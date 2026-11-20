# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar

import msgspec
import torch


class H3CausalKVCache:
    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.owners: tuple[int, ...] = ()
        self.layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._active = False

    def _idle(self) -> None:
        if self._active:
            raise RuntimeError("cannot change H3 cache during a forward")

    def reset(self) -> None:
        self._idle()
        self.owners = ()
        self.layers = {}

    def fork(self) -> H3CausalKVCache:
        self._idle()
        other = H3CausalKVCache(self.num_layers)
        other.owners = self.owners
        other.layers = self.layers.copy()
        return other

    def replace_prefix(self, prefix: H3CausalKVCache) -> None:
        self._idle()
        prefix._idle()
        count = len(prefix.owners)
        if (not count or prefix.num_layers != self.num_layers
                or prefix.owners != self.owners[:count]
                or any(owner != -1 for owner in prefix.owners)
                or set(prefix.layers) != set(self.layers)
                or len(self.layers) != self.num_layers):
            raise ValueError("incompatible H3 cache prefix")
        prototype = self.layers[0][0]
        storage = prototype.new_empty((self.num_layers, 2, len(self.owners), *prototype.shape[1:]))
        replacement = {}
        for layer, old in self.layers.items():
            targets = storage[layer].unbind(0)
            for new, original, target in zip(prefix.layers[layer], old, targets):
                if (new.shape != original[:count].shape or new.dtype != original.dtype
                        or new.device != original.device):
                    raise ValueError("incompatible H3 cache prefix tensor")
                torch.cat((new, original[count:]), dim=0, out=target)
            replacement[layer] = targets
        self.layers = replacement

    def retain(self, recent_start: int) -> None:
        self._idle()
        keep = [
            i
            for i, owner in enumerate(self.owners)
            if owner < 0 or owner >= recent_start
        ]
        if len(keep) == len(self.owners):
            return
        retained = {}
        prototype = next(iter(self.layers.values()))[0] if self.layers else None
        storage = (
            prototype.new_empty((self.num_layers, 2, len(keep), *prototype.shape[1:]))
            if prototype is not None
            else None
        )
        indexes_by_device: dict[torch.device, torch.Tensor] = {}
        for layer, (k, v) in self.layers.items():
            if k.device not in indexes_by_device:
                indexes_by_device[k.device] = torch.tensor(
                    keep, device=k.device, dtype=torch.long
                )
            indexes = indexes_by_device[k.device]
            targets = storage[layer].unbind(0)
            for source, target in zip((k, v), targets):
                torch.index_select(source, 0, indexes, out=target)
            retained[layer] = targets
        self.layers = retained
        self.owners = tuple(self.owners[i] for i in keep)


class H3CausalForward(msgspec.Struct, frozen=True):
    cache: H3CausalKVCache
    owners: tuple[int, ...]
    write: bool = False
    recent_start: int | None = None


class H3CausalFusedForward(msgspec.Struct, frozen=True):
    cache: H3CausalKVCache
    owners: tuple[int, ...]
    history_stop: int
    condition_stop: int
    recent_start: int

    @property
    def write(self) -> bool:
        return True


class H3CausalTransaction(msgspec.Struct):
    request: H3CausalForward | H3CausalFusedForward
    pending: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    visited: dict[int, dict[tuple[int, int], bool]] = {}
    total_heads: dict[int, int] = {}
    # Derived from request in __post_init__.
    retained_indexes: tuple[int, ...] = ()
    committed_indexes: tuple[int, ...] = ()
    _committed_indexes_by_device: dict[torch.device, torch.Tensor] = {}
    committed_owners: tuple[int, ...] = ()
    _output: torch.Tensor | None = None
    _joined: torch.Tensor | None = None
    _denoise: torch.Tensor | None = None
    _heads: int = 0
    _head_dim: int = 0

    def __post_init__(self) -> None:
        request = self.request
        self.committed_owners = request.cache.owners + request.owners
        if isinstance(request, H3CausalFusedForward):
            history = request.cache.owners + request.owners[: request.history_stop]
            self.retained_indexes = tuple(
                i
                for i, owner in enumerate(history)
                if owner < 0 or owner >= request.recent_start
            )
            self.committed_owners = (
                tuple(history[i] for i in self.retained_indexes)
                + request.owners[request.history_stop : request.condition_stop]
            )
            old_rows = len(request.cache.owners)
            self.committed_indexes = self.retained_indexes + tuple(
                range(
                    old_rows + request.history_stop, old_rows + request.condition_stop
                )
            )
        elif request.write and request.recent_start is not None:
            self.committed_indexes = tuple(
                i
                for i, owner in enumerate(self.committed_owners)
                if owner < 0 or owner >= request.recent_start
            )
            self.committed_owners = tuple(
                self.committed_owners[i] for i in self.committed_indexes
            )
        if isinstance(request, H3CausalFusedForward) or (
            request.write and request.recent_start is not None
        ):
            # Head-group streams wait on the caller stream before attention.
            for k, _ in request.cache.layers.values():
                if k.device not in self._committed_indexes_by_device:
                    self._committed_indexes_by_device[k.device] = torch.tensor(
                        self.committed_indexes, device=k.device, dtype=torch.long
                    )
        if request.cache.layers:
            prototype = next(iter(request.cache.layers.values()))[0]
            self._prepare_storage(prototype, prototype.shape[1])

    def _prepare_storage(self, prototype: torch.Tensor, heads: int) -> None:
        request = self.request
        self._heads = heads
        self._head_dim = prototype.shape[-1]
        rows = len(request.cache.owners) + len(request.owners)
        width = heads * prototype.shape[-1]
        if request.write:
            self._output = prototype.new_empty(
                (request.cache.num_layers, 2, len(self.committed_owners) * width)
            )
        if request.cache.owners and (
            not request.write
            or isinstance(request, H3CausalFusedForward)
            or request.recent_start is not None
        ):
            self._joined = prototype.new_empty((2, rows * width))
        if isinstance(request, H3CausalFusedForward):
            denoise_rows = (
                len(self.committed_owners)
                + len(request.owners)
                - request.condition_stop
            )
            self._denoise = prototype.new_empty((2, denoise_rows * width))

    def _views(
        self, storage: torch.Tensor, head_start: int, head_count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = storage.shape[-1] // (self._heads * self._head_dim)
        size = rows * head_count * self._head_dim
        # Pack head groups separately to preserve contiguous attention inputs.
        start = rows * head_start * self._head_dim
        if storage.is_cuda:
            # A failed head group can skip the caller's normal stream joins.
            storage.record_stream(torch.cuda.current_stream(storage.device))
        return tuple(
            value.narrow(0, start, size).view(rows, head_count, self._head_dim)
            for value in storage.unbind(0)
        )

    def _commit_indexes(self, device: torch.device) -> torch.Tensor:
        keep = self._committed_indexes_by_device.get(device)
        if keep is None:
            keep = torch.tensor(self.committed_indexes, device=device, dtype=torch.long)
        if keep.is_cuda:
            # Failed attention may skip the normal head-group stream joins.
            keep.record_stream(torch.cuda.current_stream(device))
        return keep

    def attend(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        head_start: int,
        total_heads: int,
    ) -> torch.Tensor:
        cache = self.request.cache
        used = len(self.request.owners)
        if not self._heads:
            self._prepare_storage(k, total_heads)
        head_stop = head_start + q.shape[1]
        self.total_heads[layer] = total_heads
        # A slice stays unfinished until attention returns; complete() rejects gaps.
        ranges = self.visited.setdefault(layer, {})
        ranges[head_start, head_stop] = False
        live_k, live_v = k[:used], v[:used]
        targets = (
            self._views(self._output[layer], head_start, q.shape[1])
            if self.request.write
            else None
        )
        if cache.owners:
            joined = (
                self._views(self._joined, head_start, q.shape[1])
                if self._joined is not None
                else targets
            )
            for target, old, live in zip(joined, cache.layers[layer], (live_k, live_v)):
                torch.cat((old[:, head_start:head_stop], live), dim=0, out=target)
            live_k, live_v = joined
        if isinstance(self.request, H3CausalFusedForward):
            h, c = self.request.history_stop, self.request.condition_stop
            old_rows = len(cache.owners)
            keep = self._commit_indexes(k.device)
            committed = targets
            for value, target in zip((live_k, live_v), committed):
                torch.index_select(value, 0, keep, out=target)
            history_result = attention(
                q[:h], live_k[: old_rows + h], live_v[: old_rows + h]
            )
            condition_result = attention(q[h:c], *committed)
            denoise_kv = self._views(self._denoise, head_start, q.shape[1])
            for value, live, target in zip(committed, (live_k, live_v), denoise_kv):
                torch.cat((value, live[old_rows + c :]), out=target)
            result = torch.cat(
                (history_result, condition_result, attention(q[c:used], *denoise_kv))
            )
            live_k, live_v = committed
        else:
            result = attention(q[:used], live_k, live_v)
            if self.request.write and self.request.recent_start is not None:
                keep = self._commit_indexes(k.device)
                for value, target in zip((live_k, live_v), targets):
                    torch.index_select(value, 0, keep, out=target)
                live_k, live_v = targets
            elif self.request.write and not cache.owners:
                for value, target in zip((live_k, live_v), targets):
                    target.copy_(value)
                live_k, live_v = targets
        out = torch.zeros_like(q)
        out[:used] = result
        if self.request.write:
            self.pending.setdefault(layer, {})[head_start] = tuple(
                value.detach() for value in (live_k, live_v)
            )
        ranges[head_start, head_stop] = True
        return out

    def complete(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        if set(self.visited) != set(range(self.request.cache.num_layers)):
            raise RuntimeError("causal H3 forward did not visit every cache layer")
        for layer, ranges in self.visited.items():
            cursor = 0
            for (start, stop), done in sorted(ranges.items()):
                if start != cursor or not done:
                    raise RuntimeError("causal H3 forward did not complete every head")
                cursor = stop
            if cursor != self.total_heads[layer]:
                raise RuntimeError("causal H3 forward did not complete every head")
        layers = {}
        merged = None
        for layer, slices in self.pending.items():
            parts = [slices[start] for start in sorted(slices)]
            if parts[0][0].is_cuda:
                stream = torch.cuda.current_stream(parts[0][0].device)
                for pair in parts:
                    for tensor in pair:
                        tensor.record_stream(stream)
            if len(parts) == 1:
                layers[layer] = parts[0]
            else:
                if merged is None:
                    merged = torch.empty_like(self._output)
                targets = self._views(merged[layer], 0, self._heads)
                for tensors, target in zip(zip(*parts), targets):
                    torch.cat(tensors, dim=1, out=target)
                layers[layer] = targets
        return layers


_CURRENT: ContextVar[H3CausalTransaction | None] = ContextVar(
    "h3_causal_forward", default=None
)


def current_h3_causal_transaction() -> H3CausalTransaction:
    transaction = _CURRENT.get()
    if transaction is None:
        raise RuntimeError("causal H3 attention requires a scoped forward transaction")
    return transaction


def h3_causal_fused_forward_active() -> bool:
    transaction = _CURRENT.get()
    return transaction is not None and isinstance(
        transaction.request, H3CausalFusedForward
    )


@contextmanager
def h3_causal_transaction(request: H3CausalForward | H3CausalFusedForward):
    cache = request.cache
    cache._idle()
    if _CURRENT.get() is not None:
        raise RuntimeError("nested causal H3 forwards are not supported")
    transaction = H3CausalTransaction(request)
    cache._active = True
    token = _CURRENT.set(transaction)
    try:
        yield transaction
        layers = transaction.complete()
        if request.write:
            cache.layers = layers
            cache.owners = transaction.committed_owners
    finally:
        _CURRENT.reset(token)
        cache._active = False
