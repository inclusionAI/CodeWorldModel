from dataclasses import dataclass
import hashlib
import time

import torch
import torch.distributed as dist

from ..inference_optimizations import _tensor_stamp


_EAGER, _WARM_CAPTURE, _REPLAY = range(3)
_WARMUPS = 3


def _layout(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), tensor.device.type)


def _storage(tensor):
    return (tensor.data_ptr(), tensor.device, _layout(tensor))


def _fingerprint(layout, key):
    logical_key = None if key is None else (key[0].type, *key[1:])
    digest = hashlib.sha256(repr((layout, logical_key)).encode()).digest()
    return tuple(int.from_bytes(digest[i:i+4], 'little') for i in range(0, 16, 4))


def _agreed_action(records):
    if any(record[1] for record in records):
        return _EAGER, True
    if any(tuple(record) != tuple(records[0]) for record in records[1:]):
        return _EAGER, False
    return records[0][0], False


@dataclass
class _AgreementWorkspace:
    device: object = None
    group: object = None
    world_size: int = 0
    control: torch.Tensor = None
    received: torch.Tensor = None
    views: list = None
    host_received: torch.Tensor = None


def _agree(record, device, group, workspace):
    world_size = dist.get_world_size(group)
    if (workspace.device != device or workspace.group is not group
            or workspace.world_size != world_size):
        control = torch.empty(len(record), dtype=torch.int64, device=device)
        received = torch.empty((world_size, len(record)), dtype=torch.int64, device=device)
        host_received = torch.empty(received.shape, dtype=torch.int64, device='cpu')
        workspace.control, workspace.received = control, received
        workspace.views, workspace.host_received = list(received.unbind(0)), host_received
        workspace.device, workspace.group, workspace.world_size = device, group, world_size
    workspace.control.copy_(torch.tensor(record, dtype=torch.int64, device='cpu'))
    dist.all_gather(workspace.views, workspace.control, group=group)
    workspace.host_received.copy_(workspace.received)
    return _agreed_action(workspace.host_received.tolist())


@dataclass
class _Variant:
    graph: object
    output: torch.Tensor
    projections: tuple
    inputs: dict
    slot: int
    fingerprint: tuple


@dataclass(frozen=True)
class _WindowAdmission:
    token: object
    context: tuple


class _DenoiseGraphs:
    def __init__(self):
        self.variants = {}
        self.layout = self.binding = None
        self.inputs = self.arenas = self.weights = None
        self.device = None
        self._agreement = _AgreementWorkspace()
        self._window_admission = None
        self._stats = dict(warmup_calls=0, captures=0, replays=0, invalidations=0,
                           eager_reasons={}, capture_host_ms=0.)

    @property
    def stats(self):
        return dict(self._stats, eager_reasons=dict(self._stats['eager_reasons']),
                    entries=len(self.variants))

    def end_window(self):
        self._window_admission = None

    def close(self):
        self.end_window()
        if self.variants:
            torch.cuda.synchronize(self.device)
        self.variants.clear()
        self._agreement = _AgreementWorkspace()
        self.layout = self.binding = self.inputs = self.arenas = self.weights = self.device = None

    def _eager(self, reason, model, hidden, kwargs):
        self.end_window()
        reasons = self._stats['eager_reasons']
        reasons[reason] = reasons.get(reason, 0) + 1
        return model._run_blocks(hidden, **kwargs)

    def _step_guard(self, model, hidden, kwargs, group, sp_size, tp_size):
        cache = kwargs.get('past_key_values')
        metadata = kwargs.get('attention_metadata')
        tensors = (hidden, kwargs['combined_indices'], *kwargs['rope_cache'])
        key = getattr(kwargs['adaln_input'], '_h3_adaln_key', None)
        layout = (tuple(_layout(t) for t in tensors), None if metadata is None else
                  (metadata.q_bounds, metadata.k_bounds, metadata.max_seqlen_q, metadata.max_seqlen_k))
        context = (id(model), getattr(model, '_h3_inference_generation', None), id(group),
                   hidden.device, tuple(sorted(getattr(model, '_h3_inference', {}).items())),
                   sp_size, tp_size, layout)
        reason = None
        if model.training or torch.is_grad_enabled():
            reason = 'training_or_grad'
        elif hidden.device.type != 'cuda' or hidden.dtype != torch.bfloat16 or sp_size != 8 or tp_size != 1:
            reason = 'device_dtype_or_parallelism'
        elif any(t.device != hidden.device for t in tensors[1:]):
            reason = 'input_device'
        elif not all(getattr(model, '_h3_inference', {}).get(flag) for flag in ('metadata', 'kv', 'adaln', 'rope')):
            reason = 'required_options'
        elif getattr(model, '_h3_inference', {}).get('delayed_hc'):
            reason = 'delayed_hc'
        elif not getattr(cache, '_h3_dit_graph_eligible', False):
            reason = 'window_not_steady'
        elif not getattr(cache, 'supports_inference_arena', False) or kwargs['update_past_key_values']:
            reason = 'cache_or_store'
        elif (metadata is None or len(metadata.q_bounds) != 3 or len(metadata.k_bounds) != 3
              or metadata.k_bounds[-1] - metadata.k_bounds[-2] != metadata.q_bounds[-1] - metadata.q_bounds[-2]
              or any(kwargs.get(name) is not None for name in
                     ('cache_segments', 'attention_mask', 'q_ranges', 'k_ranges', 'attn_type_map', 'attn_workloads'))):
            reason = 'attention_layout'
        elif key is None or kwargs['adaln_input'].is_inference():
            reason = 'adaln_key'
        return reason, layout, key, context

    def _inspect(self, model, hidden, kwargs, guard):
        _, layout, _, context = guard
        cache, metadata = kwargs.get('past_key_values'), kwargs.get('attention_metadata')
        parameters = tuple(model.blocks.parameters())
        buffers = tuple(model.blocks.buffers())
        weights = parameters + buffers
        arenas = getattr(cache, '_h3_kv_arenas', {})
        arena_stamp = tuple((layer, tuple(_storage(t) for t in pair)) for layer, pair in sorted(arenas.items()))
        binding = context[:5], tuple(_tensor_stamp(t) for t in weights), arena_stamp
        invalidated = self.binding is not None and self.binding != binding
        reason = None
        if (any(t.is_inference() for t in weights) or not parameters
              or any(t.device != hidden.device or t.dtype != torch.bfloat16 for t in parameters)
              or any(t.device != hidden.device for t in buffers)):
            reason = 'untracked_or_nonresident_weights'
        if reason is None:
            past_len = metadata.k_bounds[1] - metadata.q_bounds[1]
            for block in model.blocks:
                layer = block.attn.layer_idx
                arena = arenas.get(layer)
                past = (cache.key_cache[layer], cache.value_cache[layer])
                if (arena is None or tuple(cache.kvlens) != (past_len, 0)
                        or any(p is None or len(p) != past_len or p.data_ptr() != a.data_ptr()
                               or p.shape[1:] != a.shape[1:] or p.stride() != a.stride()
                               or p.device != a.device or p.dtype != a.dtype
                               or len(a) < metadata.k_bounds[-1]
                               for p, a in zip(past, arena))):
                    reason = 'history_not_in_arena'
                    break
        if reason is None and self.layout is not None and layout != self.layout:
            reason = 'layout_changed'
        return reason, invalidated, binding, tuple(arenas.values())

    def _stage(self, hidden, kwargs):
        live = (hidden, kwargs['combined_indices'], *kwargs['rope_cache'])
        if self.inputs is None:
            self.inputs = tuple(torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
                                for t in live)
        for target, source in zip(self.inputs, live):
            target.copy_(source)

    def _capture(self, model, kwargs, projections, slot, fingerprint):
        started = time.perf_counter()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(_WARMUPS):
                model._run_blocks(self.inputs[0], **kwargs)
                self._stats['warmup_calls'] += 1
        stream.synchronize()
        # Both NCCL exchange paths must enter capture in the same ordinal order.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = model._run_blocks(self.inputs[0], **kwargs)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self._stats['captures'] += 1
        self._stats['capture_host_ms'] += (time.perf_counter() - started) * 1000
        held = {name: value for name, value in kwargs.items() if name != 'past_key_values'}
        return _Variant(graph, output, projections, held, slot, fingerprint)

    def run(self, model, hidden, kwargs, *, group, sp_size, tp_size):
        token = getattr(kwargs.get('past_key_values'), '_h3_dit_graph_window', None)
        admission = self._window_admission
        guard = self._step_guard(model, hidden, kwargs, group, sp_size, tp_size)
        reason, layout, key, context = guard
        if reason is not None:
            self.end_window()
            invalidated, binding, arenas = False, None, None
        elif (token is not None and admission is not None and admission.token is token
              and context == admission.context):
            invalidated, binding, arenas = False, self.binding, self.arenas
        else:
            self.end_window()
            reason, invalidated, binding, arenas = self._inspect(model, hidden, kwargs, guard)
        variant = self.variants.get(key)
        slot = variant.slot if variant is not None else len(self.variants)
        fingerprint = (variant.fingerprint if variant is not None and layout == self.layout
                       else _fingerprint(layout, key))
        if reason is None and variant is None and len(self.variants) == 4:
            reason = 'variant_limit'
        proposed = _EAGER if reason else (_REPLAY if variant is not None else _WARM_CAPTURE)
        record = (proposed, int(invalidated), slot, len(self.variants), _WARMUPS, *fingerprint)
        action, reset = _agree(record, hidden.device, group, self._agreement)
        if reset:
            self.close()
            self._stats['invalidations'] += 1
            return self._eager('invalidated', model, hidden, kwargs)
        if action == _EAGER:
            return self._eager(reason or 'rank_disagreement', model, hidden, kwargs)

        self._stage(hidden, kwargs)
        if action == _WARM_CAPTURE:
            self.device, self.layout, self.binding, self.arenas = hidden.device, layout, binding, arenas
            self.weights = tuple(model.blocks.parameters()) + tuple(model.blocks.buffers())
            projections = tuple(block.adaln_proj(kwargs['adaln_input']) for block in model.blocks)
            staged = dict(kwargs, combined_indices=self.inputs[1], rope_cache=self.inputs[2:])
            # A capture/collective failure is fatal: no rank-local eager recovery.
            self.variants[key] = self._capture(model, staged, projections, slot, fingerprint)
        variant = self.variants[key]
        variant.graph.replay()
        self._stats['replays'] += 1
        if token is not None:
            self._window_admission = _WindowAdmission(token, context)
        return variant.output
