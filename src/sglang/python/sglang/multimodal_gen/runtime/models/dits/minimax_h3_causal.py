# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
from sglang.multimodal_gen.configs.models.dits.minimax_h3_causal import (
    MiniMaxH3CausalDiTArchConfig,
)
from sglang.multimodal_gen.runtime.layers.kvcache.minimax_h3_causal import (
    H3CausalForward,
    H3CausalFusedForward,
    current_h3_causal_transaction,
    h3_causal_fused_forward_active,
    h3_causal_transaction,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_ulysses_ctx
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    get_global_forced_attn_backend,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3AdalnProj,
    MiniMaxH3Attention,
    MiniMaxH3DiTModel,
    _adaln_plan_key,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_CAUSAL_FUSED_ADALN_KEY = -1
_DENSE_ATTENTION_BACKENDS = (
    AttentionBackendEnum.FA,
    AttentionBackendEnum.TORCH_SDPA,
    AttentionBackendEnum.TORCH_CUDNN_SDPA,
    AttentionBackendEnum.DYNAMIC_CUDNN_SDPA,
)


def _causal_adaln_plan_key(timesteps: torch.Tensor, *, fused: bool):
    key = _adaln_plan_key(timesteps)
    return (_CAUSAL_FUSED_ADALN_KEY, *key) if fused else key


class MiniMaxH3CausalAdalnProj(MiniMaxH3AdalnProj):
    def project_local(self, adaln_input: torch.Tensor) -> torch.Tensor:
        output = super().project_local(adaln_input)
        if h3_causal_fused_forward_active():
            output[1:] = super().project_local(adaln_input[1:])
        return output


class MiniMaxH3CausalAttention(MiniMaxH3Attention):
    def _attention_after_exchange(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        head_start: int,
        head_count: int,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int,
    ) -> torch.Tensor:
        transaction = current_h3_causal_transaction()
        return transaction.attend(
            self.causal_layer_idx,
            q,
            k,
            v,
            lambda query, key, value: self._attention_impl.forward(
                query[None],
                key[None],
                value[None],
                None,
            )[0],
            head_start=head_start,
            total_heads=self.num_heads // get_ulysses_ctx()[0],
        )


class MiniMaxH3CausalDiTModel(MiniMaxH3DiTModel):
    param_names_mapping = MiniMaxH3CausalDiTArchConfig().param_names_mapping
    _aliases = ["CausalMiniMaxH3DiTModel", "CausalMiniMaxH3DiTModelSP"]
    _minimax_h3_native_checkpoint_classes = (
        "MiniMaxH3DiTModel",
        "MiniMaxH3CausalDiTModel",
        *_aliases,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.arch.hybrid_attention is not None or self.arch.has_gate_compress:
            raise ValueError("causal H3 requires the dense H3 architecture")
        for index, block in enumerate(self.blocks):
            block.attn.__class__ = MiniMaxH3CausalAttention
            block.attn.causal_layer_idx = index
            if block.adaln_proj is not None:
                block.adaln_proj.__class__ = MiniMaxH3CausalAdalnProj

    def prepare_adaln_plans(
        self,
        step_timesteps: list[torch.Tensor],
        *,
        fused_plan_indices: frozenset[int] = frozenset(),
    ) -> torch.Tensor | None:
        cache = self.adaln_cache
        if cache is None:
            return None
        if fused_plan_indices and cache.weight_files is None:
            raise ValueError(
                "fused causal H3 AdaLN requires an online checkpoint cache"
            )
        keys = [
            _causal_adaln_plan_key(timesteps, fused=index in fused_plan_indices)
            for index, timesteps in enumerate(step_timesteps)
        ]
        if cache.weight_files is not None:

            def embed(timesteps: torch.Tensor) -> torch.Tensor:
                out = torch.nn.functional.silu(self.time_embedder(timesteps))
                return out.to(torch.bfloat16) if cache.precision == "match" else out

            cache.build(
                step_timesteps,
                embed=embed,
                keys=keys,
                singleton_tail_keys=frozenset(keys[i] for i in fused_plan_indices),
            )
        return cache.resolve_slots(step_timesteps, keys=keys)

    def _time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        output = super()._time_embedding(timesteps)
        if h3_causal_fused_forward_active():
            output[1:] = super()._time_embedding(timesteps[1:])
        return output

    def _resolve_attention_backend_once(self):
        super()._resolve_attention_backend_once()
        requested = (
            get_global_forced_attn_backend()
            or self._component_attention_backend_override
        )
        if requested is not None and requested != self._resolved_attention_backend:
            raise ValueError(
                f"causal H3 requested {requested.name}, but native resolution selected "
                f"{self._resolved_attention_backend.name}"
            )
        if self._resolved_attention_backend not in _DENSE_ATTENTION_BACKENDS:
            raise ValueError("causal H3 requires dense FlashAttention or SDPA")

    def forward(
        self, *, causal_context: H3CausalForward | H3CausalFusedForward, **kwargs: Any
    ):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError(
                "causal H3 fixed-history inference requires eval and no_grad"
            )
        if kwargs["x"].device.type == "mps":
            raise ValueError(
                "causal H3 does not support the MPS streamed attention path"
            )
        if self._h3_spectrum_requested(sp_ws=1):
            raise ValueError("causal H3 requires every block on every forward")
        if self.adaln_cache is not None and kwargs.get("adaln_cache_slot") is None:
            timesteps = kwargs["unique_timesteps"].view(-1)
            key = _causal_adaln_plan_key(
                timesteps, fused=isinstance(causal_context, H3CausalFusedForward)
            )
            kwargs["adaln_cache_slot"] = self.adaln_cache.resolve_slots(
                [timesteps], keys=[key]
            )[0]
        with h3_causal_transaction(causal_context):
            return super().forward(**kwargs)


EntryClass = MiniMaxH3CausalDiTModel
