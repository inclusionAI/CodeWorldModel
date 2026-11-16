"""Ref packed-forward adapter without a training meta-model or config."""
from __future__ import annotations
from typing import Any

import torch
from common.distributed.ops import get_device
from common.distributed.unified_parallel import (get_unified_parallel_world_size,
                                                 is_unified_parallel_initialized)
from projects.minimax_h3.modeling.causal_layout import CausalLayout as _Layout
from projects.minimax_h3.modeling.ref_condition import CWMRefConditionMixin
from projects.minimax_h3.modeling.ref_guidance import RefSamplingGuidance
from projects.minimax_h3.modeling.packed_tokens import (minimax_h3_patchify_video_latent,
                                                     minimax_h3_unpatchify_video_tokens)

_PATCH_SIZE = (1, 2, 2)


class RefModel(RefSamplingGuidance, CWMRefConditionMixin):
    def _condition_options(self):
        return self.recipe.input_options

    def _stream_conditioning_mode(self, requested):
        if requested != self.recipe.conditioning_mode:
            raise ValueError('request conditioning differs from inference recipe')
        return self._condition_options()['conditioning_mode']

    def _guidance_scale(self):
        return self.recipe.guidance['scale']

    @classmethod
    def _to_device(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(get_device())
        if isinstance(value, list):
            return [cls._to_device(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._to_device(item) for key, item in value.items()}
        return value

    @staticmethod
    def _video_rows(latent: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        return minimax_h3_patchify_video_latent(
            latent[:, start:stop].unsqueeze(0), patch_size=_PATCH_SIZE
        )

    @staticmethod
    def _video_from_rows(rows: torch.Tensor, layout: _Layout) -> torch.Tensor:
        latent_t, patch_h, patch_w = layout.video_patch_grid
        return minimax_h3_unpatchify_video_tokens(
            rows,
            latent_shape=(latent_t, patch_h, patch_w, layout.latent_shape[0]),
            patch_size=_PATCH_SIZE,
        )[0]

    @staticmethod
    def _num_layers(model: Any) -> int:
        assert hasattr(model, "dit"), (
            "MiniMax-H3 model nodes must be MiniMaxH3X0Model wrappers; the raw DiT "
            "returns velocity and would silently flip the x0 sign"
        )
        return len(model.dit.blocks)

    @staticmethod
    def _sp_size() -> int:
        if is_unified_parallel_initialized():
            return get_unified_parallel_world_size()
        return 1

    @classmethod
    def _pad_rows(cls, rows: int) -> int:
        """Rows to append so a packed sequence shards evenly across the SP group.

        Both H3 SP models refuse a ``seq_len`` that the Ulysses world size does
        not divide -- they shard rows and never pad on the caller's behalf. The
        native bidirectional layout is already 64-aligned, but the causal
        layouts come from the dataset's chunk structure and are arbitrary.

        When SP is active this is deliberately never zero: the padding travels
        as its own attention document (and, on the cached path, its own cache
        sample), and a zero-length varlen segment is a needless edge case for
        ``alignment`` rows of savings.
        """
        alignment = cls._sp_size()
        if alignment <= 1:
            return 0
        remainder = rows % alignment
        return alignment - remainder if remainder else alignment

    @staticmethod
    def _generator(rng: Any) -> torch.Generator:
        """The rng's generator for the current device.

        ``RandomState.torch_cuda_generator`` is None without CUDA, and
        ``normal_(generator=None)`` would silently fall back to the global
        stream and break resume determinism.
        """
        device = get_device()
        return rng.torch_cuda_generator if device.type == "cuda" else rng.torch_generator
