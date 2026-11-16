# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field

import msgspec

from sglang.multimodal_gen.configs.models.dits.minimax_h3_causal import (
    MiniMaxH3CausalDiTConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3_causal import H3FixedHistoryConfig


class H3CausalDecoderOptions(msgspec.Struct, frozen=True):
    decoder_config: str
    decoder_weights: str
    window: int = 16
    backend: str = "reference"
    compile: bool = False
    cuda_graph: bool = False
    # Runs the decoder's suffix attention under SDPBackend.CUDNN_ATTENTION.
    cudnn_sdpa: bool = False
    spatial_attention_backend: str = "fa2"
    compact_suffix_kv: bool = False
    suffix_attention_backend: str = "sdpa"
    compile_mode: str = "default"
    tp_fused_post: bool = False


class H3CodecParallel(msgspec.Struct, frozen=True):
    encoder_ranks: tuple[int, ...]
    decoder_ranks: tuple[int, ...] = (0, 1)


@dataclass
class MiniMaxH3CausalPipelineConfig(MiniMaxH3PipelineConfig):
    """Causal fixed-history H3 served one realtime session at a time."""

    dit_config: MiniMaxH3CausalDiTConfig = field(
        default_factory=MiniMaxH3CausalDiTConfig
    )
    transformer_class_override: str | None = "MiniMaxH3CausalDiTModel"
    # Raw fixed-history v2 recipe; H3FixedHistorySession parses it per session.
    recipe: dict | None = None
    decoder: H3CausalDecoderOptions | None = None
    codec_parallel: H3CodecParallel | None = None
    # Encoder pair encodes the request's next source while the decoder pair decodes.
    codec_overlap: bool = False
    fuse_first_denoise: bool = False
    lossless_norm: bool = False
    cwm_silence_path: str | None = None
    cwm_proxy_encode_size: tuple[int, int] = (768, 1344)
    cwm_proxy_resize: str = "lanczos"
    # Optional encoder-only TAE; the static image always uses the full H3 VAE.
    tae_weights: str | None = None

    def __post_init__(self) -> None:
        if self.tae_weights is not None and (
                not isinstance(self.tae_weights, str) or not self.tae_weights):
            raise ValueError("tae_weights must be a nonempty path or None")
        if self.cwm_proxy_encode_size not in ((256, 448), (768, 1344)):
            raise ValueError('CWM proxy encoder size must be 256x448 or 768x1344 (H,W)')
        if self.cwm_proxy_resize not in ("none", "lanczos", "nearest"):
            raise ValueError("unsupported CWM proxy resize")
        if self.recipe is not None:
            H3FixedHistoryConfig.from_recipe(self.recipe)
            if (self.recipe.get('protocol') == 'cwm_causal/v1') != bool(self.cwm_silence_path):
                raise ValueError('CWM causal requires an explicit frozen silence artifact')
        if self.codec_overlap and (
            self.codec_parallel is None
            or (tuple(self.codec_parallel.encoder_ranks), tuple(self.codec_parallel.decoder_ranks))
            not in (((2, 3), (0, 1)), ((2, 3), (0, 1, 4, 5)),
                    ((2, 3, 6, 7), (0, 1, 4, 5)))
        ):
            raise ValueError(
                "native codec overlap requires E2 ranks 2/3 or E4 ranks 2/3/6/7 with a disjoint decoder"
            )
