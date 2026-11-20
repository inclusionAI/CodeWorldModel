# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_causal import (
    MiniMaxH3CausalPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.realtime import (
    H3RealtimeStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.realtime_codec import (
    H3RealtimeCodec,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

# GPUWorker publishes from the first rank of the single replica.
_OUTPUT_RANK = 0


def load_h3_realtime_modules(server_args: ServerArgs) -> dict[str, object]:
    from sglang.kernels.ops.diffusion import mount_minimax_h3_norm_modulate
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_world_group,
    )
    from sglang.multimodal_gen.runtime.loader.minimax_h3_causal import (
        bind_h3_codec_groups,
        load_h3_causal_components,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.causal_streaming import (
        H3CausalVideoDecoder,
    )

    config = server_args.pipeline_config
    if config.recipe is None or config.decoder is None:
        raise ValueError("causal H3 realtime requires a recipe and decoder options")
    encoder_group = decoder_group = None
    if config.codec_parallel is not None:
        # Every rank creates the codec communicators before any weights load.
        encoder_group, decoder_group = bind_h3_codec_groups(
            encoder_ranks=tuple(config.codec_parallel.encoder_ranks),
            decoder_ranks=tuple(config.codec_parallel.decoder_ranks),
            output_rank=_OUTPUT_RANK,
        )
    components = load_h3_causal_components(server_args, encoder_group=encoder_group)
    if config.lossless_norm and not mount_minimax_h3_norm_modulate(components.model):
        raise RuntimeError("requested lossless H3 norm modulation could not be mounted")
    decoder = None
    if decoder_group is None or decoder_group.is_member:
        options = config.decoder
        decoder = H3CausalVideoDecoder(
            decoder_config=options.decoder_config,
            decoder_weights=options.decoder_weights,
            device=components.device,
            window=options.window,
            backend=options.backend,
            compile=options.compile,
            cuda_graph=options.cuda_graph,
            tp_group=None if decoder_group is None else decoder_group.coordinator,
            cudnn_sdpa=options.cudnn_sdpa,
            spatial_attention_backend=options.spatial_attention_backend,
            compact_suffix_kv=options.compact_suffix_kv,
            suffix_attention_backend=options.suffix_attention_backend,
            compile_mode=options.compile_mode,
            tp_fused_post=options.tp_fused_post,
        )
    codec = H3RealtimeCodec(decoder, get_world_group(), output_rank=_OUTPUT_RANK,
        device=components.device, decoder_group=decoder_group)
    return {"h3_components": components, "h3_codec": codec}


class MiniMaxH3CausalRealtimePipeline(ComposedPipelineBase):
    """Select with ``pipeline_class_name="MiniMaxH3CausalRealtimePipeline"``."""

    pipeline_name = "MiniMaxH3CausalRealtimePipeline"
    is_video_pipeline = True
    pipeline_config_cls = MiniMaxH3CausalPipelineConfig
    sampling_params_cls = SamplingParams
    # Components come from the native H3 bundle's component_paths, not model_index.json.
    _required_config_modules = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError("causal H3 realtime only supports monolithic deployment")

    def load_modules(self, server_args: ServerArgs, loaded_modules=None):
        return load_h3_realtime_modules(server_args)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        config = server_args.pipeline_config
        self.add_stage(
            H3RealtimeStage(
                components=self.get_module("h3_components"),
                codec=self.get_module("h3_codec"),
                recipe=config.recipe,
                codec_overlap=config.codec_overlap,
                fuse_first_denoise=config.fuse_first_denoise,
                cwm_silence_path=config.cwm_silence_path,
                cwm_proxy_encode_size=config.cwm_proxy_encode_size,
                cwm_proxy_resize=config.cwm_proxy_resize,
            )
        )


EntryClass = MiniMaxH3CausalRealtimePipeline
