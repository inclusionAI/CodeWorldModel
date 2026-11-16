# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
from pathlib import Path

import torch
from sglang.multimodal_gen.configs.models.dits.minimax_h3_causal import (
    MiniMaxH3CausalDiTArchConfig,
    MiniMaxH3CausalDiTConfig,
)
from sglang.multimodal_gen.runtime.layers.kvcache.minimax_h3_causal import H3CausalForward
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_local_torch_device,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    component_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
    get_forward_context_or_none,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_causal import (
    MiniMaxH3CausalDiTModel,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.causal_latent import (
    H3FixedHistorySession,
    H3SourceLatents,
    H3StaticCondition,
    _Frame,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs as SrtServerArgs


RECIPE = Path(__file__).resolve().parents[2] / "configs/recipes/omni-h3-longlive.json"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--sp-size", type=int, default=1)
    parser.add_argument(
        "--attention-backend",
        choices=("FA", "TORCH_SDPA", "TORCH_CUDNN_SDPA", "DYNAMIC_CUDNN_SDPA"),
        default="TORCH_SDPA",
    )
    options = parser.parse_args()
    if options.tp_size < 1 or options.sp_size < 1:
        parser.error("TP and Ulysses sizes must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != options.tp_size * options.sp_size:
        parser.error("torchrun world size must equal TP * Ulysses")
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=options.tp_size,
        sp_size=options.sp_size,
        ulysses_degree=options.sp_size,
    )
    publish(
        SrtServerArgs(model_path="dummy", tp_size=options.tp_size),
        role="diffusion_gpu_worker",
    )
    device = get_local_torch_device()
    if device.type != "cuda":
        raise RuntimeError(
            "native H3 smoke requires CUDA and the native SGLang dependencies"
        )
    torch.manual_seed(37)
    # Scale H3 widths by 1/7 while keeping its native head and rotary dimensions.
    arch = MiniMaxH3CausalDiTArchConfig(
        num_layers=2,
        token_refiner_num_layers=1,
        hidden_size=768,
        num_attention_heads=8,
        attention_head_dim=128,
        ffn_hidden_size=2048,
        latents_dim=2,
        audio_latents_dim=8,
        text_dim=8,
        timestep_input_dim=16,
        time_embed_hidden_size=768,
        time_embed_dim=384,
        adaln_out_features=18 * 768,
        final_adaln_out_features=2 * 768,
        rope_inv_freq_len=16,
    )
    config = MiniMaxH3CausalDiTConfig(arch_config=arch)
    cls, _ = ModelRegistry.resolve_model_cls("MiniMaxH3CausalDiTModel")
    assert cls is MiniMaxH3CausalDiTModel
    mapper = get_param_names_mapping(cls.param_names_mapping)
    assert (
        mapper("dit.blocks.0.attn.qkv_proj.weight")[0]
        == "blocks.0.attn.qkv_proj.weight"
    )
    with component_attn_backend_context_manager(
        AttentionBackendEnum[options.attention_backend], component_name="transformer"
    ):
        model = cls(config=config, hf_config={}).to(device).eval()
        for name, parameter in model.named_parameters():
            if (
                name.endswith("norm.weight")
                or name.endswith("norm_q.weight")
                or name.endswith("norm_k.weight")
            ):
                parameter.fill_(1)
            else:
                parameter.normal_(std=0.02)
        model.rope.inv_freq.copy_(
            10000
            ** (
                -torch.arange(
                    arch.rope_inv_freq_len, device=device, dtype=torch.float32
                )
                / arch.rope_inv_freq_len
            )
        )
        base = MiniMaxH3DiTModel(config=config, hf_config={}).to(device).eval()
        base.load_state_dict(model.state_dict(), strict=True)
        recipe = json.loads(RECIPE.read_text())
        static = H3StaticCondition(
            torch.randn(3, 8, device=device),
            torch.tensor([1, 0, 1], device=device),
            torch.randn(2, 1, 4, 4, device=device),
            torch.randn(2, 1, 4, 4, device=device),
        )
        session = H3FixedHistorySession(model, static, recipe=recipe)
        tensor = lambda count: torch.randn(2, count, 4, 4, device=device)
        source = H3SourceLatents((tensor(5),), (tensor(5),), tensor(5), tensor(5))
        frame = _Frame(
            0,
            (source.references[0][:, :1],),
            (source.reference_noise[0][:, :1],),
            source.target_noise[:, :1],
            source.clean_noise[:, :1],
        )
        packed, owners, _ = session._pack(
            [frame],
            static=True,
            conditions=True,
            video=True,
            prefix=False,
            denoise=True,
            sigma=session.sigmas[0],
        )
        with set_forward_context(current_timestep=0, attn_metadata=None):
            native, _ = model(
                causal_context=H3CausalForward(session.cache, owners), **packed
            )
            dense, _ = base(**packed)
        torch.testing.assert_close(native, dense, rtol=0.01, atol=0.01)
        assert session.cache.owners == ()
        calls = []

        def record_context(module, args, kwargs):
            context = get_forward_context()
            assert context.attn_metadata is None
            calls.append(
                (
                    kwargs["causal_context"].write,
                    context.current_timestep,
                    kwargs["unique_timesteps"].detach().cpu().clone(),
                )
            )

        handle = model.register_forward_pre_hook(record_context, with_kwargs=True)
        assert session.append(source) == ()
        (bootstrap,) = session.append(source)
        assert bootstrap.plan.bootstrap and bootstrap.video.shape[1] == 9
        (tail,) = session.finish(
            H3SourceLatents((tensor(2),), (tensor(2),), tensor(2), tensor(2))
        )
        assert tail.plan.eos and tail.video.shape[1] == 3
        assert session.state == "ended" and torch.isfinite(tail.video).all()
        handle.remove()
        denoise_calls = [(step, times) for write, step, times in calls if not write]
        assert [step for step, _ in denoise_calls] == list(range(4)) * 2
        for step, times in denoise_calls:
            assert (times == 1 - session.sigmas[step].float().cpu()).any()
        assert all(step == 0 for write, step, _ in calls if write)
        assert get_forward_context_or_none() is None
        for key, value in session.cache.layers.values():
            assert key.shape == value.shape
            assert key.shape[1] == arch.num_attention_heads // (
                options.tp_size * options.sp_size
            )
            assert key.shape[2] == arch.attention_head_dim
            assert key.shape[0] == len(session.cache.owners)
        session.reset()
        assert session.state == "open" and not session.cache.owners
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(
            "native random-weight H3 registry, dense bootstrap and fixed-history "
            f"session passed; TP={options.tp_size} Ulysses={options.sp_size} "
            f"backend={model._resolved_attention_backend.name}"
        )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
