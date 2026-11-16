# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import msgspec
import numpy as np
import torch

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


_H3_CODEC_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))


def _codec_group_partitions(members):
    target = tuple(members)
    selected_pairs = [pair for pair in _H3_CODEC_PAIRS if set(pair).issubset(target)]
    if sum(len(pair) for pair in selected_pairs) != len(target):
        raise ValueError("H3 codec groups must contain complete rank pairs")
    return [list(target)] + [list(pair) for pair in _H3_CODEC_PAIRS if pair not in selected_pairs]


class H3CodecGroup(msgspec.Struct, frozen=True):
    coordinator: object
    members: tuple[int, ...] = (0, 1)

    @property
    def is_member(self) -> bool:
        return self.coordinator.rank in self.members


def bind_h3_codec_groups(*, encoder_ranks=(0, 1), decoder_ranks=(0, 1), output_rank: int = 0):
    """Create independent codec communicators in the same order on every rank."""
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        GroupCoordinator, get_world_group,
    )
    from sglang.multimodal_gen.runtime.platforms import current_platform

    world = get_world_group()
    if world.world_size != 8 or world.ranks != list(range(8)) or output_rank != 0:
        raise ValueError("H3 codec groups require world ranks 0..7 and output rank 0")
    if any(not isinstance(ranks, (tuple, list))
           or any(type(rank) is not int for rank in ranks)
           for ranks in (encoder_ranks, decoder_ranks)):
        raise ValueError("H3 codec groups require a pair encoder and a pair-aligned decoder containing rank 0")
    encoder = tuple(encoder_ranks)
    decoder = tuple(decoder_ranks)
    if (
            encoder not in (*_H3_CODEC_PAIRS, (2, 3, 6, 7))
            or len(decoder) not in (2, 4)
            or tuple(sorted(decoder)) != decoder
            or any(tuple(decoder[i:i + 2]) not in _H3_CODEC_PAIRS
                   for i in range(0, len(decoder), 2))
            or 0 not in decoder):
        raise ValueError("H3 codec groups require a pair encoder or E4 ranks 2/3/6/7 and a pair-aligned decoder containing rank 0")
    if current_platform.is_cuda():
        from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
            can_p2p,
        )

        # SRT's first P2P cache read barriers across the world, not just decoder ranks.
        can_p2p(0, 2)
    # Pair-aligned groups keep the subgroup creation order deterministic.
    groups = []
    for name, members in (("h3_encoder", tuple(encoder_ranks)),
                          ("h3_decoder", tuple(decoder_ranks))):
        group = GroupCoordinator(_codec_group_partitions(members), local_rank=world.local_rank,
            torch_distributed_backend=torch.distributed.get_backend(world.device_group),
            use_device_communicator=name == "h3_decoder" and world.rank in members,
            use_srt_custom_allreduce=name == "h3_decoder" and world.rank in members,
            group_name=name)
        groups.append(H3CodecGroup(group, members))
    return groups[0], groups[1]


class H3CausalComponents(msgspec.Struct):
    model: torch.nn.Module
    text_encoder: torch.nn.Module
    video_encoder: torch.nn.Module | None
    processor: object
    encoder_group: H3CodecGroup | None = None
    world_group: object = None
    proxy_encode_size: tuple[int, int] | None = None
    proxy_resize: str = "lanczos"
    proxy_encoder: str = "h3"
    proxy_tae: torch.nn.Module | None = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def condition(self, prompt: str, first_rgb: np.ndarray):
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
            minimax_h3_multi_image_presentation,
        )

        _rgb(first_rgb, video=False)
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        image = self.processor.image_processor(images=[first_rgb], return_tensors="pt")
        grid = image["image_grid_thw"]
        count = int(grid[0].prod()) // self.processor.image_processor.merge_size**2
        ids, tags = minimax_h3_multi_image_presentation(
            self.processor.tokenizer,
            prompt=prompt,
            image_token_counts=[count],
        )
        with set_forward_context(current_timestep=0, attn_metadata=None):
            hidden = self.text_encoder.encode_ids(
                ids,
                pixel_values=image["pixel_values"],
                image_grid_thw=grid,
            )
        return hidden, tags.to(hidden.device)

    def encode_image(self, rgb: np.ndarray) -> torch.Tensor:
        _rgb(rgb, video=False)
        return self._encode(rgb[None], count=1)

    def encode_proxy(self, rgb: np.ndarray, *, eos: bool) -> torch.Tensor:
        local = self.encode_proxy_local(rgb, eos=eos)
        return self.broadcast_proxy(local, rgb_shape=rgb.shape, eos=eos)

    @torch.no_grad()
    def encode_proxy_local(self, rgb: np.ndarray, *, eos: bool) -> torch.Tensor | None:
        _rgb(rgb, video=True)
        if type(eos) is not bool or len(rgb) != (5 if eos else 17):
            raise ValueError(
                "proxy encoding requires 17 RGB frames, or exactly 5 for EOS"
            )
        if self.proxy_encoder == "tae":
            group = self.encoder_group
            if group is not None and group.coordinator.rank != group.members[0]:
                return None
            return self._encode_tae(rgb)
        if self.encoder_group is None or self.encoder_group.is_member:
            if self.proxy_encode_size is not None:
                if self.proxy_resize == "nearest":
                    return self._encode_local(rgb, count=2 if eos else 5, pad_to=17,
                                              resize_to=self.proxy_encode_size)
                from PIL import Image
                height, width = self.proxy_encode_size
                rgb = np.stack([np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))
                                for frame in rgb])
            return self._encode_local(rgb, count=2 if eos else 5, pad_to=17)
        return None

    def broadcast_proxy(self, local_proxy, *, rgb_shape, eos: bool) -> torch.Tensor:
        if self.proxy_encode_size is not None:
            rgb_shape = (rgb_shape[0], *self.proxy_encode_size, 3)
        return self._broadcast_encoded(local_proxy, rgb_shape=rgb_shape, count=2 if eos else 5)

    @torch.no_grad()
    def _encode(
        self, rgb: np.ndarray, *, count: int, pad_to: int | None = None,
    ) -> torch.Tensor:
        group = self.encoder_group
        latent = (self._encode_local(rgb, count=count, pad_to=pad_to)
                  if group is None or group.is_member else None)
        return self._broadcast_encoded(latent, rgb_shape=rgb.shape, count=count)

    def _broadcast_encoded(self, latent, *, rgb_shape, count):
        group = self.encoder_group
        if group is None:
            return latent
        if latent is None:
            latent = torch.empty(24, count, rgb_shape[1] // 16, rgb_shape[2] // 16,
                device=self.device, dtype=torch.bfloat16)
        return self.world_group.broadcast(latent,
            src=self.world_group.ranks.index(group.members[0]))

    def _encode_local(
        self, rgb: np.ndarray, *, count: int, pad_to: int | None = None,
        resize_to: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        vae = self.video_encoder
        device = next(vae.parameters()).device
        pixels = vae.processor.convert_numpy_to_tensor(
            np.ascontiguousarray(rgb), device
        )
        if resize_to is not None:
            pixels = torch.nn.functional.interpolate(pixels, size=resize_to, mode="nearest")
        pixels = vae.processor.transform_tensor(pixels)
        pixels = pixels.transpose(0, 1)[None].to(torch.bfloat16)
        if pad_to is not None and pixels.shape[2] < pad_to:
            pixels = torch.cat(
                (
                    pixels,
                    pixels[:, :, -1:].expand(-1, -1, pad_to - pixels.shape[2], -1, -1),
                ),
                dim=2,
            )
        # Fixed 17/5 packets must bypass encode_temporal's clip padding and token drop.
        with torch.autocast(device_type=device.type, enabled=False):
            moments = vae._adaptive_encode(pixels)
        # Conditioning uses the posterior mean; the log-variance half is unused.
        mean = moments.chunk(2, dim=1)[0][0].float()
        arch = vae.sglang_config.arch_config
        center = mean.new_tensor(arch.latents_mean).reshape(-1, 1, 1, 1)
        scale = mean.new_tensor(arch.latents_std).reshape(-1, 1, 1, 1)
        encoded_count = 5 if pad_to is not None else 1
        height, width = resize_to if resize_to is not None else rgb.shape[1:3]
        if mean.shape != (24, encoded_count, height // 16, width // 16):
            raise RuntimeError(
                f"native video encoder returned unexpected shape {tuple(mean.shape)}"
            )
        return ((mean - center) / scale)[:, :count].to(torch.bfloat16).contiguous()

    def _encode_tae(self, rgb: np.ndarray) -> torch.Tensor:
        if self.proxy_encode_size is not None and self.proxy_resize == "lanczos":
            from PIL import Image
            height, width = self.proxy_encode_size
            rgb = np.stack([np.asarray(Image.fromarray(frame).resize(
                (width, height), Image.Resampling.LANCZOS)) for frame in rgb])
        pixels = torch.as_tensor(np.ascontiguousarray(rgb), device=self.device)
        pixels = pixels.permute(0, 3, 1, 2).float().div_(255)
        if self.proxy_encode_size is not None and self.proxy_resize == "nearest":
            pixels = torch.nn.functional.interpolate(
                pixels, size=self.proxy_encode_size, mode="nearest")
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            latent = self.proxy_tae.encode_video(pixels[None])[0].permute(1, 0, 2, 3)
        height, width = self.proxy_encode_size or rgb.shape[1:3]
        if latent.shape != (24, 2 if len(rgb) == 5 else 5, height // 16, width // 16):
            raise RuntimeError(f"native TAE returned unexpected shape {tuple(latent.shape)}")
        # TAE emits normalized diffusion latents, so H3 posterior statistics do not apply.
        return latent.to(torch.bfloat16).contiguous()

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def _rgb(value: np.ndarray, *, video: bool) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.ndim != (4 if video else 3)
        or value.shape[-1] != 3
        or min(value.shape) <= 0
        or value.shape[-3] % 32
        or value.shape[-2] % 32
    ):
        raise ValueError(
            "RGB requires uint8 HWC/THWC with positive dimensions and H/W divisible by 32"
        )


def load_h3_causal_components(
    server_args: ServerArgs, *, encoder_group: H3CodecGroup | None = None
) -> H3CausalComponents:
    from sglang.multimodal_gen.configs.models.dits.minimax_h3_causal import (
        MiniMaxH3CausalDiTConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
        MiniMaxH3PipelineConfig,
    )
    from sglang.multimodal_gen.runtime.distributed import (
        get_local_torch_device,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_world_group,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
        AutoProcessorLoader,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
        TextEncoderLoader,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
        TransformerLoader,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
        VAELoader,
    )
    from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
        MiniMaxH3Qwen3VLEncoder,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    if not model_parallel_is_initialized():
        raise RuntimeError(
            "initialize the SGLang distributed worker before loading H3 components"
        )
    args = deepcopy(server_args)
    if not isinstance(args.pipeline_config, MiniMaxH3PipelineConfig):
        raise ValueError("causal H3 requires MiniMaxH3PipelineConfig")
    args.pipeline_config.dit_config = MiniMaxH3CausalDiTConfig()
    args.pipeline_config.transformer_class_override = "MiniMaxH3CausalDiTModel"
    args.component_precisions["video_vae"] = "bf16"
    args.model_variant = args.model_variant or "ref2va"
    paths = args.component_paths
    names = ("transformer", "text_encoder", "video_vae", "processor")
    if any(not paths.get(name) for name in names):
        raise ValueError(f"component_paths must provide {names}")
    device = get_local_torch_device()
    components = []
    for name, loader, library in (
        ("transformer", TransformerLoader, "diffusers"),
        ("text_encoder", TextEncoderLoader, "transformers"),
        ("video_vae", VAELoader, "diffusers"),
        ("processor", AutoProcessorLoader, "transformers"),
    ):
        if name == "video_vae" and encoder_group is not None and not encoder_group.is_member:
            components.append(None)
            continue
        model, _ = loader(device=device).load(
            paths[name],
            args,
            name,
            library,
            allow_native_fallback=False,
        )
        components.append(model)
    model, text, video, processor = components
    # The DiT class is forced above; the encoders follow the checkpoint config.
    if (
        not isinstance(text, MiniMaxH3Qwen3VLEncoder)
        or (not isinstance(video, MiniMaxH3VideoVAE)
            and (encoder_group is None or encoder_group.is_member))
    ):
        raise TypeError(
            "H3 component configuration did not resolve the required native classes"
        )
    if args.encoder_parallel == "fold" and (
        text._encoder_tp_group is None
        or text._encoder_tp_group.world_size != args.num_gpus
    ):
        raise ValueError(
            "native Qwen dimensions did not permit requested replica folding"
        )
    backend = args.component_attention_backends.get("transformer")
    if backend is None and args.is_arg_explicitly_set("attention_backend"):
        backend = args.attention_backend
    if backend is not None:
        model._component_attention_backend_override = AttentionBackendEnum[
            backend.upper()
        ]
    if any(
        next(module.parameters()).device != device for module in (model, text, video)
        if module is not None
    ):
        raise ValueError(
            "causal components require resident weights on the worker device"
        )
    from sglang.multimodal_gen.configs.models.vaes.minimax_h3_contract import (
        validate_minimax_h3_vae_latent_stats,
    )

    if video is not None:
        validate_minimax_h3_vae_latent_stats(
            video.sglang_config.arch_config,
            component_name="video_vae",
            expected_channels=24,
        )
        if encoder_group is not None:
            video._tile_group = encoder_group.coordinator
            if not video.encoder_tiling or not video.parallel_tiling:
                raise ValueError("H3 E2 requires encoder_tiling and parallel_tiling")
        # shortcut: loads the full VAE first; use encoder-only loading if peak memory matters.
        del video.decoder, video.post_quant_conv
    for module in (model, text, video):
        if module is not None:
            module.eval().requires_grad_(False)
    world = get_world_group() if encoder_group is not None else None
    tae_weights = getattr(args.pipeline_config, "tae_weights", None)
    tae = None
    if tae_weights is not None and (encoder_group is None
                                    or world.rank == encoder_group.members[0]):
        from sglang.multimodal_gen.runtime.models.vaes.taeh3 import MiniMaxH3TAEEncoder
        tae = MiniMaxH3TAEEncoder.from_weights(tae_weights, device=device)
    return H3CausalComponents(model, text, video, processor, encoder_group, world,
        proxy_encoder="tae" if tae_weights is not None else "h3", proxy_tae=tae)
