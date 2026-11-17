"""TAE encoder and causal ViT decoder behind RAVEN's video-VAE interface.

All public latents are normalized H3 diffusion latents. Public pixels are
RGB in [0, 1]; identity statistics and a raw processor preserve the existing
meta-model normalization contract. The original H3 VAE remains unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn

from .streaming_video_decoder import CausalViTVideoDecoder
from .taeh3 import MiniMaxH3NativeStreamingTAE


class StreamRenderH3VideoVAE(nn.Module):
    """Strictly load a TAE encoder and a standalone causal ViT student.

    Constructor loading requires `meta_init: false` in RAVEN model configs.
    Keep parameters FP32 and use CUDA autocast for reduced-precision compute.
    """

    latent_channels = 24
    image_channels = 3
    vae_ratio = 16
    vae_ratio_t = 4
    use_3d_conv = True

    def __init__(
        self,
        *,
        tae_weights: str | Path,
        decoder_weights: str | Path,
        decoder_config: dict | str | Path,
        window: int = 16,
        parallel_encode: bool = True,
        backend: str = "reference",
        compile: bool = False,
        cuda_graph: bool = False,
    ) -> None:
        super().__init__()
        self.tae = MiniMaxH3NativeStreamingTAE(parallel_encode=parallel_encode)
        path = Path(tae_weights)
        state = (
            load_file(str(path), device="cpu")
            if path.suffix in {".safetensors", ".safetensor"}
            else torch.load(path, map_location="cpu", weights_only=True)
        )
        self.tae.load_state_dict(state, strict=True, assign=True)
        del state
        if any(t.is_meta for t in (*self.tae.parameters(), *self.tae.buffers())):
            raise RuntimeError("TAE weights were not materialized; use meta_init: false")
        self.tae.float()
        self.decoder = CausalViTVideoDecoder(
            decoder_config, decoder_weights, window=window,
            backend=backend, compile=compile, cuda_graph=cuda_graph,
        )
        self.processor = self.tae.processor
        self.register_buffer("latents_mean", torch.zeros(24), persistent=False)
        self.register_buffer("latents_std", torch.ones(24), persistent=False)
        self.param_dtype = torch.float32

    def encode_images(
        self, images: Any, transform_input: bool = False,
        use_fp16_latent: bool = False, verbose: bool = False,
    ) -> list[torch.Tensor]:
        """Return normalized C1HW latents; PIL/numpy inputs use raw RGB."""
        return self.tae.encode_images(
            images, transform_input=transform_input,
            use_fp16_latent=use_fp16_latent, verbose=verbose,
        )

    def encode_videos(
        self, videos: Any, transform_input: bool = False,
        use_fp16_latent: bool = False, verbose: bool = False,
        encode_prefix: bool = False,
    ) -> list[torch.Tensor]:
        """Return normalized CTHW latents for 17n or 17n+5 RGB clips."""
        return self.tae.encode_videos(
            videos, transform_input=transform_input,
            use_fp16_latent=use_fp16_latent, verbose=verbose,
            encode_prefix=encode_prefix,
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Differentiable NCHW RGB -> N1CHW normalized latent."""
        return self.tae.encode_image(images)

    def encode_video(self, pixels: torch.Tensor) -> torch.Tensor:
        """Differentiable NTCHW RGB -> NTCHW normalized latents."""
        return self.tae.encode_video(pixels)

    def decode_base(
        self, latent: torch.Tensor, frame_num: int | None = None,
        process_image: bool = False,
    ) -> torch.Tensor:
        """Decode normalized NCTHW to RGB NCTHW (NCHW for an image).

        Like the original H3 VAE, a smaller frame_num keeps the trailing
        frames. Streaming prefix publication uses create_stream instead.
        """
        if latent.ndim == 4 and process_image:
            latent = latent.unsqueeze(2)
        if latent.ndim != 5 or latent.shape[1] != self.latent_channels:
            raise ValueError("decode_base expects normalized NCTHW with 24 channels")
        if latent.shape[2] <= 0:
            raise ValueError("decode_base requires at least one latent frame")
        if process_image and latent.shape[2] != 1:
            raise ValueError("process_image requires exactly one latent frame")
        if frame_num is not None and (
            isinstance(frame_num, bool) or not isinstance(frame_num, int) or frame_num <= 0
        ):
            raise ValueError("frame_num must be a positive integer")
        pixels = self.decoder.decode(latent.transpose(1, 2)).transpose(1, 2)
        if frame_num is not None:
            if frame_num > pixels.shape[2]:
                raise ValueError("frame_num exceeds the decoded RGB frame count")
            pixels = pixels[:, :, -frame_num:]
        return (pixels.squeeze(2) if process_image else pixels).contiguous()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Differentiable normalized NCTHW -> RGB NCTHW."""
        return self.decode_base(latent)

    def create_stream(self) -> StreamRenderH3Stream:
        return StreamRenderH3Stream(self)


class StreamRenderH3Stream:
    """Independent native encoder and two-latent causal decoder state.

    encode/decode use NTCHW throughout. The encoder ignores an incomplete
    four-frame group at flush; the decoder flushes its one pending latent.
    Start a new stream after changing the codec device or dtype.
    """

    pixel_frames_per_block = 17
    latent_frames_per_block = 5
    requires_single_frame_encode_dispatch = True
    requires_single_latent_dispatch = False

    def __init__(self, codec: StreamRenderH3VideoVAE) -> None:
        self.encoder = codec.tae.create_stream()
        self.decoder = codec.decoder.create_stream()

    def reset(self) -> None:
        self.encoder.reset()
        self.decoder.reset()

    def encode_one(self, pixel: torch.Tensor) -> torch.Tensor | None:
        return self.encoder.encode_one(pixel)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor | None:
        return self.encoder.encode(pixels)

    def flush_encoder(self) -> None:
        return self.encoder.flush_encoder()

    def decode(self, latents: torch.Tensor) -> torch.Tensor | None:
        return self.decoder.decode(latents)

    def flush_decoder(self) -> torch.Tensor | None:
        return self.decoder.flush()
