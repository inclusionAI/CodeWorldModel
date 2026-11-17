# SPDX-License-Identifier: Apache-2.0
"""One H3 causal ViT decode stream: GPU CTHW latents in, GPU THWC uint8 RGB out.

The stream holds four past latents of K/V and at most one pending latent;
EOS flushes that tail, and ``begin_session`` starts the next stream.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from .causal_vit_decoder import PATCH_T, CausalViTDecoder
from .processor import VAEProcessor

LATENT_CHUNK = 2
LATENT_CHANNELS = 24


def _load_student(
    decoder_config: dict | str | Path, student_weights: str | Path, *, window: int
) -> tuple[CausalViTDecoder, str]:
    if isinstance(decoder_config, dict):
        config = decoder_config
    else:
        with open(decoder_config) as handle:
            config = yaml.safe_load(handle)
    kwargs = config["decoder_kwargs"]
    if (
        config["embed_dim"] != LATENT_CHANNELS
        or kwargs["in_channels"] != LATENT_CHANNELS
        or kwargs["out_channels"] != 3
    ):
        raise ValueError(
            "H3 decoder requires 24 latent channels and three RGB channels"
        )
    vit = CausalViTDecoder(config, window=window, chunk=LATENT_CHUNK, cache_latents=4)
    vit.load_state_dict(
        load_file(str(student_weights), device="cpu"), strict=True, assign=True
    )
    if any(t.is_meta for t in (*vit.parameters(), *vit.buffers())):
        raise RuntimeError("Student checkpoint left unmaterialized tensors")
    return vit.float(), config["pixel_norm_type"]


class H3CausalVideoDecoder:
    """``backend='fast'`` decodes in fixed BF16 on CUDA, batch one."""

    def __init__(
        self,
        *,
        decoder_config: dict | str | Path,
        decoder_weights: str | Path,
        device: torch.device | str,
        window: int = 16,
        backend: str = "reference",
        compile: bool = False,
        cuda_graph: bool = False,
        tp_group=None,
        cudnn_sdpa: bool = False,
        spatial_attention_backend: str = "fa2",
        compact_suffix_kv: bool = False,
        suffix_attention_backend: str = "sdpa",
        compile_mode: str = "default",
        tp_fused_post: bool = False,
    ) -> None:
        if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
            raise ValueError("window must be a positive integer")
        if backend not in {"reference", "fast"}:
            raise ValueError("backend must be 'reference' or 'fast'")
        if backend == "reference" and (compile or cuda_graph or compile_mode != "default" or tp_fused_post):
            raise ValueError("compile and cuda_graph require backend='fast'")
        if tp_group is not None and (backend != "fast" or tp_group.world_size not in (2, 4)):
            raise ValueError("decoder TP requires backend='fast' and a two- or four-rank group")
        if suffix_attention_backend != "sdpa" and backend != "fast":
            raise ValueError("FA3 suffix attention requires backend='fast'")
        self.device = torch.device(device)
        vit, pixel_norm_type = _load_student(
            decoder_config, decoder_weights, window=window
        )
        self.vit = vit.to(self.device).eval().requires_grad_(False)
        self.processor = VAEProcessor(
            vae_ratio=16,
            vae_ratio_t=4,
            clip_length=17,
            frame_overlap=5,
            token_overlap=2,
            tokens_chunk_size=5,
            isolated_last_frame=False,
            latent_patch_size=1,
            crop_mode="top_left",
            pixel_norm_type=pixel_norm_type,
            use_3d_conv=True,
        )
        self.sdpa_backends = [SDPBackend.CUDNN_ATTENTION] if cudnn_sdpa else None
        self.fast_backend = None
        if backend == "fast":
            from .causal_vit_fast import FastCausalViT

            self.fast_backend = FastCausalViT(
                self.vit, compile=compile, cuda_graph=cuda_graph, tp_group=tp_group,
                spatial_attention_backend=spatial_attention_backend,
                compact_suffix_kv=compact_suffix_kv,
                suffix_attention_backend=suffix_attention_backend,
                compile_mode=compile_mode,
                tp_fused_post=tp_fused_post,
            )
        # Fixed-resolution FA2 buffers survive begin_session; built on the first chunk.
        self.fast_state = None
        self.begin_session()

    def begin_session(self) -> None:
        if self.fast_state is not None:
            self.fast_state.reset()
        self.state = self.vit.init_state()
        self.pending: torch.Tensor | None = None
        self.geometry = None
        self.closed = False

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, *, eos: bool) -> torch.Tensor | None:
        """CTHW normalized latents -> THWC uint8 frames, or None when no frame is ready."""
        sdpa = (
            nullcontext()
            if self.sdpa_backends is None
            else sdpa_kernel(self.sdpa_backends)
        )
        with sdpa, torch.autocast(self.device.type, dtype=torch.bfloat16):
            pixels = self._push(latent.permute(1, 0, 2, 3).unsqueeze(0))
            tail = self._flush() if eos else None
        if tail is not None:
            pixels = tail if pixels is None else torch.cat((pixels, tail), dim=1)
        if pixels is None:
            return None
        return pixels[0].mul_(255).byte().permute(0, 2, 3, 1).contiguous()

    def _push(self, latents: torch.Tensor) -> torch.Tensor | None:
        if (
            latents.ndim != 5
            or latents.shape[2] != LATENT_CHANNELS
            or min(latents.shape[0], *latents.shape[-2:]) <= 0
            or not latents.is_floating_point()
        ):
            raise ValueError("Expected floating CTHW H3 latents with 24 channels")
        latents = latents.float()
        if self.closed:
            raise RuntimeError(
                "Decoder stream was flushed; call begin_session before decoding again"
            )
        if latents.shape[1] == 0:
            return None
        geometry = (latents.shape[0], *latents.shape[-2:], latents.device)
        if self.geometry is not None and geometry != self.geometry:
            raise ValueError("Decoder stream batch, resolution and device must stay fixed")
        self.geometry = geometry
        if self.pending is not None:
            latents = torch.cat((self.pending, latents), dim=1)
        stop = latents.shape[1] // LATENT_CHUNK * LATENT_CHUNK
        outputs = [
            self._chunk(latents[:, i : i + LATENT_CHUNK])
            for i in range(0, stop, LATENT_CHUNK)
        ]
        # Own the small tail rather than retaining the caller's full input.
        self.pending = (
            latents[:, stop:].detach().clone() if stop < latents.shape[1] else None
        )
        return self._rgb(torch.cat(outputs, dim=2)) if outputs else None

    def _flush(self) -> torch.Tensor | None:
        if self.closed:
            return None
        self.closed = True
        if self.pending is None:
            return None
        z, self.pending = self.pending, None
        # The odd tail decodes as a duplicated pair; keep only its own frames.
        frames = PATCH_T - (
            self.vit.frames_to_trim
            if self.state.t % self.vit.block_latents == 0
            else 0
        )
        pixels = self._chunk(torch.cat((z, z), dim=1))
        return self._rgb(pixels[:, :, :frames])

    def _chunk(self, latents: torch.Tensor) -> torch.Tensor:
        if self.fast_backend is None:
            return self.vit.decode_chunk(latents, self.state)
        if self.fast_state is None:
            from .causal_vit_fast import FastDecodeState

            if latents.shape[0] != 1:
                raise ValueError("The fast decoder supports batch one")
            self.fast_state = FastDecodeState(self.fast_backend, *latents.shape[-2:])
        result = self.fast_state.decode_chunk(latents)
        self.state.t = self.fast_state.t
        return result

    def _rgb(self, pixels: torch.Tensor) -> torch.Tensor:
        # NCTHW decode_base space -> NTCHW RGB in [0, 1].
        # _push/_flush own these pixels; graph-owned outputs were copied in _chunk.
        return self.processor.revert_tensor(pixels.float(), runtime_owned=True).transpose(1, 2)
