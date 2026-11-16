"""H3 causal ViT24 decoding from normalized latents to RGB.

Weights remain FP32; use torch.autocast for reduced-precision CUDA compute.
The ordinary decode/forward is differentiable. Per-stream methods are for
inference and retain only four past latents of K/V and one pending latent.
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file
from torch import nn

from .video_vae.causal_vit_decoder import CausalViTDecoder, PATCH_T
from .video_vae.processor import VAEProcessor


class CausalViTVideoDecoder(nn.Module):
    """Load a student checkpoint directly, without constructing its teacher.

    Inputs are normalized NTCHW H3 latents; outputs are NTCHW RGB in [0, 1].
    This is the StreamRender two-latent chunk / four-latent cache contract.
    ``backend='fast'`` uses fixed BF16 compute for eval/no_grad CUDA decoding,
    independent of caller autocast. Training and autograd use the reference.
    """

    latent_chunk = 2
    latent_channels = 24

    def __init__(self, decoder_config: dict | str | Path,
                 student_weights: str | Path, *, window: int = 16,
                 backend: str = "reference", compile: bool = False,
                 cuda_graph: bool = False, tp_group=None) -> None:
        super().__init__()
        if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
            raise ValueError("window must be a positive integer")
        if backend not in {"reference", "fast"}:
            raise ValueError("backend must be 'reference' or 'fast'")
        if backend == "reference" and (compile or cuda_graph):
            raise ValueError("compile and cuda_graph require backend='fast'")
        if tp_group is not None and (backend != "fast" or tp_group.world_size != 2):
            raise ValueError("decoder TP requires backend='fast' and a two-rank group")
        self.tp_group = tp_group
        self.backend, self.compile, self.cuda_graph = backend, compile, cuda_graph
        self._fast_backend, self._fast_signature, self._fast_generation = None, None, 0
        self.register_load_state_dict_post_hook(self._invalidate_fast)
        if isinstance(decoder_config, dict):
            config = decoder_config
        else:
            with open(decoder_config) as handle:
                config = yaml.safe_load(handle)
        kwargs = config["decoder_kwargs"]
        if (config["embed_dim"] != self.latent_channels
                or kwargs["in_channels"] != self.latent_channels
                or kwargs["out_channels"] != 3):
            raise ValueError("H3 decoder requires 24 latent channels and three RGB channels")
        self.vit = CausalViTDecoder.from_config(config, window=window, chunk=2, cache_latents=4)
        self.vit.load_state_dict(load_file(str(student_weights), device="cpu"), strict=True, assign=True)
        if any(t.is_meta for t in (*self.vit.parameters(), *self.vit.buffers())):
            raise RuntimeError("Student checkpoint left unmaterialized tensors")
        self.vit.float()
        self.processor = VAEProcessor(
            vae_ratio=16, vae_ratio_t=4, clip_length=17, frame_overlap=5,
            token_overlap=2, tokens_chunk_size=5, isolated_last_frame=False,
            latent_patch_size=1, crop_mode="top_left",
            pixel_norm_type=config["pixel_norm_type"], use_3d_conv=True,
        )

    def _validate(self, latents: torch.Tensor) -> None:
        if (latents.ndim != 5 or latents.shape[2] != self.latent_channels
                or min(latents.shape[0], *latents.shape[-2:]) <= 0
                or not latents.is_floating_point()):
            raise ValueError("Expected floating NTCHW H3 latents with 24 channels")

    def _rgb(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.processor.revert_tensor(pixels.float()).transpose(1, 2)

    def _invalidate_fast(self, *_):
        self._fast_backend = None
        self._fast_signature = None
        self._fast_generation += 1

    def _weight_signature(self):
        tensors = (*self.vit.parameters(), *self.vit.buffers())
        return tuple((p.data_ptr(), None if p.is_inference() else p._version, p.dtype, p.device)
                     for p in tensors)

    def _apply(self, fn, recurse=True):
        self._invalidate_fast()
        return super()._apply(fn, recurse=recurse)

    def train(self, mode=True):
        if mode:
            self._invalidate_fast()
        return super().train(mode)

    def _use_fast(self):
        if self.tp_group is not None and (self.training or torch.is_grad_enabled()):
            raise ValueError("decoder TP requires eval and no_grad")
        return self.backend == "fast" and not self.training and not torch.is_grad_enabled()

    def _make_fast_state(self, latents):
        from .video_vae.causal_vit_fast import FastCausalViT, FastDecodeState

        if latents.shape[0] != 1:
            raise ValueError("The fast decoder supports batch one")
        signature = self._weight_signature()
        if self._fast_backend is None or self._fast_signature != signature:
            self._fast_backend = FastCausalViT(self.vit, compile=self.compile, cuda_graph=self.cuda_graph,
                tp_group=self.tp_group)
            self._fast_signature = signature
        return FastDecodeState(self._fast_backend, *latents.shape[-2:])

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a complete sequence, including the stream's padded odd tail."""
        self._validate(latents)
        latents = latents.float()
        count = latents.shape[1]
        if count == 0:
            return latents.new_empty(latents.shape[0], 0, 3,
                                     latents.shape[-2] * 16, latents.shape[-1] * 16)
        if self._use_fast():
            stream = self.create_stream()
            parts = [stream.decode(latents), stream.flush()]
            return torch.cat([part for part in parts if part is not None], dim=1)
        if count % self.latent_chunk:
            latents = torch.cat((latents, latents[:, -1:]), dim=1)
        pixels = self.vit(latents)
        frames = PATCH_T * count - self.vit.frames_to_trim * (
            (count + self.vit.block_latents - 1) // self.vit.block_latents
        )
        return self._rgb(pixels[:, :, :frames])

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode(latents)

    def create_stream(self) -> CausalViTDecodeStream:
        return CausalViTDecodeStream(self)


class CausalViTDecodeStream:
    """Independent inference state. Flush ends the stream; reset starts another."""

    requires_single_latent_dispatch = False

    def __init__(self, codec: CausalViTVideoDecoder) -> None:
        self.codec = codec
        self.reset()

    def reset(self) -> None:
        fast = getattr(self, "fast", None)
        if (fast is not None and self.fast_generation == self.codec._fast_generation
                and self.fast_signature == self.codec._weight_signature()):
            fast.reset()
        else:
            fast = None
        self.state = self.codec.vit.init_state()
        self.pending: torch.Tensor | None = None
        self.geometry = None
        self.closed = False
        self.fast = fast
        self.fast_signature = self.codec._fast_signature
        self.fast_generation = self.codec._fast_generation

    def _chunk(self, latents):
        if self.fast_generation != self.codec._fast_generation:
            raise RuntimeError("Decoder parameters changed; reset the stream")
        if self.codec._use_fast():
            if self.fast is None:
                if self.state["t"]:
                    raise RuntimeError("Decoder mode changed; reset the stream")
                self.fast = self.codec._make_fast_state(latents)
                self.fast_signature = self.codec._fast_signature
            if self.fast_signature != self.codec._weight_signature():
                raise RuntimeError("Decoder weights changed; reset the stream")
            result = self.fast.decode_chunk(latents)
            self.state["t"] = self.fast.t
            return result
        if self.fast is not None:
            raise RuntimeError("Decoder mode changed; reset the stream")
        return self.codec.vit.decode_chunk(latents, self.state)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor | None:
        self.codec._validate(latents)
        latents = latents.float()
        if self.closed:
            raise RuntimeError("Decoder stream was flushed; call reset before decoding again")
        if latents.shape[1] == 0:
            return None
        geometry = (latents.shape[0], *latents.shape[-2:], latents.device, latents.dtype)
        if self.geometry is not None and geometry != self.geometry:
            raise ValueError("Decoder stream batch, resolution, device and dtype must stay fixed")
        self.geometry = geometry
        if self.pending is not None:
            latents = torch.cat((self.pending, latents), dim=1)
        stop = latents.shape[1] // self.codec.latent_chunk * self.codec.latent_chunk
        outputs = [
            self._chunk(latents[:, i:i + self.codec.latent_chunk])
            for i in range(0, stop, self.codec.latent_chunk)
        ]
        # Own the small tail, rather than retaining or sharing the caller's full input.
        self.pending = latents[:, stop:].detach().clone() if stop < latents.shape[1] else None
        return self.codec._rgb(torch.cat(outputs, dim=2)) if outputs else None

    push = decode

    @torch.no_grad()
    def flush(self) -> torch.Tensor | None:
        if self.closed:
            return None
        self.closed = True
        if self.pending is None:
            return None
        z, self.pending = self.pending, None
        frames = PATCH_T - (
            self.codec.vit.frames_to_trim
            if self.state["t"] % self.codec.vit.block_latents == 0 else 0
        )
        pixels = self._chunk(torch.cat((z, z), dim=1))
        return self.codec._rgb(pixels[:, :, :frames])

    flush_decoder = flush
