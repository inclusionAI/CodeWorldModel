# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor

from sglang.multimodal_gen.runtime.loader.minimax_h3_causal import (
    H3CausalComponents,
    load_h3_causal_components,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.conv import BaseConv3d
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.norm import (
    TemporalIsolatedGroupNorm,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.processor import VAEProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _tokenizer():
    # A real byte-level tokenizer, with no checkpoint files or network downloads.
    special = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE({v: i for i, v in enumerate(alphabet + special)}, []))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, additional_special_tokens=special)


class _CaptureText(torch.nn.Module):
    def encode_ids(self, ids, **vision):
        self.ids, self.vision = ids, vision
        return ids[:, None].float()


def test_first_rgb_reaches_qwen_as_one_labelled_image():
    tokenizer = _tokenizer()
    processor = SimpleNamespace(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(patch_size=14, temporal_patch_size=2, merge_size=2),
    )
    first = (np.arange(64 * 96 * 3).reshape(64, 96, 3) % 251).astype(np.uint8)
    text = _CaptureText()
    components = H3CausalComponents(torch.nn.Linear(1, 1), text, None, processor)
    prompt = "Keep moving.\nPreserve the original style."
    hidden, tags = components.condition(prompt, first)
    expected_vision = processor.image_processor(images=[first], return_tensors="pt")
    grid = expected_vision["image_grid_thw"]
    assert set(text.vision) == {"pixel_values", "image_grid_thw"}
    for key, value in text.vision.items():
        torch.testing.assert_close(value, expected_vision[key], rtol=0, atol=0)
    count = int(grid[0].prod()) // processor.image_processor.merge_size**2
    prefix_ids = tokenizer("<Picture 1>: ", add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    vision_ids = [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
    vision_ids += [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * count
    vision_ids += [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
    assert text.ids.tolist() == prefix_ids + vision_ids + prompt_ids
    assert tags.tolist() == [1] * len(prefix_ids) + [0] * len(vision_ids) + [1] * len(prompt_ids)
    assert hidden.shape == (len(text.ids), 1)


class _MomentBoundary(torch.nn.Module):
    """Capture the real pixel processor's output at the expensive VAE boundary."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((), dtype=torch.bfloat16))
        self.processor = VAEProcessor(
            vae_ratio=16, vae_ratio_t=4, clip_length=17, frame_overlap=0,
            token_overlap=0, tokens_chunk_size=None, isolated_last_frame=False,
            latent_patch_size=2, crop_mode="center", use_3d_conv=True,
        )
        self.sglang_config = SimpleNamespace(arch_config=SimpleNamespace(
            latents_mean=[0.125] * 24, latents_std=[1.5] * 24,
        ))

    def _adaptive_encode(self, pixels):
        self.pixels = pixels.clone()
        t = 5 if pixels.shape[2] == 17 else 1
        shape = (1, 24, t, pixels.shape[-2] // 16, pixels.shape[-1] // 16)
        mean = torch.full(shape, 0.234567, dtype=pixels.dtype)
        logvar = torch.full(shape, -1.234567, dtype=pixels.dtype)
        self.moments = torch.cat((mean, logvar), dim=1)
        return self.moments


@pytest.mark.parametrize("mode,frames,count", [("image", 1, 1), ("proxy", 17, 5), ("eos", 5, 2)])
def test_conditioning_latents_are_the_bf16_posterior_mean(mode, frames, count):
    vae = _MomentBoundary()
    components = H3CausalComponents(torch.nn.Linear(1, 1), _CaptureText(), vae, None)
    rgb = np.full((frames, 32, 64, 3), 117, dtype=np.uint8)
    output = components.encode_image(rgb[0]) if mode == "image" else components.encode_proxy_local(rgb, eos=mode == "eos")
    assert vae.pixels.dtype == torch.bfloat16
    normalized = (117 / 255.0 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(vae.pixels[0, :, 0, 0, 0], normalized.to(torch.bfloat16))
    assert vae.pixels.shape[2] == (1 if mode == "image" else 17)
    # No posterior noise: a nonzero log-variance must not move the result off the mean.
    mean = vae.moments.float().chunk(2, dim=1)[0]
    expected = ((mean[:, :, :count] - 0.125) / 1.5)[0].to(torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.shape == (24, count, 2, 4) and output.dtype == torch.bfloat16


@pytest.mark.parametrize("size", [(256, 448), (768, 1344)])
def test_wire_pixels_follow_the_selected_encoder_size(monkeypatch, size):
    received = []

    def encode(self, rgb, **kw):
        received.append(rgb)
        return torch.zeros(24, 5, rgb.shape[1] // 16, rgb.shape[2] // 16)

    monkeypatch.setattr(H3CausalComponents, "_encode_local", encode)
    components = H3CausalComponents(torch.nn.Linear(1, 1), None, None, None,
        proxy_encode_size=None if size == (256, 448) else size)
    wire = np.zeros((17, 256, 448, 3), dtype=np.uint8)
    wire[:, :, :, 0] = 255
    out = components.encode_proxy_local(wire, eos=False)
    assert wire.shape == (17, 256, 448, 3) and out.shape == (24, 5, size[0] // 16, size[1] // 16)
    assert received[0].shape == (17, *size, 3) and received[0][0, 0, 0].tolist() == [255, 0, 0]
    if size == (256, 448):
        assert received[0] is wire


@pytest.mark.parametrize("eos", [False, True])
def test_nearest_encoder_resize_matches_pixel_replication(eos):
    class Processor:
        @staticmethod
        def convert_numpy_to_tensor(rgb, device):
            return torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float().div_(255)

        @staticmethod
        def transform_tensor(pixels):
            return pixels

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.processor = Processor()
            self.sglang_config = SimpleNamespace(arch_config=SimpleNamespace(
                latents_mean=[0.0] * 24, latents_std=[1.0] * 24))

        def _adaptive_encode(self, pixels):
            self.pixels = pixels
            return torch.zeros(1, 48, 5, pixels.shape[-2] // 16, pixels.shape[-1] // 16)

    vae = VAE()
    components = H3CausalComponents(torch.nn.Linear(1, 1), None, vae, None,
        proxy_encode_size=(96, 192), proxy_resize="nearest")
    rgb = np.random.default_rng(3).integers(0, 256, size=(5 if eos else 17, 32, 64, 3), dtype=np.uint8)
    result = components.encode_proxy_local(rgb, eos=eos)
    expected = np.repeat(np.repeat(rgb, 3, axis=1), 3, axis=2)
    expected = Processor.convert_numpy_to_tensor(expected, "cpu").transpose(0, 1)[None].bfloat16()
    torch.testing.assert_close(vae.pixels[:, :, :len(rgb)], expected, rtol=0, atol=0)
    assert result.shape == (24, 2 if eos else 5, 6, 12)


def test_first_moment_is_independent_of_future_repeats_in_causal_encoder_primitives():
    # Exercise real encoder operators rather than assuming a one-frame shortcut.
    encoder = torch.nn.Sequential(
        BaseConv3d(3, 4, 3, padding=1, causal=True),
        TemporalIsolatedGroupNorm(2, 4),
        torch.nn.SiLU(),
        BaseConv3d(4, 4, 3, stride=(2, 1, 1), padding=1, causal=True),
        TemporalIsolatedGroupNorm(2, 4),
    ).eval()
    first = torch.arange(3 * 8 * 8).float().reshape(1, 3, 1, 8, 8) / 255
    with torch.no_grad():
        single = encoder(first)
        repeated = encoder(first.expand(-1, -1, 17, -1, -1).contiguous())
    torch.testing.assert_close(single, repeated[:, :, :1], rtol=1e-5, atol=1e-6)


def test_vae_precision_is_bf16_before_checkpoint_loader(monkeypatch):
    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import MiniMaxH3PipelineConfig
    from sglang.multimodal_gen.runtime import distributed
    from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import TextEncoderLoader
    from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import TransformerLoader
    from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader

    args = SimpleNamespace(
        pipeline_config=MiniMaxH3PipelineConfig(), component_precisions={"video_vae": "fp32"},
        model_variant="ref2va", component_paths={name: name for name in ("transformer", "text_encoder", "video_vae", "processor")},
    )
    monkeypatch.setattr(distributed, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_local_torch_device", lambda: torch.device("cpu"))

    def unused_load(self, *args, **kwargs):
        return torch.nn.Linear(1, 1), None

    monkeypatch.setattr(TransformerLoader, "load", unused_load)
    monkeypatch.setattr(TextEncoderLoader, "load", unused_load)

    class BoundaryReached(Exception):
        pass

    def capture_load(self, path, configured, name, library, **kwargs):
        assert name == "video_vae"
        assert configured.component_precisions[name] == "bf16"
        assert kwargs["allow_native_fallback"] is False
        raise BoundaryReached

    monkeypatch.setattr(VAELoader, "load", capture_load)
    with pytest.raises(BoundaryReached):
        load_h3_causal_components(args)
