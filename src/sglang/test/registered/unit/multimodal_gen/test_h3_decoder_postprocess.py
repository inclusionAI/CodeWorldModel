# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.causal_streaming import (
    H3CausalVideoDecoder,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.processor import (
    VAEProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def decoder(norm, **kwargs):
    result = H3CausalVideoDecoder.__new__(H3CausalVideoDecoder)
    result.device = torch.device("cpu")
    result.sdpa_backends = None
    result.processor = VAEProcessor(
        vae_ratio=16, vae_ratio_t=4, clip_length=17, frame_overlap=5,
        token_overlap=2, tokens_chunk_size=5, isolated_last_frame=False,
        latent_patch_size=1, crop_mode="top_left", pixel_norm_type=norm,
        use_3d_conv=True, **kwargs,
    )
    return result


@pytest.mark.parametrize("norm", ["imagenet", "simple", "raw"])
def test_owned_rgb_matches_reference_and_preserves_published_bytes(norm):
    model = decoder(norm)
    pixels = torch.linspace(-4, 4, 2 * 3 * 4 * 5).reshape(1, 3, 2, 4, 5)
    expected = model.processor.revert_tensor(pixels.clone()).transpose(1, 2)
    actual = model._rgb(pixels.clone())
    assert torch.equal(actual, expected)
    expected_bytes = (expected[0].clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).contiguous()
    model._push = lambda latent: model._rgb(pixels.clone())
    model._flush = lambda: None
    latent = torch.zeros(24, 2, 1, 1)
    published = model.decode(latent, eos=False)
    assert torch.equal(published, expected_bytes)
    model.decode(latent, eos=True)
    assert torch.equal(published, expected_bytes)
    assert torch.count_nonzero(latent) == 0


def test_custom_reverse_transform_without_owned_variant_keeps_original_contract():
    model = decoder("raw", transform_rev=lambda value: value + 0.25)
    pixels = torch.zeros(1, 3, 2, 4, 5)
    assert torch.equal(model._rgb(pixels), torch.full((1, 2, 3, 4, 5), 0.25))
    assert torch.count_nonzero(pixels) == 0
