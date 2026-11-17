# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from projects.minimax_h3.modeling.inference_runtime import InferenceCodec
from projects.minimax_h3.modeling.video_vae.causal_vit_decoder import CausalViTDecoder
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.causal_streaming import (
    H3CausalVideoDecoder,
)
from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module

CONFIG = dict(
    embed_dim=24,
    pixel_norm_type='imagenet',
    latents_mean=[0.1 * i for i in range(24)],
    latents_std=[1.0 + 0.05 * i for i in range(24)],
    decoder_kwargs=dict(
        in_channels=24, out_channels=3, num_layers=3, heads=2, dim_head=16,
        norm_type='rms_norm', norm_affine=True, qk_norm_type='rms_norm',
        qk_norm_affine=False, ffn_activation_fn='silu', ffn_use_gated=True,
        rope_dim_ratio=0.75, rope_theta=100.0,
    ),
)
WINDOW = 2


@pytest.fixture(autouse=True)
def server_args():
    # SGLang's ViT block builds a USPAttention on CUDA hosts, which reads global server args.
    previous = server_args_module._global_server_args
    server_args_module.set_global_server_args(SimpleNamespace(
        attention_backend=None, attention_backend_config=None, kv_gather_degree=1,
        sp_split_auto=False, pipeline_config=SimpleNamespace(dit_precision='bfloat16')))
    yield
    server_args_module.set_global_server_args(previous)


@pytest.fixture(scope='module')
def weights(tmp_path_factory):
    generator = torch.Generator().manual_seed(0)
    shapes = CausalViTDecoder.from_config(CONFIG, window=WINDOW).state_dict()
    state = {name: torch.randn(value.shape, generator=generator) * 0.2
             for name, value in shapes.items()}
    path = tmp_path_factory.mktemp('decoder') / 'student.safetensors'
    save_file(state, str(path))
    return str(path)


def run(decoder, latents, splits):
    outputs, start = [], 0
    for index, count in enumerate(splits):
        rgb = decoder.decode(latents[:, start:start + count], eos=index == len(splits) - 1)
        outputs.append(rgb)
        start += count
    assert start == latents.shape[1]
    return outputs


@pytest.mark.parametrize('splits', [(1, 2, 3, 0, 4, 1), (5, 4), (2, 2, 2)])
def test_port_matches_parent_inference_codec_bitwise_across_sessions(weights, splits):
    options = dict(decoder='causal', decoder_config=CONFIG, decoder_weights=weights,
                   window=WINDOW, backend='reference', compile=False, cuda_graph=False)
    parent = InferenceCodec(None, options, device='cpu', load_encoder=False)
    port = H3CausalVideoDecoder(decoder_config=CONFIG, decoder_weights=weights, device='cpu',
                                window=WINDOW, backend='reference')
    latents = torch.randn(24, sum(splits), 3, 5, generator=torch.Generator().manual_seed(1))
    count = latents.shape[1]
    first = None
    for _ in range(2):
        parent.begin_session()
        port.begin_session()
        expected, actual = run(parent, latents, splits), run(port, latents, splits)
        assert [None if x is None else tuple(x.shape) for x in actual] == \
               [None if x is None else tuple(x.shape) for x in expected]
        for want, got in zip(expected, actual):
            if want is not None:
                assert got.dtype == torch.uint8 and torch.equal(got, want)
        frames = sum(x.shape[0] for x in actual if x is not None)
        assert frames == 4 * count - 3 * ((count + 4) // 5)
        video = torch.cat([x for x in actual if x is not None])
        assert video.unique().numel() > 64, 'random weights must not saturate the comparison'
        assert first is None or torch.equal(video, first), 'begin_session must reset the stream'
        first = video
    with pytest.raises(RuntimeError, match='flushed'):
        port.decode(latents[:, :1], eos=False)
