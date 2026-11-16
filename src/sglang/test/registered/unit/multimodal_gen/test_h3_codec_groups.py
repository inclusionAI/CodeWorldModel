# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.loader.minimax_h3_causal import (
    H3CausalComponents, H3CodecGroup, bind_h3_codec_groups,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import klvae
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.srt.distributed.device_communicators import custom_all_reduce_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize('cuda', [False, True])
@pytest.mark.parametrize('encoder_ranks', [None, [0, 1], [2, 3], [4, 5], [6, 7], [2, 3, 6, 7]])
def test_group_creation_includes_idle_pairs_in_deterministic_order(monkeypatch, cuda, encoder_ranks):
    calls = []
    checks = []
    world = SimpleNamespace(world_size=8, ranks=list(range(8)), local_rank=0, device_group=object())
    monkeypatch.setattr(parallel_state, 'get_world_group', lambda: world)
    monkeypatch.setattr(torch.distributed, 'get_backend', lambda group: 'gloo')
    monkeypatch.setattr(current_platform, 'is_cuda', lambda: cuda)
    def p2p(src, size):
        checks.append((world.rank, src, size))
        return False
    monkeypatch.setattr(custom_all_reduce_utils, 'can_p2p', p2p)
    for rank in range(8):
        world.rank = rank
        def create(partitions, **kwargs):
            assert checks == ([(r, 0, 2) for r in range(rank + 1)] if cuda else [])
            calls.append((partitions, kwargs))
            return SimpleNamespace(rank=rank, ranks=next(p for p in partitions if rank in p))
        monkeypatch.setattr(parallel_state, 'GroupCoordinator', create)
        encoder, decoder = bind_h3_codec_groups(
            **({} if encoder_ranks is None else {'encoder_ranks': encoder_ranks}))
        assert encoder.members == tuple(encoder_ranks or [0, 1])
        assert encoder.is_member == (rank in encoder.members)
        assert decoder.is_member == (rank < 2)
        assert encoder.coordinator is not decoder.coordinator
        assert [entry[1]['group_name'] for entry in calls[-2:]] == ['h3_encoder', 'h3_decoder']
    assert all(sorted(sum(entry[0], [])) == list(range(8)) for entry in calls)
    assert all(entry[1]['use_device_communicator'] == entry[1]['use_srt_custom_allreduce']
               for entry in calls)
    assert all(entry[1]['use_srt_custom_allreduce'] == (index in (1, 3))
               for index, entry in enumerate(calls))
    with pytest.raises(ValueError, match='output rank 0'):
        bind_h3_codec_groups(output_rank=1)


@pytest.mark.parametrize('ranks', [None, 2, [], [0], [0, 0], [0, 2], [1, 2], [3, 2],
    [6, 8], [-2, -1], [0, 1, 2], [False, True], [0.0, 1.0], ['2', '3']])
def test_invalid_encoder_membership_fails_before_preflight_or_group_creation(monkeypatch, ranks):
    world = SimpleNamespace(world_size=8, ranks=list(range(8)))
    monkeypatch.setattr(parallel_state, 'get_world_group', lambda: world)
    def no_communication(*args, **kwargs):
        raise AssertionError('invalid ranks reached communication setup')
    monkeypatch.setattr(custom_all_reduce_utils, 'can_p2p', no_communication)
    monkeypatch.setattr(parallel_state, 'GroupCoordinator', no_communication)
    with pytest.raises(ValueError, match='pair encoder'):
        bind_h3_codec_groups(encoder_ranks=ranks)


@pytest.mark.parametrize('ranks', [None, [2, 3], [1, 0], [False, True]])
def test_decoder_membership_stays_on_output_group(monkeypatch, ranks):
    monkeypatch.setattr(parallel_state, 'get_world_group',
        lambda: SimpleNamespace(world_size=8, ranks=list(range(8))))
    with pytest.raises(ValueError, match='pair-aligned decoder'):
        bind_h3_codec_groups(decoder_ranks=ranks)


def test_tp4_decoder_group_uses_idle_pairs_in_one_communicator(monkeypatch):
    calls = []
    world = SimpleNamespace(world_size=8, ranks=list(range(8)), local_rank=0,
        device_group=object(), rank=0)
    monkeypatch.setattr(parallel_state, 'get_world_group', lambda: world)
    monkeypatch.setattr(torch.distributed, 'get_backend', lambda group: 'gloo')
    monkeypatch.setattr(current_platform, 'is_cuda', lambda: False)
    def create(partitions, **kwargs):
        calls.append((partitions, kwargs))
        return SimpleNamespace(rank=world.rank,
            ranks=next(p for p in partitions if world.rank in p))
    monkeypatch.setattr(parallel_state, 'GroupCoordinator', create)
    _, decoder = bind_h3_codec_groups(encoder_ranks=(2, 3), decoder_ranks=(0, 1, 4, 5))
    assert calls[0][0] == [[2, 3], [0, 1], [4, 5], [6, 7]]
    assert calls[1][0] == [[0, 1, 4, 5], [2, 3], [6, 7]]
    assert decoder.members == (0, 1, 4, 5)


def test_official_p2p_preflight_preserves_skip_check(monkeypatch):
    monkeypatch.setenv('SGLANG_SKIP_P2P_CHECK', '1')
    checks = []
    monkeypatch.setattr(torch.cuda, 'can_device_access_peer',
        lambda src, tgt: checks.append((src, tgt)) or True)
    def no_probe(*args):
        raise AssertionError('skip-check requested the world-barrier P2P probe')
    monkeypatch.setattr(custom_all_reduce_utils, 'gpu_p2p_access_check', no_probe)
    assert custom_all_reduce_utils.can_p2p(0, 2)
    assert checks == [(0, 1)]


@pytest.mark.parametrize('mode,frames,count', [('image', 1, 1), ('proxy', 17, 5), ('eos', 5, 2)])
@pytest.mark.parametrize('encoder_ranks', [(0, 1), (2, 3), (4, 5), (6, 7), (2, 3, 6, 7)])
def test_full_bf16_latents_reach_every_rank_without_an_encoder_on_followers(
        monkeypatch, mode, frames, count, encoder_ranks):
    expected = torch.randn(24, count, 16, 28).bfloat16()
    messages = []
    module = torch.nn.Linear(1, 1)
    for rank in [encoder_ranks[0], *(r for r in range(8) if r != encoder_ranks[0])]:
        group = H3CodecGroup(SimpleNamespace(rank=rank,
            ranks=[rank // 2 * 2, rank // 2 * 2 + 1]), encoder_ranks)
        def broadcast(value, *, src):
            assert src == encoder_ranks[0]
            if rank == encoder_ranks[0]:
                messages.append(value.clone())
            return messages[0].clone()
        components = H3CausalComponents(module, None, None, None, group,
            SimpleNamespace(ranks=list(range(8)), broadcast=broadcast))
        calls = []
        def encode(*args, **kwargs):
            calls.append(kwargs)
            return expected.clone()
        monkeypatch.setattr(H3CausalComponents, '_encode_local',
            lambda self, *args, **kwargs: encode(*args, **kwargs))
        rgb = np.zeros((frames, 256, 448, 3), dtype=np.uint8)
        actual = (components.encode_image(rgb[0]) if mode == 'image'
                  else components.encode_proxy(rgb, eos=mode == 'eos'))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(calls) == (1 if rank in encoder_ranks else 0)


def test_local_proxy_encode_does_not_publish_until_the_codec_join(monkeypatch):
    expected = torch.randn(24, 5, 2, 2).bfloat16()
    rgb = np.zeros((17, 32, 32, 3), dtype=np.uint8)
    traces, local_values = [], []
    components = []
    for rank in range(8):
        def broadcast(value, *, src, rank=rank):
            assert src == 2
            traces.append(('world', rank))
            return expected.clone()
        group = H3CodecGroup(SimpleNamespace(rank=rank,
            ranks=[rank // 2 * 2, rank // 2 * 2 + 1]), (2, 3))
        component = H3CausalComponents(torch.nn.Linear(1, 1), None, None, None, group,
            SimpleNamespace(ranks=list(range(8)), broadcast=broadcast))
        def encode(*args, rank=rank, **kwargs):
            traces.append(('local', rank))
            return expected.clone()
        monkeypatch.setattr(H3CausalComponents, '_encode_local',
            lambda self, *args, **kwargs: encode(*args, **kwargs))
        components.append(component)
        local_values.append(component.encode_proxy_local(rgb, eos=False))
    assert traces == [('local', 2), ('local', 3)]
    for component, local in zip(components, local_values):
        torch.testing.assert_close(component.broadcast_proxy(local, rgb_shape=rgb.shape,
            eos=False), expected, rtol=0, atol=0)
    assert traces[2:] == [('world', rank) for rank in range(8)]


def vae_fixture(group=None):
    vae = klvae.AutoencoderKL.__new__(klvae.AutoencoderKL)
    torch.nn.Module.__init__(vae)
    vae.encoder = torch.nn.Identity()
    vae.parallel_tiling, vae.stack_tiling = True, False
    vae.tile_size, vae.tile_overlap_min, vae.vae_ratio = 256, 64, 16
    vae._blend_weight_cache = {}
    vae._tile_group = group
    vae.encode = lambda value: value[:, :1, :1, ::16, ::16].clone()
    return vae


def test_tile_assignment_and_gather_share_explicit_group_and_preserve_blending(monkeypatch):
    pixels = torch.arange(256 * 448).reshape(1, 1, 1, 256, 448).float()
    baseline = vae_fixture()
    baseline.parallel_tiling = False
    expected = baseline.tiled_encode(pixels)
    tile_outputs = [baseline.encode(pixels[..., start:start + 256]) for start in (0, 192)]
    def no_global():
        raise AssertionError('explicit encoder consulted the global decode group')
    monkeypatch.setattr(klvae, 'get_tile_parallel_state', no_global)
    monkeypatch.setattr(klvae, 'get_decode_parallel_group_coordinator', no_global)
    for rank in (0, 1):
        calls = []
        def gather(stacked, *, separate_tensors):
            assert separate_tensors
            torch.testing.assert_close(stacked[0], tile_outputs[rank], rtol=0, atol=0)
            calls.append(stacked.shape)
            return [tile[None] for tile in tile_outputs]
        vae = vae_fixture(SimpleNamespace(rank_in_group=rank, world_size=2, all_gather=gather))
        assigned = []
        run = vae._run_tile_tasks
        def tracked(tiles, indices, *args):
            assigned.append(indices)
            return run(tiles, indices, *args)
        vae._run_tile_tasks = tracked
        torch.testing.assert_close(vae.tiled_encode(pixels), expected, rtol=0, atol=0)
        assert assigned == [[rank]] and len(calls) == 1
        assigned.clear()
        vae.tiled_encode(pixels[..., :256])
        assert assigned == [[0]] and len(calls) == 1


def test_four_encoder_ranks_preserve_full_canvas_tile_assembly():
    pixels = torch.arange(768 * 1344).reshape(1, 1, 1, 768, 1344).float()
    baseline = vae_fixture()
    baseline.parallel_tiling = False
    expected = baseline.tiled_encode(pixels)
    ys, _, _ = baseline.split_tiles(768)
    xs, _, _ = baseline.split_tiles(1344)
    tiles = [baseline.encode(pixels[..., y:y + 256, x:x + 256]) for y in ys for x in xs]
    assert len(tiles) == 28
    for rank in range(4):
        def gather(stacked, *, separate_tensors):
            assert separate_tensors
            torch.testing.assert_close(stacked, torch.stack(tiles[rank::4]), rtol=0, atol=0)
            return [torch.stack(tiles[i::4]) for i in range(4)]
        vae = vae_fixture(SimpleNamespace(rank_in_group=rank, world_size=4, all_gather=gather))
        torch.testing.assert_close(vae.tiled_encode(pixels), expected, rtol=0, atol=0)
