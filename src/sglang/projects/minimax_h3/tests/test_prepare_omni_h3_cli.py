"""Recipe semantics and deployment acceleration have independent defaults."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from inference import omni_h3_assets


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.setitem(sys.modules, 'projects.minimax_h3.inference.omni_h3_assets', omni_h3_assets)
    path = Path(__file__).resolve().parents[1] / 'tools/prepare_omni_h3.py'
    spec = importlib.util.spec_from_file_location('prepare_h3_cli_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = ['--artifact', 'artifact.json', '--output', 'native', '--decoder-config', 'decoder.json',
        '--decoder-weights', 'decoder.bin', '--state-reservation-gib', '32']
CWM = BASE + ['--cwm-silence', 'silence.safetensors']


@pytest.mark.parametrize('profile', ['production', 'reference'])
def test_cwm_profiles_keep_model_numerics_and_leave_decoder_untuned(cli, profile):
    args = cli.parse_args(CWM + ([] if profile == 'production' else ['--deployment-profile', profile]))
    assert (args.recipe, args.deployment_profile) == ('cwm', profile)
    if profile == 'production':
        assert args.codec_parallel and args.codec_encoder_ranks == [2, 3, 6, 7]
        assert args.decoder_ranks == [0, 1, 4, 5] and args.decoder_backend == 'fast'
    else:
        assert not args.codec_parallel and args.codec_encoder_ranks is None and args.decoder_ranks is None
        assert args.decoder_backend == 'reference'
    assert not any((args.decoder_compile, args.decoder_cuda_graph, args.decoder_tp_fused_post,
                    args.decoder_compact_suffix_kv))
    assert (args.decoder_compile_mode, args.decoder_suffix_sdpa_backend,
            args.decoder_spatial_attention_backend) == ('default', 'default', 'fa3' if profile == 'production' else 'fa2')
    assert args.fuse_first_denoise and args.dit_norm_quality == 'lossless'
    assert (args.cwm_proxy_width, args.cwm_proxy_resize) == (1344, 'nearest')
    execution = json.loads(Path(args.execution_config).read_text())
    assert (execution['tp_size'], execution['ulysses_degree'], execution['num_gpus']) == (1, 8, 8)
    assert execution['minimax_h3_adaln_online'] is True


def test_overlap_enables_only_deployment_acceleration(cli):
    args = cli.parse_args(CWM + ['--deployment-profile', 'overlap'])
    assert args.codec_encoder_ranks == [2, 3, 6, 7] and args.decoder_ranks == [0, 1, 4, 5]
    assert args.codec_parallel and args.decoder_backend == 'fast'
    assert all((args.decoder_compile, args.decoder_cuda_graph, args.decoder_tp_fused_post,
                args.decoder_compact_suffix_kv))
    assert args.decoder_compile_mode == 'max-autotune-no-cudagraphs'
    assert (args.decoder_suffix_sdpa_backend, args.decoder_spatial_attention_backend) == ('fa3', 'fa3')
    assert args.dit_norm_quality == 'lossless'
    assert args.fuse_first_denoise and (args.cwm_proxy_width, args.cwm_proxy_resize) == (1344, 'nearest')
    assert json.loads(Path(args.execution_config).read_text())['minimax_h3_adaln_online'] is True


@pytest.mark.parametrize('flags, expected', [
    (['--no-codec-parallel'], dict(codec_parallel=False, codec_encoder_ranks=None, decoder_ranks=None,
        decoder_backend='reference', decoder_compile=False, decoder_cuda_graph=False,
        decoder_tp_fused_post=False, decoder_compile_mode='default', decoder_compact_suffix_kv=False,
        decoder_suffix_sdpa_backend='default', decoder_spatial_attention_backend='fa2')),
    (['--no-decoder-compile'], dict(decoder_compile=False, decoder_compile_mode='default', decoder_tp_fused_post=False)),
    (['--no-decoder-compact-suffix-kv'], dict(decoder_compact_suffix_kv=False, decoder_suffix_sdpa_backend='default')),
    (['--no-decoder-cuda-graph', '--no-decoder-tp-fused-post'], dict(decoder_cuda_graph=False, decoder_tp_fused_post=False)),
])
def test_negative_flags_clear_only_implicit_dependents(cli, flags, expected):
    args = cli.parse_args(CWM + ['--deployment-profile', 'overlap', *flags])
    assert {name: getattr(args, name) for name in expected} == expected


@pytest.mark.parametrize('flags', [
    ['--no-codec-parallel', '--codec-encoder-ranks', '2', '3'],
    ['--no-codec-parallel', '--decoder-ranks', '0', '1'],
    ['--no-codec-parallel', '--decoder-tp-fused-post'],
    ['--no-decoder-compile', '--decoder-compile-mode', 'max-autotune-no-cudagraphs'],
    ['--no-decoder-compile', '--decoder-tp-fused-post'],
    ['--no-decoder-compact-suffix-kv', '--decoder-suffix-sdpa-backend', 'fa3'],
    ['--decoder-ranks', '0', '1'],
    ['--codec-encoder-ranks', '2', '3', '6'],
])
def test_explicit_conflicts_are_not_silently_overridden(cli, flags):
    with pytest.raises(SystemExit) as error:
        cli.parse_args(CWM + ['--deployment-profile', 'overlap', *flags])
    assert error.value.code == 2


def test_legacy_recipe_and_explicit_e2_topology(cli):
    args = cli.parse_args(BASE + ['--recipe', 'longlive', '--deployment-profile', 'reference', '--codec-parallel',
        '--codec-encoder-ranks', '2', '3', '--decoder-ranks', '0', '1'])
    assert args.codec_encoder_ranks == [2, 3] and args.decoder_ranks == [0, 1]
    assert args.decoder_backend == 'fast' and not args.decoder_compile
    assert Path(args.execution_config).name == 'omni-h3-execution.json'
    assert json.loads(Path(args.execution_config).read_text())['tp_size'] == 2


def test_cwm_default_requires_frozen_silence(cli):
    with pytest.raises(SystemExit) as error:
        cli.parse_args(BASE)
    assert error.value.code == 2


def test_tae_proxy_flags_are_explicit_and_preserve_the_cwm_profile(cli):
    args = cli.parse_args(CWM + ['--proxy-encoder', 'tae', '--tae-weights', 'tae.pth'])
    assert args.proxy_encoder == 'tae' and args.tae_weights == 'tae.pth'
    assert args.codec_parallel and args.codec_encoder_ranks == [2, 3, 6, 7]


@pytest.mark.parametrize('flags', [['--proxy-encoder', 'tae'], ['--tae-weights', 'tae.pth']])
def test_tae_proxy_flags_require_a_matching_weight_argument(cli, flags):
    with pytest.raises(SystemExit) as error:
        cli.parse_args(CWM + flags)
    assert error.value.code == 2


def test_main_forwards_effective_profile_to_bundle_preparation(cli, monkeypatch):
    from unittest.mock import Mock
    prepare = Mock()
    monkeypatch.setattr(cli, 'prepare_native_bundle', prepare)
    monkeypatch.setattr(sys, 'argv', ['prepare', *CWM, '--deployment-profile', 'overlap'])
    cli.main()
    options = prepare.call_args.kwargs
    assert options['recipe'] == 'cwm' and options['cwm_silence'] == 'silence.safetensors'
    assert options['codec_parallel'] == dict(encoder_ranks=[2, 3, 6, 7], decoder_ranks=[0, 1, 4, 5])
    assert options['decoder_compile'] and options['decoder_tp_fused_post']
    assert options['fuse_first_denoise'] and options['cwm_proxy_resize'] == 'nearest'
    assert options['state_reservation_bytes'] == 32 << 30


def test_proxy_resize_is_a_cwm_full_size_option(cli):
    assert cli.parse_args(CWM + ['--cwm-proxy-width', '448']).cwm_proxy_resize == 'lanczos'
    assert cli.parse_args(CWM + ['--cwm-proxy-resize', 'lanczos']).cwm_proxy_resize == 'lanczos'
    with pytest.raises(SystemExit) as error:
        cli.parse_args(BASE + ['--recipe', 'longlive', '--deployment-profile', 'reference', '--cwm-proxy-resize', 'nearest'])
    assert error.value.code == 2


def test_diffusers_execution_preset_only_disables_online_adaln():
    runs = Path(__file__).resolve().parents[1] / 'configs/runs'
    native = json.loads((runs / 'omni-h3-cwm-execution.json').read_text())
    diffusers = json.loads((runs / 'omni-h3-cwm-diffusers-execution.json').read_text())
    assert native['minimax_h3_adaln_online'] is True
    assert diffusers == dict(native, minimax_h3_adaln_online=False)
