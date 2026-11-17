import json
from pathlib import Path
import subprocess
import sys

import pytest

from inference.omni_h3_assets import (derive_native_recipe, digest, native_bundle_artifact,
    prepare_native_bundle, read_native_bundle, verify_native_bundle)


@pytest.fixture
def assets(tmp_path):
    checkpoint = tmp_path / 'encoders'
    checkpoint.mkdir()
    (checkpoint / 'video_vae/source').mkdir(parents=True)
    (checkpoint / 'video_vae/config.json').write_text('{}')
    (checkpoint / 'video_vae/source/config.json').write_text('{}')
    (checkpoint / 'source.bin').write_bytes(b'original encoder')
    weights, decoder_config, decoder_weights = (tmp_path / p for p in ('merged.bin', 'decoder.json', 'decoder.bin'))
    weights.write_bytes(b'original merged tensor payload')
    decoder_config.write_text('{}')
    decoder_weights.write_bytes(b'causal decoder')
    root = Path(__file__).resolve().parents[1]
    bundle = dict(recipe=json.loads((root / 'configs/recipes/omni-h3-longlive.json').read_text()),
        model={'args': {'hidden_size': 5376}},
        weights=dict(format='raven_full_or_merged', path=str(weights), sha256=digest(weights)),
        encoders=dict(checkpoint=str(checkpoint), recipe=dict(
            conditioning_mode='static_image_qwen_proxy_vae_v1', posterior='mean',
            model_hashes={'source.bin': digest(checkpoint / 'source.bin')},
            height=768, width=1344, proxy_height=256, proxy_width=448)))
    artifact = tmp_path / 'artifact.json'
    artifact.write_text(json.dumps(bundle))
    execution = tmp_path / 'execution.json'
    execution.write_text(json.dumps(dict(num_gpus=8, tp_size=2, sp_degree=4,
        ulysses_degree=4, ring_degree=1, encoder_parallel='auto', attention_backend='fa')))
    return artifact, bundle, dict(decoder_config=decoder_config, decoder_weights=decoder_weights,
        state_reservation_bytes=1 << 30, execution_config=execution)


def test_prepare_keeps_tensor_payload_and_declares_native_qkv(tmp_path, assets):
    artifact, bundle, options = assets
    output = tmp_path / 'native'
    prepare_native_bundle(artifact, output, **options)
    manifest = read_native_bundle(output)
    assert 'codec_parallel' not in manifest
    assert manifest['dit'] == {'fuse_first_denoise': False, 'norm_quality': 'exact'}
    assert manifest['codec']['backend'] == 'reference'
    assert manifest['codec']['suffix_sdpa_backend'] == 'default'
    assert manifest['server_args']['transformer_weights_path'] == bundle['weights']['path']
    assert digest(bundle['weights']['path']) == bundle['weights']['sha256']
    assert json.loads((output / 'transformer/config.json').read_text())['checkpoint_qkv_layout'] == 'native'
    assert not list(output.rglob('*.bin'))
    with pytest.raises(FileExistsError):
        prepare_native_bundle(artifact, output, **options)


def test_tae_proxy_bundle_pins_weights_and_keeps_h3_assets(tmp_path, assets):
    artifact, _, options = assets
    tae_weights = tmp_path / 'tae.pth'
    tae_weights.write_bytes(b'CPU fixture TAE')
    baseline = prepare_native_bundle(artifact, tmp_path / 'h3', **options,
        codec_parallel=dict(encoder_ranks=[2, 3], decoder_ranks=[0, 1]), decoder_backend='fast')
    candidate = prepare_native_bundle(artifact, tmp_path / 'tae', **options,
        codec_parallel=dict(encoder_ranks=[2, 3], decoder_ranks=[0, 1]), decoder_backend='fast',
        proxy_encoder='tae', tae_weights=tae_weights)
    assert read_native_bundle(tmp_path / 'tae') == candidate
    assert candidate['codec'] == dict(baseline['codec'], encoder='tae',
        tae_weights=str(tae_weights.resolve()))
    assert candidate['assets'][str(tae_weights.resolve())] == digest(tae_weights)
    assert candidate['server_args']['component_paths']['video_vae'] == baseline['server_args']['component_paths']['video_vae']
    del candidate['assets'][str(tae_weights.resolve())]
    (tmp_path / 'tae/native-runtime.json').write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match='TAE proxy weights must be pinned'):
        read_native_bundle(tmp_path / 'tae')


@pytest.mark.parametrize('updates', [dict(proxy_encoder='tae'),
    dict(tae_weights='unused.pth'), dict(proxy_encoder='unknown')])
def test_invalid_proxy_encoder_selection_fails_before_creating_bundle(tmp_path, assets, updates):
    artifact, _, options = assets
    with pytest.raises(ValueError):
        prepare_native_bundle(artifact, tmp_path / 'native', **options, **updates)
    assert not (tmp_path / 'native').exists()


def _safetensors(path, metadata):
    header = json.dumps({'__metadata__': metadata}).encode()
    path.write_bytes(len(header).to_bytes(8, 'little') + header)


@pytest.mark.parametrize('kind, expected', [
    ('per_head', ('MiniMaxH3CausalDiTModel', 'grouped')),
    ('runtime', ('MiniMaxH3CausalDiTModel', 'native')),
    ('diffusers', ('MiniMaxH3Transformer3DModel', 'native')),
])
def test_transformer_config_follows_the_stored_checkpoint_layout(tmp_path, assets, kind, expected):
    artifact, bundle, options = assets
    if kind == 'diffusers':
        weights = tmp_path / 'hf/diffusion_pytorch_model.safetensors.index.json'
        weights.parent.mkdir()
        weights.write_text('{"weight_map": {}}')
        (weights.parent / 'config.json').write_text('{"_class_name": "MiniMaxH3Transformer3DModel"}')
    else:
        weights = tmp_path / f'{kind}.safetensors'
        _safetensors(weights, {'qkv_layout': 'released_per_head'} if kind == 'per_head' else {})
    bundle['weights'].update(path=str(weights), sha256=digest(weights))
    artifact.write_text(json.dumps(bundle))
    prepare_native_bundle(artifact, tmp_path / 'native', **options)
    config = json.loads((tmp_path / 'native/transformer/config.json').read_text())
    assert (config['_class_name'], config['checkpoint_qkv_layout']) == expected


def test_execution_topology_does_not_rewrite_the_frozen_recipe(tmp_path, assets):
    artifact, bundle, options = assets
    first = prepare_native_bundle(artifact, tmp_path / 'tp2', **options)
    execution = json.loads(options['execution_config'].read_text())
    execution.update(tp_size=4, sp_degree=2, ulysses_degree=2)
    options['execution_config'].write_text(json.dumps(execution))
    second = prepare_native_bundle(artifact, tmp_path / 'tp4', **options)
    assert first['recipe'] == second['recipe'] == bundle['recipe']
    assert first['server_args']['tp_size'] == 2
    assert second['server_args']['tp_size'] == 4
    assert second['server_args']['encoder_parallel'] == 'auto'


def test_execution_config_cannot_override_model_assets(tmp_path, assets):
    artifact, bundle, options = assets
    options['execution_config'].write_text(json.dumps(dict(num_gpus=8, model_path='another-model')))
    with pytest.raises(ValueError, match='only native topology'):
        prepare_native_bundle(artifact, tmp_path / 'native', **options)
    assert not (tmp_path / 'native').exists()


def test_changed_weights_fail_before_output_is_created(tmp_path, assets):
    artifact, bundle, options = assets
    Path(bundle['weights']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='digest differs'):
        prepare_native_bundle(artifact, tmp_path / 'native', **options)
    assert not (tmp_path / 'native').exists()


def test_changed_asset_is_rejected_again_at_runtime_start(tmp_path, assets):
    artifact, bundle, options = assets
    output = tmp_path / 'native'
    prepare_native_bundle(artifact, output, **options)
    options['decoder_weights'].write_bytes(b'changed after preparation')
    with pytest.raises(ValueError, match='prepared native asset changed'):
        verify_native_bundle(read_native_bundle(output))


def test_native_vae_config_is_pinned_with_weights(tmp_path, assets):
    artifact, bundle, options = assets
    output = tmp_path / 'native'
    prepare_native_bundle(artifact, output, **options)
    config = Path(bundle['encoders']['checkpoint']) / 'video_vae/config.json'
    config.write_text('{"latents_mean": [100]}')
    with pytest.raises(ValueError, match='prepared native asset changed'):
        verify_native_bundle(read_native_bundle(output))


def test_incompatible_recipe_is_not_silently_reinterpreted(tmp_path, assets):
    artifact, bundle, options = assets
    bundle['recipe']['recent_latents'] = 30
    artifact.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match='frozen step600'):
        prepare_native_bundle(artifact, tmp_path / 'native', **options)


@pytest.mark.parametrize('name', ['longlive', 'continuous5'])
def test_named_recipe_preserves_source_and_changes_only_segment_boundaries(tmp_path, assets, name):
    artifact, bundle, options = assets
    original = artifact.read_bytes()
    manifest = prepare_native_bundle(artifact, tmp_path / 'native', recipe=name, **options)
    expected = json.loads(original)['recipe']
    if name == 'continuous5':
        expected['segment_boundaries'] = []
    assert manifest['recipe'] == expected
    assert artifact.read_bytes() == original
    assert bundle['recipe']['segment_boundaries'] == [45, 70]
    assert native_bundle_artifact(manifest, artifact) == (digest(artifact), bundle, expected)
    verify_native_bundle(read_native_bundle(tmp_path / 'native'))


def test_canonical_recipe_matches_historical_frozen_recipe(assets):
    _, bundle, _ = assets
    assert derive_native_recipe(bundle['recipe']) == bundle['recipe']


@pytest.mark.parametrize('wire_size', [(256, 448), (384, 672), (768, 1344)])
@pytest.mark.parametrize('proxy_width', [448, 1344])
def test_cwm_keeps_wire_geometry_and_pins_encoder_size_and_silence(tmp_path, assets, proxy_width, wire_size):
    artifact, bundle, options = assets
    root = Path(__file__).resolve().parents[1]
    bundle['recipe'] = json.loads((root/'configs/recipes/omni-h3-cwm.json').read_text())
    bundle['encoders']['recipe'].update(proxy_height=wire_size[0], proxy_width=wire_size[1])
    artifact.write_text(json.dumps(bundle))
    execution = json.loads(options['execution_config'].read_text())
    execution.update(tp_size=1,sp_degree=8,ulysses_degree=8)
    options['execution_config'].write_text(json.dumps(execution))
    silence = tmp_path/'silence.safetensors'; silence.write_bytes(b'frozen stereo silence')
    output = tmp_path/'cwm'
    manifest = prepare_native_bundle(artifact,output,recipe='cwm',cwm_silence=silence,
        cwm_proxy_width=proxy_width,**options)
    assert manifest['model_name'] == 'invoker-render'
    assert manifest['canvas']['proxy_width']==wire_size[1] and manifest['cwm']['proxy_encode_width']==proxy_width
    assert manifest['cwm']['proxy_encode_height']==proxy_width*4//7
    assert manifest['cwm']['proxy_resize']==('none' if proxy_width==448 else 'lanczos')
    assert manifest['recipe']['bootstrap_latents']==9 and manifest['recipe']['position_policy']=='cwm_frame_aligned_v1'
    assert (manifest['recipe']['sink_latents'], manifest['recipe']['recent_latents']) == (1, 12)
    assert manifest['recipe']['segment_boundaries'] == [45, 70]
    assert manifest['recipe']['diffusion']['sampler']['class_name'] == 'EulerSampler'
    assert manifest['cwm']['posterior'] == 'mean'
    assert native_bundle_artifact(read_native_bundle(output),artifact)[1]==bundle
    invalid=json.loads((output/'native-runtime.json').read_text())
    invalid['cwm']['proxy_encode_height']=384
    (output/'native-runtime.json').write_text(json.dumps(invalid))
    with pytest.raises(ValueError,match='encoder contract'):read_native_bundle(output)
    (output/'native-runtime.json').write_text(json.dumps(manifest))
    silence.write_bytes(b'changed')
    with pytest.raises(ValueError,match='asset changed'):
        verify_native_bundle(read_native_bundle(output))


def test_cwm_nearest_resize_is_pinned_and_limited_to_the_full_size_encoder(tmp_path, assets):
    artifact, bundle, options = assets
    root = Path(__file__).resolve().parents[1]
    bundle['recipe'] = json.loads((root / 'configs/recipes/omni-h3-cwm.json').read_text())
    artifact.write_text(json.dumps(bundle))
    execution = json.loads(options['execution_config'].read_text())
    execution.update(tp_size=1, sp_degree=8, ulysses_degree=8)
    options['execution_config'].write_text(json.dumps(execution))
    silence = tmp_path / 'silence.safetensors'
    silence.write_bytes(b'frozen stereo silence')
    with pytest.raises(ValueError, match='nearest resize requires'):
        prepare_native_bundle(artifact, tmp_path / 'small', recipe='cwm', cwm_silence=silence,
            cwm_proxy_width=448, cwm_proxy_resize='nearest', **options)
    output = tmp_path / 'nearest'
    prepare_native_bundle(artifact, output, recipe='cwm', cwm_silence=silence, cwm_proxy_resize='nearest', **options)
    assert read_native_bundle(output)['cwm']['proxy_resize'] == 'nearest'
    invalid = json.loads((output / 'native-runtime.json').read_text())
    invalid['cwm'].update(proxy_encode_width=448, proxy_encode_height=256)
    (output / 'native-runtime.json').write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match='encoder contract'):
        read_native_bundle(output)


def test_cwm_proxy_width_rejects_unsupported_size_before_preparation(tmp_path, assets):
    artifact, _, options=assets
    with pytest.raises(ValueError,match='448 or 1344'):
        prepare_native_bundle(artifact,tmp_path/'invalid',cwm_proxy_width=672,**options)
    assert not (tmp_path/'invalid').exists()


def test_untagged_legacy_bundle_keeps_longlive_identity(tmp_path, assets):
    artifact, bundle, options = assets
    output = tmp_path / 'native'
    manifest = prepare_native_bundle(artifact, output, **options)
    del manifest['recipe_selection']
    (output / 'native-runtime.json').write_text(json.dumps(manifest))
    verify_native_bundle(read_native_bundle(output))
    assert native_bundle_artifact(manifest, artifact)[2] == bundle['recipe']


@pytest.mark.parametrize('change', ['source', 'effective', 'selection', 'unpinned'])
def test_named_recipe_rejects_tampered_identity(tmp_path, assets, change):
    artifact, bundle, options = assets
    manifest = prepare_native_bundle(artifact, tmp_path / 'native', recipe='continuous5', **options)
    if change == 'source':
        bundle['recipe']['segment_boundaries'] = []
        artifact.write_text(json.dumps(bundle))
    elif change == 'effective':
        manifest['recipe']['bootstrap_latents'] = 5
    elif change == 'selection':
        manifest['recipe_selection']['name'] = 'longlive'
    else:
        manifest['assets'].pop(str(artifact))
    (tmp_path / 'native/native-runtime.json').write_text(json.dumps(manifest))
    if change == 'source':
        # Reading the bundle does not hash weights or the source artifact; naming the artifact does.
        with pytest.raises(ValueError, match='not pinned'):
            native_bundle_artifact(read_native_bundle(tmp_path / 'native'), artifact)
        with pytest.raises(ValueError, match='prepared native asset changed'):
            verify_native_bundle(read_native_bundle(tmp_path / 'native'))
        return
    with pytest.raises(ValueError):
        read_native_bundle(tmp_path / 'native')


@pytest.mark.parametrize('encoder_ranks', [[0, 1], [2, 3], [4, 5], [6, 7]])
def test_codec_parallel_is_separate_from_dit_topology(tmp_path, assets, encoder_ranks):
    artifact, bundle, options = assets
    spec = dict(encoder_ranks=encoder_ranks, decoder_ranks=[0, 1])
    manifest = prepare_native_bundle(artifact, tmp_path / 'native', **options,
        codec_parallel=spec, decoder_backend='fast')
    assert manifest['codec_parallel'] == spec
    assert 'codec_parallel' not in manifest['server_args']
    assert manifest['server_args']['tp_size'] == 2
    assert manifest['recipe'] == bundle['recipe']
    assert manifest['codec']['compile'] is False
    assert read_native_bundle(tmp_path / 'native') == manifest


def test_codec_parallel_accepts_pair_aligned_tp4_decoder(tmp_path, assets):
    artifact, _, options = assets
    spec = dict(encoder_ranks=[2, 3], decoder_ranks=[0, 1, 4, 5])
    manifest = prepare_native_bundle(artifact, tmp_path / 'native', **options,
        codec_parallel=spec, decoder_backend='fast')
    assert manifest['codec_parallel'] == spec
    assert read_native_bundle(tmp_path / 'native') == manifest


@pytest.mark.parametrize('spec', [None, {}, {'encoder_ranks': [0, 1]},
    dict(encoder_ranks=[0, 1], decoder_ranks=[0, 1], overlap=True),
    dict(encoder_ranks=None, decoder_ranks=[0, 1]),
    dict(encoder_ranks=[], decoder_ranks=[0, 1]),
    dict(encoder_ranks=[0, 0], decoder_ranks=[0, 1]),
    dict(encoder_ranks=[0, 2], decoder_ranks=[0, 1]),
    dict(encoder_ranks=[1, 2], decoder_ranks=[0, 1]),
    dict(encoder_ranks=[3, 2], decoder_ranks=[0, 1]),
    dict(encoder_ranks=[6, 8], decoder_ranks=[0, 1]),
    dict(encoder_ranks=[False, True], decoder_ranks=[0, 1]),
    dict(encoder_ranks=[2.0, 3.0], decoder_ranks=[0, 1]),
    dict(encoder_ranks=[2, 3], decoder_ranks=[2, 3]),
    dict(encoder_ranks=[2, 3], decoder_ranks=[False, True])])
def test_runtime_rejects_invalid_codec_topology(tmp_path, assets, spec):
    artifact, _, options = assets
    output = tmp_path / 'native'
    manifest = prepare_native_bundle(artifact, output, **options)
    manifest['codec_parallel'] = spec
    (output / 'native-runtime.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='codec_parallel'):
        read_native_bundle(output)


def test_codec_parallel_rejects_reference_backend_before_creating_bundle(tmp_path, assets):
    artifact, _, options = assets
    with pytest.raises(ValueError, match='fast causal'):
        prepare_native_bundle(artifact, tmp_path / 'native', **options,
            codec_parallel=dict(encoder_ranks=[0, 1], decoder_ranks=[0, 1]))
    assert not (tmp_path / 'native').exists()


def test_first_denoise_fusion_is_execution_only_and_old_bundles_default_off(tmp_path, assets):
    artifact, bundle, options = assets
    output = tmp_path / 'native'
    manifest = prepare_native_bundle(artifact, output, **options, fuse_first_denoise=True)
    assert manifest['dit'] == {'fuse_first_denoise': True, 'norm_quality': 'exact'}
    assert manifest['recipe'] == bundle['recipe']
    assert 'fuse_first_denoise' not in manifest['server_args']
    assert read_native_bundle(output) == manifest
    del manifest['dit']
    (output / 'native-runtime.json').write_text(json.dumps(manifest))
    assert read_native_bundle(output) == manifest


@pytest.mark.parametrize('spec', [None, {'fuse_first_denoise': 'true'}, {'fuse_first_denoise': 1}, {'unknown': True}])
def test_native_bundle_rejects_invalid_dit_options(tmp_path, assets, spec):
    artifact, _, options = assets
    output = tmp_path / 'native'
    manifest = prepare_native_bundle(artifact, output, **options)
    manifest['dit'] = spec
    (output / 'native-runtime.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='dit accepts only'):
        read_native_bundle(output)


@pytest.mark.parametrize('quality', ['exact', 'lossless'])
def test_norm_quality_is_execution_only_and_old_bundles_keep_exact(tmp_path, assets, quality):
    artifact, bundle, options = assets
    output = tmp_path / 'native'
    manifest = prepare_native_bundle(artifact, output, **options, dit_norm_quality=quality)
    assert manifest['dit']['norm_quality'] == quality
    assert manifest['recipe'] == bundle['recipe']
    assert 'norm_quality' not in manifest['server_args']
    assert read_native_bundle(output) == manifest
    verify_native_bundle(manifest)
    del manifest['dit']['norm_quality']
    (output / 'native-runtime.json').write_text(json.dumps(manifest))
    assert read_native_bundle(output) == manifest
    verify_native_bundle(manifest)
    del manifest['dit']
    (output / 'native-runtime.json').write_text(json.dumps(manifest))
    assert read_native_bundle(output) == manifest
    verify_native_bundle(manifest)


@pytest.mark.parametrize('quality', [None, True, False, 0, 1, 1.0, [], {}, 'high', '', 'EXACT'])
def test_invalid_norm_quality_is_rejected_at_every_manifest_boundary(tmp_path, assets, quality):
    artifact, _, options = assets
    output = tmp_path / 'native'
    with pytest.raises(ValueError, match='dit.norm_quality'):
        prepare_native_bundle(artifact, output, **options, dit_norm_quality=quality)
    assert not output.exists()
    manifest = prepare_native_bundle(artifact, output, **options)
    manifest['dit']['norm_quality'] = quality
    (output / 'native-runtime.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='dit.norm_quality'):
        read_native_bundle(output)


@pytest.mark.parametrize('quality', [None, 'exact', 'lossless'])
@pytest.mark.parametrize('codec_args, encoder_ranks', [([], None),
    (['--codec-parallel'], [0, 1]),
    (['--codec-parallel', '--codec-encoder-ranks', '2', '3'], [2, 3]),
    (['--codec-parallel', '--decoder-backend', 'fast', '--decoder-compile',
      '--decoder-compile-mode', 'max-autotune-no-cudagraphs', '--decoder-tp-fused-post'], [0, 1]),
    (['--codec-encoder-ranks', '2', '3'], 'missing-parallel')])
def test_cli_bundle_options_round_trip_without_heavy_runtime_imports(
        tmp_path, assets, quality, codec_args, encoder_ranks):
    artifact, _, options = assets
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'native'
    script = """
import runpy, sys
from inference import omni_h3_assets
sys.modules['projects.minimax_h3.inference.omni_h3_assets'] = omni_h3_assets
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
assert not {'torch', 'sglang', 'sglang_omni'} & sys.modules.keys()
"""
    command = [sys.executable, '-c', script, str(root / 'tools/prepare_omni_h3.py'),
        '--recipe', 'longlive', '--deployment-profile', 'reference',
        '--artifact', str(artifact), '--output', str(output),
        '--decoder-config', str(options['decoder_config']),
        '--decoder-weights', str(options['decoder_weights']),
        '--state-reservation-gib', '1', '--execution-config', str(options['execution_config'])]
    if quality is not None:
        command.extend(['--dit-norm-quality', quality])
    command.extend(codec_args)
    if encoder_ranks == 'missing-parallel':
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        assert result.returncode == 2
        assert 'explicit codec ranks require --codec-parallel' in result.stderr
        assert not output.exists()
        return
    subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    manifest = read_native_bundle(output)
    assert manifest['dit']['norm_quality'] == (quality or 'exact')
    assert manifest.get('codec_parallel') == (None if encoder_ranks is None else
        dict(encoder_ranks=encoder_ranks, decoder_ranks=[0, 1]))
    assert manifest['codec']['tp_fused_post'] == ('--decoder-tp-fused-post' in codec_args)
    assert manifest['codec']['compile_mode'] == (
        'max-autotune-no-cudagraphs' if '--decoder-compile-mode' in codec_args else 'default')


def test_suffix_sdpa_policy_is_explicit_and_old_bundles_keep_default(tmp_path, assets):
    artifact, bundle, options = assets
    output = tmp_path / 'native'
    manifest = prepare_native_bundle(artifact, output, **options,
        decoder_backend='fast', decoder_suffix_sdpa_backend='cudnn')
    assert manifest['codec']['suffix_sdpa_backend'] == 'cudnn'
    assert manifest['recipe'] == bundle['recipe']
    assert read_native_bundle(output) == manifest
    del manifest['codec']['suffix_sdpa_backend']
    (output / 'native-runtime.json').write_text(json.dumps(manifest))
    assert read_native_bundle(output) == manifest
    verify_native_bundle(manifest)


def test_decoder_attention_options_round_trip_and_old_bundles_default(tmp_path, assets):
    artifact, _, options = assets
    output = tmp_path / 'native'
    manifest = prepare_native_bundle(artifact, output, **options,
        decoder_backend='fast', decoder_spatial_attention_backend='fa3',
        decoder_compact_suffix_kv=True)
    assert manifest['codec']['spatial_attention_backend'] == 'fa3'
    assert manifest['codec']['compact_suffix_kv'] is True
    assert read_native_bundle(output) == manifest
    del manifest['codec']['spatial_attention_backend']
    del manifest['codec']['compact_suffix_kv']
    (output / 'native-runtime.json').write_text(json.dumps(manifest))
    assert read_native_bundle(output) == manifest


@pytest.mark.parametrize('kwargs, message', [
    ({'decoder_spatial_attention_backend': 'cutlass'}, 'spatial_attention_backend'),
    ({'decoder_spatial_attention_backend': None}, 'spatial_attention_backend'),
    ({'decoder_compact_suffix_kv': 1}, 'compact_suffix_kv'),
])
def test_invalid_decoder_attention_options_fail_before_bundle_write(tmp_path, assets, kwargs, message):
    artifact, _, options = assets
    with pytest.raises(ValueError, match=message):
        prepare_native_bundle(artifact, tmp_path / 'native', **options,
            decoder_backend='fast', **kwargs)
    assert not (tmp_path / 'native').exists()


@pytest.mark.parametrize('backend, policy', [('reference', 'cudnn'), ('fast', 'unknown'),
    ('fast', None), ('fast', True)])
def test_invalid_suffix_sdpa_policy_fails_before_creating_bundle(tmp_path, assets, backend, policy):
    artifact, _, options = assets
    with pytest.raises(ValueError, match='suffix'):
        prepare_native_bundle(artifact, tmp_path / 'native', **options,
            decoder_backend=backend, decoder_suffix_sdpa_backend=policy)
    assert not (tmp_path / 'native').exists()


@pytest.mark.parametrize('options_update', [dict(suffix_sdpa_backend='unknown'),
    dict(suffix_sdpa_backend=None), dict(suffix_sdpa_backend=True),
    dict(suffix_sdpa_backend='cudnn'),
    dict(decoder='h3', backend='fast', suffix_sdpa_backend='cudnn')])
def test_runtime_rejects_invalid_suffix_sdpa_policy(tmp_path, assets, options_update):
    artifact, _, options = assets
    output = tmp_path / 'native'
    manifest = prepare_native_bundle(artifact, output, **options)
    manifest['codec'].update(options_update)
    (output / 'native-runtime.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='suffix'):
        read_native_bundle(output)


def test_fa3_suffix_is_explicit_and_requires_compact_fast_decoder():
    from inference.omni_h3_assets import _validate_codec_options
    codec=dict(decoder='causal',backend='fast',suffix_sdpa_backend='fa3',compact_suffix_kv=True)
    _validate_codec_options(codec)
    for update in (dict(compact_suffix_kv=False),dict(backend='reference'),dict(decoder='h3')):
        with pytest.raises(ValueError,match='FA3 suffix'):
            _validate_codec_options(dict(codec,**update))


def test_fa3_suffix_option_survives_bundle_round_trip(tmp_path, assets):
    artifact, _, options = assets
    output=tmp_path/'fa3-bundle'
    manifest=prepare_native_bundle(artifact,output,**options,decoder_backend='fast',
        decoder_suffix_sdpa_backend='fa3',decoder_compact_suffix_kv=True)
    assert read_native_bundle(output)==manifest
    assert manifest['codec']['suffix_sdpa_backend']=='fa3'


def test_decoder_compile_options_round_trip_and_old_bundle_defaults(tmp_path, assets):
    artifact, _, options = assets
    output = tmp_path / 'compiled'
    manifest = prepare_native_bundle(artifact, output, **options, decoder_backend='fast',
        decoder_compile=True, decoder_compile_mode='max-autotune-no-cudagraphs',
        decoder_tp_fused_post=True,
        codec_parallel=dict(encoder_ranks=[2, 3], decoder_ranks=[0, 1, 4, 5]))
    assert read_native_bundle(output) == manifest
    assert manifest['codec']['compile_mode'] == 'max-autotune-no-cudagraphs'
    assert manifest['codec']['tp_fused_post'] is True
    del manifest['codec']['compile_mode'], manifest['codec']['tp_fused_post']
    (output / 'native-runtime.json').write_text(json.dumps(manifest))
    assert read_native_bundle(output) == manifest


@pytest.mark.parametrize('codec', [
    dict(compile_mode='max-autotune'),
    dict(tp_fused_post=1),
    dict(compile_mode='max-autotune-no-cudagraphs', compile=False),
    dict(tp_fused_post=True, backend='reference'),
])
def test_invalid_decoder_compile_options_fail_at_bundle_boundary(codec):
    from inference.omni_h3_assets import _validate_codec_options
    with pytest.raises(ValueError, match='compile|tp_fused_post'):
        _validate_codec_options(dict(dict(decoder='causal', backend='fast', compile=True), **codec))


def test_online_adaln_execution_option_requires_boolean(tmp_path, assets):
    artifact, _, options = assets
    execution = json.loads(options['execution_config'].read_text())
    execution['minimax_h3_adaln_online'] = 'false'
    options['execution_config'].write_text(json.dumps(execution))
    with pytest.raises(ValueError, match='must be boolean'):
        prepare_native_bundle(artifact, tmp_path / 'native', **options)
    assert not (tmp_path / 'native').exists()


def test_hf_checkpoint_rejects_online_adaln_before_writing_bundle(tmp_path, assets):
    artifact, bundle, options = assets
    weights = tmp_path / 'hf/diffusion_pytorch_model.safetensors.index.json'
    weights.parent.mkdir()
    weights.write_text('{"weight_map": {}}')
    (weights.parent / 'config.json').write_text('{"_class_name": "MiniMaxH3Transformer3DModel"}')
    bundle['weights'].update(path=str(weights), sha256=digest(weights))
    artifact.write_text(json.dumps(bundle))
    execution = json.loads(options['execution_config'].read_text())
    execution['minimax_h3_adaln_online'] = True
    options['execution_config'].write_text(json.dumps(execution))
    with pytest.raises(ValueError, match='Diffusers checkpoints require'):
        prepare_native_bundle(artifact, tmp_path / 'native', **options)
    assert not (tmp_path / 'native').exists()


def test_cwm_four_encoder_ranks_preserve_original_conditioning(tmp_path, assets):
    artifact, bundle, options = assets
    root = Path(__file__).resolve().parents[1]
    bundle['recipe'] = json.loads((root / 'configs/recipes/omni-h3-cwm.json').read_text())
    bundle['encoders']['recipe'].update(proxy_width=672, proxy_height=384)
    artifact.write_text(json.dumps(bundle))
    execution = json.loads(options['execution_config'].read_text())
    execution.update(tp_size=1, sp_degree=8, ulysses_degree=8, minimax_h3_adaln_online=False)
    options['execution_config'].write_text(json.dumps(execution))
    silence = tmp_path / 'silence.safetensors'
    silence.write_bytes(b'frozen stereo silence')
    spec = dict(encoder_ranks=[2, 3, 6, 7], decoder_ranks=[0, 1, 4, 5])
    output = tmp_path / 'e4'
    manifest = prepare_native_bundle(artifact, output, recipe='cwm', cwm_silence=silence,
        codec_parallel=spec, decoder_backend='fast', **options)
    assert read_native_bundle(output)['codec_parallel'] == spec
    assert manifest['canvas']['proxy_width'] == 672
    assert manifest['cwm']['proxy_resize'] == 'lanczos'
    assert manifest['cwm']['posterior'] == 'mean'
    assert manifest['recipe']['bootstrap_latents'] == 9
    manifest['codec_parallel']['decoder_ranks'] = [0, 1]
    (output / 'native-runtime.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='codec_parallel'):
        read_native_bundle(output)
