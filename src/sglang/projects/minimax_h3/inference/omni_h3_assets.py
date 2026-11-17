import hashlib
import json
from copy import deepcopy
from pathlib import Path

MODEL_NAME = 'invoker-render'

RECIPES = ('longlive', 'continuous5', 'cwm')
FROZEN_RECIPE = Path(__file__).resolve().parents[1] / 'configs/recipes/omni-h3-longlive.json'


def _frozen_recipe(name='longlive'):
    path = FROZEN_RECIPE.with_name('omni-h3-cwm.json') if name == 'cwm' else FROZEN_RECIPE
    return json.loads(path.read_text(encoding='utf-8'))


def derive_native_recipe(source, name='longlive'):
    if name not in RECIPES:
        raise ValueError(f'unknown native H3 recipe: {name}')
    if source != _frozen_recipe(name):
        expected = 'CWM causal' if name == 'cwm' else 'step600 fixed-v2'
        raise ValueError(f'native integration requires the frozen {expected} recipe')
    effective = deepcopy(source)
    if name == 'continuous5':
        effective['segment_boundaries'] = []
    return effective


def native_bundle_artifact(manifest, artifact_path):
    path = Path(artifact_path).resolve()
    artifact_id = digest(path)
    if manifest['assets'].get(str(path)) != artifact_id:
        raise ValueError('source artifact is not pinned by this native bundle')
    artifact = json.loads(path.read_text(encoding='utf-8'))
    selection = manifest.get('recipe_selection')
    name = 'longlive' if selection is None else selection['name']
    effective = derive_native_recipe(artifact['recipe'], name)
    if effective != manifest['recipe']:
        raise ValueError('source artifact recipe differs from the native bundle')
    if selection is not None and selection != dict(name=name, source_artifact=str(path),
            source_artifact_sha256=artifact_id):
        raise ValueError('native recipe selection differs from the pinned source artifact')
    spec = artifact['encoders']['recipe']
    if any(spec[key] != value for key, value in manifest['canvas'].items()):
        raise ValueError('source artifact geometry differs from the native bundle')
    return artifact_id, artifact, effective


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare_native_bundle(artifact, destination, *, decoder_config, decoder_weights,
                          state_reservation_bytes, execution_config, codec_parallel=None,
                          decoder_backend='reference', decoder_compile=False,
                          decoder_cuda_graph=False, decoder_suffix_sdpa_backend='default',
                          decoder_spatial_attention_backend='fa2',
                          decoder_compact_suffix_kv=False,
                          decoder_compile_mode='default', decoder_tp_fused_post=False,
                          fuse_first_denoise=False, dit_norm_quality='exact', recipe='longlive', cwm_silence=None,
                          cwm_proxy_width=1344, cwm_proxy_resize='lanczos',
                          proxy_encoder='h3', tae_weights=None):
    _validate_codec_options(dict(encoder=proxy_encoder, tae_weights=(
        str(tae_weights) if tae_weights is not None else None)))
    if type(cwm_proxy_width) is not int or cwm_proxy_width not in (448, 1344):
        raise ValueError('CWM proxy encoder width must be 448 or 1344')
    if recipe != 'cwm' and cwm_proxy_width != 1344:
        raise ValueError('cwm_proxy_width requires the explicit CWM recipe')
    if cwm_proxy_resize not in ('lanczos', 'nearest'):
        raise ValueError('CWM proxy resize must be lanczos or nearest')
    if cwm_proxy_resize != 'lanczos' and (recipe != 'cwm' or cwm_proxy_width != 1344):
        raise ValueError('nearest resize requires the explicit full-size CWM recipe')
    artifact, destination = Path(artifact).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    bundle = json.loads(artifact.read_text(encoding='utf-8'))
    effective_recipe = derive_native_recipe(bundle['recipe'], recipe)
    weights = bundle['weights']
    if weights['format'] != 'raven_full_or_merged':
        raise ValueError('native H3 requires the complete merged runtime checkpoint')
    execution_config = Path(execution_config).resolve()
    execution = json.loads(execution_config.read_text(encoding='utf-8'))
    allowed = {'num_gpus', 'tp_size', 'sp_degree', 'ulysses_degree', 'ring_degree',
        'encoder_parallel', 'attention_backend', 'minimax_h3_adaln_online'}
    if not isinstance(execution, dict) or execution.keys() - allowed:
        raise ValueError('execution config accepts only native topology and backend fields')
    if ('minimax_h3_adaln_online' in execution
            and type(execution['minimax_h3_adaln_online']) is not bool):
        raise ValueError('minimax_h3_adaln_online must be boolean')
    if (execution.get('minimax_h3_adaln_online', False)
            and checkpoint_layout(weights['path'])['_class_name'] == 'MiniMaxH3Transformer3DModel'):
        raise ValueError('HF Diffusers checkpoints require minimax_h3_adaln_online=false '
                         '(configs/runs/omni-h3-cwm-diffusers-execution.json)')
    if type(execution.get('num_gpus')) is not int or execution['num_gpus'] not in (1, 2, 4, 8):
        raise ValueError('native H3 supports a single-node group of 1, 2, 4 or 8 GPUs')
    if state_reservation_bytes <= 0:
        raise ValueError('state_reservation_bytes must be a positive deployment reservation')
    checkpoint = Path(bundle['encoders']['checkpoint']).resolve()
    encoder_recipe = bundle['encoders']['recipe']
    if (encoder_recipe['conditioning_mode'] != 'static_image_qwen_proxy_vae_v1'
            or encoder_recipe['posterior'] != 'mean'):
        raise ValueError('native H3 requires static-image conditioning and the mean posterior')
    paths = {Path(weights['path']).resolve(): weights['sha256']}
    for relative, expected in encoder_recipe['model_hashes'].items():
        path = (checkpoint / relative).resolve()
        if not path.is_relative_to(checkpoint):
            raise ValueError('encoder hash entry escapes its checkpoint directory')
        paths[path] = expected
    for relative in ('video_vae/config.json', 'video_vae/source/config.json'):
        path = checkpoint / relative
        paths[path] = digest(path)
    for path, expected in paths.items():
        if digest(path) != expected:
            raise ValueError(f'asset digest differs from the frozen artifact: {path}')
    decoder_config, decoder_weights = Path(decoder_config).resolve(), Path(decoder_weights).resolve()
    if decoder_backend != 'fast' and (decoder_compile or decoder_cuda_graph):
        raise ValueError('decoder compile and CUDA Graph require the fast backend')
    codec = dict(encoder=proxy_encoder, decoder='causal', backend=decoder_backend, compile=decoder_compile,
        cuda_graph=decoder_cuda_graph, window=16, decoder_config=str(decoder_config),
        decoder_weights=str(decoder_weights), suffix_sdpa_backend=decoder_suffix_sdpa_backend,
        spatial_attention_backend=decoder_spatial_attention_backend,
        compact_suffix_kv=decoder_compact_suffix_kv, compile_mode=decoder_compile_mode,
        tp_fused_post=decoder_tp_fused_post)
    if proxy_encoder == 'tae':
        tae_weights = Path(tae_weights).resolve()
        codec['tae_weights'] = str(tae_weights)
        paths[tae_weights] = digest(tae_weights)
    _validate_codec_options(codec)
    paths[decoder_config] = digest(decoder_config)
    paths[decoder_weights] = digest(decoder_weights)
    paths[artifact] = digest(artifact)
    paths[execution_config] = digest(execution_config)
    server = dict(execution, model_path=str(checkpoint), model_variant='ref2va',
        nnodes=1, dp_size=1, enable_cfg_parallel=False, cfg_parallel_degree=1, kv_gather_degree=1,
        transformer_weights_path=str(Path(weights['path']).resolve()),
        component_paths=dict(transformer=str(destination / 'transformer'),
            text_encoder=str(checkpoint / 'text_encoder'), video_vae=str(checkpoint / 'video_vae'),
            processor=str(checkpoint / 'processor')))
    manifest = dict(schema='h3_omni_native/v1', recipe=effective_recipe, server_args=server,
        recipe_selection=dict(name=recipe, source_artifact=str(artifact),
            source_artifact_sha256=paths[artifact]),
        codec=codec, state_reservation_bytes=state_reservation_bytes,
        dit=dict(fuse_first_denoise=fuse_first_denoise, norm_quality=dit_norm_quality),
        assets={str(path): expected for path, expected in paths.items()},
        canvas={key: encoder_recipe[key] for key in ('height', 'width', 'proxy_height', 'proxy_width')})
    if recipe == 'cwm':
        manifest['model_name'] = MODEL_NAME
        if cwm_silence is None or execution['tp_size'] != 1:
            raise ValueError('CWM requires frozen silence and TP1')
        canvas = manifest['canvas']
        if ((canvas['height'], canvas['width']) != (768, 1344)
                or any(type(canvas[key]) is not int or canvas[key] <= 0 or canvas[key] % 32
                       for key in ('proxy_height', 'proxy_width'))
                or canvas['proxy_width'] * 4 != canvas['proxy_height'] * 7):
            raise ValueError('CWM requires a 1344x768 output and a matching-aspect proxy on a 32-pixel grid')
        silence = str(Path(cwm_silence).resolve())
        manifest['assets'][silence] = digest(silence)
        manifest['cwm'] = dict(silence_path=silence,
            proxy_resize='none' if cwm_proxy_width == 448 else cwm_proxy_resize, proxy_encode_width=cwm_proxy_width,
            proxy_encode_height=cwm_proxy_width * 4 // 7, posterior='mean')
    elif cwm_silence is not None:
        raise ValueError('cwm_silence requires the explicit CWM recipe')
    if codec_parallel is not None:
        manifest['codec_parallel'] = codec_parallel
    _validate_codec_parallel(manifest)
    _validate_dit_options(manifest)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'transformer').mkdir()
    files = {
        'config.json': {'architectures': ['MiniMaxH3CausalDiTModel']},
        'native-runtime.json': manifest,
        'transformer/config.json': dict(bundle['model']['args'], **checkpoint_layout(weights['path'])),
    }
    for relative, value in files.items():
        (destination / relative).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    for relative in ('config.json', 'transformer/config.json'):
        manifest['assets'][str(destination / relative)] = digest(destination / relative)
    (destination / 'native-runtime.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def checkpoint_layout(weights_path):
    """Transformer class and QKV row layout that the weight file actually stores.

    Diffusers exports keep separate to_q/to_k/to_v and [value, gate] FFN rows; the
    loader converts them only when the checkpoint class is the Diffusers name.
    Native exports tagged ``qkv_layout: released_per_head`` interleave Q/K/V per
    head and must be regrouped; untagged native exports already hold q/k/v blocks.
    """
    path = Path(weights_path)
    if path.name.endswith('.safetensors.index.json'):
        config = path.with_name('config.json')
        if config.is_file() and json.loads(config.read_text(encoding='utf-8')).get(
                '_class_name') == 'MiniMaxH3Transformer3DModel':
            return dict(_class_name='MiniMaxH3Transformer3DModel', checkpoint_qkv_layout='native')
    elif path.suffix == '.safetensors':
        with path.open('rb') as handle:
            length = int.from_bytes(handle.read(8), 'little')
            metadata = json.loads(handle.read(length)).get('__metadata__') or {}
        if metadata.get('qkv_layout') == 'released_per_head':
            return dict(_class_name='MiniMaxH3CausalDiTModel', checkpoint_qkv_layout='grouped')
    return dict(_class_name='MiniMaxH3CausalDiTModel', checkpoint_qkv_layout='native')


def read_native_bundle(model_path):
    path = Path(model_path) / 'native-runtime.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['schema'] != 'h3_omni_native/v1':
        raise ValueError('unsupported native H3 bundle')
    _validate_codec_options(manifest['codec'])
    if (manifest['codec'].get('encoder') == 'tae'
            and manifest['codec']['tae_weights'] not in manifest['assets']):
        raise ValueError('TAE proxy weights must be pinned in native assets')
    _validate_codec_parallel(manifest)
    _validate_dit_options(manifest)
    _validate_recipe_selection(manifest)
    cwm = manifest.get('cwm')
    if manifest['recipe'].get('protocol') == 'cwm_causal/v1':
        size = (cwm.get('proxy_encode_width'), cwm.get('proxy_encode_height')) if isinstance(cwm, dict) else None
        resize = cwm.get('proxy_resize') if isinstance(cwm, dict) else None
        if (size not in ((448, 256), (1344, 768))
                or resize not in (('none',) if size == (448, 256) else ('lanczos', 'nearest'))
                or cwm != dict(silence_path=cwm.get('silence_path'),
                proxy_resize=resize,
                proxy_encode_width=size[0], proxy_encode_height=size[1],
                posterior='mean') or cwm['silence_path'] not in manifest['assets']):
            raise ValueError('CWM requires pinned silence and a 448x256 or 1344x768 proxy encoder contract')
    elif cwm is not None:
        raise ValueError('CWM assets require the CWM recipe')
    return manifest


def _validate_recipe_selection(manifest):
    selection = manifest.get('recipe_selection')
    if selection is None:
        return
    if (not isinstance(selection, dict)
            or set(selection) != {'name', 'source_artifact', 'source_artifact_sha256'}
            or selection['name'] not in RECIPES
            or not isinstance(selection['source_artifact'], str)
            or not isinstance(selection['source_artifact_sha256'], str)
            or manifest['assets'].get(selection['source_artifact']) != selection['source_artifact_sha256']):
        raise ValueError('invalid native recipe selection or unpinned source artifact')
    # The source artifact itself is hashed only where a caller names it (native_bundle_artifact).
    if manifest['recipe'] != derive_native_recipe(_frozen_recipe(selection['name']), selection['name']):
        raise ValueError('native bundle recipe differs from its recipe selection')


def _validate_dit_options(manifest):
    spec = manifest.get('dit', {})
    if (not isinstance(spec, dict) or spec.keys() - {'fuse_first_denoise', 'norm_quality'}
            or type(spec.get('fuse_first_denoise', False)) is not bool):
        raise ValueError('dit accepts only fuse_first_denoise (boolean) and norm_quality options')
    quality = spec.get('norm_quality', 'exact')
    if type(quality) is not str or quality not in ('exact', 'lossless'):
        raise ValueError('dit.norm_quality must be exact or lossless')


def _validate_codec_options(codec):
    encoder = codec.get('encoder', 'h3')
    if encoder not in ('h3', 'tae'):
        raise ValueError('codec.encoder must be h3 or tae')
    weights = codec.get('tae_weights')
    if encoder == 'tae' and (not isinstance(weights, str) or not weights):
        raise ValueError('TAE proxy encoding requires tae_weights')
    if encoder == 'h3' and weights is not None:
        raise ValueError('tae_weights requires the TAE proxy encoder')
    mode = codec.get('compile_mode', 'default')
    fused = codec.get('tp_fused_post', False)
    if mode not in ('default', 'max-autotune-no-cudagraphs'):
        raise ValueError('codec.compile_mode must be default or max-autotune-no-cudagraphs')
    if type(fused) is not bool:
        raise ValueError('codec.tp_fused_post must be boolean')
    if (mode != 'default' or fused) and (codec.get('backend') != 'fast'
            or codec.get('decoder') != 'causal' or codec.get('compile') is not True):
        raise ValueError('decoder compile tuning requires a compiled fast causal decoder')
    backend = codec.get('suffix_sdpa_backend', 'default')
    if backend not in ('default', 'cudnn', 'fa3'):
        raise ValueError('codec.suffix_sdpa_backend must be default, cudnn or fa3')
    if backend == 'cudnn' and (codec.get('decoder') != 'causal' or codec.get('backend') != 'fast'):
        raise ValueError('cuDNN suffix SDPA requires the fast causal decoder')
    if backend == 'fa3' and (codec.get('decoder') != 'causal' or codec.get('backend') != 'fast'
                             or not codec.get('compact_suffix_kv', False)):
        raise ValueError('FA3 suffix requires the fast causal decoder and compact_suffix_kv')
    spatial = codec.get('spatial_attention_backend', 'fa2')
    if spatial not in ('fa2', 'fa3'):
        raise ValueError('codec.spatial_attention_backend must be fa2 or fa3')
    if type(codec.get('compact_suffix_kv', False)) is not bool:
        raise ValueError('codec.compact_suffix_kv must be boolean')


def _validate_codec_parallel(manifest):
    if manifest['codec'].get('tp_fused_post', False) and 'codec_parallel' not in manifest:
        raise ValueError('codec.tp_fused_post requires codec_parallel')
    if 'codec_parallel' not in manifest:
        return
    spec = manifest['codec_parallel']
    decoder = spec.get('decoder_ranks') if isinstance(spec, dict) else None
    valid_decoder = (isinstance(decoder, list) and len(decoder) in (2, 4)
        and tuple(sorted(decoder)) == tuple(decoder) and decoder[0] == 0
        and len(set(decoder)) == len(decoder)
        and all(decoder[i:i + 2] in ([0, 1], [2, 3], [4, 5], [6, 7])
                for i in range(0, len(decoder), 2)))
    if (not isinstance(spec, dict) or set(spec) != {'encoder_ranks', 'decoder_ranks'}
            or any(type(spec[key]) is not list
                or any(type(rank) is not int for rank in spec[key]) for key in spec)
            or spec['encoder_ranks'] not in ([0, 1], [2, 3], [4, 5], [6, 7], [2, 3, 6, 7])
            or (spec['encoder_ranks'] == [2, 3, 6, 7] and decoder != [0, 1, 4, 5])
            or not valid_decoder):
        raise ValueError('codec_parallel requires supported encoder ranks and a pair-aligned decoder group containing rank 0')
    codec = manifest['codec']
    if (manifest['server_args']['num_gpus'] != 8 or codec.get('backend') != 'fast'
            or codec.get('decoder') != 'causal' or codec.get('encoder') not in ('h3', 'tae')
            or type(codec.get('compile')) is not bool or type(codec.get('cuda_graph')) is not bool):
        raise ValueError('codec_parallel requires eight GPUs and the fast causal H3 or TAE codec')


def verify_native_bundle(manifest):
    """Re-hash every pinned asset; costly (full weights), so only on explicit request."""
    for path, expected in manifest['assets'].items():
        if digest(path) != expected:
            raise ValueError(f'prepared native asset changed: {path}')
