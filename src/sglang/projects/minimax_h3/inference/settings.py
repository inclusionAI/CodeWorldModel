"""Torch-free run manifests, deployment options and comparable settings receipts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

import yaml

RUN_SCHEMA = 'h3_inference_run/v1'
EFFECTIVE_SCHEMA = 'h3_inference_effective/v1'
PROTOCOLS = {
    'infer_ref_fixed_history': 'ref_fixed_recent_history/v1',
    'infer_ref_bidir_history': 'ref_bidir_recent_history/v1',
    'infer_ref_fixed_history_streaming_text': 'ref_fixed_recent_history_streaming_text/v1',
    'infer_ref_bidir_history_streaming_text': 'ref_bidir_recent_history_streaming_text/v1',
}
ENTRYPOINTS = (*PROTOCOLS, 'ref2va_infer', 'serve_ref', 'infer_ref_raw_latent')
PATH_FIELDS = {'artifact', 'config', 'weights', 'checkpoint', 'output_dir', 'bundle', 'gpu_occupier_config',
               'proxy_video', 'first_frame', 'prompt_file', 'prompt_updates', 'prompts_jsonl'}
GRAPH_RUNTIME_ENVIRONMENT = (
    'NCCL_GRAPH_REGISTER', 'NCCL_GRAPH_MIXING_SUPPORT', 'PYTORCH_ALLOC_CONF',
    'PYTORCH_CUDA_ALLOC_CONF', 'TORCH_NCCL_AVOID_RECORD_STREAMS',
    'TORCH_NCCL_ASYNC_ERROR_HANDLING', 'NCCL_CUMEM_HOST_ENABLE',
    'CUDA_MODULE_LOADING', 'MINIMAX_H3_ATTN_BACKEND')


def require_dit_graph_environment(enabled):
    """Validate the launch contract before CUDA/NCCL initialization."""
    if not enabled:
        return
    if os.environ.get('NCCL_GRAPH_REGISTER') != '0':
        raise RuntimeError('dit.cuda_graph requires NCCL_GRAPH_REGISTER=0 in the launch environment '
                           'before CUDA/NCCL initialization. Restart with NCCL_GRAPH_REGISTER=0.')
    if os.environ.get('NCCL_GRAPH_MIXING_SUPPORT') not in (None, '1'):
        raise RuntimeError('dit.cuda_graph requires NCCL_GRAPH_MIXING_SUPPORT=1 or unset before '
                           'CUDA/NCCL initialization. Restart with NCCL_GRAPH_MIXING_SUPPORT=1.')


class UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ValueError('configuration keys must be strings')
        if key in result:
            raise ValueError(f'duplicate configuration key: {key}')
        result[key] = loader.construct_object(value_node, deep=True)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def read_settings(path):
    value = yaml.load(Path(path).read_text(encoding='utf-8'), Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise ValueError('configuration must be a mapping')
    return value


def _path(value, base):
    path = Path(value).expanduser()
    if path.is_absolute() or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return str(path)
    return str((Path(base) / path).resolve())


def load_inference_options(path=None, *, data=None, base=None):
    """Resolve the existing closed deployment schema before loading any models."""
    if path is not None and data is not None:
        raise ValueError('provide inference options as path or data, not both')
    options = dict(profile=False, resident_weights=True, qwen_fsdp=False,
        dit=dict(metadata=False, kv=False, adaln=False, delayed_hc=False, fuse_first_denoise=False,
                 rope=False, cuda_graph=False),
        codec=dict(encoder='h3', decoder='h3', tae_weights=None, decoder_weights=None,
                   decoder_config=None, window=16, backend='reference', compile=False, cuda_graph=False),
        output=dict(async_write=False, queue_size=2))
    source = read_settings(path) if path is not None else data
    def merge(dst, src, prefix=''):
        if not isinstance(src, dict):
            raise ValueError(f'{prefix or "inference config"} must be a mapping')
        for key, value in src.items():
            name = prefix + str(key)
            if key not in dst:
                raise ValueError(f'unknown inference option: {name}')
            if isinstance(dst[key], dict):
                merge(dst[key], value, name + '.')
            elif isinstance(dst[key], bool) and type(value) is not bool:
                raise ValueError(f'{name} must be a boolean')
            else:
                dst[key] = value
    if source is not None:
        merge(options, source)
    if options['dit']['fuse_first_denoise']:
        if not options['dit']['delayed_hc'] or options['dit']['cuda_graph']:
            raise ValueError('dit.fuse_first_denoise requires dit.delayed_hc: true and dit.cuda_graph: false')
    codec = options['codec']
    for key, choices in [('encoder', ('h3', 'tae')), ('decoder', ('h3', 'causal')),
                         ('backend', ('reference', 'fast'))]:
        if codec[key] not in choices:
            raise ValueError(f'codec.{key} must be one of {choices}')
    for value, name in [(codec['window'], 'codec.window'),
                        (options['output']['queue_size'], 'output.queue_size')]:
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if codec['decoder'] == 'h3' and (codec['backend'] != 'reference' or codec['compile'] or codec['cuda_graph']):
        raise ValueError('fast/compile/cuda_graph require codec.decoder: causal')
    if codec['backend'] != 'fast' and (codec['compile'] or codec['cuda_graph']):
        raise ValueError('compile/cuda_graph require codec.backend: fast')
    required = [] if codec['encoder'] == 'h3' else ['tae_weights']
    if codec['decoder'] == 'causal':
        required += ['decoder_weights', 'decoder_config']
    asset_base = Path(path).resolve().parent if path is not None else Path(base or Path.cwd())
    for key in ('tae_weights', 'decoder_weights', 'decoder_config'):
        if codec[key] is not None and (not isinstance(codec[key], str) or not codec[key]):
            raise ValueError(f'codec.{key} must be a nonempty path string')
        if key in required:
            if not codec[key]:
                raise ValueError(f'codec.{key} is required')
            file = Path(_path(codec[key], asset_base))
            if not file.is_file():
                raise ValueError(f'missing codec.{key}: {file}')
            codec[key] = str(file.resolve())
    return options


def backbone_dit_options(options):
    return {key: value for key, value in options['dit'].items() if key != 'fuse_first_denoise'}


def _validate_entrypoint_options(options, entrypoint):
    if options['dit']['fuse_first_denoise'] and entrypoint != 'serve_ref':
        raise ValueError('dit.fuse_first_denoise is supported only by serve_ref')
    if options['dit']['delayed_hc'] and entrypoint not in ('serve_ref', 'infer_ref_raw_latent'):
        raise ValueError('delayed_hc requires a fixed-history v2 recipe')
    if options['qwen_fsdp'] and entrypoint != 'serve_ref':
        raise ValueError('qwen_fsdp is supported only by serve_ref')


def _manifest(path, entrypoint=None):
    record = read_settings(path)
    if set(record) != {'schema', 'entrypoint', 'run', 'deployment'} or record['schema'] != RUN_SCHEMA:
        raise ValueError(f'run config requires schema {RUN_SCHEMA}, entrypoint, run and deployment only')
    if not isinstance(record['entrypoint'], str) or record['entrypoint'] not in ENTRYPOINTS:
        raise ValueError('unsupported inference entrypoint')
    if entrypoint is not None and record['entrypoint'] != entrypoint:
        raise ValueError(f'run config entrypoint {record["entrypoint"]} does not match {entrypoint}')
    if not isinstance(record['run'], dict) or not isinstance(record['deployment'], dict):
        raise ValueError('run and deployment must be mappings')
    return record


def _value_tokens(action, value, base):
    name = action.dest
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if type(value) is not bool:
            raise ValueError(f'run.{name} must be a boolean')
        return [action.option_strings[0]] if value == action.const else []
    sequence = action.nargs == '+'
    values = value if isinstance(action, argparse._AppendAction) or sequence else [value]
    if not isinstance(values, list) or not values:
        raise ValueError(f'run.{name} must be a nonempty list')
    result = [action.option_strings[0]] if sequence else []
    for item in values:
        expected = action.type or str
        valid = (type(item) is int if expected is int else
                 type(item) in (int, float) and math.isfinite(item) if expected is float else
                 isinstance(item, str))
        if not valid:
            raise ValueError(f'run.{name} has an invalid value type')
        if name in PATH_FIELDS:
            item = _path(item, base)
        result.extend([str(item)] if sequence else [action.option_strings[0], str(item)])
    return result


def parse_run_arguments(parser, argv, entrypoint):
    """Expand a manifest through the caller's real argparse actions and constraints."""
    if entrypoint not in ENTRYPOINTS:
        raise ValueError('unsupported inference entrypoint')
    if '--run-config' not in parser._option_string_actions:
        parser.add_argument('--run-config', help='Unified inference run YAML')
    cli = list(sys.argv[1:] if argv is None else argv)
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument('--run-config')
    config, _ = probe.parse_known_args(cli)
    if config.run_config is None:
        args = parser.parse_args(cli)
        args._settings_entrypoint = entrypoint
        return args
    try:
        record = _manifest(config.run_config, entrypoint)
        actions = {a.dest: a for a in parser._actions if a.option_strings}
        supplied = []
        for token in cli:
            if token.startswith('-'):
                option = token.split('=', 1)[0]
                action = parser._option_string_actions.get(option)
                if action is None and parser.allow_abbrev and option.startswith('--'):
                    matches = {a.dest: a for flag, a in parser._option_string_actions.items()
                               if flag.startswith(option)}
                    if len(matches) == 1:
                        action = next(iter(matches.values()))
                if action is not None:
                    supplied.append(action.dest)
        if supplied.count('run_config') != 1:
            raise ValueError('--run-config must be specified exactly once')
        repeated = {name for name in supplied if supplied.count(name) > 1}
        if repeated:
            raise ValueError(f'duplicate CLI fields: {", ".join(sorted(repeated))}')
        if 'inference_config' in supplied or 'inference_config' in record['run']:
            raise ValueError('--inference-config cannot be mixed with --run-config')
        if 'run_config' in record['run']:
            raise ValueError('run.run_config is not allowed')
        duplicates = set(supplied) & record['run'].keys()
        if duplicates:
            raise ValueError(f'duplicate YAML/CLI fields: {", ".join(sorted(duplicates))}')
        unexpected = set(supplied) - {'run_config', 'output_dir', 'check_only', 'help'}
        if unexpected:
            raise ValueError(f'manifest mode accepts only --output-dir/--check-only CLI controls: {sorted(unexpected)}')
        tokens = []
        base = Path(config.run_config).resolve().parent
        for name, value in record['run'].items():
            if name not in actions or name == 'help':
                raise ValueError(f'unknown run option: {name}')
            tokens.extend(_value_tokens(actions[name], value, base))
        options = load_inference_options(data=record['deployment'], base=base)
        _validate_entrypoint_options(options, entrypoint)
        inference_action = actions.get('inference_config')
        was_required = inference_action.required if inference_action else False
        if inference_action:
            inference_action.required = False
        try:
            args = parser.parse_args(tokens + cli)
        finally:
            if inference_action:
                inference_action.required = was_required
        args.run_settings = dict(path=str(Path(config.run_config).resolve()),
            sha256=hashlib.sha256(Path(config.run_config).read_bytes()).hexdigest(),
            entrypoint=entrypoint, requested_deployment=deepcopy(record['deployment']), options=options)
        args._settings_entrypoint = entrypoint
        return args
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))


def options_for_args(args):
    settings = getattr(args, 'run_settings', None)
    options = deepcopy(settings['options']) if settings else load_inference_options(getattr(args, 'inference_config', None))
    _validate_entrypoint_options(options, getattr(args, '_settings_entrypoint', None))
    return options


def inspection_parser(entrypoint):
    """Shared public CLI parser for execution and CPU-only inspection."""
    if entrypoint not in ENTRYPOINTS:
        raise ValueError('unsupported inference entrypoint')
    parser = argparse.ArgumentParser(description=f'Run {entrypoint} with a unified manifest or explicit legacy flags.')
    parser.add_argument('--inference-config', required=entrypoint == 'serve_ref',
        help='Deployment-only YAML; defaults to resident weights with optional DiT caches disabled')
    parser.add_argument('--output-dir', required=True, help='New output directory; existing outputs are never overwritten')
    if entrypoint == 'serve_ref':
        parser.add_argument('--artifact', action='append', required=True)
        parser.add_argument('--bind', default='127.0.0.1:8080')
        parser.add_argument('--timeout', type=float, default=300)
        parser.add_argument('--idle-timeout', type=float, default=300)
        parser.add_argument('--gpu-occupier-config', help='Trusted local Linux process identity JSON; Ray runtime only')
        parser.add_argument('--max-body-mib', type=int, default=128)
        parser.add_argument('--runtime', choices=('torchrun', 'ray'), default='torchrun')
        parser.add_argument('--ray-address', help='Existing Ray cluster address; omit to start a local cluster')
        parser.add_argument('--ray-placement', choices=('PACK', 'SPREAD', 'STRICT_PACK', 'STRICT_SPREAD'), default='PACK')
        parser.add_argument('--ray-cpus-per-worker', type=int, default=2)
        parser.add_argument('--ray-master-port', type=int, default=29500)
        parser.add_argument('--ray-startup-timeout', type=float, default=900)
        parser.add_argument('--vae-rank', type=int, default=0)
        parser.add_argument('--decoder-rank', type=int, default=0)
        parser.add_argument('--codec-overlap', action='store_true',
                            help='Prefetch one pending media proxy during decode; Ray fixed v2 only')
        return parser
    if entrypoint == 'ref2va_infer':
        for key in ('config', 'weights', 'checkpoint'):
            parser.add_argument('--' + key, required=True)
    else:
        parser.add_argument('--artifact', required=True, help='Exported immutable Ref inference artifact JSON')
    from .ref2va import request_arguments
    request_arguments(parser)
    if entrypoint.endswith('_streaming_text'):
        prompts = parser.add_mutually_exclusive_group(required=True)
        prompts.add_argument('--prompt-updates')
        prompts.add_argument('--prompts-jsonl')
    if entrypoint == 'infer_ref_raw_latent':
        from .raw_recipe import window_arguments
        window_arguments(parser)
        prompts = parser.add_mutually_exclusive_group()
        prompts.add_argument('--prompt-updates')
        prompts.add_argument('--prompts-jsonl')
    return parser


def _recipe_class():
    if __package__.startswith('projects.'):
        from ..modeling.ref_recipe import RefRecipe
    else:
        from modeling.ref_recipe import RefRecipe
    return RefRecipe


def _parse_recipe(value):
    from .raw_recipe import parse_recipe
    return parse_recipe(value)


def _training_config(path):
    source = Path(__file__).resolve().parents[1] / 'frameworks/raven/common/config.py'
    if not source.is_file():
        raise ValueError('native settings inspection requires the pinned frameworks/raven checkout')
    spec = importlib.util.spec_from_file_location('_h3_settings_raven_config', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CfgNode.from_file(path)


def effective_settings(args, bundle=None, recipe=None, options=None, implementation=None, training=None, request=None):
    """Describe resolved settings; asset verification and hardware observation are separate."""
    options = deepcopy(options if options is not None else options_for_args(args))
    settings = getattr(args, 'run_settings', {})
    run = {k: deepcopy(v) for k, v in vars(args).items() if not k.startswith('_') and k not in ('run_settings', 'run_config')}
    entrypoint = settings.get('entrypoint', getattr(args, '_settings_entrypoint', None))
    _validate_entrypoint_options(options, entrypoint)
    artifacts = []
    paths = run.get('artifact', [])
    paths = [paths] if isinstance(paths, str) else paths
    if bundle is not None and 'recipe' not in bundle:
        bundles = [value[0] for value in bundle.values()]
    else:
        bundles = [bundle] if bundle is not None else [read_settings(p) for p in paths]
    for index, item in enumerate(bundles):
        original = deepcopy(item['recipe'])
        value = _parse_recipe(original)
        if entrypoint in PROTOCOLS and value.protocol != PROTOCOLS[entrypoint]:
            raise ValueError(f'entrypoint requires {PROTOCOLS[entrypoint]}, artifact selects {value.protocol}')
        if getattr(args, 'steps', None) is not None:
            value = value.with_steps(args.steps)
        if recipe is not None and len(bundles) == 1:
            value = recipe if hasattr(recipe, 'to_dict') else _parse_recipe(recipe)
        if entrypoint == 'infer_ref_raw_latent':
            from .raw_recipe import RAW_PROTOCOLS, window_overrides
            from dataclasses import replace
            if value.protocol not in RAW_PROTOCOLS:
                raise ValueError('raw-latent entrypoint requires a v2 artifact')
            value = replace(value, **window_overrides(args, value.to_dict()))
        if entrypoint in ('serve_ref', 'infer_ref_raw_latent'):
            from .raw_recipe import validate_online_guidance
            validate_online_guidance(value)
        if options['dit']['fuse_first_denoise']:
            from .raw_recipe import RAW_FIXED
            if (value.protocol != RAW_FIXED or value.guidance['mode'] != 'plain'
                    or value.guidance['scale'] != 1):
                raise ValueError('dit.fuse_first_denoise requires a static fixed v2 recipe with plain scale-1 guidance')
            if value.diffusion['sampling_timesteps'].get('args', {}).get('dynamic_shift', False):
                raise ValueError('dit.fuse_first_denoise requires a static sampling grid')
        if options['dit']['delayed_hc']:
            from .raw_recipe import RawLatentRecipe
            if not isinstance(value, RawLatentRecipe) or value.bidirectional:
                raise ValueError('delayed_hc requires a fixed-history v2 recipe')
        artifacts.append(dict(path=paths[index] if index < len(paths) else None,
            recipe=value.to_dict(), source_recipe=original, weights=deepcopy(item['weights']),
            encoders=deepcopy(item['encoders']), model=deepcopy(item.get('model')),
            deployment=deepcopy(item['deployment']), provenance=deepcopy(item.get('provenance')),
            verification='declared artifact identities; not rehashed by settings inspection'))
    if not artifacts and run.get('config'):
        training = training if training is not None else _training_config(run['config'])
        value = _recipe_class().from_training(training)
        source = value.to_dict()
        if recipe is not None:
            value = recipe if hasattr(recipe, 'to_dict') else _parse_recipe(recipe)
        if getattr(args, 'steps', None) is not None:
            value = value.with_steps(args.steps)
        artifacts.append(dict(recipe=value.to_dict(), source_recipe=source,
            weights=dict(path=run['weights'], format='raven_full_or_merged', sha256=None),
            encoders=dict(checkpoint=run['checkpoint'], verification='encoder hashes require runtime validation'),
            model={key: deepcopy(training['models']['backbone'][key]) for key in ('module', 'class_name', 'args')
                   if key in training['models']['backbone']},
            deployment=dict(distributed=training['distributed'], placement=training['models']['backbone']['placement'],
                            active_up_size=training['engine']['up_size']['train'])))
    if entrypoint == 'serve_ref':
        if not options['resident_weights'] or options['codec']['decoder'] != 'causal':
            raise ValueError('serving requires causal decoder and resident_weights: true')
        if options['output']['async_write']:
            raise ValueError('serving requires output.async_write: false')
        if run['codec_overlap']:
            from .raw_recipe import RAW_FIXED
            if (run['runtime'] != 'ray' or run['vae_rank'] == run['decoder_rank']
                    or options['codec']['encoder'] != 'h3'
                    or any(a['recipe']['protocol'] != RAW_FIXED for a in artifacts)):
                raise ValueError('codec-overlap requires Ray, distinct codec owners, H3 encoder and static fixed v2')
        identities = [deepcopy({key: a[key] for key in ('model', 'weights', 'encoders', 'deployment')}) for a in artifacts]
        for identity in identities:
            identity['encoders']['recipe'].pop('encoder_code', None)
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError('all served artifacts must have identical resident assets and deployment')
    if options['codec']['decoder'] == 'causal' and run.get('decode_mode', 'strict') != 'strict':
        raise ValueError('causal decoder requires --decode-mode strict')
    if entrypoint == 'infer_ref_raw_latent' and options['codec']['decoder'] == 'h3' and run.get('decode_mode', 'strict') == 'strict':
        raise ValueError('raw-latent strict output requires codec.decoder: causal')
    resolved_request = None
    if entrypoint != 'serve_ref':
        start = run.get('start_frame')
        window = run.get('window_index')
        count = run.get('num_frames', 158)
        if start is not None and window is not None:
            raise ValueError('choose --start-frame or --window-index, not both')
        if count < 5 or count % 17 != 5:
            raise ValueError('--num-frames must be 17*n+5')
        if (start is not None and start < 0) or (window is not None and window < 0):
            raise ValueError('start/window index must be nonnegative')
        resolved_request = dict(seed=run.get('seed', 2026),
            condition_seed=run.get('seed', 2026) if run.get('condition_seed') is None else run['condition_seed'],
            first_source_frame=start if start is not None else (window or 0) * 158,
            num_frames=count, decode_mode=run.get('decode_mode', 'strict'))
        if request is not None:
            resolved_request['input_spec'] = deepcopy(request)
    selected = ('ResidentInferencePlacement' if options['resident_weights'] else 'artifact_or_training_placement')
    from .raw_recipe import RAW_PROTOCOLS
    for artifact in artifacts:
        artifact['units'] = dict(history_window='native_latents' if artifact['recipe']['protocol'] in RAW_PROTOCOLS
                                 else 'past_time_groups', decoder_window='spatial')
    history_units = {artifact['units']['history_window'] for artifact in artifacts}
    return dict(schema=EFFECTIVE_SCHEMA, source=dict(entrypoint=entrypoint,
            **{k: settings[k] for k in ('path', 'sha256') if k in settings}),
        run=run, resolved_request=resolved_request, requested_deployment=deepcopy(settings.get('requested_deployment')),
        deployment=dict(options=options, selected_placement=selected,
            qwen_ranks=[0, 1] if options['qwen_fsdp'] else [0],
            distributed=[a['deployment'] for a in artifacts], world_size_environment=os.environ.get('WORLD_SIZE'),
            observed_runtime=None, optimization_state='requested flags; activation and cache hits not observed'),
        artifacts=artifacts, sampling_steps_overridden=getattr(args, 'steps', None) is not None,
        units=dict(history_window=next(iter(history_units)) if len(history_units) == 1
                   else 'mixed_by_artifact' if history_units else 'past_time_groups', decoder_window='spatial'),
        implementation=implementation, inspection='structural settings inspection; no hardware or weight verification')


def inspect_config(path):
    record = _manifest(path)
    parser = inspection_parser(record['entrypoint'])
    parser._option_string_actions['--output-dir'].required = False
    args = parse_run_arguments(parser, ['--run-config', str(path)], record['entrypoint'])
    return effective_settings(args)


def diff_settings(left, right, prefix=''):
    """Return dotted paths, preserving the difference between missing and null."""
    differences = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(left.keys() | right.keys()):
            path = f'{prefix}.{key}' if prefix else key
            if key not in left:
                differences.append(dict(path=path, change='added', after=right[key]))
            elif key not in right:
                differences.append(dict(path=path, change='removed', before=left[key]))
            else:
                differences.extend(diff_settings(left[key], right[key], path))
    elif isinstance(left, list) and isinstance(right, list):
        for index in range(max(len(left), len(right))):
            path = f'{prefix}[{index}]'
            if index >= len(left):
                differences.append(dict(path=path, change='added', after=right[index]))
            elif index >= len(right):
                differences.append(dict(path=path, change='removed', before=left[index]))
            else:
                differences.extend(diff_settings(left[index], right[index], path))
    elif type(left) is not type(right) or left != right:
        differences.append(dict(path=prefix, change='changed', before=left, after=right))
    return differences
