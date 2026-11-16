"""Stream a proxy video through a trained causal Ref2VA export.

One Qwen call on the first RGB picture and the fixed prompt, then only proxy
VAE chunks. Run with torchrun inside an allocated GPU task. --check-only is CPU.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path

from projects.minimax_h3.inference.ref2va import Ref2VAStreamEngine, input_spec
from projects.minimax_h3.inference.settings import parse_run_arguments, options_for_args, effective_settings, inspection_parser, require_dit_graph_environment


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_recipe(config, checkpoint, prompt_file):
    """Read the training encoder recipe, without loading model/tensor values."""
    from projects.minimax_h3.data.ref_latent import STATIC_CONDITIONING_MODE, read_manifest
    manifest, entries, _ = read_manifest(config['data']['args']['manifest'])
    entry = entries[0]
    path = Path(entry['path'])
    path = path if path.is_absolute() else manifest.parent / path
    recipe_path = path.parent / 'config.json'
    recipe = json.loads(recipe_path.read_text())
    if (config['data']['args'].get('conditioning_mode') != STATIC_CONDITIONING_MODE
            or recipe.get('conditioning_mode') != STATIC_CONDITIONING_MODE):
        raise ValueError('inference requires a static_image_qwen_proxy_vae_v1 training config and cache')
    prompt = Path(prompt_file).read_text(encoding='utf-8').strip()
    if not prompt or prompt != recipe['system_prompt']:
        raise ValueError('fixed prompt differs from the training encoder recipe')
    if recipe.get('posterior') != 'mean':
        raise ValueError('live inference requires the deterministic posterior-mean recipe')
    saved_path = manifest.parent / 'encoder_hashes.json'
    if not saved_path.exists():
        # Smoke manifests live outside the frozen cache; reuse its stat-checked hashes.
        saved_path = recipe_path.parents[2] / 'encoder_hashes.json'
    saved = json.loads(saved_path.read_text()) if saved_path.exists() else {}
    if not recipe.get('model_hashes'):
        raise ValueError('training recipe has no frozen encoder content hashes')
    base = Path(__file__).resolve().parents[1]
    for relative, expected in recipe.get('encoder_code', {}).items():
        if sha256(base/relative) != expected:
            raise ValueError(f'encoder/preprocessing implementation differs from training: {relative}')
    for relative, expected in recipe['model_hashes'].items():
        current = (Path(checkpoint) / relative).resolve()
        st = current.stat()
        stamp = {k: getattr(st, 'st_'+k) for k in ('dev','ino','size','mtime_ns','ctime_ns')}
        cached = saved.get(str(current), {})
        digest = cached.get('sha256') if cached.get('stamp') == stamp else sha256(current)
        after = current.stat()
        if stamp != {k: getattr(after, 'st_'+k) for k in ('dev','ino','size','mtime_ns','ctime_ns')}:
            raise ValueError(f'encoder changed during verification: {current}')
        if digest != expected:
            raise ValueError(f'frozen encoder differs from training: {current}')
    return recipe, recipe_path


def inference_config(args, training, spec):
    """Keep the trained causal layout/sampler, with only the exported generator."""
    from common.config import CfgNode
    options = options_for_args(args)
    if options['codec']['decoder'] == 'causal' and args.decode_mode != 'strict':
        raise ValueError('causal decoder requires --decode-mode strict (online publication)')
    from projects.minimax_h3.modeling.ref_recipe import RefRecipe
    recorded = 'inference_recipe' in training
    if recorded:
        RefRecipe.from_training(training)
    cfg = CfgNode(copy.deepcopy(training))
    cfg.pop('inference_recipe', None)
    stage = cfg.data.args.stage
    if stage not in ('tf', 'tscd', 'dmd'):
        raise ValueError('AR inference requires a TF/TSCD/DMD config, not bidirectional SFT')
    backbone = copy.deepcopy(cfg.models.backbone)
    for key in ('optimizer', 'adapter', 'ema'):
        backbone.pop(key, None)
    backbone['weight'] = CfgNode(dict(module='common.model.weights', class_name='DefaultWeightLoader',
                                     path=str(Path(args.weights).resolve()), strict=True))
    backbone['runtime'] = CfgNode(dict(training=False, requires_grad=False, param_dtype='bfloat16'))
    if options['resident_weights']:
        backbone['placement'] = CfgNode(dict(module='projects.minimax_h3.modeling.inference_runtime',
                                             class_name='ResidentInferencePlacement'))
    cfg.models = dict(backbone=backbone)
    cfg.engine.resume = 'never'
    cfg.engine.pop('resume_dir', None)
    cfg.logging = {}
    output = Path(args.output_dir).resolve()
    cfg.persistence = dict(output_dir=str(output.parent), proj_name=output.name, exp_name='runtime',
                           checkpoint={'async': False, 'keep_latest_k': 2})
    cfg.entry = dict(module='projects.minimax_h3.tools.ref2va_infer', class_name='Ref2VAStreamEngine')
    cfg.inference = dict(spec=spec, weights=str(Path(args.weights).resolve()),
                         checkpoint=str(Path(args.checkpoint).resolve()), output_dir=str(output),
                         decode_mode=args.decode_mode, seed=args.seed,
                         runtime=options,
                         conditioning_matches_training=options['codec']['encoder'] == 'h3',
                         condition_seed=args.condition_seed if args.condition_seed is not None else args.seed)
    cfg.lineage['source_training_config'] = str(Path(args.config).resolve())
    cfg.lineage['inference_weights'] = str(Path(args.weights).resolve())
    if args.steps is not None:
        if args.steps <= 0:
            raise ValueError('--steps must be positive')
        target = cfg.validation if stage == 'tscd' else cfg.diffusion
        for name in ('sampling_timesteps', 'audio_sampling_timesteps'):
            target[name].args.num_sampling_steps = args.steps
    if recorded:
        cfg.inference_recipe = RefRecipe.from_training(cfg).to_dict()
    return cfg


def main(argv=None):
    parser = inspection_parser('ref2va_infer')
    args = parse_run_arguments(parser, argv, entrypoint='ref2va_infer')
    from common.config import CfgNode
    training = CfgNode.from_file(args.config)
    recipe, recipe_path = load_recipe(training, args.checkpoint, args.prompt_file)
    spec = input_spec(args, recipe)
    if not Path(args.weights).is_file():
        parser.error('generator export is missing; train and export the selected stage first')
    output = Path(args.output_dir)
    if any((output/name).exists() for name in ('inference.json','effective-settings.json','video.mp4','video.partial.mp4','chunks')):
        parser.error('inference outputs already exist; choose a new --output-dir')
    config = inference_config(args, training, spec)
    from projects.minimax_h3.modeling.ref_recipe import RefRecipe
    config.inference.effective_settings = effective_settings(args, recipe=RefRecipe.from_training(config),
        options=dict(config.inference.runtime), implementation="local_training_config", training=training, request=spec)
    if args.check_only:
        print(json.dumps(dict(status='checked', recipe=str(recipe_path), spec=spec,
                              stage=config.data.args.stage, weights=args.weights, output_dir=args.output_dir,
                              runtime=dict(config.inference.runtime),
                              effective_settings=dict(config.inference.effective_settings),
                              conditioning_matches_training=config.inference.conditioning_matches_training),indent=2))
        return
    require_dit_graph_environment(config.inference.runtime['dit']['cuda_graph'])
    from common.distributed.init import configure_distributed, destroy_distributed
    from common.logging import configure_logging
    from common.persistence import configure_persistence, resolve_run_dirs
    config = configure_logging(resolve_run_dirs(config, Path(args.config)))
    config = configure_distributed(config)
    config = configure_persistence(config, Path(args.config), list(argv if argv is not None else sys.argv[1:]))
    cls = getattr(importlib.import_module(config.meta_model.module), config.meta_model.class_name)
    failed = True
    try:
        result = Ref2VAStreamEngine(config, cls(config)).run()
        failed = False
        return result
    finally:
        config.checkpointer.finalize()
        config.logger.close(exit_code=int(failed))
        if not failed:
            destroy_distributed()


if __name__ == '__main__':
    main()
