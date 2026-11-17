"""Stream a proxy using an exported Ref recipe, without training configuration."""
from copy import deepcopy
import json
from pathlib import Path
import sys

from projects.minimax_h3.inference.artifact import load_bundle, runtime_backbone, sha256
from projects.minimax_h3.inference.ref2va import Ref2VAStreamEngine, input_spec
from projects.minimax_h3.inference.settings import parse_run_arguments, options_for_args, effective_settings, inspection_parser, require_dit_graph_environment
from projects.minimax_h3.modeling.ref_recipe import (BIDIR_RECIPE, FIXED_RECIPE,
    FIXED_STREAMING_TEXT_RECIPE, BIDIR_STREAMING_TEXT_RECIPE, STREAMING_TEXT_RECIPES)
from projects.minimax_h3.inference.raw_recipe import (
    RAW_PROTOCOLS, ALL_STREAMING_TEXT, RawLatentRecipe, window_overrides, validate_online_guidance)


def runtime_config(args, bundle, recipe, spec):
    from common.config import CfgNode
    validate_online_guidance(recipe)
    options = options_for_args(args)
    if options['codec']['decoder'] == 'causal' and args.decode_mode != 'strict':
        raise ValueError('causal decoder requires --decode-mode strict')
    if isinstance(recipe, RawLatentRecipe) and options['codec']['decoder'] == 'h3' and args.decode_mode == 'strict':
        raise ValueError('raw-latent strict output requires codec.decoder: causal')
    backbone = runtime_backbone(bundle)
    backbone.update(runtime=dict(training=False,requires_grad=False,param_dtype='bfloat16'),
        placement=deepcopy(bundle['deployment']['placement']))
    if options['resident_weights']:
        backbone['placement'] = dict(module='projects.minimax_h3.modeling.inference_runtime', class_name='ResidentInferencePlacement')
    output = Path(args.output_dir).resolve()
    return CfgNode(dict(models=dict(backbone=backbone), distributed=deepcopy(bundle['deployment']['distributed']),
        engine=dict(resume='never',up_size=dict(train=bundle['deployment']['active_up_size'])),seed=args.seed,logging={},
        persistence=dict(output_dir=str(output.parent),proj_name=output.name,exp_name='runtime',
                         checkpoint={'async':False,'keep_latest_k':2}),
        inference=dict(spec=spec,weights=bundle['weights']['path'],checkpoint=bundle['encoders']['checkpoint'],
            output_dir=str(output),decode_mode=args.decode_mode,seed=args.seed,
            condition_seed=args.seed if args.condition_seed is None else args.condition_seed,
            runtime=options,conditioning_matches_training=(options['codec']['encoder']=='h3'
                and recipe.protocol not in (*ALL_STREAMING_TEXT, *RAW_PROTOCOLS)),
            artifact_sha256=sha256(args.artifact), recipe=recipe.to_dict(),
            recipe_provenance=deepcopy(bundle.get('provenance', {})),
            sampling_steps_overridden=args.steps is not None)))


def _main(argv, *, expected_protocol):
    entrypoint = 'infer_ref_raw_latent' if expected_protocol is None else {
        FIXED_RECIPE: 'infer_ref_fixed_history', BIDIR_RECIPE: 'infer_ref_bidir_history',
        FIXED_STREAMING_TEXT_RECIPE: 'infer_ref_fixed_history_streaming_text',
        BIDIR_STREAMING_TEXT_RECIPE: 'infer_ref_bidir_history_streaming_text'}[expected_protocol]
    parser = inspection_parser(entrypoint)
    args = parse_run_arguments(parser, argv, entrypoint=entrypoint)
    bundle, recipe = load_bundle(args.artifact)
    if expected_protocol is None:
        if not isinstance(recipe, RawLatentRecipe):
            parser.error('raw-latent entry requires a v2 artifact')
        from dataclasses import replace
        recipe = replace(recipe, **window_overrides(args, recipe.to_dict()))
    elif recipe.protocol != expected_protocol:
        parser.error(f'this entry requires {expected_protocol}, artifact selects {recipe.protocol}')
    streaming = recipe.protocol in ALL_STREAMING_TEXT
    if expected_protocol is None and streaming != bool(args.prompt_updates or args.prompts_jsonl):
        parser.error('streaming-text artifacts require a prompt source; fixed-text artifacts reject it')
    if args.steps is not None:
        recipe = recipe.with_steps(args.steps)
    encoder_recipe = bundle['encoders']['recipe']
    if Path(args.prompt_file).read_text().strip() != encoder_recipe['system_prompt']:
        parser.error('fixed prompt differs from the exported encoder recipe')
    spec = input_spec(args, encoder_recipe)
    output = Path(args.output_dir)
    existing_outputs = ('inference.json', 'effective-settings.json', 'video.mp4', 'video.partial.mp4', 'chunks')
    if streaming:
        existing_outputs += ('prompt-events.jsonl', 'prompt-ledger.jsonl')
    if any((output/name).exists() for name in existing_outputs):
        parser.error('inference outputs already exist; choose a new --output-dir')
    config = runtime_config(args,bundle,recipe,spec)
    config.inference.effective_settings = effective_settings(args, bundle=bundle, recipe=recipe,
        options=dict(config.inference.runtime), implementation="local_artifact", request=spec)
    if streaming:
        from projects.minimax_h3.inference.streaming_text import prompt_source
        source = dict(kind='live_append_jsonl' if args.prompt_updates else 'replay_jsonl',
                      path=str(Path(args.prompt_updates or args.prompts_jsonl).resolve()))
        config.inference.prompt_source = source
        config.inference.update(effective_protocol=recipe.protocol,
            base_conditioning_mode=recipe.conditioning_mode,
            prompt_extension='independent_complete_text_per_chunk',
            inference_packing='raw_latent' if isinstance(recipe, RawLatentRecipe) else 'legacy')
        if args.check_only:
            checked_source = prompt_source(source, (spec['num_frames']+16)//17)
            checked_source(0)
            config.inference.prompt_source = checked_source.provenance
    if args.check_only:
        print(json.dumps(dict(status='checked',**dict(config.inference)),indent=2))
        return
    from projects.minimax_h3.inference.recipes.ref_fixed_history import RefFixedHistory
    from projects.minimax_h3.inference.recipes.ref_bidir_history import RefBidirHistory
    from projects.minimax_h3.inference.recipes.ref_streaming_text import (
        RefFixedHistoryStreamingText, RefBidirHistoryStreamingText)
    from projects.minimax_h3.inference.raw_latent import RawLatentDriver
    runner = RawLatentDriver if isinstance(recipe, RawLatentRecipe) else {FIXED_RECIPE: RefFixedHistory, BIDIR_RECIPE: RefBidirHistory,
              FIXED_STREAMING_TEXT_RECIPE: RefFixedHistoryStreamingText,
              BIDIR_STREAMING_TEXT_RECIPE: RefBidirHistoryStreamingText}[expected_protocol]
    require_dit_graph_environment(config.inference.runtime['dit']['cuda_graph'])
    from common.distributed.init import configure_distributed, destroy_distributed
    from common.logging import configure_logging
    from common.persistence import configure_persistence, resolve_run_dirs
    config = configure_logging(resolve_run_dirs(config,Path(args.artifact)))
    config = configure_distributed(config)
    config = configure_persistence(config,Path(args.artifact),list(argv if argv is not None else sys.argv[1:]))
    failed = True
    try:
        result = Ref2VAStreamEngine(config,runner(recipe)).run()
        failed = False
        return result
    finally:
        config.checkpointer.finalize()
        config.logger.close(exit_code=int(failed))
        if not failed:
            destroy_distributed()


def main(argv=None):
    return _main(argv, expected_protocol=FIXED_RECIPE)


if __name__ == '__main__':
    main()
