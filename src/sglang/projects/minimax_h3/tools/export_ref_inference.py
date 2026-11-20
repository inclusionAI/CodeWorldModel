"""Resolve a training config/cache once into an independent inference artifact."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True, help='Complete runtime safetensors export')
    parser.add_argument('--weights-format', choices=('raven_full_or_merged', 'raven_full_unmerged_lora'),
        default='raven_full_or_merged', help='Unmerged requires the frozen base and adapter in one complete file')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prompt-file', required=True,
        help='Frozen system prompt file matching the encoder recipe')
    parser.add_argument('--output', required=True, help='New artifact JSON; never overwrite')
    parser.add_argument('--recipe-override', type=Path, help='Complete RefRecipe JSON; experimental training compatibility is unverified')
    args = parser.parse_args(argv)
    import yaml
    from projects.minimax_h3.inference.artifact import export_bundle, sha256
    from projects.minimax_h3.tools.ref2va_infer import load_recipe
    training = yaml.safe_load(Path(args.config).read_text())
    encoder_recipe, recipe_path = load_recipe(training, args.checkpoint, args.prompt_file)
    bundle = export_bundle(training, args.weights, args.checkpoint, encoder_recipe,
        weights_format=args.weights_format,
        provenance=dict(source_stage=training['data']['args']['stage'],
                        training_config_sha256=sha256(args.config), encoder_recipe=str(recipe_path)),
        recipe_override=None if args.recipe_override is None else json.loads(args.recipe_override.read_text()),
        recipe_override_sha256=None if args.recipe_override is None else sha256(args.recipe_override))
    with Path(args.output).open('x') as handle:
        json.dump(bundle, handle, indent=2)
        handle.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
