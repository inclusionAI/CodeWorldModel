"""Resolve a unified run YAML or compare resolved settings without torch or GPUs."""
import argparse
import json
from pathlib import Path
import sys

import yaml

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from inference.settings import RUN_SCHEMA, diff_settings, inspect_config, read_settings
else:
    from ..inference.settings import RUN_SCHEMA, diff_settings, inspect_config, read_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    resolve = commands.add_parser('resolve')
    resolve.add_argument('config')
    resolve.add_argument('-o', '--output')
    diff = commands.add_parser('diff')
    diff.add_argument('left')
    diff.add_argument('right')
    diff.add_argument('--fail-on-difference', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'resolve':
            value = inspect_config(args.config)
        else:
            def resolved(path):
                record = read_settings(path)
                return inspect_config(path) if record.get('schema') == RUN_SCHEMA else record
            value = diff_settings(resolved(args.left), resolved(args.right))
        output = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
        if args.command == 'resolve' and args.output:
            Path(args.output).write_text(output, encoding='utf-8')
        else:
            print(output, end='')
        return int(args.command == 'diff' and args.fail_on_difference and bool(value))
    except (ValueError, OSError, KeyError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
