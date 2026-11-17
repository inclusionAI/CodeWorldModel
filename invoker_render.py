#!/usr/bin/env python3
"""Invoker-Render command line.

Run with the serving environment's interpreter:

    python invoker_render.py <command> [options]

Each command forwards its options to the matching tool in
src/sglang/projects/minimax_h3/tools/; add --help after a command to list them.
"""
import runpy
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent / 'src/sglang/projects/minimax_h3/tools'
COMMANDS = {
    'prepare': ('prepare_omni_h3', 'verify model assets and write a deployment bundle'),
    'serve': ('serve_omni_h3', 'start the streaming session service'),
    'infer': ('infer_omni_h3', 'render a video from a first image, a proxy video, and a prompt'),
    'web-adapter': ('serve_omni_h3_web', 'start the CPU media adapter for the browser client'),
    'web': ('serve_web', 'start the browser gateway'),
}
CHECK_ENV = 'check_omni_installed_requirements.py'


def usage():
    rows = [*((name, text) for name, (_, text) in COMMANDS.items()),
            ('check-env', 'compare installed packages with the declared requirements')]
    return __doc__ + '\ncommands:\n' + ''.join(f'  {name:<12} {text}\n' for name, text in rows)


def main(argv):
    if not argv or argv[0] in ('-h', '--help'):
        print(usage())
        return 0
    command, args = argv[0], argv[1:]
    if command == 'check-env':
        sys.argv = [str(TOOLS / CHECK_ENV), *args]
        runpy.run_path(sys.argv[0], run_name='__main__')
        return 0
    if command not in COMMANDS:
        print(f'unknown command: {command}\n\n{usage()}', file=sys.stderr)
        return 2
    launch = runpy.run_path(str(TOOLS / 'launch.py'))
    launch['main']([f'projects.minimax_h3.tools.{COMMANDS[command][0]}', *args])
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
