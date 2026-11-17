"""Run the CPU browser gateway without importing the training runtime."""
import argparse
import math
from pathlib import Path
import sys
import types


def bootstrap():
    root = Path(__file__).resolve().parents[1]
    for name, path in (('projects', root.parent), ('projects.minimax_h3', root)):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--backend', help='Existing Ref HTTP service URL')
    source.add_argument('--fixture', action='store_true', help='Synthetic CPU echo, no model inference')
    parser.add_argument('--artifact-id')
    parser.add_argument('--native-game-proxy-url', help='Native localhost preview URL; requires a loopback gateway')
    parser.add_argument('--host', choices=('127.0.0.1', 'localhost', '::1', '0.0.0.0'), default='127.0.0.1')
    parser.add_argument('--public-url', help='External HTTPS URL including proxy path and trailing slash')
    parser.add_argument('--access-key-file', type=Path, help='Private file containing a 32+ byte application access key')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--no-media-control', action='store_true', help='Use in-band media ACK/cancel for the baseline')
    parser.add_argument('--incremental-proxy', action='store_true', help='Upload individual proxy frames before chunk commit')
    parser.add_argument('--threejs-racing', action='store_true', help='Enable the locally built TORCS racing source')
    parser.add_argument('--ballpoint-breach', action='store_true', help='Enable the locally built Ballpoint Breach source')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('port and timeout must be positive and finite')
    if bool(args.public_url) != bool(args.access_key_file):
        parser.error('--public-url and --access-key-file must be configured together')
    if args.host == '0.0.0.0' and not args.public_url:
        parser.error('--host 0.0.0.0 requires --public-url and --access-key-file')
    bootstrap()
    if args.native_game_proxy_url:
        from projects.minimax_h3.inference.web_native_game import validate_proxy_url
        try:
            validate_proxy_url(args.native_game_proxy_url)
        except ValueError as error:
            parser.error(str(error))
        if args.public_url or args.host not in ('127.0.0.1', 'localhost', '::1'):
            parser.error('--native-game-proxy-url requires a loopback-only gateway')
    from aiohttp import web
    from projects.minimax_h3.inference.web_serving import create_app
    web.run_app(create_app(backend=args.backend, fixture=args.fixture,
        artifact_id=args.artifact_id, timeout=args.timeout, public_url=args.public_url,
        access_key_file=args.access_key_file, media_control=not args.no_media_control,
        incremental_proxy=args.incremental_proxy, native_game_proxy_url=args.native_game_proxy_url,
        threejs_racing=args.threejs_racing,
        ballpoint_breach=args.ballpoint_breach),
        host=args.host, port=args.port)


if __name__ == '__main__':
    main()
