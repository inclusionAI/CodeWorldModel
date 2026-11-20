"""Run the loopback CPU media adapter for an explicitly selected native bundle."""
import argparse
import math


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True, help='Prepared native bundle directory')
    parser.add_argument('--artifact', required=True, help='Source artifact pinned by the bundle')
    parser.add_argument('--native-uri', default='ws://127.0.0.1:18860/v1/session')
    parser.add_argument('--port', type=int, default=18861)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('port and timeout must be positive and finite')
    from tools.serve_web import bootstrap
    bootstrap()
    from aiohttp import web
    from projects.minimax_h3.inference.omni_h3_web import bundle_artifact, create_app
    artifact_id, artifact, codec = bundle_artifact(args.model_path, args.artifact)
    web.run_app(create_app(args.native_uri, artifact_id, artifact, codec=codec,
                          timeout=args.timeout), host='127.0.0.1', port=args.port,
                handler_cancellation=True)


if __name__ == '__main__':
    main()
