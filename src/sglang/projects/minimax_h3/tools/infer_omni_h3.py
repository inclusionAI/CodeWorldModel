"""Run invoker-render file inference through the Omni /v1/session endpoint."""
import argparse
import asyncio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True, help='Prepared native bundle directory')
    parser.add_argument('--artifact', required=True, help='Original source artifact pinned by the bundle')
    parser.add_argument('--uri', required=True, help='Omni ws:// or wss:// /v1/session endpoint')
    parser.add_argument('--first-image', required=True, help='RGB image matching the bundle output canvas')
    parser.add_argument('--proxy-video', required=True,
        help='Exact 24 FPS proxy canvas video, 17*N+5 frames, minimum 39; no resampling or truncation')
    parser.add_argument('--prompt-file', required=True, help='Static nonblank UTF-8 text, at most 16 KiB')
    parser.add_argument('--output-dir', required=True, help='New directory for video.mp4 and inference.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=float,
        help='Seconds allowed per session step (default 900, enough for first-run decoder compilation)')
    args = parser.parse_args()
    from projects.minimax_h3.inference.omni_h3_client import STEP_TIMEOUT, infer_files
    if args.timeout is None:
        args.timeout = STEP_TIMEOUT
    receipt = asyncio.run(infer_files(**vars(args)))
    print(f'{args.output_dir}: {receipt["output_frames"]} frames, {receipt["status"]}')


if __name__ == '__main__':
    main()
