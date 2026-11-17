"""Infra-only dummy seek/play inputs; report source-associated RGB receipt, not game response."""
import argparse
import json
from pathlib import Path
import time
import uuid

from projects.minimax_h3.inference.dummy_streaming import DummySource, InputEvent, PromptEvent, MAX_CACHE_BYTES, MAX_CLIP_FRAMES
from projects.minimax_h3.inference.serving_http import RefClient
from projects.minimax_h3.tools.replay_ref import FrameProducer, run_session
from projects.minimax_h3.inference.raw_recipe import ALL_STREAMING_TEXT, window_arguments, window_overrides


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--artifact-id', required=True)
    parser.add_argument('--proxy-video', required=True)
    parser.add_argument('--first-frame', required=True)
    parser.add_argument('--num-frames', type=int, default=158)
    window_arguments(parser)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--condition-seed', type=int)
    parser.add_argument('--pace', choices=('24', 'unpaced'), default='24')
    parser.add_argument('--queue-size', type=int, default=2)
    parser.add_argument('--events', required=True, help='JSON file containing ordered event_id/offset_ms/seek_frame/playing objects')
    parser.add_argument('--prompt-events', help='JSON file containing ordered revision/offset_ms/text objects; first offset must be zero')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--save-rgb', action='store_true', help='Save received rgb.raw; bytes are validated and hashed either way')
    args = parser.parse_args(argv)
    if args.queue_size < 1:
        parser.error('queue-size must be positive')
    client = RefClient(args.url)
    health, _ = client.call('/health', method='GET')
    if args.artifact_id not in health.metadata['artifacts']:
        parser.error('artifact ID is not offered by the service')
    artifact = health.metadata['artifacts'][args.artifact_id]
    streaming = artifact['recipe']['protocol'] in ALL_STREAMING_TEXT
    overrides = window_overrides(args, artifact['recipe'])
    if streaming != (args.prompt_events is not None):
        parser.error('streaming artifacts require --prompt-events; static artifacts reject it')
    prompt_events = []
    if streaming:
        prompt_data = json.loads(Path(args.prompt_events).read_text())
        if not isinstance(prompt_data, list) or not prompt_data:
            parser.error('prompt-events must be a nonempty JSON array')
        prompt_events = [PromptEvent(**row) for row in prompt_data]
    from projects.minimax_h3.inference.ref2va import input_spec
    from projects.minimax_h3.tools.ref2va_prepare import iter_video_chunks
    import numpy as np
    from PIL import Image
    args.bundle, args.start_frame, args.window_index = None, None, None
    spec = input_spec(args, artifact['encoder_recipe'])
    clip_count = min(args.num_frames, MAX_CLIP_FRAMES,
        MAX_CACHE_BYTES//(spec['proxy_height']*spec['proxy_width']*3))
    if clip_count < 1:
        parser.error('one proxy frame exceeds the 256 MiB cache bound')
    event_data = json.loads(Path(args.events).read_text())
    if not isinstance(event_data, list):
        parser.error('events must be a JSON array')
    events = [InputEvent(**row) for row in event_data]
    preparation_start = time.perf_counter_ns()
    clip = [batch[0] for batch in iter_video_chunks(spec['proxy'], start=0, num_frames=clip_count,
        height=spec['proxy_height'], width=spec['proxy_width'], nearest=True, chunk_frames=1)]
    with Image.open(spec['first_frame']) as image:
        first = np.asarray(image.convert('RGB').resize((spec['width'], spec['height']), Image.Resampling.LANCZOS))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = DummySource(clip, events, output,
        preparation_ns=time.perf_counter_ns()-preparation_start, prompt_events=prompt_events)
    start = dict(session_id=uuid.uuid4().hex, artifact_id=args.artifact_id,
        seed=args.seed, condition_seed=args.seed if args.condition_seed is None else args.condition_seed,
        prompt=spec['prompt'], **overrides)
    producer = FrameProducer(source.frames(), args.num_frames,
        pace=24 if args.pace=='24' else None, queue_size=args.queue_size)
    report = run_session(client, producer, first=first, start=start, spec=spec, output=output,
        pace=args.pace, save_rgb=args.save_rgb, source=source,
        prompt_snapshot=source.prompt_snapshot if streaming else None)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    main()
