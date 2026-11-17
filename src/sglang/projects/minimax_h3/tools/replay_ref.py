"""Replay independently arriving proxy frames and record actual RGB reception."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time
import uuid

from projects.minimax_h3.inference.serving import SessionError
from projects.minimax_h3.inference.serving_http import RefClient
from projects.minimax_h3.inference.dummy_streaming import CapturedFrame, DummySource, PromptSnapshot
from projects.minimax_h3.inference.streaming_text import MAX_TEXT_ROWS


class FrameProducer:
    def __init__(self, frames, count, *, start=0, pace=24, queue_size=2, start_ns=None):
        self.frames, self.count, self.start, self.pace = frames, count, start, pace
        self.start_ns = start_ns
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop = threading.Event()
        self.done = threading.Event()
        self.error = None
        self.max_queued = 0
        self.backpressure_ns = 0
        self.produced_frames = 0
        self.error_ns = None
        self.thread = threading.Thread(target=self._produce, daemon=True)

    def _produce(self):
        import numpy as np
        began = self.start_ns if self.start_ns is not None else time.perf_counter_ns()
        pixels, captures, lateness = [], [], []
        try:
            for index in range(self.count):
                if self.stop.is_set():
                    return
                scheduled = began + round(index * 1e9 / self.pace) if self.pace else None
                if scheduled is not None:
                    while (remaining := scheduled-time.perf_counter_ns()) > 0:
                        if self.stop.wait(remaining/1e9):
                            return
                frame = next(self.frames)
                if self.stop.is_set():
                    return
                captured = frame.capture_ns if isinstance(frame, CapturedFrame) else time.perf_counter_ns()
                if type(captured) is not int or captured < 0:
                    raise ValueError('capture_ns must be a nonnegative integer on the client monotonic clock')
                self.produced_frames += 1
                pixels.append(frame.pixels if isinstance(frame, CapturedFrame) else frame)
                captures.append(captured)
                lateness.append(None if scheduled is None else captured-scheduled)
                eos = index == self.count-1
                if len(pixels) == 17 or eos:
                    first = index-len(pixels)+1
                    item = dict(chunk_id=first//17, first_source_frame=self.start+first,
                        capture_ns=captures, eos=eos,
                        pixels=np.stack(pixels).tobytes(), schedule_lateness_ns=lateness)
                    item['assembled_ns'] = time.perf_counter_ns()
                    before = time.perf_counter_ns()
                    if self.pace:
                        try:
                            self.queue.put_nowait(item)
                        except queue.Full as exc:
                            raise RuntimeError('paced producer overflow; no source frames were silently dropped') from exc
                    else:
                        while not self.stop.is_set():
                            try:
                                self.queue.put(item, timeout=.1)
                                break
                            except queue.Full:
                                pass
                    self.backpressure_ns += time.perf_counter_ns()-before
                    self.max_queued = max(self.max_queued, self.queue.qsize())
                    pixels, captures, lateness = [], [], []
        except StopIteration:
            if not self.stop.is_set():
                self.error = ValueError('proxy source ended before the requested frame count')
        except BaseException as exc:
            self.error = exc
        finally:
            if self.error is not None:
                self.error_ns = time.perf_counter_ns()
            try:
                close = getattr(self.frames, 'close', None)
                if close is not None:
                    close()
            except BaseException as exc:
                if self.error is None:
                    self.error = exc
                    self.error_ns = time.perf_counter_ns()
            finally:
                self.done.set()

    def __iter__(self):
        self.thread.start()
        while True:
            if self.error is not None:
                raise self.error
            try:
                yield self.queue.get(timeout=.1)
            except queue.Empty:
                if self.done.is_set():
                    if self.error is not None:
                        raise self.error
                    return

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('source reader did not stop within five seconds')
def run_session(client, producer, *, first, start, spec, output, pace,
                save_rgb=True, mp4=False, source=None, prompt_snapshot=None):
    output = Path(output)
    sid = start['session_id']
    report = dict(status='failed', session_id=sid, artifact_id=start['artifact_id'],
        pace=pace, spec=spec, start_request=start, frame_count=0, received_rgb_bytes=0,
        presentation='not_measured', clock='client_perf_counter_ns', save_rgb=save_rgb,
        output_processing_ns=0)
    active = False
    writer = None
    digest = hashlib.sha256()
    pending = {}
    submitted = 0
    previous_prompt = previous_applied = None
    failure = None
    try:
        with (output/'requests.jsonl').open('x') as requests, (output/'frames.jsonl').open('x') as frame_log, \
                ((output/'rgb.raw').open('xb') if save_rgb else nullcontext()) as rgb_file:
            active = True
            request_start = time.perf_counter_ns()
            report['session_request_ns'] = request_start
            try:
                reply, timing = client.call('/session', start, first.tobytes())
                if (reply.metadata.get('session_id') != sid or reply.metadata.get('status') != 'accepted'
                        or reply.metadata.get('artifact_id') != start['artifact_id'] or reply.pixels):
                    raise SessionError('invalid session acknowledgement', 502)
                report['effective_recipe'] = reply.metadata['recipe']
            except BaseException as exc:
                requests.write(json.dumps(dict(operation='start', request_start_ns=request_start,
                    error_ns=time.perf_counter_ns(), error=str(exc)))+'\n')
                requests.flush()
                raise
            requests.write(json.dumps(dict(operation='start', input=start, client=timing, server=reply.metadata))+'\n')
            requests.flush()
            report['session_init_ns'] = timing['receive_ns']-request_start
            report['stream_start_ns'] = time.perf_counter_ns()
            producer.start_ns = report['stream_start_ns']
            if source is not None:
                source.start(report['stream_start_ns'])
            for produced in producer:
                item = dict(produced)
                payload = item.pop('pixels')
                lateness = item.pop('schedule_lateness_ns')
                frozen_ns = None
                if prompt_snapshot is not None:
                    snapshot = prompt_snapshot()
                    frozen_ns = time.perf_counter_ns()
                    if not isinstance(snapshot, PromptSnapshot):
                        raise ValueError('prompt_snapshot must return an immutable PromptSnapshot')
                    if previous_prompt is not None and (snapshot.revision < previous_prompt.revision or
                            snapshot.revision == previous_prompt.revision and snapshot != previous_prompt):
                        raise ValueError('prompt revisions must increase or retain the identical snapshot')
                    item['prompt'] = asdict(snapshot)
                    previous_prompt = snapshot
                for offset, capture in enumerate(item['capture_ns']):
                    pending[item['first_source_frame']+offset] = dict(chunk_id=item['chunk_id'],
                        source_frame=item['first_source_frame']+offset, capture_ns=capture)
                    if prompt_snapshot is not None:
                        pending[item['first_source_frame']+offset]['prompt_revision'] = snapshot.revision
                submitted += len(item['capture_ns'])
                request_start = time.perf_counter_ns()
                try:
                    reply, timing = client.call(f'/session/{sid}/chunk', dict(session_id=sid, **item), payload)
                    m = reply.metadata
                    count = m['frame_count']
                    if (any(type(m[key]) is not int for key in ('frame_count', 'chunk_id', 'output_id',
                            'available_after_chunk', 'height', 'width', 'output_frame_start', 'total_frames', 'accepted_frames'))
                            or count < 0 or m['session_id'] != sid
                            or m['chunk_id'] != item['chunk_id'] or m['output_id'] != item['chunk_id']
                            or m['available_after_chunk'] != item['chunk_id']
                            or m['height'] != spec['height'] or m['width'] != spec['width']
                            or m['output_frame_start'] != report['frame_count']
                            or m['total_frames'] != report['frame_count']+count
                            or m['accepted_frames'] != submitted
                            or len(m['source_frames']) != count
                            or len(reply.pixels) != count*spec['height']*spec['width']*3
                            or m['status'] != ('complete' if item['eos'] else 'accepted')):
                        raise SessionError('invalid returned RGB sequence or geometry', 502)
                    if prompt_snapshot is not None:
                        submitted_prompt = m.get('submitted_prompt')
                        applied = m.get('submitted_prompt_encoding', m.get('applied_prompt'))
                        same_text = previous_applied is not None and previous_applied['text'] == snapshot.text
                        expected_first_chunk = previous_applied['first_chunk'] if same_text else item['chunk_id']
                        if (not isinstance(submitted_prompt, dict) or submitted_prompt != item['prompt']
                                or any(type(submitted_prompt[key]) is not type(value) for key, value in item['prompt'].items())
                                or not isinstance(applied, dict) or set(applied) != {'text', 'sha256', 'first_chunk', 'rows'}
                                or applied['text'] != snapshot.text
                                or applied['sha256'] != hashlib.sha256(snapshot.text.encode('utf-8')).hexdigest()
                                or type(applied['rows']) is not int or not 1 <= applied['rows'] <= MAX_TEXT_ROWS
                                or type(applied['first_chunk']) is not int
                                or applied['first_chunk'] != expected_first_chunk
                                or same_text and applied != previous_applied):
                            raise SessionError('invalid returned prompt identity', 502)
                        for association in pending.values():
                            if association['chunk_id'] == item['chunk_id']:
                                association.update(prompt_sha256=applied['sha256'],
                                    prompt_first_chunk=applied['first_chunk'], prompt_rows=applied['rows'])
                    rows = []
                    for offset, association in enumerate(m['source_frames']):
                        expected = spec['start']+report['frame_count']+offset
                        if (not isinstance(association, dict) or association != pending.get(expected)
                                or any(type(association[key]) is not type(value) for key, value in pending[expected].items())):
                            raise SessionError('returned source association is out of order or altered', 502)
                        rows.append(dict(output_frame=report['frame_count']+offset, **association,
                            receive_ns=timing['receive_ns'],
                            capture_to_receive_ns=timing['receive_ns']-association['capture_ns']))
                    if item['eos'] and report['frame_count']+count != producer.count:
                        raise SessionError('EOS did not complete the requested stream', 502)
                    if prompt_snapshot is not None:
                        previous_applied = dict(applied)
                except BaseException as exc:
                    requests.write(json.dumps(dict(operation='chunk', chunk_id=item['chunk_id'], input=item,
                        prompt_frozen_ns=frozen_ns, request_start_ns=request_start,
                        error_ns=time.perf_counter_ns(), error=str(exc)))+'\n')
                    requests.flush()
                    raise
                processing_start = time.perf_counter_ns()
                requests.write(json.dumps(dict(operation='chunk', chunk_id=item['chunk_id'],
                    input=item, client=timing, schedule_lateness_ns=lateness, server=m,
                    prompt_frozen_ns=frozen_ns,
                    received_rgb_bytes=len(reply.pixels), rgb_sha256=hashlib.sha256(reply.pixels).hexdigest(),
                    queued_chunks=producer.queue.qsize(), produced_frames=producer.produced_frames,
                    submitted_frames=submitted, unreturned_submitted_frames=submitted-report['frame_count']-count))+'\n')
                requests.flush()
                for row in rows:
                    frame_log.write(json.dumps(row)+'\n')
                    pending.pop(row['source_frame'])
                frame_log.flush()
                digest.update(reply.pixels)
                report['frame_count'] += count
                report['received_rgb_bytes'] += len(reply.pixels)
                if item['eos']:
                    active = False
                if rgb_file is not None:
                    rgb_file.write(reply.pixels)
                if mp4 and count:
                    import numpy as np
                    import torch
                    from projects.minimax_h3.inference.ref2va import VideoWriter
                    if writer is None:
                        writer = VideoWriter(output, spec['height'], spec['width'])
                    writer.write(torch.from_numpy(np.frombuffer(reply.pixels, np.uint8).copy().reshape(
                        count, spec['height'], spec['width'], 3)))
                report['output_processing_ns'] += time.perf_counter_ns()-processing_start
            if active or report['frame_count'] != producer.count:
                raise SessionError('source ended without successful EOS', 502)
            report['status'] = 'complete'
    except BaseException as exc:
        failure = exc
        report['error'] = str(exc)
        raise
    finally:
        report['stream_end_ns'] = time.perf_counter_ns()
        producer.stop.set()
        cleanup_errors = []
        # Acquisition owners must wake blocked reads before the producer is joined.
        for cleanup in (source.close if source is not None else None, producer.close):
            if cleanup is not None:
                try:
                    cleanup()
                except BaseException as exc:
                    cleanup_errors.append(exc)
        if active:
            try:
                client.call(f'/session/{sid}/cancel', dict(session_id=sid, reason='replay failed'))
            except Exception as exc:
                report['cancel_error'] = str(exc)
        if writer is not None:
            try:
                writer.close(success=report['status']=='complete')
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            report.update(status='failed', cleanup_error='; '.join(str(exc) for exc in cleanup_errors))
        report['cleanup_ns'] = time.perf_counter_ns()-report['stream_end_ns']
        report.update(rgb_sha256=digest.hexdigest(), max_queued_chunks=producer.max_queued,
            source_backpressure_ns=producer.backpressure_ns, produced_frames=producer.produced_frames,
            submitted_frames=submitted, producer_error=str(producer.error) if producer.error else None,
            producer_error_ns=producer.error_ns, queued_chunks_at_stop=producer.queue.qsize())
        (output/'report.json').write_text(json.dumps(report, indent=2))
        if isinstance(source, DummySource):
            from projects.minimax_h3.inference.dummy_streaming import summarize_dummy_run
            try:
                summarize_dummy_run(output)
            except BaseException:
                if failure is None and not cleanup_errors:
                    raise
        if cleanup_errors and failure is None:
            raise cleanup_errors[0]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--artifact-id', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bundle')
    source.add_argument('--proxy-video')
    parser.add_argument('--first-frame')
    parser.add_argument('--start-frame', type=int)
    parser.add_argument('--window-index', type=int)
    parser.add_argument('--num-frames', type=int, default=158)
    from projects.minimax_h3.inference.raw_recipe import ALL_STREAMING_TEXT, window_arguments, window_overrides
    window_arguments(parser)
    parser.add_argument('--prompts-jsonl', help='Complete prompt changes keyed by source clip')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--condition-seed', type=int)
    parser.add_argument('--pace', choices=('24', 'unpaced'), default='24')
    parser.add_argument('--queue-size', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mp4', action='store_true')
    args = parser.parse_args(argv)
    if args.queue_size < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('queue-size and timeout must be positive')
    from projects.minimax_h3.inference.ref2va import input_spec
    from projects.minimax_h3.tools.ref2va_prepare import iter_video_chunks
    import numpy as np
    from PIL import Image
    client = RefClient(args.url, timeout=args.timeout)
    health, _ = client.call('/health', method='GET')
    if args.artifact_id not in health.metadata['artifacts']:
        parser.error('artifact ID is not offered by the service')
    artifact = health.metadata['artifacts'][args.artifact_id]
    overrides = window_overrides(args, artifact['recipe'])
    streaming = artifact['recipe']['protocol'] in ALL_STREAMING_TEXT
    if streaming != bool(args.prompts_jsonl):
        parser.error('streaming artifacts require --prompts-jsonl; static artifacts reject it')
    spec = input_spec(args, artifact['encoder_recipe'])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if 'first_frame' in spec:
        with Image.open(spec['first_frame']) as image:
            first = np.asarray(image.convert('RGB').resize((spec['width'], spec['height']), Image.Resampling.LANCZOS))
    else:
        first = next(iter_video_chunks(spec['rgb'], start=spec['start'], num_frames=1,
            height=spec['height'], width=spec['width'], chunk_frames=1))[0]
    sid = uuid.uuid4().hex
    start = dict(session_id=sid, artifact_id=args.artifact_id,
        seed=args.seed, condition_seed=args.seed if args.condition_seed is None else args.condition_seed,
        prompt=spec['prompt'], **overrides)
    frames = (batch[0] for batch in iter_video_chunks(spec['proxy'], start=spec['start'],
        num_frames=spec['num_frames'], height=spec['proxy_height'], width=spec['proxy_width'],
        nearest=True, chunk_frames=1))
    producer = FrameProducer(frames, args.num_frames, start=spec['start'],
        pace=24 if args.pace == '24' else None, queue_size=args.queue_size)
    snapshot = None
    if streaming:
        from projects.minimax_h3.inference.streaming_text import ReplayPromptSource
        prompts = ReplayPromptSource(args.prompts_jsonl, (args.num_frames+16)//17)
        clip, revision, previous = 0, 0, None
        def snapshot():
            nonlocal clip, revision, previous
            text = prompts(clip)
            clip += 1
            if text is not None and (previous is None or text != previous.text):
                revision += 1
                previous = PromptSnapshot(revision, text, time.perf_counter_ns())
            return previous
    report = run_session(client, producer, first=first, start=start, spec=spec,
        output=output, pace=args.pace, save_rgb=True, mp4=args.mp4, prompt_snapshot=snapshot)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    main()
