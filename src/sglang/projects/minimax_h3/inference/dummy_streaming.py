from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import threading
import time

from .streaming_text import validate_prompt

MAX_CACHE_BYTES = 256 << 20
MAX_CLIP_FRAMES = 158


@dataclass(frozen=True)
class CapturedFrame:
    pixels: object
    capture_ns: int


@dataclass(frozen=True)
class PromptSnapshot:
    revision: int
    text: str
    emitted_ns: int

    def __post_init__(self):
        if type(self.revision) is not int or not 1 <= self.revision < 2**63:
            raise ValueError('prompt revision must be an integer in 1..2**63-1')
        validate_prompt(self.text)
        if type(self.emitted_ns) is not int or self.emitted_ns < 0:
            raise ValueError('prompt emitted_ns must be a nonnegative integer')


@dataclass(frozen=True)
class PromptEvent:
    revision: int
    offset_ms: float
    text: str

    def __post_init__(self):
        PromptSnapshot(self.revision, self.text, 0)
        if (type(self.offset_ms) not in (int, float) or not math.isfinite(self.offset_ms)
                or self.offset_ms < 0):
            raise ValueError('prompt offset_ms must be finite and nonnegative')


@dataclass(frozen=True)
class InputEvent:
    event_id: str
    offset_ms: float
    seek_frame: int
    playing: bool

    def __post_init__(self):
        if not isinstance(self.event_id, str) or not self.event_id or len(self.event_id) > 128:
            raise ValueError('event_id must be a nonempty string of at most 128 characters')
        if (type(self.offset_ms) not in (int, float) or not math.isfinite(self.offset_ms)
                or self.offset_ms < 0):
            raise ValueError('offset_ms must be finite and nonnegative')
        if type(self.seek_frame) is not int or self.seek_frame < 0 or type(self.playing) is not bool:
            raise ValueError('seek_frame must be a nonnegative integer and playing must be boolean')


class DummySource:
    def __init__(self, clip, events, output, *, preparation_ns=None, prompt_events=()):
        import numpy as np
        if (not clip or len(clip) > MAX_CLIP_FRAMES
                or sum(frame.nbytes for frame in clip) > MAX_CACHE_BYTES):
            raise ValueError('clip must contain 1..158 frames and fit the 256 MiB cache bound')
        if any(frame.dtype != np.uint8 or frame.shape != clip[0].shape for frame in clip):
            raise ValueError('clip must contain equally shaped uint8 frames')
        if len(clip[0].shape) != 3 or clip[0].shape[-1] != 3:
            raise ValueError('clip frames must be HWC RGB')
        self.clip, self.events, self.output = clip, tuple(events), Path(output)
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError('event IDs must be unique')
        if any(event.seek_frame >= len(clip) for event in self.events):
            raise ValueError('event seek_frame is outside the cached clip')
        if any(a.offset_ms > b.offset_ms for a, b in zip(self.events, self.events[1:])):
            raise ValueError('events must be ordered by offset_ms')
        self.prompt_events = tuple(prompt_events)
        if self.prompt_events and (self.prompt_events[0].offset_ms != 0 or any(
                a.offset_ms > b.offset_ms or a.revision >= b.revision
                for a, b in zip(self.prompt_events, self.prompt_events[1:]))):
            raise ValueError('prompt events must start at offset zero with ordered offsets and increasing revisions')
        self.lock = threading.Lock()
        self.proxy_lock = threading.Lock()
        self.stop = threading.Event()
        self.initial_inputs_ready = threading.Event()
        self.thread = threading.Thread(target=self._schedule, name='dummy-inputs', daemon=True)
        self.error = None
        self.epoch = None
        self.cursor = self.source_frame = 0
        self.playing = True
        self.current_event = None
        self.current_prompt = None
        self.proxy_log = None
        (self.output/'input_events.json').write_text(json.dumps([asdict(event) for event in self.events], indent=2))
        (self.output/'prompt_events.json').write_text(json.dumps([asdict(event) for event in self.prompt_events], indent=2))
        (self.output/'dummy_source.json').write_text(json.dumps(dict(workload='dummy_infra',
            clip_frames=len(clip), cache_bytes=sum(frame.nbytes for frame in clip),
            video_and_first_image_preparation_ns=preparation_ns,
            cache_limit_bytes=MAX_CACHE_BYTES, clip_frame_limit=MAX_CLIP_FRAMES,
            cursor_repeat='wrap_at_clip_end', initial_seek_frame=0, initial_playing=True,
            association='first consumed proxy source frame, not semantic action response'), indent=2))

    def start(self, epoch):
        self.epoch = epoch
        self.proxy_log = (self.output/'proxy_frames.jsonl').open('x')
        self.thread.start()
        self.initial_inputs_ready.wait()
        if self.error is not None:
            raise self.error

    def _schedule(self):
        try:
            with (self.output/'events.jsonl').open('x') as log, (self.output/'prompt_events.jsonl').open('x') as prompt_log:
                scheduled = sorted([*self.events, *self.prompt_events], key=lambda event: event.offset_ms)
                for event in scheduled:
                    planned = self.epoch + round(event.offset_ms * 1e6)
                    if event.offset_ms > 0:
                        self.initial_inputs_ready.set()
                    while not self.stop.is_set() and (remaining := planned-time.perf_counter_ns()) > 0:
                        self.stop.wait(remaining/1e9)
                    if self.stop.is_set():
                        return
                    with self.lock:
                        emitted = time.perf_counter_ns()
                        if isinstance(event, PromptEvent):
                            self.current_prompt = PromptSnapshot(event.revision, event.text, emitted)
                            target = prompt_log
                        else:
                            self.cursor, self.playing, self.current_event = event.seek_frame, event.playing, event.event_id
                            target = log
                        target.write(json.dumps(dict(**asdict(event), planned_ns=planned,
                            emitted_ns=emitted, schedule_lateness_ns=emitted-planned))+'\n')
                        target.flush()
        except BaseException as exc:
            self.error = exc
        finally:
            self.initial_inputs_ready.set()

    def prompt_snapshot(self):
        with self.lock:
            if self.error is not None:
                raise self.error
            if self.current_prompt is None:
                raise ValueError('streaming prompt requires an offset-zero event')
            return self.current_prompt

    def frames(self):
        while not self.stop.is_set():
            if self.error is not None:
                raise self.error
            with self.lock:
                selected_ns = time.perf_counter_ns()
                cursor, event_id = self.cursor, self.current_event
                if self.playing:
                    self.cursor = (self.cursor+1) % len(self.clip)
            frame = self.clip[cursor].copy()
            completed = time.perf_counter_ns()
            with self.proxy_lock:
                if self.stop.is_set():
                    return
                self.proxy_log.write(json.dumps(dict(source_frame=self.source_frame,
                    clip_frame=cursor, event_id=event_id, input_consumed_ns=selected_ns,
                    proxy_ready_ns=completed, proxy_generation_ns=completed-selected_ns))+'\n')
                self.proxy_log.flush()
            self.source_frame += 1
            yield CapturedFrame(frame, completed)

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('input scheduler did not stop within five seconds')
        with self.proxy_lock:
            if self.proxy_log is not None:
                self.proxy_log.close()
        if self.error is not None:
            raise self.error


def percentiles(values):
    ordered = sorted(values)
    return dict(count=len(ordered), p50=ordered[math.ceil(len(ordered)*.50)-1] if ordered else None,
        p95=ordered[math.ceil(len(ordered)*.95)-1] if ordered else None,
        max=ordered[-1] if ordered else None, method='nearest_rank')


def summarize_dummy_run(output):
    """Rebuild metrics only from durable records, including the validated failure prefix."""
    output = Path(output)
    def rows(name):
        path = output/name
        return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
    report = json.loads((output/'report.json').read_text())
    source = json.loads((output/'dummy_source.json').read_text())
    plan = json.loads((output/'input_events.json').read_text())
    events, proxies, frames, requests = (rows(name) for name in (
        'events.jsonl', 'proxy_frames.jsonl', 'frames.jsonl', 'requests.jsonl'))
    epoch, end = report.get('stream_start_ns'), report['stream_end_ns']
    late_events = [row for row in events if row['emitted_ns'] > end]
    late_proxies = [row for row in proxies if row['proxy_ready_ns'] > end]
    events = [row for row in events if row['emitted_ns'] <= end]
    proxies = [row for row in proxies if row['proxy_ready_ns'] <= end]
    duration = end-epoch if epoch is not None else None
    received = {row['source_frame']: row for row in frames}
    first_proxy = {}
    for row in proxies:
        if row['event_id'] is not None:
            first_proxy.setdefault(row['event_id'], row)
    emitted = {row['event_id']: row for row in events}
    event_results = []
    for event in plan:
        result = dict(event)
        arrival = emitted.get(event['event_id'])
        proxy = first_proxy.get(event['event_id'])
        result['planned_ns'] = epoch+round(event['offset_ms']*1e6) if epoch is not None else None
        if arrival is None:
            result['status'] = 'unemitted'
        elif proxy is None:
            result.update(arrival)
            result['status'] = 'superseded' if arrival['event_id'] != events[-1]['event_id'] else 'never_applied'
        else:
            result.update(arrival)
            result.update(first_source_frame=proxy['source_frame'], input_consumed_ns=proxy['input_consumed_ns'],
                input_to_proxy_ns=proxy['proxy_ready_ns']-arrival['emitted_ns'])
            rgb = received.get(proxy['source_frame'])
            result['status'] = 'returned' if rgb is not None else 'unreturned'
            if rgb is not None:
                result.update(receive_ns=rgb['receive_ns'],
                    input_to_associated_rgb_receive_ns=rgb['receive_ns']-arrival['emitted_ns'])
        event_results.append(result)
    prompt_plan_path = output/'prompt_events.json'
    prompt_plan = json.loads(prompt_plan_path.read_text()) if prompt_plan_path.exists() else []
    prompt_events = rows('prompt_events.jsonl')
    late_prompts = [row for row in prompt_events if row['emitted_ns'] > end]
    prompt_events = [row for row in prompt_events if row['emitted_ns'] <= end]
    prompt_arrivals = {row['revision']: row for row in prompt_events}
    prompt_results = []
    for event in prompt_plan:
        revision = event['revision']
        arrival = prompt_arrivals.get(revision)
        submissions = [row for row in requests if row['operation'] == 'chunk'
            and row.get('input', {}).get('prompt', {}).get('revision') == revision]
        associated = [row for row in frames if row.get('prompt_revision') == revision]
        result = dict(event, planned_ns=epoch+round(event['offset_ms']*1e6) if epoch is not None else None,
            submitted_chunks=len(submissions), submitted_frames=sum(len(row['input']['capture_ns']) for row in submissions),
            returned_frames=len(associated))
        result['censored_submitted_frames'] = result['submitted_frames']-result['returned_frames']
        if arrival is None:
            result['status'] = 'unemitted'
        else:
            result.update(arrival)
            if not submissions:
                result['status'] = ('superseded_before_submission' if revision != prompt_events[-1]['revision']
                    else 'never_submitted')
            else:
                first = submissions[0]
                result.update(first_submitted_chunk=first['chunk_id'],
                    first_source_frame=first['input']['first_source_frame'], prompt_frozen_ns=first['prompt_frozen_ns'])
                result['status'] = 'returned' if associated else 'submitted_unreturned'
                if associated:
                    result.update(receive_ns=associated[0]['receive_ns'],
                        prompt_to_associated_rgb_receive_ns=associated[0]['receive_ns']-arrival['emitted_ns'])
        prompt_results.append(result)
    bins = []
    if duration is not None and duration > 0:
        for index in range(math.ceil(duration/1e9)):
            begin, stop = epoch+index*1_000_000_000, min(epoch+(index+1)*1_000_000_000, end)
            bins.append(dict(start_ns=begin, end_ns=stop, duration_ns=stop-begin, frame_count=0))
        for row in frames:
            bins[min((row['receive_ns']-epoch)//1_000_000_000, len(bins)-1)]['frame_count'] += 1
        for item in bins:
            item['fps'] = item['frame_count']*1e9/item['duration_ns']
    batches = [row for row in requests if row['operation'] == 'chunk' and 'server' in row
               and row['server']['frame_count'] > 0]
    submitted_chunks = [row['input'] for row in requests if row['operation']=='chunk' and 'input' in row]
    arrivals = [row['client']['receive_ns'] for row in batches]
    gaps = [b-a for a, b in zip(arrivals, arrivals[1:])]
    boundaries = [epoch, *arrivals, end] if epoch is not None else []
    no_frame_gaps = [b-a for a, b in zip(boundaries, boundaries[1:])]
    latencies = [row['input_to_associated_rgb_receive_ns'] for row in event_results if row['status']=='returned']
    status_counts = {name: sum(row['status']==name for row in event_results)
        for name in ('returned', 'unreturned', 'unemitted', 'superseded', 'never_applied')}
    failure_ns = report.get('producer_error_ns')
    paced = report['pace']=='24'
    overload = (paced and epoch is not None and failure_ns is not None and epoch <= failure_ns <= end
        and 'paced producer overflow;' in (report.get('producer_error') or ''))
    summary = dict(workload='dummy_infra', status=report['status'], pace=report['pace'], clock='client_perf_counter_ns',
        association='source association only; semantic response not measured',
        primary_metric='input_to_associated_rgb_receive_ns',
        input_to_associated_rgb_receive_ns=percentiles(latencies), events=event_results,
        prompt_events=prompt_results,
        prompt_to_associated_rgb_receive_ns=percentiles([row['prompt_to_associated_rgb_receive_ns']
            for row in prompt_results if row['status'] == 'returned']),
        prompt_event_counts=dict(planned=len(prompt_plan), emitted=len(prompt_events),
            submitted=sum(row['submitted_chunks'] > 0 for row in prompt_results),
            pending_censored=sum(row['status'] != 'returned' for row in prompt_results),
            **{status: sum(row['status'] == status for row in prompt_results) for status in
                ('unemitted', 'superseded_before_submission', 'never_submitted', 'submitted_unreturned', 'returned')}),
        event_counts=dict(planned=len(plan), emitted=len(events), applied=len(first_proxy),
            pending_censored=len(plan)-len(latencies), **status_counts),
        shutdown_records=dict(input_events=len(late_events), prompt_events=len(late_prompts), proxy_frames=len(late_proxies)),
        source_counts=dict(planned=report['spec']['num_frames'], produced=len(proxies),
            submitted=report['submitted_frames'], returned=len(frames),
            unreturned_produced=len(proxies)-len(frames)),
        stream_start_ns=epoch, stream_end_ns=end, observed_duration_ns=duration,
        session_init_ns=report.get('session_init_ns'),
        cleanup_ns=report.get('cleanup_ns'),
        video_and_first_image_preparation_ns=source.get('video_and_first_image_preparation_ns'),
        first_rgb_receive_latency_ns=arrivals[0]-epoch if arrivals else None,
        actual_rgb_fps=dict(numerator_frames=len(frames), denominator_ns=duration,
            window_start_ns=epoch, window_end_ns=end,
            fps=len(frames)*1e9/duration if duration else None),
        wall_second_bins=bins, batch_receive_gap_ns=percentiles(gaps),
        no_new_frame_gap_ns=percentiles(no_frame_gaps),
        startup_no_frame_ns=arrivals[0]-epoch if arrivals else duration,
        tail_no_frame_ns=end-arrivals[-1] if arrivals else duration,
        capture_age_ns=percentiles([row['capture_to_receive_ns'] for row in frames]),
        input_schedule_lateness_ns=percentiles([row['schedule_lateness_ns'] for row in events]),
        source_schedule_lateness_ns=percentiles([row['proxy_ready_ns']-epoch-round(row['source_frame']*1e9/24)
            for row in proxies] if report['pace']=='24' and epoch is not None else []),
        proxy_generation_ns=percentiles([row['proxy_generation_ns'] for row in proxies]),
        submitted_capture_to_assembly_ns=percentiles([chunk['assembled_ns']-capture
            for chunk in submitted_chunks for capture in chunk['capture_ns']]),
        submitted_chunk_assembly_span_ns=percentiles([chunk['capture_ns'][-1]-chunk['capture_ns'][0]
            for chunk in submitted_chunks]),
        assembly_to_send_ns=percentiles([row['client']['send_ns']-row['input']['assembled_ns'] for row in batches]),
        source_lag_frames=percentiles([row['produced_frames']-row['server']['total_frames'] for row in batches]),
        unreturned_submitted_frames=percentiles([row['unreturned_submitted_frames'] for row in batches]),
        max_queued_chunks=report['max_queued_chunks'], queued_chunks_at_stop=report['queued_chunks_at_stop'],
        source_backpressure_ns=report['source_backpressure_ns'],
        overload=overload,
        sustainable_duration_ns=max(0, min(failure_ns if overload else end, end)-epoch)
            if paced and epoch is not None else None,
        sustainable_duration_censored=not overload if paced and epoch is not None else None,
        sustainable_duration_basis='observed_paced_queue_overflow' if overload else
            'no_overflow_observed' if paced else 'not_applicable_unpaced',
        errors={key: report[key] for key in ('error', 'producer_error', 'cleanup_error', 'cancel_error') if report.get(key)},
        received_rgb_bytes=report['received_rgb_bytes'], rgb_sha256=report['rgb_sha256'],
        output_processing_ns=report['output_processing_ns'],
        output_processing='measured separately; delays to subsequent requests remain in run metrics',
        gpu_rgb_ready='not_measured; existing server upper bound stays in server clock', presentation='not_measured')
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
