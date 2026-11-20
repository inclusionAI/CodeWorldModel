import asyncio
import hmac
import secrets
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
import time
from uuid import uuid4

from aiohttp import ClientSession, ClientTimeout

from .serving import PromptSnapshot, Request, SessionError, integer
from .serving_http import MAX_HEADER, pack, unpack
from .media_waits import WAIT_SCHEMA, interval, with_waits
from .native_game_sampling import MediaSlots, SECOND, FPS

MAX_RECORD = (2 << 20) + MAX_HEADER + 4
NATIVE_GAME_CAPTURE_WAIT_NS = 63_000_000
NATIVE_GAME_CAPTURE_HTTP_TIMEOUT = .2


@dataclass
class PartialProxy:
    chunk_id: int
    first_source_frame: int
    received_ns: int
    pixels: bytearray = field(repr=False)
    capture_ns: list = field(default_factory=list)
    timer: asyncio.TimerHandle | None = None


class MediaBrowserSession:
    def __init__(self, gateway, socket):
        self.gateway, self.socket = gateway, socket
        self.sid = uuid4().hex
        self.inflight = self.output_task = None
        self.closing = self.complete = self.attempted = False
        self.starting = False
        self.started = False
        self.native_game_source = None
        self.native_game_bridge = None
        self.native_game_samples = deque()
        self.native_game_pending = False
        self.native_game_captured = self.native_game_acked = 0
        self.native_game_last_capture = -1
        self.native_game_last_evidence = None
        self.native_game_slots = MediaSlots()
        self.native_game_segment = 0
        self.native_game_generation = None
        self.incremental_proxy = gateway.config['media'].get('proxy_protocol') == 'proxy_frames/v1'
        self.partial = None
        self.proxy_assembly = None
        self.control_socket = self.pending_control = None
        self.control_capability = secrets.token_urlsafe(32) if gateway.config['media'].get('control_protocol') else None
        self.cleanup_task = self.termination_task = None
        self.terminal = None
        self.send_lock = asyncio.Lock()
        self.frame_released = asyncio.Event()
        self.unacknowledged = deque()
        self.sources = deque()
        self.frames = deque()
        self.next_chunk = self.output_id = self.emitted = self.next_frame = 0
        self.last_sent_frame = -1
        self.next_source = None
        self.last_capture = -1
        self.input_closed = False
        self.video = None
        self.observe_waits = False
        self.wait_counts = dict(interrupted=0, omitted=0)
        self.interrupted_waits = {}
        self.cancel_receipt = None
        self.first_close = None
        self.profiling = dict(status='unknown', source='start')
        timeout = gateway.client.timeout
        self.http = ClientSession(timeout=ClientTimeout(total=None, sock_connect=min(timeout, 15), sock_read=timeout))
        host = gateway.client.host
        self.url = f'http://{"[" + host + "]" if ":" in host else host}:{gateway.client.port}'

    def bind_control(self, capability, socket):
        if (self.gateway.owner is not self or self.closing or self.control_socket is not None
                or self.control_capability is None or not isinstance(capability, str)
                or len(capability) != 43 or not capability.isascii()
                or not hmac.compare_digest(capability, self.control_capability)):
            raise SessionError('invalid control binding')
        self.control_capability = None
        self.control_socket = socket

    def begin_cleanup(self):
        if self.cleanup_task is None:
            self.note_close('session', 'unknown')
            self.closing = True
            self.native_game_samples.clear()
            self.drop_partial('error' if self.terminal and self.terminal['op'] == 'error' else 'cancelled')
            self.control_capability = None
            self.frame_released.set()
            self.cleanup_task = asyncio.create_task(self.cleanup())
            self.termination_task = asyncio.create_task(self.gateway.finish_media(self))
            for task in (self.cleanup_task, self.termination_task):
                self.gateway.cleanups.add(task)
                task.add_done_callback(self.gateway.cleanups.discard)
        return self.cleanup_task

    def note_close(self, channel, trigger, code=None, exception=None):
        if self.first_close is None:
            self.first_close = dict(channel=channel, trigger=trigger, close_code=code,
                error_class=type(exception).__name__ if exception is not None else None,
                gateway_ns=time.perf_counter_ns(), outstanding=len(self.unacknowledged),
                output_frame_head=self.unacknowledged[0] if self.unacknowledged else None,
                last_sent_frame=self.last_sent_frame if self.last_sent_frame >= 0 else None,
                last_acked_frame=self.native_game_acked - 1 if self.native_game_acked else None,
                source_queue=len(self.sources), frame_queue=len(self.frames),
                native_game_sample_queue=len(self.native_game_samples))

    def drop_partial(self, outcome='complete'):
        partial, self.partial = self.partial, None
        if partial is not None:
            partial.timer.cancel()
            partial.pixels.clear()
            if self.observe_waits and outcome != 'complete':
                self.interrupted_waits['proxy_assembly'] = interval(
                    partial.received_ns, time.perf_counter_ns(), outcome)
                self.wait_counts['interrupted'] += 1

    def expire_partial(self, partial):
        if self.gateway.owner is self and not self.closing and self.partial is partial:
            self.note_close('proxy', 'timeout')
            self.terminal = dict(op='error', error='proxy upload deadline exceeded')
            self.drop_partial('timeout')
            if self.observe_waits:
                self.terminal = with_waits(self.terminal, 'gateway', self.interrupted_waits, self.wait_counts)
            self.begin_cleanup()

    def prepare_input(self, metadata, pixels, *, received_ns=None):
        if self.closing or self.complete:
            raise SessionError('session is closing')
        if self.inflight is not None and not self.inflight.done():
            raise SessionError('one backend request is already in flight', 409)
        op = metadata.get('op')
        if op == 'start':
            return self.prepare(metadata, pixels, received_ns=received_ns)
        if not self.started or self.input_closed:
            raise SessionError('media input is not ready or is closed')
        if self.native_game_source is not None:
            count = 5 if metadata.get('eos') is True else 17
            samples = list(self.native_game_samples)[:count]
            if (op != 'native_game_chunk' or pixels or len(samples) != count or self.native_game_pending
                    or metadata.get('eos') is True and len(self.native_game_samples) != count
                    or metadata.get('first_source_frame') != samples[0][0]['source_frame']
                    or metadata.get('capture_ns') != [sample[0]['capture_ns'] for sample in samples]):
                raise SessionError('Native game commit differs from staged samples')
            job = self.prepare(dict(metadata, op='chunk'), b''.join(sample[1] for sample in samples),
                               received_ns=received_ns)
            for _ in samples:
                self.native_game_samples.popleft()
            return job
        if not self.incremental_proxy:
            return self.prepare(metadata, pixels, received_ns=received_ns)
        if op not in ('proxy_frame', 'proxy_commit'):
            raise SessionError('expected proxy_frame or proxy_commit')
        partial = self.partial
        loop = asyncio.get_running_loop()
        if partial is not None and loop.time() >= partial.timer.when():
            self.expire_partial(partial)
            raise SessionError('proxy upload deadline exceeded')
        if op == 'proxy_frame':
            if set(metadata) != {'op', 'chunk_id', 'source_frame', 'capture_ns'}:
                raise SessionError('invalid proxy frame fields')
            cid = integer(metadata['chunk_id'], 'chunk_id')
            source = integer(metadata['source_frame'], 'source_frame')
            stamp = integer(metadata['capture_ns'], 'capture_ns')
            count = len(partial.capture_ns) if partial is not None else 0
            expected_source = partial.first_source_frame + count if partial is not None else self.next_source
            last_capture = partial.capture_ns[-1] if partial is not None else self.last_capture
            if (cid != self.next_chunk or expected_source is not None and source != expected_source
                    or stamp < last_capture or count >= 17):
                raise SessionError('out-of-order proxy frame')
            spec = self.gateway.config['encoder_recipe']
            frame_bytes = spec['proxy_width'] * spec['proxy_height'] * 3
            if len(pixels) != frame_bytes:
                raise SessionError('proxy frame differs from artifact geometry')
            if partial is None:
                partial = PartialProxy(cid, source, received_ns if received_ns is not None else time.perf_counter_ns(),
                                       bytearray(17 * frame_bytes))
                remaining = self.gateway.client.timeout - (time.perf_counter_ns() - partial.received_ns) / 1e9
                partial.timer = loop.call_later(max(0, remaining), self.expire_partial, partial)
                self.partial = partial
            partial.pixels[count * frame_bytes:(count + 1) * frame_bytes] = pixels
            partial.capture_ns.append(stamp)
            return None
        if (partial is None or pixels or metadata.get('chunk_id') != partial.chunk_id
                or metadata.get('first_source_frame') != partial.first_source_frame
                or metadata.get('capture_ns') != partial.capture_ns):
            raise SessionError('proxy commit differs from staged frames')
        spec = self.gateway.config['encoder_recipe']
        size = len(partial.capture_ns) * spec['proxy_width'] * spec['proxy_height'] * 3
        job = self.prepare(dict(metadata, op='chunk'), bytes(memoryview(partial.pixels)[:size]),
                           received_ns=received_ns)
        if self.observe_waits:
            self.proxy_assembly = interval(partial.received_ns, job.received_ns)
        self.drop_partial()
        return job

    async def capture_native_game(self, source_frame, capture_ns, segment=0):
        bridge = self.native_game_bridge
        if (self.native_game_source is None or bridge.media is not self or self.gateway.owner is not self
                or not self.started or self.closing or self.input_closed or self.native_game_pending):
            raise SessionError('Native game capture is unavailable')
        source_frame = integer(source_frame, 'source_frame')
        capture_ns = integer(capture_ns, 'capture_ns')
        segment = integer(segment, 'segment')
        if (source_frame != self.native_game_captured or capture_ns < self.native_game_last_capture
                or capture_ns > 2**53 - 1 or not self.native_game_segment <= segment <= 2**53 - 1):
            raise SessionError('out-of-order native game sample')
        limit = self.native_game_source['budget_frames']
        if (len(self.native_game_samples) >= 34 or len(self.sources) + len(self.frames) + len(self.native_game_samples) >= 256
                or limit is not None and self.native_game_captured - self.native_game_acked >= limit):
            return dict(blocked='source_budget')
        self.native_game_pending = True
        native = bridge.native
        generation = native.generation
        begin = time.perf_counter_ns()
        slots = self.native_game_slots

        def valid_owner():
            if (bridge.native is not native or native.instance != self.native_game_source['instance_id']
                    or native.epoch != self.native_game_source['epoch']):
                raise SessionError('Native game capture owner changed')
            if (self.closing and (bridge.media is None or bridge.media is self)
                    and (self.gateway.owner is None or self.gateway.owner is self)):
                return False
            if (self.closing or self.input_closed or bridge.media is not self or self.gateway.owner is not self):
                raise SessionError('Native game capture owner changed')
            return native.generation == generation and (source_frame == 0 or native.live)

        try:
            if source_frame == 0:
                async with native.lock:
                    if bridge.prepared is None:
                        raise SessionError('Native game prepared frame expired')
                    if not (await native.status())['paused']:
                        raise SessionError('Native game prepared source is no longer paused')
                    publication = bridge.publication(bridge.prepared, time.perf_counter_ns())
                    evidence = publication.evidence
                media = dict(media_segment=-1, media_slot=0, sample_kind='prepared',
                             media_reason='prepared', media_skipped=0, media_omitted_ns=0,
                             media_target_ns=evidence['published_monotonic_ns'])
                slots.cursor = evidence['frame_id']
            else:
                if not native.live:
                    raise SessionError('Native game sampling is paused')
                if self.native_game_generation != generation:
                    slots.restart('resume', begin)
                elif segment != self.native_game_segment:
                    slots.restart('capacity', begin)
                elif slots.origin_ns is not None and begin > slots.deadline_ns() + SECOND // FPS:
                    slots.restart('scheduling_gap', begin)
                self.native_game_generation, self.native_game_segment = generation, segment
                selected = None
                http_deadline = asyncio.get_running_loop().time() + getattr(native, 'capture_timeout', NATIVE_GAME_CAPTURE_HTTP_TIMEOUT)
                while selected is None:
                    now = time.perf_counter_ns()
                    if now - begin >= NATIVE_GAME_CAPTURE_WAIT_NS:
                        return dict(waiting=True)
                    remaining = NATIVE_GAME_CAPTURE_WAIT_NS - (now - begin)
                    if slots.origin_ns is not None:
                        remaining = min(remaining, max(0, slots.deadline_ns() - now))
                    async with asyncio.timeout_at(http_deadline):
                        result = await native.frame(after_frame_id=slots.cursor, latest=True,
                                                    wait_ms=(remaining + 999_999) // 1_000_000)
                        if (result is None and slots.origin_ns is not None
                                and time.perf_counter_ns() >= slots.deadline_ns() and valid_owner()):
                            result = await native.frame(after_frame_id=slots.cursor, latest=True, wait_ms=0)
                    if not valid_owner():
                        return dict(cancelled=True)
                    acquired_ns = time.perf_counter_ns()
                    if slots.origin_ns is not None and acquired_ns > slots.deadline_ns() + SECOND // FPS:
                        slots.restart('native_wait_gap', acquired_ns)
                    if result is not None:
                        slots.offer(bridge.publication(result, acquired_ns))
                    selected = slots.take(time.perf_counter_ns())
                    if selected is None:
                        await asyncio.sleep(.001)
                publication, media = selected
            if not valid_owner():
                return dict(cancelled=True)
            selected_ns = time.perf_counter_ns()
            if publication.pixels is None:
                async with asyncio.timeout(12):
                    publication.pixels = await bridge.rgb(publication.jpeg, self.gateway.config['encoder_recipe'])
                publication.ready_ns = time.perf_counter_ns()
            if not valid_owner():
                return dict(cancelled=True)
            evidence, pixels = publication.evidence, publication.pixels
            ready = time.perf_counter_ns()
            previous = self.native_game_last_evidence
            receipt = dict(source_frame=source_frame, capture_ns=capture_ns, **media,
                capture_clock='browser_performance_now', gateway_clock='python_perf_counter_ns',
                gateway_acquired_ns=publication.acquired_ns, gateway_acquire_ns=max(0, publication.acquired_ns - begin),
                gateway_selected_ns=selected_ns, gateway_sample_ready_ns=ready,
                gateway_rgb_ready_ns=publication.ready_ns,
                gateway_transform_ns=max(0, publication.ready_ns - selected_ns), transform=self.native_game_source['transform'],
                source=dict(evidence, publication_clock=evidence.get('publication_clock', 'native_monotonic'),
                    repeated_native_frame=previous is not None and previous['frame_id'] == evidence['frame_id'],
                    repeated_jpeg=previous is not None and previous['jpeg_sha256'] == evidence['jpeg_sha256'],
                    input_association='last-browser-input-not-native-action-proof'))
            self.native_game_samples.append((receipt, pixels))
            self.native_game_captured += 1
            self.native_game_last_capture = capture_ns
            self.native_game_last_evidence = evidence
            if source_frame == 0:
                bridge.prepared = None
            return dict(sample=receipt)
        except ValueError:
            if not valid_owner():
                return dict(cancelled=True)
            raise
        finally:
            self.native_game_pending = False

    async def send(self, metadata, pixels=b''):
        started = time.perf_counter_ns() if self.observe_waits else None
        acquired = False
        try:
            async with self.send_lock:
                acquired = True
                if started is not None:
                    metadata = with_waits(metadata, 'gateway',
                        dict(send_lock=interval(started, time.perf_counter_ns())), self.wait_counts)
                if not self.socket.closed:
                    await self.socket.send_bytes(pack(metadata, pixels))
                    if metadata.get('op') == 'video_frame':
                        self.last_sent_frame = metadata['output_frame']
        finally:
            if started is not None and not acquired:
                self.interrupted_waits['send_lock'] = interval(started, time.perf_counter_ns(), 'cancelled')
                self.wait_counts['interrupted'] += 1

    def prepare(self, metadata, pixels, *, received_ns=None):
        if self.closing:
            raise SessionError('session is closing')
        m = dict(metadata)
        op = m.pop('op', None)
        if op == 'start':
            source = m.pop('source', None)
            observe_waits = m.pop('observe_waits', False)
            if type(observe_waits) is not bool:
                raise SessionError('observe_waits must be boolean')
            if self.attempted or set(m) != {'seed', 'condition_seed', 'video'}:
                raise SessionError('media start requires seed, condition_seed and video')
            policy = self.gateway.config['media']
            video = m['video']
            if not isinstance(video, dict) or set(video) != {'bitrate_bps'}:
                raise SessionError('video requires bitrate_bps')
            bitrate = integer(video['bitrate_bps'], 'bitrate_bps', policy['bitrate_min_bps'])
            if bitrate > policy['bitrate_max_bps']:
                raise SessionError('bitrate exceeds server maximum')
            self.video = dict(video)
            self.observe_waits = observe_waits
            if observe_waits:
                m['observe_waits'] = True
            m.update(artifact_id=self.gateway.config['artifact_id'],
                     prompt=self.gateway.config['encoder_recipe']['system_prompt'])
            if source is not None:
                if not isinstance(source, dict):
                    raise SessionError('Native game source is unavailable')
                bridge = self.gateway.native_game_sources.get(source.get('protocol'))
                if bridge is None:
                    raise SessionError('Native game source is unavailable')
                self.native_game_bridge = bridge
                self.native_game_source = bridge.claim_prepared(source, self)
                spec = self.gateway.config['encoder_recipe']
                if (source.get('reference') == 'prepared' and pixels
                        or source.get('reference') == 'upload' and len(pixels) != spec['width'] * spec['height'] * 3):
                    raise SessionError('invalid native game reference payload')
        elif op == 'chunk':
            fields = {'chunk_id', 'first_source_frame', 'capture_ns', 'assembled_ns', 'eos'}
            if self.input_closed or not self.attempted or set(m) - {'prompt'} != fields:
                raise SessionError('invalid or closed media input')
            if type(m['eos']) is not bool:
                raise SessionError('eos must be boolean')
            cid = integer(m['chunk_id'], 'chunk_id')
            first = integer(m['first_source_frame'], 'first_source_frame')
            assembled = integer(m['assembled_ns'], 'assembled_ns')
            count = 5 if m['eos'] else 17
            stamps = m['capture_ns']
            if (cid != self.next_chunk or self.next_source is not None and first != self.next_source
                    or not isinstance(stamps, list) or len(stamps) != count):
                raise SessionError('out-of-order media input')
            for stamp in stamps:
                integer(stamp, 'capture_ns')
            if (stamps[0] < self.last_capture or any(a > b for a, b in zip(stamps, stamps[1:]))
                    or assembled < stamps[-1]):
                raise SessionError('invalid capture timestamps')
            spec = self.gateway.config['encoder_recipe']
            if len(pixels) != count * spec['proxy_width'] * spec['proxy_height'] * 3:
                raise SessionError('proxy payload differs from artifact geometry')
            if len(self.sources) + len(self.frames) + count > 256:
                raise SessionError('pending source ledger exceeded its bound')
            if 'prompt' in m:
                PromptSnapshot.parse(m['prompt'])
            self.sources.extend(dict(source_frame=first+i, capture_ns=stamp, chunk_id=cid)
                                for i, stamp in enumerate(stamps))
            self.next_chunk += 1
            self.next_source = first + count
            self.last_capture = stamps[-1]
            self.input_closed = m['eos']
        else:
            raise SessionError('expected media start or chunk')
        m['session_id'] = self.sid
        return Request(op, m, pixels) if received_ns is None else Request(op, m, pixels, received_ns)

    async def call(self, path, metadata, pixels=b''):
        begin = time.perf_counter_ns()
        async with self.http.post(self.url + path, data=pack(metadata, pixels),
                                  headers={'Content-Type': 'application/octet-stream'}) as response:
            size = response.content_length
            if size is None or not 6 <= size <= MAX_HEADER + 4:
                raise SessionError('media acknowledgement exceeds header bound', 502)
            body = await response.content.readexactly(size)
            result, data = unpack(body)
            if response.status != 200:
                raise SessionError(result.get('error', 'media request failed'), response.status)
            if data:
                raise SessionError('media acknowledgement contained pixels', 502)
        return result, (time.perf_counter_ns() - begin) / 1e6

    async def run(self, job):
        from .web_serving import profiling_status, source_budget_capability
        if self.closing:
            return
        rpc_started = time.perf_counter_ns() if self.observe_waits else None
        rpc_ended = None
        registered = False
        try:
            if job.op == 'start':
                self.starting = True
                if self.native_game_source is not None:
                    bridge = self.native_game_bridge
                    async with bridge.native.lock:
                        if not (await bridge.native.status())['paused']:
                            raise SessionError('Native game prepared source is no longer paused')
                    if self.native_game_source['reference'] == 'prepared':
                        pixels = await bridge.reference_rgb(self.gateway.config['encoder_recipe'])
                        job = Request(job.op, job.metadata, pixels, job.received_ns)
                    if self.closing:
                        return
                self.attempted = True
                try:
                    m, rpc_ms = await self.call('/media/session', job.metadata, job.pixels)
                    rpc_ended = time.perf_counter_ns() if self.observe_waits else None
                except SessionError as exc:
                    if exc.status == 409:
                        self.attempted = False
                    raise
                media = m.get('media', {})
                if (m.get('session_id') != self.sid or media.get('bitrate_bps') != self.video['bitrate_bps']
                        or media.get('protocol') != 'h264_annexb_v1'):
                    raise SessionError('backend media start receipt differs from request', 502)
                self.profiling = profiling_status(m, fixture=self.gateway.config['mode'] == 'fixture', source='start')
                observation = m.get('wait_observability', {})
                enabled = isinstance(observation, dict) and observation.get('schema') == WAIT_SCHEMA and observation.get('enabled') is True
                if enabled != self.observe_waits:
                    raise SessionError('backend wait observation receipt differs from request', 502)
                registered = True
                self.starting = False
                m = dict(m, wait_observability=dict(schema=WAIT_SCHEMA, enabled=enabled))
                if self.native_game_source is not None:
                    m['source'] = self.native_game_source
                if self.observe_waits:
                    m = with_waits(m, 'gateway', dict(request_complete_ns=job.received_ns,
                        rpc_start_ns=rpc_started, rpc_end_ns=rpc_ended), self.wait_counts)
                if not self.closing:
                    capability = source_budget_capability(dict(self.gateway.config, media=m.get('media'),
                        recipe=m.get('recipe') if m.get('recipe') is not None else self.gateway.config['recipe']))
                    await self.send(dict(m, op='started', profiling=self.profiling, rpc_ms=rpc_ms,
                                         credit=1, source_budget=capability))
                    if not self.closing and not self.socket.closed:
                        self.started = True
                        self.output_task = asyncio.create_task(self.pump())
            else:
                if self.native_game_source is not None and job.metadata['eos']:
                    await self.native_game_bridge.pause_media(self)
                m, rpc_ms = await self.call(f'/media/session/{self.sid}/chunk', job.metadata, job.pixels)
                rpc_ended = time.perf_counter_ns() if self.observe_waits else None
                if m.get('session_id') != self.sid or m.get('chunk_id') != job.metadata['chunk_id'] or m.get('status') != 'admitted':
                    raise SessionError('backend admission receipt differs from request', 502)
                if not self.closing:
                    if self.observe_waits:
                        m = with_waits(m, 'gateway', dict(request_complete_ns=job.received_ns,
                            rpc_start_ns=rpc_started, rpc_end_ns=rpc_ended), self.wait_counts)
                        if self.proxy_assembly is not None:
                            m = with_waits(m, 'gateway', dict(proxy_assembly=self.proxy_assembly), self.wait_counts)
                    await self.send(dict(m, op='admitted', rpc_ms=rpc_ms, credit=1))
        except Exception as exc:
            if job.op == 'start' and self.attempted and not registered:
                self.gateway.quarantine = 'Backend media registration uncertain; check media service'
            if not self.closing:
                error = dict(op='error', session_id=self.sid, error=str(exc))
                if rpc_started is not None:
                    rpc_ended = rpc_ended or time.perf_counter_ns()
                    error = with_waits(error, 'gateway', dict(admission_rpc=interval(rpc_started, rpc_ended, 'error')),
                        self.wait_counts)
                    error['wait_coverage'] = dict(gateway=dict(self.wait_counts), gateway_complete=False)
                self.terminal = error
                self.note_close('run', 'error', exception=exc)
                self.begin_cleanup()
        finally:
            if rpc_started is not None and rpc_ended is None:
                self.interrupted_waits['admission_rpc'] = interval(rpc_started, time.perf_counter_ns(),
                    'cancelled' if self.closing else 'error')
                self.wait_counts['interrupted'] += 1
            self.starting = False
            self.proxy_assembly = None

    def batch(self, m, data):
        spec = self.gateway.config['encoder_recipe']
        count = integer(m.get('frame_count'), 'frame_count')
        if (data or m.get('output_id') != self.output_id or m.get('output_frame_start') != self.emitted
                or m.get('width') != spec['width'] or m.get('height') != spec['height']
                or not isinstance(m.get('source_frames'), list) or len(m['source_frames']) != count
                or count > len(self.sources) or m.get('status') not in ('accepted', 'complete')):
            raise SessionError('invalid media batch provenance', 502)
        for source in m['source_frames']:
            expected = self.sources.popleft()
            if not isinstance(source, dict) or any(type(source.get(k)) is not int or source[k] != v for k, v in expected.items()):
                raise SessionError('media source mapping differs from submitted input', 502)
            self.frames.append(dict(expected, output_id=self.output_id, output_frame=self.emitted,
                                    available_after_chunk=m['chunk_id']))
            self.emitted += 1
        self.output_id += 1
        if m['status'] == 'complete' and (self.sources or not self.input_closed):
            raise SessionError('media EOS did not account for all input', 502)

    def frame(self, m, data):
        if not self.frames:
            raise SessionError('media frame has no batch receipt', 502)
        expected = self.frames.popleft()
        if (not data or len(data) > 2 << 20 or m.get('encoded_bytes') != len(data)
                or type(m.get('key_frame')) is not bool
                or any(type(m.get(k)) is not int or m[k] != expected[k]
                       for k in ('output_id', 'output_frame', 'source_frame', 'capture_ns'))
                or m.get('timestamp_us') != expected['output_frame'] * 1000000 // 24
                or m.get('chunk_id') != expected['available_after_chunk']):
            raise SessionError('invalid encoded frame provenance', 502)
        self.next_frame += 1

    def acknowledge(self, m):
        if (set(m) != {'op', 'output_frame'} or type(m['output_frame']) is not int
                or not self.unacknowledged or self.unacknowledged[0] != m['output_frame']):
            raise SessionError('out-of-order frame acknowledgement')
        self.unacknowledged.popleft()
        self.native_game_acked += 1
        self.frame_released.set()

    async def wait_window(self, limit):
        started = time.perf_counter_ns() if self.observe_waits else None
        outcome = 'cancelled'
        try:
            while len(self.unacknowledged) >= limit and not self.closing:
                self.frame_released.clear()
                await asyncio.wait_for(self.frame_released.wait(), self.gateway.client.timeout)
            outcome = 'cancelled' if self.closing else 'complete'
        except asyncio.TimeoutError:
            outcome = 'timeout'
            raise
        finally:
            if started is not None:
                span = interval(started, time.perf_counter_ns(), outcome)
                if outcome != 'complete':
                    self.interrupted_waits['ack_window'] = span
                    self.wait_counts['interrupted'] += 1
        return span if started is not None else None

    async def pump(self):
        try:
            async with self.http.get(self.url + f'/media/session/{self.sid}/events') as response:
                if response.status != 200:
                    raise SessionError('backend rejected media stream', response.status)
                while not self.closing:
                    started = time.perf_counter_ns() if self.observe_waits else None
                    read_complete = False
                    try:
                        size = int.from_bytes(await response.content.readexactly(4), 'big')
                        if not 6 <= size <= MAX_RECORD:
                            raise SessionError('media record exceeds its bound', 502)
                        m, data = unpack(await response.content.readexactly(size))
                        read_complete = True
                    finally:
                        if started is not None:
                            span = interval(started, time.perf_counter_ns(), 'complete' if read_complete else 'cancelled' if self.closing else 'error')
                            if not read_complete:
                                self.interrupted_waits['record_read'] = span
                                self.wait_counts['interrupted'] += 1
                    if self.observe_waits:
                        m = with_waits(m, 'gateway', dict(record_read=span), self.wait_counts)
                    if m.get('session_id') != self.sid:
                        raise SessionError('media event belongs to another session', 502)
                    op = m.get('op')
                    if op == 'batch':
                        self.batch(m, data)
                        m = dict(m, profiling=self.profiling)
                    elif op == 'video_frame':
                        self.frame(m, data)
                        ack_window = await self.wait_window(self.gateway.config['media']['frame_window'])
                        if self.observe_waits:
                            m = with_waits(m, 'gateway', dict(ack_window=ack_window), self.wait_counts)
                        if self.closing:
                            return
                        self.unacknowledged.append(m['output_frame'])
                    elif op == 'complete':
                        if self.frames or self.sources or self.next_frame != self.emitted or m.get('total_frames') != self.emitted:
                            raise SessionError('media completed with missing frames', 502)
                        self.complete = True
                        if self.observe_waits:
                            coverage = m.get('wait_coverage')
                            m = dict(m, wait_coverage=dict(coverage if isinstance(coverage, dict) else {}, gateway=dict(self.wait_counts),
                                gateway_complete=False, terminal_ack_drain='not_collected', socket_write='not_collected'))
                    elif op == 'error':
                        raise SessionError(m.get('error', 'media stream failed'), 502)
                    else:
                        raise SessionError('unknown media event', 502)
                    await self.send(m, data)
                    if self.complete:
                        await self.wait_window(1)
                        self.note_close('pump', 'normal_eos')
                        self.begin_cleanup()
                        return
        except Exception as exc:
            if not self.closing:
                error = dict(op='error', session_id=self.sid, error=str(exc) or type(exc).__name__)
                if self.observe_waits:
                    error = with_waits(error, 'gateway', self.interrupted_waits, self.wait_counts)
                    error['wait_coverage'] = dict(gateway=dict(self.wait_counts), gateway_complete=False)
                self.terminal = error
                self.note_close('pump', 'error', exception=exc)
                self.begin_cleanup()

    async def cleanup(self):
        try:
            if self.native_game_source is not None:
                try:
                    await self.native_game_bridge.detach_media(self)
                except Exception:
                    self.gateway.quarantine = 'Native game cleanup uncertain; reconnect source and check native service'
            for task in (self.inflight, self.output_task):
                if task is not None:
                    if task is not self.inflight or not self.starting:
                        task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
            if self.attempted and not self.complete:
                m, _ = await self.call(f'/media/session/{self.sid}/cancel',
                                       dict(session_id=self.sid, reason='browser closed'))
                if m.get('status') != 'cancelled' or m.get('session_id') != self.sid:
                    raise SessionError('backend did not confirm media cancellation')
                self.cancel_receipt = m
        except Exception as exc:
            self.gateway.quarantine = f'Backend cleanup uncertain; check media service: {exc}'
        finally:
            if self.observe_waits and self.terminal is not None:
                self.terminal = with_waits(self.terminal, 'gateway', self.interrupted_waits, self.wait_counts)
            if self.observe_waits and self.cancel_receipt is not None:
                self.cancel_receipt = with_waits(self.cancel_receipt, 'gateway', self.interrupted_waits, self.wait_counts)
                coverage = self.cancel_receipt.get('wait_coverage')
                self.cancel_receipt['wait_coverage'] = dict(coverage if isinstance(coverage, dict) else {},
                    gateway=dict(self.wait_counts), gateway_complete=True, socket_write='not_collected')
            try:
                await self.http.close()
            except Exception as exc:
                self.gateway.quarantine = f'Media client cleanup failed: {exc}'
            finally:
                self.sources.clear()
                self.frames.clear()
                self.unacknowledged.clear()
                if not self.gateway.quarantine and self.gateway.owner is self:
                    self.gateway.owner = None
