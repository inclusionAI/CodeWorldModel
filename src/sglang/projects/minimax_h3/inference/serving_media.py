from collections import deque
from fractions import Fraction
import re
import threading
import time

from .serving import Reply, Request, SessionError
from .serving_stages import CpuOutput
from .media_waits import WAIT_SCHEMA, interval, with_waits


FPS = 24
MAX_ACCESS_UNIT = 2 << 20
MAX_RGB_BATCH = 128 << 20
MAX_RGB_RESERVATIONS = 2
MAX_RECORD = MAX_ACCESS_UNIT + (64 << 10) + 4
MEDIA_SETTINGS = dict(protocol='h264_annexb_v1', codec='avc1.42C028',
    bitrate_min_bps=500_000, bitrate_max_bps=20_000_000,
    bitrate_default_bps=8_000_000, fps=FPS, frame_window=4)


class H264Encoder:
    def __init__(self, width, height, bitrate):
        if (type(width) is not int or type(height) is not int
                or min(width, height) < 2 or width % 2 or height % 2):
            raise SessionError('H264 requires positive even image dimensions')
        import av
        self.av = av
        self.width, self.height = width, height
        self.codec = av.CodecContext.create('libx264', 'w')
        self.codec.width, self.codec.height = width, height
        self.codec.pix_fmt = 'yuv420p'
        self.codec.time_base = Fraction(1, FPS)
        self.codec.framerate = Fraction(FPS, 1)
        self.codec.bit_rate = bitrate
        self.codec.thread_count = 2
        self.codec.thread_type = 'SLICE'
        self.codec.max_b_frames = 0
        self.codec.gop_size = 48
        kbps = max(1, bitrate // 1000)
        self.codec.options = dict(preset='ultrafast', tune='zerolatency', profile='baseline',
            level='4.0', **{'x264-params': f'annexb=1:repeat-headers=1:bframes=0:'
                f'rc-lookahead=0:sync-lookahead=0:vbv-maxrate={kbps}:vbv-bufsize={kbps}'})
        self.codec.open()

    def encode(self, rgb, ordinal):
        import numpy as np
        frame = self.av.VideoFrame.from_numpy_buffer(
            np.frombuffer(rgb, dtype=np.uint8).reshape(self.height, self.width, 3), format='rgb24')
        frame.pts = ordinal
        packets = self.codec.encode(frame)
        if len(packets) != 1:
            raise SessionError('zero-latency encoder must emit one access unit per frame', 500)
        packet = packets[0]
        payload = bytes(packet)
        if len(payload) > MAX_ACCESS_UNIT:
            raise SessionError('encoded frame exceeds 2 MiB', 500)
        return payload, bool(packet.is_keyframe)

    def flush(self):
        if self.codec.encode(None):
            raise SessionError('zero-latency encoder retained output frames', 500)

    def close(self):
        self.codec = None


class MediaSession:
    def __init__(self, session_id, encoder, settings, observe_waits=False):
        self.session_id, self.encoder, self.settings = session_id, encoder, settings
        self.results, self.events = deque(), deque()
        self.reservations = 0
        self.cancelled = False
        self.error = None
        self.attached = self.reader_done = False
        self.worker_done, self.encoder_done = threading.Event(), threading.Event()
        self.attach_timer = None
        self.observe_waits = observe_waits
        self.wait_counts = dict(published=0, discarded=0, interrupted=0, omitted=0)
        self.interrupted_waits = None


class MediaSessions:
    def __init__(self, mailbox, health, *, encoder_factory=H264Encoder):
        self.mailbox, self.health = mailbox, health
        self.condition = mailbox.condition
        self.timeout = mailbox.timeout
        self.encoder_factory = encoder_factory
        self.active = self.last = None
        self.available = False
        try:
            probe = encoder_factory(16, 16, MEDIA_SETTINGS['bitrate_default_bps'])
            probe.close()
            self.available = True
            health['media'] = dict(MEDIA_SETTINGS)
        except (ImportError, OSError, RuntimeError, ValueError, SessionError) as exc:
            self.unavailable_reason = str(exc)

    def start(self, metadata, pixels):
        if not self.available:
            raise SessionError('H264 media encoder unavailable', 503)
        metadata = dict(metadata)
        observe_waits = metadata.pop('observe_waits', False)
        if type(observe_waits) is not bool:
            raise SessionError('observe_waits must be boolean')
        video = metadata.pop('video', {})
        if not isinstance(video, dict) or set(video) - {'bitrate_bps'}:
            raise SessionError('video accepts only bitrate_bps')
        bitrate = video.get('bitrate_bps', MEDIA_SETTINGS['bitrate_default_bps'])
        if type(bitrate) is not int or not 500_000 <= bitrate <= 20_000_000:
            raise SessionError('video.bitrate_bps must be an integer from 500000 to 20000000')
        sid = metadata.get('session_id')
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', sid):
            raise SessionError('invalid session_id')
        artifact_id = metadata.get('artifact_id')
        if not isinstance(artifact_id, str):
            raise SessionError('unknown artifact_id')
        artifact = self.health.get('artifacts', {}).get(artifact_id)
        if artifact is None:
            raise SessionError('unknown artifact_id')
        geometry = artifact['encoder_recipe']
        with self.condition:
            if self.active is not None or self.mailbox.current is not None or self.mailbox.pending is not None:
                raise SessionError('a session is already active or draining', 409)
            try:
                encoder = self.encoder_factory(geometry['width'], geometry['height'], bitrate)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise SessionError(f'H264 media encoder unavailable: {exc}', 503) from exc
            settings = dict(protocol=MEDIA_SETTINGS['protocol'], codec=MEDIA_SETTINGS['codec'],
                bitrate_bps=bitrate, fps=FPS, width=geometry['width'], height=geometry['height'], frame_window=4)
            session = MediaSession(sid, encoder, settings, observe_waits)
            self.active = session
            try:
                job = self.mailbox.submit(Request('start', metadata, pixels), media=session)
            except BaseException:
                self.active = None
                encoder.close()
                raise
            threading.Thread(target=self._encode, args=(session,), daemon=True,
                name=f'h264-{sid}').start()
            return job

    def submit(self, metadata, pixels):
        with self.condition:
            session = self._session(metadata.get('session_id'))
            if session.cancelled or session.worker_done.is_set():
                raise SessionError('media session is closing', 409)
            return self.mailbox.submit(Request('chunk', metadata, pixels), media=session)

    def _session(self, sid):
        if self.active is None or self.active.session_id != sid:
            raise SessionError('unknown media session', 409)
        return self.active

    def reserve(self, job, controller):
        session = job.media
        started = time.perf_counter_ns() if job.waits is not None else None
        with self.condition:
            if job.request.op not in ('start', 'chunk'):
                return
            if session.cancelled:
                raise SessionError('media session cancelled', 409)
            if job.request.op != 'chunk':
                return
            available = self.condition.wait_for(
                lambda: session.cancelled or session.reservations < MAX_RGB_RESERVATIONS, self.timeout)
            if started is not None:
                job.waits['rgb_reserve'] = interval(started, time.perf_counter_ns(),
                    'timeout' if not available else 'cancelled' if session.cancelled else 'complete')
                if not available or session.cancelled:
                    session.interrupted_waits = dict(job.waits)
                    session.wait_counts['interrupted'] += 1
            if not available:
                self.abort(session, 'RGB output queue timeout')
            if session.cancelled:
                raise SessionError(session.error or 'media session cancelled', 409)
            active = controller.active
            pending = len(active.pending) if active is not None else 0
            count = 5 if job.request.metadata.get('eos') else 17
            if (pending + count) * session.settings['width'] * session.settings['height'] * 3 > MAX_RGB_BATCH:
                raise SessionError('possible RGB output batch exceeds 128 MiB', 413)
            session.reservations += 1
            job.reserved = True

    def admitted(self, job):
        if job.waits is not None:
            job.waits['prepared_ns'] = time.perf_counter_ns()
        if job.request.op == 'chunk':
            with self.condition:
                if job.media.cancelled:
                    raise SessionError('media session cancelled', 409)
                job.reply = Reply(dict(session_id=job.media.session_id,
                    chunk_id=job.request.metadata['chunk_id'], status='admitted'))
                if job.waits is not None:
                    job.reply = Reply(with_waits(job.reply.metadata, 'frontend', job.waits, job.media.wait_counts))
                job.ready.set()

    def release_reservation(self, job):
        if job.reserved:
            job.media.reservations -= 1
            job.reserved = False
            self.condition.notify_all()

    def result(self, job, reply):
        session = job.media
        with self.condition:
            if reply.status != 200:
                if not job.ready.is_set():
                    job.reply = reply
                    job.ready.set()
                self.release_reservation(job)
                if job.request.op == 'start':
                    session.worker_done.set()
                self.abort(session, reply.metadata['error'])
                return False
            if job.request.op == 'chunk':
                if session.cancelled:
                    self.release_reservation(job)
                    return False
                if len(reply.pixels) > MAX_RGB_BATCH:
                    self.release_reservation(job)
                    self.abort(session, 'RGB output batch exceeds 128 MiB')
                    return False
                if session.observe_waits:
                    reply = Reply(with_waits(reply.metadata, 'frontend',
                        dict(rgb_enqueued_ns=time.perf_counter_ns()), session.wait_counts), reply.pixels, reply.status)
                session.results.append(reply)
                job.reserved = False
                self.condition.notify_all()
                return True
            if job.request.op == 'start':
                reply.metadata['media'] = dict(session.settings)
                reply.metadata['wait_observability'] = dict(schema=WAIT_SCHEMA, enabled=session.observe_waits)
                if job.waits is not None:
                    reply.metadata = with_waits(reply.metadata, 'frontend', job.waits, session.wait_counts)
                session.attach_timer = threading.Timer(self.timeout, self._orphan, args=(session,))
                session.attach_timer.daemon = True
                session.attach_timer.start()
            return None

    def worker_closed(self, job):
        if job is not None and job.media is not None:
            with self.condition:
                job.media.worker_done.set()
                self.condition.notify_all()
                self._retire(job.media)

    def abort(self, session, reason, *, schedule=True):
        with self.condition:
            if not session.cancelled:
                session.cancelled, session.error = True, reason
                if session.observe_waits:
                    session.wait_counts['discarded'] += len(session.events) + len(session.results)
                session.events.clear()
                session.reservations -= len(session.results)
                session.results.clear()
                if session.attach_timer is not None:
                    session.attach_timer.cancel()
                pending = self.mailbox.pending
                if pending is not None and pending.media is session:
                    if pending.waits is not None:
                        session.wait_counts['discarded'] += 1
                    self.mailbox.pending = None
                    pending.reply = Reply(dict(status='error', error=reason), status=409)
                    pending.ready.set()
                    pending.released.set()
                    if pending.request.op == 'start':
                        session.worker_done.set()
                current = self.mailbox.current
                cleaned = current is not None and current.media is session and current.cleaned
                if schedule and not cleaned and not session.worker_done.is_set() and self.mailbox.control is None:
                    self.mailbox.submit(Request('cancel', dict(session_id=session.session_id,
                        reason=reason[:1024])), internal=True, media=session)
            self.condition.notify_all()
            self._retire(session)

    def cancel(self, metadata, pixels=b''):
        if (set(metadata) - {'session_id', 'reason'} or pixels
                or not isinstance(metadata.get('reason', ''), str) or len(metadata.get('reason', '')) > 1024):
            raise SessionError('cancel takes session_id and an optional reason up to 1024 characters')
        sid = metadata.get('session_id')
        with self.condition:
            session = self.active if self.active is not None and self.active.session_id == sid else self.last
            if session is None or session.session_id != sid:
                raise SessionError('unknown media session', 409)
            self.abort(session, metadata.get('reason') or 'media session cancelled')
        deadline = time.monotonic() + self.timeout
        for event in (session.worker_done, session.encoder_done):
            if not event.wait(max(0, deadline - time.monotonic())):
                raise SessionError('media cleanup is still pending', 504)
        with self.condition:
            self._retire(session)
        metadata = dict(session_id=sid, status='cancelled')
        if session.observe_waits:
            metadata['wait_coverage'] = dict(frontend=dict(session.wait_counts), frontend_complete=True,
                interrupted_snapshots_omitted=max(0, session.wait_counts['interrupted'] - int(session.interrupted_waits is not None)))
            if session.interrupted_waits:
                metadata = with_waits(metadata, 'frontend', session.interrupted_waits, session.wait_counts)
        return Reply(metadata)

    def attach(self, sid):
        with self.condition:
            session = self._session(sid)
            if session.attached:
                raise SessionError('media output already has a reader', 409)
            session.attached = True
            if session.attach_timer is not None:
                session.attach_timer.cancel()
            return session

    def next_event(self, session):
        with self.condition:
            if not self.condition.wait_for(lambda: session.events or session.cancelled, self.timeout):
                self.abort(session, 'media output timeout')
            if session.cancelled:
                return Reply(dict(op='error', session_id=session.session_id, error=session.error))
            result = session.events.popleft()
            if session.observe_waits:
                metadata = with_waits(result.metadata, 'frontend',
                    dict(event_dequeued_ns=time.perf_counter_ns()), session.wait_counts)
                if metadata.get('op') == 'complete':
                    metadata['wait_coverage'] = dict(frontend=dict(session.wait_counts), frontend_complete=True)
                result = Reply(metadata, result.pixels, result.status)
            self.condition.notify_all()
            return result

    def detach(self, session, complete):
        with self.condition:
            session.reader_done = True
            if not complete:
                self.abort(session, 'media reader disconnected')
            self._retire(session)

    def _orphan(self, session):
        with self.condition:
            if self.active is session and not session.attached:
                self.abort(session, 'media reader attach timeout')

    def _retire(self, session):
        if (self.active is session and session.worker_done.is_set() and session.encoder_done.is_set()
                and (session.cancelled or session.reader_done)):
            self.last, self.active = session, None
            self.condition.notify_all()

    def _publish(self, session, reply):
        started = time.perf_counter_ns() if session.observe_waits else None
        with self.condition:
            available = self.condition.wait_for(lambda: session.cancelled or len(session.events) < 4, self.timeout)
            if started is not None:
                publication = interval(started, time.perf_counter_ns(),
                    'timeout' if not available else 'cancelled' if session.cancelled else 'complete')
                if not available or session.cancelled:
                    session.interrupted_waits = dict(publication=publication)
                    session.wait_counts['interrupted'] += 1
                else:
                    reply = Reply(with_waits(reply.metadata, 'frontend', dict(publication=publication),
                        session.wait_counts), reply.pixels, reply.status)
                    session.wait_counts['published'] += 1
            if not available:
                self.abort(session, 'encoded output queue timeout')
            if session.cancelled:
                return False
            session.events.append(reply)
            self.condition.notify_all()
            return True

    def _encode(self, session):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: session.cancelled or session.results)
                    if session.cancelled:
                        return
                    reply = session.results.popleft()
                    if session.observe_waits:
                        reply = Reply(with_waits(reply.metadata, 'frontend',
                            dict(rgb_taken_ns=time.perf_counter_ns()), session.wait_counts), reply.pixels, reply.status)
                def release_rgb():
                    nonlocal reply
                    with self.condition:
                        if reply is not None:
                            reply = None
                            session.reservations -= 1
                            self.condition.notify_all()
                try:
                    if isinstance(reply.pixels, CpuOutput):
                        pixels, delivery = reply.pixels.read()
                        reply = Reply(dict(reply.metadata, **delivery), pixels, reply.status)
                        pixels = None
                        if session.cancelled:
                            return
                    metadata = reply.metadata
                    if not metadata['source_frames']:
                        release_rgb()
                    if not self._publish(session, Reply(dict(metadata, op='batch'))):
                        return
                    frame_size = session.settings['width'] * session.settings['height'] * 3
                    for index, source in enumerate(metadata['source_frames']):
                        ordinal = metadata['output_frame_start'] + index
                        started = time.perf_counter_ns()
                        payload, key = session.encoder.encode(
                            memoryview(reply.pixels)[index*frame_size:(index+1)*frame_size], ordinal)
                        ended = time.perf_counter_ns()
                        if index + 1 == len(metadata['source_frames']):
                            release_rgb()
                        frame = dict(op='video_frame', session_id=session.session_id,
                            chunk_id=metadata['chunk_id'], output_id=metadata['output_id'], output_frame=ordinal,
                            source_frame=source['source_frame'], capture_ns=source['capture_ns'], source=source,
                            timestamp_us=ordinal*1_000_000//FPS,
                            duration_us=(ordinal+1)*1_000_000//FPS-ordinal*1_000_000//FPS,
                            key_frame=key, encode_start_ns=started, encode_end_ns=ended, encoded_bytes=len(payload))
                        if not self._publish(session, Reply(frame, payload)):
                            return
                    if metadata['status'] == 'complete':
                        session.encoder.flush()
                        self._publish(session, Reply(dict(op='complete', status='complete',
                            session_id=session.session_id, total_frames=metadata['total_frames'])))
                        return
                finally:
                    release_rgb()
        except Exception as exc:
            self.abort(session, f'H264 encoding failed: {exc}')
        finally:
            session.encoder.close()
            with self.condition:
                session.encoder_done.set()
                self.condition.notify_all()
                self._retire(session)
