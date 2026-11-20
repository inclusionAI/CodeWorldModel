"""CPU media adapter between the existing browser gateway and native Omni WS."""
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import re
import struct
import time
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
import msgpack

from .media_waits import WAIT_SCHEMA
from .omni_h3_assets import native_bundle_artifact, read_native_bundle
from .serving import PromptSnapshot, SessionError, integer
from .serving_http import MAX_BODY, pack, unpack
from .serving_media import FPS, H264Encoder, MEDIA_SETTINGS

ADAPTER = web.AppKey('omni_h3_media_adapter', object)
PROFILE_OFF = dict(deployment=dict(options=dict(profile=False)))


def bundle_artifact(model_path, artifact_path):
    """Read the explicit source artifact already pinned by the native bundle."""
    manifest = read_native_bundle(model_path)
    artifact_id, artifact, effective = native_bundle_artifact(manifest, artifact_path)
    spec = artifact['encoders']['recipe']
    return artifact_id, dict(recipe=effective, encoder_recipe=spec), manifest['codec']


@dataclass
class NativeInput:
    accepted: asyncio.Future
    eos: bool
    accepted_frames: int
    done: bool = False
    outputs: int = 0


@dataclass
class RawBatch:
    metadata: dict
    rgb: bytes = b''
    terminal: bool = False


class MediaSession:
    def __init__(self, adapter, sid, bitrate):
        self.adapter, self.sid = adapter, sid
        self.spec = adapter.artifact['encoder_recipe']
        self.max_rgb_bytes = 64 * self.spec['width'] * self.spec['height'] * 3
        self.settings = dict(MEDIA_SETTINGS, width=self.spec['width'], height=self.spec['height'],
                             bitrate_bps=bitrate)
        self.http = self.socket = self.encoder = None
        self.identity = None
        self.inputs, self.sources = {}, deque()
        self.next_chunk, self.next_source, self.last_capture = 0, None, -1
        self.input_frames = self.output_frames = self.output_id = self.wire_seq = 0
        self.input_eos = self.native_eos = self.finished = self.stopping = False
        self.native_confirmed = False
        self.raw = asyncio.Queue(maxsize=2)
        self.raw_slots = asyncio.Semaphore(2)
        self.events = asyncio.Queue(maxsize=4)
        self.credit = asyncio.Event()
        self.credit.set()
        self.append_lock, self.close_lock, self.cleanup_lock = asyncio.Lock(), asyncio.Lock(), asyncio.Lock()
        self.receive_task = self.encode_task = self.failure_task = None
        self.attached = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='omni-h264')

    async def cpu(self, function, *args):
        future = asyncio.get_running_loop().run_in_executor(self.executor, function, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # An executing encode owns its buffers until the CPU call returns.
            await future
            raise

    async def send_native(self, message):
        await self.socket.send_bytes(msgpack.packb(message, use_bin_type=True))

    async def read_native(self):
        message = await self.socket.receive(timeout=self.adapter.timeout)
        self.last_native_received_ns = time.perf_counter_ns()
        if message.type != WSMsgType.BINARY:
            raise SessionError('native session disconnected before confirmed close', 502)
        value = msgpack.unpackb(message.data, raw=False)
        if not isinstance(value, dict):
            raise SessionError('native session frame must be a map', 502)
        return value

    async def open(self, seed, prompt, pixels):
        self.encoder = await self.cpu(self.adapter.encoder_factory,
            self.spec['width'], self.spec['height'], self.settings['bitrate_bps'])
        self.http = ClientSession(timeout=ClientTimeout(total=None, sock_connect=self.adapter.timeout))
        self.socket = await self.http.ws_connect(self.adapter.native_uri, compress=0,
            max_msg_size=self.max_rgb_bytes + (64 << 10), heartbeat=None)
        await self.send_native(dict(op='open', params=dict(seed=seed, prompt=prompt,
            first_image=dict(width=self.spec['width'], height=self.spec['height'], rgb=pixels,
                frames=[dict(source_frame_id='first_image', pts_ms=0., duration_ms=1000./FPS)]))))
        answer = await self.read_native()
        if answer.get('op') != 'opened' or not isinstance(answer.get('session'), dict):
            raise SessionError(f'native open failed: {answer.get("message", answer.get("op"))}', 502)
        self.identity = answer['session']
        self.receive_task = asyncio.create_task(self.receive())
        self.encode_task = asyncio.create_task(self.encode())

    def batch(self, cid, sources, received_ns, *, terminal=False):
        accepted = self.inputs[cid].accepted_frames if cid in self.inputs else self.input_frames
        metadata = dict(op='batch', session_id=self.sid, chunk_id=cid,
            output_id=self.output_id, output_frame_start=self.output_frames,
            frame_count=len(sources), source_frames=sources,
            accepted_frames=accepted, total_frames=self.output_frames+len(sources),
            width=self.spec['width'], height=self.spec['height'],
            status='complete' if terminal else 'accepted', native_received_ns=received_ns)
        self.output_id += 1
        self.output_frames += len(sources)
        return metadata

    async def queue_batch(self, batch):
        await self.raw_slots.acquire()
        try:
            await self.raw.put(batch)
        except BaseException:
            self.raw_slots.release()
            raise

    async def append(self, m, pixels):
        async with self.append_lock:
            fields = {'session_id', 'chunk_id', 'first_source_frame', 'capture_ns', 'assembled_ns', 'eos'}
            if set(m) - {'prompt'} != fields or type(m['eos']) is not bool:
                raise SessionError('chunk requires fixed source metadata')
            if 'prompt' in m and PromptSnapshot.parse(m['prompt']).text != self.spec['system_prompt']:
                raise SessionError('native H3 prompt must match the frozen system prompt')
            if self.stopping or self.input_eos:
                raise SessionError('media input is closed', 409)
            cid = integer(m['chunk_id'], 'chunk_id')
            first = integer(m['first_source_frame'], 'first_source_frame')
            assembled = integer(m['assembled_ns'], 'assembled_ns')
            count = 5 if m['eos'] else 17
            stamps = m['capture_ns']
            if (cid != self.next_chunk or self.next_source not in (None, first)
                    or not isinstance(stamps, list) or len(stamps) != count):
                raise SessionError('out-of-order media input')
            for stamp in stamps:
                integer(stamp, 'capture_ns')
            if (stamps[0] < self.last_capture or any(a > b for a, b in zip(stamps, stamps[1:]))
                    or assembled < stamps[-1]):
                raise SessionError('invalid capture timestamps')
            if len(pixels) != count * self.spec['proxy_width'] * self.spec['proxy_height'] * 3:
                raise SessionError('proxy payload differs from artifact geometry')
            if len(self.sources) + count > 256:
                raise SessionError('pending source ledger exceeded its bound', 413)
            async with asyncio.timeout(self.adapter.timeout):
                while sum(not value.done for value in self.inputs.values()) >= 2:
                    self.credit.clear()
                    await self.credit.wait()
                    if self.stopping:
                        raise SessionError('media session is closing', 409)
                if self.stopping:
                    raise SessionError('media session is closing', 409)
                accepted = asyncio.get_running_loop().create_future()
                self.inputs[cid] = NativeInput(accepted, m['eos'], self.input_frames+count)
                frames = []
                for index, capture in enumerate(stamps):
                    ordinal = self.input_frames + index
                    start, end = ordinal * 1000. / FPS, (ordinal + 1) * 1000. / FPS
                    stamp = dict(source_frame_id=str(first + index), pts_ms=start, duration_ms=end-start)
                    frames.append(stamp)
                    self.sources.append((stamp, dict(source_frame=first+index, capture_ns=capture, chunk_id=cid)))
                self.next_chunk += 1
                self.next_source, self.last_capture = first + count, stamps[-1]
                self.input_frames += count
                self.input_eos = m['eos']
                await self.send_native(dict(op='append', chunk=dict(modality='video', format='h3.rgb24',
                    seq=cid, t_start_ms=frames[0]['pts_ms'],
                    duration_ms=frames[-1]['pts_ms']+frames[-1]['duration_ms']-frames[0]['pts_ms'],
                    payload=dict(width=self.spec['proxy_width'], height=self.spec['proxy_height'],
                        frames=frames, rgb=pixels), eos=m['eos'], turn=None)))
                await asyncio.shield(accepted)
            return dict(session_id=self.sid, chunk_id=cid, status='admitted')

    async def receive(self):
        try:
            while True:
                answer = await self.read_native()
                received_ns = self.last_native_received_ns
                op = answer.get('op')
                if op in ('accepted', 'rejected'):
                    seq = answer.get('seq')
                    item = self.inputs.get(seq) if type(seq) is int else None
                    if item is None or item.accepted.done():
                        raise SessionError('native acknowledgment has unknown or duplicate input', 502)
                    if op == 'rejected':
                        raise SessionError(f'native rejected input {seq}: {answer.get("reason", "queue full")}', 502)
                    item.accepted.set_result(None)
                    if item.done:
                        del self.inputs[seq]
                elif op == 'output':
                    chunk = answer.get('chunk')
                    if (not isinstance(chunk, dict) or chunk.get('session_identity') != self.identity
                            or chunk.get('seq') != self.wire_seq):
                        raise SessionError('native output has stale identity or sequence', 502)
                    self.wire_seq += 1
                    seq = chunk.get('input_seq')
                    item = self.inputs.get(seq) if type(seq) is int else None
                    if item is None or item.done:
                        raise SessionError('native output has unknown or completed input', 502)
                    if chunk.get('kind') == 'input_done':
                        item.done = True
                        self.credit.set()
                        if not item.outputs:
                            await self.queue_batch(RawBatch(self.batch(seq, [], received_ns)))
                        if item.accepted.done():
                            del self.inputs[seq]
                    elif chunk.get('kind') == 'data':
                        if chunk.get('modality') != 'video' or chunk.get('format') != 'h3.rgb24':
                            raise SessionError('native output must be H3 RGB video', 502)
                        if chunk.get('eos'):
                            if self.native_eos or not item.eos or chunk.get('payload') is not None:
                                raise SessionError('invalid native output EOS', 502)
                            self.native_eos = True
                        else:
                            if self.native_eos:
                                raise SessionError('native RGB follows EOS', 502)
                            packet = chunk.get('payload')
                            sources, rgb = self.output_packet(packet)
                            item.outputs += 1
                            metadata = self.batch(seq, sources, received_ns)
                            if packet.get('orchestration') is not None:
                                metadata['orchestration'] = packet['orchestration']
                            await self.queue_batch(RawBatch(metadata, rgb))
                    else:
                        raise SessionError('unknown native output kind', 502)
                else:
                    raise SessionError(f'native session failed: {answer.get("message", op)}', 502)
                if self.native_eos and not self.inputs and not self.finished:
                    if self.sources or not self.input_eos:
                        raise SessionError('native EOS left source frames unaccounted', 502)
                    self.finished = True
                    await self.queue_batch(RawBatch(self.batch(self.next_chunk-1, [], received_ns,
                        terminal=True), terminal=True))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.fail(exc)

    def output_packet(self, packet):
        if not isinstance(packet, dict):
            raise SessionError('native RGB packet must be a map', 502)
        frames, rgb = packet.get('frames'), packet.get('rgb')
        if (packet.get('width') != self.spec['width'] or packet.get('height') != self.spec['height']
                or not isinstance(frames, list) or not 1 <= len(frames) <= 64
                or not isinstance(rgb, bytes) or len(rgb) > self.max_rgb_bytes
                or len(rgb) != len(frames)*self.spec['width']*self.spec['height']*3
                or len(frames) > len(self.sources)):
            raise SessionError('invalid native RGB geometry or frame count', 502)
        sources = []
        for stamp in frames:
            expected, source = self.sources.popleft()
            if stamp != expected:
                raise SessionError('native frame provenance differs from submitted source', 502)
            sources.append(source)
        return sources, rgb

    def encode_frame(self, rgb, ordinal):
        started = time.perf_counter_ns()
        payload, key = self.encoder.encode(rgb, ordinal)
        return payload, key, started, time.perf_counter_ns()

    async def publish(self, metadata, pixels=b''):
        started = time.perf_counter_ns()
        await self.events.put((metadata, pixels))
        metadata.update(event_queue_wait_start_ns=started, event_queue_wait_end_ns=time.perf_counter_ns())

    async def encode(self):
        try:
            while True:
                batch = await self.raw.get()
                try:
                    m = batch.metadata
                    await self.publish(m)
                    stride = self.spec['width'] * self.spec['height'] * 3
                    for index, source in enumerate(m['source_frames']):
                        ordinal = m['output_frame_start'] + index
                        payload, key, started, ended = await self.cpu(self.encode_frame,
                            memoryview(batch.rgb)[index*stride:(index+1)*stride], ordinal)
                        frame = dict(op='video_frame', session_id=self.sid, chunk_id=m['chunk_id'],
                            output_id=m['output_id'], output_frame=ordinal, source=source,
                            source_frame=source['source_frame'], capture_ns=source['capture_ns'],
                            timestamp_us=ordinal*1_000_000//FPS,
                            duration_us=(ordinal+1)*1_000_000//FPS-ordinal*1_000_000//FPS,
                            key_frame=key, encoded_bytes=len(payload), encode_start_ns=started,
                            encode_end_ns=ended, native_received_ns=m['native_received_ns'])
                        await self.publish(frame, payload)
                    if batch.terminal:
                        await self.cpu(self.encoder.flush)
                        await self.close_native()
                        await self.publish(dict(op='complete', session_id=self.sid,
                            status='complete', total_frames=self.output_frames))
                        return
                finally:
                    self.raw_slots.release()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.fail(exc)
        finally:
            try:
                if self.encoder is not None:
                    await self.cpu(self.encoder.close)
            finally:
                self.encoder = None
                self.executor.shutdown(wait=False)

    def fail(self, exc):
        if self.failure_task is None and not self.stopping:
            self.failure_task = asyncio.create_task(self.abort(str(exc) or type(exc).__name__))

    async def close_native(self):
        async with self.close_lock:
            if self.native_confirmed:
                return
            if self.receive_task is not None:
                self.receive_task.cancel()
                await asyncio.gather(self.receive_task, return_exceptions=True)
            if self.socket is not None:
                async with asyncio.timeout(self.adapter.timeout):
                    await self.send_native(dict(op='close'))
                    while True:
                        answer = await self.read_native()
                        if answer.get('op') == 'closed':
                            break
                        if answer.get('op') == 'error':
                            raise SessionError(f'native close failed: {answer.get("message")}', 502)
                await self.socket.close()
            self.native_confirmed = True

    async def cleanup(self):
        async with self.cleanup_lock:
            self.stopping = True
            self.credit.set()
            for item in self.inputs.values():
                if not item.accepted.done():
                    item.accepted.set_exception(SessionError('native session was closed', 502))
            close_error = None
            try:
                await self.close_native()
            except Exception as exc:
                close_error = exc
                self.adapter.quarantine = f'native cleanup unconfirmed: {exc}'
            if self.encode_task is not None:
                self.encode_task.cancel()
                await asyncio.gather(self.encode_task, return_exceptions=True)
            elif self.encoder is not None:
                await self.cpu(self.encoder.close)
                self.encoder = None
                self.executor.shutdown(wait=False)
            else:
                self.executor.shutdown(wait=False)
            if self.http is not None:
                await self.http.close()
            self.sources.clear()
            while not self.raw.empty():
                self.raw.get_nowait()
            if self.adapter.active is self and close_error is None:
                self.adapter.last, self.adapter.active = self, None
            if close_error is not None:
                raise SessionError(str(close_error), 503)

    async def abort(self, message):
        try:
            await self.cleanup()
        except Exception as exc:
            message += f'; {exc}'
        while not self.events.empty():
            self.events.get_nowait()
        self.events.put_nowait((dict(op='error', session_id=self.sid, error=message), b''))


class MediaAdapter:
    def __init__(self, native_uri, artifact_id, artifact, *, codec=None, timeout=600,
                 encoder_factory=H264Encoder):
        uri = urlsplit(native_uri)
        if uri.scheme != 'ws' or uri.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('native URI must use a loopback ws endpoint')
        self.native_uri, self.artifact_id, self.artifact = native_uri, artifact_id, artifact
        self.timeout, self.encoder_factory = timeout, encoder_factory
        self.active = self.last = self.quarantine = None
        self.health = dict(status='ready', artifacts={artifact_id: artifact},
            media=dict(MEDIA_SETTINGS), codec=codec or {}, effective_settings=PROFILE_OFF,
            native_transport='omni_session_msgpack', wait_observability=dict(schema=WAIT_SCHEMA, enabled=False))

    def session(self, sid):
        for session in (self.active, self.last):
            if session is not None and session.sid == sid:
                return session
        raise SessionError('unknown media session', 409)

    async def start(self, m, pixels):
        fields = {'session_id', 'artifact_id', 'seed', 'condition_seed', 'prompt', 'video'}
        if set(m) - {'observe_waits'} != fields:
            raise SessionError('start requires session, artifact, seeds, prompt and video')
        if m.get('observe_waits', False) is not False:
            raise SessionError('this adapter does not implement wait observation')
        sid = m['session_id']
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', sid):
            raise SessionError('invalid session_id')
        seed = integer(m['seed'], 'seed')
        if type(m['condition_seed']) is not int or m['condition_seed'] != seed or seed >= 2**63:
            raise SessionError('native H3 requires condition_seed == seed in [0, 2**63)')
        spec = self.artifact['encoder_recipe']
        if m['artifact_id'] != self.artifact_id or m['prompt'] != spec['system_prompt']:
            raise SessionError('artifact or prompt differs from the frozen native recipe')
        video = m['video']
        if not isinstance(video, dict) or set(video) != {'bitrate_bps'}:
            raise SessionError('video requires bitrate_bps')
        bitrate = integer(video['bitrate_bps'], 'bitrate_bps', MEDIA_SETTINGS['bitrate_min_bps'])
        if bitrate > MEDIA_SETTINGS['bitrate_max_bps']:
            raise SessionError('bitrate exceeds server maximum')
        if len(pixels) != spec['width']*spec['height']*3:
            raise SessionError('first image differs from artifact geometry')
        if self.quarantine:
            raise SessionError(self.quarantine, 503)
        if self.active is not None:
            raise SessionError('a media session is active or draining', 409)
        session = self.active = MediaSession(self, sid, bitrate)
        try:
            async with asyncio.timeout(self.timeout):
                await session.open(seed, m['prompt'], pixels)
        except BaseException:
            await asyncio.shield(session.cleanup())
            raise
        return dict(session_id=sid, status='accepted', recipe=self.artifact['recipe'],
            media=session.settings, effective_settings=PROFILE_OFF,
            wait_observability=dict(schema=WAIT_SCHEMA, enabled=False))


@web.middleware
async def errors(request, handler):
    try:
        return await handler(request)
    except SessionError as exc:
        return web.Response(body=pack(dict(error=str(exc))), status=exc.status,
                            content_type='application/octet-stream')
    except asyncio.TimeoutError:
        return web.Response(body=pack(dict(error='native media operation timed out')), status=504)


def create_app(native_uri, artifact_id, artifact, *, codec=None, timeout=600,
               encoder_factory=H264Encoder):
    adapter = MediaAdapter(native_uri, artifact_id, artifact, codec=codec, timeout=timeout,
                           encoder_factory=encoder_factory)
    app = web.Application(middlewares=[errors], client_max_size=MAX_BODY)
    app[ADAPTER] = adapter

    async def health(request):
        return web.Response(body=pack(dict(adapter.health,
            status='quarantined' if adapter.quarantine else 'ready')))

    async def start(request):
        metadata, pixels = unpack(await request.read())
        receipt = await adapter.start(metadata, pixels)
        return web.Response(body=pack(receipt))

    async def append(request):
        session = adapter.session(request.match_info['sid'])
        metadata, pixels = unpack(await request.read())
        if metadata.get('session_id') != session.sid:
            raise SessionError('session identity differs from URL')
        try:
            receipt = await session.append(metadata, pixels)
        except (asyncio.TimeoutError, ConnectionError) as exc:
            await session.abort(str(exc) or 'native append timed out')
            raise SessionError('native append failed', 502) from exc
        return web.Response(body=pack(receipt))

    async def cancel(request):
        session = adapter.session(request.match_info['sid'])
        metadata, pixels = unpack(await request.read())
        if (pixels or set(metadata) - {'reason'} != {'session_id'}
                or metadata['session_id'] != session.sid):
            raise SessionError('invalid media cancellation')
        await asyncio.shield(session.cleanup())
        while not session.events.empty():
            session.events.get_nowait()
        session.events.put_nowait((dict(op='error', session_id=session.sid, error='cancelled'), b''))
        return web.Response(body=pack(dict(session_id=session.sid, status='cancelled')))

    async def events(request):
        session = adapter.session(request.match_info['sid'])
        if session.attached:
            raise SessionError('media event reader is already attached', 409)
        session.attached = True
        response = web.StreamResponse(headers={'Content-Type': 'application/octet-stream'})
        await response.prepare(request)
        try:
            while True:
                async with asyncio.timeout(timeout):
                    metadata, pixels = await session.events.get()
                    body = pack(metadata, pixels)
                    await response.write(struct.pack('!I', len(body)) + body)
                if metadata['op'] in ('complete', 'error'):
                    break
            await response.write_eof()
        finally:
            await asyncio.shield(session.cleanup())
        return response

    async def cleanup(application):
        if adapter.active is not None:
            await adapter.active.cleanup()

    app.router.add_get('/health', health)
    app.router.add_post('/media/session', start)
    app.router.add_post('/media/session/{sid}/chunk', append)
    app.router.add_post('/media/session/{sid}/cancel', cancel)
    app.router.add_get('/media/session/{sid}/events', events)
    app.on_cleanup.append(cleanup)
    return app
