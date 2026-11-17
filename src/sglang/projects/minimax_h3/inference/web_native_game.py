import asyncio
from contextlib import suppress
import hashlib
import hmac
import io
import ipaddress
import json
import logging
import math
import re
import secrets
import time
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
from PIL import Image, ImageOps

from .serving_http import pack
from .serving import SessionError
from .native_game_sampling import Publication


NATIVE_GAME_TRANSFORM = 'pillow-centered-cover-bilinear/v1'
logger = logging.getLogger(__name__)


def source_rgb(jpeg, spec, *, reference=False):
    with Image.open(io.BytesIO(jpeg)) as image:
        proxy = ImageOps.fit(image.convert('RGB'), (spec['proxy_width'], spec['proxy_height']),
                             method=Image.Resampling.BILINEAR)
        if reference:
            proxy = proxy.resize((spec['width'], spec['height']), Image.Resampling.BILINEAR)
        return proxy.tobytes()


def validate_proxy_url(value):
    try:
        url = urlsplit(value)
        if (url.scheme != 'http' or url.hostname != '127.0.0.1' or not url.port
                or url.netloc != f'127.0.0.1:{url.port}' or url.path not in ('', '/')
                or url.query or url.fragment or value != url.geturl()):
            raise ValueError()
    except (TypeError, ValueError, AttributeError):
        raise ValueError('native_game_proxy_url must be http://127.0.0.1:<port>') from None
    return f'http://127.0.0.1:{url.port}'


ZERO = dict(move_lr=0, move_ud=0, look_lr=0, look_ud=0, buttons=0)


def parse_input(value):
    if not isinstance(value, dict) or set(value) != set(ZERO):
        raise ValueError('Invalid native game input fields')
    for field in ('move_lr', 'move_ud', 'look_lr', 'look_ud'):
        if type(value[field]) not in (int, float) or not math.isfinite(value[field]) or abs(value[field]) > 1:
            raise ValueError('Invalid native game input axis')
    if type(value['buttons']) is not int or value['buttons'] < 0 or value['buttons'] & ~2047:
        raise ValueError('Invalid native game buttons')
    return value


class NativeSource:
    def __init__(self, url, http):
        self.url, self.http = url, http
        self.token = None
        self.instance = None
        self.epoch = None
        self.reset_pending = False
        self.live = False
        self.input = dict(ZERO)
        self.progress = {}
        self.generation = 0
        self.step = 'connect'
        self.lock = asyncio.Lock()

    async def request(self, path, body=None, *, params=None):
        endpoint = urlsplit(path).path
        limit = 4 * 1024 * 1024 if endpoint in ('/frame-snapshot', '/frame-delivery') else 256 * 1024
        headers = {'Cache-Control': 'no-store'}
        if body is not None:
            headers['X-Proxy-Token'] = self.token
            body = dict(body, instance_id=self.instance)
            if self.epoch is not None:
                body['epoch'] = self.epoch
        async with self.http.request('POST' if body is not None else 'GET', self.url + path,
                                     json=body, params=params, headers=headers, allow_redirects=False) as response:
            if response.status == 204 and endpoint == '/frame-delivery' and params and params['selection'] == 'latest':
                return None
            if response.status != 200:
                raise ValueError(f'Native {path} returned HTTP {response.status}')
            data = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                data.extend(chunk)
                if len(data) > limit:
                    raise ValueError('Native response exceeds size limit')
        if endpoint in ('/', '/frame-snapshot', '/frame-delivery'):
            return bytes(data)
        return json.loads(data)

    async def status(self, *, new_epoch=False, allow_waiting=False):
        status = await self.request('/status')
        phases = ('live', 'waiting_for_native_frames') if allow_waiting else ('live',)
        if (not isinstance(status, dict) or status.get('phase') not in phases
                or not isinstance(status.get('instance_id'), str) or not 1 <= len(status['instance_id']) <= 128
                or type(status.get('epoch')) is not int or status['epoch'] < 0
                or type(status.get('paused')) is not bool
                or type(status.get('frame_id')) is not int or status['frame_id'] < 0):
            raise ValueError('Native source is unavailable or status is malformed')
        if self.instance is not None and status['instance_id'] != self.instance:
            raise ValueError('Native instance changed; reconnect the native game source')
        if not new_epoch and self.epoch is not None and status['epoch'] != self.epoch:
            raise ValueError('Native epoch changed outside this controller')
        return status

    async def wait_status(self, predicate, *, new_epoch=False, require_live=True):
        async with asyncio.timeout(5):
            while True:
                status = await self.status(new_epoch=new_epoch, allow_waiting=True)
                if (not require_live or status['phase'] == 'live') and predicate(status):
                    return status
                await asyncio.sleep(.05)

    async def release(self):
        self.input = dict(ZERO)
        if self.instance:
            await self.request('/input', self.input)

    async def pause(self, *, cleanup=False):
        self.generation += 1
        self.progress.clear()
        self.step = 'pause release'
        self.live = False
        if cleanup and self.reset_pending:
            status = await self.status(new_epoch=True, allow_waiting=True)
            if status['epoch'] not in (self.epoch, self.epoch + 1):
                raise ValueError('Native epoch changed outside this controller')
            self.epoch = status['epoch']
            self.reset_pending = False
        await self.release()
        await self.status(allow_waiting=True)
        self.step = 'pause command'
        await self.request('/command', dict(op='pause'))
        self.step = 'pause observation'
        return await self.wait_status(lambda status: status['paused'])

    async def wait_frame(self, *, paused):
        async with asyncio.timeout(5):
            while True:
                try:
                    return await self.frame(paused=paused)
                except ValueError as error:
                    if 'HTTP 503' not in str(error):
                        raise
                await asyncio.sleep(.05)

    async def prepare(self):
        self.generation += 1
        self.progress.clear()
        self.step = 'prepare token'
        self.live = False
        page = await self.request('/')
        token = re.search(rb"const token='([A-Za-z0-9_-]{32})'", page)
        if not token:
            raise ValueError('Native controller token is unavailable')
        self.token = token[1].decode('ascii')
        self.step = 'prepare status'
        before = await self.status(allow_waiting=True)
        self.instance, self.epoch = before['instance_id'], before['epoch']
        await self.release()
        self.step = 'reset command'
        self.reset_pending = True
        await self.request('/command', dict(op='reset'))
        self.step = 'reset epoch observation'
        reset = await self.wait_status(lambda status: status['epoch'] > before['epoch'],
                                       new_epoch=True, require_live=False)
        if reset['epoch'] != before['epoch'] + 1:
            raise ValueError('Native epoch changed outside this controller')
        self.epoch = reset['epoch']
        self.reset_pending = False
        self.step = 'post-reset live preview'
        await self.wait_status(lambda status: not status['paused'] and status['frame_id'] > reset['frame_id'])
        await self.pause()
        self.step = 'post-pause preview'
        return await self.wait_frame(paused=True)

    async def resume(self):
        self.generation += 1
        self.progress.clear()
        self.step = 'resume command'
        await self.status()
        await self.request('/command', dict(op='resume'))
        self.step = 'resume observation'
        await self.wait_status(lambda status: not status['paused'])
        self.live = True

    async def frame(self, *, paused=False, after_frame_id=None, preview=False, latest=False, wait_ms=0):
        self.step = 'frame acquisition'
        params = None
        if after_frame_id is not None:
            if type(after_frame_id) is not int or not 0 <= after_frame_id <= 2**53 - 1 or paused:
                raise ValueError('Invalid native snapshot cursor')
            params = dict(instance_id=self.instance, epoch=self.epoch, after_frame_id=after_frame_id,
                          selection='latest' if preview or latest else 'next', wait_ms=250 if preview else wait_ms)
        generation = self.generation
        lane = 'preview' if preview else 'capture'
        if latest and lane not in self.progress:
            self.progress[lane] = (after_frame_id, time.monotonic())
        packet = await self.request('/frame-delivery' if params is not None else '/frame-snapshot', params=params)
        if generation != self.generation:
            raise ValueError('Native snapshot generation changed')
        if packet is None:
            if latest:
                if (await self.status())['paused']:
                    raise ValueError('Native sampling is paused')
                if generation != self.generation:
                    raise ValueError('Native snapshot generation changed')
                if time.monotonic() - self.progress[lane][1] > 2:
                    raise ValueError('Native preview stopped advancing')
            return None
        if len(packet) < 5:
            raise ValueError('Native snapshot is truncated')
        size = int.from_bytes(packet[:4], 'big')
        if not 0 < size <= 65536 or 4 + size >= len(packet):
            raise ValueError('Native snapshot metadata size is invalid')
        try:
            evidence = json.loads(packet[4:4 + size])
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError('Native snapshot metadata is malformed') from error
        jpeg = packet[4 + size:]
        if (not isinstance(evidence, dict) or evidence.get('schema') != 'cwm-preview-snapshot/1'
                or evidence.get('atomic') is not True or evidence.get('phase') != 'live'
                or evidence.get('instance_id') != self.instance or type(evidence.get('epoch')) is not int
                or evidence['epoch'] < 1 or evidence['epoch'] != self.epoch
                or evidence.get('paused') is not paused or type(evidence.get('seated_vehicle')) is not bool
                or any(type(evidence.get(field)) is not int or evidence[field] < 0
                       for field in ('frame_id', 'game_frame', 'draw_frame', 'width', 'height',
                                     'published_monotonic_ns'))
                or evidence['frame_id'] == 0 or not 1 <= evidence['width'] <= 4096
                or not 1 <= evidence['height'] <= 4096
                or evidence.get('clock_method') not in ('exact_script_sample', 'bracketed_native_game_time')
                or not isinstance(evidence.get('clock_frames'), (list, tuple))
                or len(evidence['clock_frames']) != 2
                or any(type(value) is not int or value < 0 for value in evidence['clock_frames'])
                or not evidence['clock_frames'][0] <= evidence['game_frame'] <= evidence['clock_frames'][1]
                or type(evidence.get('clock_bracket_ms')) not in (int, float)
                or not math.isfinite(evidence['clock_bracket_ms']) or not 0 <= evidence['clock_bracket_ms'] <= 250
                or evidence['clock_method'] == 'exact_script_sample' and
                   (evidence['clock_frames'] != [evidence['game_frame']] * 2 or evidence['clock_bracket_ms'] != 0)
                or not isinstance(evidence.get('jpeg_sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', evidence['jpeg_sha256'])):
            raise ValueError('Native snapshot provenance is invalid')
        if hashlib.sha256(jpeg).hexdigest() != evidence['jpeg_sha256']:
            raise ValueError('Native snapshot JPEG hash differs from provenance')
        previous_id, progress = self.progress.get(lane, (None, None))
        if ((previous_id is not None and evidence['frame_id'] < previous_id) or
                after_frame_id is not None and evidence['frame_id'] < after_frame_id or
                (preview or latest) and evidence['frame_id'] <= after_frame_id):
            raise ValueError('Native snapshot frame moved backwards')
        if not paused:
            if evidence['frame_id'] != previous_id:
                progress = time.monotonic()
            elif progress is not None and time.monotonic() - progress > 2:
                raise ValueError('Native preview stopped advancing')
        with Image.open(io.BytesIO(jpeg)) as image:
            if image.format != 'JPEG' or image.size != (evidence['width'], evidence['height']):
                raise ValueError('Native JPEG geometry differs from provenance')
            image.verify()
        if not paused:
            self.progress[lane] = (evidence['frame_id'], progress)
        return dict(evidence, kind='native-jpeg'), jpeg


class NativeGameBridge:
    protocol = 'native_game_commands/v1'
    kind = 'native-jpeg'
    transform = NATIVE_GAME_TRANSFORM
    operations = ('prepare', 'pause', 'resume', 'frame', 'capture')

    def __init__(self, url):
        self.url = validate_proxy_url(url)
        self.owner = None
        self.done = None
        self.native = self.prepared = self.capability = self.media = None
        self.decode_task = None
        self.preview_task = None
        self.preview_started = None
        self.expires = 0

    def create_native(self, http):
        return NativeSource(self.url, http)

    def preview(self, frame):
        return frame

    def publication(self, frame, acquired_ns):
        return Publication(*frame, acquired_ns)

    async def reference_rgb(self, spec):
        return await self.rgb(self.prepared[1], spec, reference=True)

    def prepare_payload(self, frame):
        return {}, self.preview(frame)[1]

    async def release_native(self, native):
        await native.pause(cleanup=True)

    async def refresh_input(self, native, pressed_buttons):
        if native.live:
            native.step = 'input heartbeat'
            value = dict(native.input, buttons=native.input['buttons'] | pressed_buttons)
            native.input = dict(native.input, look_lr=0, look_ud=0)
            await native.request('/input', value)

    def claim_prepared(self, binding, media):
        fields = {'protocol', 'capability', 'instance_id', 'epoch', 'frame_id', 'reference', 'budget'}
        capability = binding.get('capability') if isinstance(binding, dict) else None
        if (self.capability is None or not isinstance(capability, str) or len(capability) != 43
                or not capability.isascii() or not hmac.compare_digest(capability, self.capability)):
            raise SessionError('invalid native game source binding')
        self.capability = None
        if (set(binding) != fields or binding['protocol'] != self.protocol or self.media is not None
                or self.owner is None or self.owner.closed or self.prepared is None
                or time.monotonic() >= self.expires
                or binding['reference'] not in ('prepared', 'upload')
                or binding['budget'] not in ('off', 'bounded40', 'bounded45', 'bounded51')):
            raise SessionError('invalid native game source binding')
        evidence = self.preview(self.prepared)[0]
        if (any(type(binding[k]) is not type(evidence[k]) or binding[k] != evidence[k]
                for k in ('instance_id', 'epoch', 'frame_id')) or self.native.live
                or self.native.instance != evidence['instance_id'] or self.native.epoch != evidence['epoch']):
            raise SessionError('Native game prepared identity changed')
        self.media = media
        return dict(protocol=self.protocol, kind=self.kind,
                    fixture_only=media.gateway.config['mode'] == 'fixture',
                    **{k: evidence[k] for k in ('instance_id', 'epoch', 'frame_id')},
                    reference=binding['reference'], transform=self.transform,
                    capture_clock='browser_performance_now', capture_semantics='sample-request',
                    budget_frames={'off': None, 'bounded40': 40, 'bounded45': 45, 'bounded51': 51}[binding['budget']])

    async def pause_media(self, media, *, cleanup=False):
        if self.media is media:
            if self.native is None:
                raise SessionError('Native game controller disappeared before pause was confirmed')
            self.native.live = False
            await self.cancel_preview()
            async with self.native.lock:
                if cleanup:
                    await self.release_native(self.native)
                else:
                    await self.native.pause()

    async def cancel_preview(self):
        task, started = self.preview_task, self.preview_started
        if task is not None:
            await started.wait()
            if not task.done() and not task.cancelling():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if self.preview_task is task:
                self.preview_task = self.preview_started = None

    async def rgb(self, jpeg, spec, *, reference=False):
        if self.decode_task is not None:
            raise SessionError('Native game transform already pending')
        task = self.decode_task = asyncio.create_task(asyncio.to_thread(source_rgb, jpeg, spec, reference=reference))
        try:
            return await asyncio.shield(task)
        finally:
            # The worker thread must finish before a replacement session can decode.
            try:
                with suppress(asyncio.CancelledError):
                    await task
            finally:
                self.decode_task = None

    async def detach_media(self, media):
        if self.media is not media:
            return
        self.capability = self.prepared = None
        try:
            await self.pause_media(media, cleanup=True)
            if self.decode_task is not None:
                await asyncio.shield(self.decode_task)
        except Exception:
            await self.owner.close()
            raise
        finally:
            self.media = None

    async def close(self):
        if self.owner:
            await self.owner.close()
        if self.done:
            await asyncio.wait_for(asyncio.shield(self.done.wait()), 8)

    async def socket(self, request):
        try:
            peer = ipaddress.ip_address(request.remote)
        except (ValueError, TypeError):
            raise web.HTTPForbidden()
        if not peer.is_loopback or request.headers.get('Origin') != 'http://' + request.host:
            raise web.HTTPForbidden(text='Native game requires an exact localhost Origin and loopback peer')
        if self.owner is not None and self.owner.closed:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.done.wait(), 4)
        if self.owner is not None:
            raise web.HTTPConflict(text='Native game already has a controller')
        socket = web.WebSocketResponse(max_msg_size=2048, heartbeat=5)
        self.owner, self.done = socket, asyncio.Event()
        done = self.done
        tasks = []
        try:
            await socket.prepare(request)
            async with ClientSession(timeout=ClientTimeout(total=2), trust_env=False) as http:
                native = self.create_native(http)
                self.native = native
                queue = asyncio.Queue(maxsize=2)
                last_seen = time.monotonic()
                input_changed = asyncio.Event()
                pressed_buttons = 0

                async def worker():
                    while True:
                        message = await queue.get()
                        op, request_id = message['op'], message['id']
                        if op == 'capture':
                            if self.media is None:
                                raise ValueError('Native game source is not bound')
                            result = await self.media.capture_native_game(message['source_frame'], message['capture_ns'],
                                                                  message['segment'])
                            await socket.send_bytes(pack(dict(op='result', id=request_id, **result)))
                            continue
                        await self.cancel_preview()
                        async with native.lock, asyncio.timeout(12):
                            payload = b''
                            evidence = None
                            binding = None
                            extra = {}
                            if op == 'prepare':
                                if self.media is not None:
                                    raise ValueError('Native game source is bound')
                                self.capability = self.prepared = None
                                self.prepared = await native.prepare()
                                evidence, payload = self.preview(self.prepared)
                                extra, payload = self.prepare_payload(self.prepared)
                                self.capability = secrets.token_urlsafe(32)
                                self.expires = time.monotonic() + 30
                                binding = dict(protocol=self.protocol, capability=self.capability,
                                               **{k: evidence[k] for k in ('instance_id', 'epoch', 'frame_id')})
                            elif op == 'pause':
                                if self.media is None:
                                    self.prepared = self.capability = None
                                await native.pause()
                                evidence, payload = self.preview(await native.wait_frame(paused=True))
                            elif op == 'resume':
                                if self.media is not None and (self.media.closing or self.media.input_closed
                                        or not self.media.started or self.media.native_game_captured == 0):
                                    raise ValueError('Native game media source cannot resume yet')
                                if self.media is None:
                                    self.prepared = self.capability = None
                                await native.resume()
                            await socket.send_bytes(pack(dict(op='result', id=request_id, evidence=evidence, **extra,
                                                              **({'binding': binding} if binding else {})), payload))

                async def preview(message, started):
                    started.set()
                    try:
                        result = await native.frame(after_frame_id=message['after_frame_id'], preview=True)
                    except asyncio.CancelledError:
                        if not socket.closed:
                            await socket.send_bytes(pack(dict(op='result', id=message['id'], cancelled=True)))
                        raise
                    if result is None:
                        await socket.send_bytes(pack(dict(op='result', id=message['id'], unchanged=True)))
                    else:
                        evidence, payload = self.preview(result)
                        await socket.send_bytes(pack(dict(op='result', id=message['id'], evidence=evidence), payload))

                async def heartbeat():
                    nonlocal pressed_buttons
                    while True:
                        with suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(input_changed.wait(), .1)
                        input_changed.clear()
                        if time.monotonic() - last_seen > 2:
                            raise ValueError('Native game browser stopped responding')
                        async with native.lock:
                            if self.prepared is not None and self.media is None and time.monotonic() >= self.expires:
                                self.prepared = self.capability = None
                                await self.release_native(native)
                            await self.refresh_input(native, pressed_buttons)
                            pressed_buttons = 0

                async def guarded(operation, *args):
                    try:
                        await operation(*args)
                    except Exception:
                        logger.exception('Native game source failed during %s', native.step)
                        await socket.send_json(dict(op='error',
                                                    error=f'Native game source unavailable during {native.step}; reset or reconnect'))
                        await socket.close()

                tasks = [asyncio.create_task(guarded(worker)), asyncio.create_task(guarded(heartbeat))]
                try:
                    async for incoming in socket:
                        if incoming.type != WSMsgType.TEXT:
                            break
                        message = json.loads(incoming.data)
                        if not isinstance(message, dict):
                            raise ValueError('Invalid native game message')
                        op = message.get('op')
                        if op == 'input' and set(message) == {'op', 'value'}:
                            value = parse_input(message['value'])
                            native.input = value if native.live else dict(ZERO)
                            pressed_buttons |= native.input['buttons']
                            input_changed.set()
                        elif op == 'release' and set(message) == {'op'}:
                            native.input = dict(ZERO)
                            pressed_buttons = 0
                            input_changed.set()
                        elif op == 'keepalive' and set(message) == {'op'}:
                            pass
                        elif (op in self.operations
                              and set(message) == ({'op', 'id', 'source_frame', 'capture_ns', 'segment'} if op == 'capture'
                                                   else {'op', 'id', 'after_frame_id'} if op == 'frame' else {'op', 'id'})
                              and type(message['id']) is int and 0 <= message['id'] <= 2**53 - 1):
                            if op == 'prepare' and self.media is not None:
                                raise ValueError('Native game source is bound')
                            if op in ('prepare', 'pause', 'resume'):
                                native.generation += 1
                                native.live = False
                                native.input = dict(ZERO)
                                pressed_buttons = 0
                            if op == 'frame':
                                if (native.instance is None or type(message['after_frame_id']) is not int
                                        or not 0 <= message['after_frame_id'] <= 2**53 - 1
                                        or self.preview_task is not None and not self.preview_task.done()):
                                    raise ValueError('Invalid native game preview request')
                                if native.live:
                                    self.preview_started = asyncio.Event()
                                    self.preview_task = asyncio.create_task(guarded(preview, message, self.preview_started))
                                else:
                                    await socket.send_bytes(pack(dict(op='result', id=message['id'], cancelled=True)))
                            else:
                                queue.put_nowait(message)
                        else:
                            raise ValueError('Unsupported native game message')
                        last_seen = time.monotonic()
                finally:
                    async def cleanup():
                        media = self.media
                        if media is not None:
                            media.begin_cleanup()
                        self.capability = self.prepared = None
                        await self.cancel_preview()
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        if media is not None:
                            await asyncio.shield(media.cleanup_task)
                        else:
                            with suppress(Exception):
                                async with asyncio.timeout(5):
                                    async with native.lock:
                                        await self.release_native(native)
                    cleanup_task = asyncio.create_task(cleanup())
                    try:
                        await asyncio.shield(cleanup_task)
                    except asyncio.CancelledError:
                        await cleanup_task
                        raise
        except (ValueError, asyncio.QueueFull):
            with suppress(ConnectionError):
                await socket.send_json(dict(op='error', error='Invalid native game source request'))
        finally:
            await socket.close()
            self.owner = None
            self.native = self.prepared = self.capability = None
            done.set()
        return socket
