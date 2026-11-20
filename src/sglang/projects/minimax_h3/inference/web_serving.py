"""One browser owns one Ref session. RGB and provenance use the existing wire format."""
import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass
import hmac
import json
import logging
from pathlib import Path
import secrets
from urllib.parse import urlsplit
from uuid import uuid4

from aiohttp import WSMsgType, web

from .raw_recipe import parse_recipe, validate_online_guidance
from .serving import Request, SessionController, SessionError, integer
from .serving_http import MAX_BODY, MAX_HEADER, RefClient, pack, unpack
from ..modeling.ref_recipe import RefRecipe, STATIC_CONDITIONING_MODE

ROOT = Path(__file__).resolve().parents[1] / 'web' / 'playable'
LOCAL_HOSTS = {'127.0.0.1', 'localhost', '::1'}
GATEWAY = web.AppKey('gateway', object)
PUBLIC_ACCESS = web.AppKey('public_access', object)
SESSION_COOKIE = '__Secure-h3_session'
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, repr=False)
class PublicAccess:
    origin: str
    cookie_path: str
    key: bytes
    session_token: str


def read_public_access(public_url, access_key_file):
    if public_url is None and access_key_file is None:
        return None
    if not public_url or not access_key_file:
        raise ValueError('public_url and access_key_file must be configured together')
    try:
        source = urlsplit(public_url)
        if (not isinstance(public_url, str) or source.scheme != 'https' or not source.hostname
                or source.username is not None or source.password is not None
                or source.query or source.fragment or source.netloc.endswith(':')
                or source.geturl() != public_url or not source.path.endswith('/')
                or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in public_url)
                or any(char in public_url for char in ('\\', ';', '%'))
                or any(part in ('.', '..') for part in source.path.split('/'))
                or not 1 <= (443 if source.port is None else source.port) <= 65535):
            raise ValueError('url')
    except (ValueError, TypeError, AttributeError):
        raise ValueError('public_url must be an HTTPS URL with a trailing slash and no credentials, query or fragment') from None
    key = Path(access_key_file).read_bytes().strip()
    if not 32 <= len(key) <= 1024:
        raise ValueError('access key must contain 32 to 1024 bytes')
    authority = source.netloc.lower()
    if source.port == 443:
        authority = authority.rsplit(':', 1)[0]
    return PublicAccess(f'https://{authority}', source.path, key, secrets.token_urlsafe(32))


def login_page(*, failed=False):
    error = '<p role="alert">访问口令错误，请重试。</p>' if failed else ''
    return web.Response(status=401 if failed else 200, content_type='text/html', text=f'''<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>invoker-render · 访问口令</title><style>
*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;background:#0c141f;color:#e5edf5;font:16px/1.6 system-ui,sans-serif}}
main{{width:min(100%,440px);padding:32px;border:1px solid #304052;border-radius:16px;background:#172131}}h1{{margin:0 0 12px;font-size:26px}}p{{color:#a9b9cb}}form{{display:grid;gap:12px;margin-top:24px}}
input,button{{font:inherit;border-radius:8px;padding:12px}}input{{width:100%;color:inherit;background:#0c141f;border:1px solid #40546a}}button{{border:0;background:#76dfc8;color:#12342f;font-weight:650;cursor:pointer}}input:focus-visible,button:focus-visible{{outline:3px solid #76dfc8;outline-offset:3px}}[role=alert]{{color:#ffaaa5}}
</style></head><body>
<main><h1>invoker-render 流式服务</h1><p>请输入管理员提供的访问口令。</p>{error}
<form method="post" action="./login"><label for="password">访问口令</label>
<input id="password" name="password" type="password" autocomplete="current-password" required maxlength="1024">
<button type="submit">进入网页</button></form></main></body></html>''')


async def login(request):
    if request.method == 'GET':
        return login_page()
    if request.content_type != 'application/x-www-form-urlencoded':
        raise web.HTTPUnsupportedMediaType()
    form = await request.clone(client_max_size=4096).post()
    access = request.app[PUBLIC_ACCESS]
    if len(form) != 1 or not hmac.compare_digest(form.get('password', '').encode('utf-8'), access.key):
        return login_page(failed=True)
    response = web.Response(status=303, headers={'Location': './'})
    response.set_cookie(SESSION_COOKIE, access.session_token, path=access.cookie_path,
                        secure=True, httponly=True, samesite='Strict')
    return response


def profiling_status(metadata, *, fixture, source):
    if fixture:
        return dict(status='fixture', source='fixture')
    value = metadata
    for key in ('effective_settings', 'deployment', 'options', 'profile'):
        value = value.get(key) if isinstance(value, dict) else None
    status = ('diagnostic' if value else 'baseline') if type(value) is bool else 'unknown'
    return dict(status=status, source=source)


def source_budget_capability(config):
    if (config.get('media') or {}).get('protocol') != 'h264_annexb_v1':
        return dict(eligible=False, reason='requires_h264')
    if config['mode'] == 'fixture':
        return dict(eligible=True, reason='fixture_h264')
    recipe = config['recipe']
    schedule = recipe.get('current_latent_schedule')
    cwm = recipe.get('protocol') == 'cwm_causal/v1'
    supported = (recipe.get('protocol') in ('ref_fixed_recent_history/v2', 'cwm_causal/v1')
        and recipe.get('conditioning_mode') == 'static_image_qwen_proxy_vae_v1'
        and recipe.get('bootstrap_latents') == (5 if cwm else 9)
        and schedule == [5]
        and config['codec'].get('decoder') == 'causal')
    return dict(eligible=supported, reason=('cwm_chunk5_window6_causal' if supported and cwm else 'fixed_v2_bootstrap9_current5_causal' if supported
                                            else 'unsupported_recipe_or_decoder'))


@web.middleware
async def browser_access(request, handler):
    try:
        host = urlsplit('http://' + request.host)
        access = request.app.get(PUBLIC_ACCESS)
        public_origin = access.origin if access else None
        allowed_host = (host.netloc == urlsplit(public_origin).netloc if public_origin
                        else host.hostname in LOCAL_HOSTS)
        if (not allowed_host or host.username or host.password
                or host.path or host.query or host.fragment or not 1 <= (host.port or 80) <= 65535):
            raise ValueError('host')
        origin = request.headers.get('Origin')
        if public_origin:
            if (origin is not None or request.path in ('/stream', '/control') or request.method == 'POST') and origin != public_origin:
                raise ValueError('origin')
        elif request.path == '/control' and origin != 'http://' + host.netloc:
            raise ValueError('origin')
        elif origin:
            source = urlsplit(origin)
            if (source.scheme != 'http' or source.hostname != host.hostname
                    or (source.port or 80) != (host.port or 80)
                    or source.path or source.query or source.fragment or source.username or source.password):
                raise ValueError('origin')
    except ValueError:
        raise web.HTTPForbidden(text='Request Host or Origin is not allowed')
    if access and request.path != '/login' and not hmac.compare_digest(
            request.cookies.get(SESSION_COOKIE, '').encode('utf-8'), access.session_token.encode('ascii')):
        response = (web.Response(status=303, headers={'Location': 'login'}) if request.path in ('/', '/index.html')
                    else web.Response(status=401, text='Sign in with the application access key'))
    else:
        response = await handler(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Cache-Control'] = 'no-store'
    return response


class FixtureBackend:
    """Synthetic RGB echo served over the real Ref HTTP transport."""
    def __init__(self, *, width=672, height=384):
        recipe = RefRecipe(STATIC_CONDITIONING_MODE, 6, 37,
            dict(schedule=dict(module='common.diffusion.schedule.lerp', class_name='LinearInterpolationSchedule',
                               args=dict(pred_type='x_0', T=1.0)),
                 sampling_timesteps=dict(module='common.diffusion.timestep.sampling.trailing',
                     class_name='TrailingSamplingTimesteps', args=dict(T=1.0, num_sampling_steps=4)),
                 sampler=dict(module='projects.minimax_h3.modeling.sampling', class_name='EulerSampler')),
            dict(mode='plain', drop='proxy', scale=1.0))
        self.spec = dict(height=height, width=width, proxy_height=height, proxy_width=width,
                         system_prompt='Synthetic CPU fixture. No model inference.')
        self.artifacts = {'synthetic-echo': (dict(encoders=dict(recipe=self.spec)), recipe)}
        self.controller = SessionController(self.artifacts)
        self.health = dict(status='ready', artifacts={'synthetic-echo':
            dict(recipe=recipe.to_dict(), encoder_recipe=self.spec)})
        self.pending = b''

    async def handle(self, request):
        if request.method == 'GET':
            return web.Response(body=pack(self.health), content_type='application/octet-stream')
        try:
            metadata, pixels = unpack(await request.read())
            op = 'start' if request.path == '/session' else request.match_info['op']
            if op not in ('start', 'chunk', 'cancel'):
                raise SessionError('unknown fixture operation')
            if op != 'start' and request.match_info['sid'] != metadata.get('session_id'):
                raise SessionError('path and session differ')
            job = Request(op, metadata, pixels)
            self.controller.prepare(job)
            rgb = await self.render(job)
            reply = self.controller.finish(job, rgb, {})
            if reply.metadata['status'] in ('complete', 'cancelled'):
                self.controller.active = None
                self.pending = b''
            return web.Response(body=pack(reply.metadata, reply.pixels), content_type='application/octet-stream')
        except (SessionError, ValueError) as exc:
            return web.Response(status=getattr(exc, 'status', 400), body=pack(dict(error=str(exc))))

    async def render(self, job):
        if job.op != 'chunk':
            return b''
        from PIL import Image
        spec = self.spec
        stride = spec['proxy_height'] * spec['proxy_width'] * 3
        return b''.join(Image.frombytes('RGB', (spec['proxy_width'], spec['proxy_height']),
            job.pixels[i:i+stride]).resize((spec['width'], spec['height'])).tobytes()
            for i in range(0, len(job.pixels), stride))

    async def start(self):
        app = web.Application(client_max_size=MAX_BODY)
        app.router.add_get('/health', self.handle)
        app.router.add_post('/session', self.handle)
        app.router.add_post('/session/{sid}/{op}', self.handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        return 'http://127.0.0.1:' + str(self.runner.addresses[0][1])

    async def close(self):
        await self.runner.cleanup()


class BrowserSession:
    def __init__(self, gateway, socket):
        self.gateway, self.socket = gateway, socket
        self.sid = uuid4().hex
        self.controller = SessionController(gateway.artifacts)
        self.inflight = None
        self.closing = False
        self.attempted = False
        self.complete = False
        self.profiling = profiling_status({}, fixture=gateway.config['mode'] == 'fixture', source='start')

    async def send(self, metadata, pixels=b''):
        if not self.socket.closed:
            await self.socket.send_bytes(pack(metadata, pixels))

    def prepare(self, metadata, pixels):
        metadata = dict(metadata)
        op = metadata.pop('op', None)
        if op == 'start':
            if set(metadata) != {'seed', 'condition_seed'}:
                raise SessionError('start requires exactly seed and condition_seed')
            metadata.update(artifact_id=self.gateway.config['artifact_id'],
                            prompt=self.gateway.config['encoder_recipe']['system_prompt'])
        elif op != 'chunk':
            raise SessionError('expected start or chunk')
        if 'session_id' in metadata:
            raise SessionError('session_id is assigned by the gateway')
        metadata['session_id'] = self.sid
        job = Request(op, metadata, pixels)
        self.controller.prepare(job)
        return job

    async def run(self, job):
        path = '/session' if job.op == 'start' else f'/session/{self.sid}/chunk'
        self.attempted = True
        try:
            try:
                reply, timings = await asyncio.to_thread(self.gateway.client.call, path, job.metadata, job.pixels)
            except SessionError as exc:
                if job.op == 'start' and exc.status == 409:
                    self.attempted = False
                raise
            prompt_receipt = {key: reply.metadata[key] for key in ('applied_prompt', 'submitted_prompt_encoding')
                              if key in reply.metadata}
            expected = self.controller.finish(job, reply.pixels, prompt_receipt).metadata
            keys = ('session_id', 'status', 'recipe', 'artifact_id') if job.op == 'start' else (
                'session_id', 'status', 'chunk_id', 'output_id', 'output_frame_start', 'frame_count',
                'source_frames', 'height', 'width', 'available_after_chunk', 'total_frames', 'accepted_frames')
            actual_fields = {key: reply.metadata.get(key) for key in keys}
            expected_fields = {key: expected.get(key) for key in keys}
            if json.dumps(actual_fields, sort_keys=True) != json.dumps(expected_fields, sort_keys=True):
                raise SessionError('backend output differs from ordered source provenance', 502)
            if job.op == 'start' and reply.pixels:
                raise SessionError('start returned unexpected RGB', 502)
            if job.op == 'start':
                self.profiling = profiling_status(reply.metadata,
                    fixture=self.gateway.config['mode'] == 'fixture', source='start')
            self.complete = reply.metadata['status'] == 'complete'
            if not self.closing:
                forwarded = dict(reply.metadata, op='started' if job.op == 'start' else 'output',
                    profiling=self.profiling, rpc_ms=(timings['receive_ns']-timings['send_ns'])/1e6, credit=1)
                if job.op == 'start':
                    forwarded['source_budget'] = source_budget_capability(dict(self.gateway.config,
                        media=reply.metadata.get('media'),
                        recipe=reply.metadata.get('recipe') if reply.metadata.get('recipe') is not None
                            else self.gateway.config['recipe']))
                await self.send(forwarded, reply.pixels)
                if self.complete:
                    await self.socket.close()
        except Exception as exc:
            if not self.closing:
                self.closing = True
                await self.send(dict(op='error', error=str(exc)))
                await self.socket.close()

    async def cleanup(self):
        self.closing = True
        if self.inflight is not None:
            await self.inflight
        if self.attempted and not self.complete:
            try:
                reply, _ = await asyncio.to_thread(self.gateway.client.call,
                    f'/session/{self.sid}/cancel', dict(session_id=self.sid, reason='browser closed'))
                if reply.metadata.get('status') != 'cancelled' or reply.metadata.get('session_id') != self.sid:
                    raise SessionError('backend did not confirm cancellation')
            except Exception as exc:
                self.gateway.quarantine = f'Backend cleanup uncertain; restart gateway after checking service: {exc}'
                return
        if self.gateway.owner is self:
            self.gateway.owner = None


class Gateway:
    def __init__(self, client, config, artifacts, *, cleanup_timeout=5, native_game=None):
        self.client, self.config, self.artifacts = client, config, artifacts
        self.native_game = native_game
        self.native_game_sources = {native_game.protocol: native_game} if native_game is not None else {}
        self.cleanup_timeout = cleanup_timeout
        self.owner = None
        self.quarantine = None
        self.cleanups = set()

    async def finish_media(self, session):
        await asyncio.shield(session.cleanup_task)
        terminal = session.terminal
        if terminal is not None:
            if self.quarantine:
                terminal = dict(op='error', error=self.quarantine)
            elif terminal['op'] == 'cancelled':
                terminal = dict(session.cancel_receipt or {}, op='cancelled', session_id=session.sid)
            with suppress(Exception):
                if session.control_socket is not None:
                    await asyncio.wait_for(session.control_socket.send_json(terminal), 1)
                else:
                    await asyncio.wait_for(session.send(terminal), 1)
        async def close(socket):
            if socket is not None:
                with suppress(Exception):
                    await asyncio.wait_for(socket.close(), 1)
        await asyncio.gather(*(close(socket) for socket in
            (session.socket, session.control_socket, session.pending_control)))
        LOGGER.warning('%s', json.dumps(dict(event='session_close_first', sid=session.sid,
            **session.first_close, terminal=session.terminal['op'] if session.terminal else None,
            quarantine=bool(self.quarantine), media_close_code=session.socket.close_code,
            control_close_code=session.control_socket.close_code if session.control_socket else None)))

    async def control(self, request):
        session = self.owner
        if (not self.config.get('media', {}).get('control_protocol') or session is None
                or session.closing or session.control_socket is not None or session.pending_control is not None):
            raise web.HTTPConflict(text='Control connection unavailable')
        socket = web.WebSocketResponse(max_msg_size=1024, heartbeat=30)
        session.pending_control = socket
        bound = False
        cancelled = False
        try:
            await socket.prepare(request)
            message = await asyncio.wait_for(socket.receive(), 5)
            if message.type != WSMsgType.TEXT:
                raise SessionError('expected control binding')
            value = json.loads(message.data)
            if not isinstance(value, dict) or set(value) != {'op', 'capability'} or value['op'] != 'bind':
                raise SessionError('invalid control binding')
            session.bind_control(value['capability'], socket)
            bound = True
            session.pending_control = None
            await asyncio.wait_for(socket.send_json(dict(op='bound', protocol='media_control/v1')), 1)
            async for message in socket:
                if session.closing:
                    break
                if message.type != WSMsgType.TEXT:
                    raise SessionError('expected text control command')
                value = json.loads(message.data)
                if value == {'op': 'cancel'}:
                    session.terminal = dict(op='cancelled')
                    cancelled = True
                    session.note_close('control', 'explicit_cancel')
                    session.begin_cleanup()
                    break
                if not isinstance(value, dict) or value.get('op') != 'frame_ack':
                    raise SessionError('invalid control command')
                session.acknowledge(value)
        except (SessionError, ValueError, asyncio.TimeoutError, ConnectionError) as exc:
            if bound and not session.closing:
                session.terminal = dict(op='error', error='Control protocol failed')
                session.note_close('control', 'error', code=socket.close_code, exception=socket.exception() or exc)
                session.begin_cleanup()
        finally:
            if bound:
                if not session.closing:
                    session.note_close('control', 'peer_close', code=socket.close_code, exception=socket.exception())
                session.begin_cleanup()
                if cancelled:
                    await asyncio.shield(session.termination_task)
                else:
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(asyncio.shield(session.termination_task), self.cleanup_timeout)
            else:
                try:
                    with suppress(Exception):
                        await asyncio.wait_for(socket.close(), 1)
                finally:
                    if session.pending_control is socket:
                        session.pending_control = None
        return socket

    async def media_stream(self, request, session):
        socket = session.socket
        split = bool(self.config['media'].get('control_protocol'))
        try:
            await socket.prepare(request)
            if split:
                await socket.send_json(dict(op='control_offer', protocol='media_control/v1',
                                            capability=session.control_capability))
            async for message in socket:
                if session.closing:
                    break
                if message.type == WSMsgType.TEXT:
                    if split:
                        raise SessionError('media controls require the control connection')
                    value = json.loads(message.data)
                    if value == {'op': 'cancel'}:
                        session.terminal = dict(op='cancelled')
                        session.note_close('media', 'explicit_cancel')
                        session.begin_cleanup()
                        break
                    if not isinstance(value, dict) or value.get('op') != 'frame_ack':
                        raise SessionError('invalid media control')
                    session.acknowledge(value)
                    continue
                if message.type != WSMsgType.BINARY:
                    if message.type == WSMsgType.ERROR:
                        raise SessionError('WebSocket message exceeds limits or is malformed')
                    break
                if session.complete or split and session.control_socket is None:
                    raise SessionError('session is closing or control is not bound')
                received_ns = time.perf_counter_ns() if session.observe_waits or session.incremental_proxy else None
                metadata, pixels = unpack(message.data)
                if not session.observe_waits and metadata.get('observe_waits') is True:
                    received_ns = time.perf_counter_ns()
                job = session.prepare_input(metadata, pixels, received_ns=received_ns)
                if job is not None:
                    session.inflight = asyncio.create_task(session.run(job))
        except (SessionError, ValueError, ConnectionError) as exc:
            if not session.closing:
                session.terminal = dict(op='error', error=str(exc))
                session.note_close('media', 'error', code=socket.close_code, exception=socket.exception() or exc)
        finally:
            if not session.closing:
                session.note_close('media', 'peer_close', code=socket.close_code, exception=socket.exception())
            session.begin_cleanup()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(session.termination_task), self.cleanup_timeout)
        return socket

    async def stream(self, request):
        if self.owner is not None or self.quarantine:
            raise web.HTTPConflict(text=self.quarantine or 'Another browser owns the backend session')
        socket = web.WebSocketResponse(max_msg_size=self.config['max_input_bytes'], heartbeat=30)
        if self.config.get('media'):
            from .web_media import MediaBrowserSession
            session = MediaBrowserSession(self, socket)
        else:
            session = BrowserSession(self, socket)
        self.owner = session
        if self.config.get('media'):
            return await self.media_stream(request, session)
        cancelled = False
        try:
            await socket.prepare(request)
            async for message in socket:
                if message.type == WSMsgType.TEXT:
                    if message.data != '{"op":"cancel"}':
                        try:
                            value, _ = unpack(len(message.data.encode()).to_bytes(4, 'big') + message.data.encode())
                        except SessionError:
                            value = None
                        if value != {'op': 'cancel'}:
                            raise SessionError('only cancel may use a text message')
                    session.closing = True
                    cancelled = True
                    break
                if message.type != WSMsgType.BINARY:
                    if message.type == WSMsgType.ERROR:
                        raise SessionError('WebSocket message exceeds limits or is malformed')
                    break
                if session.closing or session.complete:
                    raise SessionError('session is closing')
                if session.inflight is not None and not session.inflight.done():
                    raise SessionError('one backend request is already in flight', 409)
                metadata, pixels = unpack(message.data)
                job = session.prepare(metadata, pixels)
                session.inflight = asyncio.create_task(session.run(job))
        except (SessionError, ValueError) as exc:
            session.closing = True
            await session.send(dict(op='error', error=str(exc)))
        finally:
            session.closing = True
            cleanup = asyncio.create_task(session.cleanup())
            self.cleanups.add(cleanup)
            cleanup.add_done_callback(self.cleanups.discard)
            if cancelled:
                await asyncio.shield(cleanup)
                await session.send(dict(op='error', error=self.quarantine) if self.quarantine else
                                   dict(getattr(session, 'cancel_receipt', None) or {}, op='cancelled', session_id=session.sid))
            await socket.close()
            if not cancelled:
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(cleanup), self.cleanup_timeout)
        return socket


async def create_app(*, backend=None, fixture=False, artifact_id=None, timeout=300,
                     fixture_backend=None, cleanup_timeout=5, public_url=None, access_key_file=None,
                     media_control=True, incremental_proxy=False, native_game_proxy_url=None,
                     threejs_racing=False, ballpoint_breach=False):
    if type(threejs_racing) is not bool:
        raise ValueError('threejs_racing must be boolean')
    racing_root = (ROOT.parent / 'threejs-racing' / 'dist-streaming').resolve()
    if threejs_racing and not (racing_root / 'index.html').is_file():
        raise ValueError('Build racing assets first: npm --prefix web/threejs-racing run build:streaming')
    if type(ballpoint_breach) is not bool:
        raise ValueError('ballpoint_breach must be boolean')
    ballpoint_root = (ROOT.parent / 'ballpoint-breach' / 'dist-streaming').resolve()
    if ballpoint_breach and not (ballpoint_root / 'index.html').is_file():
        raise ValueError('Build Ballpoint Breach assets first: npm --prefix web/ballpoint-breach run build:streaming')
    if type(media_control) is not bool:
        raise ValueError('media_control must be boolean')
    if type(incremental_proxy) is not bool:
        raise ValueError('incremental_proxy must be boolean')
    access = read_public_access(public_url, access_key_file)
    native_game = None
    if native_game_proxy_url is not None:
        from .web_native_game import NativeGameBridge
        if access:
            raise ValueError('Native game requires a loopback-only gateway')
        native_game = NativeGameBridge(native_game_proxy_url)
    if bool(backend) == bool(fixture):
        raise ValueError('choose one explicit backend or synthetic fixture')
    if fixture and fixture_backend is None:
        from .web_media_fixture import MediaFixtureBackend
        echo = MediaFixtureBackend()
    else:
        echo = fixture_backend if fixture else None
    if echo is not None:
        backend = await echo.start()
    try:
        client = RefClient(backend, timeout=timeout)
        health, _ = await asyncio.to_thread(client.call, '/health', method='GET')
        if health.pixels or health.metadata.get('status') != 'ready':
            raise ValueError('backend is not ready')
        entries = health.metadata.get('artifacts')
        if not isinstance(entries, dict) or not entries:
            raise ValueError('backend health has no artifacts')
        if artifact_id is None:
            if len(entries) != 1:
                raise ValueError('backend has multiple artifacts; pass --artifact-id')
            artifact_id = next(iter(entries))
        if artifact_id not in entries:
            raise ValueError('selected artifact is absent from backend health')
        entry = entries[artifact_id]
        recipe, spec = parse_recipe(entry['recipe']), entry['encoder_recipe']
        validate_online_guidance(recipe)
        for key in ('height', 'width', 'proxy_height', 'proxy_width'):
            integer(spec[key], key, 1)
        if not isinstance(spec['system_prompt'], str) or not spec['system_prompt'].strip():
            raise ValueError('artifact needs a frozen system prompt')
        max_input = max(spec['height']*spec['width'], 17*spec['proxy_height']*spec['proxy_width'])*3 + MAX_HEADER + 4
        if max_input > MAX_BODY or 34*spec['height']*spec['width']*3 + MAX_HEADER + 4 > MAX_BODY:
            raise ValueError('artifact geometry exceeds browser RGB memory limits')
        config = dict(mode='fixture' if fixture else 'model', artifact_id=artifact_id,
            profiling=profiling_status(health.metadata, fixture=fixture, source='health'),
            codec={key: health.metadata.get('codec', {}).get(key)
                   for key in ('decoder', 'backend', 'compile', 'cuda_graph')},
            recipe=recipe.to_dict(), encoder_recipe=spec, fps=24, max_capture_frames=34,
            max_message_bytes=MAX_BODY, max_input_bytes=max_input)
        if threejs_racing:
            config['torcs_source'] = dict(url='./threejs-racing/index.html?controlled=1',
                                         style='racing-proxy/v1', training_aligned=False)
        if ballpoint_breach:
            config['ballpoint_source'] = dict(url='./ballpoint-breach/index.html?stream=1',
                                             style='ballpoint-rgb/v1', training_aligned=False)
        if native_game:
            config['native_game_source'] = dict(protocol='native_game_commands/v1', preview='native_game_jpeg/v1', fixture_only=fixture)
        if health.metadata.get('media'):
            media = health.metadata['media']
            if (not isinstance(media, dict) or media.get('protocol') != 'h264_annexb_v1'
                    or media.get('codec') != 'avc1.42C028' or media.get('fps') != 24
                    or type(media.get('frame_window')) is not int or not 1 <= media['frame_window'] <= 8):
                raise ValueError('unsupported backend media configuration')
            for field in ('bitrate_min_bps', 'bitrate_default_bps', 'bitrate_max_bps'):
                integer(media.get(field), field, 1)
            if not media['bitrate_min_bps'] <= media['bitrate_default_bps'] <= media['bitrate_max_bps']:
                raise ValueError('backend media bitrate bounds disagree')
            config['media'] = dict(media, width=spec['width'], height=spec['height'])
            config['media'].pop('control_protocol', None)
            config['media'].pop('proxy_protocol', None)
            if media_control:
                config['media']['control_protocol'] = 'media_control/v1'
            if incremental_proxy:
                config['media']['proxy_protocol'] = 'proxy_frames/v1'
        elif incremental_proxy:
            raise ValueError('incremental proxy requires backend media support')
        if native_game and 'media' not in config:
            raise ValueError('Native game requires backend media support')
        config['source_budget'] = source_budget_capability(config)
        gateway = Gateway(client, config, {artifact_id: (dict(encoders=dict(recipe=spec)), recipe)},
                          cleanup_timeout=cleanup_timeout, native_game=native_game)
        app = web.Application(middlewares=[browser_access], client_max_size=max_input)
        app[GATEWAY] = gateway
        if access:
            app[PUBLIC_ACCESS] = access
            app.router.add_get('/login', login)
            app.router.add_post('/login', login)
        async def configuration(request):
            return web.json_response(gateway.config)
        async def asset(request):
            name = request.match_info.get('asset') or 'index.html'
            if name not in {'index.html', 'app.js', 'style.css', 'stream.mjs', 'charts.mjs', 'game.js', 'native-game-source.mjs', 'torcs-source.mjs', 'ballpoint-source.mjs',
                    'sources.mjs', 'gantt.mjs', 'gantt-replay.mjs', 'media.mjs', 'recording.mjs', 'waits.mjs',
                    'boxing-scene.js', 'boxing-logic.mjs', 'racing-scene.js', 'racing-logic.mjs',
                    'vendor/three-0.180.0/three.module.min.js', 'vendor/three-0.180.0/three.core.min.js',
                    'vendor/three-0.180.0/LICENSE', 'vendor/echarts-6.1.0/echarts.min.js',
                    'vendor/echarts-6.1.0/LICENSE', 'vendor/echarts-6.1.0/NOTICE'} or not (ROOT / name).is_file():
                raise web.HTTPNotFound()
            return web.FileResponse(ROOT / name)
        app.router.add_get('/config', configuration)
        app.router.add_get('/stream', gateway.stream)
        app.router.add_get('/control', gateway.control)
        if native_game:
            app.router.add_get('/native_game', native_game.socket)
        if threejs_racing:
            async def racing_asset(request):
                name = request.match_info['asset']
                target = (racing_root / name).resolve()
                allowed = name in {'index.html', 'palette.json', 'torcs/centerline.json',
                    'torcs/car1-stock1.json', 'torcs/e-track-3.json', 'torcs/semantic_materials.json'}
                allowed = allowed or (name.startswith('assets/') and target.suffix == '.js')
                allowed = allowed or (name.startswith('torcs/textures/') and target.suffix == '.png')
                if (not allowed or not target.is_relative_to(racing_root) or not target.is_file()
                        or any(part.startswith('.') for part in Path(name).parts)):
                    raise web.HTTPNotFound()
                return web.FileResponse(target)
            app.router.add_get('/threejs-racing/{asset:.*}', racing_asset)
        if ballpoint_breach:
            async def ballpoint_asset(request):
                name = request.match_info['asset']
                target = (ballpoint_root / name).resolve()
                allowed = name == 'index.html' or (name.startswith('assets/') and target.suffix in {'.js', '.css', '.woff', '.woff2'})
                if (not allowed or not target.is_relative_to(ballpoint_root) or not target.is_file()
                        or any(part.startswith('.') for part in Path(name).parts)):
                    raise web.HTTPNotFound()
                return web.FileResponse(target)
            app.router.add_get('/ballpoint-breach/{asset:.*}', ballpoint_asset)
        app.router.add_get('/', asset)
        app.router.add_get('/{asset:.*}', asset)
        async def close(app):
            for source in gateway.native_game_sources.values():
                await source.close()
            if gateway.owner is not None:
                if gateway.config.get('media'):
                    gateway.owner.note_close('gateway', 'shutdown')
                    gateway.owner.begin_cleanup()
                else:
                    await gateway.owner.socket.close()
            if gateway.cleanups:
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(asyncio.gather(*gateway.cleanups)), cleanup_timeout)
            if echo is not None:
                await echo.close()
        app.on_shutdown.append(close)
        return app
    except BaseException:
        if echo is not None:
            await echo.close()
        raise
