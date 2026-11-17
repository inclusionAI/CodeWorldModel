import asyncio
import hashlib
import io
import json
import unittest
from unittest.mock import Mock, patch

from aiohttp import ClientSession, WSMsgType, WSServerHandshakeError, web
from aiohttp.test_utils import TestServer
from PIL import Image

from tools.serve_web import bootstrap, main

bootstrap()
from projects.minimax_h3.inference.web_native_game import NativeGameBridge, NativeSource, ZERO, parse_input, validate_proxy_url
from projects.minimax_h3.inference.serving_http import unpack
from projects.minimax_h3.inference.web_serving import browser_access, create_app


class NativeFixture:
    def __init__(self):
        self.state = dict(phase='live', instance_id='native-one', epoch=2, paused=True, frame_id=10)
        self.commands, self.inputs = [], []
        self.reset_reads = 0
        self.reset_reply_gate = None
        self.reset_applied = asyncio.Event()
        self.freeze_reset = False
        self.bad_jpeg = False
        self.bad_snapshot = None
        self.snapshot_packet = None
        self.status_reads = 0
        self.snapshot_reads = 0
        self.publication_ns = 123456789
        self.snapshot_queries = []
        self.preview_started = asyncio.Event()
        self.capture_started = asyncio.Event()
        self.capture_gate = None
        self.freeze_status = False
        self.preview_gate = None
        self.preview_timeout = False
        self.legacy_snapshot_only = False
        self.preview_reads = 0
        self.reset_preview_delay = 0
        self.waiting_phase = True
        self.pause_started = asyncio.Event()
        self.pause_gate = asyncio.Event()
        self.pause_gate.set()
        self.token = 'x' * 32
        image = io.BytesIO()
        Image.new('RGB', (2, 1), (32, 64, 128)).save(image, format='JPEG')
        self.jpeg = image.getvalue()

    async def handle(self, request):
        if request.path == '/':
            return web.Response(text=f"const token='{self.token}'")
        if request.path == '/status':
            self.status_reads += 1
            if self.preview_reads:
                if not self.state['paused']:
                    self.preview_reads -= 1
                if not self.preview_reads:
                    self.state['phase'] = 'live'
            if self.state['phase'] == 'live' and not self.preview_reads and not self.freeze_status:
                self.state['frame_id'] += 1
            if self.reset_reads and not self.freeze_reset:
                self.reset_reads -= 1
                if self.reset_reads == 0:
                    self.state.update(epoch=self.state['epoch'] + 1, paused=False)
                    self.preview_reads = self.reset_preview_delay
                    if self.preview_reads and self.waiting_phase:
                        self.state['phase'] = 'waiting_for_native_frames'
            return web.json_response(self.state)
        if request.path == '/frame-delivery' and self.legacy_snapshot_only:
            raise web.HTTPNotFound()
        if request.path in ('/frame-snapshot', '/frame-delivery'):
            self.snapshot_reads += 1
            self.snapshot_queries.append(dict(request.query))
            if request.query.get('selection') == 'latest':
                self.preview_started.set()
                if request.query.get('wait_ms') != '250':
                    self.capture_started.set()
                    if self.capture_gate is not None:
                        await self.capture_gate.wait()
                if self.preview_gate is not None and request.query.get('wait_ms') == '250':
                    await self.preview_gate.wait()
                if self.preview_timeout or int(request.query['after_frame_id']) >= self.state['frame_id']:
                    await asyncio.sleep(int(request.query.get('wait_ms', 0)) / 1000)
                    return web.Response(status=204)
            if self.snapshot_packet is not None:
                return web.Response(body=self.snapshot_packet, content_type='application/octet-stream')
            if self.state['phase'] != 'live' or self.preview_reads:
                raise web.HTTPServiceUnavailable()
            jpeg = b'bad jpeg' if self.bad_jpeg else self.jpeg
            metadata = dict(schema='cwm-preview-snapshot/1', atomic=True, **self.state,
                            game_frame=self.state['frame_id'], draw_frame=self.state['frame_id'],
                            width=2, height=1, seated_vehicle=False, clock_method='exact_script_sample',
                            clock_frames=[self.state['frame_id'], self.state['frame_id']], clock_bracket_ms=0,
                            published_monotonic_ns=self.publication_ns, jpeg_sha256=hashlib.sha256(jpeg).hexdigest())
            if self.bad_snapshot:
                metadata.update(self.bad_snapshot)
            encoded = json.dumps(metadata).encode()
            return web.Response(body=len(encoded).to_bytes(4, 'big') + encoded + jpeg,
                                content_type='application/octet-stream')
        if request.headers.get('X-Proxy-Token') != self.token:
            raise web.HTTPForbidden()
        value = await request.json()
        if value.get('instance_id') != self.state['instance_id']:
            raise web.HTTPConflict()
        if value.get('epoch') != self.state['epoch']:
            raise web.HTTPConflict()
        if request.path == '/input':
            self.inputs.append(value)
        elif request.path == '/command':
            self.commands.append(value['op'])
            if value['op'] == 'reset':
                if self.reset_reply_gate is None:
                    self.reset_reads = 3
                else:
                    self.state.update(epoch=self.state['epoch'] + 1, paused=False)
                    self.reset_applied.set()
                    await self.reset_reply_gate.wait()
            else:
                if value['op'] == 'pause':
                    self.pause_started.set()
                    await self.pause_gate.wait()
                self.state['paused'] = value['op'] == 'pause'
        else:
            raise web.HTTPNotFound()
        return web.json_response(dict(accepted=True))


class NativeGameTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.native = NativeFixture()
        upstream = web.Application()
        upstream.router.add_route('*', '/{path:.*}', self.native.handle)
        self.upstream = TestServer(upstream)
        await self.upstream.start_server()
        self.bridge = NativeGameBridge(str(self.upstream.make_url('/')))
        app = web.Application(middlewares=[browser_access])
        app.router.add_get('/native_game', self.bridge.socket)
        self.server = TestServer(app)
        await self.server.start_server()
        self.http = ClientSession()
        self.origin = str(self.server.make_url('')).rstrip('/')

    async def asyncTearDown(self):
        self.native.pause_gate.set()
        if self.native.preview_gate:
            self.native.preview_gate.set()
        await self.bridge.close()
        await self.http.close()
        await self.server.close()
        await self.upstream.close()

    async def connect(self):
        return await self.http.ws_connect(self.server.make_url('/native_game'), origin=self.origin)

    async def operation(self, socket, op, request_id=1):
        await socket.send_json(dict(op=op, id=request_id, **({'after_frame_id': 0} if op == 'frame' else {})))
        reply = await socket.receive(timeout=16)
        self.assertEqual(reply.type, WSMsgType.BINARY, reply.data)
        metadata, pixels = unpack(reply.data)
        self.assertEqual(metadata['id'], request_id)
        return metadata, pixels

    async def test_prepare_waits_for_reset_pause_and_new_preview(self):
        socket = await self.connect()
        metadata, jpeg = await self.operation(socket, 'prepare')
        self.assertEqual(jpeg, self.native.jpeg)
        self.assertEqual(self.native.commands[:2], ['reset', 'pause'])
        self.assertEqual(metadata['evidence']['epoch'], 3)
        self.assertTrue(metadata['evidence']['atomic'])
        self.assertGreater(metadata['evidence']['frame_id'], 10)
        self.assertNotIn(self.native.token, str(metadata))
        self.assertTrue(self.native.state['paused'])
        await socket.close()

    async def test_exclusive_owner_origin_and_host(self):
        for headers in ({}, {'Origin': 'http://127.0.0.1:9'}, {'Origin': self.origin, 'Host': 'evil.invalid'}):
            with self.assertRaises(WSServerHandshakeError) as rejected:
                await self.http.ws_connect(self.server.make_url('/native_game'), headers=headers)
            self.assertEqual(rejected.exception.status, 403)
        socket = await self.connect()
        with self.assertRaises(WSServerHandshakeError) as rejected:
            await self.connect()
        self.assertEqual(rejected.exception.status, 409)
        await socket.close()

    async def test_prepare_waits_for_live_preview_after_epoch_changes(self):
        self.native.reset_preview_delay = 4
        socket = await self.connect()
        metadata, jpeg = await self.operation(socket, 'prepare')
        self.assertEqual(jpeg, self.native.jpeg)
        self.assertEqual(metadata['evidence']['epoch'], 3)
        self.assertEqual(self.native.preview_reads, 0)
        self.assertTrue(self.native.state['paused'])
        await socket.close()

    async def test_prepare_waits_for_new_preview_even_if_reset_status_is_live(self):
        self.native.reset_preview_delay = 4
        self.native.waiting_phase = False
        socket = await self.connect()
        metadata, jpeg = await self.operation(socket, 'prepare')
        self.assertEqual(jpeg, self.native.jpeg)
        self.assertEqual(metadata['evidence']['epoch'], 3)
        self.assertEqual(self.native.preview_reads, 0)
        self.assertTrue(self.native.state['paused'])
        await socket.close()

    async def test_prepare_timeout_reports_step_without_token_or_stale_frame(self):
        self.native.reset_preview_delay = 1000
        socket = await self.connect()
        async def keepalive():
            while True:
                await socket.send_json(dict(op='keepalive'))
                await asyncio.sleep(.25)
        alive = asyncio.create_task(keepalive())
        try:
            await socket.send_json(dict(op='prepare', id=1))
            error = await socket.receive_json(timeout=7)
            self.assertIn('post-reset live preview', error['error'])
            self.assertNotIn(self.native.token, str(error))
        finally:
            alive.cancel()
            await asyncio.gather(alive, return_exceptions=True)
            await socket.close()

    async def test_reconnect_waits_for_disconnected_owner_cleanup(self):
        socket = await self.connect()
        await self.operation(socket, 'prepare')
        await self.operation(socket, 'resume', 2)
        self.native.pause_started.clear()
        self.native.pause_gate.clear()
        await socket.close()
        await asyncio.wait_for(self.native.pause_started.wait(), 1)
        old_done = self.bridge.done
        reconnect = asyncio.create_task(self.connect())
        await asyncio.sleep(.1)
        self.assertFalse(reconnect.done(), 'new controller must wait for native cleanup')
        self.native.pause_gate.set()
        again = await asyncio.wait_for(reconnect, 2)
        self.assertTrue(old_done.is_set())
        self.assertTrue(self.native.state['paused'])
        await self.operation(again, 'prepare')
        await again.close()

    async def test_input_heartbeat_and_disconnect_release(self):
        socket = await self.connect()
        await self.operation(socket, 'prepare')
        await self.operation(socket, 'resume', 2)
        await socket.send_json(dict(op='input', value=dict(ZERO, move_ud=-1, buttons=32)))
        await asyncio.sleep(.36)
        held = [value for value in self.native.inputs if value['buttons'] == 32]
        self.assertGreaterEqual(len(held), 2, 'held controls refresh the native 24-frame TTL')
        self.assertTrue(all(value['instance_id'] == 'native-one' for value in held))
        await socket.send_json(dict(op='release'))
        await asyncio.sleep(.15)
        self.assertEqual(self.native.inputs[-1]['buttons'], 0)
        await socket.close()
        await asyncio.wait_for(self.bridge.done.wait(), 3)
        self.assertTrue(self.native.state['paused'])
        self.assertEqual(self.native.inputs[-1], dict(ZERO, instance_id='native-one', epoch=3))
        again = await self.connect()
        await again.close()

    async def test_changed_instance_fails_without_controlling_replacement(self):
        socket = await self.connect()
        await self.operation(socket, 'prepare')
        self.native.state['instance_id'] = 'replacement'
        await socket.send_json(dict(op='resume', id=2))
        error = await socket.receive_json(timeout=3)
        self.assertEqual(error['op'], 'error')
        await socket.close()
        await self.bridge.done.wait()
        self.assertNotIn('resume', self.native.commands)

    async def test_malformed_jpeg_and_input_close_controller(self):
        socket = await self.connect()
        self.native.bad_jpeg = True
        await socket.send_json(dict(op='prepare', id=1))
        self.assertEqual((await socket.receive_json(timeout=3))['op'], 'error')
        await socket.close()
        await self.bridge.done.wait()
        self.native.bad_jpeg = False
        socket = await self.connect()
        await socket.send_json(dict(op='input', value=dict(ZERO, buttons=4096)))
        self.assertEqual((await socket.receive_json(timeout=3))['op'], 'error')
        await socket.close()

    async def test_frame_and_input_heartbeat_do_not_read_status(self):
        socket = await self.connect()
        await self.operation(socket, 'prepare')
        await self.operation(socket, 'resume', 2)
        reads = self.native.status_reads
        snapshots = self.native.snapshot_reads
        await self.operation(socket, 'frame', 3)
        await asyncio.sleep(.23)
        self.assertEqual(self.native.status_reads, reads)
        self.assertEqual(self.native.snapshot_reads, snapshots + 1)
        await socket.close()

    async def test_preview_wait_does_not_hold_control_lock_and_pause_cancels_it(self):
        socket = await self.connect()
        await self.operation(socket, 'prepare')
        await self.operation(socket, 'resume', 2)
        self.native.preview_gate = asyncio.Event()
        await socket.send_json(dict(op='frame', id=3, after_frame_id=0))
        await asyncio.wait_for(self.native.preview_started.wait(), 1)
        self.assertFalse(self.bridge.native.lock.locked())
        await socket.send_json(dict(op='input', value=dict(ZERO, move_ud=-1)))
        await asyncio.sleep(.15)
        self.assertEqual(self.native.inputs[-1]['move_ud'], -1)
        await socket.send_json(dict(op='pause', id=4))
        cancelled, _ = unpack((await socket.receive(timeout=1)).data)
        paused, _ = unpack((await socket.receive(timeout=1)).data)
        self.assertEqual(cancelled, dict(op='result', id=3, cancelled=True))
        self.assertEqual(paused['id'], 4)
        self.assertTrue(paused['evidence']['paused'])
        await socket.close()

    async def test_frame_pause_burst_replies_to_both_requests(self):
        socket = await self.connect()
        await self.operation(socket, 'prepare')
        await self.operation(socket, 'resume', 2)
        await socket.send_json(dict(op='frame', id=3, after_frame_id=0))
        await socket.send_json(dict(op='pause', id=4))
        replies = [unpack((await socket.receive(timeout=2)).data)[0] for _ in range(2)]
        self.assertEqual({reply['id'] for reply in replies}, {3, 4})
        preview = next(reply for reply in replies if reply['id'] == 3)
        self.assertTrue(preview.get('cancelled') or preview.get('evidence'))
        await self.operation(socket, 'resume', 5)
        await socket.close()

    async def test_concurrent_preview_cancel_preserves_one_reply(self):
        socket = await self.connect()
        await self.operation(socket, 'prepare')
        await self.operation(socket, 'resume', 2)
        self.native.preview_gate = asyncio.Event()
        await socket.send_json(dict(op='frame', id=3, after_frame_id=0))
        await asyncio.wait_for(self.native.preview_started.wait(), 1)
        sending, release = asyncio.Event(), asyncio.Event()
        send_bytes = web.WebSocketResponse.send_bytes

        async def delayed_send(server, data, **kwargs):
            metadata, _ = unpack(data)
            if metadata.get('id') == 3 and metadata.get('cancelled'):
                sending.set()
                await release.wait()
            await send_bytes(server, data, **kwargs)

        with patch.object(web.WebSocketResponse, 'send_bytes', delayed_send):
            first = asyncio.create_task(self.bridge.cancel_preview())
            await asyncio.wait_for(sending.wait(), 1)
            second = asyncio.create_task(self.bridge.cancel_preview())
            await asyncio.sleep(0)
            release.set()
            await asyncio.wait_for(asyncio.gather(first, second), 2)
        reply, _ = unpack((await socket.receive(timeout=1)).data)
        self.assertEqual(reply, dict(op='result', id=3, cancelled=True))
        await self.operation(socket, 'pause', 4)
        await self.operation(socket, 'resume', 5)
        await socket.close()

    async def test_preview_timeout_and_separate_capture_progress(self):
        async with ClientSession() as http:
            source = NativeSource(str(self.upstream.make_url('')).rstrip('/'), http)
            prepared, _ = await source.prepare()
            await source.resume()
            self.native.state['frame_id'] = 30
            await source.frame(after_frame_id=0, preview=True)
            self.native.state['frame_id'] = 20
            captured, _ = await source.frame(after_frame_id=prepared['frame_id'])
            self.assertEqual(captured['frame_id'], 20)
            self.assertEqual(source.progress['preview'][0], 30)
            self.assertEqual(source.progress['capture'][0], 20)
            self.native.preview_timeout = True
            self.assertIsNone(await source.frame(after_frame_id=30, preview=True))
            self.assertEqual(self.native.snapshot_queries[-1]['wait_ms'], '250')
            for cursor in (-1, True, 1.5, 2**53):
                with self.assertRaisesRegex(ValueError, 'cursor'):
                    await source.frame(after_frame_id=cursor, preview=True)

    async def test_pause_during_preview_send_emits_only_one_response_per_id(self):
        socket = await self.connect()
        await self.operation(socket, 'prepare')
        await self.operation(socket, 'resume', 2)
        written = asyncio.Event()
        drain = asyncio.Event()
        send_bytes = web.WebSocketResponse.send_bytes

        async def delayed_send(server, data, **kwargs):
            await send_bytes(server, data, **kwargs)
            metadata, _ = unpack(data)
            if metadata.get('id') == 3 and metadata.get('evidence'):
                written.set()
                await drain.wait()

        with patch.object(web.WebSocketResponse, 'send_bytes', delayed_send):
            await socket.send_json(dict(op='frame', id=3, after_frame_id=0))
            await asyncio.wait_for(written.wait(), 1)
            frame, _ = unpack((await socket.receive(timeout=1)).data)
            self.assertEqual(frame['id'], 3)
            paused, _ = await self.operation(socket, 'pause', 4)
            self.assertTrue(paused['evidence']['paused'])
            await self.operation(socket, 'resume', 5)
        self.assertFalse(socket.closed)
        await socket.close()

    async def test_preview_generation_and_duplicate_lane_fail_closed(self):
        socket = await self.connect()
        await self.operation(socket, 'prepare')
        await self.operation(socket, 'resume', 2)
        self.native.preview_gate = asyncio.Event()
        await socket.send_json(dict(op='frame', id=3, after_frame_id=0))
        await asyncio.wait_for(self.native.preview_started.wait(), 1)
        await socket.send_json(dict(op='frame', id=4, after_frame_id=0))
        cancelled, _ = unpack((await socket.receive(timeout=1)).data)
        self.assertTrue(cancelled['cancelled'])
        error = await socket.receive_json(timeout=1)
        self.assertEqual(error['op'], 'error')
        await socket.close()

    async def test_query_snapshot_size_limit_and_no_legacy_fallback(self):
        async with ClientSession() as http:
            source = NativeSource(str(self.upstream.make_url('')).rstrip('/'), http)
            await source.prepare()
            await source.resume()
            self.native.snapshot_packet = b'x' * (4 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(ValueError, 'size limit'):
                await source.frame(after_frame_id=0)
            self.native.snapshot_packet = b'legacy'
            with self.assertRaisesRegex(ValueError, 'metadata size'):
                await source.frame(after_frame_id=0, preview=True)

    async def test_delayed_preview_rejects_changed_generation(self):
        async with ClientSession() as http:
            source = NativeSource(str(self.upstream.make_url('')).rstrip('/'), http)
            await source.prepare()
            await source.resume()
            self.native.preview_gate = asyncio.Event()
            waiting = asyncio.create_task(source.frame(after_frame_id=0, preview=True))
            await asyncio.wait_for(self.native.preview_started.wait(), 1)
            await source.pause()
            self.native.preview_gate.set()
            with self.assertRaisesRegex(ValueError, 'generation changed'):
                await waiting

    async def test_legacy_query_ignoring_server_rejects_delivery_immediately(self):
        self.native.legacy_snapshot_only = True
        async with ClientSession() as http:
            source = NativeSource(str(self.upstream.make_url('')).rstrip('/'), http)
            await source.prepare()
            await source.resume()
            reads = self.native.snapshot_reads
            for preview in (False, True):
                with self.subTest(preview=preview), self.assertRaisesRegex(ValueError, '/frame-delivery returned HTTP 404'):
                    await source.frame(after_frame_id=0, preview=preview)
            self.assertEqual(self.native.snapshot_reads, reads, 'delivery must never fall back to the query-ignoring endpoint')

    async def test_snapshot_rejects_malformed_and_mismatched_provenance(self):
        async with ClientSession() as http:
            source = NativeSource(str(self.upstream.make_url('')).rstrip('/'), http)
            await source.prepare()
            for packet in (b'', b'\x00\x01', b'\xff\xff\xff\xff{}',
                           b'\x00\x00\x00\x02{xjpeg'):
                with self.subTest(packet=packet), self.assertRaises(ValueError):
                    self.native.snapshot_packet = packet
                    await source.frame(paused=True)
            self.native.snapshot_packet = None
            for bad in ({'atomic': False}, {'instance_id': 'replacement'},
                        {'epoch': 99}, {'paused': False}, {'width': 3},
                        {'jpeg_sha256': '0' * 64}, {'frame_id': 0},
                        {'clock_bracket_ms': float('nan')}, {'clock_bracket_ms': 251},
                        {'epoch': 3.0}, {'clock_method': 'latest_status'},
                        {'clock_frames': [999, 999]}, {'clock_frames': [0, 999]}):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    self.native.bad_snapshot = bad
                    await source.frame(paused=True)
            self.native.bad_snapshot = None
            evidence, jpeg = await source.frame(paused=True)
            self.assertEqual(evidence['frame_id'], self.native.state['frame_id'])
            self.assertEqual(jpeg, self.native.jpeg)

    async def test_cancel_during_prepare_does_not_resume(self):
        self.native.freeze_reset = True
        socket = await self.connect()
        await socket.send_json(dict(op='prepare', id=1))
        await asyncio.sleep(.1)
        await socket.close()
        await asyncio.wait_for(self.bridge.done.wait(), 3)
        self.assertNotIn('resume', self.native.commands)
        self.assertEqual(self.native.inputs[-1]['buttons'], 0)

    async def test_cancel_after_applied_reset_pauses_owned_successor(self):
        self.native.reset_reply_gate = asyncio.Event()
        socket = await self.connect()
        await socket.send_json(dict(op='prepare', id=1))
        await asyncio.wait_for(self.native.reset_applied.wait(), 2)
        await socket.close()
        await asyncio.wait_for(self.bridge.done.wait(), 3)
        self.assertEqual(self.native.state['epoch'], 3)
        self.assertTrue(self.native.state['paused'])
        self.assertEqual(self.native.commands[-1], 'pause')
        self.assertEqual(self.native.inputs[-1], dict(ZERO, instance_id='native-one', epoch=3))
        self.native.reset_reply_gate.set()

    async def test_cancel_after_reset_does_not_control_unexpected_epoch(self):
        self.native.reset_reply_gate = asyncio.Event()
        socket = await self.connect()
        await socket.send_json(dict(op='prepare', id=1))
        await asyncio.wait_for(self.native.reset_applied.wait(), 2)
        self.native.state['epoch'] += 1
        commands, inputs = len(self.native.commands), len(self.native.inputs)
        await socket.close()
        await asyncio.wait_for(self.bridge.done.wait(), 3)
        self.assertEqual(len(self.native.commands), commands)
        self.assertEqual(len(self.native.inputs), inputs)
        self.native.reset_reply_gate.set()

    async def test_cancel_after_reset_does_not_control_replacement_instance(self):
        self.native.reset_reply_gate = asyncio.Event()
        socket = await self.connect()
        await socket.send_json(dict(op='prepare', id=1))
        await asyncio.wait_for(self.native.reset_applied.wait(), 2)
        self.native.state['instance_id'] = 'replacement'
        commands, inputs = len(self.native.commands), len(self.native.inputs)
        await socket.close()
        await asyncio.wait_for(self.bridge.done.wait(), 3)
        self.assertEqual(len(self.native.commands), commands)
        self.assertEqual(len(self.native.inputs), inputs)
        self.native.reset_reply_gate.set()

    async def test_stalled_browser_releases_even_when_websocket_stays_open(self):
        socket = await self.connect()
        await self.operation(socket, 'prepare')
        await self.operation(socket, 'resume', 2)
        await socket.send_json(dict(op='input', value=dict(ZERO, move_ud=-1)))
        self.assertEqual((await socket.receive_json(timeout=4))['op'], 'error')
        await socket.close()
        await asyncio.wait_for(self.bridge.done.wait(), 3)
        self.assertTrue(self.native.state['paused'])
        self.assertEqual(self.native.inputs[-1], dict(ZERO, instance_id='native-one', epoch=3))

    async def test_short_action_tap_is_delivered_without_an_input_queue(self):
        socket = await self.connect()
        await self.operation(socket, 'prepare')
        await self.operation(socket, 'resume', 2)
        await socket.send_json(dict(op='input', value=dict(ZERO, buttons=32)))
        await socket.send_json(dict(op='input', value=dict(ZERO)))
        await asyncio.sleep(.25)
        self.assertTrue(any(value['buttons'] == 32 for value in self.native.inputs))
        self.assertEqual(self.native.inputs[-1]['buttons'], 0)
        await socket.close()

    async def test_redirect_and_bounded_response(self):
        async with ClientSession() as http:
            source = NativeSource(str(self.upstream.make_url('')).rstrip('/'), http)
            self.native.token = 'a' * (256 * 1024)
            with self.assertRaisesRegex(ValueError, 'size limit'):
                await source.prepare()
        app = web.Application()
        async def redirect_response(request):
            return web.Response(status=302, headers={'Location': self.origin})
        app.router.add_get('/', redirect_response)
        async with TestServer(app) as redirect, ClientSession() as http:
            source = NativeSource(str(redirect.make_url('')).rstrip('/'), http)
            with self.assertRaisesRegex(ValueError, '302'):
                await source.request('/')

    def test_cli_accepts_explicit_backend_and_local_native_game_without_serving(self):
        with patch('projects.minimax_h3.inference.web_serving.create_app', new=Mock(return_value=object())) as create, \
                patch('aiohttp.web.run_app') as run:
            main(['--backend', 'http://127.0.0.1:9000', '--native-game-proxy-url', self.bridge.url])
        self.assertEqual(create.call_args.kwargs['backend'], 'http://127.0.0.1:9000')
        self.assertEqual(create.call_args.kwargs['native_game_proxy_url'], self.bridge.url)
        self.assertEqual(run.call_count, 1)

    def test_url_and_input_boundaries(self):
        for url in ('https://127.0.0.1:8771', 'http://localhost:8771', 'http://127.0.0.1:8771/path',
                    'http://127.0.0.1:8771?target=x', 'http://u@127.0.0.1:8771', 'http://8.8.8.8:8771'):
            with self.assertRaises(ValueError):
                validate_proxy_url(url)
        for value in (dict(ZERO, move_lr=float('nan')), dict(ZERO, look_ud=2), dict(ZERO, buttons=True), {}):
            with self.assertRaises(ValueError):
                parse_input(value)
        self.assertEqual(validate_proxy_url('http://127.0.0.1:8771/'), 'http://127.0.0.1:8771')


if __name__ == '__main__':
    unittest.main()
