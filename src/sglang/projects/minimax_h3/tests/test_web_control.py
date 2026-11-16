import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock

from aiohttp import WSServerHandshakeError, WSMsgType, web

from tests import test_web_media as fixtures
from tests.test_web_media import create_app, pack, unpack
from projects.minimax_h3.inference.web_serving import SESSION_COOKIE


class MediaControlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = fixtures.MediaGatewayTests()
        self.fixture.media_control = True
        await self.fixture.asyncSetUp()
        self.gateway = self.fixture.gateway
        self.client = self.fixture.client
        self.url = self.fixture.url

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def data(self):
        socket = await self.client.ws_connect(self.url + '/stream', origin=self.url)
        self.fixture.sockets.append(socket)
        offer = await socket.receive_json(timeout=2)
        self.assertEqual(set(offer), {'op', 'protocol', 'capability'})
        self.assertEqual(offer['protocol'], 'media_control/v1')
        return socket, offer['capability']

    async def bind(self, capability):
        socket = await self.client.ws_connect(self.url + '/control', origin=self.url)
        self.fixture.sockets.append(socket)
        await socket.send_json(dict(op='bind', capability=capability))
        self.assertEqual(await socket.receive_json(timeout=2), dict(op='bound', protocol='media_control/v1'))
        return socket

    def start(self):
        return pack(dict(op='start', seed=1, condition_seed=1, video=dict(bitrate_bps=8000000)), bytes(12))

    async def connected(self):
        data, capability = await self.data()
        control = await self.bind(capability)
        await data.send_bytes(self.start())
        self.assertEqual((await self.fixture.receive(data))[0]['op'], 'started')
        return data, control

    async def cancelled(self, control):
        await control.send_json(dict(op='cancel'))
        self.assertEqual((await control.receive_json(timeout=3))['op'], 'cancelled')
        self.assertIsNone(self.gateway.owner)

    async def test_registration_barrier_suppresses_late_start_and_cancels_once(self):
        self.fixture.start_gate = asyncio.Event()
        data, capability = await self.data()
        control = await self.bind(capability)
        owner = self.gateway.owner
        await data.send_bytes(self.start())
        await asyncio.wait_for(self.fixture.start_received.wait(), 2)
        await control.send_json(dict(op='cancel'))
        async with asyncio.timeout(2):
            while owner.cleanup_task is None:
                await asyncio.sleep(0)
        self.assertTrue(owner.closing)
        self.assertIs(owner.begin_cleanup(), owner.cleanup_task)
        self.assertEqual(self.fixture.cancel_count, 0)
        self.assertIs(self.gateway.owner, owner)
        self.fixture.start_gate.set()
        self.assertEqual((await control.receive_json(timeout=3))['op'], 'cancelled')
        self.assertEqual((await data.receive(timeout=3)).type, WSMsgType.CLOSE)
        self.assertEqual(self.fixture.cancel_count, 1)
        self.assertIsNone(self.gateway.quarantine)

    async def test_scheduled_start_cannot_run_after_release(self):
        data, capability = await self.data()
        await self.bind(capability)
        owner = self.gateway.owner
        metadata, pixels = unpack(self.start())
        job = owner.prepare(metadata, pixels)
        owner.inflight = asyncio.create_task(owner.run(job))
        cleanup = owner.begin_cleanup()
        self.assertIs(cleanup, owner.begin_cleanup())
        await asyncio.wait_for(asyncio.shield(cleanup), 2)
        self.assertFalse(self.fixture.start_received.is_set())
        self.assertEqual(self.fixture.cancel_count, 0)
        self.assertIsNone(self.gateway.owner)

    async def test_registered_start_blocked_on_data_lock_does_not_block_cancel(self):
        data, capability = await self.data()
        control = await self.bind(capability)
        owner = self.gateway.owner
        await owner.send_lock.acquire()
        try:
            await data.send_bytes(self.start())
            async with asyncio.timeout(2):
                while not self.fixture.started or owner.starting:
                    await asyncio.sleep(0)
            self.assertFalse(owner.inflight.done())
            await self.cancelled(control)
            self.assertEqual(self.fixture.cancel_count, 1)
        finally:
            owner.send_lock.release()

    async def test_wrong_old_consumed_capabilities_and_pending_connection_limit(self):
        data, capability = await self.data()
        owner = self.gateway.owner
        pending = await self.client.ws_connect(self.url + '/control', origin=self.url)
        self.fixture.sockets.append(pending)
        with self.assertRaises(WSServerHandshakeError) as caught:
            await self.client.ws_connect(self.url + '/control', origin=self.url)
        self.assertEqual(caught.exception.status, 409)
        close_entered, close_gate = asyncio.Event(), asyncio.Event()
        original_close = owner.pending_control.close
        async def held_close(*args, **kwargs):
            close_entered.set()
            await close_gate.wait()
            return await original_close(*args, **kwargs)
        owner.pending_control.close = held_close
        await pending.send_json(dict(op='bind', capability=owner.sid))
        await asyncio.wait_for(close_entered.wait(), 2)
        with self.assertRaises(WSServerHandshakeError) as caught:
            await self.client.ws_connect(self.url + '/control', origin=self.url)
        self.assertEqual(caught.exception.status, 409, 'rejected socket retains its reservation until close finishes')
        close_gate.set()
        self.assertEqual((await pending.receive(timeout=2)).type, WSMsgType.CLOSE)
        for invalid in ('x' * 43, capability + 'x', 17):
            rejected = await self.client.ws_connect(self.url + '/control', origin=self.url)
            await rejected.send_json(dict(op='bind', capability=invalid))
            self.assertEqual((await rejected.receive(timeout=2)).type, WSMsgType.CLOSE)
            await rejected.close()
            self.assertIs(self.gateway.owner, owner)
            self.assertFalse(owner.closing)
        control = await self.bind(capability)
        self.assertIsNone(owner.control_capability)
        with self.assertRaises(WSServerHandshakeError) as caught:
            await self.client.ws_connect(self.url + '/control', origin=self.url)
        self.assertEqual(caught.exception.status, 409)
        await self.cancelled(control)
        later, fresh = await self.data()
        self.assertNotEqual(fresh, capability)
        rejected = await self.client.ws_connect(self.url + '/control', origin=self.url)
        await rejected.send_json(dict(op='bind', capability=capability))
        self.assertEqual((await rejected.receive(timeout=2)).type, WSMsgType.CLOSE)
        await rejected.close()
        await self.cancelled(await self.bind(fresh))

    async def test_control_origin_is_required_and_exact_on_loopback(self):
        await self.data()
        for origin in (None, 'null', 'http://evil.example', self.url + '/'):
            with self.subTest(origin=origin), self.assertRaises(WSServerHandshakeError) as caught:
                await self.client.ws_connect(self.url + '/control', origin=origin)
            self.assertEqual(caught.exception.status, 403)

    async def test_public_cookie_and_origin_matrix_on_both_routes(self):
        origin = 'https://gateway.example.com'
        with tempfile.TemporaryDirectory() as directory:
            key_path = Path(directory) / 'key'
            key_path.write_text('fixture-access-key-' * 3)
            backend = f'http://{self.gateway.client.host}:{self.gateway.client.port}'
            app = await create_app(backend=backend, public_url=origin + '/proxy/fixture:8765/',
                                   access_key_file=key_path)
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, '127.0.0.1', 0).start()
            url = 'http://127.0.0.1:' + str(runner.addresses[0][1])
            headers = dict(Host='gateway.example.com')
            try:
                for route in ('/stream', '/control'):
                    for rejected in (None, 'null', 'https://evil.example', origin + '/'):
                        with self.assertRaises(WSServerHandshakeError) as caught:
                            await self.client.ws_connect(url + route, headers=headers, origin=rejected)
                        self.assertEqual(caught.exception.status, 403)
                    for cookie in ('', f'{SESSION_COOKIE}=wrong'):
                        with self.assertRaises(WSServerHandshakeError) as caught:
                            await self.client.ws_connect(url + route, headers=dict(headers, Cookie=cookie), origin=origin)
                        self.assertEqual(caught.exception.status, 401)
                async with self.client.post(url + '/login', headers=dict(headers, Origin=origin),
                        data=dict(password=key_path.read_text()), allow_redirects=False) as response:
                    token = response.cookies[SESSION_COOKIE].value
                    self.assertEqual(response.cookies[SESSION_COOKIE]['path'], '/proxy/fixture:8765/')
                signed = dict(headers, Cookie=f'{SESSION_COOKIE}={token}')
                data = await self.client.ws_connect(url + '/stream', headers=signed, origin=origin)
                offer = await data.receive_json(timeout=2)
                intruder = await self.client.ws_connect(url + '/control', headers=signed, origin=origin)
                await intruder.send_json(dict(op='bind', capability='x' * 43))
                self.assertEqual((await intruder.receive(timeout=2)).type, WSMsgType.CLOSE)
                await intruder.close()
                control = await self.client.ws_connect(url + '/control', headers=signed, origin=origin)
                await control.send_json(dict(op='bind', capability=offer['capability']))
                self.assertEqual((await control.receive_json(timeout=2))['op'], 'bound')
                await control.send_json(dict(op='cancel'))
                receipt = await control.receive_json(timeout=2)
                self.assertEqual(receipt['op'], 'cancelled')
                self.assertNotIn(offer['capability'], json.dumps(receipt))
                await control.close()
                await data.close()
            finally:
                await runner.cleanup()

    async def test_unbound_timeout_and_protocol_errors_do_not_terminate_owner(self):
        data, capability = await self.data()
        owner = self.gateway.owner
        for value in (b'binary', 'x' * 1025, dict(op='bind', capability=capability, session_id=owner.sid)):
            rejected = await self.client.ws_connect(self.url + '/control', origin=self.url)
            if isinstance(value, bytes):
                await rejected.send_bytes(value)
            elif isinstance(value, str):
                await rejected.send_str(value)
            else:
                await rejected.send_json(value)
            self.assertEqual((await rejected.receive(timeout=2)).type, WSMsgType.CLOSE)
            await rejected.close()
            self.assertFalse(owner.closing)
            self.assertIs(self.gateway.owner, owner)
        pending = await self.client.ws_connect(self.url + '/control', origin=self.url)
        self.assertEqual((await pending.receive(timeout=6)).type, WSMsgType.CLOSE)
        await pending.close()
        self.assertFalse(owner.closing)
        await self.cancelled(await self.bind(capability))

    async def test_start_before_bind_and_in_band_commands_are_rejected(self):
        data, _ = await self.data()
        await data.send_bytes(self.start())
        self.assertEqual((await self.fixture.receive(data))[0]['op'], 'error')
        self.assertFalse(self.fixture.start_received.is_set())
        data, control = await self.connected()
        await data.send_json(dict(op='cancel'))
        self.assertEqual((await control.receive_json(timeout=3))['op'], 'error')
        self.assertEqual(self.fixture.cancel_count, 1)

    async def test_control_schema_size_binary_and_fifo_rejections(self):
        invalid = [dict(op='cancel', extra=True), dict(op='bind', capability='x' * 43),
                   dict(op='frame_ack', output_frame=True), dict(op='frame_ack', output_frame=4),
                   dict(op='frame_ack', output_frame=0, extra=True), [], 'x' * 1025, b'binary']
        for value in invalid:
            with self.subTest(value=type(value).__name__):
                data, control = await self.connected()
                owner = self.gateway.owner
                if isinstance(value, bytes):
                    await control.send_bytes(value)
                elif isinstance(value, str):
                    await control.send_str(value)
                else:
                    await control.send_json(value)
                async with asyncio.timeout(3):
                    while owner.cleanup_task is None:
                        await asyncio.sleep(0)
                await asyncio.wait_for(asyncio.shield(owner.termination_task), 3)
                self.assertIsNone(self.gateway.owner)
                self.assertTrue(owner.http.closed)
                await data.close()
                await control.close()
        self.assertEqual(self.fixture.cancel_count, len(invalid))

    async def test_duplicate_ack_still_fails_after_one_valid_release(self):
        data, control = await self.connected()
        await data.send_bytes(self.fixture.chunk())
        await self.fixture.receive(data)
        self.fixture.ready.set()
        await self.fixture.receive(data)
        for _ in range(4):
            await self.fixture.receive(data)
        await control.send_json(dict(op='frame_ack', output_frame=0))
        self.assertEqual((await self.fixture.receive(data))[0]['output_frame'], 4)
        await control.send_json(dict(op='frame_ack', output_frame=0))
        self.assertEqual((await control.receive_json(timeout=3))['op'], 'error')
        self.assertEqual(self.fixture.cancel_count, 1)

    async def test_eos_waits_for_final_control_ack_without_cancel(self):
        data, control = await self.connected()
        owner = self.gateway.owner
        await data.send_bytes(self.fixture.chunk(eos=True))
        await self.fixture.receive(data)
        self.fixture.ready.set()
        await self.fixture.receive(data)
        for ordinal in range(4):
            self.assertEqual((await self.fixture.receive(data))[0]['output_frame'], ordinal)
            await control.send_json(dict(op='frame_ack', output_frame=ordinal))
        self.assertEqual((await self.fixture.receive(data))[0]['output_frame'], 4)
        self.assertEqual((await self.fixture.receive(data))[0]['op'], 'complete')
        self.assertFalse(owner.closing)
        self.assertIs(self.gateway.owner, owner)
        await control.send_json(dict(op='frame_ack', output_frame=4))
        self.assertEqual((await data.receive(timeout=3)).type, WSMsgType.CLOSE)
        self.assertEqual(self.fixture.cancel_count, 0)
        self.assertIsNone(self.gateway.owner)
        self.assertEqual((owner.first_close['channel'], owner.first_close['trigger']), ('pump', 'normal_eos'))
        self.assertEqual(owner.first_close['last_acked_frame'], 4)

    async def test_channel_loss_during_eos_drain_releases_without_backend_cancel(self):
        data, control = await self.connected()
        owner = self.gateway.owner
        await data.send_bytes(self.fixture.chunk(eos=True))
        await self.fixture.receive(data)
        self.fixture.ready.set()
        await self.fixture.receive(data)
        for ordinal in range(4):
            await self.fixture.receive(data)
            await control.send_json(dict(op='frame_ack', output_frame=ordinal))
        await self.fixture.receive(data)
        await self.fixture.receive(data)
        await control.close()
        await asyncio.wait_for(asyncio.shield(owner.cleanup_task), 2)
        self.assertEqual((owner.first_close['channel'], owner.first_close['trigger']), ('control', 'peer_close'))
        self.assertEqual(owner.first_close['outstanding'], 1)
        self.assertEqual(owner.first_close['output_frame_head'], 4)
        self.assertIsNone(self.gateway.owner)
        self.assertEqual(self.fixture.cancel_count, 0)

    async def test_cancel_disconnect_and_shutdown_share_cleanup(self):
        data, control = await self.connected()
        owner = self.gateway.owner
        owner.http.close = AsyncMock(wraps=owner.http.close)
        await control.send_json(dict(op='cancel'))
        await asyncio.gather(data.close(), control.close(), self.fixture.runner.shutdown())
        await asyncio.wait_for(asyncio.shield(owner.termination_task), 3)
        owner.http.close.assert_awaited_once()
        self.assertEqual(self.fixture.cancel_count, 1)
        self.assertEqual((owner.first_close['channel'], owner.first_close['trigger']), ('gateway', 'shutdown'))
        self.assertIsNone(self.gateway.owner)
        self.assertFalse(self.gateway.cleanups)

    async def test_first_close_log_is_bounded_and_media_first_is_preserved(self):
        data, control = await self.connected()
        owner = self.gateway.owner
        with self.assertLogs('projects.minimax_h3.inference.web_serving', level='INFO') as logs:
            await data.close()
            await asyncio.wait_for(asyncio.shield(owner.termination_task), 3)
        self.assertEqual((owner.first_close['channel'], owner.first_close['trigger']), ('media', 'peer_close'))
        event = json.loads(logs.records[-1].message)
        self.assertEqual(event['event'], 'session_close_first')
        self.assertEqual(event['sid'], owner.sid)
        self.assertEqual(event['channel'], 'media')
        self.assertEqual(event['trigger'], 'peer_close')
        self.assertNotIn('capability', logs.records[-1].message)
        self.assertEqual(event['source_queue'], 0)
        self.assertEqual(event['frame_queue'], 0)

    async def test_pump_failure_while_data_send_lock_held_cleans_without_deadlock(self):
        data, control = await self.connected()
        owner = self.gateway.owner
        await owner.send_lock.acquire()
        try:
            await self.fixture.events.put((dict(op='error', session_id=owner.sid, error='fixture failure'), b''))
            self.assertEqual((await control.receive_json(timeout=3))['op'], 'error')
            self.assertEqual(self.fixture.cancel_count, 1)
            self.assertIsNone(self.gateway.owner)
            self.assertEqual((owner.first_close['channel'], owner.first_close['trigger']), ('pump', 'error'))
            self.assertEqual(owner.first_close['error_class'], 'SessionError')
        finally:
            owner.send_lock.release()

    async def test_start_409_does_not_cancel_external_owner(self):
        self.fixture.start_status = 409
        data, capability = await self.data()
        control = await self.bind(capability)
        await data.send_bytes(self.start())
        self.assertEqual((await control.receive_json(timeout=3))['op'], 'error')
        self.assertEqual(self.fixture.cancel_count, 0)
        self.assertIsNone(self.gateway.quarantine)
        self.assertIsNone(self.gateway.owner)

    async def test_malformed_start_and_failed_cancel_quarantine(self):
        self.fixture.corrupt_start = True
        data, capability = await self.data()
        control = await self.bind(capability)
        owner = self.gateway.owner
        await data.send_bytes(self.start())
        self.assertEqual((await control.receive_json(timeout=3))['op'], 'error')
        self.assertTrue(self.gateway.quarantine)
        self.assertIs(self.gateway.owner, owner)
        self.assertEqual(self.fixture.cancel_count, 1)
        with self.assertRaises(WSServerHandshakeError) as caught:
            await self.client.ws_connect(self.url + '/stream', origin=self.url)
        self.assertEqual(caught.exception.status, 409)

    async def test_failed_cancel_quarantines_owner(self):
        self.fixture.cancel_status = 500
        data, control = await self.connected()
        owner = self.gateway.owner
        await control.send_json(dict(op='cancel'))
        self.assertEqual((await control.receive_json(timeout=3))['op'], 'error')
        self.assertTrue(self.gateway.quarantine)
        self.assertIs(self.gateway.owner, owner)
        self.assertEqual(self.fixture.cancel_count, 1)

    async def test_start_timeout_keeps_quarantine_and_cleanup_survives_handler_timeout(self):
        self.fixture.start_gate = asyncio.Event()
        data, capability = await self.data()
        control = await self.bind(capability)
        owner = self.gateway.owner
        await data.send_bytes(self.start())
        await asyncio.wait_for(self.fixture.start_received.wait(), 2)
        await control.send_json(dict(op='cancel'))
        async with asyncio.timeout(3):
            while owner.cleanup_task is None:
                await asyncio.sleep(0)
            await asyncio.shield(owner.cleanup_task)
        self.assertTrue(self.gateway.quarantine)
        self.assertIs(self.gateway.owner, owner)
        self.assertTrue(owner.http.closed)
        self.fixture.start_gate.set()

    async def test_mode_config_and_strict_boolean(self):
        async with self.client.get(self.url + '/config') as response:
            config = await response.json()
        self.assertEqual(config['media']['control_protocol'], 'media_control/v1')
        self.assertNotIn('capability', json.dumps(config))
        for value in (0, 1, 'false', None):
            with self.assertRaisesRegex(ValueError, 'media_control must be boolean'):
                await create_app(fixture=True, media_control=value)
