import asyncio
import json
import subprocess
import sys
import tracemalloc
import unittest
from unittest.mock import patch

from tests import test_web_media as fixtures
from projects.minimax_h3.inference.serving import SessionError
from projects.minimax_h3.inference.web_media import MediaBrowserSession
from projects.minimax_h3.inference.web_serving import create_app

pack, unpack = fixtures.pack, fixtures.unpack


class ProxyGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = fixtures.MediaGatewayTests()
        self.fixture.incremental_proxy = True
        await self.fixture.asyncSetUp()
        self.gateway = self.fixture.gateway

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def connected(self, observe_waits=None):
        socket = await self.fixture.connect(observe_waits=observe_waits)
        self.assertEqual((await self.fixture.receive(socket))[0]['op'], 'started')
        return socket

    def frame(self, source=0, chunk=0):
        return dict(op='proxy_frame', chunk_id=chunk, source_frame=source, capture_ns=source * 10), bytes([source % 256]) * 12

    def commit(self, first=0, chunk=0, eos=False, **extra):
        count = 5 if eos else 17
        return dict(op='proxy_commit', chunk_id=chunk, first_source_frame=first,
            capture_ns=[i * 10 for i in range(first, first + count)], assembled_ns=(first + count) * 10,
            eos=eos, **extra)

    async def wait_partial(self, owner, count):
        async with asyncio.timeout(1):
            while owner.partial is None or len(owner.partial.capture_ns) != count:
                await asyncio.sleep(0)

    async def stage(self, owner, count=5, first=0, chunk=0):
        for source in range(first, first + count):
            self.assertIsNone(owner.prepare_input(*self.frame(source, chunk)))

    async def test_first_frame_reaches_gateway_before_later_capture_and_without_model_admission(self):
        socket = await self.connected()
        owner = self.gateway.owner
        await socket.send_bytes(pack(*self.frame()))
        await self.wait_partial(owner, 1)
        self.assertEqual(owner.partial.pixels[:12], self.frame()[1])
        self.assertEqual(len(owner.partial.pixels), 17 * 12)
        self.assertEqual((owner.next_chunk, owner.next_source, len(owner.sources)), (0, None, 0))
        self.assertEqual(self.fixture.chunks, [])
        self.assertTrue(owner.inflight.done())
        self.assertIsNone(owner.proxy_assembly)

    async def test_17_then_5_matches_legacy_requests_and_backend_bytes(self):
        socket = await self.connected()
        owner = self.gateway.owner
        legacy = MediaBrowserSession(self.gateway, owner.socket)
        legacy.sid = owner.sid
        legacy.attempted = True
        try:
            for chunk, first, count, eos in ((0, 0, 17, False), (1, 17, 5, True)):
                prompt = dict(revision=chunk + 1, text=f'Prompt {chunk}', emitted_ns=first)
                metadata = self.commit(first, chunk, eos, prompt=prompt)
                payload = b''.join(self.frame(source, chunk)[1] for source in range(first, first + count))
                baseline = legacy.prepare(dict(metadata, op='chunk'), payload)
                for source in range(first, first + count):
                    await socket.send_bytes(pack(*self.frame(source, chunk)))
                await self.wait_partial(owner, count)
                self.assertEqual(len(self.fixture.chunks), chunk)
                await socket.send_bytes(pack(metadata))
                admitted, _ = await self.fixture.receive(socket)
                self.assertEqual((admitted['op'], admitted['chunk_id']), ('admitted', chunk))
                self.assertNotIn('waits', admitted)
                self.assertEqual(self.fixture.chunks[-1], (baseline.metadata, baseline.pixels))
                self.assertIsNone(owner.partial)
            self.assertEqual(owner.next_source, 22)
            self.assertTrue(owner.input_closed)
            with self.assertRaises(SessionError):
                owner.prepare_input(*self.frame(22, 2))
        finally:
            await legacy.http.close()

    async def test_strict_frame_validation_does_not_mutate_staging_or_committed_state(self):
        await self.connected()
        owner = self.gateway.owner
        metadata, pixels = self.frame()
        invalid = [(dict(metadata, extra=0), pixels), (dict(metadata, op='chunk'), pixels),
            (dict(metadata, chunk_id=1), pixels), (dict(metadata, chunk_id=True), pixels),
            (dict(metadata, source_frame=-1), pixels), (dict(metadata, source_frame=False), pixels),
            (dict(metadata, capture_ns=True), pixels), (dict(metadata, capture_ns=0.5), pixels),
            (dict(metadata, capture_ns=-1), pixels), (metadata, pixels[:-1]), (metadata, pixels + b'x')]
        for fields, data in invalid:
            with self.subTest(fields=fields, size=len(data)), self.assertRaises(SessionError):
                owner.prepare_input(fields, data)
            self.assertIsNone(owner.partial)
        await self.stage(owner, 1)
        original = owner.partial
        for fields, data in ((self.frame()[0], pixels), (self.frame(2)[0], pixels),
                (dict(self.frame(1)[0], chunk_id=1), pixels), (dict(self.frame(1)[0], capture_ns=-1), pixels)):
            with self.assertRaises(SessionError):
                owner.prepare_input(fields, data)
        self.assertIs(owner.partial, original)
        self.assertEqual(original.capture_ns, [0])
        await self.stage(owner, 16, first=1)
        with self.assertRaises(SessionError):
            owner.prepare_input(*self.frame(17))
        self.assertEqual(len(original.pixels), 204)
        self.assertEqual(len(original.capture_ns), 17)
        self.assertEqual((owner.next_chunk, owner.next_source, len(owner.sources)), (0, None, 0))

    async def test_commit_requires_exact_provenance_count_fields_and_zero_payload(self):
        await self.connected()
        owner = self.gateway.owner
        with self.assertRaises(SessionError):
            owner.prepare_input(self.commit(eos=True), b'')
        await self.stage(owner)
        good = self.commit(eos=True)
        invalid = [dict(good, chunk_id=1), dict(good, chunk_id=False), dict(good, first_source_frame=1),
            dict(good, first_source_frame=False), dict(good, capture_ns=[0, 10, 20, 30, 41]),
            dict(good, capture_ns=[False, 10, 20, 30, 40]), dict(good, assembled_ns=39),
            dict(good, assembled_ns=True), dict(good, eos=1), dict(good, eos=False),
            dict(good, capture_ns=good['capture_ns'][:-1]), dict(good, extra=1),
            dict(good, prompt={'revision': 1}), dict(good, prompt={'revision': True, 'text': 'x', 'emitted_ns': 0})]
        for metadata in invalid:
            with self.subTest(metadata=metadata), self.assertRaises(SessionError):
                owner.prepare_input(metadata, b'')
            self.assertEqual((owner.next_chunk, owner.next_source, len(owner.sources)), (0, None, 0))
        with self.assertRaises(SessionError):
            owner.prepare_input(good, b'x')
        timer = owner.partial.timer
        job = owner.prepare_input(good, b'', received_ns=12345)
        self.assertEqual(job.received_ns, 12345)
        self.assertEqual(job.op, 'chunk')
        self.assertTrue(timer.cancelled())
        self.assertIsNone(owner.partial)

    async def test_registration_and_admission_reject_frames_until_receipt_delivered(self):
        self.fixture.start_gate = asyncio.Event()
        socket = await self.fixture.connect()
        await asyncio.wait_for(self.fixture.start_received.wait(), 1)
        owner = self.gateway.owner
        self.assertTrue(owner.attempted)
        self.assertFalse(owner.started)
        with self.assertRaises(SessionError):
            owner.prepare_input(*self.frame())
        self.fixture.start_gate.set()
        await self.fixture.receive(socket)
        await owner.send_lock.acquire()
        try:
            await self.stage(owner, 17)
            await socket.send_bytes(pack(self.commit()))
            async with asyncio.timeout(1):
                while not self.fixture.chunks:
                    await asyncio.sleep(0)
            with self.assertRaises(SessionError):
                owner.prepare_input(*self.frame(17, 1))
        finally:
            owner.send_lock.release()
        self.assertEqual((await self.fixture.receive(socket))[0]['op'], 'admitted')
        owner.prepare_input(*self.frame(17, 1))

    async def test_waits_use_pre_unpack_commit_dispatch_and_separate_assembly_interval(self):
        socket = await self.connected(observe_waits=True)
        owner = self.gateway.owner
        for source in range(5):
            await socket.send_bytes(pack(*self.frame(source)))
        await self.wait_partial(owner, 5)
        first = owner.partial.received_ns
        import projects.minimax_h3.inference.web_serving as serving
        dispatch = []
        def unpack_at_boundary(body):
            import time
            dispatch.append(time.perf_counter_ns())
            return unpack(body)
        with patch.object(serving, 'unpack', side_effect=unpack_at_boundary):
            await socket.send_bytes(pack(self.commit(eos=True)))
            admitted, _ = await self.fixture.receive(socket)
        waits = admitted['waits']['gateway']
        self.assertEqual(waits['proxy_assembly']['start_ns'], first)
        self.assertEqual(waits['proxy_assembly']['end_ns'], waits['request_complete_ns'])
        self.assertLessEqual(waits['request_complete_ns'], dispatch[0])
        self.assertLessEqual(dispatch[0], waits['rpc_start_ns'])

    async def test_observed_start_keeps_legacy_after_unpack_boundary(self):
        import projects.minimax_h3.inference.web_serving as serving
        import time
        dispatch = []
        def unpack_at_boundary(body):
            result = unpack(body)
            dispatch.append(time.perf_counter_ns())
            return result
        with patch.object(serving, 'unpack', side_effect=unpack_at_boundary):
            socket = await self.fixture.connect(observe_waits=True)
            started, _ = await self.fixture.receive(socket)
        waits = started['waits']['gateway']
        self.assertGreaterEqual(waits['request_complete_ns'], dispatch[0])
        self.assertLessEqual(waits['request_complete_ns'], waits['rpc_start_ns'])

    async def test_unobserved_legacy_route_does_not_add_dispatch_timestamps(self):
        import projects.minimax_h3.inference.web_serving as serving
        self.gateway.config['media'].pop('proxy_protocol')
        with patch.object(serving, 'time') as clock:
            socket = await self.connected()
            await socket.send_bytes(self.fixture.chunk(eos=True))
            admitted, _ = await self.fixture.receive(socket)
            clock.perf_counter_ns.assert_not_called()
        self.assertNotIn('waits', admitted)

    async def test_registration_success_waits_for_started_receipt_before_accepting_frames(self):
        socket = await self.fixture.client.ws_connect(self.fixture.url + '/stream', origin=self.fixture.url)
        self.fixture.sockets.append(socket)
        owner = self.gateway.owner
        await owner.send_lock.acquire()
        try:
            await socket.send_bytes(pack(dict(op='start', seed=1, condition_seed=1,
                video=dict(bitrate_bps=8000000)), bytes(12)))
            async with asyncio.timeout(1):
                while not self.fixture.started or owner.starting:
                    await asyncio.sleep(0)
            self.assertFalse(owner.started)
            with self.assertRaises(SessionError):
                owner.prepare_input(*self.frame())
        finally:
            owner.send_lock.release()
        await self.fixture.receive(socket)
        self.assertIsNone(owner.prepare_input(*self.frame()))

    async def test_absolute_deadline_expires_despite_dribble_and_cleans_once(self):
        socket = await self.connected(observe_waits=True)
        owner = self.gateway.owner
        self.gateway.client.timeout = .15
        await socket.send_bytes(pack(*self.frame()))
        await self.wait_partial(owner, 1)
        partial = owner.partial
        deadline = partial.timer.when()
        for source in (1, 2, 3):
            await asyncio.sleep(.025)
            await socket.send_bytes(pack(*self.frame(source)))
            await self.wait_partial(owner, source + 1)
            self.assertEqual(owner.partial.timer.when(), deadline)
        error, _ = await self.fixture.receive(socket)
        self.assertEqual(error['op'], 'error')
        self.assertIn('deadline', error['error'])
        self.assertEqual(error['waits']['gateway']['proxy_assembly']['outcome'], 'timeout')
        self.assertEqual(partial.pixels, b'')
        self.assertEqual(self.fixture.chunks, [])
        self.assertEqual(self.fixture.cancel_count, 1)
        self.assertIsNone(self.gateway.owner)

    async def test_cleanup_releases_pixels_immediately_and_stale_timer_cannot_touch_new_owner(self):
        socket = await self.connected()
        owner = self.gateway.owner
        await self.stage(owner, 1)
        partial = owner.partial
        cleanup = owner.begin_cleanup()
        self.assertIsNone(owner.partial)
        self.assertFalse(partial.pixels)
        self.assertTrue(partial.timer.cancelled())
        with self.assertRaises(SessionError):
            owner.prepare_input(*self.frame(1))
        await cleanup
        await owner.termination_task
        self.assertTrue(socket.closed or (await socket.receive()).type in (fixtures.WSMsgType.CLOSE, fixtures.WSMsgType.CLOSED))
        await self.connected()
        fresh = self.gateway.owner
        await self.stage(fresh, 1)
        owner.expire_partial(partial)
        self.assertIs(self.gateway.owner, fresh)
        self.assertFalse(fresh.closing)
        self.assertEqual(len(fresh.partial.capture_ns), 1)
        self.assertEqual(self.fixture.cancel_count, 1)

    async def test_malformed_frame_on_wire_cleans_without_model_admission(self):
        for observed in (False, True):
            with self.subTest(observed=observed):
                socket = await self.connected(observe_waits=observed)
                await socket.send_bytes(pack(*self.frame()))
                owner = self.gateway.owner
                await self.wait_partial(owner, 1)
                partial = owner.partial
                await socket.send_bytes(pack(*self.frame(2)))
                error, _ = await self.fixture.receive(socket)
                self.assertEqual(error['op'], 'error')
                self.assertIn('out-of-order', error['error'])
                if observed:
                    span = error['waits']['gateway']['proxy_assembly']
                    self.assertEqual(span['start_ns'], partial.received_ns)
                    self.assertGreaterEqual(span['end_ns'], span['start_ns'])
                    self.assertEqual(span['outcome'], 'error')
                else:
                    self.assertNotIn('waits', error)
                self.assertEqual(self.fixture.chunks, [])
                self.assertFalse(partial.pixels)
                self.assertTrue(partial.timer.cancelled())
                await owner.termination_task
                self.assertIsNone(self.gateway.owner)
        self.assertEqual(self.fixture.cancel_count, 2)

    async def test_staged_protocol_errors_deliver_waits_on_control_channel(self):
        self.gateway.config['media']['control_protocol'] = 'media_control/v1'
        for observed in (False, True):
            for invalid_channel in ('data', 'control'):
                with self.subTest(observed=observed, invalid_channel=invalid_channel):
                    socket = await self.fixture.client.ws_connect(self.fixture.url + '/stream', origin=self.fixture.url)
                    self.fixture.sockets.append(socket)
                    offer = await socket.receive_json()
                    control = await self.fixture.client.ws_connect(self.fixture.url + '/control', origin=self.fixture.url)
                    self.fixture.sockets.append(control)
                    await control.send_json(dict(op='bind', capability=offer['capability']))
                    self.assertEqual((await control.receive_json())['op'], 'bound')
                    await socket.send_bytes(pack(dict(op='start', seed=1, condition_seed=1,
                        video=dict(bitrate_bps=8000000), observe_waits=observed), bytes(12)))
                    await self.fixture.receive(socket)
                    owner = self.gateway.owner
                    await socket.send_bytes(pack(*self.frame()))
                    await self.wait_partial(owner, 1)
                    partial = owner.partial
                    if invalid_channel == 'data':
                        await socket.send_bytes(pack(*self.frame(2)))
                    else:
                        await control.send_json(dict(op='invalid'))
                    error = await control.receive_json(timeout=2)
                    self.assertEqual(error['op'], 'error')
                    if observed:
                        span = error['waits']['gateway']['proxy_assembly']
                        self.assertEqual(span['start_ns'], partial.received_ns)
                        self.assertGreaterEqual(span['end_ns'], span['start_ns'])
                        self.assertEqual(span['outcome'], 'error')
                    else:
                        self.assertNotIn('waits', error)
                    self.assertFalse(partial.pixels)
                    self.assertTrue(partial.timer.cancelled())
                    await owner.termination_task
                    self.assertIsNone(self.gateway.owner)
        self.assertEqual(self.fixture.cancel_count, 4)
        self.assertEqual(self.fixture.chunks, [])

    async def test_legacy_mode_rejects_incremental_messages(self):
        self.gateway.config['media'].pop('proxy_protocol')
        socket = await self.connected()
        await socket.send_bytes(pack(*self.frame()))
        self.assertEqual((await self.fixture.receive(socket))[0]['op'], 'error')
        self.assertEqual(self.fixture.chunks, [])

    async def test_control_disconnect_releases_staging_and_cancels_once(self):
        self.gateway.config['media']['control_protocol'] = 'media_control/v1'
        socket = await self.fixture.client.ws_connect(self.fixture.url + '/stream', origin=self.fixture.url)
        self.fixture.sockets.append(socket)
        offer = await socket.receive_json()
        control = await self.fixture.client.ws_connect(self.fixture.url + '/control', origin=self.fixture.url)
        self.fixture.sockets.append(control)
        await control.send_json(dict(op='bind', capability=offer['capability']))
        await control.receive_json()
        await socket.send_bytes(pack(dict(op='start', seed=1, condition_seed=1,
            video=dict(bitrate_bps=8000000)), bytes(12)))
        await self.fixture.receive(socket)
        owner = self.gateway.owner
        await socket.send_bytes(pack(*self.frame()))
        await self.wait_partial(owner, 1)
        partial = owner.partial
        await control.close()
        async with asyncio.timeout(2):
            while owner.cleanup_task is None:
                await asyncio.sleep(0)
            await owner.cleanup_task
        self.assertFalse(partial.pixels)
        self.assertTrue(partial.timer.cancelled())
        self.assertIsNone(self.gateway.owner)
        self.assertEqual(self.fixture.cancel_count, 1)
        self.assertEqual(self.fixture.chunks, [])

    async def test_disconnect_and_shutdown_release_staging_without_admission(self):
        for ending in ('disconnect', 'shutdown'):
            with self.subTest(ending=ending):
                socket = await self.connected()
                owner = self.gateway.owner
                await socket.send_bytes(pack(*self.frame()))
                await self.wait_partial(owner, 1)
                partial = owner.partial
                if ending == 'disconnect':
                    await socket.close()
                else:
                    await self.fixture.runner.shutdown()
                async with asyncio.timeout(2):
                    while owner.cleanup_task is None:
                        await asyncio.sleep(0)
                    await owner.cleanup_task
                self.assertFalse(partial.pixels)
                self.assertTrue(partial.timer.cancelled())
                self.assertIsNone(self.gateway.owner)
                self.assertEqual(self.fixture.chunks, [])
        self.assertEqual(self.fixture.cancel_count, 2)

    async def test_frames_before_start_and_unknown_or_mixed_operations_fail(self):
        socket = await self.fixture.client.ws_connect(self.fixture.url + '/stream', origin=self.fixture.url)
        self.fixture.sockets.append(socket)
        await socket.send_bytes(pack(*self.frame()))
        self.assertEqual((await self.fixture.receive(socket))[0]['op'], 'error')
        self.assertFalse(self.fixture.started)
        self.assertEqual(self.fixture.cancel_count, 0)
        await self.connected()
        owner = self.gateway.owner
        for metadata, pixels in ((dict(op='unknown'), b''), unpack(self.fixture.chunk())):
            with self.assertRaises(SessionError):
                owner.prepare_input(metadata, pixels)
        self.gateway.config['media'].pop('proxy_protocol')
        await self.stage(owner, 1)
        self.assertEqual(len(owner.partial.capture_ns), 1, 'protocol is fixed for the owner')

    async def test_commit_after_deadline_is_rejected_before_timer_callback_runs(self):
        await self.connected()
        owner = self.gateway.owner
        await self.stage(owner)
        partial = owner.partial
        with patch.object(asyncio.get_running_loop(), 'time', return_value=partial.timer.when()):
            with self.assertRaisesRegex(SessionError, 'deadline'):
                owner.prepare_input(self.commit(eos=True), b'')
        self.assertTrue(owner.closing)
        self.assertFalse(partial.pixels)
        self.assertEqual(self.fixture.chunks, [])

    async def test_staging_conversion_and_http_pack_copy_accounting(self):
        await self.connected()
        owner = self.gateway.owner
        spec = self.gateway.config['encoder_recipe']
        spec.update(proxy_width=64, proxy_height=64)
        frame_bytes = 64 * 64 * 3
        payload_bytes = frame_bytes * 17
        pixels = bytes(frame_bytes)
        tracemalloc.start()
        try:
            for source in range(17):
                owner.prepare_input(self.frame(source)[0], pixels)
            partial = owner.partial
            self.assertEqual(len(partial.pixels), payload_bytes)
            staging_current, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            job = owner.prepare_input(self.commit(), b'')
            converted_current, conversion_peak = tracemalloc.get_traced_memory()
            self.assertIsInstance(job.pixels, bytes)
            self.assertEqual(len(job.pixels), payload_bytes)
            self.assertFalse(partial.pixels)
            self.assertGreaterEqual(conversion_peak - staging_current, payload_bytes)
            self.assertLess(conversion_peak - staging_current, payload_bytes + 65536)
            tracemalloc.reset_peak()
            body = pack(job.metadata, job.pixels)
            packed_current, pack_peak = tracemalloc.get_traced_memory()
            self.assertGreaterEqual(packed_current - converted_current, len(body))
            self.assertLess(pack_peak - converted_current, len(body) + 65536)
            self.assertEqual(unpack(body), (job.metadata, job.pixels))
            type(self).copy_receipt = dict(frame_bytes=frame_bytes, payload_bytes=payload_bytes,
                staging_current_bytes=staging_current, conversion_peak_extra_bytes=conversion_peak - staging_current,
                packed_body_bytes=len(body), pack_peak_extra_bytes=pack_peak - converted_current)
        finally:
            tracemalloc.stop()


class ProxyConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def test_requires_media_and_boolean_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'media support'):
            await create_app(fixture=True, fixture_backend=fixtures.FixtureBackend(width=2, height=2), incremental_proxy=True)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            await create_app(fixture=True, incremental_proxy=1)

    async def test_upstream_proxy_advertisement_is_stripped_and_local_choice_applied(self):
        fixture = fixtures.MediaGatewayTests()
        await fixture.asyncSetUp()
        try:
            fixture.health['media']['proxy_protocol'] = 'untrusted/v2'
            for enabled in (False, True):
                app = await create_app(backend=f'http://{fixture.gateway.client.host}:{fixture.gateway.client.port}',
                    incremental_proxy=enabled)
                media = app[fixtures.GATEWAY].config['media']
                self.assertEqual(media.get('proxy_protocol'), 'proxy_frames/v1' if enabled else None)
        finally:
            await fixture.asyncTearDown()

    async def test_cli_advertises_opt_in(self):
        result = await asyncio.to_thread(subprocess.run, [sys.executable, '-m', 'tools.serve_web', '--help'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--incremental-proxy', result.stdout)
