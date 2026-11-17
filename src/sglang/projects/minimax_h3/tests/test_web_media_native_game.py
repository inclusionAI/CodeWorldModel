import asyncio
import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from aiohttp import ClientSession, WSMsgType, web
from aiohttp.test_utils import TestServer
from PIL import Image

from tests.test_web_native_game import NativeFixture
from projects.minimax_h3.inference.serving import SessionError
from projects.minimax_h3.inference.serving_http import pack, unpack
from projects.minimax_h3.inference.web_native_game import source_rgb
from projects.minimax_h3.inference.web_media_fixture import MediaFixtureBackend
from projects.minimax_h3.inference.web_serving import GATEWAY, create_app


class RecordingFixture(MediaFixtureBackend):
    def __init__(self):
        super().__init__(width=2, height=2)
        self.calls = []

    def execute(self, op, metadata, pixels):
        self.calls.append((op, metadata, pixels))
        return super().execute(op, metadata, pixels)


class NativeGameMediaHarness(unittest.IsolatedAsyncioTestCase):
    use_backend = False

    async def asyncSetUp(self):
        self.native = NativeFixture()
        native_app = web.Application()
        native_app.router.add_route('*', '/{path:.*}', self.native.handle)
        self.native_server = TestServer(native_app)
        await self.native_server.start_server()
        self.fixture = RecordingFixture()
        self.backend_url = await self.fixture.start() if self.use_backend else None
        self.app = await create_app(backend=self.backend_url, fixture=not self.use_backend,
            fixture_backend=None if self.use_backend else self.fixture,
            native_game_proxy_url=str(self.native_server.make_url('/')), media_control=False, incremental_proxy=True)
        self.gateway = self.app[GATEWAY]
        self.bridge = self.gateway.native_game
        self.server = TestServer(self.app)
        await self.server.start_server()
        self.http = ClientSession()
        self.origin = str(self.server.make_url('')).rstrip('/')
        self.controller = await self.http.ws_connect(self.server.make_url('/native_game'), origin=self.origin)
        self.serial = 0
        self.media = None

    async def asyncTearDown(self):
        self.native.pause_gate.set()
        if self.native.capture_gate:
            self.native.capture_gate.set()
        if self.native.preview_gate:
            self.native.preview_gate.set()
        if self.media:
            await self.media.close()
        await self.controller.close()
        await self.http.close()
        await self.server.close()
        if self.use_backend:
            await self.fixture.close()
        await self.native_server.close()

    async def command(self, op, **fields):
        self.serial += 1
        await self.controller.send_json(dict(op=op, id=self.serial, **({'segment': self.gateway.owner.native_game_segment} if op == 'capture' else {}), **fields))
        message = await self.controller.receive(timeout=4)
        self.assertEqual(message.type, WSMsgType.BINARY, message.data)
        return unpack(message.data)

    async def receive(self):
        message = await self.media.receive(timeout=4)
        self.assertEqual(message.type, WSMsgType.BINARY, message.data)
        return unpack(message.data)

    async def start(self, *, reference=None, budget='bounded51', prepare=None):
        prepared, _ = await self.command('prepare') if prepare is None else prepare
        binding = dict(prepared['binding'], reference='prepared' if reference is None else 'upload', budget=budget)
        self.media = await self.http.ws_connect(self.server.make_url('/stream'), origin=self.origin)
        await self.media.send_bytes(pack(dict(op='start', seed=1, condition_seed=1,
            video=dict(bitrate_bps=8000000), source=binding), b'' if reference is None else reference))
        reply, payload = await self.receive()
        self.assertEqual(reply['op'], 'started', reply)
        expected_budget = dict(eligible=not self.use_backend,
            reason='unsupported_recipe_or_decoder' if self.use_backend else 'fixture_h264')
        self.assertEqual(self.gateway.config['source_budget'], expected_budget)
        self.assertEqual(reply['source_budget'], expected_budget)
        self.assertFalse(payload)
        self.assertNotIn(binding['capability'], str(reply))
        self.owner = self.gateway.owner
        self.assertEqual(reply['source']['protocol'], 'native_game_commands/v1')
        return binding

    async def capture(self, number):
        if number == 1:
            await self.command('resume')
        if number:
            self.native.state['frame_id'] += 1
        self.native.publication_ns += 1_000_000_000 // 24
        while True:
            result, pixels = await self.command('capture', source_frame=number, capture_ns=number + 1000)
            self.assertFalse(pixels)
            if not result.get('waiting'):
                return result

    def commit(self, first, count, chunk=0):
        return dict(op='native_game_chunk', chunk_id=chunk, first_source_frame=first,
            capture_ns=list(range(first + 1000, first + 1000 + count)), assembled_ns=first + 1000 + count,
            eos=count == 5)


class NativeGameMediaTests(NativeGameMediaHarness):
    async def test_slow_native_repeats_rgb_without_rebasing_or_new_decode(self):
        await self.start()
        await self.capture(0)
        first = (await self.capture(1))['sample']
        clock = [first['gateway_sample_ready_ns']]

        async def no_publication(**request):
            clock[0] += request['wait_ms'] * 1_000_000
            return None

        original = self.bridge.rgb
        with (patch.object(self.bridge, 'rgb', wraps=original) as decode,
              patch.object(self.bridge.native, 'frame', no_publication),
              patch('projects.minimax_h3.inference.web_media.time.perf_counter_ns', lambda: clock[0])):
            for number in (2, 3, 4):
                while True:
                    result = await self.owner.capture_native_game(number, number + 1000)
                    if not result.get('waiting'):
                        break
                sample = result['sample']
                self.assertEqual(sample['sample_kind'], 'repeat')
                self.assertEqual(sample['media_segment'], first['media_segment'])
                self.assertEqual(sample['source']['frame_id'], first['source']['frame_id'])
                self.assertEqual(sample['gateway_rgb_ready_ns'], first['gateway_rgb_ready_ns'])
                self.assertGreater(sample['gateway_sample_ready_ns'], first['gateway_sample_ready_ns'])
                self.assertGreaterEqual(sample['gateway_sample_ready_ns'], sample['gateway_selected_ns'])
            self.assertEqual(decode.call_count, 0)
        self.assertEqual(self.owner.native_game_captured, 5)

    async def test_empty_wait_is_bounded_and_real_http_timeout_fails(self):
        await self.start()
        await self.capture(0)
        await self.command('resume')
        self.native.freeze_status = self.native.preview_timeout = True
        begin = time.monotonic()
        result = await self.owner.capture_native_game(1, 1001)
        self.assertEqual(result, dict(waiting=True))
        self.assertLess(time.monotonic() - begin, .3)
        self.assertEqual(self.owner.native_game_captured, 1)
        self.assertFalse(self.bridge.native.lock.locked())
        self.native.capture_gate = asyncio.Event()
        begin = time.monotonic()
        with self.assertRaises(TimeoutError):
            await self.owner.capture_native_game(1, 1001)
        self.assertLess(time.monotonic() - begin, .5)
        self.assertFalse(self.owner.native_game_pending)
        self.assertEqual(len(self.owner.native_game_samples), 1)
        self.native.capture_gate.set()

    async def test_pause_invalidates_wait_before_queued_control_and_one_reply(self):
        await self.start()
        await self.capture(0)
        await self.command('resume')
        self.native.capture_started.clear()
        self.native.capture_gate = asyncio.Event()
        await self.controller.send_json(dict(op='capture', id=100, source_frame=1, capture_ns=1001, segment=0))
        await asyncio.wait_for(self.native.capture_started.wait(), 1)
        async with asyncio.timeout(.05):
            async with self.bridge.native.lock:
                self.assertTrue(self.owner.native_game_pending)
        generation = self.bridge.native.generation
        await self.controller.send_json(dict(op='pause', id=101))
        for _ in range(50):
            if self.bridge.native.generation != generation:
                break
            await asyncio.sleep(.001)
        self.assertNotEqual(self.bridge.native.generation, generation)
        self.native.capture_gate.set()
        cancelled, _ = unpack((await self.controller.receive(timeout=1)).data)
        paused, _ = unpack((await self.controller.receive(timeout=1)).data)
        self.assertEqual(cancelled, dict(op='result', id=100, cancelled=True))
        self.assertEqual(paused['id'], 101)
        self.assertEqual(self.owner.native_game_captured, 1)
        resumed, _ = await self.command('resume')
        self.assertEqual(resumed['id'], self.serial, 'no duplicate terminal response remains queued')

    async def test_cleanup_during_http_wait_discards_late_publication(self):
        await self.start()
        await self.capture(0)
        await self.command('resume')
        self.native.capture_started.clear()
        self.native.capture_gate = asyncio.Event()
        capture = asyncio.create_task(self.owner.capture_native_game(1, 1001))
        await asyncio.wait_for(self.native.capture_started.wait(), 1)
        cleanup = self.owner.begin_cleanup()
        await asyncio.sleep(.01)
        self.native.capture_gate.set()
        self.assertEqual(await capture, dict(cancelled=True))
        await asyncio.wait_for(cleanup, 3)
        self.assertFalse(self.owner.native_game_samples)
        self.assertEqual(self.owner.native_game_captured, 1)
        self.assertFalse(self.owner.native_game_pending)

    async def test_media_cancel_during_capture_preserves_controller_and_single_reply(self):
        await self.start()
        await self.capture(0)
        await self.command('resume')
        self.native.capture_started.clear()
        self.native.capture_gate = asyncio.Event()
        await self.controller.send_json(dict(op='capture', id=100, source_frame=1, capture_ns=1001, segment=0))
        await asyncio.wait_for(self.native.capture_started.wait(), 1)
        await self.media.send_json(dict(op='cancel'))
        cancelled, _ = await self.receive()
        self.assertEqual(cancelled['op'], 'cancelled')
        await asyncio.wait_for(self.owner.cleanup_task, 1)
        self.assertIsNone(self.bridge.media)
        self.assertIsNone(self.gateway.owner)
        self.native.capture_gate.set()
        response = await self.controller.receive(timeout=1)
        self.assertEqual(response.type, WSMsgType.BINARY, response.data)
        receipt, _ = unpack(response.data)
        self.assertEqual(receipt, dict(op='result', id=100, cancelled=True))
        self.assertFalse(self.owner.native_game_samples)
        self.assertEqual(self.owner.native_game_captured, 1)
        paused, _ = await self.command('pause')
        self.assertTrue(paused['evidence']['paused'], 'late capture has exactly one reply before pause')
        prepared, _ = await self.command('prepare')
        self.assertIn('binding', prepared, 'normal cancellation keeps the controller reusable')
        self.assertIsNone(self.gateway.quarantine)

    async def test_closing_capture_still_rejects_foreign_owners_and_epoch(self):
        await self.start()
        await self.capture(0)
        await self.command('resume')
        for target, name, value in ((self.bridge, 'media', object()),
                                    (self.gateway, 'owner', object()),
                                    (self.bridge, 'native', object()),
                                    (self.bridge.native, 'instance', 'another-native'),
                                    (self.bridge.native, 'epoch', self.bridge.native.epoch + 1)):
            with self.subTest(field=name):
                original = getattr(target, name)
                self.native.capture_started.clear()
                self.native.capture_gate = asyncio.Event()
                capture = asyncio.create_task(self.owner.capture_native_game(1, 1001))
                await asyncio.wait_for(self.native.capture_started.wait(), 1)
                self.owner.closing = True
                setattr(target, name, value)
                self.native.capture_gate.set()
                try:
                    with self.assertRaisesRegex(SessionError, 'owner changed'):
                        await capture
                finally:
                    setattr(target, name, original)
                    self.owner.closing = False
                self.assertEqual(self.owner.native_game_captured, 1)

    async def test_slow_conversion_then_capacity_discards_slot_debt(self):
        await self.start()
        await self.capture(0)
        await self.command('resume')
        original = self.bridge.rgb

        async def delayed(*args, **kwargs):
            await asyncio.sleep(.15)
            return await original(*args, **kwargs)

        with patch.object(self.bridge, 'rgb', delayed):
            first = (await self.owner.capture_native_game(1, 1001))['sample']
        self.native.state['frame_id'] += 1
        self.native.publication_ns += 200_000_000
        second = (await self.owner.capture_native_game(2, 1002))['sample']
        self.assertGreater(second['media_segment'], first['media_segment'])
        self.assertEqual(second['media_slot'], 0)
        self.assertEqual(second['media_reason'], 'scheduling_gap')
        self.native.state['frame_id'] += 1
        self.native.publication_ns += 10_000_000_000
        third = (await self.owner.capture_native_game(3, 1003, 1))['sample']
        self.assertEqual(third['media_slot'], 0)
        self.assertEqual(third['media_reason'], 'capacity')
        self.assertEqual(self.owner.native_game_captured, 4)

    async def test_delayed_http_reply_rebases_after_wait_before_staging(self):
        await self.start()
        await self.capture(0)
        first = (await self.capture(1))['sample']
        self.native.state['frame_id'] += 1
        self.native.publication_ns += 150_000_000
        original = self.bridge.native.frame

        async def delayed(*args, **kwargs):
            await asyncio.sleep(.15)
            return await original(*args, **kwargs)

        with patch.object(self.bridge.native, 'frame', delayed):
            second = (await self.owner.capture_native_game(2, 1002))['sample']
        self.assertEqual(second['media_segment'], first['media_segment'] + 1)
        self.assertEqual(second['media_slot'], 0)
        self.assertEqual(second['media_reason'], 'native_wait_gap')
        self.assertGreater(second['media_omitted_ns'], 80_000_000)
        self.assertEqual(self.owner.native_game_captured, 3)

    async def test_empty_wire_payload_uses_pinned_rgb_and_17_5_fixture_output(self):
        prepared = await self.command('prepare')
        pinned = prepared[1]
        changed = io.BytesIO()
        Image.new('RGB', (2, 1), (220, 20, 40)).save(changed, format='JPEG')
        self.native.jpeg = changed.getvalue()
        binding = await self.start(prepare=prepared)
        result = await self.capture(0)
        self.assertEqual(result['sample']['source']['frame_id'], binding['frame_id'])
        self.assertTrue(result['sample']['source']['paused'])
        self.assertEqual(self.owner.native_game_samples[0][1], source_rgb(pinned, self.fixture.spec))
        self.assertEqual(self.fixture.calls[0][2], source_rgb(pinned, self.fixture.spec, reference=True))
        with self.assertRaises(SessionError):
            self.bridge.claim_prepared(binding, object())
        for number in range(1, 17):
            await self.capture(number)
        await self.media.send_bytes(pack(self.commit(0, 17)))
        for number in range(17, 22):
            await self.capture(number)
        admitted = False
        frames = []
        while not admitted:
            m, _ = await self.receive()
            admitted = m['op'] == 'admitted'
            if m['op'] == 'video_frame':
                self.assertEqual(m['timestamp_us'], m['output_frame'] * 1_000_000 // 24)
                frames.append(m['source_frame'])
                await self.media.send_json(dict(op='frame_ack', output_frame=m['output_frame']))
        self.native.preview_gate = asyncio.Event()
        await self.controller.send_json(dict(op='frame', id=1000, after_frame_id=0))
        await asyncio.wait_for(self.native.preview_started.wait(), 1)
        await self.media.send_bytes(pack(self.commit(17, 5, 1)))
        while True:
            m, _ = await self.receive()
            if m['op'] == 'video_frame':
                self.assertEqual(m['timestamp_us'], m['output_frame'] * 1_000_000 // 24)
                frames.append(m['source_frame'])
                await self.media.send_json(dict(op='frame_ack', output_frame=m['output_frame']))
            if m['op'] == 'complete':
                break
            self.assertNotEqual(m['op'], 'error', m)
        self.assertEqual(frames, list(range(22)))
        chunks = [call for call in self.fixture.calls if call[0] == 'chunk']
        self.assertEqual([len(call[2]) for call in chunks], [17 * 12, 5 * 12])
        self.assertEqual([call[1]['eos'] for call in chunks], [False, True])
        self.assertEqual(chunks[0][2][:12], source_rgb(pinned, self.fixture.spec))
        self.assertEqual(chunks[0][2][12:24], source_rgb(self.native.jpeg, self.fixture.spec))
        self.assertTrue(self.native.state['paused'])
        self.assertIsNone(self.bridge.prepared)
        cancelled, _ = unpack((await self.controller.receive(timeout=1)).data)
        self.assertEqual(cancelled, dict(op='result', id=1000, cancelled=True))
        late, _ = await self.command('frame', after_frame_id=0)
        self.assertTrue(late['cancelled'])
        await self.owner.cleanup_task
        self.native.preview_gate.set()
        prepared, _ = await self.command('prepare')
        self.assertEqual(prepared['evidence']['epoch'], binding['epoch'] + 1)

    async def test_uploaded_reference_preserved_and_bad_bindings_rejected(self):
        prepared = await self.command('prepare')
        binding = dict(prepared[0]['binding'], reference='upload', budget='bounded51')
        with self.assertRaises(SessionError):
            self.bridge.claim_prepared(dict(binding, capability='x' * 43), object())
        self.assertEqual(self.bridge.capability, binding['capability'], 'foreign token cannot invalidate preparation')
        for changed in ({'epoch': 99}, {'frame_id': True},
                        {'instance_id': 'another'}, {'protocol': 'rgb'}, {'budget': 'bounded999'}):
            with self.subTest(changed=changed), self.assertRaises(SessionError):
                self.bridge.claim_prepared(dict(binding, **changed), object())
            self.assertIsNone(self.bridge.capability, 'authenticated failed claim consumes preparation')
            prepared = await self.command('prepare')
            binding = dict(prepared[0]['binding'], reference='upload', budget='bounded51')
        self.bridge.expires = time.monotonic() - 1
        with self.assertRaises(SessionError):
            self.bridge.claim_prepared(binding, object())
        prepared = await self.command('prepare')
        uploaded = bytes(range(12))
        await self.start(reference=uploaded, prepare=prepared)
        self.assertEqual(self.fixture.calls[0][2], uploaded)
        self.assertFalse(any('capability' in str(call) for call in self.fixture.calls))
        await self.capture(0)
        self.assertNotEqual(self.owner.native_game_samples[0][1], uploaded)
        await self.media.close()
        await asyncio.wait_for(self.owner.cleanup_task, 3)
        self.assertIsNone(self.bridge.media)
        self.assertFalse(self.owner.native_game_samples)
        self.assertTrue(self.native.state['paused'])
        self.assertIsNone(self.gateway.quarantine)

    async def test_capture_cursor_and_budget_are_independent_of_preview_wait(self):
        await self.start()
        initial = await self.capture(0)
        await self.command('resume')
        self.native.preview_gate = asyncio.Event()
        await self.controller.send_json(dict(op='frame', id=100, after_frame_id=0))
        await asyncio.wait_for(self.native.preview_started.wait(), 1)
        self.native.state['frame_id'] += 1
        self.native.publication_ns += 1_000_000_000 // 24
        result, _ = await self.command('capture', source_frame=1, capture_ns=1001)
        self.assertEqual(result['sample']['source_frame'], 1)
        sample_query = next(value for value in reversed(self.native.snapshot_queries) if value.get('selection') == 'latest' and value.get('wait_ms') != '250')
        self.assertEqual(int(sample_query['after_frame_id']), initial['sample']['source']['frame_id'])
        self.owner.native_game_source['budget_frames'] = 2
        result, _ = await self.command('capture', source_frame=2, capture_ns=1002)
        self.assertEqual(result['blocked'], 'source_budget')
        self.native.preview_gate.set()
        preview, _ = unpack((await self.controller.receive(timeout=1)).data)
        self.assertEqual(preview['id'], 100)
        self.assertEqual(self.owner.native_game_captured, 2)

    async def test_queue_budget_ack_and_exact_prefix(self):
        await self.start(budget='bounded40')
        for number in range(34):
            await self.capture(number)
        self.assertEqual((await self.capture(34))['blocked'], 'source_budget')
        for metadata, pixels in ((self.commit(1, 17), b''), (self.commit(0, 17), b'foreign'),
                                 (dict(self.commit(0, 17), op='chunk'), b''), (self.commit(0, 5), b'')):
            with self.assertRaises(SessionError):
                self.owner.prepare_input(metadata, pixels)
        job = self.owner.prepare_input(self.commit(0, 17), b'')
        self.assertEqual(len(job.pixels), 17 * 12)
        for number in range(34, 40):
            await self.capture(number)
        self.assertEqual((await self.capture(40))['blocked'], 'source_budget')
        with self.assertRaises(SessionError):
            self.owner.acknowledge(dict(op='frame_ack', output_frame=0))
        self.owner.unacknowledged.append(0)
        self.owner.acknowledge(dict(op='frame_ack', output_frame=0))
        self.assertIn('sample', await self.capture(40))
        self.assertEqual(self.owner.native_game_captured - self.owner.native_game_acked, 40)

    async def test_cleanup_invalidates_pending_decode_and_releases_controller(self):
        await self.start()
        entered, release = threading.Event(), threading.Event()
        original = source_rgb
        def delayed(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test decode gate expired')
            return original(*args, **kwargs)
        with patch('projects.minimax_h3.inference.web_native_game.source_rgb', delayed):
            pending = asyncio.create_task(self.owner.capture_native_game(0, 1000))
            await asyncio.to_thread(entered.wait, 2)
            self.assertTrue(self.owner.native_game_pending)
            with self.assertRaises(SessionError):
                await self.owner.capture_native_game(0, 1000)
            cleanup = self.owner.begin_cleanup()
            await asyncio.sleep(.03)
            self.assertFalse(cleanup.done())
            release.set()
            self.assertEqual(await pending, dict(cancelled=True))
            await asyncio.wait_for(cleanup, 3)
        self.assertFalse(self.owner.native_game_samples)
        self.assertEqual(self.owner.native_game_captured, 0)
        self.assertIsNone(self.bridge.media)
        self.assertIsNone(self.bridge.decode_task)
        self.assertTrue(self.native.state['paused'])

    async def test_controller_close_and_prepare_while_bound_clean_media(self):
        await self.start()
        await self.capture(0)
        await self.command('resume')
        await self.controller.send_json(dict(op='prepare', id=99))
        self.assertEqual((await self.controller.receive_json(timeout=3))['op'], 'error')
        await self.controller.close()
        await asyncio.wait_for(self.bridge.done.wait(), 3)
        if self.owner.cleanup_task:
            await asyncio.wait_for(self.owner.cleanup_task, 3)
        self.assertTrue(self.owner.closing)
        self.assertFalse(self.owner.native_game_samples)
        self.assertIsNone(self.bridge.media)
        self.assertEqual(self.native.state['epoch'], 3, 'bound prepare must not reset native')
        self.assertTrue(self.native.state['paused'])

    async def test_epoch_change_and_stale_native_sample_fail(self):
        await self.start()
        await self.capture(0)
        await self.command('resume')

        async def capture(number):
            while True:
                result = await self.owner.capture_native_game(number, number + 1000)
                if not result.get('waiting'):
                    return result

        self.native.bad_snapshot = dict(epoch=self.native.state['epoch'] + 1)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            await capture(1)
        self.native.bad_snapshot = None
        await capture(1)
        self.bridge.native.progress['capture'] = (self.owner.native_game_last_evidence['frame_id'], time.monotonic() - 3)
        with self.assertRaisesRegex(ValueError, 'stopped advancing'):
            await capture(2)

    async def test_failed_start_releases_consumed_source_and_preserves_no_pixels(self):
        from projects.minimax_h3.inference.web_media import MediaBrowserSession
        prepared, _ = await self.command('prepare')
        binding = dict(prepared['binding'], reference='prepared', budget='bounded51')
        async def rejected(*args, **kwargs):
            raise SessionError('registration rejected', 409)
        with patch.object(MediaBrowserSession, 'call', rejected):
            self.media = await self.http.ws_connect(self.server.make_url('/stream'), origin=self.origin)
            owner = self.gateway.owner
            await self.media.send_bytes(pack(dict(op='start', seed=1, condition_seed=1,
                video=dict(bitrate_bps=8000000), source=binding)))
            m, _ = await self.receive()
            self.assertEqual(m['op'], 'error')
            await asyncio.wait_for(owner.cleanup_task, 3)
        self.assertIsNone(self.bridge.media)
        self.assertIsNone(self.bridge.prepared)
        self.assertIsNone(self.bridge.capability)
        self.assertFalse(owner.native_game_samples)
        self.assertTrue(self.native.state['paused'])
        self.assertIsNone(self.gateway.quarantine)

    async def test_controller_disconnect_with_failed_pause_quarantines_gateway(self):
        await self.start()
        async def failed_pause(*args, **kwargs):
            raise ValueError('native pause failed')
        with patch.object(self.bridge.native, 'pause', failed_pause):
            await self.controller.close()
            await asyncio.wait_for(self.bridge.done.wait(), 3)
            await asyncio.wait_for(self.owner.cleanup_task, 3)
        self.assertIn('Native game cleanup uncertain', self.gateway.quarantine)
        self.assertIs(self.gateway.owner, self.owner)
        self.assertIsNone(self.bridge.media)
        self.assertIsNone(self.bridge.prepared)

    async def test_pending_reservation_is_the_34th_sample_and_off_retains_ledger_bound(self):
        await self.start(budget='off')
        for number in range(33):
            await self.capture(number)
        entered, release = threading.Event(), threading.Event()
        def delayed(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test decode gate expired')
            return source_rgb(*args, **kwargs)
        with patch('projects.minimax_h3.inference.web_native_game.source_rgb', delayed):
            self.native.state['frame_id'] += 1
            self.native.publication_ns += 1_000_000_000 // 24
            pending = asyncio.create_task(self.owner.capture_native_game(33, 1033, 1))
            await asyncio.to_thread(entered.wait, 2)
            self.assertEqual(len(self.owner.native_game_samples) + self.owner.native_game_pending, 34)
            with self.assertRaises(SessionError):
                await self.owner.capture_native_game(34, 1034)
            release.set()
            await pending
        self.assertEqual((await self.capture(34))['blocked'], 'source_budget')
        self.owner.prepare_input(self.commit(0, 17), b'')
        self.owner.sources.extend({} for _ in range(222))
        self.assertEqual((await self.capture(34))['blocked'], 'source_budget')

    def test_centered_cover_transform(self):
        image = Image.new('RGB', (6, 2), (255, 0, 0))
        for x in (2, 3):
            for y in range(2):
                image.putpixel((x, y), (0, 255, 0))
        stream = io.BytesIO()
        image.save(stream, format='JPEG', quality=100, subsampling=0)
        rgb = source_rgb(stream.getvalue(), dict(proxy_width=2, proxy_height=2))
        self.assertTrue(all(rgb[i + 1] > 240 and rgb[i] < 10 for i in range(0, len(rgb), 3)))


class NativeGameModelMediaTests(NativeGameMediaHarness):
    use_backend = True

    async def test_prepared_reference_completes_model_mode_media_protocol(self):
        self.assertEqual(self.gateway.config['mode'], 'model')
        self.assertFalse(self.gateway.config['native_game_source']['fixture_only'])
        prepared = await self.command('prepare')
        pinned = prepared[1]
        await self.start(prepare=prepared)
        self.assertFalse(self.owner.native_game_source['fixture_only'])
        for number in range(17):
            await self.capture(number)
        await self.media.send_bytes(pack(self.commit(0, 17)))
        for number in range(17, 22):
            await self.capture(number)
        frames = []
        while True:
            metadata, pixels = await self.receive()
            if metadata['op'] == 'video_frame':
                self.assertTrue(pixels, 'backend produced H.264 payload')
                frames.append(metadata['source_frame'])
                await self.media.send_json(dict(op='frame_ack', output_frame=metadata['output_frame']))
            if metadata['op'] == 'admitted':
                break
            self.assertNotEqual(metadata['op'], 'error', metadata)
        await self.media.send_bytes(pack(self.commit(17, 5, 1)))
        while True:
            metadata, pixels = await self.receive()
            if metadata['op'] == 'video_frame':
                self.assertTrue(pixels)
                frames.append(metadata['source_frame'])
                await self.media.send_json(dict(op='frame_ack', output_frame=metadata['output_frame']))
            if metadata['op'] == 'complete':
                break
            self.assertNotEqual(metadata['op'], 'error', metadata)
        self.assertEqual(frames, list(range(22)))
        self.assertEqual(self.fixture.calls[0][2], source_rgb(pinned, self.fixture.spec, reference=True))
        chunks = [call for call in self.fixture.calls if call[0] == 'chunk']
        self.assertEqual([len(call[2]) for call in chunks], [17 * 12, 5 * 12])
        self.assertEqual([call[1]['eos'] for call in chunks], [False, True])
        self.assertEqual(chunks[0][2][:12], source_rgb(pinned, self.fixture.spec))
        for _, metadata, _ in self.fixture.calls:
            self.assertNotIn('source', metadata)
            self.assertNotIn('capability', str(metadata))
            self.assertNotIn('native_game_commands', str(metadata))
        self.assertTrue(self.native.state['paused'])
        self.assertIsNone(self.bridge.prepared)

    async def test_uploaded_reference_and_cancel_cleanup_in_model_mode(self):
        uploaded = bytes(range(12))
        await self.start(reference=uploaded)
        self.assertEqual(self.fixture.calls[0][2], uploaded)
        self.assertEqual(self.owner.native_game_source['reference'], 'upload')
        await self.capture(0)
        await self.command('resume')
        await self.media.send_json(dict(op='cancel'))
        metadata, _ = await self.receive()
        self.assertEqual(metadata['op'], 'cancelled', metadata)
        await asyncio.wait_for(self.owner.cleanup_task, 3)
        self.assertTrue(self.native.state['paused'])
        self.assertIsNone(self.bridge.media)
        self.assertFalse(self.owner.native_game_samples)
        self.assertIsNone(self.gateway.quarantine)

    async def test_public_gateway_and_backend_without_media_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / 'access-key'
            key.write_text('test-only-access-key-' * 3)
            with self.assertRaisesRegex(ValueError, 'loopback-only gateway'):
                await create_app(backend=self.backend_url,
                    native_game_proxy_url=self.bridge.url, public_url='https://example.com/', access_key_file=key)
        self.fixture.health.pop('media')
        with self.assertRaisesRegex(ValueError, 'backend media support'):
            await create_app(backend=self.backend_url, native_game_proxy_url=self.bridge.url)


if __name__ == '__main__':
    unittest.main()
