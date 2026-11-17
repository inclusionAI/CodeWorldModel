import asyncio
import struct
import unittest

from aiohttp import ClientSession, WSMsgType, web

from tools.serve_web import bootstrap

bootstrap()

from projects.minimax_h3.inference.serving_http import pack, unpack
from projects.minimax_h3.inference.web_serving import FixtureBackend, GATEWAY, create_app


class MediaGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        fixture = FixtureBackend(width=2, height=2)
        self.health = fixture.health
        self.policy = dict(protocol='h264_annexb_v1', codec='avc1.42C028', fps=24,
            bitrate_min_bps=500000, bitrate_max_bps=20000000, bitrate_default_bps=8000000, frame_window=4)
        self.health['media'] = self.policy
        self.events = asyncio.Queue(maxsize=4)
        self.producers = []
        self.cancel_count = 0
        self.chunks = []
        self.started = False
        self.total = 0
        self.ready = asyncio.Event()
        self.start_gate = None
        self.start_received = asyncio.Event()
        self.rpc_delay = 0
        backend = web.Application()
        async def health(request):
            return web.Response(body=pack(self.health))
        backend.router.add_get('/health', health)
        backend.router.add_post('/media/session', self.start_remote)
        backend.router.add_post('/media/session/{sid}/chunk', self.chunk_remote)
        backend.router.add_post('/media/session/{sid}/cancel', self.cancel_remote)
        backend.router.add_get('/media/session/{sid}/events', self.output_remote)
        self.backend = web.AppRunner(backend, shutdown_timeout=.1)
        await self.backend.setup()
        await web.TCPSite(self.backend, '127.0.0.1', 0).start()
        remote = 'http://127.0.0.1:' + str(self.backend.addresses[0][1])
        self.app = await create_app(backend=remote, timeout=2, cleanup_timeout=.1,
                                    media_control=getattr(self, 'media_control', False),
                                    incremental_proxy=getattr(self, 'incremental_proxy', False))
        self.gateway = self.app[GATEWAY]
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        await web.TCPSite(self.runner, '127.0.0.1', 0).start()
        self.url = 'http://127.0.0.1:' + str(self.runner.addresses[0][1])
        self.client = ClientSession()
        self.sockets = []

    async def asyncTearDown(self):
        for socket in self.sockets:
            await socket.close()
        await self.client.close()
        await self.runner.cleanup()
        for task in self.producers:
            task.cancel()
        await asyncio.gather(*self.producers, return_exceptions=True)
        await self.backend.cleanup()

    async def start_remote(self, request):
        m, _ = unpack(await request.read())
        self.start_metadata = m
        self.start_received.set()
        if self.start_gate is not None:
            await self.start_gate.wait()
        if getattr(self, 'start_status', 200) != 200:
            return web.Response(status=self.start_status, body=pack(dict(error='start rejected')))
        self.started = True
        self.sid = m['session_id']
        receipt = dict(session_id='wrong' if getattr(self, 'corrupt_start', False) else self.sid, status='accepted',
            wait_observability=dict(schema='stream_waits/v1', enabled=m.get('observe_waits', False)),
            media=dict(self.policy, width=2, height=2, bitrate_bps=m['video']['bitrate_bps']))
        if hasattr(self, 'start_recipe'):
            receipt['recipe'] = self.start_recipe
        return web.Response(body=pack(receipt))

    async def chunk_remote(self, request):
        m, pixels = unpack(await request.read())
        self.chunks.append((m, pixels))
        await asyncio.sleep(self.rpc_delay)
        self.producers.append(asyncio.create_task(self.produce(m)))
        return web.Response(body=pack(dict(session_id=self.sid, chunk_id=m['chunk_id'], status='admitted')))

    async def produce(self, m):
        await self.ready.wait()
        count = len(m['capture_ns'])
        start = self.total
        self.total += count
        sources = [dict(source_frame=m['first_source_frame']+i, capture_ns=stamp, chunk_id=m['chunk_id'])
                   for i, stamp in enumerate(m['capture_ns'])]
        await self.events.put((dict(op='batch', session_id=self.sid, chunk_id=m['chunk_id'],
            output_id=m['chunk_id'], output_frame_start=start, frame_count=count, width=2, height=2,
            source_frames=sources, status='complete' if m['eos'] else 'accepted'), b''))
        for i, source in enumerate(sources):
            ordinal = start+i
            await self.events.put((dict(source, op='video_frame', session_id=self.sid,
                output_id=m['chunk_id'], output_frame=ordinal, timestamp_us=ordinal*1000000//24,
                duration_us=41667, key_frame=i == 0, encoded_bytes=5), b'video'))
        if m['eos']:
            await self.events.put((dict(op='complete', session_id=self.sid, total_frames=self.total), b''))

    async def output_remote(self, request):
        response = web.StreamResponse()
        await response.prepare(request)
        try:
            while True:
                m, data = await self.events.get()
                body = pack(m, data)
                await response.write(struct.pack('!I', len(body))+body)
                if m['op'] == 'complete':
                    break
        except (ConnectionError, asyncio.CancelledError):
            pass
        return response

    async def cancel_remote(self, request):
        if not self.started:
            return web.Response(status=409, body=pack(dict(error='unknown media session')))
        self.cancel_count += 1
        if getattr(self, 'cancel_status', 200) != 200:
            return web.Response(status=self.cancel_status, body=pack(dict(error='cancel rejected')))
        return web.Response(body=pack(dict(session_id=self.sid, status='cancelled')))

    async def receive(self, socket):
        message = await socket.receive(timeout=3)
        self.assertEqual(message.type, WSMsgType.BINARY, message)
        return unpack(message.data)

    async def connect(self, bitrate=8000000, observe_waits=None):
        socket = await self.client.ws_connect(self.url+'/stream', origin=self.url)
        self.sockets.append(socket)
        metadata = dict(op='start', seed=1, condition_seed=1, video=dict(bitrate_bps=bitrate))
        if observe_waits is not None:
            metadata['observe_waits'] = observe_waits
        await socket.send_bytes(pack(metadata, bytes(12)))
        return socket

    def chunk(self, index=0, eos=False):
        count = 5 if eos else 17
        first = index*17
        return pack(dict(op='chunk', chunk_id=index, first_source_frame=first,
            capture_ns=list(range(first, first+count)), assembled_ns=first+count, eos=eos), bytes(count*12))

    async def test_admission_does_not_wait_for_output_and_eos_drains_all_frames(self):
        self.start_recipe = None
        async with self.client.get(self.url + '/config') as response:
            config = await response.json()
        self.assertEqual(config['source_budget'], dict(eligible=False, reason='unsupported_recipe_or_decoder'))
        socket = await self.connect()
        start, _ = await self.receive(socket)
        self.assertEqual(start['source_budget'], config['source_budget'])
        self.assertEqual(start['media']['bitrate_bps'], 8000000)
        self.assertNotIn('observe_waits', self.start_metadata, 'default-off gateway preserves old backend start fields')
        self.assertNotIn('waits', start)
        await socket.send_bytes(self.chunk(eos=True))
        admitted, _ = await self.receive(socket)
        self.assertEqual(admitted['op'], 'admitted')
        self.assertFalse(self.ready.is_set())
        self.ready.set()
        received = []
        while True:
            m, data = await self.receive(socket)
            if m['op'] == 'video_frame':
                received.append(m['source_frame'])
                self.assertEqual(data, b'video')
                await socket.send_json(dict(op='frame_ack', output_frame=m['output_frame']))
            if m['op'] == 'complete':
                break
        self.assertEqual(received, list(range(5)))
        await socket.receive(timeout=3)
        self.assertEqual(self.cancel_count, 0)

    async def test_started_budget_uses_effective_receipt_recipe(self):
        self.gateway.config['codec']['decoder'] = 'causal'
        self.start_recipe = dict(protocol='ref_fixed_recent_history/v2',
            conditioning_mode='static_image_qwen_proxy_vae_v1', bootstrap_latents=9,
            current_latent_schedule=[5])
        socket = await self.connect()
        started, _ = await self.receive(socket)
        self.assertEqual(started['source_budget'], dict(eligible=True,
            reason='fixed_v2_bootstrap9_current5_causal'))
        self.assertEqual(started['recipe'], self.start_recipe)

    async def test_frame_window_blocks_delivery_but_cancel_still_cleans(self):
        socket = await self.connect()
        await self.receive(socket)
        await socket.send_bytes(self.chunk())
        self.assertEqual((await self.receive(socket))[0]['op'], 'admitted')
        self.ready.set()
        self.assertEqual((await self.receive(socket))[0]['op'], 'batch')
        for index in range(4):
            self.assertEqual((await self.receive(socket))[0]['output_frame'], index)
        self.assertEqual(len(self.gateway.owner.unacknowledged), 4)
        with self.assertRaises(asyncio.TimeoutError):
            await socket.receive(timeout=.05)
        await socket.send_json(dict(op='cancel'))
        self.assertEqual((await self.receive(socket))[0]['op'], 'cancelled')
        self.assertEqual(self.cancel_count, 1)
        self.assertIsNone(self.gateway.owner)

    async def test_invalid_bitrate_rejected_before_remote_start(self):
        socket = await self.connect(bitrate=True)
        self.assertEqual((await self.receive(socket))[0]['op'], 'error')
        self.assertFalse(self.started)

    async def test_observed_rpc_read_ack_and_send_lock_keep_frame_order(self):
        socket = await self.connect(observe_waits=True)
        started, _ = await self.receive(socket)
        self.assertTrue(started['wait_observability']['enabled'])
        owner = self.gateway.owner
        await owner.send_lock.acquire()
        waiting = asyncio.Event()
        acquire = owner.send_lock.acquire
        async def acquire_when_waiting():
            waiting.set()
            return await acquire()
        owner.send_lock.acquire = acquire_when_waiting
        self.rpc_delay = .03
        try:
            await socket.send_bytes(self.chunk(eos=True))
            await asyncio.wait_for(waiting.wait(), 1)
            await asyncio.sleep(.03)
        finally:
            owner.send_lock.acquire = acquire
            owner.send_lock.release()
        admitted, _ = await self.receive(socket)
        gateway = admitted['waits']['gateway']
        self.assertGreater(gateway['rpc_end_ns'] - gateway['rpc_start_ns'], 20_000_000)
        self.assertGreater(gateway['send_lock']['end_ns'] - gateway['send_lock']['start_ns'], 20_000_000)
        self.assertLessEqual(gateway['request_complete_ns'], gateway['rpc_start_ns'])
        self.ready.set()
        batch, _ = await self.receive(socket)
        read = batch['waits']['gateway']['record_read']
        self.assertGreater(read['end_ns'] - read['start_ns'], 20_000_000)
        for ordinal in range(4):
            frame, data = await self.receive(socket)
            self.assertEqual((frame['output_frame'], data), (ordinal, b'video'))
        await asyncio.sleep(.04)
        await socket.send_json(dict(op='frame_ack', output_frame=0))
        last, data = await self.receive(socket)
        self.assertEqual((last['output_frame'], data), (4, b'video'))
        ack = last['waits']['gateway']['ack_window']
        self.assertGreater(ack['end_ns'] - ack['start_ns'], 20_000_000)
        complete, _ = await self.receive(socket)
        self.assertEqual(complete['wait_coverage']['socket_write'], 'not_collected')
        self.assertFalse(complete['wait_coverage']['gateway_complete'])
        for ordinal in range(1, 5):
            await socket.send_json(dict(op='frame_ack', output_frame=ordinal))
        await socket.receive(timeout=3)
        self.assertEqual(self.cancel_count, 0)

    async def test_observed_cancel_retains_interrupted_ack_window(self):
        socket = await self.connect(observe_waits=True)
        await self.receive(socket)
        owner = self.gateway.owner
        await socket.send_bytes(self.chunk())
        await self.receive(socket)
        self.ready.set()
        await self.receive(socket)
        for _ in range(4):
            await self.receive(socket)
        await asyncio.sleep(.03)
        await socket.send_json(dict(op='cancel'))
        cancelled, _ = await self.receive(socket)
        self.assertEqual(cancelled['op'], 'cancelled')
        self.assertEqual(cancelled['waits']['gateway']['ack_window']['outcome'], 'cancelled')
        self.assertGreater(cancelled['wait_coverage']['gateway']['interrupted'], 0)
        self.assertIsNone(self.gateway.owner)
        self.assertEqual((owner.first_close['channel'], owner.first_close['trigger']), ('media', 'explicit_cancel'))
        self.assertEqual(owner.first_close['outstanding'], 4)

    async def test_invalid_observe_flag_is_a_start_protocol_error(self):
        socket = await self.connect(observe_waits='yes')
        self.assertEqual((await self.receive(socket))[0]['op'], 'error')
        self.assertFalse(self.started)

    async def test_forged_frame_ack_closes_session(self):
        socket = await self.connect()
        await self.receive(socket)
        await socket.send_json(dict(op='frame_ack', output_frame=0))
        self.assertEqual((await self.receive(socket))[0]['op'], 'error')
        await socket.receive(timeout=3)
        async with asyncio.timeout(3):
            while self.gateway.owner is not None:
                await asyncio.sleep(.01)
        self.assertEqual(self.cancel_count, 1)

    async def test_cancel_waits_for_start_registration_without_quarantining_gateway(self):
        self.start_gate = asyncio.Event()
        socket = await self.connect()
        await asyncio.wait_for(self.start_received.wait(), 2)
        await socket.send_json(dict(op='cancel'))
        await asyncio.sleep(.05)
        self.assertEqual(self.cancel_count, 0)
        self.start_gate.set()
        self.assertEqual((await self.receive(socket))[0]['op'], 'cancelled')
        self.assertEqual(self.cancel_count, 1)
        self.assertIsNone(self.gateway.owner)
        self.assertFalse(self.gateway.quarantine)
