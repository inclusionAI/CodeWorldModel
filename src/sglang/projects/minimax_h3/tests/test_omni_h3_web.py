import asyncio
from copy import deepcopy
import json
from pathlib import Path
import threading
import unittest

from aiohttp import ClientSession, WSMsgType, web
import msgpack
from test_omni_h3_assets import assets

from tools.serve_web import bootstrap

bootstrap()

from projects.minimax_h3.inference.omni_h3_web import ADAPTER, create_app
from projects.minimax_h3.inference.serving_http import pack, unpack
from projects.minimax_h3.inference.web_serving import create_app as create_gateway


def test_web_bundle_uses_effective_recipe_without_changing_frozen_prompt(tmp_path, assets):
    from projects.minimax_h3.inference.omni_h3_assets import prepare_native_bundle
    from projects.minimax_h3.inference.omni_h3_web import bundle_artifact
    artifact, source, options = assets
    source['encoders']['recipe']['system_prompt'] = 'Frozen prompt remains the web gate.'
    artifact.write_text(json.dumps(source))
    output = tmp_path/'continuous'
    prepare_native_bundle(artifact, output, recipe='continuous5', **options)
    _, adapted, _ = bundle_artifact(output, artifact)
    assert adapted['recipe']['segment_boundaries'] == []
    assert adapted['recipe']['bootstrap_latents'] == 9
    assert adapted['encoder_recipe']['system_prompt'] == source['encoders']['recipe']['system_prompt']


class FakeOmni:
    def __init__(self):
        self.opens, self.appends = [], []
        self.closed = 0
        self.gates = {}
        self.accept_gate = None
        self.done_before_accepted = False
        self.reject = self.bad_source = self.omit_eos = self.unconfirmed_close = False
        self.output_ready = asyncio.Event()
        self.workers = []
        self.orchestration = None

    async def handle(self, request):
        socket = web.WebSocketResponse(max_msg_size=1 << 20)
        await socket.prepare(request)
        identity = dict(id=f'native-{len(self.opens)}', open_index=1)
        sequence = 0
        pending = []
        queue = asyncio.Queue()

        async def send(value):
            await socket.send_bytes(msgpack.packb(value, use_bin_type=True))

        async def output(seq, payload=None, *, kind='data', eos=False):
            nonlocal sequence
            await send(dict(op='output', chunk=dict(session_identity=identity, seq=sequence,
                input_seq=seq, modality='video', format='h3.rgb24', kind=kind,
                t_start_ms=0., duration_ms=0., payload=payload, eos=eos, turn=None)))
            sequence += 1

        async def process():
            while True:
                chunk = await queue.get()
                seq, payload = chunk['seq'], chunk['payload']
                if seq in self.gates:
                    await self.gates[seq].wait()
                pending.extend(payload['frames'])
                if seq or chunk['eos']:
                    while pending:
                        frames, pending[:] = deepcopy(pending[:8]), pending[8:]
                        if self.bad_source:
                            frames[0]['source_frame_id'] = 'wrong-source'
                        packet = dict(width=16, height=16, frames=frames,
                                      rgb=bytes([seq % 256])*len(frames)*16*16*3)
                        if self.orchestration is not None:
                            packet['orchestration'] = self.orchestration
                        await output(seq, packet)
                    self.output_ready.set()
                if chunk['eos'] and not self.omit_eos:
                    await output(seq, eos=True)
                await output(seq, kind='input_done', eos=chunk['eos'])
                if self.done_before_accepted:
                    await asyncio.sleep(.01)
                    await send(dict(op='accepted', seq=seq))

        worker = asyncio.create_task(process())
        self.workers.append(worker)
        try:
            async for message in socket:
                if message.type != WSMsgType.BINARY:
                    break
                value = msgpack.unpackb(message.data, raw=False)
                if value['op'] == 'open':
                    self.opens.append(value['params'])
                    await send(dict(op='opened', session=identity))
                elif value['op'] == 'append':
                    chunk = value['chunk']
                    self.appends.append(chunk)
                    if self.reject:
                        await send(dict(op='rejected', seq=chunk['seq'], reason='fake queue full'))
                    else:
                        if self.accept_gate is not None:
                            await self.accept_gate.wait()
                        if not self.done_before_accepted:
                            await send(dict(op='accepted', seq=chunk['seq']))
                        queue.put_nowait(chunk)
                elif value['op'] == 'close':
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
                    self.closed += 1
                    if self.unconfirmed_close:
                        await socket.close()
                    else:
                        await send(dict(op='closed'))
                    break
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        return socket


class OmniMediaTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fake = FakeOmni()
        native = web.Application()
        native.router.add_get('/v1/session', self.fake.handle)
        self.runners = []
        native_url = await self.serve(native)
        recipe = json.loads((Path(__file__).resolve().parents[1] /
            'configs/recipes/omni-h3-longlive.json').read_text())
        self.artifact = dict(recipe=recipe, encoder_recipe=dict(width=16, height=16,
            proxy_width=16, proxy_height=16, system_prompt='Frozen fake prompt.'))
        self.app = create_app(native_url.replace('http:', 'ws:')+'/v1/session',
            'artifact', self.artifact, codec=dict(decoder='causal'), timeout=2)
        self.adapter = self.app[ADAPTER]
        self.url = await self.serve(self.app)
        self.client = ClientSession()
        self.sockets = []

    async def serve(self, app):
        runner = web.AppRunner(app, shutdown_timeout=.1, handler_cancellation=True)
        await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', 0).start()
        self.runners.append(runner)
        return 'http://127.0.0.1:'+str(runner.addresses[0][1])

    async def asyncTearDown(self):
        for socket in self.sockets:
            await socket.close()
        await self.client.close()
        for runner in reversed(self.runners):
            try:
                await runner.cleanup()
            except Exception:
                if not self.adapter.quarantine:
                    raise

    async def call(self, path, metadata, pixels=b''):
        async with self.client.post(self.url+path, data=pack(metadata, pixels)) as response:
            receipt, extra = unpack(await response.read())
            self.assertEqual(extra, b'')
            return response.status, receipt

    def start_metadata(self, sid='browser'):
        return dict(session_id=sid, artifact_id='artifact', seed=2026, condition_seed=2026,
            prompt=self.artifact['encoder_recipe']['system_prompt'], video=dict(bitrate_bps=8_000_000))

    async def start(self, sid='browser'):
        status, receipt = await self.call('/media/session', self.start_metadata(sid), bytes(16*16*3))
        self.assertEqual(status, 200, receipt)
        return receipt

    def chunk(self, seq, *, eos=False, sid='browser'):
        count, first = (5 if eos else 17), seq*17
        stamps = [10_000_000_000+index*13 for index in range(first, first+count)]
        return dict(session_id=sid, chunk_id=seq, first_source_frame=first,
            capture_ns=stamps, assembled_ns=stamps[-1]+1, eos=eos), bytes(count*16*16*3)

    async def submit(self, seq, **kwargs):
        return await self.call('/media/session/browser/chunk', *self.chunk(seq, **kwargs))

    async def cancel(self):
        return await self.call('/media/session/browser/cancel', dict(session_id='browser', reason='test'))

    async def read_record(self, response):
        size = int.from_bytes(await response.content.readexactly(4), 'big')
        return unpack(await response.content.readexactly(size))

    async def eventually(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.005)

    async def test_real_h264_gateway_startup_provenance_and_eos_drain(self):
        import av
        self.fake.orchestration = dict(clock='native_worker_monotonic', stages=[
            dict(stage='render', dispatch_ns=1000000, complete_ns=2000000, wall_ms=1.)])
        self.fake.done_before_accepted = True
        gateway = await create_gateway(backend=self.url, timeout=3, media_control=False)
        gateway_url = await self.serve(gateway)
        socket = await self.client.ws_connect(gateway_url+'/stream', origin=gateway_url)
        self.sockets.append(socket)
        await socket.send_bytes(pack(dict(op='start', seed=2026, condition_seed=2026,
            video=dict(bitrate_bps=8_000_000)), bytes(16*16*3)))

        async def receive():
            message = await socket.receive(timeout=3)
            self.assertEqual(message.type, WSMsgType.BINARY, message)
            value, rgb = unpack(message.data)
            self.assertNotEqual(value.get('op'), 'error', value)
            return value, rgb

        started, _ = await receive()
        self.assertEqual(started['op'], 'started')
        self.assertEqual(started['profiling']['status'], 'baseline')
        self.assertTrue(started['source_budget']['eligible'])
        decoder = av.CodecContext.create('h264', 'r')
        frames, decoded, batches = [], 0, []
        for seq in range(3):
            m, pixels = self.chunk(seq, eos=seq == 2)
            m.pop('session_id')
            m['prompt'] = dict(revision=1, text=self.artifact['encoder_recipe']['system_prompt'],
                              emitted_ns=123456789)
            await socket.send_bytes(pack(dict(m, op='chunk'), pixels))
            admitted, total = False, (0 if seq == 0 else 34 if seq == 1 else 39)
            while True:
                value, payload = await receive()
                if value['op'] == 'admitted':
                    admitted = True
                elif value['op'] == 'batch':
                    batches.append(value)
                elif value['op'] == 'video_frame':
                    frames.append(value)
                    decoded += len(decoder.decode(av.Packet(payload)))
                    await socket.send_json(dict(op='frame_ack', output_frame=value['output_frame']))
                elif value['op'] == 'complete':
                    self.assertEqual(value['total_frames'], 39)
                    break
                if seq < 2 and admitted and len(frames) == total and batches[-1]['chunk_id'] == seq:
                    break
            if seq == 0:
                self.assertEqual((batches[0]['frame_count'], batches[0]['accepted_frames'],
                                  batches[0]['total_frames']), (0, 17, 0))
        decoded += len(decoder.decode(None))
        self.assertEqual(decoded, 39)
        self.assertTrue(all(b['orchestration'] == self.fake.orchestration
                            for b in batches if b['frame_count']))
        self.assertEqual([f['source_frame'] for f in frames], list(range(39)))
        self.assertEqual([f['capture_ns'] for f in frames], [10_000_000_000+i*13 for i in range(39)])
        self.assertEqual([f['timestamp_us'] for f in frames], [i*1_000_000//24 for i in range(39)])
        self.assertTrue(all(f['encode_start_ns'] <= f['encode_end_ns'] for f in frames))
        self.assertTrue(any(len({s['chunk_id'] for s in b['source_frames']}) > 1 for b in batches))
        await self.eventually(lambda: self.adapter.active is None)
        self.assertEqual(self.fake.closed, 1)
        wire_frames = [f for chunk in self.fake.appends for f in chunk['payload']['frames']]
        self.assertEqual([f['pts_ms'] for f in wire_frames], [i*1000./24 for i in range(39)])
        self.assertEqual(self.fake.opens[0]['first_image']['rgb'], bytes(16*16*3))
        await self.start('restart')

    async def test_http_admitted_waits_for_accepted_and_native_credit_waits_for_done(self):
        self.fake.accept_gate = asyncio.Event()
        self.fake.gates[0] = asyncio.Event()
        self.fake.gates[1] = asyncio.Event()
        await self.start()
        first = asyncio.create_task(self.submit(0))
        await self.eventually(lambda: len(self.fake.appends) == 1)
        self.assertFalse(first.done())
        self.fake.accept_gate.set()
        self.assertEqual((await first)[0], 200)
        self.assertEqual((await self.submit(1))[0], 200)
        third = asyncio.create_task(self.submit(2, eos=True))
        await asyncio.sleep(.03)
        self.assertEqual(len(self.fake.appends), 2)
        self.assertFalse(third.done())
        self.fake.gates[0].set()
        self.assertEqual((await third)[0], 200)
        self.assertEqual(len(self.fake.appends), 3)
        self.assertEqual((await self.cancel())[0], 200)

    async def test_browser_prompt_snapshot_accepts_frozen_text_and_rejects_changes(self):
        await self.start()
        metadata, pixels = self.chunk(0)
        metadata['prompt'] = dict(revision=1,
            text=self.artifact['encoder_recipe']['system_prompt'], emitted_ns=123456789)
        changed = dict(metadata, prompt=dict(metadata['prompt'], revision=2, text='Different prompt.'))
        status, receipt = await self.call('/media/session/browser/chunk', changed, pixels)
        self.assertEqual(status, 400, receipt)
        self.assertEqual(self.fake.appends, [])
        status, receipt = await self.call('/media/session/browser/chunk', metadata, pixels)
        self.assertEqual(status, 200, receipt)
        self.assertEqual(len(self.fake.appends), 1)
        self.assertNotIn('prompt', self.fake.appends[0])

    async def test_native_rejection_is_not_admission_and_confirmed_cleanup_allows_restart(self):
        self.fake.reject = True
        await self.start()
        status, receipt = await self.submit(0)
        self.assertEqual(status, 502, receipt)
        await self.eventually(lambda: self.adapter.active is None)
        self.assertEqual(self.fake.closed, 1)
        self.fake.reject = False
        await self.start('new')

    async def test_missing_native_eos_does_not_complete_on_input_eos_or_done(self):
        self.fake.omit_eos = True
        await self.start()
        async with self.client.get(self.url+'/media/session/browser/events') as response:
            self.assertEqual((await self.submit(0, eos=True))[0], 200)
            seen = []
            while len([m for m in seen if m['op'] == 'video_frame']) < 5:
                metadata, _ = await self.read_record(response)
                seen.append(metadata)
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(self.read_record(response), .04)
            self.assertFalse(self.adapter.active.finished)
            self.assertEqual((await self.cancel())[0], 200)

    async def test_bad_native_provenance_fails_and_closes_session(self):
        self.fake.bad_source = True
        await self.start()
        async with self.client.get(self.url+'/media/session/browser/events') as response:
            self.assertEqual((await self.submit(0, eos=True))[0], 200)
            metadata, _ = await self.read_record(response)
            self.assertEqual(metadata['op'], 'error', metadata)
            self.assertIn('provenance', metadata['error'])
        self.assertEqual(self.fake.closed, 1)

    async def test_slow_reader_is_bounded_and_cancel_cleans_encoder_and_native(self):
        await self.start()
        session = self.adapter.active
        self.assertEqual((await self.submit(0))[0], 200)
        self.assertEqual((await self.submit(1))[0], 200)
        await self.eventually(lambda: session.events.full() and session.raw.qsize() >= 1)
        self.assertEqual(session.events.qsize(), 4)
        self.assertLessEqual(session.raw.qsize(), 2)
        self.assertEqual((await self.cancel())[0], 200)
        self.assertTrue(session.encode_task.done())
        self.assertTrue(session.receive_task.done())
        self.assertIsNone(session.encoder)
        self.assertIsNone(self.adapter.active)
        await self.start('new')

    async def test_cancel_waits_for_executing_cpu_encode_before_reuse(self):
        from projects.minimax_h3.inference.serving_media import H264Encoder
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()

        class BlockingEncoder(H264Encoder):
            def encode(self, rgb, ordinal):
                entered.set()
                release.wait(3)
                return super().encode(rgb, ordinal)

            def close(self):
                super().close()
                closed.set()

        self.adapter.encoder_factory = BlockingEncoder
        await self.start()
        await self.submit(0, eos=True)
        await self.eventually(entered.is_set)
        cancellation = asyncio.create_task(self.cancel())
        try:
            await self.eventually(lambda: self.fake.closed == 1)
            self.assertFalse(cancellation.done())
            self.assertFalse(closed.is_set())
            status, _ = await self.call('/media/session', self.start_metadata('new'), bytes(16*16*3))
            self.assertEqual(status, 409)
        finally:
            release.set()
        self.assertEqual((await cancellation)[0], 200)
        self.assertTrue(closed.is_set())
        self.assertIsNone(self.adapter.active)

    async def test_event_reader_disconnect_confirms_native_cleanup(self):
        await self.start()
        response = await self.client.get(self.url+'/media/session/browser/events')
        response.close()
        await self.eventually(lambda: self.adapter.active is None)
        self.assertEqual(self.fake.closed, 1)

    async def test_unconfirmed_close_quarantines_restart(self):
        self.fake.unconfirmed_close = True
        await self.start()
        status, _ = await self.cancel()
        self.assertEqual(status, 503)
        self.assertIn('unconfirmed', self.adapter.quarantine)
        status, _ = await self.call('/media/session', self.start_metadata('new'), bytes(16*16*3))
        self.assertEqual(status, 503)

    async def test_unsupported_seeds_and_wait_observation_reject_before_native_open(self):
        for changes in (dict(condition_seed=3), dict(observe_waits=True)):
            metadata = dict(self.start_metadata(), **changes)
            status, _ = await self.call('/media/session', metadata, bytes(16*16*3))
            self.assertEqual(status, 400)
        self.assertEqual(self.fake.opens, [])


if __name__ == '__main__':
    unittest.main()
