from dataclasses import asdict, dataclass
import json
import hashlib
import http.client
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from projects.minimax_h3.inference.serving import Request, SessionController, SessionError, Worker, load_artifacts
from projects.minimax_h3.inference.serving_http import Ledger, RefClient, SessionMailbox, ServingHTTPServer, pack, unpack
from projects.minimax_h3.tools.replay_ref import FrameProducer
from projects.minimax_h3.modeling.ref_recipe import STREAMING_TEXT_RECIPES


@dataclass(frozen=True)
class Recipe:
    window_size: int = 6
    protocol: str = 'fixed'

    def __post_init__(self):
        if not 0 <= self.window_size <= 6:
            raise ValueError('window outside frozen RoPE range')

    def to_dict(self):
        return asdict(self)


def artifacts():
    bundle = dict(encoders=dict(recipe=dict(height=1, width=1, proxy_height=1, proxy_width=1,
                                          system_prompt='frozen prompt')))
    return {'fixed': (bundle, Recipe()), 'bidir': (bundle, Recipe(protocol='bidir'))}


def start(sid='one', artifact='fixed', seed=7):
    return Request('start', dict(session_id=sid, artifact_id=artifact, window_size=6,
        seed=seed, condition_seed=seed, prompt='frozen prompt'), bytes(3))


def chunk(index=0, *, sid='one', eos=False):
    count = 5 if eos else 17
    first = index * 17
    return Request('chunk', dict(session_id=sid, chunk_id=index, first_source_frame=first,
        capture_ns=list(range(first, first+count)), assembled_ns=first+count, eos=eos), bytes(count*3))


class LocalGroup:
    rank = 0

    def broadcast(self, value):
        return value

    def rank0(self, fn):
        try:
            return fn()
        except ValueError as exc:
            raise SessionError(str(exc), 500) from exc

    def barrier(self):
        pass


class FakeBackend:
    def __init__(self, group=None):
        self.group = group or LocalGroup()
        self.weight_load_count = 1
        self.latents = self.emitted = 0
        self.seed = 0
        self.sessions = self.closed = 0
        self.fail = None
        self.gate = None
        self.calls = []

    def execute(self, op, metadata, payload):
        self.calls.append((op, threading.current_thread().name))
        if op == 'start':
            self.latents = self.emitted = 0
            self.seed = metadata['seed']
            self.sessions += 1
        if op != 'chunk':
            return b'', {}
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail is not None:
            raise self.fail
        self.latents += 2 if metadata['eos'] else 5
        consumed = self.latents if metadata['eos'] else self.latents//2*2
        total = 4*consumed - 3*((consumed+4)//5)
        rgb = bytes([self.seed % 256]) * ((total-self.emitted)*3)
        self.emitted = total
        return rgb, dict(cpu_ready_ns=time.perf_counter_ns())

    def close_session(self):
        self.closed += 1
        self.latents = self.emitted = 0


class Harness:
    def __init__(self, directory, *, timeout=2, idle_timeout=10, occupier=None):
        self.ledger = Ledger(Path(directory)/'server.jsonl')
        self.box = SessionMailbox(self.ledger, timeout=timeout)
        self.controller = SessionController(artifacts())
        self.backend = FakeBackend()
        self.error = None
        self.worker = Worker(self.backend, LocalGroup(), self.controller, self.box, idle_timeout=idle_timeout,
                             occupier=occupier)
        self.thread = threading.Thread(target=self.run, name='main-worker')
        self.thread.start()

    def run(self):
        try:
            self.worker.serve()
        except BaseException as exc:
            self.error = exc
            self.box.fail(str(exc))

    def exchange(self, request, *, delivered=True):
        job = self.box.submit(request)
        assert job.ready.wait(3), 'worker did not answer'
        self.box.acknowledge(job, delivered, None if delivered else 'test disconnect')
        assert job.released.wait(3), 'rank boundary did not release'
        return job.reply

    def close(self):
        self.box.stop()
        self.thread.join(5)
        assert not self.thread.is_alive(), 'worker did not shut down'
        self.ledger.close()


class StreamingPromptTests(unittest.TestCase):
    def controller(self, protocol=STREAMING_TEXT_RECIPES[0]):
        values = artifacts()
        values['streaming'] = (values['fixed'][0], Recipe(protocol=protocol))
        controller = SessionController(values)
        controller.prepare(start(artifact='streaming'))
        return controller

    def prompt(self, revision=1, text='A', emitted_ns=0):
        return dict(revision=revision, text=text, emitted_ns=emitted_ns)

    def applied(self, text='A', first_chunk=0, rows=1):
        return dict(text=text, sha256=hashlib.sha256(text.encode()).hexdigest(),
                    first_chunk=first_chunk, rows=rows)

    def test_prompt_validation_happens_before_counters_change(self):
        invalid = [None, {}, self.prompt(True), self.prompt(0), self.prompt(2**63),
                   self.prompt(text=''), self.prompt(text=' '), self.prompt(text='\ud800'),
                   self.prompt(text='a' * 16385), self.prompt(emitted_ns=True),
                   self.prompt(emitted_ns=-1), dict(self.prompt(), extra=1)]
        for value in invalid:
            with self.subTest(value=repr(value)[:80]):
                controller = self.controller()
                request = chunk()
                request.metadata['prompt'] = value
                with self.assertRaises(SessionError):
                    controller.prepare(request)
                self.assertEqual(controller.active.next_chunk, 0)
                self.assertEqual(controller.active.accepted, 0)
                self.assertFalse(controller.active.pending)
                self.assertIsNone(controller.active.prompt)
        with self.assertRaisesRegex(SessionError, 'chunk 0'):
            self.controller().prepare(chunk())

    def test_snapshot_repetition_omission_and_revision_integrity(self):
        controller = self.controller()
        request = chunk()
        request.metadata['prompt'] = self.prompt(3)
        controller.prepare(request)
        controller.finish(request, bytes(51), dict(applied_prompt=self.applied()))
        for value in (self.prompt(2), self.prompt(3, 'B'), self.prompt(3, emitted_ns=1), None):
            request = chunk(1)
            request.metadata['prompt'] = value
            with self.assertRaises(SessionError):
                controller.prepare(request)
            self.assertEqual(controller.active.next_chunk, 1)
        request = chunk(1)
        request.metadata['prompt'] = self.prompt(3)
        self.assertEqual(controller.prepare(request)['prompt'], self.prompt(3))
        controller.finish(request, bytes(51), dict(applied_prompt=self.applied()))
        self.assertEqual(controller.prepare(chunk(2))['prompt'], self.prompt(3))
        request.metadata['prompt']['text'] = 'changed after admission'
        self.assertEqual(controller.active.prompt.text, 'A')

    def test_delayed_frames_keep_originating_prompt_for_both_protocols(self):
        for protocol in STREAMING_TEXT_RECIPES:
            controller = self.controller(protocol)
            replies = []
            for index, (text, count) in enumerate(zip(('A', 'BBB', 'A'), (13, 21, 5))):
                request = chunk(index, eos=index == 2)
                request.metadata['prompt'] = self.prompt(index + 1, text)
                controller.prepare(request)
                reply = controller.finish(request, bytes(count * 3),
                    dict(applied_prompt=self.applied(text, index, len(text))))
                self.assertEqual(reply.metadata['submitted_prompt'], request.metadata['prompt'])
                self.assertEqual(reply.metadata['applied_prompt']['first_chunk'], index)
                replies.append(reply.metadata)
            self.assertEqual([r['frame_count'] for r in replies], [13, 21, 5])
            self.assertEqual([r['prompt_revision'] for r in replies[1]['source_frames']], [1]*4 + [2]*17)
            self.assertEqual([r['prompt_first_chunk'] for r in replies[1]['source_frames']], [0]*4 + [1]*17)
            self.assertEqual([r['prompt_rows'] for r in replies[1]['source_frames']], [1]*4 + [3]*17)
            self.assertEqual(replies[-1]['status'], 'complete')

    def test_zero_rgb_and_same_text_new_revision_retain_encoding_identity(self):
        controller = self.controller()
        for index, count in enumerate((0, 34, 5)):
            request = chunk(index, eos=index == 2)
            request.metadata['prompt'] = self.prompt(index + 1)
            controller.prepare(request)
            reply = controller.finish(request, bytes(count * 3), dict(applied_prompt=self.applied()))
            self.assertEqual(reply.metadata['submitted_prompt']['revision'], index + 1)
            self.assertEqual(reply.metadata['applied_prompt']['first_chunk'], 0)
            if index == 1:
                self.assertEqual([r['prompt_revision'] for r in reply.metadata['source_frames']], [1]*17 + [2]*17)
                self.assertTrue(all(r['prompt_first_chunk'] == 0 for r in reply.metadata['source_frames']))

    def test_bad_applied_identity_does_not_drain_source_queue(self):
        controller = self.controller()
        request = chunk()
        request.metadata['prompt'] = self.prompt()
        controller.prepare(request)
        with self.assertRaisesRegex(SessionError, 'applied prompt'):
            controller.finish(request, bytes(39), dict(applied_prompt=self.applied('B')))
        self.assertEqual(len(controller.active.pending), 17)

    def test_static_chunk_rejects_prompt(self):
        controller = SessionController(artifacts())
        controller.prepare(start())
        request = chunk()
        request.metadata['prompt'] = self.prompt()
        with self.assertRaisesRegex(SessionError, 'fields'):
            controller.prepare(request)

    def test_maximum_unicode_prompt_fits_unprofiled_response_header(self):
        for text in ('左' * (16384 // 3) + 'A', '\U0001f600' * (16384 // 4)):
            controller = self.controller()
            request = chunk()
            request.metadata['prompt'] = self.prompt(text=text)
            self.assertEqual(len(text.encode('utf-8')), 16384)
            controller.prepare(request)
            reply = controller.finish(request, bytes(39), dict(applied_prompt=self.applied(text)))
            self.assertEqual(unpack(pack(reply.metadata, reply.pixels)), (reply.metadata, reply.pixels))
        with self.assertRaisesRegex(SessionError, '64 KiB'):
            pack(dict(oversized='x' * (64 << 10)))


class ServingTests(unittest.TestCase):
    def test_start_reply_records_actual_session_settings(self):
        settings = dict(schema='h3_inference_effective/v1',
            deployment=dict(options=dict(resident_weights=True)), artifacts=[])
        controller = SessionController(artifacts(), settings=settings)
        request = start(seed=29)
        request.metadata['window_size'] = 2
        checked = controller.prepare(request)
        reply = controller.finish(request, b'', {})
        snapshot = reply.metadata['effective_settings']
        self.assertEqual(checked['recipe']['window_size'], 2)
        self.assertEqual(snapshot['session']['recipe']['window_size'], 2)
        self.assertEqual(snapshot['session']['seed'], 29)
        self.assertEqual(snapshot['session']['condition_seed'], 29)
        self.assertTrue(snapshot['deployment']['options']['resident_weights'])
        self.assertNotIn('session', settings)


    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.h = Harness(self.directory.name)

    def tearDown(self):
        self.h.close()
        self.directory.cleanup()

    def test_eos_parity_source_ledger_and_fresh_recipe_sessions(self):
        for normal in range(4):
            sid = f'session{normal}'
            reply = self.h.exchange(start(sid, 'fixed' if normal % 2 else 'bidir'))
            self.assertEqual(reply.metadata['status'], 'accepted')
            source = []
            counts = []
            for index in range(normal+1):
                reply = self.h.exchange(chunk(index, sid=sid, eos=index==normal))
                self.assertEqual(reply.status, 200)
                counts.append(reply.metadata['frame_count'])
                source += [f['source_frame'] for f in reply.metadata['source_frames']]
                self.assertEqual(len(reply.pixels), counts[-1]*3)
            self.assertEqual(source, list(range(17*normal+5)))
            self.assertEqual(reply.metadata['status'], 'complete')
            self.assertIsNone(self.h.controller.active)
            if normal:
                self.assertEqual(counts[0], 13)
        self.assertEqual(self.h.backend.sessions, 4)
        self.assertEqual(self.h.backend.weight_load_count, 1)
        self.assertEqual({thread for _, thread in self.h.backend.calls}, {'main-worker'})

    def test_delivery_ack_gates_eos_and_releases_on_disconnect(self):
        self.h.exchange(start())
        job = self.h.box.submit(chunk(eos=True))
        self.assertTrue(job.ready.wait(3))
        self.assertIsNone(self.h.controller.active)
        with self.assertRaisesRegex(SessionError, 'in flight'):
            self.h.box.submit(start('second'))
        self.h.box.acknowledge(job, False, 'EOS socket failed')
        self.assertTrue(job.released.wait(3))
        self.assertIsNone(self.h.controller.active)
        self.assertEqual(self.h.exchange(start('second')).status, 200)
        self.h.exchange(Request('cancel', dict(session_id='second')))

    def test_cancel_has_reserved_slot_while_compute_is_running(self):
        self.h.exchange(start())
        self.h.backend.gate = threading.Event()
        job = self.h.box.submit(chunk())
        cancel = self.h.box.submit(Request('cancel', dict(session_id='one', reason='test')))
        with self.assertRaises(SessionError):
            self.h.box.submit(chunk(1))
        self.h.backend.gate.set()
        self.assertTrue(job.ready.wait(3))
        self.h.box.acknowledge(job, True)
        self.assertTrue(cancel.ready.wait(3))
        self.h.box.acknowledge(cancel, True)
        self.assertTrue(cancel.released.wait(3))
        self.assertEqual(cancel.reply.metadata['status'], 'cancelled')
        self.assertIsNone(self.h.controller.active)

    def test_error_cleanup_and_restart(self):
        self.h.exchange(start())
        self.h.backend.fail = SessionError('encode failed', 500)
        self.assertEqual(self.h.exchange(chunk()).status, 500)
        self.assertIsNone(self.h.controller.active)
        self.h.backend.fail = None
        self.h.exchange(start('new'))
        reply = self.h.exchange(chunk(sid='new', eos=True))
        self.assertEqual(reply.pixels, bytes([7])*15)

    def test_malformed_shutdown_does_not_stop_the_worker(self):
        for request in (Request('shutdown', {'extra': True}), Request('shutdown', {}, b'bad')):
            with self.assertRaisesRegex(SessionError, 'no payload'):
                self.h.box.submit(request)
            self.assertFalse(self.h.box.stopping)
        self.assertEqual(self.h.exchange(start()).status, 200)
        self.assertEqual(self.h.exchange(chunk(eos=True)).status, 200)
        self.assertEqual(self.h.exchange(Request('shutdown', {})).status, 200)

    def test_collective_runtime_failure_is_fatal(self):
        self.h.exchange(start())
        self.h.backend.fail = RuntimeError('collective failed')
        job = self.h.box.submit(chunk())
        self.assertTrue(job.ready.wait(3))
        self.h.thread.join(3)
        self.assertIsInstance(self.h.error, RuntimeError)
        self.assertTrue(self.h.box.stopping)
        self.assertEqual(self.h.backend.closed, 0)

    def test_input_guards_do_not_advance_or_infer(self):
        for field, value in [('seed', True), ('prompt', 'different'), ('artifact_id', 'unknown'), ('window_size', 7)]:
            request = start()
            request.metadata[field] = value
            self.assertGreaterEqual(self.h.exchange(request).status, 400)
        self.h.exchange(start())
        before = len(self.h.backend.calls)
        for mutate in (lambda r: r.metadata.update(chunk_id=2),
                       lambda r: r.metadata.update(eos=1),
                       lambda r: r.metadata.update(capture_ns=[1]*16),
                       lambda r: setattr(r, 'pixels', bytes(15)),
                       lambda r: r.metadata.update(capture_ns=list(reversed(range(17))))):
            request = chunk()
            mutate(request)
            self.assertGreaterEqual(self.h.exchange(request).status, 400)
        self.assertEqual(len(self.h.backend.calls), before)
        self.assertEqual(self.h.controller.active.next_chunk, 0)
        self.h.exchange(Request('cancel', dict(session_id='one')))

    def test_wrong_eos_frame_count_is_session_error(self):
        self.h.exchange(start())
        self.h.backend.execute = lambda *_: (b'', {})
        reply = self.h.exchange(chunk(eos=True))
        self.assertEqual(reply.status, 500)
        self.assertIn('EOS', reply.metadata['error'])
        self.assertIsNone(self.h.controller.active)

    def test_eos_needs_five_real_frames_and_stale_chunks_are_rejected(self):
        self.h.exchange(start())
        request = chunk()
        request.metadata['eos'] = True
        self.assertEqual(self.h.exchange(request).status, 400)
        self.assertEqual(self.h.exchange(chunk(eos=True)).status, 200)
        self.assertEqual(self.h.exchange(chunk(1)).status, 409)

    def test_capture_time_cannot_move_backwards_across_chunks(self):
        self.h.exchange(start())
        self.h.exchange(chunk())
        request = chunk(1, eos=True)
        request.metadata['capture_ns'] = [0]*5
        self.assertEqual(self.h.exchange(request).status, 400)
        self.assertEqual(self.h.exchange(chunk(1, eos=True)).status, 200)

    def test_real_http_returns_rgb_and_separate_send_ledger(self):
        server = ServingHTTPServer(('127.0.0.1', 0), self.h.box, dict(artifacts={}), timeout=2)
        thread = threading.Thread(target=server.serve_forever, kwargs=dict(poll_interval=.01))
        thread.start()
        client = RefClient(f'http://127.0.0.1:{server.server_port}', timeout=2)
        try:
            request = start()
            client.call('/session', request.metadata, request.pixels)
            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
            connection.request('POST', '/session/one/chunk', body=b'bad')
            bad = connection.getresponse()
            self.assertEqual(bad.status, 413)
            bad.read()
            connection.close()
            request = chunk(eos=True)
            reply, timing = client.call('/session/one/chunk', request.metadata, request.pixels)
            self.assertEqual(reply.pixels, bytes([7])*15)
            self.assertNotIn('send_end_ns', reply.metadata)
            self.assertGreaterEqual(timing['receive_ns'], timing['send_ns'])
            client.call('/shutdown', {})
            self.h.thread.join(3)
            events = [json.loads(line) for line in (Path(self.directory.name)/'server.jsonl').read_text().splitlines()]
            self.assertTrue(any(row['event']=='send_end' and row.get('chunk_id')==0 for row in events))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)


class ProtocolTests(unittest.TestCase):
    def test_assets_must_be_identical_across_served_recipes(self):
        from projects.minimax_h3.modeling.ref_recipe import FIXED_RECIPE
        from types import SimpleNamespace
        bundle = dict(model={}, weights={'sha256':'same'}, encoders={'recipe':{}}, deployment={})
        other = dict(bundle, weights={'sha256':'different'})
        with patch('projects.minimax_h3.inference.serving.Path.read_text',
                   side_effect=[json.dumps(bundle), json.dumps(other)]), \
                patch('projects.minimax_h3.inference.serving.validate_bundle',
                      return_value=SimpleNamespace(protocol=FIXED_RECIPE)), \
                patch('projects.minimax_h3.inference.serving.verify_bundle_assets') as verify, \
                patch('projects.minimax_h3.inference.serving.sha256', return_value='id'):
            with self.assertRaisesRegex(ValueError, 'identical resident assets'):
                load_artifacts(['fixed.json', 'bidir.json'])
            verify.assert_not_called()

    def test_shared_frozen_assets_are_read_once_for_multiple_recipes(self):
        from dataclasses import replace
        from projects.minimax_h3.inference import artifact
        from projects.minimax_h3.modeling.ref_recipe import BIDIR_RECIPE, RefRecipe
        from projects.minimax_h3.tests.test_ref_bidir_artifact import assets, override_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, weights, encoder = assets(root)
            recipe = RefRecipe.from_training(cfg)
            paths = []
            for name, selected in [('fixed', recipe), ('bidir', replace(recipe, protocol=BIDIR_RECIPE))]:
                bundle = override_bundle(root, cfg, weights, encoder, selected.to_dict())
                bundle['encoders']['recipe']['encoder_code'] = {'historic.py': name}
                path = root/f'{name}.json'
                path.write_text(json.dumps(bundle))
                paths.append(path)
            with patch.object(artifact, 'sha256', wraps=artifact.sha256) as digest:
                self.assertEqual(len(load_artifacts(paths)), 2)
            self.assertEqual([Path(call.args[0]) for call in digest.call_args_list],
                             [weights, root/'encoder.bin'])

    def test_idle_and_delivery_timeout_drop_session_state(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = Harness(directory, timeout=.05, idle_timeout=.05)
            try:
                harness.exchange(start())
                deadline = time.monotonic()+2
                while (harness.controller.active is not None or harness.box.current is not None) and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertIsNone(harness.controller.active)
                self.assertIsNone(harness.box.current)
                harness.exchange(start('timeout'))
                job = harness.box.submit(chunk(sid='timeout', eos=True))
                self.assertTrue(job.ready.wait(2))
                self.assertTrue(job.released.wait(2))
                self.assertFalse(job.delivery_ok)
                self.assertIsNone(harness.controller.active)
            finally:
                harness.close()

    def test_binary_roundtrip_and_malformed_headers(self):
        self.assertEqual(unpack(pack(dict(name='RGB'), bytes(range(256)))), (dict(name='RGB'), bytes(range(256))))
        for body in (b'', struct.pack('!I', 65537), struct.pack('!I', 2)+b'[]',
                     struct.pack('!I', 13)+b'{"x":1,"x":2}', struct.pack('!I', 9)+b'{"x":NaN}'):
            with self.assertRaises(SessionError):
                unpack(body)

    def test_codec_reset_preserves_modules(self):
        from projects.minimax_h3.modeling.inference_runtime import InferenceCodec
        class Stream:
            def __init__(self):
                self.pending = object()
            def reset(self):
                self.pending = None
        codec = InferenceCodec.__new__(InferenceCodec)
        codec.decoder = object()
        codec.stream = Stream()
        original = codec.decoder
        codec.begin_session()
        self.assertIs(codec.decoder, original)
        self.assertIsNone(codec.stream.pending)

    def test_unpaced_producer_is_bounded_and_preserves_tail(self):
        import numpy as np
        frames = iter([np.zeros((1,1,3), dtype=np.uint8) for _ in range(39)])
        producer = FrameProducer(frames, 39, pace=None, queue_size=1)
        try:
            items = list(producer)
        finally:
            producer.close()
        self.assertEqual([len(item['capture_ns']) for item in items], [17, 17, 5])
        self.assertEqual([item['first_source_frame'] for item in items], [0, 17, 34])
        self.assertEqual([item['eos'] for item in items], [False, False, True])
        self.assertLessEqual(producer.max_queued, 1)

    def test_paced_source_overflow_fails_instead_of_waiting_for_consumer(self):
        import numpy as np
        producer = FrameProducer(iter([np.zeros((1,1,3), dtype=np.uint8) for _ in range(56)]),
                                 56, pace=100000, queue_size=1)
        producer.thread.start()
        self.assertTrue(producer.done.wait(3))
        self.assertRegex(str(producer.error), 'overflow')
        self.assertEqual(producer.queue.qsize(), 1)
        producer.close()


if __name__ == '__main__':
    unittest.main()
