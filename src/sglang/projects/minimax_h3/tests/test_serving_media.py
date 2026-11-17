import http.client
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tools.serve_web import bootstrap

bootstrap()

from projects.minimax_h3.inference.serving import Request, SessionController, SessionError, Worker
from projects.minimax_h3.inference.serving_http import Ledger, RefClient, SessionMailbox, ServingHTTPServer, unpack
from projects.minimax_h3.inference.serving_media import H264Encoder, MAX_RECORD, MEDIA_SETTINGS
from projects.minimax_h3.inference.media_waits import with_waits
from projects.minimax_h3.tests.test_serving import LocalGroup, artifacts, chunk, start
from projects.minimax_h3.tests.test_serving_occupier import Occupier


class Backend:
    def __init__(self):
        self.calls = []
        self.gate = None
        self.plan = None
        self.closed = 0

    def execute(self, op, metadata, pixels):
        self.calls.append((op, metadata.get('chunk_id'), time.perf_counter_ns()))
        if op != 'chunk':
            return b'', {}
        if self.gate is not None:
            assert self.gate.wait(3), 'test backend gate timed out'
        count = self.plan[metadata['chunk_id']] if self.plan else 5 if metadata['eos'] else 17
        return bytes([64]) * (16 * 16 * 3 * count), dict(cpu_ready_ns=time.perf_counter_ns())

    def close_session(self):
        self.closed += 1


class Harness:
    def __init__(self, *, timeout=2):
        self.directory = tempfile.TemporaryDirectory()
        self.ledger = Ledger(Path(self.directory.name) / 'server.jsonl')
        self.box = SessionMailbox(self.ledger, timeout=timeout)
        self.artifacts = artifacts()
        for bundle, _ in self.artifacts.values():
            bundle['encoders']['recipe'].update(width=16, height=16)
        health = dict(artifacts={key: dict(encoder_recipe=bundle['encoders']['recipe'])
            for key, (bundle, _) in self.artifacts.items()})
        self.server = ServingHTTPServer(('127.0.0.1', 0), self.box, health, timeout=timeout)
        self.media = self.server.media
        self.backend, self.occupier = Backend(), Occupier()
        self.controller = SessionController(self.artifacts)
        self.worker = Worker(self.backend, LocalGroup(), self.controller, self.box,
            idle_timeout=timeout, occupier=self.occupier)
        self.error = None
        self.worker_thread = threading.Thread(target=self.run, daemon=True)
        self.http_thread = threading.Thread(target=self.server.serve_forever,
            kwargs=dict(poll_interval=.01), daemon=True)
        self.worker_thread.start()
        self.http_thread.start()
        self.client = RefClient(f'http://127.0.0.1:{self.server.server_port}', timeout=timeout)

    def run(self):
        try:
            self.worker.serve()
        except BaseException as exc:
            self.error = exc
            self.box.fail(str(exc))

    def begin(self, sid='one', video=None, observe_waits=None):
        request = start(sid)
        request.pixels = bytes(16 * 16 * 3)
        if video is not None:
            request.metadata['video'] = video
        if observe_waits is not None:
            request.metadata['observe_waits'] = observe_waits
        job = self.media.start(request.metadata, request.pixels)
        self.admitted(job)
        assert job.released.wait(2), 'start did not release'
        return job.reply

    def admitted(self, job):
        assert job.ready.wait(2), 'request did not reach worker preparation'
        self.box.acknowledge(job, True)
        return job.reply

    def submit(self, request):
        return self.media.submit(request.metadata, request.pixels)

    def close(self):
        if self.backend.gate is not None:
            self.backend.gate.set()
        self.box.stop()
        self.worker_thread.join(3)
        self.server.shutdown()
        self.server.server_close()
        self.http_thread.join(3)
        self.ledger.close()
        self.directory.cleanup()
        assert not self.worker_thread.is_alive(), 'worker did not shut down'
        assert self.error is None, self.error


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()

    def test_pending_peek_preserves_admission_and_cancel_releases_job(self):
        from copy import deepcopy
        h = self.h
        h.begin()
        h.backend.gate = threading.Event()
        current = h.submit(chunk())
        h.admitted(current)
        self.assertIsNone(h.box.peek_pending_chunk())
        following = h.submit(chunk(1, eos=True))
        snapshot = deepcopy(h.controller.active)
        checked, payload = h.worker.pending_proxy()
        self.assertEqual(checked, following.request.metadata)
        self.assertEqual(payload, following.request.pixels)
        self.assertEqual(h.controller.active, snapshot)
        self.assertIs(h.box.pending, following)
        self.assertFalse(following.reserved)
        self.assertFalse(following.ready.is_set())
        following.request.metadata['chunk_id'] = 7
        self.assertIsNone(h.worker.pending_proxy())
        following.request.metadata['chunk_id'] = 1
        h.media.abort(current.media, 'cancel pending prefetch')
        self.assertIsNone(h.box.peek_pending_chunk())
        h.backend.gate.set()
        self.assertTrue(current.released.wait(2))
        self.assertTrue(following.ready.wait(2))
        self.assertTrue(following.released.wait(2))

    def tearDown(self):
        self.h.close()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(predicate())

    def test_wait_observation_flag_and_owner_handoffs(self):
        for value in (1, 'true', [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(SessionError, 'observe_waits'):
                self.h.begin(observe_waits=value)
        response = self.h.begin()
        self.assertFalse(response.metadata['wait_observability']['enabled'])
        self.assertNotIn('waits', response.metadata)
        self.h.media.cancel(dict(session_id='one'))
        response = self.h.begin('observed', observe_waits=True)
        self.assertTrue(response.metadata['wait_observability']['enabled'])
        self.assertEqual(response.metadata['waits']['schema'], 'stream_waits/v1')
        times = response.metadata['waits']['frontend']
        self.assertLessEqual(times['request_complete_ns'], times['enqueued_ns'])
        self.assertLessEqual(times['enqueued_ns'], times['taken_ns'])
        self.assertLessEqual(times['taken_ns'], times['prepared_ns'])
        self.assertNotIn('observe_waits', self.h.controller.active.__dict__)

    def test_observation_budget_omits_diagnostics_without_losing_media_metadata(self):
        counts = {}
        original = dict(op='video_frame', output_frame=1, waits=dict(schema='stream_waits/v1',
            frontend=dict(unexpected='x' * 2048)))
        result = with_waits(original, 'gateway', dict(send_lock=dict(start_ns=1, end_ns=2, outcome='complete')), counts)
        self.assertNotIn('waits', result)
        self.assertEqual(result, dict(op='video_frame', output_frame=1))
        self.assertEqual(counts['omitted'], 1)
        self.assertIn('waits', original, 'ownership transfer must not mutate the previous receipt')
        bounded = with_waits(dict(op='video_frame', waits=dict(schema='future')), 'gateway',
            dict(send_lock=dict(start_ns=1, end_ns=2, outcome='complete')), counts)
        self.assertEqual(counts['unsupported'], 1)
        self.assertEqual(bounded['waits']['schema'], 'stream_waits/v1')

    def test_injected_mailbox_encoder_rgb_slot_and_publication_waits(self):
        self.h.begin(observe_waits=True)
        session = self.h.media.attach('one')
        self.h.backend.gate = threading.Event()
        encoder_entered, encoder_gate = threading.Event(), threading.Event()
        original_encode = session.encoder.encode
        def encode(rgb, ordinal):
            if ordinal == 0:
                encoder_entered.set()
                assert encoder_gate.wait(2)
            return original_encode(rgb, ordinal)
        session.encoder.encode = encode
        first = self.h.submit(chunk(0))
        self.h.admitted(first)
        second = self.h.submit(chunk(1))
        time.sleep(.04)
        self.h.backend.gate.set()
        self.assertTrue(encoder_entered.wait(2))
        second_receipt = self.h.admitted(second)
        third = self.h.submit(chunk(2, eos=True))
        self.wait_for(lambda: self.h.box.current is third and len(session.results) == 1)
        time.sleep(.04)
        encoder_gate.set()
        self.wait_for(lambda: len(session.events) == 4)
        time.sleep(.04)
        collected = []
        def read():
            while True:
                event = self.h.media.next_event(session)
                collected.append(event)
                if event.metadata['op'] in ('complete', 'error'):
                    self.h.media.detach(session, event.metadata['op'] == 'complete')
                    return
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        admitted = self.h.admitted(third)
        reader.join(2)
        self.assertFalse(reader.is_alive())
        self.assertEqual(collected[-1].metadata['op'], 'complete')
        admission = second_receipt.metadata['waits']['frontend']
        self.assertGreater(admission['taken_ns'] - admission['enqueued_ns'], 20_000_000)
        slot = admitted.metadata['waits']['frontend']['rgb_reserve']
        self.assertGreater(slot['end_ns'] - slot['start_ns'], 20_000_000)
        batches = [row.metadata for row in collected if row.metadata['op'] == 'batch']
        queue = batches[1]['waits']['frontend']
        self.assertGreater(queue['rgb_taken_ns'] - queue['rgb_enqueued_ns'], 20_000_000)
        waits = [row.metadata['waits']['frontend'] for row in collected]
        self.assertTrue(any(row['publication']['end_ns'] - row['publication']['start_ns'] > 20_000_000 for row in waits))
        self.assertTrue(all(row['publication']['end_ns'] <= row['event_dequeued_ns'] for row in waits))
        self.assertTrue(all(row['publication']['outcome'] == 'complete' for row in waits))
        self.assertEqual(collected[-1].metadata['wait_coverage']['frontend']['omitted'], 0)
        self.assertEqual(session.reservations, 0)
        self.assertEqual([row.metadata['source_frame'] for row in collected if row.metadata['op'] == 'video_frame'], list(range(39)))

    def test_observed_cancel_reports_interrupted_and_discarded_queues(self):
        self.h.begin(observe_waits=True)
        session = self.h.media.attach('one')
        self.h.admitted(self.h.submit(chunk(0)))
        self.h.admitted(self.h.submit(chunk(1)))
        self.wait_for(lambda: len(session.events) == 4 and len(session.results) == 1)
        third = self.h.submit(chunk(2))
        self.assertFalse(third.ready.wait(.04))
        result = self.h.media.cancel(dict(session_id='one'))
        coverage = result.metadata['wait_coverage']
        self.assertTrue(coverage['frontend_complete'])
        self.assertGreaterEqual(coverage['frontend']['discarded'], 5)
        self.assertGreaterEqual(coverage['frontend']['interrupted'], 1)
        self.assertTrue(any(value.get('outcome') == 'cancelled'
            for value in result.metadata['waits']['frontend'].values() if isinstance(value, dict)))
        self.assertEqual(session.reservations, 0)
        self.assertFalse(self.h.begin('two').metadata['wait_observability']['enabled'])

    def test_actual_encoder_annexb_baseline_decode_and_bitrate_validation(self):
        import av
        encoder = H264Encoder(16, 16, 500_000)
        decoder = av.CodecContext.create('h264', 'r')
        try:
            payload, key = encoder.encode(bytes([80]) * (16 * 16 * 3), 0)
            self.assertTrue(key)
            self.assertIn(b'\x00\x00\x00\x01\x67\x42\xc0\x28', payload)
            frames = decoder.decode(av.Packet(payload))
            self.assertEqual([(f.width, f.height) for f in frames], [(16, 16)])
            encoder.flush()
        finally:
            encoder.close()
        for value in (True, 499_999, 20_000_001, 8_000_000.0, '8000000'):
            with self.subTest(value=value), self.assertRaises(SessionError):
                self.h.begin(video=dict(bitrate_bps=value))
        reply = self.h.begin(video=dict(bitrate_bps=20_000_000))
        self.assertEqual(reply.metadata['media']['bitrate_bps'], 20_000_000)
        self.assertNotIn('video', self.h.box.current.request.metadata if self.h.box.current else {})
        self.assertEqual(self.h.server.health['media'], MEDIA_SETTINGS)

    def test_encoder_borrows_exact_rgb_for_even_widths_and_buffer_types(self):
        import av
        import numpy as np

        for width in (16, 18, 32, 34):
            source = bytes((index * 17 + 31) % 256 for index in range(width * 2 * 3))
            for rgb in (source, memoryview(source), bytearray(source), memoryview(bytearray(source))):
                with self.subTest(width=width, kind=type(rgb).__name__, readonly=memoryview(rgb).readonly):
                    frames = []

                    class Codec:
                        def encode(self, frame):
                            frames.append(frame)
                            return [av.Packet(b'\x01')]

                    encoder = object.__new__(H264Encoder)
                    encoder.av, encoder.width, encoder.height, encoder.codec = av, width, 2, Codec()
                    self.assertEqual(encoder.encode(rgb, 7)[0], b'\x01')
                    frame = frames[0]
                    self.assertEqual((frame.width, frame.height, frame.pts), (width, 2, 7))
                    self.assertEqual(frame.planes[0].line_size, width * 3)
                    np.testing.assert_array_equal(frame.to_ndarray(),
                        np.frombuffer(source, dtype=np.uint8).reshape(2, width, 3))
                    self.assertTrue(np.shares_memory(frame.to_ndarray(), np.frombuffer(rgb, dtype=np.uint8)))
                    self.assertEqual(bytes(rgb), source)

    def test_compute_overlaps_delivery_and_saturation_cancel_restores_occupier(self):
        self.h.begin()
        session = self.h.media.attach('one')
        first = self.h.submit(chunk(0))
        self.assertEqual(self.h.admitted(first).metadata['status'], 'admitted')
        second = self.h.submit(chunk(1))
        self.h.admitted(second)
        self.wait_for(lambda: len(session.events) == 4 and len(session.results) == 1)
        self.assertEqual(len([c for c in self.h.backend.calls if c[0] == 'chunk']), 2)
        self.assertEqual(session.reservations, 2)
        third = self.h.submit(chunk(2))
        self.assertFalse(third.ready.wait(.05))
        fourth = self.h.submit(chunk(3))
        with self.assertRaisesRegex(SessionError, 'in flight'):
            self.h.submit(chunk(4))
        self.assertEqual(len([c for c in self.h.backend.calls if c[0] == 'chunk']), 2)
        self.assertEqual(self.h.media.cancel(dict(session_id='one')).metadata['status'], 'cancelled')
        self.assertTrue(third.ready.is_set())
        self.assertTrue(fourth.ready.is_set())
        self.assertEqual(fourth.reply.status, 409)
        self.assertEqual(session.reservations, 0)
        self.assertFalse(self.h.occupier.paused)
        self.assertEqual(self.h.backend.closed, 1)
        self.assertEqual(self.h.media.cancel(dict(session_id='one')).metadata['status'], 'cancelled')
        self.assertEqual(self.h.begin('two').metadata['session_id'], 'two')

    def test_admission_precedes_compute_and_cancel_does_not_preempt_compute(self):
        self.h.begin()
        self.h.media.attach('one')
        self.h.backend.gate = threading.Event()
        job = self.h.submit(chunk())
        self.assertEqual(self.h.admitted(job).metadata['status'], 'admitted')
        self.assertFalse(job.released.is_set())
        replies = []
        cancel = threading.Thread(target=lambda: replies.append(self.h.media.cancel(dict(session_id='one'))))
        cancel.start()
        self.assertFalse(job.released.wait(.05))
        self.assertTrue(self.h.occupier.paused)
        self.assertTrue(cancel.is_alive())
        self.h.backend.gate.set()
        cancel.join(2)
        self.assertFalse(cancel.is_alive())
        self.assertEqual(replies[0].metadata['status'], 'cancelled')
        self.assertFalse(self.h.occupier.paused)

    def test_zero_frame_and_26_frame_batch_keep_authoritative_sources(self):
        import av
        self.h.backend.plan = [0, 26, 13]
        self.h.begin(observe_waits=True)
        session = self.h.media.attach('one')
        collected, errors = [], []
        def read():
            try:
                while True:
                    event = self.h.media.next_event(session)
                    collected.append(event)
                    if event.metadata['op'] in ('complete', 'error'):
                        self.h.media.detach(session, event.metadata['op'] == 'complete')
                        return
            except Exception as exc:
                errors.append(exc)
        reader = threading.Thread(target=read)
        reader.start()
        for index in range(3):
            self.h.admitted(self.h.submit(chunk(index, eos=index == 2)))
        reader.join(2)
        self.assertFalse(reader.is_alive())
        self.assertEqual(errors, [])
        batches = [event.metadata for event in collected if event.metadata['op'] == 'batch']
        self.assertEqual([row['frame_count'] for row in batches], [0, 26, 13])
        self.assertTrue(all(row['waits']['frontend']['rgb_enqueued_ns'] <= row['waits']['frontend']['rgb_taken_ns'] for row in batches))
        self.assertEqual([row['output_id'] for row in batches], [0, 1, 2])
        frames = [event for event in collected if event.metadata['op'] == 'video_frame']
        self.assertEqual([row.metadata['source_frame'] for row in frames], list(range(39)))
        self.assertEqual([row.metadata['capture_ns'] for row in frames], list(range(39)))
        self.assertEqual([row.metadata['output_frame'] for row in frames], list(range(39)))
        self.assertEqual(frames[0].metadata['chunk_id'], 1)
        self.assertEqual(frames[0].metadata['source']['chunk_id'], 0)
        decoder = av.CodecContext.create('h264', 'r')
        self.assertEqual(sum(len(decoder.decode(av.Packet(row.pixels))) for row in frames), 39)
        self.assertEqual(collected[-1].metadata['op'], 'complete')
        self.assertTrue(all(not row.pixels for row in collected if row.metadata['op'] == 'batch'))
        self.assertEqual(session.reservations, 0)
        self.wait_for(lambda: self.h.media.active is None)
        self.assertFalse(self.h.occupier.paused)

    def test_persistent_http_records_and_raw_session_exclusion(self):
        request = start()
        request.metadata['video'] = dict(bitrate_bps=800_000)
        reply, _ = self.h.client.call('/media/session', request.metadata, bytes(16 * 16 * 3))
        self.assertEqual(reply.metadata['media']['bitrate_bps'], 800_000)
        connection = http.client.HTTPConnection('127.0.0.1', self.h.server.server_port, timeout=2)
        connection.request('GET', '/media/session/one/events')
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIsNone(response.getheader('Content-Length'))
        with self.assertRaises(SessionError):
            self.h.client.call('/session', start('raw').metadata, bytes(16 * 16 * 3))
        request = chunk(eos=True)
        admitted, _ = self.h.client.call('/media/session/one/chunk', request.metadata, request.pixels)
        self.assertEqual(admitted.metadata['status'], 'admitted')
        events = []
        try:
            while True:
                size, = struct.unpack('!I', response.read(4))
                self.assertLessEqual(size, MAX_RECORD)
                metadata, pixels = unpack(response.read(size))
                events.append((metadata, pixels))
                if metadata['op'] in ('error', 'complete'):
                    break
        finally:
            connection.close()
        self.assertEqual([row[0]['op'] for row in events], ['batch'] + ['video_frame'] * 5 + ['complete'])
        self.wait_for(lambda: self.h.media.active is None)
        self.assertEqual(self.h.begin('another').metadata['status'], 'accepted')

    def test_detach_and_invalid_chunk_discard_pending_inputs(self):
        self.h.begin()
        session = self.h.media.attach('one')
        with self.assertRaisesRegex(SessionError, 'already has a reader'):
            self.h.media.attach('one')
        request = chunk(1)
        rejected = self.h.submit(request)
        self.assertEqual(self.h.admitted(rejected).status, 409)
        self.wait_for(lambda: self.h.media.active is None)
        self.assertFalse(self.h.occupier.paused)
        self.h.begin('two')
        session = self.h.media.attach('two')
        self.h.media.detach(session, False)
        self.wait_for(lambda: self.h.media.active is None)
        self.assertFalse(self.h.occupier.paused)


class MediaTimeoutTests(unittest.TestCase):
    def test_cancel_before_worker_takes_start_does_not_wait_for_model_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory) / 'server.jsonl')
            box = SessionMailbox(ledger, timeout=.2)
            health = dict(artifacts={'artifact': dict(encoder_recipe=dict(width=16, height=16))})
            server = ServingHTTPServer(('127.0.0.1', 0), box, health, timeout=.2)
            try:
                job = server.media.start(dict(session_id='early', artifact_id='artifact'), b'')
                reply = server.media.cancel(dict(session_id='early'))
                self.assertEqual(reply.metadata['status'], 'cancelled')
                self.assertTrue(job.released.is_set())
                self.assertIsNone(box.pending)
                self.assertIsNone(box.control)
                self.assertIsNone(server.media.active)
            finally:
                server.server_close()
                ledger.close()

    def test_server_close_after_worker_failure_does_not_wait_for_dead_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory) / 'server.jsonl')
            box = SessionMailbox(ledger, timeout=.2)
            health = dict(artifacts={'artifact': dict(encoder_recipe=dict(width=16, height=16))})
            server = ServingHTTPServer(('127.0.0.1', 0), box, health, timeout=.2)
            server.media.start(dict(session_id='failed', artifact_id='artifact'), b'')
            box.take(None, None)
            session = server.media.active
            box.fail('worker exited unexpectedly')
            try:
                server.server_close()
                self.assertTrue(session.encoder_done.is_set())
                self.assertFalse(session.worker_done.is_set())
            finally:
                ledger.close()

    def test_orphan_attach_timeout_cleans_up_and_allows_restart(self):
        harness = Harness(timeout=.15)
        try:
            harness.begin()
            session = harness.media.active
            self.assertTrue(session.worker_done.wait(2))
            self.assertTrue(session.encoder_done.wait(2))
            self.assertFalse(harness.occupier.paused)
            self.assertEqual(harness.begin('two').metadata['session_id'], 'two')
        finally:
            harness.close()

    def test_missing_pyav_keeps_raw_routes_available(self):
        with patch.object(H264Encoder, '__init__', side_effect=ImportError('av')):
            harness = Harness()
        try:
            self.assertNotIn('media', harness.server.health)
            with self.assertRaisesRegex(SessionError, 'unavailable'):
                harness.begin()
            request = start()
            reply, _ = harness.client.call('/session', request.metadata, bytes(16 * 16 * 3))
            self.assertEqual(reply.metadata['status'], 'accepted')
            harness.client.call('/session/one/cancel', dict(session_id='one'))
        finally:
            harness.close()


if __name__ == '__main__':
    unittest.main()
