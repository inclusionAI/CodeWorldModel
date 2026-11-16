import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np

from projects.minimax_h3.inference.dummy_streaming import DummySource, InputEvent, percentiles, summarize_dummy_run
from projects.minimax_h3.inference.serving import Request, SessionController, SessionError
from projects.minimax_h3.tests.test_serving import FakeBackend, artifacts, start
from projects.minimax_h3.tools.replay_ref import FrameProducer, run_session


class Client:
    def __init__(self, before_chunk=None, corrupt=None):
        self.controller = SessionController(artifacts())
        self.backend = FakeBackend()
        self.before_chunk, self.corrupt = before_chunk, corrupt

    def call(self, path, metadata, pixels=b''):
        sent = time.perf_counter_ns()
        op = 'start' if path=='/session' else path.rsplit('/', 1)[-1]
        request = Request(op, metadata, pixels)
        prepared = self.controller.prepare(request)
        if op == 'chunk' and self.before_chunk:
            self.before_chunk(metadata['chunk_id'])
        rgb, timings = self.backend.execute(op, prepared, pixels)
        reply = self.controller.finish(request, rgb, timings)
        if op == 'chunk' and self.corrupt:
            self.corrupt(reply)
        received = time.perf_counter_ns()
        return reply, dict(send_ns=sent, headers_ns=received, receive_ns=received)


def clip(count=32):
    return [np.full((1, 1, 3), index, dtype=np.uint8) for index in range(count)]


def read_rows(directory, name):
    return [json.loads(line) for line in (Path(directory)/name).read_text().splitlines()]


def run(directory, producer, source=None, client=None, **kwargs):
    return run_session(client or Client(), producer, first=clip(1)[0], start=start().metadata,
        spec=dict(start=0, num_frames=producer.count, height=1, width=1), output=directory,
        pace='unpaced' if producer.pace is None else '24', source=source, **kwargs)


class DummyStreamingTests(unittest.TestCase):
    def test_seek_pause_play_change_pixels_and_source_ids_keep_increasing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = DummySource(clip(4), [InputEvent('paused', 0, 2, False),
                InputEvent('play', 80, 3, True)], directory)
            source.start(time.perf_counter_ns())
            frames = source.frames()
            try:
                self.assertEqual([int(next(frames).pixels[0, 0, 0]) for _ in range(2)], [2, 2])
                source.thread.join(2)
                self.assertFalse(source.thread.is_alive())
                self.assertEqual([int(next(frames).pixels[0, 0, 0]) for _ in range(3)], [3, 0, 1])
            finally:
                frames.close()
                source.close()
            records = read_rows(directory, 'proxy_frames.jsonl')
            self.assertEqual([row['source_frame'] for row in records], list(range(5)))
            self.assertEqual([row['event_id'] for row in records], ['paused']*2+['play']*3)

    def test_scheduler_runs_while_producer_is_blocked_on_bounded_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            source = DummySource(clip(), [InputEvent('pause', 50, 7, False)], directory)
            source.start(time.perf_counter_ns())
            producer = FrameProducer(source.frames(), 90, pace=None, queue_size=1)
            producer.thread.start()
            try:
                source.thread.join(2)
                self.assertFalse(source.thread.is_alive())
                self.assertFalse(producer.done.is_set())
                self.assertEqual(read_rows(directory, 'events.jsonl')[0]['event_id'], 'pause')
                self.assertEqual(producer.queue.qsize(), 1)
            finally:
                producer.close()
                source.close()

    def test_supersession_and_unemitted_events_are_censored(self):
        with tempfile.TemporaryDirectory() as directory:
            source = DummySource(clip(), [InputEvent('superseded', 0, 0, True),
                InputEvent('used', 0, 2, False), InputEvent('future', 100000, 4, True)], directory)
            producer = FrameProducer(source.frames(), 5, pace=None)
            run(directory, producer, source, save_rgb=False)
            summary = summarize_dummy_run(directory)
            self.assertEqual([row['status'] for row in summary['events']], ['superseded', 'returned', 'unemitted'])
            self.assertEqual(summary['event_counts']['pending_censored'], 2)
            self.assertEqual(summary['events'][1]['first_source_frame'], 0)
            self.assertFalse((Path(directory)/'rgb.raw').exists())
            self.assertEqual(summary['received_rgb_bytes'], 15)

    def test_cross_chunk_13_21_5_association_and_ordinary_replay_contract(self):
        for dummy in (False, True):
            with self.subTest(dummy=dummy), tempfile.TemporaryDirectory() as directory:
                source = DummySource(clip(), [InputEvent('initial', 0, 1, True),
                    InputEvent('cross_chunk', 50, 2, False)], directory) if dummy else None
                def gated_frames():
                    frames = source.frames()
                    for index in range(39):
                        if index == 14:
                            source.thread.join(2)
                            self.assertFalse(source.thread.is_alive())
                        yield next(frames)
                producer = FrameProducer(gated_frames() if source else iter(clip(39)), 39, pace=None)
                report = run(directory, producer, source)
                requests = read_rows(directory, 'requests.jsonl')[1:]
                self.assertEqual([row['server']['frame_count'] for row in requests], [13, 21, 5])
                frames = read_rows(directory, 'frames.jsonl')
                self.assertEqual([row['source_frame'] for row in frames], list(range(39)))
                self.assertEqual(frames[13]['chunk_id'], 0)
                self.assertEqual(report['frame_count'], 39)
                self.assertEqual((Path(directory)/'rgb.raw').read_bytes(), bytes([7])*117)
                self.assertEqual(report['rgb_sha256'], hashlib.sha256(bytes([7])*117).hexdigest())
                if dummy:
                    summary = summarize_dummy_run(directory)
                    self.assertEqual(summary['events'][0]['receive_ns'], frames[0]['receive_ns'])
                    self.assertEqual(summary['events'][1]['first_source_frame'], 14)
                    self.assertEqual(summary['events'][1]['receive_ns'], requests[1]['client']['receive_ns'])

    def test_entire_invalid_response_is_rejected_before_logging(self):
        for corrupt in (lambda reply: reply.metadata['source_frames'][-1].update(source_frame=999),
                        lambda reply: reply.metadata.update(width=2),
                        lambda reply: setattr(reply, 'pixels', reply.pixels[:-1])):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                producer = FrameProducer(iter(clip(5)), 5, pace=None)
                with self.assertRaises(SessionError):
                    run(directory, producer, client=Client(corrupt=corrupt))
                self.assertEqual(read_rows(directory, 'frames.jsonl'), [])
                self.assertIn('error', read_rows(directory, 'requests.jsonl')[-1])
                self.assertEqual(json.loads((Path(directory)/'report.json').read_text())['frame_count'], 0)

    def test_paced_overflow_preserves_received_prefix_and_censors_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = DummySource(clip(), [InputEvent('used', 0, 0, True),
                InputEvent('future', 100000, 1, False)], directory)
            in_rpc = threading.Event()
            def gated_frames():
                for index, frame in enumerate(source.frames()):
                    if index == 17:
                        if not in_rpc.wait(2):
                            raise RuntimeError('first RPC did not start')
                    yield frame
            producer = FrameProducer(gated_frames(), 90, pace=100000, queue_size=1)
            def wait_for_overflow(index):
                self.assertEqual(index, 0)
                in_rpc.set()
                self.assertTrue(producer.done.wait(2))
            with self.assertRaisesRegex(RuntimeError, 'overflow'):
                run(directory, producer, source, Client(before_chunk=wait_for_overflow), save_rgb=False)
            summary = summarize_dummy_run(directory)
            self.assertEqual(summary['source_counts']['returned'], 13)
            self.assertGreater(summary['source_counts']['unreturned_produced'], 0)
            self.assertEqual(summary['event_counts']['unemitted'], 1)
            self.assertTrue(summary['overload'])
            self.assertFalse(summary['sustainable_duration_censored'])
            self.assertFalse(source.thread.is_alive())
            self.assertFalse(producer.thread.is_alive())

    def test_fixed_wall_windows_include_zero_and_partial_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            source = DummySource(clip(), [InputEvent('initial', 0, 0, True)], directory)
            run(directory, FrameProducer(source.frames(), 5, pace=None), source, save_rgb=False)
            output = Path(directory)
            report = json.loads((output/'report.json').read_text())
            epoch = report['stream_start_ns']
            report['stream_end_ns'] = epoch+2_500_000_000
            (output/'report.json').write_text(json.dumps(report))
            for name in ('frames.jsonl', 'requests.jsonl'):
                records = read_rows(directory, name)
                for row in records:
                    if name == 'frames.jsonl':
                        row['receive_ns'] = epoch+1_100_000_000
                        row['capture_to_receive_ns'] = row['receive_ns']-row['capture_ns']
                    elif row['operation'] == 'chunk':
                        row['client']['receive_ns'] = epoch+1_100_000_000
                (output/name).write_text(''.join(json.dumps(row)+'\n' for row in records))
            summary = summarize_dummy_run(directory)
            self.assertEqual([row['frame_count'] for row in summary['wall_second_bins']], [0, 5, 0])
            self.assertEqual(summary['wall_second_bins'][-1]['duration_ns'], 500_000_000)
            self.assertEqual(summary['actual_rgb_fps']['fps'], 2)
            self.assertEqual(summary['no_new_frame_gap_ns']['max'], 1_400_000_000)
            self.assertEqual(percentiles(range(1, 21))['p95'], 19)

    def test_producer_waits_before_consuming_pixels(self):
        selected = []
        def frames():
            for frame in clip(5):
                selected.append(time.perf_counter_ns())
                yield frame
        epoch = time.perf_counter_ns()+30_000_000
        producer = FrameProducer(frames(), 5, pace=1000, start_ns=epoch)
        try:
            chunks = list(producer)
        finally:
            producer.close()
        self.assertGreaterEqual(selected[0], epoch)
        self.assertTrue(all(capture >= began for capture, began in zip(chunks[0]['capture_ns'], selected)))

    def test_input_validation_and_session_start_failure_summary(self):
        for args in (('a', float('nan'), 0, True), ('a', 0, -1, True), ('a', 0, 0, 1)):
            with self.assertRaises(ValueError):
                InputEvent(*args)
        with tempfile.TemporaryDirectory() as directory:
            source = DummySource(clip(), [InputEvent('e0', 0, 0, True)], directory)
            class FailedClient:
                def call(self, *args):
                    raise RuntimeError('original HTTP failure')
            with self.assertRaisesRegex(RuntimeError, 'original HTTP failure'):
                run(directory, FrameProducer(source.frames(), 5, pace=None), source, FailedClient())
            summary = summarize_dummy_run(directory)
            self.assertEqual(summary['event_counts']['unemitted'], 1)
            self.assertIsNone(summary['actual_rgb_fps']['denominator_ns'])

    def test_applied_event_is_unreturned_on_http_failure_and_original_error_survives_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            source = DummySource(clip(), [InputEvent('e0', 0, 0, True)], directory)
            producer = FrameProducer(source.frames(), 5, pace=None)
            close = producer.close
            cleanup_began = []
            def failed_close():
                cleanup_began.append(time.perf_counter_ns())
                close()
                raise RuntimeError('cleanup failed')
            producer.close = failed_close
            def failed_chunk(index):
                raise RuntimeError('original chunk failure')
            with self.assertRaisesRegex(RuntimeError, 'original chunk failure'):
                run(directory, producer, source, Client(before_chunk=failed_chunk))
            summary = summarize_dummy_run(directory)
            self.assertEqual(summary['event_counts']['applied'], 1)
            self.assertEqual(summary['event_counts']['unreturned'], 1)
            self.assertEqual(summary['errors']['cleanup_error'], 'cleanup failed')
            self.assertIsNone(summary['sustainable_duration_ns'])
            report = json.loads((Path(directory)/'report.json').read_text())
            self.assertLessEqual(report['stream_end_ns'], cleanup_began[0])
            self.assertGreaterEqual(report['cleanup_ns'], 0)
            report['pace'] = '24'
            (Path(directory)/'report.json').write_text(json.dumps(report))
            summary = summarize_dummy_run(directory)
            self.assertFalse(summary['overload'])
            self.assertTrue(summary['sustainable_duration_censored'])
            report['producer_error'] = 'paced producer overflow; no source frames were silently dropped'
            report['producer_error_ns'] = report['stream_end_ns']+1
            (Path(directory)/'report.json').write_text(json.dumps(report))
            summary = summarize_dummy_run(directory)
            self.assertFalse(summary['overload'])
            self.assertTrue(summary['sustainable_duration_censored'])


if __name__ == '__main__':
    unittest.main()
