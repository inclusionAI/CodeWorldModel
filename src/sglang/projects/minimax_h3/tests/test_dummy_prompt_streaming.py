from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from projects.minimax_h3.inference.dummy_streaming import (
    CapturedFrame, DummySource, PromptEvent, PromptSnapshot, summarize_dummy_run)
from projects.minimax_h3.inference.serving import Reply, SessionError
from projects.minimax_h3.tests.test_dummy_streaming import clip, read_rows
from projects.minimax_h3.tools import dummy_stream_ref
from projects.minimax_h3.tools.replay_ref import FrameProducer, run_session
from projects.minimax_h3.modeling.ref_recipe import FIXED_RECIPE, FIXED_STREAMING_TEXT_RECIPE


class PromptClient:
    """Wire peer with delayed RGB; no encoder or controller implementation is shared."""
    def __init__(self, counts=(13, 21, 5), before_chunk=None, corrupt=None):
        self.counts, self.before_chunk, self.corrupt = counts, before_chunk, corrupt
        self.requests = []
        self.pending = []
        self.applied = None
        self.accepted = self.returned = self.cancelled = 0

    def call(self, path, metadata, pixels=b''):
        sent = time.perf_counter_ns()
        if path == '/session':
            self.pending.clear()
            self.applied = None
            self.accepted = self.returned = 0
            reply = Reply(dict(session_id=metadata['session_id'], artifact_id=metadata['artifact_id'],
                status='accepted', recipe=dict(protocol=FIXED_STREAMING_TEXT_RECIPE)))
        elif path.endswith('/cancel'):
            self.cancelled += 1
            self.pending.clear()
            reply = Reply(dict(status='cancelled'))
        else:
            index = metadata['chunk_id']
            self.requests.append(deepcopy(metadata))
            if self.before_chunk is not None:
                self.before_chunk(index, metadata)
            prompt = metadata['prompt']
            if self.applied is None or self.applied['text'] != prompt['text']:
                self.applied = dict(text=prompt['text'], sha256=hashlib.sha256(prompt['text'].encode()).hexdigest(),
                    first_chunk=index, rows=len(prompt['text']))
            for offset, capture in enumerate(metadata['capture_ns']):
                self.pending.append(dict(chunk_id=index, source_frame=metadata['first_source_frame']+offset,
                    capture_ns=capture, prompt_revision=prompt['revision'], prompt_sha256=self.applied['sha256'],
                    prompt_first_chunk=self.applied['first_chunk'], prompt_rows=self.applied['rows']))
            self.accepted += len(metadata['capture_ns'])
            count = self.counts[index]
            rows, self.pending = self.pending[:count], self.pending[count:]
            reply = Reply(dict(session_id=metadata['session_id'], chunk_id=index, output_id=index,
                available_after_chunk=index, frame_count=count, height=1, width=1,
                output_frame_start=self.returned, total_frames=self.returned+count, accepted_frames=self.accepted,
                source_frames=rows, status='complete' if metadata['eos'] else 'accepted',
                submitted_prompt=dict(prompt), applied_prompt=dict(self.applied)), bytes([7])*count*3)
            self.returned += count
            if self.corrupt is not None:
                self.corrupt(index, reply)
        received = time.perf_counter_ns()
        return reply, dict(send_ns=sent, headers_ns=received, receive_ns=received)


def run(directory, producer, client, snapshot, source=None):
    return run_session(client, producer, first=clip(1)[0],
        start=dict(session_id='stream', artifact_id='streaming', prompt='frozen system', window_size=6,
            seed=7, condition_seed=7), spec=dict(start=0, num_frames=producer.count, height=1, width=1),
        output=directory, pace='unpaced', source=source, prompt_snapshot=snapshot)


class PromptProducerTests(unittest.TestCase):
    def test_snapshot_validation_and_immutable_identity(self):
        for values in ((True, 'A', 0), (0, 'A', 0), (2**63, 'A', 0), (1, '', 0),
                (1, '\ud800', 0), (1, 'a'*16385, 0), (1, 'A', True), (1, 'A', -1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PromptSnapshot(*values)
        with self.assertRaises(FrozenInstanceError):
            PromptSnapshot(1, 'A', 0).text = 'changed'
        for offset in (-1, True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                PromptEvent(1, offset, 'A')
        for events in ([PromptEvent(1, 1, 'A')], [PromptEvent(2, 0, 'A'), PromptEvent(1, 1, 'B')],
                [PromptEvent(1, 0, 'A'), PromptEvent(1, 0, 'B')],
                [PromptEvent(1, 0, 'A'), PromptEvent(2, 2, 'B'), PromptEvent(3, 1, 'C')]):
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                DummySource(clip(), [], directory, prompt_events=events)

    def test_prompt_changes_during_blocked_http_and_superseded_events_remain_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = DummySource(clip(), [], directory, prompt_events=[PromptEvent(1, 0, 'A'),
                PromptEvent(2, 60, 'unused'), PromptEvent(3, 60, 'B'), PromptEvent(4, 100000, 'future')])
            producer = FrameProducer(source.frames(), 39, pace=None, queue_size=1)
            def blocked(index, metadata):
                if index == 0:
                    self.assertEqual(metadata['prompt']['revision'], 1)
                    deadline = time.monotonic()+2
                    while source.prompt_snapshot().revision != 3 and time.monotonic() < deadline:
                        time.sleep(.001)
                    self.assertEqual(source.prompt_snapshot().revision, 3)
                    self.assertEqual(metadata['prompt']['revision'], 1)
            client = PromptClient(before_chunk=blocked)
            calls = []
            def snapshot():
                calls.append(True)
                return source.prompt_snapshot()
            report = run(directory, producer, client, snapshot, source)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(calls), 3)
            self.assertEqual([row['prompt']['revision'] for row in client.requests], [1, 3, 3])
            self.assertEqual([row['revision'] for row in read_rows(directory, 'prompt_events.jsonl')], [1, 2, 3])
            frames = read_rows(directory, 'frames.jsonl')
            self.assertEqual([row['prompt_revision'] for row in frames], [1]*17+[3]*22)
            summary = summarize_dummy_run(directory)
            self.assertEqual([row['status'] for row in summary['prompt_events']],
                ['returned', 'superseded_before_submission', 'returned', 'unemitted'])
            self.assertEqual(summary['prompt_events'][2]['receive_ns'], frames[17]['receive_ns'])
            self.assertEqual(summary['prompt_events'][2]['prompt_to_associated_rgb_receive_ns'],
                frames[17]['receive_ns']-summary['prompt_events'][2]['emitted_ns'])
            self.assertEqual(summary['prompt_event_counts']['pending_censored'], 2)
            for row in read_rows(directory, 'requests.jsonl')[1:]:
                self.assertLessEqual(row['input']['prompt']['emitted_ns'], row['prompt_frozen_ns'])
                self.assertLessEqual(row['prompt_frozen_ns'], row['client']['send_ns'])
            self.assertFalse(source.thread.is_alive())
            self.assertFalse(producer.thread.is_alive())

    def test_same_text_new_revision_and_a_b_a_keep_originating_chunk_identity(self):
        for texts, first_chunks in ((('A', 'A', 'B'), [0, 0, 2]), (('A', 'B', 'A'), [0, 1, 2])):
            with self.subTest(texts=texts), tempfile.TemporaryDirectory() as directory:
                snapshots = iter(PromptSnapshot(index+1, text, index) for index, text in enumerate(texts))
                client = PromptClient()
                run(directory, FrameProducer(iter(clip(39)), 39, pace=None), client, lambda: next(snapshots))
                requests = read_rows(directory, 'requests.jsonl')[1:]
                self.assertEqual([row['server']['frame_count'] for row in requests], [13, 21, 5])
                self.assertEqual([row['server']['applied_prompt']['first_chunk'] for row in requests], first_chunks)
                second = requests[1]['server']['source_frames']
                self.assertEqual([row['prompt_revision'] for row in second], [1]*4+[2]*17)
                self.assertEqual([row['prompt_first_chunk'] for row in second], [first_chunks[0]]*4+[first_chunks[1]]*17)
                self.assertEqual([row['prompt_revision'] for row in read_rows(directory, 'frames.jsonl')],
                    [1]*17+[2]*17+[3]*5)

    def test_zero_frame_receipt_retains_applied_identity_for_later_response(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshots = iter([PromptSnapshot(1, 'A', 0), PromptSnapshot(2, 'B', 1), PromptSnapshot(3, 'A', 2)])
            client = PromptClient(counts=(0, 34, 5))
            run(directory, FrameProducer(iter(clip(39)), 39, pace=None), client, lambda: next(snapshots))
            requests = read_rows(directory, 'requests.jsonl')[1:]
            self.assertEqual(requests[0]['server']['frame_count'], 0)
            self.assertEqual(requests[0]['server']['applied_prompt']['first_chunk'], 0)
            self.assertEqual([row['prompt_revision'] for row in requests[1]['server']['source_frames']], [1]*17+[2]*17)

    def test_changed_or_decreasing_revision_is_rejected_before_submission(self):
        for second in (PromptSnapshot(1, 'A', 0), PromptSnapshot(2, 'B', 0), PromptSnapshot(2, 'A', 1)):
            with self.subTest(second=second), tempfile.TemporaryDirectory() as directory:
                snapshots = iter([PromptSnapshot(2, 'A', 0), second])
                client = PromptClient()
                with self.assertRaisesRegex(ValueError, 'prompt revisions'):
                    run(directory, FrameProducer(iter(clip(39)), 39, pace=None), client, lambda: next(snapshots))
                self.assertEqual(len(client.requests), 1)
                self.assertEqual(len(read_rows(directory, 'frames.jsonl')), 13)
                self.assertEqual(client.cancelled, 1)

    def test_bad_prompt_receipt_never_persists_any_rgb_from_that_response(self):
        mutations = [lambda m: m['submitted_prompt'].update(revision=999),
            lambda m: m['submitted_prompt'].update(revision=True),
            lambda m: m['applied_prompt'].update(sha256='bad'),
            lambda m: m['applied_prompt'].update(first_chunk=True),
            lambda m: m['applied_prompt'].update(rows=0),
            lambda m: m['applied_prompt'].update(rows=4097),
            lambda m: m['source_frames'][-1].update(prompt_revision=999),
            lambda m: m['source_frames'][-1].update(prompt_rows=999),
            lambda m: m['source_frames'][0].update(prompt_first_chunk=1)]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                def corrupt(index, reply):
                    if index == 1:
                        mutate(reply.metadata)
                snapshots = iter([PromptSnapshot(1, 'A', 0), PromptSnapshot(2, 'B', 1)])
                client = PromptClient(corrupt=corrupt)
                producer = FrameProducer(iter(clip(39)), 39, pace=None)
                with self.assertRaises(SessionError):
                    run(directory, producer, client, lambda: next(snapshots))
                self.assertEqual(len(read_rows(directory, 'frames.jsonl')), 13)
                self.assertEqual((Path(directory)/'rgb.raw').read_bytes(), bytes([7])*39)
                self.assertIn('error', read_rows(directory, 'requests.jsonl')[-1])
                self.assertEqual(client.cancelled, 1)
                self.assertFalse(producer.thread.is_alive())

    def test_timestamp_is_preserved_and_replay_arrays_still_work(self):
        for captured in (True, False):
            with self.subTest(captured=captured):
                frames = [CapturedFrame(frame, 100+index) if captured else frame for index, frame in enumerate(clip(5))]
                producer = FrameProducer(iter(frames), 5, pace=None)
                try:
                    chunks = list(producer)
                finally:
                    producer.close()
                self.assertEqual(chunks[0]['pixels'], np.stack(clip(5)).tobytes())
                if captured:
                    self.assertEqual(chunks[0]['capture_ns'], [100, 101, 102, 103, 104])
                else:
                    self.assertTrue(all(value > 104 for value in chunks[0]['capture_ns']))

    def test_close_unblocks_acquisition_before_join_and_preserves_original_failure(self):
        class BlockingSource:
            def __init__(self):
                self.blocked, self.closed = threading.Event(), threading.Event()
            def start(self, epoch):
                pass
            def frames(self):
                yield from clip(17)
                self.blocked.set()
                self.closed.wait(2)
                if not self.closed.is_set():
                    raise RuntimeError('acquisition was not closed')
            def close(self):
                self.closed.set()
        with tempfile.TemporaryDirectory() as directory:
            source = BlockingSource()
            producer = FrameProducer(source.frames(), 39, pace=None)
            def fail(index, metadata):
                self.assertTrue(source.blocked.wait(2))
                raise RuntimeError('HTTP failed')
            client = PromptClient(before_chunk=fail)
            with self.assertRaisesRegex(RuntimeError, 'HTTP failed'):
                run(directory, producer, client, lambda: PromptSnapshot(1, 'A', 0), source)
            self.assertTrue(source.closed.is_set())
            self.assertFalse(producer.thread.is_alive())
            self.assertEqual(client.cancelled, 1)
            report = json.loads((Path(directory)/'report.json').read_text())
            self.assertIsNone(report['producer_error'])

    def test_dummy_close_during_capture_does_not_write_to_closed_log(self):
        copying, release = threading.Event(), threading.Event()
        class SlowFrame(np.ndarray):
            def copy(self):
                copying.set()
                if not release.wait(2):
                    raise RuntimeError('copy was not released')
                return super().copy()
        with tempfile.TemporaryDirectory() as directory:
            source = DummySource([clip(1)[0].view(SlowFrame)], [], directory)
            source.start(time.perf_counter_ns())
            producer = FrameProducer(source.frames(), 5, pace=None)
            producer.thread.start()
            try:
                self.assertTrue(copying.wait(2))
                producer.stop.set()
                source.close()
            finally:
                release.set()
                producer.close()
                source.close()
            self.assertIsNone(producer.error)
            self.assertEqual(read_rows(directory, 'proxy_frames.jsonl'), [])

    def test_eos_with_missing_frames_rejects_whole_response_and_cancels(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = FrameProducer(iter(clip(5)), 5, pace=None)
            client = PromptClient(counts=(4,))
            with self.assertRaisesRegex(SessionError, 'EOS did not complete'):
                run(directory, producer, client, lambda: PromptSnapshot(1, 'A', 0))
            self.assertEqual(read_rows(directory, 'frames.jsonl'), [])
            self.assertEqual((Path(directory)/'rgb.raw').read_bytes(), b'')
            self.assertEqual(client.cancelled, 1)

    def test_snapshot_callback_failure_cancels_and_censors_unsubmitted_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            source = DummySource(clip(), [], directory, prompt_events=[PromptEvent(1, 0, 'A')])
            client = PromptClient(counts=(5,))
            producer = FrameProducer(source.frames(), 5, pace=None)
            def failed_snapshot():
                raise RuntimeError('snapshot unavailable')
            with self.assertRaisesRegex(RuntimeError, 'snapshot unavailable'):
                run(directory, producer, client, failed_snapshot, source)
            self.assertEqual(summarize_dummy_run(directory)['prompt_events'][0]['status'], 'never_submitted')
            self.assertEqual(client.requests, [])
            self.assertEqual(client.cancelled, 1)
            self.assertFalse(source.thread.is_alive())
            self.assertFalse(producer.thread.is_alive())

    def test_interrupted_http_records_submitted_prompt_and_cancels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = DummySource(clip(), [], directory, prompt_events=[PromptEvent(1, 0, 'A')])
            producer = FrameProducer(source.frames(), 5, pace=None)
            def interrupt(index, metadata):
                raise KeyboardInterrupt('operator cancelled')
            client = PromptClient(counts=(5,), before_chunk=interrupt)
            with self.assertRaises(KeyboardInterrupt):
                run(directory, producer, client, source.prompt_snapshot, source)
            result = summarize_dummy_run(directory)['prompt_events'][0]
            self.assertEqual(result['status'], 'submitted_unreturned')
            self.assertEqual(result['censored_submitted_frames'], 5)
            self.assertEqual(client.cancelled, 1)
            self.assertFalse(source.thread.is_alive())
            self.assertFalse(producer.thread.is_alive())

    def test_failure_censors_prompt_then_fresh_session_starts_with_new_state(self):
        client = PromptClient(counts=(5,))
        for fail in (True, False):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                source = DummySource(clip(), [], directory, prompt_events=[PromptEvent(1, 0, 'new session')])
                producer = FrameProducer(source.frames(), 5, pace=None)
                def failing(index, metadata):
                    raise RuntimeError('chunk failed')
                client.before_chunk = failing if fail else None
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'chunk failed'):
                        run(directory, producer, client, source.prompt_snapshot, source)
                else:
                    run(directory, producer, client, source.prompt_snapshot, source)
                summary = summarize_dummy_run(directory)
                self.assertEqual(summary['prompt_events'][0]['status'], 'submitted_unreturned' if fail else 'returned')
                self.assertEqual(summary['prompt_events'][0]['censored_submitted_frames'], 5 if fail else 0)
                self.assertFalse(source.thread.is_alive())
                self.assertFalse(producer.thread.is_alive())

    def test_cli_checks_prompt_flag_against_artifact_protocol(self):
        # Stop before media loading; flag validation depends only on the health receipt.
        for protocol, prompt_flag in ((FIXED_RECIPE, True), (FIXED_STREAMING_TEXT_RECIPE, False)):
            with self.subTest(protocol=protocol), patch.object(dummy_stream_ref, 'RefClient') as client:
                client.return_value.call.return_value = (Reply(dict(artifacts={'chosen': dict(recipe=dict(protocol=protocol))})), {})
                argv = ['--artifact-id', 'chosen', '--proxy-video', 'proxy.mp4', '--first-frame', 'first.png',
                    '--events', 'events.json', '--output-dir', 'output']
                if prompt_flag:
                    argv.extend(['--prompt-events', 'prompts.json'])
                with self.assertRaises(SystemExit):
                    dummy_stream_ref.main(argv)


if __name__ == '__main__':
    unittest.main()
