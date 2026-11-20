import threading
import time
import unittest

from projects.minimax_h3.inference.serving import SessionError
from projects.minimax_h3.inference.serving_stages import CpuOutput, WorkKey
from projects.minimax_h3.tests.test_serving import chunk
from projects.minimax_h3.tests.test_serving_media import Backend, Harness


class DeferredBackend(Backend):
    deferred_output = True

    def __init__(self):
        super().__init__()
        self.cpu_entered, self.cpu_release, self.cpu_done = (threading.Event() for _ in range(3))
        self.next_gpu_done = threading.Event()
        self.fail_output = self.failed = False
        self.fail_cleanup = False

    def execute(self, op, metadata, pixels, *, defer_output=False):
        if self.failed:
            raise RuntimeError('execution group is unhealthy; restart required')
        pixels, timing = super().execute(op, metadata, pixels)
        if op != 'chunk':
            return pixels, timing
        key = WorkKey(1, metadata['session_id'], op, metadata['chunk_id'])
        if key.chunk_id:
            self.next_gpu_done.set()
        def read():
            if key.chunk_id == 0:
                self.cpu_entered.set()
                if not self.cpu_release.wait(5):
                    raise TimeoutError('test CPU delivery was not released')
                self.cpu_done.set()
                if self.fail_output:
                    self.failed = True
                    raise RuntimeError('injected serialization failure')
            return pixels, dict(output_delivery=dict(status='complete', part_count=int(bool(pixels))))
        output = CpuOutput(key, len(pixels), read)
        return (output, timing) if defer_output else (pixels, timing)

    def close_session(self):
        if self.cpu_entered.is_set():
            assert self.cpu_done.wait(5), 'cleanup did not drain CPU delivery'
        super().close_session()
        if self.fail_cleanup:
            self.fail_cleanup = False
            self.failed = True
            raise RuntimeError('output failed while draining EOS')


class OutputDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness(timeout=5)
        self.backend = DeferredBackend()
        self.h.backend = self.h.worker.backend = self.backend

    def tearDown(self):
        self.backend.cpu_release.set()
        self.h.close()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(predicate())

    def test_slow_zero_output_allows_next_gpu_work_and_keeps_whole_work_reservations(self):
        self.backend.plan = [0, 30, 9]
        self.h.begin()
        session = self.h.media.attach('one')
        first = self.h.submit(chunk())
        self.h.admitted(first)
        self.assertTrue(first.released.wait(2))
        self.assertTrue(self.backend.cpu_entered.wait(2))
        second = self.h.submit(chunk(1))
        self.h.admitted(second)
        self.assertTrue(second.released.wait(2))
        self.assertTrue(self.backend.next_gpu_done.is_set())
        self.assertFalse(self.backend.cpu_done.is_set())
        third = self.h.submit(chunk(2, eos=True))
        self.wait_for(lambda: self.h.box.current is third)
        self.assertEqual(session.reservations, 2)
        self.assertFalse(third.ready.is_set())
        self.assertEqual(self.h.controller.active.next_chunk, 2)
        self.backend.cpu_release.set()
        self.h.admitted(third)
        events = []
        while not events or events[-1].metadata['op'] != 'complete':
            events.append(self.h.media.next_event(session))
            self.assertNotEqual(events[-1].metadata['op'], 'error')
        self.h.media.detach(session, True)
        batches = [event.metadata for event in events if event.metadata['op'] == 'batch']
        self.assertEqual([batch['frame_count'] for batch in batches], [0, 30, 9])
        self.assertEqual([batch['output_id'] for batch in batches], [0, 1, 2])
        self.assertEqual(batches[0]['output_delivery']['part_count'], 0)
        sources = [source for batch in batches for source in batch['source_frames']]
        self.assertEqual([source['source_frame'] for source in sources], list(range(39)))
        self.assertEqual([source['capture_ns'] for source in sources], list(range(39)))
        self.assertEqual([source['chunk_id'] for source in sources], [0]*17 + [1]*17 + [2]*5)
        self.assertEqual(events[-1].metadata['total_frames'], 39)
        self.assertTrue(session.encoder_done.wait(2))
        self.assertTrue(third.released.wait(2))
        self.assertEqual(session.reservations, 0)

    def test_cancel_drains_delivery_before_new_session_and_drops_old_output(self):
        self.h.begin()
        session = self.h.media.attach('one')
        first = self.h.submit(chunk())
        self.h.admitted(first)
        self.assertTrue(first.released.wait(2))
        self.assertTrue(self.backend.cpu_entered.wait(2))
        self.h.media.abort(session, 'test cancellation')
        self.assertFalse(session.worker_done.wait(.05))
        self.assertFalse(session.encoder_done.is_set())
        with self.assertRaisesRegex(SessionError, 'active or draining'):
            self.h.begin('two')
        self.backend.cpu_release.set()
        result = self.h.media.cancel(dict(session_id='one'))
        self.assertEqual(result.metadata['status'], 'cancelled')
        self.assertFalse(session.events)
        self.assertFalse(session.results)
        self.assertEqual(session.reservations, 0)
        self.assertIsNone(self.h.controller.active)
        self.h.begin('two')
        self.assertEqual(self.h.media.active.session_id, 'two')

    def test_serialization_failure_cleans_up_and_rejects_reuse(self):
        self.backend.fail_output = True
        self.h.begin()
        session = self.h.media.attach('one')
        job = self.h.submit(chunk())
        self.h.admitted(job)
        self.assertTrue(job.released.wait(2))
        self.assertTrue(self.backend.cpu_entered.wait(2))
        self.backend.cpu_release.set()
        self.assertTrue(session.encoder_done.wait(2))
        self.assertTrue(session.worker_done.wait(2))
        self.assertEqual(session.reservations, 0)
        self.assertIsNone(self.h.controller.active)
        self.assertIn('injected serialization failure', session.error)
        self.assertEqual(self.h.media.cancel(dict(session_id='one')).metadata['status'], 'cancelled')
        self.assertIsNone(self.h.error)
        result = self.h.begin('two')
        self.assertEqual(result.status, 503)
        self.assertIn('unhealthy', result.metadata['error'])

    def test_eos_cleanup_failure_still_releases_worker_and_reservation(self):
        self.h.begin()
        session = self.h.media.attach('one')
        self.backend.fail_cleanup = True
        job = self.h.submit(chunk(eos=True))
        self.h.admitted(job)
        self.assertTrue(job.released.wait(2))
        self.assertTrue(session.encoder_done.wait(2))
        self.assertTrue(session.worker_done.is_set())
        self.assertIn('output failed while draining EOS', session.error)
        self.assertEqual(session.reservations, 0)
        self.assertIsNone(self.h.controller.active)
        self.assertFalse(self.h.occupier.paused)
        self.assertIsNone(self.h.error)


if __name__ == '__main__':
    unittest.main()
