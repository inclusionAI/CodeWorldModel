import threading
import unittest
import weakref

from tests.test_serving_media import Harness, chunk
from projects.minimax_h3.inference.serving import Reply


class RGB(bytearray):
    pass


class MediaReleaseTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness(timeout=5)
        self.gate = threading.Event()
        self.addCleanup(self.h.close)
        self.addCleanup(self.gate.set)

    def wait_for(self, predicate):
        with self.h.box.condition:
            self.assertTrue(self.h.box.condition.wait_for(predicate, 2))

    def hold_last_encode(self, *, fail=False):
        refs = []
        execute = self.h.backend.execute

        def tracked(op, metadata, pixels):
            result, timings = execute(op, metadata, pixels)
            if op == 'chunk':
                result = RGB(result)
                for index in range(len(result) // (16*16*3)):
                    shade = 32 + 4*(metadata['chunk_id']*17 + index)
                    result[index*16*16*3:(index+1)*16*16*3] = bytes([shade])*(16*16*3)
                refs.append(weakref.ref(result))
            return result, timings

        self.h.backend.execute = tracked
        self.h.begin()
        session = self.h.media.attach('one')
        entered = threading.Event()
        encode = session.encoder.encode

        def held(rgb, ordinal):
            result = encode(rgb, ordinal)
            if ordinal == 16:
                entered.set()
                assert self.gate.wait(3), 'last encode gate timed out'
                if fail:
                    raise RuntimeError('final encode failed')
            return result

        session.encoder.encode = held
        first = self.h.submit(chunk(0))
        self.h.admitted(first)
        second = self.h.submit(chunk(1))
        self.h.admitted(second)
        self.assertTrue(second.released.wait(2))
        collected = [self.h.media.next_event(session) for _ in range(13)]
        self.assertTrue(entered.wait(2))
        third = self.h.submit(chunk(2, eos=True))
        self.wait_for(lambda: self.h.box.current is third)
        self.assertFalse(third.ready.wait(.05), 'RGB was released while encode still held its input')
        self.assertEqual(session.reservations, 2)
        self.assertIsNotNone(refs[0]())
        self.assertEqual([event.metadata['output_frame'] for event in session.events], [12, 13, 14, 15])
        return session, third, collected, refs

    def drain(self, session, collected):
        while True:
            event = self.h.media.next_event(session)
            collected.append(event)
            if event.metadata['op'] in ('complete', 'error'):
                self.h.media.detach(session, event.metadata['op'] == 'complete')
                return

    def assert_clean(self, session):
        self.assertTrue(session.encoder_done.wait(2))
        self.assertTrue(session.worker_done.wait(2))
        self.assertEqual(session.reservations, 0)
        self.assertFalse(self.h.occupier.paused)

    def test_last_encode_return_admits_third_chunk_before_final_publication(self):
        import av

        session, third, collected, refs = self.hold_last_encode()
        self.gate.set()
        self.assertEqual(self.h.admitted(third).metadata['status'], 'admitted')
        self.assertTrue(third.released.wait(2))
        self.assertIsNone(refs[0](), 'encoder retained RGB after its final encode returned')
        self.assertEqual([event.metadata['output_frame'] for event in session.events], [12, 13, 14, 15])
        self.assertEqual(session.reservations, 2)
        self.drain(session, collected)
        frames = [row for row in collected if row.metadata['op'] == 'video_frame']
        for key in ('source_frame', 'capture_ns', 'output_frame'):
            self.assertEqual([row.metadata[key] for row in frames], list(range(39)))
        self.assertEqual([row.metadata['source']['chunk_id'] for row in frames], [0]*17 + [1]*17 + [2]*5)
        self.assertEqual([row.metadata['op'] for row in collected],
            ['batch'] + ['video_frame']*17 + ['batch'] + ['video_frame']*17
            + ['batch'] + ['video_frame']*5 + ['complete'])
        decoder = av.CodecContext.create('h264', 'r')
        decoded = [frame for row in frames for frame in decoder.decode(av.Packet(row.pixels))]
        decoded.extend(decoder.decode(None))
        self.assertEqual([(frame.width, frame.height) for frame in decoded], [(16, 16)]*39)
        for ordinal, frame in enumerate(decoded):
            self.assertAlmostEqual(float(frame.to_ndarray(format='rgb24').mean()),
                32 + 4*ordinal, delta=3)
        self.assert_clean(session)
        self.assertTrue(all(ref() is None for ref in refs))

    def test_cancel_before_release(self):
        self.check_failure('cancel_before')

    def test_cancel_after_release(self):
        self.check_failure('cancel_after')

    def test_final_encode_exception(self):
        self.check_failure('encode')

    def test_final_publication_timeout(self):
        self.check_failure('publication_timeout')

    def check_failure(self, failure):
        session, third, _, refs = self.hold_last_encode(fail=failure == 'encode')
        if failure == 'cancel_before':
            replies = []
            cancel = threading.Thread(target=lambda: replies.append(self.h.media.cancel(dict(session_id='one'))))
            cancel.start()
            self.wait_for(lambda: session.cancelled)
            self.assertFalse(session.encoder_done.is_set())
            self.assertEqual(session.reservations, 1)
            self.assertIsNotNone(refs[0]())
            self.gate.set()
            cancel.join(2)
            self.assertFalse(cancel.is_alive())
            self.assertEqual(replies[0].metadata['status'], 'cancelled')
        else:
            if failure == 'publication_timeout':
                self.h.media.timeout = .1
            self.gate.set()
            if failure == 'cancel_after':
                self.h.admitted(third)
                self.assertTrue(third.released.wait(2))
                self.assertIsNone(refs[0]())
                self.h.media.cancel(dict(session_id='one'))
            elif failure == 'publication_timeout':
                self.h.admitted(third)
                self.assertTrue(third.released.wait(2))
        self.assert_clean(session)
        self.assertTrue(all(ref() is None for ref in refs))
        terminal = self.h.media.next_event(session)
        self.assertEqual(terminal.metadata['op'], 'error')
        if failure == 'encode':
            self.assertIn('final encode failed', terminal.metadata['error'])
        elif failure == 'publication_timeout':
            self.assertIn('encoded output queue timeout', terminal.metadata['error'])
        self.assertEqual(self.h.media.cancel(dict(session_id='one')).metadata['status'], 'cancelled')
        self.assertEqual(session.reservations, 0)

    def test_flush_failure_never_publishes_complete(self):
        self.h.begin()
        session = self.h.media.attach('one')

        def fail_flush():
            raise RuntimeError('flush failed')

        session.encoder.flush = fail_flush
        self.h.admitted(self.h.submit(chunk(0, eos=True)))
        collected = []
        self.drain(session, collected)
        self.assertEqual(collected[-1].metadata['op'], 'error')
        self.assertIn('flush failed', collected[-1].metadata['error'])
        self.assertNotIn('complete', [row.metadata['op'] for row in collected])
        self.assert_clean(session)

    def test_empty_batch_releases_before_blocked_publication_including_eos(self):
        for status in ('accepted', 'complete'):
            with self.subTest(status=status):
                self.h.begin(status)
                session = self.h.media.attach(status)
                entered = threading.Event()
                publish = self.h.media._publish

                def publishing(current, reply):
                    if reply.metadata['op'] == 'batch':
                        entered.set()
                    return publish(current, reply)

                self.h.media._publish = publishing
                rgb = RGB()
                ref = weakref.ref(rgb)
                with self.h.box.condition:
                    session.events.extend(Reply(dict(op='occupied')) for _ in range(4))
                    session.results.append(Reply(dict(status=status, chunk_id=0, source_frames=[],
                        output_frame_start=0, total_frames=0), rgb))
                    session.reservations = 1
                    rgb = None
                    self.h.box.condition.notify_all()
                self.assertTrue(entered.wait(2))
                self.assertEqual(session.reservations, 0)
                self.assertIsNone(ref())
                self.assertEqual(len(session.events), 4)
                for _ in range(4):
                    self.assertEqual(self.h.media.next_event(session).metadata['op'], 'occupied')
                self.assertEqual(self.h.media.next_event(session).metadata['op'], 'batch')
                if status == 'complete':
                    self.assertEqual(self.h.media.next_event(session).metadata['op'], 'complete')
                self.h.media.cancel(dict(session_id=status))
                self.assert_clean(session)
                self.h.media._publish = publish


if __name__ == '__main__':
    unittest.main()
