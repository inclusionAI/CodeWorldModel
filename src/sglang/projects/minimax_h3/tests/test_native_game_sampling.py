import unittest

from tools.serve_web import bootstrap

bootstrap()
from projects.minimax_h3.inference.native_game_sampling import MediaSlots, Publication, SECOND, FPS


def publication(number, stamp):
    return Publication(dict(frame_id=number, published_monotonic_ns=stamp,
                            jpeg_sha256='equal-jpeg'), b'jpeg', stamp)


class MediaSlotTests(unittest.TestCase):
    def trace(self, timestamps):
        slots = MediaSlots()
        samples = []
        now = 0
        publications = iter(enumerate(timestamps, 1))
        upcoming = next(publications, None)
        while upcoming is not None:
            if slots.origin_ns is None or upcoming[1] <= slots.deadline_ns():
                now = upcoming[1]
                slots.offer(publication(*upcoming))
                upcoming = next(publications, None)
            else:
                now = slots.deadline_ns()
            selected = slots.take(now)
            if selected:
                samples.append(selected)
            self.assertIsNone(slots.pending, 'trace offers only at native publication events')
        return slots, samples

    def test_slow_matched_fast_and_jitter_preserve_duration(self):
        traces = [(fps, [round(n * SECOND / fps) for n in range(fps * 4 + 1)]) for fps in (12, 24, 48, 60)]
        traces.append((None, [sum(([33_000_000, 50_000_000] * 49)[:n]) for n in range(98)]))
        for fps, timestamps in traces:
            with self.subTest(interval=timestamps[1]):
                slots, samples = self.trace(timestamps)
                duration = (len(samples) - 1) * SECOND / FPS
                self.assertLessEqual(abs(duration - timestamps[-1]), SECOND / FPS)
                self.assertEqual([m['media_slot'] for _, m in samples], list(range(len(samples))))
                self.assertEqual({m['media_segment'] for _, m in samples}, {0})
                self.assertEqual(slots.cursor, len(timestamps))
                fresh = sum(m['sample_kind'] == 'fresh' for _, m in samples)
                if fps == 12:
                    self.assertLess(fresh, len(samples))
                if fps in (48, 60):
                    self.assertLess(fresh, len(timestamps))

    def test_phase_boundary_jitter_preserves_every_matched_publication(self):
        for direction in (-1, 1):
            timestamps = [0] + [n * SECOND // FPS + direction *
                                (19_000_000 if n % 2 else 22_000_000) for n in range(1, 97)]
            with self.subTest(direction=direction):
                slots, samples = self.trace(timestamps)
                self.assertEqual(len(samples), len(timestamps))
                self.assertTrue(all(metadata['sample_kind'] == 'fresh' for _, metadata in samples))
                self.assertEqual(sum(metadata['media_skipped'] for _, metadata in samples), 0)
                self.assertLessEqual(abs((len(samples) - 1) * SECOND / FPS - timestamps[-1]), SECOND / FPS)

    def test_deadline_tie_prefers_fresh_and_early_exclusion_retains_held_rgb(self):
        slots = MediaSlots()
        held = publication(1, 0)
        held.pixels, held.ready_ns = b'rgb', 10
        slots.offer(held)
        self.assertIs(slots.take(0)[0], held)
        fresh = publication(2, slots.deadline_ns())
        slots.offer(fresh)
        self.assertIs(slots.take(slots.deadline_ns())[0], fresh)
        slots.offer(publication(3, 1))
        self.assertIsNone(slots.pending)
        self.assertEqual(slots.cursor, 3)
        self.assertIsNone(slots.take(slots.deadline_ns() - 1))
        item, metadata = slots.take(slots.deadline_ns())
        self.assertIs(item, fresh)
        self.assertEqual(metadata['sample_kind'], 'repeat')
        self.assertEqual(metadata['media_skipped'], 1)

    def test_long_capacity_gap_starts_one_segment_without_catchup(self):
        slots = MediaSlots()
        slots.offer(publication(1, 0))
        slots.take(0)
        slots.restart('capacity', 10 * SECOND)
        self.assertIsNone(slots.take(10 * SECOND))
        slots.offer(publication(300, 10 * SECOND))
        item, metadata = slots.take(10 * SECOND)
        self.assertEqual(item.evidence['frame_id'], 300)
        self.assertEqual(metadata['media_slot'], 0)
        self.assertEqual(metadata['media_segment'], 1)
        self.assertEqual(metadata['media_reason'], 'capacity')
        self.assertGreater(metadata['media_omitted_ns'], 9 * SECOND)
        self.assertIsNone(slots.take(10 * SECOND))


if __name__ == '__main__':
    unittest.main()
