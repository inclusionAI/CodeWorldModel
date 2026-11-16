import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('latent_windows', ROOT/'inference/latent_windows.py')
windows = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = windows
spec.loader.exec_module(windows)


def recipe(**changes):
    return SimpleNamespace(**dict(dict(sink_latents=2, recent_latents=10,
        bootstrap_latents=9, current_latent_schedule=(3, 2), segment_boundaries=()), **changes))


class PlannerTests(unittest.TestCase):
    def test_readiness_overlap_eos(self):
        planner = windows.LatentPlanner(recipe())
        self.assertEqual(planner.ready(5), [])
        self.assertEqual([(p.current.start, p.current.stop) for p in planner.ready(10)], [(0, 9)])
        plans = planner.ready(15)
        self.assertEqual([(p.current.start, p.current.stop) for p in plans], [(9, 12), (12, 14)])
        self.assertEqual(plans[0].requested_recent, windows.LatentRange(0, 9))
        self.assertEqual(plans[0].dynamic_recent, windows.LatentRange(2, 9))
        self.assertEqual(plans[1].dynamic_recent, windows.LatentRange(2, 12))
        tail = planner.ready(17, eos=True)
        self.assertEqual(tail[-1].current, windows.LatentRange(14, 17))
        self.assertTrue(tail[-1].eos)
        with self.assertRaises(ValueError):
            planner.ready(18)

    def test_raw_units_and_zero_recent(self):
        for sink in (0, 1, 2, 4):
            for recent in (0, 10):
                plans = windows.LatentPlanner(recipe(sink_latents=sink, recent_latents=recent)).ready(42, eos=True)
                for plan in plans:
                    a = plan.current.start
                    self.assertEqual(plan.sink.stop, min(sink, a))
                    self.assertEqual(plan.dynamic_recent.count, a-max(min(sink, a), max(0, a-recent)))
                self.assertEqual(plans[-1].current.stop, 42)

    def test_step300_exact_intervals_and_readiness(self):
        expected = [(0,9,34),(9,12,51),(12,14,51),(14,17,68),(17,19,68),
            (19,22,85),(22,24,85),(24,27,102),(27,29,102),(29,32,119),(32,34,119),
            (34,37,136),(37,39,136),(39,42,153),(42,44,153),(44,45,153),
            (45,47,170),(47,49,170),(49,52,187),(52,54,187),(54,57,204),(57,59,204),
            (59,62,221),(62,64,221),(64,67,238),(67,69,238),(69,70,238),
            (70,72,255),(72,74,255),(74,77,272),(77,79,272),(79,82,289),
            (82,84,289),(84,87,306),(87,89,306),(89,92,323),(92,94,323),(94,97,328)]
        planner = windows.LatentPlanner(recipe(segment_boundaries=(45,70)))
        actual = []
        for i in range(1, 20):
            actual.extend((p.current.start, p.current.stop, i*17) for p in planner.ready(i*5))
        actual.extend((p.current.start, p.current.stop, 328) for p in planner.ready(97, eos=True))
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
