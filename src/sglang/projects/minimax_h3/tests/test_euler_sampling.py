"""Deterministic Euler contract on the production linear-flow sampling grid."""
import unittest

import torch

from common.diffusion.schedule import BaseSchedule
from common.diffusion.schedule.lerp import LinearInterpolationSchedule
from common.diffusion.timestep.sampling.trailing import TrailingSamplingTimesteps
from projects.minimax_h3.modeling.sampling import EulerSampler


class EulerSamplingTests(unittest.TestCase):
    def test_matches_analytical_euler_including_endpoints_and_ragged_samples(self):
        for total in (1.0, 1000):
            schedule = LinearInterpolationSchedule(T=total, pred_type='x_0')
            sampler = EulerSampler(schedule)
            t = torch.tensor([1.0, 1e-4], dtype=torch.float64) * total
            s = torch.tensor([0.0, 5e-5], dtype=torch.float64) * total
            x = [torch.randn(2, 3, dtype=torch.float64), torch.randn(1, 5, dtype=torch.float64)]
            pred = [torch.randn_like(v, requires_grad=True) for v in x]
            actual = sampler.step_to(pred, x, t, s)
            for i, result in enumerate(actual):
                velocity = (x[i]-pred[i]) / (t[i]/total)
                expected = x[i]+(s[i]-t[i])/total * velocity
                torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
                grad, = torch.autograd.grad(result.sum(), pred[i])
                torch.testing.assert_close(grad, torch.full_like(grad, 1-s[i]/t[i]))

    def test_twenty_step_grid_is_deterministic_and_reaches_zero(self):
        schedule = LinearInterpolationSchedule(T=1.0, pred_type='v_lerp')
        sampler = EulerSampler(schedule)
        grid = TrailingSamplingTimesteps(T=1.0, num_sampling_steps=20, shift=12.0)
        grid.set_timesteps()
        self.assertEqual(len(grid.timesteps), 20)
        self.assertEqual(grid.timesteps[0].item(), 1.0)
        x = torch.randn(1, 3, dtype=torch.float64)
        velocity = torch.randn_like(x, requires_grad=True)
        initial, rng = x.clone(), torch.random.get_rng_state().clone()
        for t in grid.timesteps:
            t = t.reshape(1)
            s = grid.get_next_timesteps(t)
            x = sampler.step_to(velocity, x, t, s)
        self.assertEqual(s.item(), 0.0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        torch.testing.assert_close(x, initial-velocity, rtol=1e-12, atol=1e-12)
        grad, = torch.autograd.grad(x.sum(), velocity)
        torch.testing.assert_close(grad, -torch.ones_like(grad), rtol=1e-12, atol=1e-12)

    def test_rejects_non_linear_flow_schedule(self):
        with self.assertRaisesRegex(ValueError, 'LinearInterpolationSchedule'):
            EulerSampler(BaseSchedule(T=1.0, pred_type='x_0'))


if __name__ == '__main__':
    unittest.main()
