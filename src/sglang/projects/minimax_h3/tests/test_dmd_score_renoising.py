"""CPU regressions for DMD scoring noise independent of the rollout solver."""
from __future__ import annotations

import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from common.diffusion.sampler.consistency import ConsistencySampler
from common.diffusion.schedule.lerp import LinearInterpolationSchedule
from projects.minimax_h3.meta_models.causal_minimax_h3_dmd import CausalMiniMaxH3DMD
from projects.minimax_h3.modeling.sampling import EulerSampler


def map_tensors(value, fn):
    if isinstance(value, (list, tuple)):
        return type(value)(map_tensors(item, fn) for item in value)
    return fn(value)


def tensors(value):
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in tensors(item)]
    return [value]


def rngs(per_sample=False):
    # ConsistencySampler historically reads torch_cuda_generator on every
    # device. A CPU generator under that name makes the exact old path testable.
    values = [SimpleNamespace(torch_cuda_generator=torch.Generator().manual_seed(seed))
              for seed in (713, 191)]
    return values if per_sample else values[0]


def rng_states(rng):
    values = rng if isinstance(rng, (list, tuple)) else [rng]
    return [state.torch_cuda_generator.get_state() for state in values]


def subject():
    value = CausalMiniMaxH3DMD.__new__(CausalMiniMaxH3DMD)
    value.schedule = LinearInterpolationSchedule(T=1.0, pred_type='x_0')
    value.sampler = EulerSampler(value.schedule)
    return value


def values(shape, dtype):
    return torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape).to(dtype) / 37


def ragged_pair(dtype):
    return ([values((2, 3, 2, 2), dtype), values((2, 5, 2, 2), dtype)],
            [values((2, 32, 0), dtype), values((2, 32, 3), dtype)])


class DMDScoreRenoisingTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized(), 'These tests must stay on CPU')

    def assert_tensors_equal(self, actual, expected):
        actual, expected = tensors(actual), tensors(expected)
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def compare_with_legacy(self, prediction, xts, times, per_sample):
        meta = subject()
        legacy = ConsistencySampler(meta.schedule)
        new_rng, old_rng = rngs(per_sample), rngs(per_sample)
        new_pred = map_tensors(prediction, lambda tensor: tensor.detach().clone().requires_grad_())
        old_pred = map_tensors(prediction, lambda tensor: tensor.detach().clone().requires_grad_())
        gen_t = torch.tensor([.27, .81], dtype=torch.float64)
        old = tuple(legacy.step_to(pred=p, x_t=x, t=gen_t, s=t, rng=old_rng)
                    for p, x, t in zip(old_pred, xts, times, strict=True))
        with patch.object(meta.sampler, 'step_to', side_effect=AssertionError('score used Euler')):
            new = tuple(meta._renoise_for_score(p, x, t, new_rng)
                        for p, x, t in zip(new_pred, xts, times, strict=True))
        self.assert_tensors_equal(new, old)
        self.assert_tensors_equal(rng_states(new_rng), rng_states(old_rng))
        # Nonlinear output weights exercise the full attached noise graph, not
        # merely a manually supplied derivative of schedule.A.
        for output in (old, new):
            sum(tensor.float().square().sum() for tensor in tensors(output)).backward()
        self.assert_tensors_equal([tensor.grad for tensor in tensors(new_pred)],
                                  [tensor.grad for tensor in tensors(old_pred)])

    def test_ragged_video_then_audio_matches_legacy_rng_dtype_and_gradients(self):
        for per_sample, pred_dtype, xt_dtype, time_dtype in itertools.product(
            (False, True), (torch.float32, torch.bfloat16),
            (torch.float32, torch.bfloat16), (torch.float32, torch.float64),
        ):
            with self.subTest(per_sample=per_sample, pred=pred_dtype, xt=xt_dtype, time=time_dtype):
                prediction = ragged_pair(pred_dtype)
                xts = map_tensors(prediction, lambda tensor: torch.full_like(tensor, .7, dtype=xt_dtype))
                times = (torch.tensor([0, .732617], dtype=time_dtype),
                         torch.tensor([1, .271839], dtype=time_dtype))
                self.compare_with_legacy(prediction, xts, times, per_sample)

    def test_batched_tensor_with_individual_generators_matches_legacy(self):
        for audio_frames in (0, 3):
            with self.subTest(audio_frames=audio_frames):
                prediction = (values((2, 3, 4), torch.bfloat16),
                              values((2, 2, audio_frames), torch.float32))
                xts = map_tensors(prediction, lambda tensor: torch.zeros_like(tensor, dtype=torch.bfloat16))
                times = (torch.tensor([.16, .94], dtype=torch.float64),
                         torch.tensor([.31, .79], dtype=torch.float64))
                self.compare_with_legacy(prediction, xts, times, True)

    def test_tuple_pack_and_tuple_generators_extend_list_behavior(self):
        meta = subject()
        prediction = tuple(ragged_pair(torch.float32)[0])
        xts = tuple(torch.zeros_like(tensor, dtype=torch.bfloat16) for tensor in prediction)
        times = torch.tensor([.2, .8], dtype=torch.float64)
        new_rng, old_rng = tuple(rngs(True)), rngs(True)
        actual = meta._renoise_for_score(prediction, xts, times, new_rng)
        expected = ConsistencySampler(meta.schedule).step_to(
            pred=list(prediction), x_t=list(xts), t=torch.tensor([.5, .5]), s=times, rng=old_rng,
        )
        self.assertIsInstance(actual, tuple)
        self.assert_tensors_equal(actual, expected)
        self.assert_tensors_equal(rng_states(new_rng), rng_states(old_rng))
        self.assertEqual(meta._renoise_for_score((), (), times[:0], new_rng), ())

    def test_noise_uses_current_default_dtype_before_output_cast(self):
        # The legacy sampler omits randn(dtype=...), rather than forcing FP32.
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            prediction = ragged_pair(torch.bfloat16)
            xts = map_tensors(prediction, lambda tensor: torch.zeros_like(tensor, dtype=torch.float32))
            times = (torch.tensor([.25, .65]), torch.tensor([.45, .85]))
            self.compare_with_legacy(prediction, xts, times, False)
        finally:
            torch.set_default_dtype(previous)

    def test_score_keeps_masked_audio_draws_and_fake_real_gradient_flags(self):
        for fake_enabled, real_enabled in itertools.product((False, True), repeat=2):
            with self.subTest(fake=fake_enabled, real=real_enabled):
                meta = subject()
                meta.fake_grad_enabled, meta.real_grad_enabled = fake_enabled, real_enabled
                prediction = map_tensors(ragged_pair(torch.float32),
                                         lambda tensor: tensor.detach().clone().requires_grad_())
                # Only xt dtype belongs to the scoring contract. NaN trajectory
                # values and absent generation timesteps must not affect scores.
                xts = map_tensors(prediction, lambda tensor: torch.full_like(tensor, float('nan')))
                times = (torch.tensor([.25, .75]), torch.tensor([.15, .85]))
                inputs = SimpleNamespace(seqlens=torch.tensor([3, 5]), audio_masks=[False, False])
                models = {name: torch.nn.Linear(1, 1, bias=False)
                          for name in ('fake_model', 'tea_model')}
                for model, weight in zip(models.values(), (2.0, 3.0), strict=True):
                    torch.nn.init.constant_(model.weight, weight)
                calls = []

                def score_forward(model, score_inputs, **kwargs):
                    calls.append((model, score_inputs, torch.is_grad_enabled(), kwargs))
                    return map_tensors((kwargs['video_xts'], kwargs['audio_xts']),
                                       lambda tensor: tensor * model.weight.flatten()[0])

                meta._bidirectional_forward = score_forward
                ctx = dict(models=models, rng=rngs(), gen_inputs=inputs, gen_pred=prediction,
                           gen_xts=xts, score_timesteps=times[0], audio_score_timesteps=times[1])
                old_rng = rngs()
                oracle = ConsistencySampler(meta.schedule)
                expected = tuple(oracle.step_to(pred=p, x_t=x, t=torch.tensor([.4, .6]), s=t, rng=old_rng)
                                 for p, x, t in zip(prediction, xts, times, strict=True))
                with meta.gen_model_context(models), \
                     patch.object(meta.sampler, 'step_to', side_effect=AssertionError('score used Euler')), \
                     patch.object(meta.schedule, 'convert_from_pred', side_effect=AssertionError('score used trajectory')):
                    self.assertIs(meta.score(ctx), ctx)
                    self.assertEqual([call[2] for call in calls], [fake_enabled, real_enabled])
                    self.assert_tensors_equal((calls[0][3]['video_xts'], calls[0][3]['audio_xts']), expected)
                    self.assert_tensors_equal(rng_states(ctx['rng']), rng_states(old_rng))
                    for key in ('video_xts', 'audio_xts', 'video_timesteps', 'audio_timesteps'):
                        self.assertIs(calls[0][3][key], calls[1][3][key])
                    for index in (0, 1):
                        self.assertIs(ctx['gen_x0s'][index], prediction[index])
                    for name, enabled in (('fake_score_x0s', fake_enabled), ('real_score_x0s', real_enabled)):
                        self.assertTrue(all(tensor.requires_grad == enabled for tensor in tensors(ctx[name])))
                    active = [ctx[name] for name, enabled in
                              (('fake_score_x0s', fake_enabled), ('real_score_x0s', real_enabled)) if enabled]
                    if active:
                        sum(tensor.sum() for tensor in tensors(active)).backward()
                        coefficient = 2 * fake_enabled + 3 * real_enabled
                        for modality, t in zip(prediction, times, strict=True):
                            for index, tensor in enumerate(modality):
                                torch.testing.assert_close(tensor.grad, torch.full_like(tensor, coefficient * (1-t[index])),
                                                           rtol=2e-7, atol=1e-7)
                self.assertTrue(all(parameter.grad is None for model in models.values() for parameter in model.parameters()))
                self.assertTrue(all(parameter.requires_grad for model in models.values() for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
