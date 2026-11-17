"""CPU checks for normalized continuous TF velocity loss and legacy SFT/CFG."""
from __future__ import annotations

from contextlib import ExitStack
import math
import unittest
from unittest.mock import patch

import torch

from common.config import CfgNode
from common.diffusion.schedule.lerp import LinearInterpolationSchedule
from projects.minimax_h3.meta_models.causal_minimax_h3_tf import CausalMiniMaxH3TF
from projects.minimax_h3.meta_models.minimax_h3_ref2va import MiniMaxH3RefTF
from projects.minimax_h3.modeling.training_timesteps import RefSFTTrainingTimesteps
from projects.minimax_h3.tests.test_ref_guidance import context, meta


def original_grid(shift):
    base = torch.linspace(1.0, 0.0, 1001, dtype=torch.float32)[:-1]
    sigmas = shift * base / (1 + (shift - 1) * base)
    values = torch.exp(-2 * ((sigmas * 1000 - 500) / 1000) ** 2)
    shifted = values - values.min()
    return sigmas, shifted * (1000 / shifted.sum()), 1000 / shifted.sum()


def reference_weight(sigma):
    # Independent float64 evaluation of the requested Gaussian, normalized by
    # the historical FP32 grid. It intentionally does not call the new helper.
    scale = original_grid(12.0)[2].double()
    return (torch.exp(-2 * (sigma.double() - .5).square()) - math.exp(-.5)) * scale


def frame_factors(layout, sigma):
    return torch.cat([(weight / t.double().square()).expand(c.video_stop-c.video_start)
        for weight,t,c in zip(reference_weight(sigma),sigma,layout.chunks,strict=True)])


class RefTrainingLossTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module in ('modeling.ref_condition','meta_models.causal_minimax_h3_base',
                       'meta_models.minimax_h3_ref2va'):
            self.stack.enter_context(patch('projects.minimax_h3.'+module+'.get_device',
                                          return_value=torch.device('cpu')))
        self.stack.enter_context(patch('utils.flex_attn.FLEX_FLASH_ATTN_AVAILABLE',True))
        self.stack.enter_context(patch('projects.minimax_h3.meta_models.minimax_h3_ref2va.all_reduce_sum'))
        self.stack.enter_context(patch('projects.minimax_h3.meta_models.minimax_h3_ref2va.get_world_size',return_value=1))

    def test_default_shift_and_explicit_legacy_sft_grid_are_exact(self):
        self.assertEqual(RefSFTTrainingTimesteps().shift,12.0)
        for shift in (2.22,12.0):
            with self.subTest(shift=shift):
                sampler = RefSFTTrainingTimesteps(shift=shift)
                sigmas,weights,_ = original_grid(shift)
                torch.testing.assert_close(sampler.sigmas,sigmas,rtol=0,atol=0)
                torch.testing.assert_close(sampler.weights,weights,rtol=0,atol=0)
                torch.testing.assert_close(sampler.loss_weight(sigmas),weights,rtol=0,atol=0)
                torch.testing.assert_close(sampler.continuous_loss_weight(sigmas),weights,
                                           rtol=3e-5,atol=1e-6)
                self.assertEqual(weights[0].item(),0)
                self.assertAlmostEqual(weights.mean().item(),1.0,places=6)

    def test_continuous_weights_are_unquantized_and_stable_at_endpoints(self):
        sampler = RefSFTTrainingTimesteps()
        sigma = torch.tensor([0,1e-8,1e-4,.137271,.5,.9999,1],dtype=torch.float32)
        actual = sampler.continuous_loss_weight(sigma)
        torch.testing.assert_close(actual.double(),reference_weight(sigma),rtol=2e-6,atol=1e-12)
        self.assertEqual(actual.dtype,torch.float32)
        self.assertGreater(actual[1].item(),0)
        self.assertEqual(actual[-1].item(),0)
        self.assertNotEqual(actual[3].item(),sampler.loss_weight(sigma[3]).item())

    def test_default_tf_uses_each_chunk_sigma_and_element_count(self):
        subject = meta('tf','plain')
        subject.config.meta_model = {}  # Exercise the new default objective.
        ctx = context(subject)
        sigmas = [torch.tensor([1e-4,.12735,1.0]),torch.tensor([.61,.9999,.3])]
        ctx['train_timesteps'] = (sigmas,sigmas)
        predictions = [torch.full_like(v,.5+i,requires_grad=True)
                       for i,v in enumerate(ctx['clean_latents'][0])]
        ctx['pred'] = (predictions,ctx['clean_latents'][1])
        subject.compute_loss(ctx)
        count = sum(p.numel() for p in predictions)
        expected = torch.zeros((),dtype=torch.float64)
        for pred,target,sigma,layout in zip(predictions,ctx['clean_latents'][0],sigmas,
                                            ctx['inputs'].layouts,strict=True):
            factor = frame_factors(layout,sigma)[None,:,None,None]
            expected += ((pred.detach().double()-target.double()).square()*factor).sum()/count
        torch.testing.assert_close(ctx['loss'],expected,rtol=2e-6,atol=1e-7)
        ctx['loss'].backward()
        for pred,target,sigma,layout in zip(predictions,ctx['clean_latents'][0],sigmas,
                                            ctx['inputs'].layouts,strict=True):
            expected_grad = (2*(pred.detach().double()-target.double())
                             *frame_factors(layout,sigma)[None,:,None,None]/count)
            torch.testing.assert_close(pred.grad.double(),expected_grad,rtol=2e-6,atol=1e-7)
        self.assertEqual(predictions[0].grad[:,10:].count_nonzero().item(),0)

    def test_gft_tf_mixed_drop_preserves_gaussian_gradient_and_beta(self):
        subject = meta('tf',dropout=.1,beta=.5)
        subject.config.meta_model.objective = 'gaussian_velocity'
        ctx = context(subject)
        sigmas = [torch.tensor([.02,.43,1.0]),torch.tensor([.71,.97,.15])]
        ctx['train_timesteps'] = (sigmas,sigmas)
        with patch('projects.minimax_h3.modeling.ref_guidance.random.Random') as rng:
            rng.return_value.random.side_effect = [0,1]
            subject.forward(ctx)
        auxiliary,primary = [c['result'] for c in ctx['models']['backbone'].calls]
        self.assertFalse(auxiliary.requires_grad)
        subject.compute_loss(ctx)
        output = torch.cat([subject._video_rows(v,0,v.shape[1]) for v in ctx['pred'][0]])
        targets = torch.cat([subject._video_rows(v,0,v.shape[1]) for v in ctx['clean_latents'][0]])
        factor = torch.cat([frame_factors(l,t).repeat_interleave(l.latent_shape[2]*l.latent_shape[3]//4)
                            for l,t in zip(ctx['inputs'].layouts,sigmas,strict=True)])[:,None]
        residual = output.detach().double()-targets.double()
        torch.testing.assert_close(ctx['loss'],(residual.square()*factor).mean(),rtol=2e-6,atol=1e-7)
        expected_grad = 2*residual*factor/residual.numel()
        first_rows = ctx['inputs'].samples[0]['video'].numel()//96
        expected_grad[first_rows:] *= .5
        ctx['loss'].backward()
        torch.testing.assert_close(primary.grad.double(),expected_grad,rtol=2e-6,atol=1e-7)

    def test_explicit_x0_preserves_legacy_loss_without_timesteps(self):
        subject = meta('tf','plain')
        ctx = context(subject)
        subject.forward(ctx)
        del ctx['train_timesteps']
        legacy = dict(ctx)
        CausalMiniMaxH3TF.compute_loss(subject,legacy)
        subject.compute_loss(ctx)
        torch.testing.assert_close(ctx['loss'],legacy['loss'],rtol=0,atol=0)

    def test_optional_resampling_defaults_to_twenty_steps_and_keeps_one_step_override(self):
        def base_init(subject,config):
            subject.config = config
            subject.schedule = LinearInterpolationSchedule(T=1.0,pred_type='x_0')
        for recipe,steps in ((None,None),(dict(warmup_steps=0),20),(dict(warmup_steps=0,steps=1),1)):
            config = CfgNode(dict(meta_model=dict(resampling=recipe),engine=dict(ga_steps=4),
                data=dict(args=dict(stage='tf',window_size=6,rope_reference_latents=37)),validation={}))
            with self.subTest(steps=steps),patch.object(CausalMiniMaxH3TF,'__init__',base_init):
                subject = MiniMaxH3RefTF(config)
            if steps is None:
                self.assertIsNone(subject.resampling)
                continue
            self.assertEqual(subject.resampling['steps'],steps)
            ctx = context(subject,count=1)
            with patch.object(subject,'_num_layers',return_value=1):
                with patch.object(subject,'_ref_forward',wraps=subject._ref_forward) as forward:
                    history = subject._resample_history(ctx['models']['backbone'],ctx['inputs'],
                        ctx['clean_latents'][0],ctx['context_eps'][0],ctx['rng'])
            noise_calls = sum(call.kwargs['selection'][0]=='noise' for call in forward.call_args_list)
            self.assertEqual(noise_calls,steps*(len(ctx['inputs'].layouts[0].chunks)-1))
            self.assertTrue(torch.isfinite(history[0]).all())

    def test_small_sigmas_use_fp32_and_unrepresentable_loss_fails(self):
        for dtype,minimum in ((torch.float16,1e-4),(torch.float32,1e-8)):
            subject = meta('tf','plain')
            subject.config.meta_model.objective = 'gaussian_velocity'
            ctx = context(subject,count=1)
            sigma = torch.tensor([minimum,.5,1.0],dtype=dtype)
            ctx['train_timesteps'] = ([sigma],[sigma])
            pred = torch.full_like(ctx['clean_latents'][0][0],.5,dtype=dtype,requires_grad=True)
            ctx['pred'] = ([pred],ctx['clean_latents'][1])
            subject.compute_loss(ctx)
            self.assertTrue(torch.isfinite(ctx['loss']))
            ctx['loss'].backward()
            self.assertTrue(torch.isfinite(pred.grad).all())
        ctx['train_timesteps'] = ([torch.tensor([1e-40,.5,1.0])],[sigma])
        with self.assertRaisesRegex(FloatingPointError,'not finite in FP32'):
            subject.compute_loss(ctx)

    def test_invalid_sigma_or_chunk_count_is_rejected_without_clamping(self):
        subject = meta('tf','plain')
        subject.config.meta_model.objective = 'gaussian_velocity'
        ctx = context(subject,count=1)
        ctx['pred'] = (ctx['noisy_latents'][0],ctx['clean_latents'][1])
        for value in (0,-.1,1.001,float('nan'),float('inf')):
            with self.subTest(value=value):
                ctx['train_timesteps'] = ([torch.tensor([value,.5,.8])],[])
                with self.assertRaisesRegex(ValueError,'finite sigma'):
                    subject.compute_loss(ctx)
        ctx['train_timesteps'] = ([torch.tensor([.5,.8])],[])
        with self.assertRaisesRegex(ValueError,'one sigma per RGB chunk'):
            subject.compute_loss(ctx)


if __name__ == '__main__':
    unittest.main()
