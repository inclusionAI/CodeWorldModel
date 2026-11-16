from dataclasses import replace
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from projects.minimax_h3.inference.raw_latent import RawLatentSession, RawLatentDriver
from projects.minimax_h3.inference.raw_recipe import RAW_PROTOCOLS, RAW_BIDIR, RawLatentRecipe
from projects.minimax_h3.tests.bidir_history_support import actual_model, full_window
from projects.minimax_h3.tests.test_ref_inference import inputs, recipe_config, Grid, Euler
from projects.minimax_h3.modeling.ref_recipe import RefRecipe
from projects.minimax_h3.modeling.inference_optimizations import configure_inference
from projects.minimax_h3.modeling.inference_runtime import StageProfiler


def session(net, protocol=RAW_BIDIR, *, fuse_first_denoise=False, **changes):
    static, picture, chunks = inputs()
    static['prompt_embeds'] = static['prompt_embeds'].bfloat16()
    source = RefRecipe.from_training(recipe_config())
    recipe = RawLatentRecipe(source.conditioning_mode, 37, source.diffusion, source.guidance,
                             protocol=protocol, **changes)
    def text(value):
        return dict(prompt_embeds=torch.full((2,32), len(value)*.01, dtype=torch.bfloat16),
                    text_token_tags=torch.ones(2, dtype=torch.long))
    result = RawLatentSession(net, recipe, static, first_frame=picture,
                              seed=73, condition_seed=94, encode_text=text,
                              fuse_first_denoise=fuse_first_denoise)
    result.timesteps, result.sampler = Grid(), Euler()
    result.timesteps.timesteps = result.timesteps.timesteps.to(result.picture.device)
    return result, chunks


class RawLatentTests(unittest.TestCase):
    @torch.no_grad()
    def test_graph_scope_ends_before_commit_and_on_forward_or_sampler_failure(self):
        with actual_model() as net:
            for failure in (None, 'forward', 'sampler'):
                with self.subTest(failure=failure):
                    stream, chunks = session(net, RAW_PROTOCOLS[0], recent_latents=0,
                                             current_latent_schedule=(5,))
                    forward, sample, stage = stream.model.forward, stream.sampler.step_to, stream._stage
                    seen, eligibility, commits = [], [], []
                    active = None
                    def checked_forward(backbone, inputs, frames, timestep, cache, **kwargs):
                        nonlocal active
                        if kwargs.get('store'):
                            self.assertFalse(hasattr(cache, '_h3_dit_graph_window'))
                            self.assertFalse(getattr(cache, '_h3_dit_graph_eligible', False))
                        else:
                            active = cache
                            self.assertTrue(hasattr(cache, '_h3_dit_graph_window'))
                            eligibility.append(cache._h3_dit_graph_eligible)
                            if not hasattr(cache, '_h3_dit_graphs'):
                                cache._h3_dit_graphs = SimpleNamespace(end_window=Mock(), stats={})
                                seen.append(cache)
                            if failure == 'forward' and cache._h3_dit_graph_eligible:
                                raise RuntimeError('forward failure')
                        return forward(backbone, inputs, frames, timestep, cache, **kwargs)
                    def checked_sample(**kwargs):
                        self.assertTrue(hasattr(active, '_h3_dit_graph_window'))
                        if failure == 'sampler' and active._h3_dit_graph_eligible:
                            raise RuntimeError('sampler failure')
                        return sample(**kwargs)
                    def checked_stage(name):
                        if name in ('latent_commit', 'clean_commit'):
                            commits.append(name)
                            for cache in seen:
                                self.assertFalse(hasattr(cache, '_h3_dit_graph_window'))
                                self.assertFalse(cache._h3_dit_graph_eligible)
                        return stage(name)
                    with patch.object(stream.model, 'forward', side_effect=checked_forward), \
                            patch.object(stream.sampler, 'step_to', side_effect=checked_sample), \
                            patch.object(stream, '_stage', side_effect=checked_stage):
                        stream.push(chunks[0])
                        stream.push(chunks[1])
                        if failure:
                            with self.assertRaisesRegex(RuntimeError, failure + ' failure'):
                                stream.push(chunks[2])
                        else:
                            stream.push(chunks[2])
                            stream.push(chunks[3])
                            self.assertIn('clean_commit', commits)
                            self.assertFalse(eligibility[-1])
                    self.assertIn(True, eligibility)
                    self.assertIn(False, eligibility)
                    for cache in seen:
                        self.assertFalse(hasattr(cache, '_h3_dit_graph_window'))
                        self.assertFalse(cache._h3_dit_graph_eligible)
                        self.assertGreater(cache._h3_dit_graphs.end_window.call_count, 0)

    def test_only_fixed_session_caches_and_forks_support_inference_arena(self):
        with actual_model() as net:
            for protocol in RAW_PROTOCOLS:
                stream, _ = session(net, protocol)
                for sp_size in (1, 4):
                    stream.model._sp_size = lambda: sp_size
                    cache = stream._cache()
                    self.assertEqual(cache.supports_inference_arena, not stream.recipe.bidirectional)
                    self.assertEqual(cache.fork().supports_inference_arena, not stream.recipe.bidirectional)
                    self.assertEqual(cache.batch_size, 1 + (sp_size > 1))

    @torch.no_grad()
    @unittest.skipIf(os.environ.get('H3_BIDIR_TEST_DEVICE', 'cpu') != 'cpu',
                     'FP32 comparison uses CPU SDPA')
    def test_profile_and_dit_options_preserve_fixed_and_bidir_windows(self):
        with actual_model(fp32=True) as net:
            for protocol in RAW_PROTOCOLS[:2]:
                def run(*, optimized=False, profile=False):
                    configure_inference(net, metadata=optimized, kv=optimized, adaln=optimized)
                    stream, chunks = session(net, protocol, current_latent_schedule=(5,))
                    profiler = StageProfiler(enabled=profile)
                    stream.profiler = profiler
                    outputs = [output for chunk in chunks for output in stream.push(chunk)]
                    return outputs, profiler

                baseline, disabled_profiler = run()
                profiled, profiler = run(profile=True)
                optimized, _ = run(optimized=True)
                self.assertEqual(disabled_profiler.records, [])
                self.assertTrue(any(o.plan.dynamic_recent.start > o.plan.sink.stop for o in baseline))
                self.assertEqual(len(baseline), len(profiled))
                self.assertEqual(len(baseline), len(optimized))
                for expected, measured, cached in zip(baseline, profiled, optimized, strict=True):
                    self.assertEqual(expected.plan, measured.plan)
                    self.assertEqual(expected.plan, cached.plan)
                    self.assertEqual(expected.sources, measured.sources)
                    self.assertEqual(expected.sources, cached.sources)
                    torch.testing.assert_close(expected.video, measured.video, rtol=0, atol=0)
                    torch.testing.assert_close(expected.video, cached.video, rtol=0, atol=1e-5)
                report = profiler.report()
                stages = set(report['stages'])
                self.assertEqual(report['stages']['raw_generate']['calls'], len(baseline))
                self.assertTrue({'raw_generate', 'denoise_inputs', 'sampling_setup',
                                 'latent_commit', 'prefix_inputs', 'denoise'} <= stages)
                if protocol == RAW_PROTOCOLS[0]:
                    self.assertTrue({'cache_retain', 'condition_inputs', 'clean_inputs',
                                     'history_inputs'} <= stages)

    @torch.no_grad()
    def test_rejected_source_does_not_change_session_identity(self):
        with actual_model() as net:
            for protocol in (RAW_PROTOCOLS[0], RAW_PROTOCOLS[2]):
                stream, chunks = session(net, protocol)
                bad = dict(chunks[0], prompt='bad', prompt_revision=True)
                with self.assertRaises(ValueError):
                    stream.push(bad)
                self.assertIsNone(stream.signature)
                self.assertIsNone(stream.text)
                self.assertEqual(stream.available, 0)
                good = dict(chunks[0], references=[dict(chunks[0]['references'][0], role='changed')])
                if stream.recipe.streaming_text:
                    good.update(prompt='valid', prompt_revision=1)
                self.assertEqual(stream.push(good), [])
                self.assertEqual(stream.signature[0][0], 'changed')

    def test_finite_driver_rejects_missing_eos(self):
        from projects.minimax_h3.tests.test_ref_inference import CacheProbe
        net = CacheProbe()
        static, picture, chunks = inputs()
        source = RefRecipe.from_training(recipe_config())
        recipe = RawLatentRecipe(source.conditioning_mode, 37, source.diffusion, source.guidance)
        with self.assertRaisesRegex(ValueError, 'without EOS'):
            list(RawLatentDriver(recipe).stream_latents(net, static, chunks[:1], first_frame=picture))

    @torch.no_grad()
    def test_all_four_protocols_zero_multi_window_eos_and_source_prompt_slices(self):
        with actual_model() as net:
            for protocol in RAW_PROTOCOLS:
                stream, chunks = session(net, protocol)
                outputs = []
                for k, item in enumerate(chunks):
                    if stream.recipe.streaming_text:
                        item = dict(item, prompt=f'instruction {k}', prompt_revision=k+1)
                    result = stream.push(item)
                    self.assertEqual(len(result), (0, 1, 2, 1)[k])
                    outputs.extend(result)
                self.assertEqual([o.video.shape[1] for o in outputs], [9,3,2,3])
                self.assertEqual([(s['source_clip'], s['latent_start'], s['latent_stop'])
                    for s in outputs[1].sources], [(1,9,10), (2,10,12)])
                self.assertEqual(sum(o.plan.current.count for o in outputs), 17)
                self.assertTrue(outputs[-1].plan.eos)
                if stream.recipe.streaming_text:
                    self.assertEqual([s['prompt']['revision'] for s in outputs[1].sources], [2,3])
                    self.assertEqual([s['prompt']['text'] for s in outputs[1].sources],
                                     ['instruction 1', 'instruction 2'])
                with self.assertRaisesRegex(ValueError, 'after EOS'):
                    stream.push(chunks[-1])

    @torch.no_grad()
    @unittest.skipIf(os.environ.get('H3_BIDIR_TEST_DEVICE', 'cpu') != 'cpu',
                     'FP32 structural oracle uses CPU SDPA; CUDA flash attention requires BF16')
    def test_isolated_prefix_matches_full_oracle_and_is_zero_timestep(self):
        with actual_model(fp32=True) as net:
            stream, chunks = session(net)
            stream.push(chunks[0])
            bootstrap = stream.push(chunks[1])[0]
            captured = []
            hook = net.register_forward_pre_hook(lambda module, args, kwargs: captured.append(dict(kwargs)),
                                                 with_kwargs=True)
            outputs = stream.push(chunks[2])
            hook.remove()
            prefix = captured[0]
            self.assertTrue(prefix['update_past_key_values'])
            self.assertEqual(prefix['unique_timesteps'].tolist(), [0.])
            torch.testing.assert_close(prefix['x'][0, -2:],
                stream.model._video_rows(bootstrap.video, 1,2).to(prefix['x'].dtype), rtol=0, atol=0)
            cache = stream.prefix['positive']
            frozen = [v.clone() for v in cache.key_cache.values()]
            first_dynamic = captured[1]
            oracle = net(**full_window(prefix, first_dynamic))[0]
            direct = net(**first_dynamic)[0]
            torch.testing.assert_close(oracle, direct, rtol=0, atol=1e-5)
            altered = dict(first_dynamic, x=first_dynamic['x'].clone())
            altered['x'][:, -6:] += 3
            self.assertFalse(torch.equal(net(**altered)[0], direct))
            for value, original in zip(cache.key_cache.values(), frozen):
                torch.testing.assert_close(value, original, rtol=0, atol=0)
            sink = [stream.frames[i] for i in range(2)]
            p = stream._inputs(sink, 'positive', include_static=True, prefix=True)
            other = stream._cache()
            stream.model.forward(net, p, sink, stream.picture.new_tensor(.8), other, store=True)
            for value, original in zip(other.key_cache.values(), frozen):
                torch.testing.assert_close(value, original, rtol=0, atol=0)
            self.assertEqual(outputs[0].metrics['prefix_prefill'], 1)
            self.assertEqual(outputs[1].metrics['prefix_cache_hits'], 1)

    @torch.no_grad()
    def test_fixed_partial_eviction_preserves_exact_old_rows_and_prefix(self):
        with actual_model() as net:
            stream, chunks = session(net, RAW_PROTOCOLS[0], recent_latents=10)
            stream.push(chunks[0])
            stream.push(chunks[1])
            stream.push(chunks[2])
            cache = stream.fixed['positive']
            owners = list(cache.owners)
            previous = [v.clone() for v in cache.key_cache.values()]
            prefix = [v.clone() for v in stream.prefix['positive'].key_cache.values()]
            stream.push(dict(chunks[0]))
            cache = stream.fixed['positive']
            old_rows = {owner:i for i,owner in enumerate(owners) if owner[0] >= 7}
            for layer, value in enumerate(cache.key_cache.values()):
                for owner, old_index in old_rows.items():
                    indices = [i for i,o in enumerate(cache.owners) if o == owner]
                    if indices:
                        old_indices = [i for i,o in enumerate(owners) if o == owner]
                        torch.testing.assert_close(value[indices], previous[layer][old_indices], rtol=0, atol=0)
            for value, original in zip(stream.prefix['positive'].key_cache.values(), prefix):
                torch.testing.assert_close(value, original, rtol=0, atol=0)
            self.assertTrue(all(p < 0 or p >= 7 for p,_ in cache.owners))

    @torch.no_grad()
    def test_long_stream_noise_identity_and_sp_padding(self):
        with actual_model() as net:
            configure_inference(net, metadata=True, kv=True, adaln=True)
            stream, chunks = session(net, sink_latents=4, recent_latents=0)
            stream.model._sp_size = lambda: 4
            stream.model._pad_rows = lambda n: 4-n%4
            captures = []
            hook = net.register_forward_pre_hook(lambda module, args, kwargs: captures.append(dict(kwargs)), with_kwargs=True)
            output = []
            for _ in range(9):
                output.extend(stream.push(chunks[0]))
            output.extend(stream.push(chunks[-1]))
            hook.remove()
            self.assertEqual(output[-1].plan.current.stop, 47)
            self.assertLessEqual(len(stream.frames), 4)
            for call in captures:
                self.assertEqual(sum(call['sample_lens']) % 4, 0)
                self.assertGreater(call['sample_lens'][-1], 0)
                self.assertEqual(call['past_key_values'].kvlens[-1], 0)
            self.assertEqual(sum(o.metrics['prefix_prefill'] for o in output), 1)

    @torch.no_grad()
    def test_cfg_has_independent_prefix_banks(self):
        with actual_model() as net:
            stream, chunks = session(net)
            stream.recipe = replace(stream.recipe, guidance=dict(mode='cfg', drop='proxy', scale=2.))
            stream.model.recipe = stream.recipe
            for item in chunks:
                stream.push(item)
            self.assertEqual(set(stream.prefix), {'positive','negative'})
            self.assertGreater(stream.prefix['positive'].kvlens[0], stream.prefix['negative'].kvlens[0])
            self.assertIsNot(stream.prefix['positive'].key_cache[0], stream.prefix['negative'].key_cache[0])


if __name__ == '__main__':
    unittest.main()
