"""Bidirectional history contracts through actual H3 blocks with CPU SDPA."""
import copy
from dataclasses import replace
import unittest

import torch

from projects.minimax_h3.inference.model import RefModel
from projects.minimax_h3.inference.recipes.ref_bidir_history import RefBidirHistory
from projects.minimax_h3.inference.recipes.ref_fixed_history import RefFixedHistory
from projects.minimax_h3.modeling.inference_optimizations import configure_inference
from projects.minimax_h3.modeling.ref_condition import GroupedKVCache
from projects.minimax_h3.modeling.ref_recipe import BIDIR_RECIPE, RefRecipe
from projects.minimax_h3.tests.bidir_history_support import actual_model, full_window
from projects.minimax_h3.tests.test_ref_inference import (CacheProbe, Euler, Grid,
    inputs as stream_inputs, recipe_config, run_stream)
from projects.minimax_h3.tools.ref2va_acceptance import synthetic


def adapter():
    driver = RefModel()
    driver.recipe = replace(RefRecipe.from_training(recipe_config()), protocol=BIDIR_RECIPE)
    return driver


def streaming_driver(mode='plain', *, window=1, drop='proxy'):
    recipe = replace(RefRecipe.from_training(recipe_config('tf' if mode != 'plain' else 'dmd', mode)),
                     protocol=BIDIR_RECIPE, window_size=window, rope_reference_latents=max(37, 5*(window+1)))
    recipe = replace(recipe, guidance=dict(recipe.guidance, drop=drop))
    driver = RefBidirHistory(recipe)
    driver.sampling_timesteps, driver.sampler = Grid(), Euler()
    return driver


def observe_windows(driver):
    records = []
    forward = driver._ref_forward
    def observe(model, inputs, **kwargs):
        selection = kwargs['selection']
        if selection[0] == 'window':
            start = inputs.layouts[0].chunks[selection[1]].video_start
            records.append(dict(start=inputs.samples[0]['metadata']['stream_chunk_start'],
                count=selection[1], history=kwargs['video_context'][0][:, :start].clone(),
                eps=kwargs['video_eps'][0][:, :start].clone(), current=kwargs['video_xts'][0][:, start:].clone(),
                references=[r['video'].clone() for r in inputs.samples[0]['references']],
                reference_eps=[v.clone() for v in inputs.reference_eps[0]], cache=kwargs['cache'],
                seqlens=inputs.seqlens.clone()))
        return forward(model, inputs, **kwargs)
    driver._ref_forward = observe
    return records


class BidirStreamTests(unittest.TestCase):
    def test_current_only_sampler_raw_history_noise_positions_and_bounded_window(self):
        for window in (0, 1, 6):
            driver = streaming_driver(window=window)
            records = observe_windows(driver)
            sampler_calls = []
            def step(**kwargs):
                self.assertEqual(kwargs['t'].dtype, torch.float64)
                self.assertEqual(kwargs['s'].dtype, torch.float64)
                self.assertEqual(kwargs['x_t'][0].shape, kwargs['pred'][0].shape)
                self.assertIn(kwargs['x_t'][0].shape[1], (5, 2))
                sampler_calls.append(kwargs['seqlens'].clone())
                return Euler().step_to(**kwargs)
            driver.sampler.step_to = step
            values, calls = run_stream(driver)
            self.assertEqual([v.shape[1] for v in values], [5, 5, 5, 2])
            self.assertEqual([call['store'] for call in calls], [True]+[False]*12)
            self.assertEqual(len(records), 12)
            for chunk in range(4):
                n = min(chunk, window)
                for record in records[3*chunk:3*chunk+3]:
                    self.assertEqual((record['start'], record['count']), (chunk-n, n))
                    expected = torch.cat(values[chunk-n:chunk], 1) if n else values[chunk][:, :0]
                    torch.testing.assert_close(record['history'], expected, rtol=0, atol=0)
                    torch.testing.assert_close(record['eps'], records[3*chunk]['eps'], rtol=0, atol=0)
                self.assertEqual(sampler_calls[3*chunk].item(), 5+2*values[chunk].shape[1]*2)
            for call in calls:
                self.assertEqual(call['groups'], [[(-1, 5)]])
                torch.testing.assert_close(call['kv'], calls[0]['kv'], rtol=0, atol=0)
                self.assertEqual(call['unique_timesteps'].dtype, torch.float32)
            if window:
                for chunk in range(1, 4):
                    previous, current = records[(chunk-1)*3], records[chunk*3]
                    torch.testing.assert_close(previous['reference_eps'][1][:, -5:],
                        current['reference_eps'][1][:, -current['current'].shape[1]-5:-current['current'].shape[1]],
                        rtol=0, atol=0)
            fixed = RefFixedHistory(replace(driver.recipe, protocol=RefFixedHistory.PROTOCOL))
            fixed.sampling_timesteps, fixed.sampler = Grid(), Euler()
            _, old = run_stream(fixed)
            for chunk in range(4):
                positions = []
                for prior in range(max(0, chunk-window), chunk):
                    positions.extend([old[1+5*prior]['img_position_ids'], old[5+5*prior]['img_position_ids']])
                positions.extend([old[1+5*chunk]['img_position_ids'], old[2+5*chunk]['img_position_ids']])
                for call in calls[1+3*chunk:1+3*chunk+3]:
                    torch.testing.assert_close(call['img_position_ids'], torch.cat(positions, dim=1), rtol=0, atol=0)
                torch.testing.assert_close(driver._video_rows(records[3*chunk]['current'], 0, values[chunk].shape[1]),
                                           old[2+5*chunk]['x'][0], rtol=0, atol=0)
                if window and chunk < 3:
                    torch.testing.assert_close(driver._video_rows(records[3*(chunk+1)]['eps'][:, -5:], 0, 5),
                                               old[5+5*chunk]['eps'][0], rtol=0, atol=0)

    @torch.no_grad()
    def test_caller_and_source_buffer_reuse_do_not_change_history_or_predictions(self):
        with actual_model() as net:
            static, picture, chunks = stream_inputs()
            static['prompt_embeds'] = static['prompt_embeds'].bfloat16()
            def run(mutate):
                driver = streaming_driver()
                outputs, consumed = [], []
                shared = torch.empty_like(chunks[0]['references'][0]['video'])
                def source():
                    for k, item in enumerate(chunks):
                        consumed.append(k)
                        shared.fill_(k+.25)
                        yield dict(item, references=[dict(item['references'][0], video=shared[:, :item['references'][0]['video'].shape[1]])])
                stream = driver.stream_latents(net, static, source(), first_frame=picture,
                                               seed=73, condition_seed=94)
                for k, value in enumerate(stream):
                    self.assertEqual(consumed, list(range(k+1)))
                    self.assertFalse(value.requires_grad)
                    outputs.append(value.clone())
                    if mutate:
                        value.fill_(999)
                        shared.fill_(-999)
                return outputs
            expected, changed = run(False), run(True)
            for a, b in zip(expected, changed, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_guidance_branches_keep_noise_and_static_caches_separate(self):
        plain = streaming_driver()
        plain_records = observe_windows(plain)
        run_stream(plain)
        for mode in ('cfg', 'gft'):
            for drop in ('proxy', 'prompt', 'prompt_proxy'):
                driver = streaming_driver(mode, drop=drop)
                records = observe_windows(driver)
                static, picture, chunks = stream_inputs()
                weak = dict(prompt_embeds=static['prompt_embeds']*.3, text_token_tags=static['text_token_tags'].clone())
                for chunk in chunks:
                    chunk['guidance_conditions'] = {drop: dict(teacher_condition=weak,
                        student_condition=dict(static=weak, chunks=[]))}
                net = CacheProbe()
                values = list(driver.stream_latents(net, static, chunks, first_frame=picture,
                                                    seed=73, condition_seed=94))
                branches = 2 if mode == 'cfg' else 1
                self.assertEqual(len(net.calls), branches*13)
                for chunk in range(4):
                    for step in range(3):
                        full = records[branches*(3*chunk+step)]
                        if step == 0:
                            torch.testing.assert_close(full['current'], plain_records[3*chunk]['current'], rtol=0, atol=0)
                        if branches == 2:
                            weak_record = records[branches*(3*chunk+step)+1]
                            torch.testing.assert_close(full['current'], weak_record['current'], rtol=0, atol=0)
                            torch.testing.assert_close(full['history'], weak_record['history'], rtol=0, atol=0)
                            torch.testing.assert_close(full['eps'], weak_record['eps'], rtol=0, atol=0)
                            self.assertIsNot(full['cache'], weak_record['cache'])
                            self.assertEqual(len(weak_record['references']), 2 if drop == 'prompt' else 1)
                self.assertEqual(values[-1].shape[1], 2)

    def test_invalid_streams_and_weak_static_drift(self):
        static, picture, chunks = stream_inputs()
        driver = streaming_driver()
        def stream(items, runner=driver):
            return runner.stream_latents(CacheProbe(), static, items, first_frame=picture, seed=73, condition_seed=94)
        first = next(stream(chunks))
        torch.testing.assert_close(first, next(stream(chunks[:1])), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'after EOS'):
            list(stream(chunks+[chunks[0]]))
        bad = copy.deepcopy(chunks)
        bad[1]['references'][0]['role'] = 'depth'
        with self.assertRaisesRegex(ValueError, 'roles or resolution'):
            list(stream(bad))
        bad = copy.deepcopy(chunks)
        bad[0]['condition'] = static
        with self.assertRaisesRegex(ValueError, 'omit Qwen'):
            list(stream(bad))
        bad = copy.deepcopy(chunks)
        for index, item in enumerate(bad):
            doc = dict(prompt_embeds=static['prompt_embeds']+index, text_token_tags=static['text_token_tags'])
            item['guidance_conditions'] = dict(prompt=dict(teacher_condition=doc, student_condition=dict(static=doc, chunks=[])))
        with self.assertRaisesRegex(ValueError, 'weak static Qwen condition changed'):
            list(stream(bad, streaming_driver('cfg', drop='prompt')))


class BidirAttentionTests(unittest.TestCase):
    @torch.no_grad()
    def test_actual_cfg_combines_independent_weak_prefix_predictions(self):
        driver = streaming_driver('cfg', drop='prompt')
        sample = synthetic(frames=39)
        doc = dict(prompt_embeds=-3*sample['teacher_condition']['prompt_embeds'],
                   text_token_tags=sample['teacher_condition']['text_token_tags'])
        sample['guidance_conditions'] = dict(prompt=dict(teacher_condition=doc, student_condition=dict(static=doc, chunks=[])))
        inputs = driver._build_inputs({'ref_samples': [sample]}, streaming=True)
        weak = driver._sampling_inputs(inputs)
        sample = inputs.samples[0]
        args = dict(video_xts=[sample['video']], video_context=[sample['video']],
                    video_eps=[sample['video']*.3], video_timesteps=[sample['video'].new_tensor([0., .7, 0.])])
        with actual_model() as net:
            cache, weak_cache = (GroupedKVCache(2, 1, windows=[1]) for _ in range(2))
            driver._sampling_forward(net, inputs, selection=('static', -1), store_kv=True,
                cache=cache, weak=weak, weak_cache=weak_cache, **args)
            strong = driver._ref_forward(net, inputs, selection=('window', 1), cache=cache, **args)[0][0]
            negative = driver._ref_forward(net, weak, selection=('window', 1), cache=weak_cache, **args)[0][0]
            guided = driver._sampling_forward(net, inputs, selection=('window', 1), cache=cache,
                                             weak=weak, weak_cache=weak_cache, **args)[0][0]
            self.assertGreater((strong-negative).abs().max().item(), .0001)
            torch.testing.assert_close(guided, negative.float()+2*(strong.float()-negative.float()), rtol=0, atol=0)

    @torch.no_grad()
    def test_actual_two_layer_history_mediation_and_full_window_oracle(self):
        driver = adapter()
        sample = synthetic(frames=39)
        inputs = driver._build_inputs({'ref_samples': [sample]}, streaming=True)
        sample = inputs.samples[0]
        g = driver._ref_geometry(inputs, ('window', 1))
        history = torch.cat([torch.arange(b.start, b.stop) for b in g['blocks'] if b.kind == 'history'])
        self.assertEqual([(b.kind, b.chunk) for b in g['blocks']],
                         [('condition', 0), ('history', 0), ('condition', 1), ('noise', 1)])
        args = dict(video_xts=[sample['video'].clone()], video_context=[sample['video'].clone()],
                    video_eps=[sample['video']*.3], video_timesteps=[sample['video'].new_tensor([0., .7, 0.])])
        with actual_model() as net:
            cache = GroupedKVCache(2, 1, windows=[1])
            captured = []
            hook = net.register_forward_pre_hook(lambda module, args, kwargs: captured.append(dict(kwargs)), with_kwargs=True)
            static_states = []
            static_hook = net.dit.blocks[0].register_forward_hook(lambda module, args, output: static_states.append(output.clone()))
            driver._ref_forward(net, inputs, selection=('static', -1), cache=cache, store_kv=True, **args)
            static_hook.remove()
            states = []
            layer_hook = net.dit.blocks[0].register_forward_hook(lambda module, args, output: states.append(output.clone()))
            baseline = driver._ref_forward(net, inputs, selection=('window', 1), cache=cache, **args)[0][0]
            baseline_h = states[-1][history].clone()
            args['video_xts'][0][:, 5:10].add_(2.)
            changed = driver._ref_forward(net, inputs, selection=('window', 1), cache=cache, **args)[0][0]
            changed_h = states[-1][history].clone()
            self.assertGreater((baseline_h-changed_h).abs().max().item(), .01)
            layer_hook.remove()
            def clamp(module, call, output):
                value = output.clone()
                value[history] = baseline_h
                return value
            clamp_hook = net.dit.blocks[0].register_forward_hook(clamp)
            clamped = driver._ref_forward(net, inputs, selection=('window', 1), cache=cache, **args)[0][0]
            clamp_hook.remove()
            hook.remove()
            self.assertGreater((changed-clamped).abs().max().item(), .00001)
            self.assertFalse(torch.equal(baseline, changed))
            oracle_args = full_window(captured[0], captured[2])
            oracle_states = []
            oracle_hook = net.dit.blocks[0].register_forward_hook(lambda module, args, output: oracle_states.append(output.clone()))
            oracle = net(**oracle_args)[0]
            expected = driver._video_rows(changed, 0, 5)
            tolerance = .02 if oracle.is_cuda else .00001
            torch.testing.assert_close(oracle, expected, rtol=0, atol=tolerance)
            torch.testing.assert_close(oracle_states[-1][:10], static_states[0], rtol=0, atol=tolerance)
            torch.testing.assert_close(oracle_states[-1][10+history], changed_h, rtol=0, atol=tolerance)
            oracle_hook.remove()
            self.assertEqual(cache.groups, [[(-1, 10)]])

    @torch.no_grad()
    def test_fixed_kv_negative_control_has_no_current_to_history_link(self):
        driver = adapter()
        sample = synthetic(frames=39)
        inputs = driver._build_inputs({'ref_samples': [sample]}, streaming=True)
        sample = inputs.samples[0]
        args = dict(video_xts=[sample['video'].clone()], video_context=[sample['video'].clone()],
                    video_eps=[sample['video']*.3], video_timesteps=[sample['video'].new_tensor([0., .7, 0.])])
        with actual_model() as net:
            cache = GroupedKVCache(2, 1, windows=[1])
            driver._ref_forward(net, inputs, selection=('static', -1), cache=cache, store_kv=True, **args)
            cache.begin_chunk(0)
            driver._ref_forward(net, inputs, selection=('condition', 0), cache=cache, store_kv=True, **args)
            driver._ref_forward(net, inputs, selection=('history', 0), cache=cache, store_kv=True, **args)
            cache.begin_chunk(1)
            driver._ref_forward(net, inputs, selection=('condition', 1), cache=cache, store_kv=True, **args)
            frozen = [value.clone() for value in cache.key_cache.values()]
            before = driver._ref_forward(net, inputs, selection=('noise', 1), cache=cache, **args)[0][0]
            args['video_xts'][0][:, 5:10].add_(2.)
            after = driver._ref_forward(net, inputs, selection=('noise', 1), cache=cache, **args)[0][0]
            self.assertFalse(torch.equal(before, after))
            for value, snapshot in zip(cache.key_cache.values(), frozen, strict=True):
                torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
            self.assertEqual([b.kind for b in driver._ref_geometry(inputs, ('noise', 1))['blocks']], ['noise'])

    @torch.no_grad()
    def test_static_cache_is_immutable_with_arena_and_isolated_sp_padding(self):
        driver = adapter()
        driver._sp_size = lambda: 4
        driver._pad_rows = lambda n: 4-n%4
        sample = synthetic(frames=39, depth=True)
        inputs = driver._build_inputs({'ref_samples': [sample]}, streaming=True)
        sample = inputs.samples[0]
        args = dict(video_xts=[sample['video'].clone()], video_context=[sample['video'].clone()],
                    video_eps=[sample['video']*.3], video_timesteps=[sample['video'].new_tensor([.9, .7, .5])])
        with actual_model() as net:
            for arena in (False, True):
                configure_inference(net, metadata=True, kv=arena, adaln=True)
                cache = GroupedKVCache(2, 1, windows=[1], padding=True)
                driver._ref_forward(net, inputs, selection=('static', -1), cache=cache, store_kv=True, **args)
                frozen = [[v.clone() for v in table.values()] for table in (cache.key_cache, cache.value_cache)]
                groups, lengths = copy.deepcopy(cache.groups), list(cache.kvlens)
                for chunk in (0, 1, 2):
                    geometry = driver._ref_geometry(inputs, ('window', chunk))
                    self.assertEqual(sum(geometry['sample_lens'])%4, 0)
                    self.assertGreater(geometry['sample_lens'][-1], 0)
                    before = driver._ref_forward(net, inputs, selection=('window', chunk), cache=cache, **args)[0][0]
                    def poison(module, call, kwargs):
                        kwargs['x'] = kwargs['x'].clone()
                        kwargs['x'][:, -geometry['pad']:] = 1000
                        return call, kwargs
                    hook = net.register_forward_pre_hook(poison, with_kwargs=True)
                    after = driver._ref_forward(net, inputs, selection=('window', chunk), cache=cache, **args)[0][0]
                    hook.remove()
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
                    self.assertEqual(cache.groups, groups)
                    self.assertEqual(cache.kvlens, lengths)
                    self.assertEqual(cache.kvlens[-1], 0)
                    for table, snapshots in zip((cache.key_cache, cache.value_cache), frozen, strict=True):
                        for value, snapshot in zip(table.values(), snapshots, strict=True):
                            torch.testing.assert_close(value, snapshot, rtol=0, atol=0)

    def test_window_rejects_writes_and_missing_or_dynamic_cache(self):
        driver = adapter()
        inputs = driver._build_inputs({'ref_samples': [synthetic(frames=5)]})
        args = dict(video_xts=[], video_timesteps=[], selection=('window', 0))
        with self.assertRaisesRegex(ValueError, 'read-only static'):
            driver._ref_forward(None, inputs, **args)
        cache = GroupedKVCache(2, 1, windows=[1])
        with self.assertRaisesRegex(ValueError, 'read-only static'):
            driver._ref_forward(None, inputs, cache=cache, store_kv=True, **args)
        cache.groups[0].append((0, 4))
        with self.assertRaisesRegex(ValueError, 'static-only'):
            driver._ref_forward(None, inputs, cache=cache, **args)


if __name__ == '__main__':
    unittest.main()
