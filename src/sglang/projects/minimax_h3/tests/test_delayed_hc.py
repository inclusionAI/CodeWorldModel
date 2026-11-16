"""Delayed clean H/C must preserve output and cache state at denoise boundaries."""
from dataclasses import replace
from copy import deepcopy
import os
import unittest
from unittest.mock import patch

import torch

from projects.minimax_h3.modeling.inference_optimizations import configure_inference
from projects.minimax_h3.inference.raw_recipe import RAW_PROTOCOLS
from projects.minimax_h3.tests.bidir_history_support import actual_model
from projects.minimax_h3.tests.test_raw_latent import session


def cache_snapshot(cache):
    return (list(cache.owners), list(cache.kvlens),
            [(cache.key_cache[i].clone(), cache.value_cache[i].clone())
             for i in cache.key_cache])


class DelayedHCTests(unittest.TestCase):
    @torch.no_grad()
    def test_failed_pair_keeps_parent_cache_and_pending_history(self):
        with actual_model(fp32=True) as net:
            configure_inference(net, delayed_hc=True)
            stream, chunks = session(net, RAW_PROTOCOLS[0], recent_latents=1)
            stream.push(chunks[0])
            stream.push(chunks[1])
            stream.push(chunks[2])
            parent = stream.fixed['positive']
            expected = cache_snapshot(parent)
            groups = deepcopy(parent.groups)
            pending = stream.pending_history
            self.assertTrue(pending)
            original = net.dit.blocks[1].forward

            def fail(*args, **kwargs):
                if kwargs.get('hc_layout') is not None:
                    raise RuntimeError('paired block failed')
                return original(*args, **kwargs)

            with patch.object(net.dit.blocks[1], 'forward', side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, 'paired block failed'):
                    stream.push(chunks[3])
            self.assertIs(stream.fixed['positive'], parent)
            self.assertIs(stream.pending_history, pending)
            self.assertEqual(parent.groups, groups)
            self.compare_cache(expected, cache_snapshot(parent))

    @torch.no_grad()
    @unittest.skipIf(os.environ.get('H3_BIDIR_TEST_DEVICE', 'cpu') != 'cpu',
                     'FP32 visibility oracle uses CPU SDPA')
    def test_two_layer_segments_preserve_pre_retain_history_visibility(self):
        from projects.minimax_h3.inference.raw_latent import LatentKVCache
        from projects.minimax_h3.modeling.transformer.causal_model import (
            CachedAttentionSegments, segmented_cached_attention)
        from torch.nn.functional import scaled_dot_product_attention
        generator = torch.Generator().manual_seed(51)
        hidden = torch.randn(6, 8, generator=generator)
        weights = [torch.randn(8, 24, generator=generator)*.2 for _ in range(2)]
        keys = [torch.randn(4, 2, 4, generator=generator) for _ in range(2)]
        values = [torch.randn(4, 2, 4, generator=generator) for _ in range(2)]
        commit = torch.tensor([0, 3, 5, 6, 7, 8])
        route = CachedAttentionSegments(
            torch.cat((torch.arange(7), commit, torch.tensor([9]))),
            torch.tensor([3, 2, 1], dtype=torch.int32),
            torch.tensor([7, 6, 1], dtype=torch.int32), commit)

        def attention(q, k, v):
            return scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1),
                v.transpose(0, 1), scale=.5).transpose(0, 1).flatten(1)

        def run(x, *, fused, old_keys=keys):
            cache = LatentKVCache(2)
            cache.kvlens[0] = 4
            cache.key_cache = {i: k.clone() for i, k in enumerate(old_keys)}
            cache.value_cache = {i: v.clone() for i, v in enumerate(values)}
            for layer, weight in enumerate(weights):
                q, k, v = (x@weight).reshape(6, 3, 2, 4).unbind(1)
                if fused:
                    out = segmented_cached_attention(q, k, v, cache, layer, route, .5).flatten(1)
                else:
                    all_k = torch.cat((old_keys[layer], k))
                    all_v = torch.cat((values[layer], v))
                    out = torch.cat((attention(q[:3], all_k[:7], all_v[:7]),
                        attention(q[3:5], all_k[commit], all_v[commit]),
                        attention(q[5:], k[5:], v[5:])))
                    cache.key_cache[layer] = all_k[commit]
                    cache.value_cache[layer] = all_v[commit]
                x = x + out
            return x, cache

        with actual_model():
            expected, expected_cache = run(hidden, fused=False)
            actual, actual_cache = run(hidden, fused=True)
            torch.testing.assert_close(expected, actual)
            for layer in range(2):
                torch.testing.assert_close(expected_cache.key_cache[layer], actual_cache.key_cache[layer])
                torch.testing.assert_close(expected_cache.value_cache[layer], actual_cache.value_cache[layer])
            future = hidden.clone()
            future[3:5] += 7
            perturbed, perturbed_cache = run(future, fused=True)
            torch.testing.assert_close(actual[:3], perturbed[:3], rtol=0, atol=0)
            torch.testing.assert_close(actual_cache.key_cache[1][:4],
                                       perturbed_cache.key_cache[1][:4], rtol=0, atol=0)
            evicted = [k.clone() for k in keys]
            evicted[0][1] += 5
            changed, _ = run(hidden, fused=True, old_keys=evicted)
            self.assertFalse(torch.allclose(actual[:3], changed[:3]))

    def compare_cache(self, expected, actual):
        self.assertEqual(expected[:2], actual[:2])
        for old, new in zip(expected[2], actual[2], strict=True):
            for a, b in zip(old, new, strict=True):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def run_stream(self, net, enabled, *, protocol=RAW_PROTOCOLS[0], cfg=None,
                   padding=False, kv=True, **recipe_options):
        configure_inference(net, delayed_hc=enabled, kv=kv)
        stream, chunks = session(net, protocol, **recipe_options)
        if cfg:
            stream.recipe = replace(stream.recipe, guidance=dict(mode='cfg', drop=cfg, scale=2.))
            stream.model.recipe = stream.recipe
        if padding:
            stream.model._sp_size = lambda: 4
            stream.model._pad_rows = lambda n: 4-n%4
        states, calls, prefixes = [], [], {}
        forward = stream.model.forward

        def capture(backbone, inputs, frames, timestep, cache, **kwargs):
            if not kwargs.get('store') and any(cache is c for c in stream.fixed.values()):
                states.append(cache_snapshot(cache))
            route = kwargs.get('cache_segments')
            if route is not None:
                self.assertTrue(bool((route.query_lens > 0).all()))
                self.assertTrue(bool((route.key_lens > 0).all()))
                calls.append(route)
            return forward(backbone, inputs, frames, timestep, cache, **kwargs)

        stream.model.forward = capture
        outputs = []
        source = chunks[:-1] + [chunks[0]]*2 + chunks[-1:]
        for i, item in enumerate(source):
            if stream.recipe.streaming_text:
                item = dict(item, prompt='same text' if i < 4 else 'changed text', prompt_revision=i+1)
            if cfg in ('prompt', 'prompt_proxy'):
                weak = {key: value.clone() for key, value in stream.static.items()}
                weak['prompt_embeds'] += .3
                item = dict(item, guidance_conditions={cfg: dict(student_condition=dict(static=weak))})
            outputs.extend(stream.push(item))
            for branch, cache in stream.prefix.items():
                if branch not in prefixes:
                    prefixes[branch] = cache_snapshot(cache)
                else:
                    self.compare_cache(prefixes[branch], cache_snapshot(cache))
            if enabled and stream.pending_history and stream.recipe.recent_latents == 0:
                self.assertTrue(any(f.position not in stream.frames for f in stream.pending_history))
        self.assertFalse(stream.pending_history)
        states.extend(cache_snapshot(stream.fixed[branch]) for branch in stream._branches())
        if padding:
            for cache in stream.fixed.values():
                self.assertEqual(cache.kvlens[-1], 0)
                self.assertEqual(len(cache.owners), len(cache.key_cache[0]))
        return outputs, states, calls

    @torch.no_grad()
    @unittest.skipIf(os.environ.get('H3_BIDIR_TEST_DEVICE', 'cpu') != 'cpu',
                     'FP32 arena comparison uses CPU SDPA')
    def test_arena_matches_scatter_with_delayed_hc_guidance_text_and_padding(self):
        with actual_model(fp32=True) as net:
            for delayed in (False, True):
                for options in (dict(protocol=RAW_PROTOCOLS[2]), dict(cfg='proxy')):
                    with self.subTest(delayed=delayed, **options):
                        options.update(padding=True, recent_latents=1)
                        expected, caches, _ = self.run_stream(net, delayed, kv=False, **options)
                        actual, arena_caches, _ = self.run_stream(net, delayed, kv=True, **options)
                        for old, new in zip(expected, actual, strict=True):
                            torch.testing.assert_close(old.video, new.video, rtol=0, atol=1e-5)
                        for old, new in zip(caches, arena_caches, strict=True):
                            self.compare_cache(old, new)

    @torch.no_grad()
    @unittest.skipIf(os.environ.get('H3_BIDIR_TEST_DEVICE', 'cpu') != 'cpu',
                     'FP32 differential uses CPU SDPA')
    def test_outputs_and_every_layer_cache_match_at_logical_boundaries(self):
        with actual_model(fp32=True) as net:
            cases = [
                dict(recent_latents=recent, current_latent_schedule=schedule)
                for recent in (0, 1, 12) for schedule in ((3, 2), (5,))]
            cases += [
                dict(recent_latents=1, segment_boundaries=(11, 16, 22), padding=True),
                dict(recent_latents=1, protocol=RAW_PROTOCOLS[2]),
                dict(recent_latents=12, protocol=RAW_PROTOCOLS[2]),
                dict(recent_latents=1, cfg='proxy', padding=True),
                dict(recent_latents=1, cfg='prompt'),
            ]
            for options in cases:
                with self.subTest(**options):
                    expected, caches, _ = self.run_stream(net, False, **options)
                    actual, fused_caches, routes = self.run_stream(net, True, **options)
                    self.assertTrue(routes)
                    self.assertEqual(len(expected), len(actual))
                    for old, new in zip(expected, actual, strict=True):
                        self.assertEqual(old.plan, new.plan)
                        self.assertEqual(old.sources, new.sources)
                        torch.testing.assert_close(old.video, new.video, rtol=1e-5, atol=1e-5)
                    self.assertEqual(len(caches), len(fused_caches))
                    for old, new in zip(caches, fused_caches, strict=True):
                        self.compare_cache(old, new)

    @torch.no_grad()
    def test_bootstrap_only_eos_has_no_pending_history(self):
        with actual_model() as net:
            configure_inference(net, delayed_hc=True)
            stream, chunks = session(net, RAW_PROTOCOLS[0])
            output = stream.push(chunks[-1])
            self.assertEqual(len(output), 1)
            self.assertTrue(output[0].plan.eos)
            self.assertFalse(stream.pending_history)
            self.assertFalse(stream.fixed)

    def test_explicit_unsupported_recipe_and_invalid_flag_fail(self):
        with actual_model() as net:
            with self.assertRaisesRegex(ValueError, 'boolean'):
                configure_inference(net, delayed_hc='true')
            configure_inference(net, delayed_hc=True)
            for protocol in (RAW_PROTOCOLS[1], RAW_PROTOCOLS[3]):
                with self.assertRaisesRegex(ValueError, 'fixed-history v2'):
                    session(net, protocol)
            from projects.minimax_h3.inference.recipes.ref_fixed_history import RefFixedHistory
            from projects.minimax_h3.inference.recipes.ref_bidir_history import RefBidirHistory
            from projects.minimax_h3.modeling.ref_recipe import RefRecipe, BIDIR_RECIPE
            from projects.minimax_h3.tests.test_ref_inference import inputs, recipe_config
            static, picture, chunks = inputs()
            recipe = RefRecipe.from_training(recipe_config())
            for driver in (RefFixedHistory(recipe), RefBidirHistory(replace(recipe, protocol=BIDIR_RECIPE))):
                with self.assertRaisesRegex(ValueError, 'fixed-history v2'):
                    next(driver.stream_latents(net, static, chunks, first_frame=picture,
                                               conditioning_mode=recipe.conditioning_mode))


if __name__ == '__main__':
    unittest.main()
