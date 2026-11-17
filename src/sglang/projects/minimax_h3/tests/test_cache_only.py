"""Cache-only forwards preserve every layer and subsequent predictions."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from projects.minimax_h3.inference.raw_latent import LatentKVCache
from projects.minimax_h3.inference.raw_recipe import RAW_PROTOCOLS
from projects.minimax_h3.modeling.inference_optimizations import configure_inference
from projects.minimax_h3.modeling.transformer import causal_model, causal_model_sp
from projects.minimax_h3.tests.bidir_history_support import actual_model
from projects.minimax_h3.tests.test_raw_latent import session


def snapshot(cache):
    return (deepcopy((cache.owners, cache.groups, cache.kvlens)),
            {layer: (key.clone(), cache.value_cache[layer].clone())
             for layer, key in cache.key_cache.items() if key is not None})


class CacheOnlyTests(unittest.TestCase):
    def assert_cache_equal(self, expected, actual):
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1].keys(), actual[1].keys())
        for layer in expected[1]:
            for old, new in zip(expected[1][layer], actual[1][layer], strict=True):
                torch.testing.assert_close(old, new, rtol=0, atol=0)

    def run_stream(self, net, *, cache_only, delayed, proxy):
        configure_inference(net, delayed_hc=delayed)
        stream, chunks = session(net, RAW_PROTOCOLS[0], recent_latents=1)
        stream.model._sp_size = lambda: 4
        stream.model._pad_rows = lambda n: 4-n%4
        if proxy:
            stream.recipe = replace(stream.recipe, guidance=dict(mode='cfg', drop='proxy', scale=2.))
            stream.model.recipe = stream.recipe
        states, blocks, heads, calls = [], [], [], []
        current = False

        def before(module, args, kwargs):
            nonlocal current
            if not cache_only:
                kwargs['cache_only'] = False
            current = kwargs.get('cache_only', False)
            calls.append(current)

        def after(module, args, kwargs, result):
            self.assertEqual(result is None, current)
            if kwargs.get('update_past_key_values'):
                states.append(snapshot(kwargs['past_key_values']))

        def head(module, args):
            self.assertFalse(current, 'cache-only called final_layer')
            heads.append(1)

        with ExitStack() as stack:
            stack.enter_context(net.dit.register_forward_pre_hook(before, with_kwargs=True))
            stack.enter_context(net.dit.register_forward_hook(after, with_kwargs=True))
            stack.enter_context(net.dit.final_layer.register_forward_pre_hook(head))
            for block in net.dit.blocks:
                stack.enter_context(block.register_forward_hook(lambda *args: blocks.append(1)))
            outputs = [output for chunk in chunks for output in stream.push(chunk)]
        states.extend(snapshot(cache) for cache in stream.fixed.values())
        self.assertEqual(len(blocks), len(calls)*len(net.dit.blocks))
        self.assertEqual(len(heads), len(calls)-sum(calls))
        if cache_only:
            self.assertTrue(any(calls))
        return outputs, states, len(blocks), len(heads)

    @torch.no_grad()
    def test_exact_kv_and_complete_stream_with_and_without_sp1(self):
        for sp1 in (False, True):
            cls = causal_model_sp.CausalMiniMaxH3DiTModelSP if sp1 else causal_model.CausalMiniMaxH3DiTModel
            with patch.object(causal_model, 'CausalMiniMaxH3DiTModel', cls), actual_model(fp32=True) as net:
                for delayed, proxy in ((False, False), (False, True)):
                    with self.subTest(sp1=sp1, delayed=delayed, proxy=proxy):
                        old, old_states, old_blocks, old_heads = self.run_stream(
                            net, cache_only=False, delayed=delayed, proxy=proxy)
                        new, new_states, new_blocks, new_heads = self.run_stream(
                            net, cache_only=True, delayed=delayed, proxy=proxy)
                        self.assertEqual(old_blocks, new_blocks)
                        self.assertLess(new_heads, old_heads)
                        for a, b in zip(old, new, strict=True):
                            self.assertEqual(a.plan, b.plan)
                            self.assertEqual(a.sources, b.sources)
                            torch.testing.assert_close(a.video, b.video, rtol=0, atol=0)
                        for a, b in zip(old_states, new_states, strict=True):
                            self.assert_cache_equal(a, b)

    def test_invalid_cache_only_fails_before_cache_mutation(self):
        with actual_model(fp32=True) as net:
            cache = LatentKVCache(2)
            original = snapshot(cache)
            valid = dict(cache_only=True, past_key_values=cache, update_past_key_values=True)
            for changes, message in ((dict(cache_only='true'), 'boolean'),
                                     (dict(past_key_values=None), 'cache'),
                                     (dict(update_past_key_values=False), 'cache')):
                with torch.no_grad(), self.assertRaisesRegex(ValueError, message):
                    net.dit(**dict(valid, **changes))
                self.assert_cache_equal(original, snapshot(cache))
            with self.assertRaisesRegex(RuntimeError, 'eval and no-grad'):
                net.dit(**valid)
            net.train()
            with torch.no_grad(), self.assertRaisesRegex(RuntimeError, 'eval and no-grad'):
                net.dit(**valid)
            self.assert_cache_equal(original, snapshot(cache))

    @torch.no_grad()
    def test_cache_write_can_still_request_predictions(self):
        with actual_model(fp32=True) as net:
            stream, chunks = session(net, RAW_PROTOCOLS[0])
            captured = []

            def capture(module, args, kwargs):
                if kwargs.get('update_past_key_values') and not captured:
                    captured.append(dict(kwargs, past_key_values=kwargs['past_key_values'].fork()))

            with net.register_forward_pre_hook(capture, with_kwargs=True):
                for chunk in chunks:
                    stream.push(chunk)
            kwargs = dict(captured[0])
            kwargs.pop('cache_only')
            positions = kwargs['img_pos_info']['position_ids']
            kwargs['img_pos_for_infer_output_info'] = dict(position_ids=positions)
            kwargs['update_mask'] = torch.ones_like(positions)
            original_cache = kwargs['past_key_values']
            cache_only = original_cache.fork()
            output = net(**kwargs)
            self.assertGreater(output[0].numel(), 0)
            self.assertIsNone(net(**dict(kwargs, past_key_values=cache_only, cache_only=True)))
            self.assert_cache_equal(snapshot(original_cache), snapshot(cache_only))

    @torch.no_grad()
    def test_x0_cache_only_keeps_clean_context_transform(self):
        with actual_model(fp32=True) as net:
            x = torch.ones(1, 2, 96)
            audio = torch.ones(1, 2, 64)
            with patch.object(net.dit, 'forward', return_value=None) as forward:
                result = net(x=x, audio_x=audio, eps=torch.zeros_like(x),
                    audio_eps=torch.zeros_like(audio), unique_timesteps=torch.tensor([0., .3]),
                    inverse_indices=torch.tensor([0, 1]), token_tags=torch.tensor([0, 0]),
                    cache_only=True)
            self.assertIsNone(result)
            kwargs = forward.call_args.kwargs
            torch.testing.assert_close(kwargs['x'][0, 0], x[0, 0]*.999, rtol=0, atol=0)
            torch.testing.assert_close(kwargs['x'][0, 1], x[0, 1], rtol=0, atol=0)
            torch.testing.assert_close(kwargs['audio_x'][0, 0], audio[0, 0]*.999, rtol=0, atol=0)
            self.assertTrue(kwargs['cache_only'])


if __name__ == '__main__':
    unittest.main()
