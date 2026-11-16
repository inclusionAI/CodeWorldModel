"""Two-layer H3 text-to-history mediation and independent masked full oracle."""
import copy
import unittest

import torch

from projects.minimax_h3.modeling.inference_optimizations import configure_inference
from projects.minimax_h3.modeling.ref_condition import GroupedKVCache
from projects.minimax_h3.tests.bidir_history_support import actual_model, full_window
from projects.minimax_h3.tests.test_streaming_text import driver, document, DRIVERS
from projects.minimax_h3.tests.test_ref_inference import inputs as raw_inputs
from projects.minimax_h3.tools.ref2va_acceptance import synthetic


def packed(runner, current='bb', previous='a'):
    sample = synthetic(frames=39)
    sample['metadata']['conditioning_mode'] = 'legacy'
    sample['student_condition']['chunks'] = [document(text, torch.bfloat16) for text in (previous, current, 'ccc')]
    inputs = runner._build_inputs({'ref_samples': [sample]}, streaming=True)
    sample = inputs.samples[0]
    args = dict(video_xts=[sample['video'].clone()], video_context=[sample['video'].clone()],
                video_eps=[sample['video']*.3], video_timesteps=[sample['video'].new_tensor([0., .7, 0.])])
    return inputs, args


class TextAttentionTests(unittest.TestCase):
    @torch.no_grad()
    def test_actual_stream_owns_reused_text_proxy_and_output_buffers(self):
        with actual_model() as net:
            for cls in DRIVERS:
                def run(mutate):
                    static, picture, chunks = raw_inputs()
                    static['prompt_embeds'] = static['prompt_embeds'].bfloat16()
                    shared_proxy = torch.empty_like(chunks[0]['references'][0]['video'])
                    shared_text = torch.empty(5, 32, dtype=torch.bfloat16)
                    consumed, outputs, retained = [], [], []
                    def source():
                        for k, item in enumerate(chunks):
                            consumed.append(k)
                            shared_proxy.fill_(k+.25)
                            yield dict(item, prompt=('A', 'BBBB', 'CC', 'DDDDD')[k],
                                references=[dict(item['references'][0], video=shared_proxy[:, :5 if k < 3 else 2])])
                    def encode(text):
                        shared_text.fill_(sum(text.encode())/100)
                        return dict(prompt_embeds=shared_text[:len(text)],
                                    text_token_tags=torch.ones(len(text), dtype=torch.long))
                    runner = driver(cls)
                    forward = runner._ref_forward
                    def observe(model, value, **kwargs):
                        if kwargs['selection'][0] == 'window':
                            retained.append(copy.deepcopy(value.samples[0]))
                        return forward(model, value, **kwargs)
                    runner._ref_forward = observe
                    for k, result in enumerate(runner.stream_latents(net, static, source(), first_frame=picture,
                            encode_text=encode, seed=73, condition_seed=94)):
                        self.assertEqual(consumed, list(range(k+1)))
                        outputs.append(result.video.clone())
                        if mutate:
                            result.video.fill_(999)
                            shared_proxy.fill_(999)
                            shared_text.fill_(999)
                            static['prompt_embeds'].fill_(999)
                            picture.fill_(999)
                    return outputs, retained
                expected, before = run(False)
                actual, after = run(True)
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                for left, right in zip(after, before):
                    for a, b in zip(left['student_condition']['chunks'], right['student_condition']['chunks']):
                        torch.testing.assert_close(a['prompt_embeds'], b['prompt_embeds'], atol=0, rtol=0)
                    for a, b in zip(left['references'], right['references']):
                        torch.testing.assert_close(a['video'], b['video'], atol=0, rtol=0)

    @torch.no_grad()
    def test_current_text_changes_h_and_mediates_current_prediction(self):
        runner = driver()
        baseline_inputs, args = packed(runner, 'bb')
        changed_inputs, changed_args = packed(runner, 'ZZZZZ')
        geometry = runner._ref_geometry(baseline_inputs, ('window', 1))
        changed_geometry = runner._ref_geometry(changed_inputs, ('window', 1))
        hist = torch.cat([torch.arange(b.start, b.stop) for b in geometry['blocks'] if b.kind == 'history'])
        changed_hist = torch.cat([torch.arange(b.start, b.stop) for b in changed_geometry['blocks'] if b.kind == 'history'])
        with actual_model() as net:
            cache = GroupedKVCache(2, 1, windows=[1])
            runner._ref_forward(net, baseline_inputs, selection=('static', -1), cache=cache, store_kv=True, **args)
            static_kv = [[v.clone() for v in table.values()] for table in (cache.key_cache, cache.value_cache)]
            states = []
            hook = net.dit.blocks[0].register_forward_hook(lambda m, a, value: states.append(value.clone()))
            baseline = runner._ref_forward(net, baseline_inputs, selection=('window', 1), cache=cache, **args)[0][0]
            baseline_h = states[-1][hist].clone()
            changed = runner._ref_forward(net, changed_inputs, selection=('window', 1), cache=cache, **changed_args)[0][0]
            changed_h = states[-1][changed_hist].clone()
            hook.remove()
            self.assertGreater((changed_h-baseline_h).abs().max().item(), .00001)
            def clamp(module, call, value):
                result = value.clone()
                result[changed_hist] = baseline_h
                return result
            hook = net.dit.blocks[0].register_forward_hook(clamp)
            clamped = runner._ref_forward(net, changed_inputs, selection=('window', 1), cache=cache, **changed_args)[0][0]
            hook.remove()
            self.assertGreater((changed-clamped).abs().max().item(), .000001)
            self.assertGreater((changed-baseline).abs().max().item(), .000001)
            for table, values in zip((cache.key_cache, cache.value_cache), static_kv):
                for actual, saved in zip(table.values(), values):
                    torch.testing.assert_close(actual, saved, atol=0, rtol=0)
            for name in ('video_context', 'video_eps', 'video_xts'):
                torch.testing.assert_close(args[name][0], changed_args[name][0], atol=0, rtol=0)
            torch.testing.assert_close(baseline_inputs.samples[0]['student_condition']['chunks'][0]['prompt_embeds'],
                changed_inputs.samples[0]['student_condition']['chunks'][0]['prompt_embeds'], atol=0, rtol=0)

    @torch.no_grad()
    def test_fixed_past_kv_is_frozen_and_keeps_prior_instruction_influence(self):
        runner = driver(DRIVERS[0])
        inputs, args = packed(runner)
        changed, changed_args = packed(runner, 'ZZZZZ')
        prior_changed, prior_args = packed(runner, previous='VVVV')
        with actual_model() as net:
            def prefill(value, kwargs):
                cache = GroupedKVCache(2, 1, windows=[1])
                for selection in (('static', -1), ('condition', 0), ('history', 0)):
                    runner._ref_forward(net, value, selection=selection, cache=cache, store_kv=True, **kwargs)
                return cache
            cache = prefill(inputs, args)
            snapshots = [[v.clone() for v in table.values()] for table in (cache.key_cache, cache.value_cache)]
            length = cache.kvlens[0]
            runner._ref_forward(net, changed, selection=('condition', 1), cache=cache, store_kv=True, **changed_args)
            output = runner._ref_forward(net, changed, selection=('noise', 1), cache=cache, **changed_args)[0][0]
            for table, old in zip((cache.key_cache, cache.value_cache), snapshots):
                for value, saved in zip(table.values(), old):
                    torch.testing.assert_close(value[:length], saved, atol=0, rtol=0)
            other_cache = prefill(prior_changed, prior_args)
            runner._ref_forward(net, changed, selection=('condition', 1), cache=other_cache, store_kv=True, **changed_args)
            other = runner._ref_forward(net, changed, selection=('noise', 1), cache=other_cache, **changed_args)[0][0]
            self.assertGreater((output-other).abs().max().item(), .000001)
            no_change_cache = prefill(inputs, args)
            runner._ref_forward(net, inputs, selection=('condition', 1), cache=no_change_cache, store_kv=True, **args)
            before = runner._ref_forward(net, inputs, selection=('noise', 1), cache=no_change_cache, **args)[0][0]
            self.assertGreater((output-before).abs().max().item(), .000001)

    @torch.no_grad()
    def test_variable_documents_full_oracle_positions_metadata_arena_and_sp_padding(self):
        for sp in (1, 4):
            for optimized in (False, True):
                runner = driver()
                runner._sp_size = lambda: sp
                runner._pad_rows = lambda n: 0 if sp == 1 else sp-n%sp
                inputs, args = packed(runner, 'bbbbbbb', previous='aaa')
                with actual_model() as net:
                    configure_inference(net, metadata=optimized, kv=optimized, adaln=optimized)
                    cache = GroupedKVCache(2, 1, windows=[1], padding=sp > 1)
                    captures = []
                    hook = net.register_forward_pre_hook(lambda m, a, kwargs: captures.append(dict(kwargs)), with_kwargs=True)
                    runner._ref_forward(net, inputs, selection=('static', -1), cache=cache, store_kv=True, **args)
                    expected = runner._ref_forward(net, inputs, selection=('window', 1), cache=cache, **args)[0][0]
                    hook.remove()
                    static, dynamic = captures
                    oracle_args = full_window(static, dynamic)
                    self.assertEqual(oracle_args['refiner_packed_seq_params']['cu_seqlens_q'].diff().tolist(), [6, 3, 7])
                    actual = net(**oracle_args)[0]
                    torch.testing.assert_close(actual, runner._video_rows(expected, 0, 5), atol=.00001, rtol=0)
                    geometry = runner._ref_geometry(inputs, ('window', 1))
                    if sp > 1:
                        self.assertGreater(geometry['pad'], 0)
                    different, _ = packed(runner, 'x', previous='aa')
                    other = runner._ref_geometry(different, ('window', 1))
                    def visual_positions(g):
                        return g['position_ids'][g['img_pos']]
                    torch.testing.assert_close(visual_positions(geometry), visual_positions(other), atol=0, rtol=0)
                    for block in geometry['blocks']:
                        rows = geometry['position_ids'][block.start:block.stop]
                        if block.kind == 'condition':
                            tags = geometry['token_tags'][block.start:block.stop]
                            text_rows = rows[tags == 1]
                            if len(text_rows):
                                self.assertEqual(len(text_rows[:, 0].unique()), 1)
                    self.assertEqual([g for g, _ in cache.groups[0]], [-1])


if __name__ == '__main__':
    unittest.main()
