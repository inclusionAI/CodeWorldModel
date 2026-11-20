"""First-denoise fusion preserves HC visibility and the D-only sampling contract."""
from contextlib import ExitStack
from dataclasses import replace
import unittest
from unittest.mock import patch
import weakref

import torch

from projects.minimax_h3.inference.raw_latent import LatentKVCache, RawLatentSession
from projects.minimax_h3.inference.raw_recipe import RAW_PROTOCOLS
from projects.minimax_h3.modeling.inference_optimizations import configure_inference
from projects.minimax_h3.modeling.transformer import causal_model, causal_model_sp, x0_model
from projects.minimax_h3.modeling.transformer.hc_layout import HCLayout
from projects.minimax_h3.tests.bidir_history_support import actual_model
from projects.minimax_h3.tests.test_cache_only import snapshot
from projects.minimax_h3.tests.test_hc_layout import route
from projects.minimax_h3.tests.test_raw_latent import session


class FirstDenoiseTests(unittest.TestCase):
    def capture_hcd(self, net, world):
        configure_inference(net, delayed_hc=True)
        stream, chunks = session(net, RAW_PROTOCOLS[0], fuse_first_denoise=True, recent_latents=1)
        stream.model._sp_size = lambda: world
        stream.model._pad_rows = lambda n: world-n % world if world > 1 else 0
        for chunk in chunks[:3]:
            stream.push(chunk)
        captured = []
        def capture(module, args, kwargs):
            captured.append(dict(kwargs))
            raise RuntimeError('captured HCD')
        with net.dit.register_forward_pre_hook(capture, with_kwargs=True):
            with self.assertRaisesRegex(RuntimeError, 'captured HCD'):
                stream.push(chunks[3])
        self.assertGreater(captured[0]['cache_segments'].denoise_rows, 0)
        return captured[0]

    def assert_cache_equal(self, expected, actual, *, atol=0):
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1].keys(), actual[1].keys())
        for layer in expected[1]:
            for a, b in zip(expected[1][layer], actual[1][layer]):
                torch.testing.assert_close(a, b, rtol=0, atol=atol)

    @torch.no_grad()
    def test_segmented_metadata_is_shared_per_forward_and_preserves_output_and_kv(self):
        with actual_model(fp32=True) as net:
            kwargs = self.capture_hcd(net, 8)
            initial = kwargs['past_key_values']

            def run(enabled):
                configure_inference(net, delayed_hc=True, metadata=enabled)
                rows = []
                def record(module, args, values):
                    rows[-1].append(values.get('attention_metadata'))
                with ExitStack() as stack:
                    stack.callback(net.dit.register_forward_pre_hook(
                        lambda *_: rows.append([])).remove)
                    for block in net.dit.blocks:
                        stack.callback(block.attn.register_forward_pre_hook(
                            record, with_kwargs=True).remove)
                    outputs, caches = [], []
                    for _ in range(2):
                        cache = initial.fork()
                        outputs.append(net.dit(**dict(kwargs, past_key_values=cache)))
                        caches.append(snapshot(cache))
                return outputs, caches, rows

            baseline, old_caches, disabled = run(False)
            candidate, new_caches, enabled = run(True)
            self.assertTrue(all(value is None for row in disabled for value in row))
            self.assertTrue(all(len(row) == len(net.dit.blocks) and
                                row[0] is not None and all(value is row[0] for value in row)
                                for row in enabled))
            self.assertIsNot(enabled[0][0], enabled[1][0])
            for old, new, old_cache, new_cache in zip(
                    baseline, candidate, old_caches, new_caches, strict=True):
                for a, b in zip(old, new, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assert_cache_equal(old_cache, new_cache)

    @torch.no_grad()
    def test_four_step_stream_preserves_sampler_rng_eos_and_published_cache(self):
        with actual_model(fp32=True) as net:
            configure_inference(net, delayed_hc=True)
            cases = [dict(recent_latents=n) for n in (0, 1, 12)]
            cases.append(dict(recent_latents=10, current_latent_schedule=(5,), segment_boundaries=(45, 70)))
            for options in cases:
                runs = []
                for fused in (False, True):
                    stream, chunks = session(net, RAW_PROTOCOLS[0], fuse_first_denoise=fused, **options)
                    stream.timesteps.timesteps = torch.tensor([1., .7, .4, .1], dtype=torch.float64)
                    stream.model._sp_size = lambda: 8
                    stream.model._pad_rows = lambda n: 8-n % 8
                    forwards, sampler_inputs, rng_ids, outputs = [], [], [], []
                    forward, sample = stream.model.forward, stream.sampler.step_to

                    def call(backbone, inputs, frames, timestep, cache, **kwargs):
                        segments = kwargs.get('cache_segments')
                        composite = segments is not None and segments.denoise_rows > 0
                        if not kwargs.get('store') and stream.fixed:
                            self.assertIs(cache, stream.fixed['positive'])
                        forwards.append((kwargs.get('store', False), composite))
                        return forward(backbone, inputs, frames, timestep, cache, **kwargs)

                    def step(**kwargs):
                        rng = kwargs['rng'][0]
                        generator = stream.model._generator(rng)
                        sampler_inputs.append((kwargs['seqlens'].clone(), kwargs['t'].clone(),
                                               generator.get_state().clone()))
                        rng_ids.append(id(rng))
                        torch.randn((), generator=generator)
                        return sample(**kwargs)

                    source = chunks[:-1] + [chunks[0]]*12 + chunks[-1:]
                    with patch.object(stream.model, 'forward', side_effect=call), \
                            patch.object(stream.sampler, 'step_to', side_effect=step):
                        for chunk in source:
                            outputs.extend(stream.push(chunk))
                    self.assertFalse(stream.pending_history)
                    self.assertTrue(outputs[-1].plan.eos)
                    self.assertEqual(len(sampler_inputs), len(outputs)*4)
                    self.assertTrue(all(out.metrics['nfe'] == 4 for out in outputs))
                    for start in range(0, len(rng_ids), 4):
                        self.assertEqual(len(set(rng_ids[start:start+4])), 1)
                    composite_count = sum(composite for _, composite in forwards)
                    prediction_count = sum(not store or composite for store, composite in forwards)
                    self.assertEqual(prediction_count, len(outputs)*4)
                    if fused:
                        self.assertEqual(composite_count, len(outputs)-2)
                    runs.append((outputs, sampler_inputs, snapshot(stream.fixed['positive']), len(forwards)))
                old, new = runs
                with self.subTest(**options):
                    self.assertEqual(old[3]-new[3], len(new[0])-2)
                    for a, b in zip(old[0], new[0], strict=True):
                        self.assertEqual(a.plan, b.plan)
                        self.assertEqual(a.sources, b.sources)
                        torch.testing.assert_close(a.video, b.video, rtol=1e-5, atol=1e-5)
                    for a, b in zip(old[1], new[1], strict=True):
                        for x, y in zip(a, b):
                            torch.testing.assert_close(x, y, rtol=0, atol=0)
                    self.assert_cache_equal(old[2], new[2], atol=1e-5)

    @torch.no_grad()
    def test_hcd_wrapper_preserves_clean_eps_timestep_and_full_pack_output_indexes(self):
        with actual_model(fp32=True) as net:
            configure_inference(net, delayed_hc=True)
            stream, chunks = session(net, RAW_PROTOCOLS[0], fuse_first_denoise=True, recent_latents=1)
            stream.model._sp_size = lambda: 8
            stream.model._pad_rows = lambda n: 8-n % 8
            for chunk in chunks[:3]:
                stream.push(chunk)
            frames = list(stream.pending_history)
            current = [replace(frames[-1], position=frames[-1].position+1, video=None)]
            all_frames = tuple(frames)+tuple(current)
            pack = stream._inputs(all_frames, 'positive', history_before=current[0].position,
                                  current_start=current[0].position)
            g = pack.geometry
            h, c = g['hc_rows']
            d = g['denoise_rows']
            self.assertEqual(g['output_pos'].tolist(), list(range(h+c, h+c+d)))
            self.assertEqual([s[1] for s in g['sources']],
                             ['history']*len(frames)+['condition']*len(current)+['noise']*len(current))
            captured = []
            def capture(module, args, kwargs):
                captured.append(kwargs)
            with net.register_forward_pre_hook(capture, with_kwargs=True), \
                    patch.object(net.dit, 'forward', return_value=(torch.zeros(d, 96), torch.zeros(0, 32))):
                result = stream.model.forward(net, pack, all_frames, torch.tensor(.7), stream._cache(), store=True)
            kwargs = captured[0]
            times = kwargs['unique_timesteps'][kwargs['inverse_indices']]
            torch.testing.assert_close(times[:h+c], torch.zeros(h+c), rtol=0, atol=0)
            torch.testing.assert_close(times[h+c:h+c+d], torch.full((d,), .7), rtol=0, atol=0)
            torch.testing.assert_close(times[h+c+d:], torch.zeros(g['pad']), rtol=0, atol=0)
            torch.testing.assert_close(kwargs['eps'][0, :h], torch.cat([
                stream.model._video_rows(f.eps, 0, 1) for f in frames]), rtol=0, atol=0)
            torch.testing.assert_close(kwargs['eps'][0, h:h+c], torch.cat([
                stream.model._video_rows(f.reference_eps[0], 0, 1) for f in current]), rtol=0, atol=0)
            torch.testing.assert_close(result[0][0], torch.cat([f.noise for f in current], 1), rtol=0, atol=0)
            self.assertTrue(kwargs['update_past_key_values'])
            self.assertFalse(kwargs.get('cache_only', False))
            torch.testing.assert_close(kwargs['img_pos_for_infer_output_info']['position_ids'], g['output_pos'])

    @torch.no_grad()
    def test_failures_in_block_head_and_x0_leave_parent_cache_and_pending_h_unchanged(self):
        with actual_model(fp32=True) as net:
            configure_inference(net, delayed_hc=True)
            for failure in ('block', 'head', 'x0'):
                with self.subTest(failure=failure):
                    stream, chunks = session(net, RAW_PROTOCOLS[0], fuse_first_denoise=True, recent_latents=1)
                    for chunk in chunks[:3]:
                        stream.push(chunk)
                    cache, pending = stream.fixed['positive'], stream.pending_history
                    expected = snapshot(cache)
                    target, name = ((net.dit.blocks[1], 'forward') if failure == 'block' else
                                    (net.dit.final_layer, 'forward') if failure == 'head' else
                                    (x0_model, 'minimax_h3_rf_v_to_x0'))
                    with patch.object(target, name, side_effect=RuntimeError(failure)):
                        with self.assertRaisesRegex(RuntimeError, failure):
                            stream.push(chunks[3])
                    self.assertIs(stream.fixed['positive'], cache)
                    self.assertIs(stream.pending_history, pending)
                    self.assert_cache_equal(expected, snapshot(cache))

    @torch.no_grad()
    def test_success_releases_old_cache_before_first_sampler_step(self):
        with actual_model(fp32=True) as net:
            configure_inference(net, delayed_hc=True)
            stream, chunks = session(net, RAW_PROTOCOLS[0], fuse_first_denoise=True, recent_latents=1)
            for chunk in chunks[:3]:
                stream.push(chunk)
            old = weakref.ref(stream.fixed['positive'])
            sample = stream.sampler.step_to
            calls = []
            def step(**kwargs):
                self.assertIsNone(old())
                calls.append(1)
                return sample(**kwargs)
            stream.sampler.step_to = step
            stream.push(chunks[3])
            self.assertTrue(calls)

    @torch.no_grad()
    def test_three_role_blocks_match_independent_h_then_c_then_read_only_d(self):
        with actual_model(fp32=True) as net:
            generator = torch.Generator().manual_seed(912)
            h, c, d, hp, cp, dp, old = 3, 2, 4, 1, 2, 2, 4
            x = torch.randn(h+c+d+hp+cp+dp, 128, generator=generator)
            ids = torch.randint(0, 3, (len(x),), generator=generator)
            cos_sin = torch.randn(len(x), 96, generator=generator)
            params = [tuple(torch.randn(3, 128, generator=generator)*.1 for _ in range(6))
                      for _ in net.dit.blocks]
            initial = LatentKVCache(len(net.dit.blocks))
            for layer in initial.key_cache:
                initial.key_cache[layer] = torch.randn(old, 2, 128, generator=generator)
                initial.value_cache[layer] = torch.randn(old, 2, 128, generator=generator)
            initial.kvlens = [old]
            retained = [0, 3, 5, 6]
            commit = retained + [old+h, old+h+1]
            dkeys = commit+list(range(old+h+c, old+h+c+d))
            hcd = route(list(range(old+h))+commit+dkeys+list(range(old+h+c+d, old+len(x))),
                [h, c, d, hp, cp, dp], [old+h, len(commit), len(dkeys), hp, cp, dp], commit, (h, c), d)
            layout = HCLayout.prepare(hcd, world_size=1, rank=0, device='cpu')
            indexes = (torch.tensor([0, 1, 2, 9]), torch.tensor([3, 4, 10, 11]),
                       torch.tensor([5, 6, 7, 8, 12, 13]))

            def run(value, rows, cache, segments, roles=None, trace=None):
                for layer, block in enumerate(net.dit.blocks):
                    value = block(value, adaln_input=torch.empty(0), adaln_params=params[layer],
                        combined_indices=ids[rows], rope_cache=(cos_sin[rows], torch.arange(len(rows))),
                        past_key_values=cache, update_past_key_values=True,
                        cache_segments=segments, hc_layout=roles)
                    if trace is not None:
                        trace.append(value.clone())
                return value

            sequential = initial.fork()
            expected_h = run(x[indexes[0]], indexes[0], sequential,
                route(range(old+h+hp), [h, hp], [old+h, hp], range(old+h)))
            for layer in sequential.key_cache:
                sequential.key_cache[layer] = sequential.key_cache[layer][retained]
                sequential.value_cache[layer] = sequential.value_cache[layer][retained]
            prefix = len(retained)
            expected_c = run(x[indexes[1]], indexes[1], sequential,
                route(range(prefix+c+cp), [c, cp], [prefix+c, cp], range(prefix+c)))
            prefix += c
            before_d = snapshot(sequential)
            expected_d = run(x[indexes[2]], indexes[2], sequential,
                route(range(prefix+d+dp), [d, dp], [prefix+d, dp], range(prefix)))
            self.assert_cache_equal(before_d, snapshot(sequential))
            fused = initial.fork()
            fused_layers = []
            shapes = {name: [] for name in ('qkv_proj', 'out_proj', 'fc1', 'fc2')}
            with ExitStack() as stack:
                for block in net.dit.blocks:
                    for name in shapes:
                        module = getattr(block.attn if name.endswith('_proj') else block.mlp, name)
                        stack.enter_context(module.register_forward_pre_hook(
                            lambda module, args, name=name: shapes[name].append(len(args[0]))))
                actual = run(x[layout.input_indexes], layout.input_indexes, fused, hcd, layout, fused_layers)
            torch.testing.assert_close(actual, torch.cat((expected_h, expected_c, expected_d)), rtol=1e-6, atol=1e-6)
            self.assert_cache_equal(snapshot(sequential), snapshot(fused), atol=1e-6)
            for values in shapes.values():
                self.assertEqual(values, [h+hp, c+cp, d+dp]*len(net.dit.blocks))
            for role, unchanged in ((indexes[2], h+hp+c+cp), (indexes[1], h+hp)):
                future = x.clone()
                future[role] += 7
                changed_cache = initial.fork()
                changed = run(future[layout.input_indexes], layout.input_indexes, changed_cache, hcd, layout)
                torch.testing.assert_close(actual[:unchanged], changed[:unchanged], rtol=0, atol=0)
                if unchanged == h+hp+c+cp:
                    self.assert_cache_equal(snapshot(fused), snapshot(changed_cache))
            evicted = initial.fork()
            evicted.key_cache[0] = evicted.key_cache[0].clone()
            evicted.key_cache[0][1] += 5
            changed_layers = []
            run(x[layout.input_indexes], layout.input_indexes, evicted, hcd, layout, changed_layers)
            self.assertFalse(torch.allclose(fused_layers[0][:h], changed_layers[0][:h]))
            torch.testing.assert_close(fused_layers[0][h+hp:], changed_layers[0][h+hp:], rtol=0, atol=0)

    @torch.no_grad()
    def test_activation_rejects_unsupported_sessions(self):
        with actual_model() as net:
            with self.assertRaisesRegex(ValueError, 'first-denoise'):
                session(net, RAW_PROTOCOLS[0], fuse_first_denoise=True)
            configure_inference(net, delayed_hc=True)
            valid, _ = session(net, RAW_PROTOCOLS[0])
            for guidance in (dict(mode='gft', drop='proxy', scale=1.), dict(mode='cfg', drop='proxy', scale=2.)):
                with self.assertRaisesRegex(ValueError, 'first-denoise'):
                    RawLatentSession(net, replace(valid.recipe, guidance=guidance), valid.static,
                                     first_frame=valid.picture, fuse_first_denoise=True)
            with self.assertRaisesRegex(ValueError, 'boolean'):
                session(net, RAW_PROTOCOLS[0], fuse_first_denoise=1)
            for protocol in RAW_PROTOCOLS[1:]:
                with self.assertRaises(ValueError):
                    session(net, protocol, fuse_first_denoise=True)
            with patch('projects.minimax_h3.modeling.transformer.model.get_tp_world_size', return_value=2):
                with self.assertRaisesRegex(ValueError, 'first-denoise'):
                    session(net, RAW_PROTOCOLS[0], fuse_first_denoise=True)
            net._h3_inference['cuda_graph'] = True
            with self.assertRaisesRegex(ValueError, 'first-denoise'):
                session(net, RAW_PROTOCOLS[0], fuse_first_denoise=True)
            net._h3_inference['cuda_graph'] = False
            with torch.enable_grad(), self.assertRaisesRegex(ValueError, 'first-denoise'):
                session(net, RAW_PROTOCOLS[0], fuse_first_denoise=True)

    @torch.no_grad()
    def test_malformed_routes_fail_before_any_cache_write(self):
        with actual_model(fp32=True) as net:
            kwargs = self.capture_hcd(net, 8)
            segments, cache = kwargs['cache_segments'], kwargs['past_key_values']
            before = snapshot(cache)
            old = cache.seq_lens(0)
            hc = sum(segments.hc_rows)
            malformed = [
                dict(cache_only=True), dict(update_past_key_values=False), dict(past_key_values=None),
                dict(cache_segments=replace(segments, denoise_rows=-1)),
                dict(cache_segments=replace(segments, hc_rows=None)),
                dict(cache_segments=replace(segments, denoise_rows=segments.denoise_rows+1)),
                dict(cache_segments=replace(segments, query_lens=segments.query_lens[:-1])),
                dict(cache_segments=replace(segments, commit_indexes=torch.tensor([old+hc]))),
                dict(cache_segments=replace(segments, commit_indexes=torch.tensor([old+len(kwargs['token_tags'])-1]))),
                dict(cache_segments=replace(segments, key_indexes=segments.key_indexes+100000)),
                dict(img_pos_for_infer_output_info=dict(position_ids=torch.tensor([0]))),
                dict(audio_pos_info=dict(position_ids=torch.tensor([hc]))),
            ]
            for changes in malformed:
                with self.subTest(changes=changes.keys()), self.assertRaises(ValueError):
                    net.dit(**dict(kwargs, **changes))
                self.assert_cache_equal(before, snapshot(cache))
            with torch.enable_grad(), self.assertRaises(ValueError):
                net.dit(**kwargs)
            with patch.object(causal_model, 'get_tp_world_size', return_value=2), self.assertRaises(ValueError):
                net.dit(**kwargs)
            net.dit.train()
            with self.assertRaises(ValueError):
                net.dit(**kwargs)
            self.assert_cache_equal(before, snapshot(cache))

    @torch.no_grad()
    def test_d_head_and_sp_gather_use_only_standalone_d_rows_and_both_projection_widths(self):
        for world in (1, 2, 8):
            with self.subTest(world=world), actual_model(fp32=True) as net:
                kwargs = self.capture_hcd(net, world)
                kwargs.pop('local_embedding_layout', None)
                segments = kwargs['cache_segments']
                hc, d = sum(segments.hc_rows), segments.denoise_rows
                dp = world-d % world if world > 1 else 0
                pad_start = len(kwargs['token_tags'])-dp
                d_indexes = torch.tensor(list(range(hc, hc+d))+list(range(pad_start, pad_start+dp)))
                head_inputs, full = [], []
                embed = net.dit._embed
                def full_embed(**values):
                    result = embed(**values)
                    full.append(result[0])
                    return result
                def head(hidden, **values):
                    head_inputs.append((hidden.clone(), values['inverse_indices'].clone()))
                    return hidden[:, :96], hidden[:, :32]
                if world == 1:
                    with patch.object(net.dit, '_embed', new=full_embed), \
                            patch.object(net.dit.final_layer, 'forward', side_effect=head):
                        output = net.dit(**kwargs)
                    self.assertEqual(len(head_inputs[0][0]), d)
                    self.assertEqual(output[0].shape, (d, 96))
                    self.assertEqual(output[1].shape, (0, 32))
                    continue
                net.dit.__class__ = causal_model_sp.CausalMiniMaxH3DiTModelSP
                for rank in range(world):
                    with ExitStack() as stack:
                        for name, value in (('is_unified_parallel_initialized', True),
                                ('get_unified_parallel_world_size', world), ('get_unified_parallel_rank', rank)):
                            stack.enter_context(patch.object(causal_model_sp, name, return_value=value))
                        stack.enter_context(patch.object(causal_model_sp, 'get_unified_parallel_group', return_value=None))
                        stack.enter_context(patch.object(causal_model_sp, '_BF16_DTYPE', torch.float32))
                        stack.enter_context(patch.object(net.dit, '_embed', new=full_embed))
                        stack.enter_context(patch.object(net.dit, '_run_blocks', side_effect=lambda hidden, **kw: hidden))
                        stack.enter_context(patch.object(net.dit.final_layer, 'forward', side_effect=head))
                        def gather(group, local, dim, grad):
                            expected = full[-1][d_indexes]
                            width = (d+dp)//world
                            shard = expected[rank*width:(rank+1)*width]
                            torch.testing.assert_close(local, torch.cat((shard[:, :96], shard[:, :32]), -1))
                            return torch.cat((expected[:, :96], expected[:, :32]), -1)
                        gather_call = stack.enter_context(patch.object(causal_model_sp.Gather, 'apply', side_effect=gather))
                        video, audio = net.dit(**kwargs)
                        self.assertEqual(len(head_inputs[-1][0]), (d+dp)//world)
                        ids = kwargs['inverse_indices'][d_indexes].chunk(world)[rank]
                        torch.testing.assert_close(head_inputs[-1][1], ids, rtol=0, atol=0)
                        torch.testing.assert_close(video, full[-1][hc:hc+d, :96], rtol=0, atol=0)
                        self.assertEqual(audio.shape, (0, 32))
                        gather_call.assert_called_once()


if __name__ == '__main__':
    unittest.main()
