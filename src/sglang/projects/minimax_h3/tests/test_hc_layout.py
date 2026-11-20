"""H/C projections keep standalone operands around a shared attention call."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from projects.minimax_h3.inference.raw_latent import LatentKVCache
from projects.minimax_h3.inference.raw_recipe import RAW_PROTOCOLS
from projects.minimax_h3.modeling.inference_optimizations import configure_inference
from projects.minimax_h3.modeling.transformer import causal_model_sp
from projects.minimax_h3.modeling.transformer.causal_model import CachedAttentionSegments
from projects.minimax_h3.modeling.transformer.hc_layout import HCLayout
from projects.minimax_h3.modeling.transformer.model import _rope_cos_sin_cache
from projects.minimax_h3.tests.bidir_history_support import actual_model
from projects.minimax_h3.tests.test_raw_latent import session


def route(keys, queries, key_lens, commit, hc_rows=None, denoise_rows=0):
    return CachedAttentionSegments(torch.tensor(keys, dtype=torch.long),
        torch.tensor(queries, dtype=torch.int32), torch.tensor(key_lens, dtype=torch.int32),
        torch.tensor(commit, dtype=torch.long), hc_rows=hc_rows, denoise_rows=denoise_rows)


class HCLayoutTests(unittest.TestCase):
    def test_three_role_shards_and_d_only_gather_keep_standalone_order(self):
        for h, c, d in ((5040, 560, 5040), (1008, 448, 4032), (7, 3, 1), (16, 8, 8), (1, 1, 1)):
            for world in (1, 2, 8):
                with self.subTest(h=h, c=c, d=d, world=world):
                    roles = (h, c, d)
                    padding = [world-n % world for n in roles] if world > 1 else [0]*3
                    segments = SimpleNamespace(hc_rows=(h, c), denoise_rows=d,
                        query_lens=torch.tensor(list(roles)+(padding if world > 1 else [])))
                    layouts = [HCLayout.prepare(segments, world_size=world, rank=r, device='cpu')
                               for r in range(world)]
                    values = torch.arange((sum(roles)+sum(padding))*8*3).reshape(-1, 8, 3)
                    role_values = []
                    live, pad = 0, sum(roles)
                    for rows, extra in zip(roles, padding):
                        role_values.append(torch.cat((values[live:live+rows], values[pad:pad+extra])))
                        live += rows
                        pad += extra
                    physical, denoise = [], []
                    for rank, layout in enumerate(layouts):
                        parts = values[layout.input_indexes].split(layout.local_rows)
                        for expected, part, width in zip(role_values, parts, layout.local_rows):
                            torch.testing.assert_close(part, expected[rank*width:(rank+1)*width], rtol=0, atol=0)
                        physical.extend(parts)
                        denoise.append(parts[-1])
                    exchanged = torch.cat(physical)
                    torch.testing.assert_close(exchanged[layouts[0].to_attention_order], values, rtol=0, atol=0)
                    torch.testing.assert_close(values[layouts[0].from_attention_order], exchanged, rtol=0, atol=0)
                    torch.testing.assert_close(torch.cat(denoise), role_values[-1], rtol=0, atol=0)

    def test_standalone_shards_round_trip_preserves_rows_heads_and_metadata(self):
        for h, c in ((5040, 560), (1008, 448), (7, 3), (16, 8), (1, 1)):
            for world in (1, 2, 8):
                with self.subTest(h=h, c=c, world=world):
                    hp, cp = (world-h % world, world-c % world) if world > 1 else (0, 0)
                    queries = [h, c] + ([hp, cp] if world > 1 else [])
                    segments = SimpleNamespace(hc_rows=(h, c), query_lens=torch.tensor(queries))
                    layouts = [HCLayout.prepare(segments, world_size=world, rank=r, device='cpu')
                               for r in range(world)]
                    total = h+c+hp+cp
                    values = torch.arange(total*8*3).reshape(total, 8, 3)
                    metadata = torch.arange(total*5).reshape(total, 5)
                    history = torch.cat((values[:h], values[h+c:h+c+hp]))
                    condition = torch.cat((values[h:h+c], values[h+c+hp:]))
                    physical = []
                    for rank, layout in enumerate(layouts):
                        rows = values[layout.input_indexes]
                        mh, mc = layout.local_rows
                        torch.testing.assert_close(rows[:mh], history[rank*mh:(rank+1)*mh])
                        torch.testing.assert_close(rows[mh:], condition[rank*mc:(rank+1)*mc])
                        physical.append(rows)
                    exchanged = torch.cat(physical)
                    canonical = exchanged[layouts[0].to_attention_order]
                    torch.testing.assert_close(canonical, values, rtol=0, atol=0)
                    restored = canonical[layouts[0].from_attention_order]
                    torch.testing.assert_close(restored, exchanged, rtol=0, atol=0)
                    actual_metadata = torch.cat([metadata[x.input_indexes] for x in layouts])
                    torch.testing.assert_close(actual_metadata[layouts[0].to_attention_order], metadata)
                    for head_rank in range(world):
                        head_slice = slice(head_rank*8//world, (head_rank+1)*8//world)
                        torch.testing.assert_close(canonical[:, head_slice], values[:, head_slice])

    def test_divisible_lengths_require_full_standalone_padding(self):
        segments = SimpleNamespace(hc_rows=(16, 8), query_lens=torch.tensor([16, 8, 8, 8]))
        layout = HCLayout.prepare(segments, world_size=8, rank=7, device='cpu')
        self.assertEqual(layout.local_rows, (3, 2))
        for lengths in ([16, 8], [16, 8, 4, 4], [16, 8, 8], [16, 8, -8, 8]):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                HCLayout.prepare(SimpleNamespace(hc_rows=(16, 8), query_lens=torch.tensor(lengths)),
                                 world_size=8, rank=7, device='cpu')

    @torch.no_grad()
    def test_sp_entry_rebuilds_prepared_rope_and_aligns_embedding_and_adaln(self):
        with actual_model(fp32=True) as net:
            configure_inference(net, delayed_hc=True)
            stream, chunks = session(net, RAW_PROTOCOLS[0], recent_latents=1)
            stream.model._sp_size = lambda: 2
            stream.model._pad_rows = lambda n: 2-n % 2
            captured = []

            def capture(module, args, kwargs):
                if kwargs.get('cache_segments') is not None:
                    captured.append(dict(kwargs))
                    raise RuntimeError('captured pair')

            with net.dit.register_forward_pre_hook(capture, with_kwargs=True):
                with self.assertRaisesRegex(RuntimeError, 'captured pair'):
                    for chunk in chunks:
                        stream.push(chunk)
            kwargs = captured[0]
            segments = kwargs['cache_segments']
            with self.assertRaisesRegex(ValueError, 'cache-only'):
                net.dit(**dict(kwargs, cache_only=False))
            kwargs.pop('local_embedding_layout', None)
            keys = tuple(net.state_dict())
            net.dit.__class__ = causal_model_sp.CausalMiniMaxH3DiTModelSP
            embed = net.dit._embed
            full_embeddings = []

            def full_embed(**values):
                result = embed(**values)
                full_embeddings.append(result[0])
                return result

            for rank in range(2):
                with self.subTest(rank=rank), ExitStack() as stack:
                    for name, value in (('is_unified_parallel_initialized', True),
                                        ('get_unified_parallel_world_size', 2),
                                        ('get_unified_parallel_rank', rank)):
                        stack.enter_context(patch.object(causal_model_sp, name, return_value=value))
                    stack.enter_context(patch.object(causal_model_sp, '_BF16_DTYPE', torch.float32))
                    stack.enter_context(patch.object(net.dit, '_embed', new=full_embed))
                    blocks = stack.enter_context(patch.object(net.dit, '_run_blocks', side_effect=lambda x, **kw: x))
                    head = stack.enter_context(patch.object(net.dit.final_layer, 'forward', side_effect=AssertionError('head')))
                    gather = stack.enter_context(patch.object(causal_model_sp.Gather, 'apply', side_effect=AssertionError('gather')))
                    poisoned_rope = (torch.full_like(kwargs['rope_cache'][0], float('nan')), kwargs['rope_cache'][1])
                    self.assertIsNone(net.dit(**dict(kwargs, rope_cache=poisoned_rope)))
                    layout = HCLayout.prepare(segments, world_size=2, rank=rank, device='cpu')
                    hidden = blocks.call_args.args[0]
                    block_args = blocks.call_args.kwargs
                    torch.testing.assert_close(hidden, full_embeddings[-1][layout.input_indexes], rtol=0, atol=0)
                    expected_ids = kwargs['token_tags'].clamp(min=0) + kwargs['inverse_indices']*3
                    torch.testing.assert_close(block_args['combined_indices'], expected_ids[layout.input_indexes])
                    freqs = net.dit.rope(kwargs['img_position_ids'][:, layout.input_indexes])
                    expected_rope = _rope_cos_sin_cache(freqs, dtype=torch.float32)
                    torch.testing.assert_close(block_args['rope_cache'][0], expected_rope, rtol=0, atol=0)
                    self.assertEqual(block_args['hc_layout'].local_rows, layout.local_rows)
                    head.assert_not_called()
                    gather.assert_not_called()
            self.assertEqual(tuple(net.state_dict()), keys)

    @torch.no_grad()
    def test_two_blocks_match_separate_roles_and_partial_h_retain(self):
        with actual_model(fp32=True) as net:
            generator = torch.Generator().manual_seed(92)
            h, c, hp, cp, old = 3, 2, 1, 2, 4
            x = torch.randn(h+c+hp+cp, 128, generator=generator)
            ids = torch.tensor([0, 1, 2, 2, 1, 0, 0, 0])
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
            hc = route(list(range(old+h))+commit+list(range(old+h+c, old+len(x))),
                       [h, c, hp, cp], [old+h, len(commit), hp, cp], commit, (h, c))
            layout = HCLayout.prepare(hc, world_size=1, rank=0, device='cpu')
            history_indexes = torch.tensor([0, 1, 2, 5])
            condition_indexes = torch.tensor([3, 4, 6, 7])

            def run_part(value, indexes, cache, segments, hc_layout=None):
                rope = (cos_sin[indexes], torch.arange(len(indexes)))
                for layer, block in enumerate(net.dit.blocks):
                    value = block(value, adaln_input=torch.empty(0), adaln_params=params[layer],
                        combined_indices=ids[indexes], rope_cache=rope, past_key_values=cache,
                        update_past_key_values=True, cache_segments=segments, hc_layout=hc_layout)
                return value

            sequential = initial.fork()
            h_route = route(range(old+h+hp), [h, hp], [old+h, hp], range(old+h))
            expected_h = run_part(x[history_indexes], history_indexes, sequential, h_route)
            for layer in sequential.key_cache:
                sequential.key_cache[layer] = sequential.key_cache[layer][retained]
                sequential.value_cache[layer] = sequential.value_cache[layer][retained]
            prefix = len(retained)
            c_route = route(range(prefix+c+cp), [c, cp], [prefix+c, cp], range(prefix+c))
            expected_c = run_part(x[condition_indexes], condition_indexes, sequential, c_route)
            paired = initial.fork()
            calls = {name: [] for name in ('qkv_proj', 'out_proj', 'fc1', 'fc2')}
            with ExitStack() as stack:
                for block in net.dit.blocks:
                    for name in calls:
                        module = getattr(block.attn if name.endswith('_proj') else block.mlp, name)
                        def record(module, args, name=name):
                            calls[name].append(len(args[0]))
                        stack.enter_context(module.register_forward_pre_hook(record))
                actual = run_part(x[layout.input_indexes], layout.input_indexes, paired, hc, layout)
            expected = torch.cat((expected_h, expected_c))
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            for shapes in calls.values():
                self.assertEqual(shapes, [h+hp, c+cp]*len(net.dit.blocks))
            for layer in paired.key_cache:
                torch.testing.assert_close(paired.key_cache[layer], sequential.key_cache[layer], rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(paired.value_cache[layer], sequential.value_cache[layer], rtol=1e-6, atol=1e-6)
                self.assertEqual(len(paired.key_cache[layer]), len(commit))


if __name__ == '__main__':
    unittest.main()
