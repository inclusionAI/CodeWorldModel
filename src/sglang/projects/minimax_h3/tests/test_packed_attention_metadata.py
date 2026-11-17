"""DIT-02 correctness and pinned Raven dispatch contracts; no GPU timing."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from utils import flash_attn
from projects.minimax_h3.modeling.inference_optimizations import configure_inference
from projects.minimax_h3.modeling.transformer.packed_attention import PackedAttentionMetadata, cached_attention
from projects.minimax_h3.modeling.transformer.causal_model import CachedAttentionSegments, CausalMiniMaxH3DiTModel
from projects.minimax_h3.tests.bidir_history_support import actual_model
from projects.minimax_h3.tests.test_raw_latent import session, RAW_PROTOCOLS


class PackedMetadataTests(unittest.TestCase):
    def test_sdpa_matches_raven_for_unequal_empty_and_padding_documents(self):
        with patch.object(flash_attn, 'FLASH_ATTN_2_AVAILABLE', False), patch.object(flash_attn, 'FLASH_ATTN_3_AVAILABLE', False):
            for dtype in (torch.float32, torch.bfloat16):
                q_lens = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
                k_lens = torch.tensor([5, 0, 4, 1], dtype=torch.int32)
                q, k, v = (torch.randn(n, 2, 8).to(dtype) for n in (6, 10, 10))
                metadata = PackedAttentionMetadata.prepare(tuple(q_lens.tolist()), (3, 0, 1, 0), query_len=6, device=q.device)
                wrapper = flash_attn.FlashAttention()
                expected = wrapper(q, k, v, q_lens=q_lens, k_lens=k_lens, softmax_scale=.37, causal=False)
                actual = cached_attention(wrapper, q, k, v, q_lens=q_lens, k_lens=k_lens, softmax_scale=.37, metadata=metadata)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                altered = v.clone()
                altered[:5] += 100
                isolated = cached_attention(wrapper, q, k, altered, q_lens=q_lens, k_lens=k_lens, softmax_scale=.37, metadata=metadata)
                torch.testing.assert_close(isolated[2:], actual[2:], rtol=0, atol=0)

    def test_preparation_rejects_invalid_document_boundaries(self):
        for query, past, total in (((), (), 0), ((2,), (0, 1), 2), ((-1, 3), (0, 0), 2), ((2,), (0,), 3), ((2,), (2**31,), 2)):
            with self.subTest(query=query, past=past), self.assertRaises(ValueError):
                PackedAttentionMetadata.prepare(query, past, query_len=total, device='cpu')

    def test_explicit_key_lengths_allow_short_keys_and_reject_bad_totals(self):
        metadata = PackedAttentionMetadata.prepare_lengths(
            (2, 0, 3), (1, 0, 2), query_len=5, key_len=3, device='cpu')
        self.assertEqual(metadata.q_bounds, (0, 2, 2, 5))
        self.assertEqual(metadata.k_bounds, (0, 1, 1, 3))
        with patch.object(flash_attn, 'FLASH_ATTN_2_AVAILABLE', False), \
                patch.object(flash_attn, 'FLASH_ATTN_3_AVAILABLE', False):
            q, k, v = (torch.randn(n, 2, 8) for n in (5, 3, 3))
            wrapper = flash_attn.FlashAttention()
            expected = wrapper(q, k, v, q_lens=torch.tensor([2, 0, 3]),
                               k_lens=torch.tensor([1, 0, 2]), softmax_scale=.37, causal=False)
            actual = cached_attention(wrapper, q, k, v, q_lens=torch.tensor([2, 0, 3]),
                                      k_lens=torch.tensor([1, 0, 2]), softmax_scale=.37,
                                      metadata=metadata)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for query, key, q_total, k_total in (
            ((), (), 0, 0), ((1,), (1, 1), 1, 2),
            ((-1, 2), (1, 1), 1, 2), ((1,), (-1,), 1, -1),
            ((2,), (1,), 1, 1), ((2,), (1,), 2, 2),
            ((2**31,), (1,), 2**31, 1), ((1,), (2**31,), 1, 2**31),
        ):
            with self.subTest(query=query, key=key), self.assertRaises(ValueError):
                PackedAttentionMetadata.prepare_lengths(
                    query, key, query_len=q_total, key_len=k_total, device='cpu')

    def test_fa2_fa3_dispatch_matches_pinned_wrapper(self):
        for fa3 in (False, True):
            for dtypes in ((torch.float32,) * 3, (torch.float16, torch.bfloat16, torch.float16)):
                with self.subTest(fa3=fa3, dtypes=dtypes), ExitStack() as stack:
                    stack.enter_context(patch.object(flash_attn, 'FLASH_ATTN_2_AVAILABLE', True))
                    stack.enter_context(patch.object(flash_attn, 'FLASH_ATTN_3_AVAILABLE', fa3))
                    calls = []
                    def kernel(**kwargs):
                        calls.append(kwargs)
                        return kwargs['q'].clone()
                    stack.enter_context(patch.object(flash_attn, 'flash_attn_varlen_func', kernel, create=True))
                    stack.enter_context(patch.object(flash_attn, 'flash_attn_varlen_func_hopper', kernel, create=True))
                    q_lens = torch.tensor([2, 1], dtype=torch.int32)
                    k_lens = torch.tensor([5, 1], dtype=torch.int32)
                    q, k, v = (torch.randn(n, 2, 8).to(dtype) for n, dtype in zip((3, 6, 6), dtypes))
                    metadata = PackedAttentionMetadata.prepare((2, 1), (3, 0), query_len=3, device='cpu')
                    wrapper = flash_attn.FlashAttention()
                    # The spy checks dispatch contracts on CPU; it does not execute a CUDA kernel.
                    stack.enter_context(patch.object(torch.Tensor, 'device', property(lambda _: torch.device('cuda'))))
                    # Raven moves its computed bounds to q.device; keep this spy-only copy on CPU.
                    original_to = torch.Tensor.to
                    def cpu_to(tensor, *args, **kwargs):
                        if args and isinstance(args[0], torch.device):
                            args = (torch.device('cpu'), *args[1:])
                        return original_to(tensor, *args, **kwargs)
                    stack.enter_context(patch.object(torch.Tensor, 'to', cpu_to))
                    expected = wrapper(q, k, v, q_lens=q_lens, k_lens=k_lens, softmax_scale=.37, causal=False)
                    actual = cached_attention(wrapper, q, k, v, q_lens=q_lens, k_lens=k_lens, softmax_scale=.37, metadata=metadata)
                    self.assertEqual(set(calls[0]), set(calls[1]))
                    for key in calls[0]:
                        if torch.is_tensor(calls[0][key]):
                            torch.testing.assert_close(calls[0][key], calls[1][key], rtol=0, atol=0)
                        else:
                            self.assertEqual(calls[0][key], calls[1][key])
                    self.assertIs(calls[1]['cu_seqlens_q'], metadata.cu_seqlens_q)
                    self.assertIs(calls[1]['cu_seqlens_k'], metadata.cu_seqlens_k)
                    self.assertEqual(actual.dtype, dtypes[0])
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_segmented_backend_receives_original_gather_and_commits_same_kv(self):
        from projects.minimax_h3.modeling.transformer import causal_model as causal

        q, k, v = (torch.randn(3, 2, 8) for _ in range(3))
        old_k, old_v = (torch.randn(2, 2, 8) for _ in range(2))
        segments = CachedAttentionSegments(torch.tensor([0, 1, 2, 4]),
                                           torch.tensor([2, 1]), torch.tensor([3, 1]),
                                           torch.tensor([0, 2, 3]))
        metadata = PackedAttentionMetadata.prepare_lengths((2, 1), (3, 1),
                                                             query_len=3, key_len=4, device='cpu')
        cache = SimpleNamespace(seq_lens=lambda _: 2, key_cache={0: old_k},
                                value_cache={0: old_v})
        with patch.object(causal, 'cached_attention', return_value=q.clone()) as backend:
            out = causal.segmented_cached_attention(q, k, v, cache, 0, segments, .37,
                                                    metadata=metadata)
        torch.testing.assert_close(out, q, rtol=0, atol=0)
        self.assertIs(backend.call_args.args[1], q)
        torch.testing.assert_close(backend.call_args.args[2],
                                   torch.cat((old_k, k))[segments.key_indexes], rtol=0, atol=0)
        torch.testing.assert_close(backend.call_args.args[3],
                                   torch.cat((old_v, v))[segments.key_indexes], rtol=0, atol=0)
        self.assertIs(backend.call_args.kwargs['metadata'], metadata)
        self.assertIs(backend.call_args.kwargs['q_lens'], segments.query_lens)
        self.assertIs(backend.call_args.kwargs['k_lens'], segments.key_lens)
        self.assertEqual(backend.call_args.kwargs['softmax_scale'], .37)
        torch.testing.assert_close(cache.key_cache[0],
                                   torch.cat((old_k, k))[segments.commit_indexes], rtol=0, atol=0)
        torch.testing.assert_close(cache.value_cache[0],
                                   torch.cat((old_v, v))[segments.commit_indexes], rtol=0, atol=0)

    def test_route_cache_hit_still_gets_fresh_pre_update_metadata(self):
        model = SimpleNamespace(training=False, gradient_checkpointing=False,
                                _h3_inference={'metadata': True}, _h3_inference_generation=object())
        cache = SimpleNamespace(batch_size=2, kvlens=[4, 0], seq_len=4)
        def update(lengths):
            cache.kvlens = [a+b for a,b in zip(cache.kvlens, lengths)]
        cache.update_kvlens = update
        kwargs = dict(past_key_values=cache, update_past_key_values=False,
                      sample_lens=torch.tensor([2, 1], dtype=torch.int32), query_len=3, device=torch.device('cpu'))
        with torch.no_grad():
            first = CausalMiniMaxH3DiTModel._prepare_cache_routing(model, **kwargs)
            second = CausalMiniMaxH3DiTModel._prepare_cache_routing(model, **kwargs)
            self.assertIs(first['packed_query_indexes'], second['packed_query_indexes'])
            self.assertIsNot(first['attention_metadata'], second['attention_metadata'])
            self.assertNotIn('attention_metadata', cache._h3_routing[1])
            third = CausalMiniMaxH3DiTModel._prepare_cache_routing(model, **dict(kwargs, update_past_key_values=True))
        self.assertEqual(third['attention_metadata'].k_bounds, (0, 6, 7))
        self.assertEqual(cache.kvlens, [6, 1])
        for training, grad in ((True, False), (False, True)):
            model.training = training
            with torch.set_grad_enabled(grad):
                result = CausalMiniMaxH3DiTModel._prepare_cache_routing(model, **kwargs)
            self.assertIsNone(result['attention_metadata'])

    @torch.no_grad()
    def test_real_two_layer_fixed_bidir_outputs_and_forward_lifetime(self):
        with actual_model(fp32=True) as net:
            for protocol in RAW_PROTOCOLS[:2]:
                def run(enabled):
                    configure_inference(net, metadata=enabled, kv=False, adaln=False)
                    forwards = []
                    with ExitStack() as hooks:
                        hooks.callback(net.register_forward_pre_hook(lambda *_: forwards.append([])).remove)
                        for block in net.dit.blocks:
                            def record(module, args, kwargs):
                                forwards[-1].append(kwargs.get('attention_metadata'))
                            hooks.callback(block.attn.register_forward_pre_hook(record, with_kwargs=True).remove)
                        stream, chunks = session(net, protocol, current_latent_schedule=(5,))
                        stream.model._sp_size = lambda: 4
                        stream.model._pad_rows = lambda n: 4-n%4
                        outputs = [o for chunk in chunks for o in stream.push(chunk)]
                    return outputs, forwards
                expected, disabled = run(False)
                actual, forwards = run(True)
                self.assertTrue(all(m is None for row in disabled for m in row))
                prepared = [row for row in forwards if row[0] is not None]
                self.assertTrue(prepared)
                self.assertTrue(all(len(row) == 2 and row[0] is row[1] for row in prepared))
                self.assertEqual(len({id(row[0]) for row in prepared}), len(prepared))
                self.assertTrue(any(len(row[0].q_bounds) > 2 for row in prepared))
                for before, after in zip(expected, actual, strict=True):
                    self.assertEqual(before.plan, after.plan)
                    self.assertEqual(before.sources, after.sources)
                    torch.testing.assert_close(before.video, after.video, rtol=0, atol=1e-5)

    @torch.no_grad()
    def test_sp_passes_global_metadata_after_exchange(self):
        from projects.minimax_h3.modeling.transformer import causal_model as causal, causal_model_sp as sp
        with actual_model() as net, ExitStack() as stack:
            attention = net.dit.blocks[0].attn
            attention.__class__ = sp.CausalMiniMaxH3AttentionSP
            stack.enter_context(patch.object(sp, 'is_unified_parallel_initialized', return_value=True))
            stack.enter_context(patch.object(sp, 'get_unified_parallel_world_size', return_value=2))
            stack.enter_context(patch.object(causal, '_apply_rope_qk', side_effect=lambda q,k,*_,**__: (q,k)))
            def exchange(q,k,v,size):
                return tuple(x.repeat(2,1,1)[:, :x.shape[1]//2] for x in (q,k,v))
            stack.enter_context(patch.object(sp, '_ulysses_scatter_heads', side_effect=exchange))
            stack.enter_context(patch.object(sp, 'gather_heads_scatter_seq', side_effect=lambda out, **_: out[:2].repeat(1,2)))
            adapter = stack.enter_context(patch.object(sp, 'cached_attention', side_effect=lambda wrapper,q,k,v,**_: torch.zeros_like(q)))
            metadata = PackedAttentionMetadata.prepare((3,1), (0,0), query_len=4, device='cpu')
            cache = SimpleNamespace(seq_lens=lambda _: 0)
            result = attention(torch.randn(2, 128).bfloat16(), rope_cache=(None,None),
                               past_key_values=cache, sample_lens=torch.tensor([3,1]),
                               key_value_lens=torch.tensor([3,1]), attention_metadata=metadata)
            self.assertEqual(result.shape, (2,128))
            self.assertEqual(adapter.call_args.args[1].shape[0], 4)
            self.assertIs(adapter.call_args.kwargs['metadata'], metadata)

    @torch.no_grad()
    def test_sp_passes_segmented_metadata_after_exchange(self):
        from projects.minimax_h3.modeling.transformer import causal_model as causal, causal_model_sp as sp
        from projects.minimax_h3.modeling.transformer.hc_layout import HCLayout

        with actual_model() as net, ExitStack() as stack:
            attention = net.dit.blocks[0].attn
            attention.__class__ = sp.CausalMiniMaxH3AttentionSP
            stack.enter_context(patch.object(sp, 'is_unified_parallel_initialized', return_value=True))
            stack.enter_context(patch.object(sp, 'get_unified_parallel_world_size', return_value=2))
            stack.enter_context(patch.object(causal, '_apply_rope_qk', side_effect=lambda q,k,*_,**__: (q,k)))
            exchanged = []
            def exchange(q, k, v, _):
                result = tuple(x.repeat(2,1,1)[:, :x.shape[1]//2] for x in (q,k,v))
                exchanged.append(result)
                return result
            stack.enter_context(patch.object(sp, '_ulysses_scatter_heads', side_effect=exchange))
            stack.enter_context(patch.object(sp, 'gather_heads_scatter_seq',
                                       side_effect=lambda out, **_: out[:2].repeat(1,2)))
            adapter = stack.enter_context(patch.object(sp, 'segmented_cached_attention',
                                                side_effect=lambda q,k,v,*_,**__: torch.zeros_like(q)))
            segments = CachedAttentionSegments(torch.arange(4), torch.tensor([1, 1, 1, 1]),
                                               torch.tensor([1, 1, 1, 1]), torch.arange(4), hc_rows=(1, 1))
            layout = HCLayout.prepare(segments, world_size=2, rank=0, device='cpu')
            metadata = PackedAttentionMetadata.prepare_lengths((1, 1, 1, 1), (1, 1, 1, 1),
                                                                 query_len=4, key_len=4, device='cpu')
            cache = SimpleNamespace(seq_lens=lambda _: 0)
            result = attention(torch.randn(2, 128).bfloat16(),
                               rope_cache=(torch.empty(2, 96), torch.arange(1)),
                               past_key_values=cache, cache_segments=segments,
                               attention_metadata=metadata, hc_layout=layout)
            self.assertEqual(result.shape, (2,128))
            self.assertEqual(adapter.call_args.args[0].shape[0], 4)
            self.assertIs(adapter.call_args.args[5], segments)
            self.assertIs(adapter.call_args.kwargs['metadata'], metadata)
            torch.testing.assert_close(adapter.call_args.args[0],
                                       exchanged[0][0].index_select(0, layout.to_attention_order))
            self.assertIs(adapter.call_args.args[1], exchanged[0][1])
            self.assertIs(adapter.call_args.args[2], exchanged[0][2])
            self.assertIs(adapter.call_args.kwargs['new_row_order'], layout.to_attention_order)


if __name__ == '__main__':
    unittest.main()
