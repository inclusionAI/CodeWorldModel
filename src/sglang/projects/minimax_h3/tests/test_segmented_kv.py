import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from projects.minimax_h3.modeling.transformer.causal_model import _collect_segment_kv
from projects.minimax_h3.modeling.transformer.causal_model import (
    CachedAttentionSegments, CausalMiniMaxH3DiTModel, segmented_cached_attention)


def check_collection(device):
    old_k = torch.arange(2*3*10, device=device).reshape(2, 3, 10)[:, :, ::2].to(torch.bfloat16)
    old_v = old_k + 100
    new_k = torch.arange(7*3*10, device=device).reshape(7, 3, 10)[:, :, ::2].to(torch.bfloat16)
    new_v = new_k + 200
    order = torch.tensor([0, 3, 6, 1, 4, 2, 5], device=device)
    indexes = torch.tensor([1, 2, 4, 1, 8, 2, 0, 6, 3], device=device)
    old_k, old_v = old_k[:, :, ::2], old_v[:, :, ::2]
    new_k, new_v = new_k[:, :, ::2], new_v[:, :, ::2]
    expected_k = torch.cat((old_k, new_k.index_select(0, order)))[indexes]
    expected_v = torch.cat((old_v, new_v.index_select(0, order)))[indexes]
    with torch.no_grad():
        actual_k, actual_v = _collect_segment_kv(old_k, old_v, new_k, new_v, indexes,
            old_len=2, new_row_order=order)
        torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)
        strided_indexes = torch.stack((indexes, indexes), dim=1).flatten()[::2]
        strided_order = torch.stack((order, order), dim=1).flatten()[::2]
        assert strided_indexes.stride(0) == strided_order.stride(0) == 2
        for selected, row_order in ((strided_indexes, order), (indexes, strided_order)):
            copied_k, copied_v = _collect_segment_kv(
                old_k, old_v, new_k, new_v, selected, old_len=2, new_row_order=row_order)
            torch.testing.assert_close(copied_k, expected_k, rtol=0, atol=0)
            torch.testing.assert_close(copied_v, expected_v, rtol=0, atol=0)
        assert actual_k.is_contiguous() and actual_v.is_contiguous()
        assert actual_k.data_ptr() != old_k.data_ptr() and actual_v.data_ptr() != old_v.data_ptr()
        empty_k, empty_v = _collect_segment_kv(old_k, old_v, new_k, new_v,
            indexes[:0], old_len=2, new_row_order=order)
        assert empty_k.shape == empty_v.shape == (0, 3, 3)
        fresh_k, fresh_v = _collect_segment_kv(None, None, new_k, new_v,
            torch.tensor([6, 0, 6], device=device), old_len=0)
        torch.testing.assert_close(fresh_k, new_k[[6, 0, 6]], rtol=0, atol=0)
        torch.testing.assert_close(fresh_v, new_v[[6, 0, 6]], rtol=0, atol=0)
        bits = torch.tensor([0, -32768, 1, -32767, 16256, -16512],
                            dtype=torch.int16, device=device).view(torch.bfloat16).reshape(6, 1, 1)
        selected = torch.tensor([0, 1, 5, 2, 3, 4, 1], device=device)
        bit_order = torch.tensor([3, 0, 2, 1], device=device)
        bit_expected = torch.cat((bits[:2], bits[2:].index_select(0, bit_order)))[selected]
        bit_k, bit_v = _collect_segment_kv(bits[:2], bits[:2], bits[2:], bits[2:], selected,
                                          old_len=2, new_row_order=bit_order)
        assert torch.equal(bit_k.view(torch.uint8), bit_expected.view(torch.uint8))
        assert torch.equal(bit_v.view(torch.uint8), bit_expected.view(torch.uint8))
        actual_k[0].zero_()
        torch.testing.assert_close(old_k[1], expected_k[0], rtol=0, atol=0)


class SegmentedKVTests(unittest.TestCase):
    def test_cpu_fallback_and_bad_indexes(self):
        check_collection('cpu')
        source = torch.arange(6).reshape(3, 1, 2).float()
        for indexes in (torch.tensor([-4]), torch.tensor([3])):
            with self.assertRaises((IndexError, RuntimeError)):
                _collect_segment_kv(None, None, source, source, indexes, old_len=0)

    def test_gradient_fallback_and_fork_ownership(self):
        source = torch.randn(3, 1, 2, requires_grad=True)
        selected, _ = _collect_segment_kv(None, None, source, source,
            torch.tensor([2, 0, 2]), old_len=0)
        selected.sum().backward()
        torch.testing.assert_close(source.grad[:, 0, 0], torch.tensor([1., 0., 2.]))

        from projects.minimax_h3.modeling.transformer import causal_model as causal
        old = torch.randn(2, 1, 2)
        parent = old.clone()
        cache = SimpleNamespace(seq_lens=lambda _: 2,
            key_cache={0: old}, value_cache={0: old.clone()})
        segments = CachedAttentionSegments(torch.tensor([1, 2, 1]),
            torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1]), torch.tensor([1, 2]))
        with torch.no_grad(), patch.object(causal, 'cached_attention', return_value=source.detach()):
            segmented_cached_attention(source, source.detach(), source.detach(), cache, 0, segments, .5)
        cache.key_cache[0].zero_()
        torch.testing.assert_close(old, parent, rtol=0, atol=0)

    def test_identity_route_checks_indexes_once_before_layers(self):
        model = SimpleNamespace(training=False, _h3_inference={'metadata': False})
        cache = SimpleNamespace(seq_lens=lambda _: 2)
        for indexes in (torch.tensor([-1]), torch.tensor([5])):
            segments = CachedAttentionSegments(indexes, torch.tensor([3]),
                torch.tensor([1]), torch.tensor([0]))
            with torch.no_grad(), self.assertRaisesRegex(ValueError, 'indexes'):
                CausalMiniMaxH3DiTModel._prepare_cache_routing(model,
                    past_key_values=cache, update_past_key_values=True,
                    sample_lens=torch.tensor([1]), query_len=3,
                    device=torch.device('cpu'), cache_segments=segments)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA preflight runs under GPU task isolation')
    def test_cuda_strided_bf16_copy(self):
        check_collection('cuda')


if __name__ == '__main__':
    unittest.main()
