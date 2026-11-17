from contextlib import contextmanager, ExitStack
import unittest
from unittest.mock import patch
import weakref

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from common.distributed import unified_parallel as parallel
from projects.minimax_h3.modeling.transformer import causal_model_sp as causal_sp
from projects.minimax_h3.modeling.transformer import model_sp


@contextmanager
def exchange_transport(sources, rank):
    world = len(sources)
    local_heads = sources[0][0].shape[1] // world
    process_group = object()

    def block(source, destination):
        start = destination * local_heads
        return torch.stack([part[:, start:start + local_heads].flatten(1)
                            for part in sources[source]], dim=1)

    def exchange_lists(outputs, inputs, *, group, async_op):
        assert group is process_group and not async_op
        for destination, part in enumerate(inputs):
            torch.testing.assert_close(part, block(rank, destination), rtol=0, atol=0)
        for source, output in enumerate(outputs):
            output.copy_(block(source, rank))

    def exchange_single(output, send, *, group, async_op):
        assert group is process_group and not async_op
        for destination, part in enumerate(send.chunk(world)):
            torch.testing.assert_close(part, block(rank, destination), rtol=0, atol=0)
        output.copy_(torch.cat([block(source, rank) for source in range(world)]))

    with ExitStack() as stack:
        stack.enter_context(patch.object(model_sp, 'get_unified_parallel_group', return_value=process_group))
        stack.enter_context(patch.object(parallel, 'get_unified_parallel_group', return_value=process_group))
        stack.enter_context(patch.object(torch.distributed, 'get_world_size', return_value=world))
        stack.enter_context(patch.object(torch.distributed, 'all_to_all', new=exchange_lists))
        stack.enter_context(patch.object(torch.distributed, 'all_to_all_single', new=exchange_single))
        yield


class PackedExchangeTests(unittest.TestCase):
    def test_rank_projection_coordinates_and_receive_views_match_legacy(self):
        heads, width = 56, 128
        for world in (1, 2, 4, 8):
            for tokens, dtype, noncontiguous in ((1, torch.float32, False),
                                                 (3, torch.bfloat16, True)):
                labels = torch.arange(world * tokens * 3 * heads * width)
                labels = labels.reshape(world, tokens, 3, heads, width)
                if dtype == torch.bfloat16:
                    labels = (labels * 17 + labels // 251 * 13) % 251 - 125
                values = labels.to(dtype)
                sources = [tuple(part.transpose(1, 2).contiguous().transpose(1, 2)
                                 if noncontiguous else part.contiguous()
                                 for part in source.unbind(1)) for source in values]
                local_heads = heads // world
                for rank in range(world):
                    with self.subTest(world=world, tokens=tokens, dtype=dtype, rank=rank):
                        with exchange_transport(sources, rank), torch.no_grad():
                            legacy = model_sp._ulysses_scatter_heads(*sources[rank], world)
                            packed = model_sp._ulysses_scatter_heads_packed(*sources[rank], world)
                        storage = packed[0].untyped_storage().data_ptr()
                        for projection, (before, after) in enumerate(zip(legacy, packed, strict=True)):
                            expected = values[:, :, projection, rank * local_heads:(rank + 1) * local_heads]
                            expected = expected.reshape(world * tokens, local_heads, width)
                            torch.testing.assert_close(after, expected, rtol=0, atol=0)
                            torch.testing.assert_close(after, before, rtol=0, atol=0)
                            self.assertEqual(after.stride(), before.stride())
                            if world * tokens > 1:
                                self.assertEqual(after.stride(), (3 * local_heads * width, width, 1))
                            self.assertEqual(after.storage_offset(), projection * local_heads * width)
                            self.assertEqual(after.storage_offset(), before.storage_offset())
                            self.assertEqual(after.untyped_storage().data_ptr(), storage)
                            self.assertEqual(after.dtype, dtype)
                            self.assertEqual(after.device, sources[rank][projection].device)

    def test_receive_views_outlive_inputs_without_retaining_send_or_other_calls(self):
        send_refs, receive_refs = [], []

        def exchange(output, send, **_):
            send_refs.append(weakref.ref(send))
            receive_refs.append(weakref.ref(output))
            output.copy_(send)

        with patch.object(model_sp, 'get_unified_parallel_group', return_value=object()), \
                patch.object(torch.distributed, 'all_to_all_single', new=exchange), torch.no_grad():
            inputs = tuple(torch.full((3, 4, 8), float(i)) for i in range(3))
            input_refs = [weakref.ref(part) for part in inputs]
            first = model_sp._ulysses_scatter_heads_packed(*inputs, 2)
            del inputs
            self.assertTrue(all(ref() is None for ref in input_refs + send_refs))
            self.assertIsNotNone(receive_refs[0]())
            second = model_sp._ulysses_scatter_heads_packed(
                *(torch.full((3, 4, 8), float(i + 10)) for i in range(3)), 2)
            self.assertNotEqual(first[0].untyped_storage().data_ptr(), second[0].untyped_storage().data_ptr())
            for projection in range(3):
                self.assertTrue(torch.all(first[projection] == projection))
                self.assertTrue(torch.all(second[projection] == projection + 10))
            del first
            self.assertIsNone(receive_refs[0]())
            self.assertIsNotNone(receive_refs[1]())
            del second
            self.assertTrue(all(ref() is None for ref in send_refs + receive_refs))

    def test_absent_group_preserves_legacy_behavior(self):
        inputs = tuple(torch.randn(2, 4, 8) for _ in range(3))
        with patch.object(model_sp, 'get_unified_parallel_group', return_value=None), \
                patch.object(parallel, 'get_unified_parallel_group', return_value=None):
            for helper in (model_sp._ulysses_scatter_heads, model_sp._ulysses_scatter_heads_packed):
                outputs = helper(*inputs, 1)
                for actual, expected in zip(outputs, inputs, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                with self.assertRaises(ValueError):
                    helper(*inputs, 2)

    def test_invalid_head_divisibility_rejects_before_collective(self):
        inputs = tuple(torch.randn(2, 7, 8) for _ in range(3))
        with patch.object(model_sp, 'get_unified_parallel_group', return_value=object()), \
                patch.object(torch.distributed, 'all_to_all_single') as collective:
            for helper in (model_sp._ulysses_scatter_heads, model_sp._ulysses_scatter_heads_packed):
                with self.assertRaisesRegex(ValueError, 'attention heads 7 must be divisible'):
                    helper(*inputs, 2)
            collective.assert_not_called()

    def test_legacy_collective_backward_preserves_projection_gradients(self):
        def exchange(outputs, inputs, **_):
            for output in outputs:
                output.copy_(inputs[0])

        inputs = tuple(torch.randn(3, 4, 8, requires_grad=True) for _ in range(3))
        with patch.object(parallel, 'get_unified_parallel_group', return_value=object()), \
                patch.object(torch.distributed, 'get_world_size', return_value=2), \
                patch.object(torch.distributed, 'all_to_all', new=exchange):
            outputs = model_sp._ulysses_scatter_heads(*inputs, 2)
            sum((projection + 1) * part.sum() for projection, part in enumerate(outputs)).backward()
        for projection, part in enumerate(inputs):
            torch.testing.assert_close(part.grad, torch.full_like(part, projection + 1), rtol=0, atol=0)


class CausalExchangeEligibilityTests(unittest.TestCase):
    def test_only_eval_no_grad_cuda_bf16_selects_packed_exchange(self):
        mode = FakeTensorMode()

        def tensor(device='cuda:0', dtype=torch.bfloat16):
            return FakeTensor(mode, torch.empty(2, 4, 8, device='meta', dtype=dtype), torch.device(device))

        cases = [(training, grad, None, None, not training and not grad)
                 for training in (False, True) for grad in (False, True)]
        cases += [(False, False, projection, unsupported, False)
                  for projection in range(3) for unsupported in ('cpu', torch.float32, torch.float16)]
        for training, grad, projection, unsupported, packed in cases:
            with self.subTest(training=training, grad=grad, projection=projection, unsupported=unsupported):
                inputs = [tensor() for _ in range(3)]
                if unsupported == 'cpu':
                    inputs[projection] = tensor(device='cpu')
                elif unsupported is not None:
                    inputs[projection] = tensor(dtype=unsupported)
                attention = causal_sp.CausalMiniMaxH3AttentionSP.__new__(causal_sp.CausalMiniMaxH3AttentionSP)
                torch.nn.Module.__init__(attention)
                attention.train(training)
                attention.head_dim = 8
                attention._project_qkv = lambda *_: tuple(inputs)
                attention.out_proj = lambda value: (value, None)
                with ExitStack() as stack, torch.set_grad_enabled(grad):
                    stack.enter_context(patch.object(causal_sp, 'is_unified_parallel_initialized', return_value=True))
                    stack.enter_context(patch.object(causal_sp, 'get_unified_parallel_world_size', return_value=2))
                    stack.enter_context(patch.object(causal_sp, '_ulysses_scatter_heads',
                                                    new=lambda *_: (torch.zeros(4, 2, 8),) * 3))
                    stack.enter_context(patch.object(causal_sp, '_ulysses_scatter_heads_packed',
                                                    new=lambda *_: (torch.ones(4, 2, 8),) * 3))
                    stack.enter_context(patch.object(causal_sp, '_CAUSAL_FLEX_ATTENTION',
                                                    new=lambda q, k, v, *_: q + k + v))
                    stack.enter_context(patch.object(causal_sp, 'gather_heads_scatter_seq',
                                                    new=lambda value, **_: value))
                    result = attention(torch.empty(2, 32), rope_cache=(None, None))
                torch.testing.assert_close(result, torch.full((4, 16), 3.0 if packed else 0.0), rtol=0, atol=0)

    def test_size_one_and_uninitialized_group_use_parent_before_projection(self):
        attention = causal_sp.CausalMiniMaxH3AttentionSP.__new__(causal_sp.CausalMiniMaxH3AttentionSP)
        torch.nn.Module.__init__(attention)
        expected = torch.ones(2, 32)
        for initialized, world in ((False, 2), (True, 1)):
            with self.subTest(initialized=initialized, world=world), \
                    patch.object(causal_sp, 'is_unified_parallel_initialized', return_value=initialized), \
                    patch.object(causal_sp, 'get_unified_parallel_world_size', return_value=world), \
                    patch.object(causal_sp.CausalMiniMaxH3Attention, 'forward', return_value=expected):
                self.assertIs(attention(torch.empty(2, 32), rope_cache=(None, None)), expected)


if __name__ == '__main__':
    unittest.main()
