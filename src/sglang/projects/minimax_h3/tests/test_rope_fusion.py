import builtins
from contextlib import ExitStack
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from projects.minimax_h3.modeling.inference_optimizations import configure_inference, inference_enabled
from projects.minimax_h3.modeling.transformer import model


def assert_bytes(test, actual, expected):
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    torch.testing.assert_close(actual.contiguous().view(torch.uint8),
                               expected.contiguous().view(torch.uint8), rtol=0, atol=0)


class RopeDispatchTests(unittest.TestCase):
    def inputs(self, *, rows=3, heads=2, width=128, rotary=96, dtype=torch.bfloat16):
        q, k = (torch.randn(rows, heads, width).to(dtype) for _ in range(2))
        cache = torch.randn(rows, rotary).to(dtype)
        return q, k, cache, torch.arange(rows).flip(0) + 1000

    def test_model_local_boolean_option_and_training_policy(self):
        first, second = (torch.nn.Sequential(torch.nn.Linear(2, 2)).eval() for _ in range(2))
        keys = tuple(first.state_dict())
        configure_inference(first)
        configure_inference(second)
        for bad in (None, 1, 'true'):
            with self.assertRaisesRegex(ValueError, 'rope must be a boolean'):
                configure_inference(first, rope=bad)
        configure_inference(first, rope=True)
        with torch.no_grad():
            self.assertTrue(inference_enabled(first[0], 'rope'))
            self.assertFalse(inference_enabled(second[0], 'rope'))
            first.train()
            self.assertFalse(inference_enabled(first[0], 'rope'))
        first.eval()
        self.assertFalse(inference_enabled(first[0], 'rope'))
        configure_inference(first, rope=False)
        with torch.no_grad():
            self.assertFalse(inference_enabled(first[0], 'rope'))
        self.assertEqual(tuple(first.state_dict()), keys)

    def test_cpu_and_default_off_never_import_optional_kernel(self):
        values = self.inputs()
        expected = model._apply_rope_qk(*values)
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name in ('rope', 'triton'):
                self.fail('CPU or default-off path imported the optional kernel')
            return original(name, *args, **kwargs)
        with torch.no_grad(), patch('builtins.__import__', side_effect=guarded):
            for enabled in (False, True):
                for actual, before in zip(model._apply_rope_qk(*values, fused=enabled), expected):
                    assert_bytes(self, actual, before)
            with patch.object(torch.Tensor, 'is_cuda', property(lambda _: True)):
                model._apply_rope_qk(*values)

    def test_unsupported_contracts_and_compilation_use_eager(self):
        supported = self.inputs()
        q, k, cache, positions = supported
        variants = [self.inputs(dtype=torch.float32), self.inputs(dtype=torch.float16), self.inputs(rows=0),
                    self.inputs(heads=0), self.inputs(width=64, rotary=48),
                    self.inputs(rotary=64),
                    (q.transpose(0, 1), k.transpose(0, 1), cache[:2], positions[:2]),
                    (q, k.transpose(0, 1).contiguous().transpose(0, 1), cache, positions),
                    (q, k, cache.t().contiguous().t(), positions),
                    (q, k, cache.float(), positions),
                    (q.detach().requires_grad_(), k, cache, positions)]
        spy = Mock(side_effect=AssertionError('unsupported inputs selected kernel'))
        name = model.__package__ + '.rope'
        with torch.no_grad(), patch.dict(sys.modules, {name: SimpleNamespace(apply_rope_qk=spy)}), \
                patch.object(torch.Tensor, 'is_cuda', property(lambda _: True)):
            for values in variants:
                expected = model._apply_rope_qk(*values)
                for actual, before in zip(model._apply_rope_qk(*values, fused=True), expected):
                    assert_bytes(self, actual, before)
            with patch.object(torch.compiler, 'is_compiling', return_value=True):
                model._apply_rope_qk(*self.inputs(), fused=True)
        spy.assert_not_called()

    def test_enabled_gradients_preserve_forward_and_backward(self):
        inputs = self.inputs()
        results = []
        for fused in (False, True):
            q, k, cache = (x.detach().clone().requires_grad_() for x in inputs[:3])
            with patch.object(torch.Tensor, 'is_cuda', property(lambda _: True)):
                out = model._apply_rope_qk(q, k, cache, inputs[3], fused=fused)
            sum(x.square().sum() for x in out).backward()
            results.append((*out, q.grad, k.grad, cache.grad))
        for actual, expected in zip(*results):
            assert_bytes(self, actual, expected)

    def test_selected_kernel_and_import_errors_are_not_hidden(self):
        values = self.inputs()
        name = model.__package__ + '.rope'
        result = (object(), object())
        spy = Mock(return_value=result)
        with torch.no_grad(), patch.object(torch.Tensor, 'is_cuda', property(lambda _: True)):
            with patch.dict(sys.modules, {name: SimpleNamespace(apply_rope_qk=spy)}):
                self.assertIs(model._apply_rope_qk(*values, fused=True), result)
                spy.assert_called_once_with(*values[:3])
                spy.side_effect = RuntimeError('kernel failed')
                with self.assertRaisesRegex(RuntimeError, 'kernel failed'):
                    model._apply_rope_qk(*values, fused=True)
            original = builtins.__import__
            for missing in ('triton', 'triton.internal_broken', 'another_dependency'):
                def import_error(name, *args, **kwargs):
                    if name == 'rope':
                        raise ModuleNotFoundError(name=missing)
                    return original(name, *args, **kwargs)
                with patch('builtins.__import__', side_effect=import_error):
                    if missing == 'triton':
                        expected = model._apply_rope_qk(*values)
                        for actual, before in zip(model._apply_rope_qk(*values, fused=True), expected):
                            assert_bytes(self, actual, before)
                    else:
                        with self.assertRaises(ModuleNotFoundError):
                            model._apply_rope_qk(*values, fused=True)

    def test_all_four_attention_forwards_pass_model_policy_after_native_norm(self):
        from projects.minimax_h3.modeling.transformer import causal_model, model_sp, causal_model_sp
        from projects.minimax_h3.tests.bidir_history_support import actual_model
        class ReachedRope(Exception):
            pass
        variants = ((model, model.MiniMaxH3Attention, False),
                    (causal_model, causal_model.CausalMiniMaxH3Attention, True),
                    (model_sp, model_sp.MiniMaxH3AttentionSP, False),
                    (causal_model_sp, causal_model_sp.CausalMiniMaxH3AttentionSP, True))
        with actual_model() as net:
            attention = net.dit.blocks[0].attn
            x = torch.randn(2, 128).bfloat16()
            for owner, cls, causal in variants:
                attention.__class__ = cls
                kwargs = {} if causal else dict(cu_seqlens=torch.tensor([0, 2]), max_seqlen=2)
                with ExitStack() as stack:
                    if owner in (model_sp, causal_model_sp):
                        stack.enter_context(patch.object(owner, 'is_unified_parallel_initialized', return_value=True))
                        stack.enter_context(patch.object(owner, 'get_unified_parallel_world_size', return_value=2))
                    normalized = []
                    native_norm = owner._apply_qk_norm
                    def record_norm(*args):
                        result = native_norm(*args)
                        normalized.append(result)
                        return result
                    def reached_rope(q, k, *args, **kwargs):
                        self.assertIs(q, normalized[-1][0])
                        self.assertIs(k, normalized[-1][1])
                        raise ReachedRope
                    norm = stack.enter_context(patch.object(owner, '_apply_qk_norm', side_effect=record_norm))
                    rope = stack.enter_context(patch.object(owner, '_apply_rope_qk', side_effect=reached_rope))
                    for enabled, training, grad in ((False, False, False), (True, False, False),
                                                    (True, True, False), (True, False, True)):
                        configure_inference(attention, rope=enabled)
                        attention.train(training)
                        with self.subTest(cls=cls.__name__, enabled=enabled, training=training, grad=grad), \
                                torch.set_grad_enabled(grad), self.assertRaises(ReachedRope):
                            attention(x, rope_cache=(None, None), **kwargs)
                        self.assertEqual(rope.call_args.kwargs['fused'], enabled and not training and not grad)
                    self.assertEqual(norm.call_count, 4)


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA and installed Triton')
class RopeCudaTests(unittest.TestCase):
    @torch.no_grad()
    def test_exact_bytes_signed_zero_rotation(self):
        from projects.minimax_h3.modeling.transformer import rope
        q = torch.zeros((1, 4, 128), device='cuda', dtype=torch.bfloat16)
        q[0, :2, 0] = -1
        q[0, 2:, 0] = 1
        q[0, 1::2, 48] = -0.0
        k = q.flip(1)
        cache = torch.zeros((1, 96), device='cuda', dtype=torch.bfloat16)
        cache[0, 48] = 1
        positions = torch.zeros(1, device='cuda', dtype=torch.long)
        expected = model._apply_rope_qk(q, k, cache, positions)
        self.assertEqual(expected[0][0, :, 0].view(torch.int16).tolist(), [-32768, 0, 0, 0])
        self.assertEqual(expected[1][0, :, 0].view(torch.int16).tolist(), [0, 0, 0, -32768])
        with patch.object(rope, 'apply_rope_qk', wraps=rope.apply_rope_qk) as launch:
            actual = model._apply_rope_qk(q, k, cache, positions, fused=True)
        launch.assert_called_once()
        for output, reference in zip(actual, expected):
            assert_bytes(self, output, reference)

    @torch.no_grad()
    def test_exact_bytes_observed_shapes_and_rounding_edges(self):
        from projects.minimax_h3.modeling.transformer import rope
        generator = torch.Generator(device='cuda').manual_seed(260621)
        for rows, heads in ((71, 56), (631, 56), (1, 1), (3, 3), (17, 2)):
            for adversarial in (False, True):
                with self.subTest(rows=rows, heads=heads, adversarial=adversarial):
                    shape = (rows, heads, 128)
                    q, k = (torch.randn((rows + 1, heads, 128), device='cuda', generator=generator,
                                        dtype=torch.bfloat16)[1:] for _ in range(2))
                    frequencies = torch.randn(rows, 96, device='cuda', generator=generator) * 100
                    cache = model._rope_cos_sin_cache(frequencies, dtype=torch.bfloat16)
                    cache = torch.cat((cache[:1], cache))[1:]
                    if adversarial:
                        bits = torch.arange(0x4181, dtype=torch.int32)
                        edges = torch.tensor([0, -32768, 1, -32767, 127, -32641, 128, -32640,
                                              0x3f7f, 0x3f80, 0x3f81, -16513, -16512, -16511,
                                              0x3f01, 0x4001], dtype=torch.int32)
                        bits = torch.cat((edges, bits, bits - 32768)).to(torch.int16).view(torch.bfloat16).cuda()
                        q.copy_(bits.repeat((q.numel() + bits.numel() - 1) // bits.numel())[:q.numel()].reshape(shape))
                        k.copy_(q.flip((0, 2)))
                        coefficients = torch.tensor([0., -0., 1., -1., .5, -.5, .50390625,
                                                     -.50390625, .99609375, -.99609375],
                                                    device='cuda', dtype=torch.bfloat16)
                        cache.copy_(coefficients.repeat((cache.numel() + 9) // 10)[:cache.numel()].reshape(rows, 96))
                    positions = torch.arange(rows, device='cuda').flip(0) * 17 + 10000
                    inputs = (q, k, cache)
                    before = tuple(x.clone() for x in inputs)
                    expected = model._apply_rope_qk(*inputs, positions)
                    with patch.object(rope, 'apply_rope_qk', wraps=rope.apply_rope_qk) as launch:
                        actual = model._apply_rope_qk(*inputs, positions, fused=True)
                    launch.assert_called_once()
                    self.assertGreater(q.storage_offset(), 0)
                    self.assertGreater(cache.storage_offset(), 0)
                    for output, reference, original in zip(actual, expected, (q, k)):
                        self.assertTrue(torch.isfinite(output).all().item())
                        assert_bytes(self, output, reference)
                        assert_bytes(self, output[..., 96:], original[..., 96:])
                    for original, copy in zip(inputs, before):
                        assert_bytes(self, original, copy)
                    pointers = [x.untyped_storage().data_ptr() for x in (*inputs, *actual)]
                    self.assertEqual(len(set(pointers)), len(pointers))
                    for output, reference in zip(model._apply_rope_qk(*inputs, positions + 99, fused=True), expected):
                        assert_bytes(self, output, reference)


if __name__ == '__main__':
    unittest.main()
