import unittest

import torch
from torch import nn

from projects.minimax_h3.modeling.inference_optimizations import (
    cached_adaln, configure_inference, prepare_adaln_input)
from projects.minimax_h3.modeling.transformer.causal_model_sp import CausalMiniMaxH3DiTModelSP


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 24)
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return cached_adaln(self, value, lambda x: self.linear(x).chunk(6, dim=-1))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.adaln_proj = Projection()
        self.calls = 0

    def forward(self, hidden, *, adaln_input, adaln_params=None, **kwargs):
        self.calls += 1
        params = self.adaln_proj(adaln_input) if adaln_params is None else adaln_params
        shift, scale, gate, shift2, scale2, gate2 = params
        hidden = hidden + torch.tanh(hidden * scale + shift) * gate
        return hidden + torch.tanh(hidden * scale2 + shift2) * gate2


class Stack(nn.Module):
    _run_blocks = CausalMiniMaxH3DiTModelSP._run_blocks

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block(), Block()])
        self.gradient_checkpointing = True
        self.block_checkpoint_step = 1
        self.block_checkpoint_start_idx = 0


class PreparedBlocksTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = configure_inference(Stack().eval())
        self.hidden = torch.randn(2, 4)

    def test_default_stack_and_checkpoint_gradients(self):
        value = torch.randn(2, 4, requires_grad=True)
        expected = self.hidden
        for block in self.model.blocks:
            expected = block(expected, adaln_input=value)
        actual = self.model._run_blocks(self.hidden, adaln_input=value, adaln_params=None)
        self.assertTrue(torch.equal(actual, expected))
        actual.sum().backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertTrue(all(p.grad is not None for p in self.model.parameters()))

    @torch.no_grad()
    def test_prepared_stack_consumes_each_layers_current_values(self):
        for timestep in (1., .8, .4, .1, .8):
            value = prepare_adaln_input(self.model, torch.full((2, 4), timestep))
            kwargs = dict(adaln_input=value, adaln_params=None)
            expected = self.model._run_blocks(self.hidden, **kwargs)
            prepared = tuple(block.adaln_proj(value) for block in self.model.blocks)
            calls = [block.adaln_proj.calls for block in self.model.blocks]
            actual = self.model._run_blocks(self.hidden, block_adaln_params=prepared, **kwargs)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual([block.adaln_proj.calls for block in self.model.blocks], calls)
            self.assertIsNone(kwargs['adaln_params'])
            wrong = self.model._run_blocks(self.hidden, block_adaln_params=prepared[::-1], **kwargs)
            self.assertFalse(torch.equal(wrong, actual))

    @torch.no_grad()
    def test_eager_preparation_preserves_cache_invalidation_and_bypass(self):
        value = prepare_adaln_input(self.model, torch.ones(2, 4))

        def compare():
            prepared = tuple(block.adaln_proj(value) for block in self.model.blocks)
            direct = tuple(block.adaln_proj.linear(value).chunk(6, dim=-1) for block in self.model.blocks)
            actual = self.model._run_blocks(self.hidden, adaln_input=value, block_adaln_params=prepared)
            expected = self.model._run_blocks(self.hidden, adaln_input=value, block_adaln_params=direct)
            self.assertTrue(torch.equal(actual, expected))
            return actual

        first = compare()
        self.assertTrue(torch.equal(first, compare()))
        value.add_(.25)
        self.assertFalse(torch.equal(first, compare()))
        layer = self.model.blocks[1].adaln_proj.linear
        layer.weight.add_(.1)
        compare()
        layer.weight = nn.Parameter(layer.weight.clone().add_(.1))
        compare()
        layer.load_state_dict({name: tensor + .1 for name, tensor in layer.state_dict().items()})
        compare()
        configure_inference(self.model)
        compare()
        configure_inference(self.model, adaln=False)
        compare()
        configure_inference(self.model)
        with torch.inference_mode():
            value = torch.ones(2, 4)
            compare()

    def test_invalid_prepared_call_fails_before_any_block(self):
        value = torch.ones(2, 4)
        prepared = tuple(tuple(value for _ in range(6)) for _ in self.model.blocks)
        with torch.no_grad():
            for supplied, common in ((prepared[:-1], None), (prepared + prepared[:1], None),
                                     (prepared, prepared[0])):
                with self.assertRaises(ValueError):
                    self.model._run_blocks(self.hidden, adaln_input=value,
                        block_adaln_params=supplied, adaln_params=common)
            self.model.train()
            with self.assertRaises(RuntimeError):
                self.model._run_blocks(self.hidden, adaln_input=value, block_adaln_params=prepared)
        self.model.eval()
        with self.assertRaises(RuntimeError):
            self.model._run_blocks(self.hidden, adaln_input=value, block_adaln_params=prepared)
        self.assertEqual([block.calls for block in self.model.blocks], [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
