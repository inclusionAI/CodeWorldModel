# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.causal_vit_fast import (
    FastCausalViT,
    _finish_pending,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.causal_vit_tp import (
    tp_post,
    tp_post_fused,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class PartialLinear(nn.Module):
    def __init__(self, width, group, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width, width).bfloat16() / width)
        self.bias = nn.Parameter(torch.randn(width).bfloat16()) if bias else None
        self.tp_group = group

    def forward(self, value):
        return nn.functional.linear(value, self.weight), self.bias


@pytest.mark.parametrize("tp,rows,bias,gated", [(2, 17, True, True), (4, 17, True, False), (4, 1, False, True)])
def test_deferred_post_preserves_rounding_permutation_and_residual_ownership(tp, rows, bias, gated):
    torch.manual_seed(32)
    width = 32
    reductions = []

    def reduce(value):
        assert value.dtype == torch.float32
        reductions.append(value.shape)
        return value * tp

    group = SimpleNamespace(all_reduce=reduce)
    projection = lambda: SimpleNamespace(linear=PartialLinear(width, group, bias))
    block = SimpleNamespace(attn=SimpleNamespace(to_out=projection()),
        ff=SimpleNamespace(w1=nn.Linear(width, width * (2 if gated else 1), dtype=torch.bfloat16),
            w2=projection(), use_gated=gated, act_fn=nn.SiLU()),
        norm2=nn.RMSNorm(width), scale1=torch.randn(width), scale2=torch.randn(width))
    with torch.no_grad():
        x, suffix = torch.randn(rows, width), torch.randn(5, width)
        owned = x.clone(), suffix.clone()
        output, tail = torch.randn(rows, 4, 8).bfloat16(), torch.randn(5, width).bfloat16()
        args = block, x, suffix, output, tail, torch.randperm(rows)
        expected = tp_post(*args)
        first, second, pending = tp_post_fused(*args)
        actual = _finish_pending(first, second, pending)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        assert reductions == [torch.Size([rows + 5, width])] * 4
        assert torch.equal(x, owned[0]) and torch.equal(suffix, owned[1])


@pytest.mark.parametrize("compile,group", [(False, object()), (True, None)])
def test_fused_post_rejects_missing_compile_or_tp_before_loading_cuda(compile, group):
    with pytest.raises(ValueError, match="compiled TP decoder"):
        FastCausalViT(None, compile=compile, tp_group=group, tp_fused_post=True)
