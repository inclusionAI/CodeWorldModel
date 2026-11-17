"""Head-parallel projections for the existing causal fast decoder state."""
from __future__ import annotations

import copy

import torch
from torch import nn


class _TensorLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear(x)[0]


@torch.no_grad()
def make_tp_blocks(ref, group):
    from sglang.multimodal_gen.runtime.layers.linear import (
        ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear,
    )

    if group.world_size != 2 or ref.heads % group.world_size:
        raise ValueError("causal decoder TP requires two ranks and divisible whole heads")
    blocks = nn.ModuleList()
    for source in ref.blocks:
        block = copy.deepcopy(source)
        for parent, name, kind in (
            (block.attn, "to_qkv", "column"),
            (block.attn, "to_out", "row"),
            (block.ff, "w1", "gated" if block.ff.use_gated else "column"),
            (block.ff, "w2", "row"),
        ):
            dense = getattr(parent, name)
            kwargs = dict(bias=dense.bias is not None, params_dtype=torch.bfloat16,
                          tp_group=group)
            if kind == "row":
                linear = RowParallelLinear(dense.in_features, dense.out_features,
                    input_is_parallel=True, reduce_results=False, skip_bias_add=True, **kwargs)
            elif kind == "gated":
                linear = MergedColumnParallelLinear(dense.in_features,
                    [dense.out_features // 2] * 2, gather_output=False, **kwargs)
            else:
                linear = ColumnParallelLinear(dense.in_features, dense.out_features,
                    gather_output=False, **kwargs)
            linear.to(dense.weight.device)
            # QKV rows are [head,Q/K/V,dim]; merged FFN loads matching gate/up halves.
            for field in ("weight", "bias"):
                param = getattr(linear, field)
                if param is not None:
                    param.weight_loader(param, getattr(dense, field))
            setattr(parent, name, _TensorLinear(linear))
        blocks.append(block)
    return blocks.eval().requires_grad_(False)


def _reduce_pair(projection, x, suffix):
    linear = projection.linear
    spatial, spatial_bias = linear(x)
    tail, _ = linear(suffix)
    # Pack only the GEMM results so spatial and suffix keep their original GEMM shapes.
    reduced = linear.tp_group.all_reduce(torch.cat((spatial.float(), tail.float()), dim=0))
    if spatial_bias is not None:
        reduced = reduced + spatial_bias.float()
    reduced = reduced.to(torch.bfloat16).float()
    return reduced[:x.shape[0]], reduced[x.shape[0]:]


def _ff_input(ff, x):
    hidden = ff.w1(x)
    if ff.use_gated:
        gate, hidden = hidden.chunk(2, dim=-1)
        return ff.act_fn(gate).mul_(hidden)
    return ff.act_fn(hidden)


def tp_post(block, x, suffix, output, suffix_output, inverse):
    spatial, tail = _reduce_pair(block.attn.to_out, output.flatten(1)[inverse], suffix_output)
    x = x + spatial * block.scale1
    suffix = suffix + tail * block.scale1
    spatial, tail = _reduce_pair(block.ff.w2,
        _ff_input(block.ff, block.norm2(x).to(torch.bfloat16)),
        _ff_input(block.ff, block.norm2(suffix).to(torch.bfloat16)))
    return x + spatial * block.scale2, suffix + tail * block.scale2
