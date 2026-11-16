"""CPU shard arithmetic using the installed native SGLang linear implementation."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modeling.video_vae.causal_vit_tp import make_tp_blocks, tp_post
from sglang.multimodal_gen.runtime.layers import linear as native_linear


def reference():
    torch.manual_seed(77)
    block = nn.Module()
    block.attn, block.ff = nn.Module(), nn.Module()
    block.attn.to_qkv = nn.Linear(8, 24)
    block.attn.to_out = nn.Linear(8, 8)
    block.ff.w1, block.ff.w2 = nn.Linear(8, 24), nn.Linear(12, 8)
    block.ff.use_gated, block.ff.act_fn = True, nn.SiLU()
    block.norm1, block.norm2 = nn.LayerNorm(8), nn.LayerNorm(8)
    block.scale1 = nn.Parameter(torch.ones(8))
    block.scale2 = nn.Parameter(torch.ones(8))
    return SimpleNamespace(heads=4, dh=2, blocks=nn.ModuleList([block]))


class Pair:
    def __init__(self):
        self.barrier = Barrier(2)
        self.inputs = [[], []]

    def group(self, rank):
        pair = self

        class Group:
            world_size, rank_in_group = 2, rank

            def all_reduce(self, value):
                assert value.dtype == torch.float32 and value.is_contiguous()
                pair.inputs[rank].append(value.clone())
                index = len(pair.inputs[rank]) - 1
                pair.barrier.wait(timeout=10)
                result = pair.inputs[0][index] + pair.inputs[1][index]
                pair.barrier.wait(timeout=10)
                return result

        return Group()


@pytest.fixture(autouse=True)
def reject_global_group(monkeypatch):
    def fail():
        raise AssertionError('decoder TP consulted the global DiT group')
    monkeypatch.setattr(native_linear, 'get_tp_group', fail)


def test_official_linears_preserve_whole_qkv_heads_and_corresponding_gate_up_channels():
    ref, pair = reference(), Pair()
    source = ref.blocks[0]
    source_parameters = {name: value.clone() for name, value in source.named_parameters()}
    for rank in range(2):
        group = pair.group(rank)
        block = make_tp_blocks(ref, group)[0]
        qkv = block.attn.to_qkv.linear
        assert isinstance(qkv, native_linear.ColumnParallelLinear)
        torch.testing.assert_close(qkv.weight,
            source.attn.to_qkv.weight.reshape(4, 3, 2, 8)[rank * 2:(rank + 1) * 2]
                .reshape(12, 8).bfloat16(), rtol=0, atol=0)
        gate, up = source.ff.w1.weight.chunk(2, dim=0)
        torch.testing.assert_close(block.ff.w1.linear.weight,
            torch.cat((gate.chunk(2)[rank], up.chunk(2)[rank])).bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(block.ff.w2.linear.weight,
            source.ff.w2.weight.chunk(2, dim=1)[rank].bfloat16(), rtol=0, atol=0)
        for projection in (qkv, block.attn.to_out.linear, block.ff.w1.linear, block.ff.w2.linear):
            assert projection.tp_group is group
            assert projection.weight.dtype == torch.bfloat16
        assert block.norm1.weight.dtype == torch.float32
    for name, value in source.named_parameters():
        torch.testing.assert_close(value, source_parameters[name], rtol=0, atol=0)
    assert ref.heads == 4


def test_two_post_gemm_fp32_reductions_return_the_summed_storage_and_add_bias_once():
    ref, pair = reference(), Pair()
    blocks = [make_tp_blocks(ref, pair.group(rank))[0] for rank in range(2)]
    x, suffix = torch.randn(9, 8), torch.randn(5, 8)
    output, suffix_output = torch.randn(9, 4, 2).bfloat16(), torch.randn(5, 8).bfloat16()
    inverse = torch.arange(8, -1, -1)
    gemm_rows = [[], []]
    for rank, block in enumerate(blocks):
        for row in (block.attn.to_out.linear, block.ff.w2.linear):
            row.register_forward_pre_hook(
                lambda module, args, rank=rank: gemm_rows[rank].append(args[0].shape[0]))

    def run(rank):
        with torch.no_grad():
            return tp_post(blocks[rank], x, suffix, output[:, rank * 2:(rank + 1) * 2],
                suffix_output.chunk(2, dim=1)[rank], inverse)

    with ThreadPoolExecutor(2) as pool:
        result = list(pool.map(run, range(2)))
    assert [len(values) for values in pair.inputs] == [2, 2]
    assert gemm_rows == [[9, 5, 9, 5]] * 2
    first = (pair.inputs[0][0] + pair.inputs[1][0]
             + blocks[0].attn.to_out.linear.bias.float()).bfloat16().float()
    second = (pair.inputs[0][1] + pair.inputs[1][1]
              + blocks[0].ff.w2.linear.bias.float()).bfloat16().float()
    expected = torch.cat((x, suffix)) + first + second
    for spatial, tail in result:
        torch.testing.assert_close(torch.cat((spatial, tail)), expected, rtol=0, atol=0)


def test_tp_rejects_wrong_group_size_before_copying_weights():
    with pytest.raises(ValueError, match='two ranks'):
        make_tp_blocks(reference(), SimpleNamespace(world_size=4))
