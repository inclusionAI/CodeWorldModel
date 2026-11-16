# SPDX-License-Identifier: Apache-2.0
from types import MethodType, SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.configs.models.dits.minimax_h3_causal import (
    MiniMaxH3CausalDiTArchConfig,
)
from sglang.multimodal_gen.runtime.layers.kvcache.minimax_h3_causal import (
    H3CausalForward,
    H3CausalFusedForward,
    H3CausalKVCache,
    h3_causal_transaction,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    CudnnSDPAImpl,
    SDPAImpl,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_causal import (
    MiniMaxH3CausalAdalnProj,
    MiniMaxH3CausalAttention,
    MiniMaxH3CausalDiTModel,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "backend,implementation",
    [
        (AttentionBackendEnum.TORCH_SDPA, SDPAImpl),
        (AttentionBackendEnum.TORCH_CUDNN_SDPA, CudnnSDPAImpl),
    ],
)
@torch.no_grad()
def test_native_dense_attention_reads_longer_kv_history_on_cpu(backend, implementation):
    generator = torch.Generator().manual_seed(37)
    query, key, value = (torch.randn(2, 2, 32, generator=generator) for _ in range(3))
    old_key, old_value = (torch.randn(3, 2, 32, generator=generator) for _ in range(2))
    attention = SimpleNamespace(
        q_norm=torch.nn.RMSNorm(32),
        k_norm=torch.nn.RMSNorm(32),
        head_dim=32,
        num_heads=2,
        bcg_breakpoint=False,
        local_inner_dim=64,
        causal_layer_idx=0,
        _attention_backend_enum=backend,
        _attention_impl=implementation(2, 32, False, 32**-0.5),
    )
    attention._attention_after_exchange = MethodType(
        MiniMaxH3CausalAttention._attention_after_exchange, attention
    )
    cache = H3CausalKVCache(1)
    cache.layers[0] = old_key, old_value
    cache.owners = (0, 1, 2)
    expected = attention._attention_impl.forward(
        attention.q_norm(query)[None],
        torch.cat((old_key, attention.k_norm(key)))[None],
        torch.cat((old_value, value))[None],
        None,
    )[0].flatten(1)
    with h3_causal_transaction(H3CausalForward(cache, (3, 4))):
        result = MiniMaxH3CausalAttention._attend(
            attention,
            query.flatten(1),
            query,
            key,
            value,
            rope_cache=None,
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            cu_seqlens_host=(0, 2),
            max_seqlen=2,
            subblock_sparse_query_block_mask=None,
            ulysses_active=False,
            ring_active=False,
        )
    torch.testing.assert_close(result, expected)
    assert cache.owners == (0, 1, 2)


def test_explicit_fa_cannot_silently_resolve_to_cpu_sdpa():
    if not current_platform.is_cpu():
        pytest.skip("requires the native CPU platform's FA-to-SDPA resolution")
    model = MiniMaxH3CausalDiTModel.__new__(MiniMaxH3CausalDiTModel)
    torch.nn.Module.__init__(model)
    model.arch = SimpleNamespace(attention_head_dim=32)
    model._resolved_attention_backend = None
    model._component_attention_backend_override = AttentionBackendEnum.FA
    with pytest.raises(ValueError, match="requested FA.*TORCH_SDPA"):
        model._resolve_attention_backend_once()


def test_fused_clean_time_projections_match_separate_rows_without_changing_denoise():
    model = MiniMaxH3CausalDiTModel.__new__(MiniMaxH3CausalDiTModel)
    torch.nn.Module.__init__(model)
    model.adaln_t_table = None
    model.time_embedder = lambda times: times[:, None] * 2 + len(times)
    projection = MiniMaxH3CausalAdalnProj.__new__(MiniMaxH3CausalAdalnProj)
    torch.nn.Module.__init__(projection)
    projection.linear = lambda rows: (rows * 3 + len(rows), None)
    times = torch.tensor([0.0, 0.999])
    original_embedding = model._time_embedding(times)
    original_projection = projection.project_local(original_embedding)
    clean_embedding = model._time_embedding(times[1:])
    clean_projection = projection.project_local(clean_embedding)
    cache = H3CausalKVCache(1)
    with h3_causal_transaction(
        H3CausalFusedForward(cache, (0, 1, 1), 1, 2, 0)
    ) as transaction:
        actual_embedding = model._time_embedding(times)
        actual_projection = projection.project_local(actual_embedding)
        torch.testing.assert_close(actual_embedding[:1], original_embedding[:1])
        torch.testing.assert_close(actual_embedding[1:], clean_embedding)
        torch.testing.assert_close(actual_projection[:1], original_projection[:1])
        torch.testing.assert_close(actual_projection[1:], clean_projection)
        value = torch.zeros(3, 1, 1)
        transaction.attend(
            0,
            value,
            value,
            value,
            lambda q, k, v: q,
            head_start=0,
            total_heads=1,
        )
    torch.testing.assert_close(model._time_embedding(times), original_embedding)
    torch.testing.assert_close(
        projection.project_local(original_embedding), original_projection
    )


@pytest.mark.parametrize(
    "tp,ulysses", [(1, 1), (1, 4), (2, 2), (4, 1), (1, 8), (2, 4), (4, 2), (8, 1)]
)
def test_native_step600_dimensions_admit_tp_ulysses_combinations(tp, ulysses):
    arch = MiniMaxH3CausalDiTArchConfig()
    MiniMaxH3CausalDiTModel._validate_tp_config(None, arch=arch, tp_size=tp)
    MiniMaxH3CausalDiTModel._validate_sequence_parallel_config(
        arch=arch, tp_size=tp, ulysses_size=ulysses, ring_size=1
    )


@pytest.mark.parametrize("tp,ulysses", [(7, 1), (2, 8), (8, 2)])
def test_native_dimensions_reject_unshardable_tp_ulysses_combinations(tp, ulysses):
    arch = MiniMaxH3CausalDiTArchConfig()
    with pytest.raises(ValueError, match="divisible"):
        MiniMaxH3CausalDiTModel._validate_tp_config(None, arch=arch, tp_size=tp)
        MiniMaxH3CausalDiTModel._validate_sequence_parallel_config(
            arch=arch, tp_size=tp, ulysses_size=ulysses, ring_size=1
        )
