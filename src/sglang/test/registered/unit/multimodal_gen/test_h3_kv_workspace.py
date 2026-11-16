# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from sglang.multimodal_gen.runtime.layers.kvcache.minimax_h3_causal import (
    H3CausalForward,
    H3CausalKVCache,
    h3_causal_transaction,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("groups", [(0,), (2, 0)])
def test_output_storage_preserves_layers_heads_and_fork(groups):
    cache = H3CausalKVCache(3)
    cache.owners = (-1, 2)
    cache.layers = {
        layer: tuple(
            torch.arange(16).reshape(2, 4, 2).float() + 100 * layer + i
            for i in range(2)
        )
        for layer in range(3)
    }
    parent = cache.fork()
    expected = {
        layer: tuple(
            torch.cat((old, torch.full((1, 4, 2), 10.0 + layer + i)))
            for i, old in enumerate(pair)
        )
        for layer, pair in cache.layers.items()
    }
    width = 4 // len(groups)
    with h3_causal_transaction(H3CausalForward(cache, (3,), True)) as tx:
        assert tx._output is not None
        for start in groups:
            for layer in range(3):
                q = torch.zeros(1, width, 2)
                k, v = (torch.full_like(q, 10.0 + layer + i) for i in range(2))
                tx.attend(
                    layer,
                    q,
                    k,
                    v,
                    lambda q, k, v: q.clone(),
                    head_start=start,
                    total_heads=4,
                )
                k.fill_(-99)
                v.fill_(-99)
    storage = set()
    for layer in range(3):
        for actual, wanted in zip(cache.layers[layer], expected[layer]):
            assert actual.is_contiguous()
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            storage.add(actual.untyped_storage().data_ptr())
        for old, wanted in zip(parent.layers[layer], expected[layer]):
            torch.testing.assert_close(old, wanted[:2], rtol=0, atol=0)
    assert len(storage) == 1


def test_temporary_storage_reuse_does_not_publish_or_damage_cache():
    cache = H3CausalKVCache(3)
    cache.owners = (-1, 2)
    cache.layers = {
        layer: (
            torch.full((2, 4, 2), float(layer)),
            torch.full((2, 4, 2), float(layer + 10)),
        )
        for layer in range(3)
    }
    original = cache.layers
    pointers = []

    def attention(q, k, v):
        layer = len(pointers)
        assert k.is_contiguous() and v.is_contiguous()
        torch.testing.assert_close(k[:2], original[layer][0])
        torch.testing.assert_close(v[:2], original[layer][1])
        assert bool((k[2:] == 70).all()) and bool((v[2:] == 80).all())
        pointers.append(k.data_ptr())
        return q.clone()

    with h3_causal_transaction(H3CausalForward(cache, (3,), False)) as tx:
        for layer in range(3):
            q = torch.zeros(1, 4, 2)
            tx.attend(layer, q, q + 70, q + 80, attention, head_start=0, total_heads=4)
    assert cache.layers is original and cache.owners == (-1, 2)
    assert len(set(pointers)) == 1
