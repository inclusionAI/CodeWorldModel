# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from msgspec.structs import replace
from sglang.multimodal_gen.configs.sample.minimax_h3_causal import H3FixedHistoryConfig
from sglang.multimodal_gen.runtime.layers.kvcache.minimax_h3_causal import (
    H3CausalForward,
    H3CausalFusedForward,
    H3CausalKVCache,
    h3_causal_transaction,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.causal_latent import (
    H3FixedHistorySession,
    H3SourceLatents,
    H3StaticCondition,
    _Frame,
)
from torch.nn.functional import scaled_dot_product_attention
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _attention(q, k, v):
    return scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    ).transpose(0, 1)


def test_retain_constructs_indexes_once_per_device_and_rebuilds_on_next_call():
    cache = H3CausalKVCache(2)
    cache.owners = (-1, 0, 2)
    cache.layers = {
        layer: tuple(torch.arange(6).reshape(3, 1, 2) + i + layer for i in range(2))
        for layer in range(2)
    }
    snapshot = cache.fork()
    with patch.object(torch, "tensor", wraps=torch.tensor) as construct:
        cache.retain(2)
        assert construct.call_count == 1
        cache.retain(3)
        assert construct.call_count == 2
    assert cache.owners == (-1,) and snapshot.owners == (-1, 0, 2)
    for layer in cache.layers:
        for result, original in zip(cache.layers[layer], snapshot.layers[layer]):
            torch.testing.assert_close(result, original[:1])
            assert original.shape[0] == 3


def test_retain_without_eviction_preserves_storage_and_active_guard():
    cache = H3CausalKVCache(1)
    cache.owners = (-1, 3, 4)
    cache.layers = {0: (torch.zeros(3, 2, 2), torch.ones(3, 2, 2))}
    layers = cache.layers
    with patch.object(torch, "tensor", wraps=torch.tensor) as construct:
        cache.retain(3)
        assert cache.layers is layers and not construct.called
    cache._active = True
    with pytest.raises(RuntimeError, match="during a forward"):
        cache.retain(3)


@pytest.mark.parametrize("head_starts", [(0,), (2, 0)])
@pytest.mark.parametrize("recent", [0, 3, 6])
def test_history_commit_retains_after_attention_without_changing_queries(
    head_starts, recent
):
    cache = H3CausalKVCache(2)
    cache.owners = (-1, 0, 3)
    generator = torch.Generator().manual_seed(12)
    cache.layers = {
        layer: tuple(torch.randn(3, 4, 2, generator=generator) for _ in range(2))
        for layer in range(2)
    }
    oracle = cache.fork()
    inputs = [
        tuple(torch.randn(2, 4, 2, generator=generator) for _ in range(3))
        for _ in range(2)
    ]
    results = []
    for target, trim in ((oracle, None), (cache, recent)):
        output = [torch.empty_like(q) for q, _, _ in inputs]
        original = target.layers
        with h3_causal_transaction(H3CausalForward(target, (4, 5), True, trim)) as tx:
            width = 4 // len(head_starts)
            for start in head_starts:
                for layer, parts in enumerate(inputs):
                    output[layer][:, start : start + width] = tx.attend(
                        layer,
                        *(v[:, start : start + width] for v in parts),
                        _attention,
                        head_start=start,
                        total_heads=4,
                    )
            assert target.layers is original
        results.append(output)
    oracle.retain(recent)
    assert cache.owners == oracle.owners
    for actual, expected in zip(results[0], results[1]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for layer in cache.layers:
        for actual, expected in zip(cache.layers[layer], oracle.layers[layer]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("populated_cache", [False, True])
def test_fused_index_construction_is_scoped_to_each_transaction(populated_cache):
    cache = H3CausalKVCache(2)
    if populated_cache:
        cache.owners = (-1, 0)
        cache.layers = {
            layer: (torch.zeros(2, 4, 2), torch.ones(2, 4, 2)) for layer in range(2)
        }
    value = torch.arange(24).float().reshape(3, 4, 2)
    with patch.object(torch, "tensor", wraps=torch.tensor) as construct:
        for recent in (1, 3):
            request = H3CausalFusedForward(
                cache, (recent, recent + 1, recent + 1), 1, 2, recent
            )
            with h3_causal_transaction(request) as tx:
                for layer in range(2):
                    for head in (2, 0):
                        part = value[:, head : head + 2]
                        tx.attend(
                            layer,
                            part,
                            part,
                            part,
                            _attention,
                            head_start=head,
                            total_heads=4,
                        )
        expected = (
            [(0, 2, 3), (0, 3, 4)] if populated_cache else [(0, 1)] * 4 + [(2, 3)]
        )
        assert [call.args[0] for call in construct.call_args_list] == expected
    assert cache.owners == ((-1, 3, 4) if populated_cache else (3, 4))


@pytest.mark.parametrize("group_order", [(0,), (2, 0, 1)])
@pytest.mark.parametrize(
    "populated_cache,recent,expected_owners",
    [
        (True, 0, (-1, 2, 5, 5, 6, 7, 8, 9)),
        (True, 6, (-1, 6, 7, 8, 9)),
        (True, 8, (-1, 8, 9)),
        (False, 6, (6, 7, 8, 9)),
    ],
)
def test_fused_attention_matches_separate_h_c_d_across_retention(
    group_order,
    populated_cache,
    recent,
    expected_owners,
):
    generator = torch.Generator().manual_seed(103)
    cache = H3CausalKVCache(2)
    if populated_cache:
        cache.owners = (-1, 2, 5)
        cache.layers = {
            layer: tuple(torch.randn(3, 6, 4, generator=generator) for _ in range(2))
            for layer in range(2)
        }
    expected_cache = cache.fork()
    original_layers = cache.layers
    original_owners = cache.owners
    owners = (5, 6, 7, 8, 9, 8, 9)
    inputs = [
        tuple(torch.randn(8, 6, 4, generator=generator) for _ in range(3))
        for _ in range(2)
    ]
    expected = [torch.zeros_like(q) for q, _, _ in inputs]
    for start, stop, write in ((0, 3, True), (3, 5, True), (5, 7, False)):
        if start == 3:
            expected_cache.retain(recent)
        with h3_causal_transaction(
            H3CausalForward(expected_cache, owners[start:stop], write)
        ) as tx:
            for layer, parts in enumerate(inputs):
                expected[layer][start:stop] = tx.attend(
                    layer,
                    *(value[start:stop] for value in parts),
                    _attention,
                    head_start=0,
                    total_heads=6,
                )
    head_count = 6 // len(group_order)
    actual = [torch.zeros_like(q) for q, _, _ in inputs]
    with h3_causal_transaction(H3CausalFusedForward(cache, owners, 3, 5, recent)) as tx:
        for group in group_order:
            head = group * head_count
            for layer, parts in enumerate(inputs):
                actual[layer][:, head : head + head_count] = tx.attend(
                    layer,
                    *(value[:, head : head + head_count] for value in parts),
                    _attention,
                    head_start=head,
                    total_heads=6,
                )
        assert cache.layers is original_layers and cache.owners == original_owners
    for result, oracle in zip(actual, expected):
        torch.testing.assert_close(result, oracle, rtol=1e-5, atol=1e-6)
        assert not result[-1].any()
    assert cache.owners == expected_cache.owners == expected_owners
    for layer in range(2):
        for result, oracle in zip(cache.layers[layer], expected_cache.layers[layer]):
            torch.testing.assert_close(result, oracle)


@pytest.mark.parametrize("failure", ["attention", "incomplete_heads"])
def test_fused_failure_preserves_old_window_and_publishes_no_partial_cache(failure):
    cache = H3CausalKVCache(1)
    cache.owners = (-1, 0)
    cache.layers = {0: (torch.zeros(2, 4, 2), torch.ones(2, 4, 2))}
    original_layers = cache.layers
    value = torch.zeros(4, 2, 2)
    calls = 0

    def attend(q, k, v):
        nonlocal calls
        calls += 1
        if failure == "attention" and calls == 2:
            raise RuntimeError("condition attention failed")
        return _attention(q, k, v)

    with (
        pytest.raises(RuntimeError),
        h3_causal_transaction(H3CausalFusedForward(cache, (1, 2, 2), 1, 2, 1)) as tx,
    ):
        tx.attend(0, value, value, value, attend, head_start=0, total_heads=4)
    assert cache.layers is original_layers and cache.owners == (-1, 0)
    assert not cache._active


class _SmallModel:
    arch = SimpleNamespace(latents_dim=2, audio_latents_dim=3)
    blocks = (None, None)
    adaln_cache = None

    def __init__(self):
        self.calls = []
        self.fail = False

    def __call__(self, *, causal_context, **packed):
        self.calls.append(causal_context)
        if self.fail and isinstance(causal_context, H3CausalFusedForward):
            raise RuntimeError("fused model failed")
        hidden = packed["x"][0].double()
        positions = packed["img_position_ids"][0]
        hidden = hidden + positions[:, [0, 1, 2, 0, 1, 2, 0, 1]] * 0.001
        times = packed["unique_timesteps"][packed["inverse_indices"]]
        hidden = hidden + times[:, None] * 0.1
        with h3_causal_transaction(causal_context) as tx:
            for layer in range(len(self.blocks)):
                q = (hidden * (layer + 1) * 0.2).reshape(-1, 4, 2)
                k, v = q + 0.1, hidden.reshape(-1, 4, 2)
                result = tx.attend(
                    layer, q, k, v, _attention, head_start=0, total_heads=4
                )
                hidden = (hidden + result.flatten(1)).tanh()
        return hidden[
            packed["img_pos_for_infer_output_info"]["position_ids"]
        ].float(), None


def _session(*, fused, recent=12):
    recipe = json.loads(
        (
            Path(__file__).with_name("h3_step600_recipe.json")
        ).read_text()
    )
    session = object.__new__(H3FixedHistorySession)
    session.config = replace(
        H3FixedHistoryConfig.from_recipe(recipe), recent_latents=recent
    )
    session.fuse_first_denoise = fused
    session.frame_aligned_rope = False
    session.device = torch.device("cpu")
    session.ulysses_world_size = 4
    session.model = _SmallModel()
    session.static = H3StaticCondition(
        torch.ones(3, 4),
        torch.tensor([1, 0, 1]),
        torch.ones(2, 1, 2, 2),
        torch.zeros(2, 1, 2, 2),
    )
    session.sigmas = torch.tensor([1, 36 / 37, 12 / 13, 4 / 5, 0], dtype=torch.float64)
    session.reset()
    return session


def _source(count, generator):
    values = tuple(torch.randn(2, count, 2, 2, generator=generator) for _ in range(4))
    return H3SourceLatents((values[0],), (values[1],), values[2], values[3])


@pytest.mark.parametrize("frame_aligned", [False, True])
def test_fused_packing_matches_separate_rows_positions_times_and_target_indexes(frame_aligned):
    session = _session(fused=True)
    session.frame_aligned_rope = frame_aligned
    value = torch.arange(8).reshape(2, 1, 2, 2).float()
    frame = _Frame(14, (value + 1,), (value - 1,), value, value * 0.2, value)
    history = [replace(frame, position=9), replace(frame, position=10)]
    current = [frame, replace(frame, position=15), replace(frame, position=16)]
    options = {"static": False, "prefix": False, "sigma": session.sigmas[0]}
    parts = [
        session._pack(
            frames, conditions=conditions, video=video, denoise=denoise, **options
        )
        for frames, conditions, video, denoise in (
            (history, False, True, False),
            (current, True, False, False),
            (current, False, True, True),
        )
    ]
    packed, owners, shape = session._pack(
        current, conditions=False, video=True, denoise=True, history=history, **options
    )
    assert owners == tuple(
        owner for _, part_owners, _ in parts for owner in part_owners
    )
    assert shape == parts[-1][2] and packed["x"].shape[1] % 4 == 0
    for key in ("x", "img_position_ids"):
        expected = torch.cat(
            [part[key][:, : len(part_owners)] for part, part_owners, _ in parts], dim=1
        )
        torch.testing.assert_close(packed[key][:, : len(owners)], expected)
    expected_times = torch.cat(
        [
            part["unique_timesteps"][part["inverse_indices"]][: len(part_owners)]
            for part, part_owners, _ in parts
        ]
    )
    torch.testing.assert_close(
        packed["unique_timesteps"][packed["inverse_indices"]][: len(owners)],
        expected_times,
    )
    assert packed["img_pos_for_infer_output_info"]["position_ids"].tolist() == [5, 6, 7]
    assert not packed["x"][:, len(owners) :].any()
    assert packed["token_tags"][len(owners) :].eq(-1).all()


@pytest.mark.parametrize("packets,recent", [(1, 12), (15, 12), (5, 2)])
def test_fused_session_matches_separate_generation_through_short_windows_eos_and_reset(
    packets, recent
):
    baseline, fused = (
        _session(fused=False, recent=recent),
        _session(fused=True, recent=recent),
    )
    generator = torch.Generator().manual_seed(316)
    for count in [5] * packets + [2]:
        source = _source(count, generator)
        outputs = [
            (session.finish if count == 2 else session.append)(source)
            for session in (baseline, fused)
        ]
        assert [span.plan for span in outputs[0]] == [span.plan for span in outputs[1]]
        for original, result in zip(*outputs):
            torch.testing.assert_close(
                result.video, original.video, rtol=1e-4, atol=1e-5
            )
    assert fused.state == "ended" and not fused.pending_history
    for session in (baseline, fused):
        assert not session.model.calls[-1].write
        for submit in (session.append, session.finish):
            with pytest.raises(RuntimeError, match="reset first"):
                submit(source)
    assert baseline.cache.owners == fused.cache.owners
    for layer in baseline.cache.layers:
        for result, original in zip(
            fused.cache.layers[layer], baseline.cache.layers[layer]
        ):
            torch.testing.assert_close(result, original, rtol=1e-4, atol=1e-5)
    if packets == 15:
        assert len(baseline.model.calls) == 101 and len(fused.model.calls) == 83
        fused_calls = [
            call for call in fused.model.calls if isinstance(call, H3CausalFusedForward)
        ]
        assert len(fused_calls) == 9
        assert all(
            call.history_stop == 5 and call.condition_stop == 10 for call in fused_calls
        )
    fused.reset()
    assert (
        not fused.pending_history and not fused.cache.owners and fused.state == "open"
    )
    assert fused.append(_source(5, generator)) == ()


def test_failed_fused_session_keeps_previous_cache_and_pending_history_until_reset():
    session = _session(fused=True)
    generator = torch.Generator().manual_seed(97)
    for _ in range(3):
        session.append(_source(5, generator))
    cache, pending = session.cache, session.pending_history
    assert pending
    session.model.fail = True
    with pytest.raises(RuntimeError, match="fused model failed"):
        session.append(_source(5, generator))
    assert (
        session.cache is cache
        and session.pending_history is pending
        and session.state == "failed"
    )
    with pytest.raises(RuntimeError, match="reset first"):
        session.append(_source(5, generator))
    session.reset()
    assert not session.pending_history and not session.cache.owners


def test_cwm_prefix_refresh_preserves_fused_history_and_short_eos():
    generator = torch.Generator().manual_seed(316)
    sessions = [_session(fused=False), _session(fused=True)]
    for session in sessions:
        session.frame_aligned_rope = True
        session.config = replace(session.config, rope_reference_latents=47)
        session.reset()
    for count in [5] * 10 + [2]:
        source = _source(count, generator)
        outputs = [(s.finish if count == 2 else s.append)(source) for s in sessions]
        assert [span.plan for span in outputs[0]] == [span.plan for span in outputs[1]]
        for original, fused in zip(*outputs):
            difference = fused.video - original.video
            assert difference.abs().max() < 2e-4
            assert difference.norm() / original.video.norm() < 1e-4
    assert sessions[1].static_rope_shift == pytest.approx(170 / 3)
    assert any(isinstance(call, H3CausalFusedForward) for call in sessions[1].model.calls)
