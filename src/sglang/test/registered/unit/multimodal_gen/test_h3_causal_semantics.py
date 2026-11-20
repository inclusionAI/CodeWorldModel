# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from msgspec.structs import replace
from sglang.multimodal_gen.configs.sample.minimax_h3_causal import H3FixedHistoryConfig
from sglang.multimodal_gen.runtime.layers.kvcache.minimax_h3_causal import (
    H3CausalForward,
    H3CausalFusedForward,
    H3CausalKVCache,
    current_h3_causal_transaction,
    h3_causal_transaction,
)
from sglang.multimodal_gen.runtime.loader.minimax_h3_causal import H3CausalComponents
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
    get_forward_context_or_none,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.causal_latent import (
    H3FixedHistorySession,
    H3SourceLatents,
    H3StaticCondition,
    H3WindowPlanner,
    H3WindowPlan,
    H3LatentRange,
    _Frame,
    h3_absolute_video_time,
    h3_fixed_euler_step,
    h3_cwm_static_shift,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_unpatchify_video_tokens,
)
from torch.nn.functional import scaled_dot_product_attention
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")



def _config():
    return H3FixedHistoryConfig.from_recipe(
        json.loads(
            Path(__file__).with_name("h3_step600_recipe.json").read_text()
        )
    )


def _sdpa(q, k, v):
    return scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    ).transpose(0, 1)


def _write(cache, owners, values):
    with h3_causal_transaction(H3CausalForward(cache, owners, True)) as transaction:
        for layer in range(cache.num_layers):
            q = torch.zeros(len(values), 1, 2)
            transaction.attend(
                layer,
                q,
                q,
                torch.tensor(values).float().view(-1, 1, 1).expand(-1, 1, 2),
                _sdpa,
                head_start=0,
                total_heads=1,
            )


def test_condition_and_history_stay_fixed_across_noisy_denoise_calls():
    cache = H3CausalKVCache(2)
    _write(cache, (-1, 0), [1, 3])
    _write(cache, (1,), [5])
    before = {
        layer: tuple(t.clone() for t in tensors)
        for layer, tensors in cache.layers.items()
    }
    for noisy in (7.0, 9000.0):
        with h3_causal_transaction(H3CausalForward(cache, (1,))) as transaction:
            for layer in range(2):
                q = torch.zeros(2, 1, 2)
                v = torch.tensor([noisy, 1e9]).view(2, 1, 1).expand(-1, 1, 2)
                result = transaction.attend(
                    layer, q, q, v, _sdpa, head_start=0, total_heads=1
                )
                torch.testing.assert_close(
                    result[0], torch.full((1, 2), (1 + 3 + 5 + noisy) / 4)
                )
                assert not result[1].any()
    assert cache.owners == (-1, 0, 1)
    for layer, tensors in cache.layers.items():
        for actual, expected in zip(tensors, before[layer]):
            assert torch.equal(actual, expected)
    _write(cache, (1,), [11])
    assert cache.layers[0][1][:, 0, 0].tolist() == [1, 3, 5, 11]


def test_failed_or_incomplete_forward_cannot_publish_partial_kv():
    cache = H3CausalKVCache(2)
    _write(cache, (-1,), [2])
    old = cache.layers
    with pytest.raises(RuntimeError, match="every cache layer"):
        with h3_causal_transaction(H3CausalForward(cache, (0,), True)) as transaction:
            q = torch.zeros(1, 1, 2)
            transaction.attend(0, q, q, q, _sdpa, head_start=0, total_heads=1)
    assert cache.layers is old and cache.owners == (-1,)
    with pytest.raises(ValueError, match="projection failed"):
        with h3_causal_transaction(H3CausalForward(cache, (0,), True)):
            raise ValueError("projection failed")
    assert cache.layers is old
    _write(cache, (0,), [7])
    assert cache.owners == (-1, 0)


@pytest.mark.parametrize("head_start,heads", [(0, 4), (1, 2)])
def test_attention_kv_share_backing_with_contiguous_independent_views(head_start, heads):
    cache = H3CausalKVCache(1)
    cache.owners = (-1, 0)
    cache.layers[0] = tuple(torch.randn(2, 4, 3) for _ in range(2))
    q, k, v = (torch.randn(4, heads, 3) for _ in range(3))
    expected = tuple(
        torch.cat((old[:, head_start : head_start + heads], live[:3]))
        for old, live in zip(cache.layers[0], (k, v))
    )

    def attention(query, key, value):
        assert key.is_contiguous() and value.is_contiguous()
        assert key.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        assert key.data_ptr() != value.data_ptr()
        assert torch.equal(key, expected[0]) and torch.equal(value, expected[1])
        return torch.zeros_like(query)

    with h3_causal_transaction(H3CausalForward(cache, (1, 2, 3))) as tx:
        if head_start:
            # Complete the other head slices without changing the tested inputs.
            for start in (0, 3):
                part = q[:, :1]
                tx.attend(0, part, part, part, _sdpa, head_start=start, total_heads=4)
        tx.attend(0, q, k, v, attention, head_start=head_start, total_heads=4)


def test_sink_and_recent_are_pruned_together_without_changing_a_fork():
    cache = H3CausalKVCache(2)
    _write(cache, (-1, -1, 1, 1, 8, 8), [1, 2, 3, 4, 5, 6])
    fork = cache.fork()
    cache.retain(8)
    assert cache.owners == (-1, -1, 8, 8)
    assert fork.owners == (-1, -1, 1, 1, 8, 8)
    assert cache.layers[0][1][:, 0, 0].tolist() == [1, 2, 5, 6]
    _write(cache, (9,), [9])
    assert fork.layers[0][1][:, 0, 0].tolist() == [1, 2, 3, 4, 5, 6]
    cache.reset()
    assert not cache.owners and not cache.layers
    _write(cache, (-1,), [20])
    assert cache.layers[1][1].flatten().tolist() == [20, 20]


def test_prefix_replacement_keeps_history_and_original_cache_unchanged():
    cache = H3CausalKVCache(2)
    _write(cache, (-1, -1, 3, 4), [1, 2, 3, 4])
    prefix = H3CausalKVCache(2)
    _write(prefix, (-1, -1), [10, 20])
    fork = cache.fork()
    fork.replace_prefix(prefix)
    assert fork.owners == cache.owners
    for layer in range(2):
        assert fork.layers[layer][1][:, 0, 0].tolist() == [10, 20, 3, 4]
        assert cache.layers[layer][1][:, 0, 0].tolist() == [1, 2, 3, 4]
    _write(prefix, (-1,), [30])
    with pytest.raises(ValueError, match='prefix'):
        fork.replace_prefix(prefix)


@pytest.mark.parametrize('start,stop,eos,expected', [
    (40, 45, False, 0), (45, 47, True, 0), (45, 50, False, 85/3),
    (45, 47, False, 85/3), (50, 55, False, 170/3), (60, 65, False, 340/3),
])
def test_cwm_static_shift_matches_official_long_stream(start, stop, eos, expected):
    plan = H3WindowPlan(0, False, H3LatentRange(0, 1), H3LatentRange(1, start),
                        H3LatentRange(start, stop), eos)
    assert h3_cwm_static_shift(plan, 47) == pytest.approx(expected)


def test_cwm_long_stream_windows_do_not_cross_rope_shift_boundaries():
    planner = H3WindowPlanner(replace(_config(), rope_reference_latents=47), frame_aligned_rope=True)
    plans = planner.ready(77, eos=True)
    spans = [(p.current.start, p.current.stop) for p in plans]
    assert (44, 45) in spans and (45, 50) in spans and (50, 55) in spans
    assert (65, 70) in spans and (70, 75) in spans
    assert spans[-1] == (75, 77)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("populated_cache", [False, True])
@pytest.mark.parametrize("group_order", [(0,), (1, 0)])
def test_published_kv_owns_projection_storage_without_changing_a_fork(
    fused, populated_cache, group_order
):
    cache = H3CausalKVCache(1)
    old = tuple(torch.arange(16).float().reshape(2, 4, 2) + i for i in (100, 200))
    if populated_cache:
        cache.owners = (-1, 0)
        cache.layers = {0: old}
    fork = cache.fork()
    old_values = tuple(value.clone() for value in old)
    inputs = tuple(torch.arange(32).float().reshape(4, 4, 2) + i for i in (0, 1, 2))
    request = (
        H3CausalFusedForward(cache, (1, 2, 2), 1, 2, 1)
        if fused
        else H3CausalForward(cache, (1, 2, 2), True)
    )
    expected_owners = ((-1,) if fused else (-1, 0)) if populated_cache else ()
    expected_owners += (1, 2) if fused else (1, 2, 2)
    old_rows = (1 if fused else 2) if populated_cache else 0
    new_rows = 2 if fused else 3
    expected = tuple(
        torch.cat((previous[:old_rows], value[:new_rows]))
        for previous, value in zip(old, inputs[1:])
    )
    head_count = 4 // len(group_order)
    with h3_causal_transaction(request) as transaction:
        for group in group_order:
            start = group * head_count
            transaction.attend(
                0,
                *(value[:, start : start + head_count] for value in inputs),
                _sdpa,
                head_start=start,
                total_heads=4,
            )
    for value in inputs:
        value.fill_(12345)
    assert cache.owners == expected_owners
    for actual, oracle in zip(cache.layers[0], expected):
        assert torch.equal(actual, oracle)
    assert fork.owners == ((-1, 0) if populated_cache else ())
    if populated_cache:
        for actual, oracle in zip(fork.layers[0], old_values):
            assert torch.equal(actual, oracle)
    else:
        assert not fork.layers


def test_cache_rejects_reentry_and_mutation_during_forward():
    cache = H3CausalKVCache(1)
    with h3_causal_transaction(H3CausalForward(cache, (0,))) as transaction:
        for mutation in (cache.reset, cache.fork, lambda: cache.retain(0)):
            with pytest.raises(RuntimeError, match="during a forward"):
                mutation()
        with pytest.raises(RuntimeError):
            with h3_causal_transaction(H3CausalForward(cache, (0,))):
                pass
        q = torch.zeros(1, 1, 2)
        transaction.attend(0, q, q, q, _sdpa, head_start=0, total_heads=1)
    with pytest.raises(RuntimeError, match="scoped forward"):
        current_h3_causal_transaction()


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
@pytest.mark.parametrize("write", [False, True])
def test_head_slices_match_full_attention_and_publish_complete_kv(order, write):
    generator = torch.Generator().manual_seed(316)
    full = H3CausalKVCache(2)
    full.owners = (-1, 0)
    full.layers = {
        layer: tuple(torch.randn(2, 6, 4, generator=generator) for _ in range(2))
        for layer in range(2)
    }
    split = full.fork()
    old = split.layers
    inputs = [
        tuple(torch.randn(4, 6, 4, generator=generator) for _ in range(3))
        for _ in range(2)
    ]
    expected = []
    with h3_causal_transaction(H3CausalForward(full, (1, 2, 3), write)) as transaction:
        for layer, (q, k, v) in enumerate(inputs):
            expected.append(
                transaction.attend(layer, q, k, v, _sdpa, head_start=0, total_heads=6)
            )
    with h3_causal_transaction(H3CausalForward(split, (1, 2, 3), write)) as transaction:
        actual = [torch.empty_like(q) for q, _, _ in inputs]
        for group in order:
            start = group * 2
            for layer, parts in enumerate(inputs):
                q, k, v = (part[:, start : start + 2] for part in parts)
                actual[layer][:, start : start + 2] = transaction.attend(
                    layer, q, k, v, _sdpa, head_start=start, total_heads=6
                )
        assert split.layers is old and split.owners == (-1, 0)
    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        assert not actual_value[3].any()
    assert split.owners == full.owners
    for layer in range(2):
        for actual_value, expected_value in zip(
            split.layers[layer], full.layers[layer]
        ):
            assert torch.equal(actual_value, expected_value)
    assert (split.layers is old) is not write
    for parts in inputs:
        for value in parts:
            value.fill_(12345)
    if write:
        assert not (split.layers[0][0] == 12345).any()


def test_missing_head_slice_cannot_publish_partial_kv():
    cache = H3CausalKVCache(1)
    cache.owners = (-1,)
    cache.layers = {0: (torch.zeros(1, 4, 2), torch.ones(1, 4, 2))}
    old = cache.layers
    with pytest.raises(RuntimeError, match="every head"):
        with h3_causal_transaction(H3CausalForward(cache, (0,), True)) as transaction:
            value = torch.zeros(2, 2, 2)
            transaction.attend(
                0, value, value, value, _sdpa, head_start=0, total_heads=4
            )
    assert cache.layers is old and cache.owners == (-1,) and not cache._active


def test_failed_head_callback_cannot_publish_even_if_caught_inside_forward():
    cache = H3CausalKVCache(1)
    old = cache.layers

    def fail(q, k, v):
        raise ValueError("attention failed")

    with pytest.raises(RuntimeError):
        with h3_causal_transaction(H3CausalForward(cache, (0,), True)) as transaction:
            value = torch.zeros(1, 2, 2)
            with pytest.raises(ValueError, match="attention failed"):
                transaction.attend(
                    0, value, value, value, fail, head_start=0, total_heads=4
                )
            transaction.attend(
                0, value, value, value, _sdpa, head_start=2, total_heads=4
            )
    assert cache.layers is old and cache.owners == () and not cache._active


def test_step600_windows_split_at_boundaries_without_advancing_nominal_schedule():
    config = _config()
    assert (config.sink_latents, config.recent_latents, config.shift) == (1, 12, 12)
    planner = H3WindowPlanner(config)
    plans = []
    for available in range(5, 71, 5):
        plans.extend(planner.ready(available))
    plans.extend(planner.ready(72, eos=True))
    ranges = [(p.current.start, p.current.stop) for p in plans]
    assert ranges[:3] == [(0, 9), (9, 14), (14, 19)]
    assert [(44, 45), (45, 49), (69, 70), (70, 72)] == [
        value for value in ranges if value[0] in (44, 45, 69, 70)
    ]
    assert sum(p.current.count for p in plans) == 72
    assert [p.index for p in plans] == list(range(len(plans)))
    assert [p.eos for p in plans].count(True) == 1 and plans[-1].eos
    assert plans[-1].dynamic_recent.start == 58
    with pytest.raises(ValueError, match="after EOS"):
        planner.ready(77)


def test_short_bootstrap_and_eos_drain_remaining_latents():
    planner = H3WindowPlanner(_config())
    assert planner.ready(5) == ()
    (short,) = planner.ready(7, eos=True)
    assert short.bootstrap and short.eos and short.current.count == 7
    (tiny,) = H3WindowPlanner(_config()).ready(2, eos=True)
    assert tiny.current.stop == 2 and tiny.bootstrap


def test_absolute_rope_keeps_native_role_origin_after_pruning():
    positions = torch.tensor([0, 1, 4, 5, 69, 70])
    actual = h3_absolute_video_time(positions, 12.0)
    assert actual.dtype == torch.float64
    assert actual[0] == 12.0
    torch.testing.assert_close(
        actual[1:3], torch.tensor([12 + 5 / 3, 12 + 65 / 3], dtype=torch.float64)
    )
    torch.testing.assert_close(
        actual[-1], torch.tensor(12 + 14 * (85 / 3), dtype=torch.float64)
    )
    assert torch.equal(actual[3:], h3_absolute_video_time(positions[3:], 12.0))


@pytest.mark.parametrize("velocity_dtype", [torch.bfloat16, torch.float32])
def test_euler_matches_reference_fp32_wrapper_and_fp64_schedule_arithmetic(
    velocity_dtype,
):
    state = torch.tensor([1.003, -2.14, 0.136], dtype=torch.float32)
    velocity = torch.tensor([0.31, -0.7, 1.5], dtype=velocity_dtype)
    t, s = (
        torch.tensor(12 / 13, dtype=torch.float64),
        torch.tensor(0.8, dtype=torch.float64),
    )
    packed = state.bfloat16()[None, :, None]
    shaped_time = (1 - t.float()).expand_as(packed)
    wrapper_x = torch.where(
        torch.zeros_like(packed, dtype=torch.bool), shaped_time * packed, packed
    )
    x0 = wrapper_x.flatten() + (1 - shaped_time.flatten()) * velocity
    epsilon = (state - (1 - t[None]) * x0) / t[None]
    expected = ((1 - s[None]) * x0 + s[None] * epsilon).float()
    actual = h3_fixed_euler_step(state, velocity, t, s)
    assert torch.equal(actual, expected)
    assert torch.equal(
        h3_fixed_euler_step(state, velocity, t, torch.zeros_like(t)), x0.float()
    )


def test_encoder_adapter_pads_eos_before_native_call_and_uses_mean_without_rng():
    class MomentsProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.processor = SimpleNamespace(
                convert_numpy_to_tensor=lambda rgb, device: (
                    torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float() / 255
                ),
                transform_tensor=lambda value: value,
            )
            self.sglang_config = SimpleNamespace(
                arch_config=SimpleNamespace(
                    latents_mean=[0.5] * 24,
                    latents_std=[2.0] * 24,
                )
            )

        def _adaptive_encode(self, value):
            self.received = value.clone()
            count = (value.shape[2] - 1) // 4 + 1
            mean = torch.arange(count, dtype=torch.bfloat16)[
                None, None, :, None, None
            ].expand(1, 24, count, 2, 2)
            return torch.cat((mean, torch.full_like(mean, 19)), dim=1)

    probe = MomentsProbe()
    components = H3CausalComponents(probe, probe, probe, None)
    rgb = (
        np.arange(5 * 32 * 32 * 3, dtype=np.int64)
        .astype(np.uint8)
        .reshape(5, 32, 32, 3)
    )
    before = torch.get_rng_state()
    result = components.encode_proxy(rgb, eos=True)
    assert probe.received.shape == (1, 3, 17, 32, 32)
    assert probe.received.dtype == torch.bfloat16
    assert torch.equal(
        probe.received[:, :, 5:], probe.received[:, :, 4:5].expand(-1, -1, 12, -1, -1)
    )
    assert result.shape == (24, 2, 2, 2) and result.dtype == torch.bfloat16
    assert torch.equal(result[:, 0], torch.full_like(result[:, 0], -0.25))
    assert torch.equal(result[:, 1], torch.full_like(result[:, 1], 0.25))
    assert torch.equal(before, torch.get_rng_state())
    assert components.encode_image(rgb[0]).shape == (24, 1, 2, 2)
    assert components.encode_proxy(np.repeat(rgb[:1], 17, axis=0), eos=False).shape == (
        24,
        5,
        2,
        2,
    )
    with pytest.raises(ValueError, match="17 RGB"):
        components.encode_proxy(rgb, eos=False)
    with pytest.raises(ValueError, match="uint8"):
        components.encode_image(rgb[0].astype(np.float32))


@pytest.mark.parametrize("fail", [False, True])
def test_condition_scopes_native_text_context_and_restores_caller(fail):
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [7] * len(text)}

        def convert_tokens_to_ids(self, token):
            return 8

    class Encoder:
        def encode_ids(self, ids, **kwargs):
            context = get_forward_context()
            assert context.current_timestep == 0 and context.attn_metadata is None
            assert context is not outer
            if fail:
                raise RuntimeError("encoder failure")
            return torch.ones(len(ids), 4)

    def image_processor(**kwargs):
        return {
            "pixel_values": torch.zeros(1, 3),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }

    image_processor.merge_size = 1
    processor = SimpleNamespace(image_processor=image_processor, tokenizer=Tokenizer())
    components = H3CausalComponents(None, Encoder(), None, processor)
    assert get_forward_context_or_none() is None
    with set_forward_context(73, object()):
        outer = get_forward_context()
        if fail:
            with pytest.raises(RuntimeError, match="encoder failure"):
                components.condition("prompt", np.zeros((32, 32, 3), dtype=np.uint8))
        else:
            hidden, tags = components.condition(
                "prompt", np.zeros((32, 32, 3), dtype=np.uint8)
            )
            assert hidden.shape == (len(tags), 4)
        assert get_forward_context() is outer
    assert get_forward_context_or_none() is None


def _packing_session(world_size=1):
    session = object.__new__(H3FixedHistorySession)
    session.frame_aligned_rope = False
    session.static_rope_shift = 0.0
    session.fuse_first_denoise = False
    session.pending_history = ()
    session.config = _config()
    session.device = torch.device("cpu")
    session.ulysses_world_size = world_size
    session._adaln_slots = None
    session.model = SimpleNamespace(
        arch=SimpleNamespace(latents_dim=2, audio_latents_dim=3)
    )
    session.static = H3StaticCondition(
        torch.ones(3, 4),
        torch.tensor([1, 0, 1]),
        torch.ones(2, 1, 4, 4),
        torch.zeros(2, 1, 4, 4),
    )
    value = torch.arange(16).reshape(2, 1, 2, 4).float()
    frame = _Frame(
        5, (value,), (torch.zeros_like(value),), value, torch.zeros_like(value), value
    )
    return session, frame


def test_cwm_frame_aligned_rope_shares_proxy_and_target_time():
    session, frame = _packing_session()
    options = dict(static=True, conditions=True, video=True, prefix=False,
                   denoise=True, sigma=torch.tensor(0.75))
    original, owners, _ = session._pack([frame], **options)
    session.frame_aligned_rope = True
    aligned, aligned_owners, _ = session._pack([frame], **options)
    positions = aligned['img_position_ids'][0]
    assert torch.equal(positions[7:9], positions[9:11])
    assert not torch.equal(original['img_position_ids'][0, 7:9, 0], original['img_position_ids'][0, 9:11, 0])
    assert aligned_owners == owners and torch.equal(aligned['x'], original['x'])


def test_cwm_shift_moves_text_and_first_frame_but_not_sink_or_current():
    session, frame = _packing_session()
    session.frame_aligned_rope = True
    session.static_rope_shift = 85 / 3
    shifted, _, _ = session._pack([frame], static=True, conditions=True, video=True,
        prefix=True, denoise=False, sigma=None)
    session.static_rope_shift = 0
    baseline, _, _ = session._pack([frame], static=True, conditions=True, video=True,
        prefix=True, denoise=False, sigma=None)
    diff = shifted['img_position_ids'][0] - baseline['img_position_ids'][0]
    torch.testing.assert_close(diff[:7, 0], torch.full((7,), 85 / 3, dtype=torch.float64))
    assert not diff[7:].any() and not diff[:, 1:].any()


def test_bootstrap_packs_dense_s_c_d_with_absolute_positions_and_qwen_tags():
    session, frame = _packing_session()
    packed, owners, shape = session._pack(
        [frame],
        static=True,
        conditions=True,
        video=True,
        prefix=False,
        denoise=True,
        sigma=torch.tensor(0.75),
    )
    assert owners == (-1,) * 7 + (5,) * 4
    assert packed["token_tags"].tolist() == [1, 0, 1] + [0] * 8
    assert packed["img_pos_for_infer_output_info"]["position_ids"].tolist() == [9, 10]
    assert packed["packed_seq_params"]["cu_seqlens_q_host"] == (0, 11)
    assert packed["x"].dtype == torch.float32
    assert torch.equal(packed["x"][0, 3:7], torch.full_like(packed["x"][0, 3:7], 0.999))
    times = packed["unique_timesteps"][packed["inverse_indices"]]
    assert torch.equal(times[:9], torch.full((9,), 0.999))
    assert torch.equal(times[9:], torch.full((2,), 0.25))
    assert shape == (1, 1, 2, 2)
    restored = minimax_h3_unpatchify_video_tokens(
        packed["x"][0, 9:11], latent_shape=shape, patch_size=(1, 2, 2)
    )
    assert torch.equal(restored[0], frame.video.bfloat16())


def test_continuation_has_no_dummy_text_and_sp_padding_never_has_an_owner():
    session, frame = _packing_session(2)
    packed, owners, _ = session._pack(
        [frame],
        static=False,
        conditions=True,
        video=False,
        prefix=False,
        denoise=False,
        sigma=None,
    )
    assert packed["x"].shape[1] == 4
    assert owners == (5, 5)
    assert packed["packed_seq_params"]["cu_seqlens_q_host"] == (0, 2, 4)
    assert packed["prompt_embeds"].shape == (0, 4)
    assert packed["text_pos_info"]["position_ids"].numel() == 0
    assert packed["token_tags"].tolist() == [0, 0, -1, -1]
    expected = 0.999 * frame.references[0].bfloat16().float()
    torch.testing.assert_close(
        packed["x"][0, :2].flatten().sort().values, expected.flatten().sort().values
    )


@pytest.mark.parametrize("fail_step", [None, 2])
def test_session_scopes_each_denoise_step_with_its_real_timestep_plan(fail_step):
    session, frame = _packing_session()
    calls = []

    class ForwardProbe:
        arch = session.model.arch
        blocks = (None,)
        adaln_cache = None

        def __call__(self, *, causal_context, **packed):
            context = get_forward_context()
            step = context.current_timestep
            assert context is not outer and context.attn_metadata is None
            calls.append(
                (step, packed["unique_timesteps"].clone(), causal_context.write)
            )
            if step == fail_step:
                raise RuntimeError("DiT failure")
            count = packed["img_pos_for_infer_output_info"]["position_ids"].numel()
            return torch.zeros(count, self.arch.latents_dim * 4), torch.empty(
                0, self.arch.audio_latents_dim
            )

    session.model = ForwardProbe()
    session.frames = {i: replace(frame, position=i) for i in range(2)}
    session.sigmas = torch.tensor([1, 36 / 37, 12 / 13, 4 / 5, 0], dtype=torch.float64)
    (plan,) = H3WindowPlanner(_config()).ready(2, eos=True)
    with set_forward_context(73, object()):
        outer = get_forward_context()
        for conditions in (True, False):
            session._forward(
                [frame],
                H3CausalKVCache(1),
                conditions=conditions,
                video=not conditions,
                write=True,
            )
            step, times, write = calls.pop()
            assert step == 0 and torch.equal(times, torch.tensor([0.999])) and write
            assert get_forward_context() is outer
        if fail_step is None:
            span = session._generate(plan)
            assert span.video.shape[1] == 2
        else:
            with pytest.raises(RuntimeError, match="DiT failure"):
                session._generate(plan)
        assert get_forward_context() is outer
    assert get_forward_context_or_none() is None
    expected_steps = list(range(4 if fail_step is None else fail_step + 1))
    assert [step for step, _, _ in calls] == expected_steps
    for step, times, write in calls:
        expected = torch.stack(
            (1 - session.sigmas[step].float(), torch.tensor(0.999))
        ).unique()
        assert torch.equal(times, expected) and not write


def test_interrupted_session_requires_reset_before_bootstrap_can_restart():
    session, frame = _packing_session()
    interruption = KeyboardInterrupt("interrupted bootstrap")

    class ForwardProbe:
        arch = session.model.arch
        blocks = (None,)
        adaln_cache = None
        interrupt = True

        def __call__(self, *, causal_context, **packed):
            if self.interrupt:
                self.interrupt = False
                raise interruption
            count = packed["img_pos_for_infer_output_info"]["position_ids"].numel()
            return torch.zeros(count, self.arch.latents_dim * 4), torch.empty(
                0, self.arch.audio_latents_dim
            )

    session.model = ForwardProbe()
    session.sigmas = torch.tensor([1.0, 0.0], dtype=torch.float64)
    session.reset()
    value = frame.noise.repeat(1, 5, 1, 1)
    source = H3SourceLatents(
        (value,), (torch.zeros_like(value),), value, torch.zeros_like(value)
    )
    assert session.append(source) == ()
    with pytest.raises(KeyboardInterrupt) as caught:
        session.append(source)
    assert caught.value is interruption
    assert session.planner.position == 9 and session.state == "failed"
    for submit in (session.append, session.finish):
        with pytest.raises(RuntimeError, match="reset first"):
            submit(source)

    session.reset()
    assert session.append(source) == ()
    (span,) = session.append(source)
    assert span.plan.bootstrap
    assert (span.plan.current.start, span.plan.current.stop) == (0, 9)
    assert span.video.shape[1] == 9


def test_recipe_and_tensor_boundaries_fail_before_computation():
    config = _config()
    with pytest.raises(ValueError, match="segment_boundaries"):
        replace(config, segment_boundaries=(70, 45))
    source = H3SourceLatents(
        (torch.zeros(2, 5, 2, 2),),
        (torch.zeros(2, 5, 2, 2),),
        torch.zeros(2, 5, 2, 2),
        torch.zeros(2, 5, 2, 2),
    )
    source.validate(5)
    with pytest.raises(ValueError, match="expected T"):
        source.validate(2)
