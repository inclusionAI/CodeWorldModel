# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from sglang.multimodal_gen.configs.models.dits.minimax_h3_causal import (
    MiniMaxH3CausalDiTArchConfig,
)
from sglang.multimodal_gen.runtime.layers.kvcache.minimax_h3_causal import (
    H3CausalForward,
    H3CausalFusedForward,
    H3CausalKVCache,
)
from sglang.multimodal_gen.runtime.models.dits import (
    minimax_h3_adaln_cache as cache_module,
)
from sglang.multimodal_gen.runtime.models.dits import minimax_h3_causal as causal_module
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3AdalnProj,
    MiniMaxH3DiTModel,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_adaln_cache import (
    MiniMaxH3AdalnCache,
    _plan_key,
    native_adaln_weight_files,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_causal import (
    MiniMaxH3CausalAdalnProj,
    MiniMaxH3CausalDiTModel,
)
from test_h3_causal_fused import _session, _source
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_ARCH = MiniMaxH3CausalDiTArchConfig(num_layers=2, hidden_size=4, time_embed_dim=3)
_TIMES = torch.tensor([0.0, 0.999])


class _TimeEmbedder(torch.nn.Module):
    def forward(self, times):
        return torch.stack((times, times.square(), times + 1), dim=-1) + len(times) / 8


def _cache(tmp_path, monkeypatch, *, prefix="", max_plans=10):
    monkeypatch.setattr(cache_module, "get_tp_world_size", lambda: 1)
    tensors = {}
    for name, width in [
        *((f"blocks.{i}", 72) for i in range(2)),
        ("final_layer", 8),
    ]:
        tensors[f"{prefix}{name}.adaln_proj.linear.weight"] = (
            (torch.arange(width * 3).reshape(width, 3) % 13 - 6) / 16
        ).bfloat16()
        tensors[f"{prefix}{name}.adaln_proj.linear.bias"] = (
            torch.arange(width) / 32
        ).bfloat16()
    path = tmp_path / "model.safetensors"
    save_file(tensors, path)
    cache = MiniMaxH3AdalnCache(
        _ARCH, weight_files=[str(path)], max_plans=max_plans, max_plan_width=2
    )
    cache.load(torch.device("cpu"))
    return cache, tensors


def _model(cache):
    model = MiniMaxH3CausalDiTModel.__new__(MiniMaxH3CausalDiTModel)
    torch.nn.Module.__init__(model)
    model.adaln_cache = cache
    model.adaln_t_table = None
    model.time_embedder = _TimeEmbedder()
    model.blocks = (None, None)
    return model.eval()


@pytest.mark.parametrize("prefix,tp_size", [("", 1), ("dit.", 1), ("dit.", 2)])
def test_checkpoint_plans_match_runtime_shapes_and_tp_local_correction(
    tmp_path, monkeypatch, prefix, tp_size
):
    cache, tensors = _cache(tmp_path, monkeypatch, prefix=prefix)
    model = _model(cache)
    monkeypatch.setattr(cache_module, "get_tp_world_size", lambda: tp_size)
    monkeypatch.setattr(cache_module, "get_tp_rank", lambda: tp_size - 1)
    gathered = []

    def gather(value):
        gathered.append(value.clone())
        return torch.cat((value, value), dim=-1)

    monkeypatch.setattr(cache_module, "tensor_model_parallel_all_gather", gather)
    linear = torch.nn.functional.linear

    def shape_sensitive_linear(value, weight, bias):
        return linear(value, weight, bias) + len(value) / 4

    monkeypatch.setattr(torch.nn.functional, "linear", shape_sensitive_linear)
    with patch.object(
        cache,
        "_project_plans_from_checkpoint",
        wraps=cache._project_plans_from_checkpoint,
    ) as project:
        slots = model.prepare_adaln_plans(
            [_TIMES, _TIMES], fused_plan_indices=frozenset({1})
        )
        assert all(len(entry) == 3 for entry in project.call_args.args[0])
        model.prepare_adaln_plans([_TIMES, _TIMES], fused_plan_indices=frozenset({1}))
        assert project.call_count == 1 and slots[0] != slots[1]
    expected_local = []
    for fused, slot in zip((False, True), slots):
        with patch.object(
            causal_module, "h3_causal_fused_forward_active", return_value=fused
        ):
            inputs = torch.nn.functional.silu(model._time_embedding(_TIMES)).bfloat16()
            for layer in range(3):
                name = f"blocks.{layer}" if layer < 2 else "final_layer"
                weight = tensors[f"{prefix}{name}.adaln_proj.linear.weight"]
                bias = tensors[f"{prefix}{name}.adaln_proj.linear.bias"]
                if tp_size > 1:
                    weight, bias = weight.chunk(tp_size)[-1], bias.chunk(tp_size)[-1]
                cls = MiniMaxH3CausalAdalnProj if layer < 2 else MiniMaxH3AdalnProj
                projection = cls.__new__(cls)
                torch.nn.Module.__init__(projection)
                projection.linear = lambda value, weight=weight, bias=bias: (
                    shape_sensitive_linear(value, weight, bias),
                    None,
                )
                local = projection.project_local(inputs)
                expected_local.append((layer, fused, local))
                expected = torch.cat((local, local), dim=-1) if tp_size > 1 else local
                actual = (
                    cache.block_params[slot, :2, layer]
                    if layer < 2
                    else cache.final_params[slot, :2]
                )
                assert torch.equal(actual, expected)
    if tp_size > 1:
        expected_local.sort(key=lambda entry: (entry[0], entry[1]))
        assert len(gathered) == 6
        assert all(
            torch.equal(actual, expected[2])
            for actual, expected in zip(gathered, expected_local)
        )
    assert (
        native_adaln_weight_files(
            str(tmp_path), param_names_mapping=_ARCH.param_names_mapping
        )
        == cache.weight_files
    )


def test_direct_forward_resolves_policy_keys_and_missing_plans_fail(
    tmp_path, monkeypatch
):
    cache, _ = _cache(tmp_path, monkeypatch)
    model = _model(cache)
    slots = model.prepare_adaln_plans(
        [_TIMES, _TIMES], fused_plan_indices=frozenset({1})
    )
    monkeypatch.setattr(
        causal_module, "h3_causal_transaction", lambda context: nullcontext()
    )
    monkeypatch.setattr(model, "_h3_spectrum_requested", lambda **kwargs: False)
    monkeypatch.setattr(
        MiniMaxH3DiTModel, "forward", lambda self, **kwargs: kwargs["adaln_cache_slot"]
    )
    requests = (
        H3CausalForward(H3CausalKVCache(2), (0,)),
        H3CausalFusedForward(H3CausalKVCache(2), (0, 1, 1), 1, 2, 0),
    )
    with torch.no_grad():
        for expected, request in zip(slots, requests):
            actual = model(
                causal_context=request, x=torch.zeros(1), unique_timesteps=_TIMES
            )
            assert actual == expected
        with pytest.raises(ValueError, match="does not cover"):
            model(
                causal_context=requests[0],
                x=torch.zeros(1),
                unique_timesteps=torch.tensor([0.5]),
            )


def test_ordinary_cache_default_and_invalid_fused_plans(tmp_path, monkeypatch):
    cache, _ = _cache(tmp_path, monkeypatch, max_plans=1)
    model = _model(cache)
    MiniMaxH3DiTModel.prepare_adaln_plans(model, [_TIMES])
    original = cache.block_params.clone()
    with pytest.raises(ValueError, match="slab holds"):
        model.prepare_adaln_plans([_TIMES, _TIMES], fused_plan_indices=frozenset({1}))
    with pytest.raises(ValueError, match="two timesteps"):
        model.prepare_adaln_plans([_TIMES[:1]], fused_plan_indices=frozenset({0}))
    assert torch.equal(cache.block_params, original) and cache.generation == 0
    model.adaln_cache = torch.nn.Module()
    model.adaln_cache.weight_files = None
    with pytest.raises(ValueError, match="online checkpoint"):
        model.prepare_adaln_plans([_TIMES], fused_plan_indices=frozenset({0}))


@pytest.mark.parametrize("failed_build", [False, True])
def test_eviction_and_invalidation_refresh_session_slots(
    tmp_path, monkeypatch, failed_build
):
    cache, tensors = _cache(tmp_path, monkeypatch, max_plans=6)
    model = _model(cache)
    session = _session(fused=True)
    session.model = model
    session.reset()
    generation = cache.generation
    if failed_build:
        with (
            patch.object(
                cache,
                "_project_plans_from_checkpoint",
                side_effect=RuntimeError("read failed"),
            ),
            pytest.raises(RuntimeError, match="read failed"),
        ):
            model.prepare_adaln_plans([torch.tensor([0.7])])
    else:
        model.prepare_adaln_plans([torch.tensor([0.7])])
    assert cache.generation > generation
    # A window refresh runs before its first H/C/DiT operation.
    original = session._prepare_adaln_plans
    with patch.object(session, "_prepare_adaln_plans", wraps=original) as prepare:
        session._range = lambda value: (_ for _ in ()).throw(
            RuntimeError("window entered")
        )
        with pytest.raises(RuntimeError, match="window entered"):
            session._generate(SimpleNamespace(current=None))
        assert prepare.call_count == 1
    assert session._adaln_generation == cache.generation
    slots = session._adaln_slots
    for step, sigma in enumerate(session.sigmas[:-1]):
        expected = torch.stack((1 - sigma.float(), torch.tensor(0.999))).unique()
        assert torch.equal(
            cache.plan_timesteps[slots[1][step], : len(expected)], expected
        )
    generation = cache.generation
    before = cache.block_params[slots[0]].clone()
    save_file(
        {name: value + 1 for name, value in tensors.items()}, cache.weight_files[0]
    )
    cache.invalidate()
    assert cache.generation == generation + 1
    session.reset()
    assert session._adaln_generation == cache.generation
    assert not torch.equal(cache.block_params[session._adaln_slots[0]], before)


@pytest.mark.parametrize(
    "world_size,fused", [(1, False), (1, True), (4, False), (4, True)]
)
def test_session_reuses_complete_inventory_and_selects_actual_packed_plans(
    tmp_path, monkeypatch, world_size, fused
):
    cache, _ = _cache(tmp_path, monkeypatch)
    model = _model(cache)
    session = _session(fused=fused)
    session.ulysses_world_size = world_size
    session.model = model
    model.arch = SimpleNamespace(latents_dim=2, audio_latents_dim=3)
    calls = []

    def forward(*, causal_context, **packed):
        times, slot = packed["unique_timesteps"], packed["adaln_cache_slot"]
        assert torch.equal(cache.plan_timesteps[slot, : len(times)], times)
        key = _plan_key(times)
        if isinstance(causal_context, H3CausalFusedForward):
            key = (-1, *key)
            assert times[0] == 0
        assert cache._slots[key] == slot
        calls.append(causal_context)
        rows = len(packed["img_pos_for_infer_output_info"]["position_ids"])
        return torch.zeros(rows, 8), None

    monkeypatch.setattr(model, "forward", forward)
    session.reset()
    assert len(cache._slots) == (9 if world_size == 1 else 5) + fused
    generator = torch.Generator().manual_seed(17)
    with patch.object(
        cache,
        "_project_plans_from_checkpoint",
        wraps=cache._project_plans_from_checkpoint,
    ) as project:
        for count in [5] * 15 + [2]:
            (session.finish if count == 2 else session.append)(
                _source(count, generator)
            )
        assert session.state == "ended" and not session.pending_history
        session.reset()
        other = _session(fused=fused)
        other.model, other.ulysses_world_size = model, world_size
        other.reset()
        assert project.call_count == 0
    assert any(isinstance(call, H3CausalFusedForward) for call in calls) == fused
