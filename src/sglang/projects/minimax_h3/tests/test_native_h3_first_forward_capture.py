# SPDX-License-Identifier: Apache-2.0
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / "tools/native_h3"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "first_forward_capture", SCRIPTS / "first_forward_capture.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class TupleLinear(nn.Linear):
    def forward(self, value):
        return super().forward(value), None


class Time(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = TupleLinear(4, 8)

    def forward(self, value):
        return self.proj_in(value[:, None].expand(-1, 4))[0]


class AdaLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = TupleLinear(8, 36)

    def project_local(self, value):
        return self.linear(value)[0]

    def split_output(self, value):
        return value.chunk(6, dim=-1)

    def forward(self, value):
        return self.split_output(self.project_local(value))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.head_dim = 4, 2
        self.qkv_proj = TupleLinear(6, 24, bias=False)
        self.out_proj = TupleLinear(8, 6, bias=False)

    def forward(self, value):
        qkv = self.qkv_proj(value)[0]
        qkv.fill_(999)
        return value


class Block(nn.Module):
    def __init__(self, fused=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(6)
        self.attn = Attention()
        self.adaln_proj = AdaLN()
        self.mlp = nn.Module()
        self.mlp.fc1 = TupleLinear(6, 16, bias=False)
        self.mlp.fc2 = TupleLinear(8, 6, bias=False)
        self.fused = fused

    def forward(self, value, *, adaln_input, adaln_params=None, **kwargs):
        if adaln_params is None:
            self.adaln_proj(adaln_input)
        norm = (
            nn.functional.layer_norm(value, (6,)) if self.fused else self.norm1(value)
        )
        norm.add_(3)
        self.attn(norm)
        return value.add_(77)


class Model(nn.Module):
    def __init__(self, fused=False, batched_adaln=False):
        super().__init__()
        self.condition_proj = TupleLinear(5, 6)
        self.token_refiner = nn.Identity()
        self.time_embedder = Time()
        self.video_patch_proj = TupleLinear(4, 6)
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq", torch.arange(2.0))
        self.blocks = nn.ModuleList([Block(fused)])
        self.batched_adaln = batched_adaln

    def forward(self, **kwargs):
        text = self.token_refiner(self.condition_proj(kwargs["prompt_embeds"])[0])
        video = self.video_patch_proj(kwargs["x"][0, [1, 3, 5, 7]])[0]
        value = torch.zeros(8, 6)
        value[[0, 2, 4, 6]], value[[1, 3, 5, 7]] = text, video
        adaln_input = self.time_embedder(kwargs["unique_timesteps"])
        adaln_params = None
        if self.batched_adaln:
            projection = self.blocks[0].adaln_proj
            adaln_params = projection.split_output(
                projection.project_local(adaln_input)
            )
        return self.blocks[0](
            value,
            adaln_input=adaln_input,
            adaln_params=adaln_params,
            combined_indices=torch.zeros(8, dtype=torch.long),
            rope_cache=(torch.zeros(8, 2), torch.arange(8)),
        )


def inputs():
    return dict(
        x=torch.arange(32.0).reshape(1, 8, 4),
        audio_x=torch.zeros(1, 8, 2),
        img_position_ids=torch.zeros(1, 8, 3),
        unique_timesteps=torch.tensor([0.0, 0.999]),
        inverse_indices=torch.zeros(8, dtype=torch.long),
        token_tags=torch.zeros(8, dtype=torch.long),
        prompt_embeds=torch.ones(4, 5),
        update_mask=torch.ones(4, dtype=torch.bool),
        img_pos_info={"position_ids": torch.tensor([1, 3, 5, 7])},
        text_pos_info={"position_ids": torch.tensor([0, 2, 4, 6])},
        audio_pos_info={"position_ids": torch.empty(0, dtype=torch.long)},
        img_pos_for_infer_output_info={"position_ids": torch.tensor([1, 3, 5, 7])},
        sample_lens=[8],
        refiner_packed_seq_params={
            "cu_seqlens_q": torch.tensor([0, 4]),
            "max_seqlen_q": 4,
        },
    )


def fixture_runtime(kind, model):
    return dict(kind=kind, norm_modulate_fused=model.blocks[0].fused)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("batched_adaln", [False, True])
def test_real_hooks_snapshot_before_inplace_and_remove_on_exit(
    tmp_path, monkeypatch, fused, batched_adaln
):
    monkeypatch.setattr(probe, "runtime_record", fixture_runtime)
    model = Model(fused, batched_adaln).eval()
    control = copy.deepcopy(model)
    writer = probe.TraceWriter(tmp_path, 0, 1)
    with (
        torch.no_grad(),
        probe.FirstForwardCapture(model, writer, kind="reference") as capture,
    ):
        actual = model(**inputs())
    assert torch.equal(actual, control(**inputs()))
    state = capture.receipt()
    assert state["norm1_observed"] is not fused
    packed = probe._load(tmp_path / "rank0000/packed/x.pt")
    assert torch.equal(packed["value"], inputs()["x"])
    captured_input = probe._load(tmp_path / "rank0000/block0/input.pt")["value"]
    assert torch.equal(actual, captured_input + 77)
    raw_q = probe._load(tmp_path / "rank0000/block0/raw_q.pt")["value"]
    assert not bool((raw_q == 999).all())
    modulated = probe._load(tmp_path / "rank0000/block0/modulated.pt")["value"]
    expected_q = model.blocks[0].attn.qkv_proj(modulated)[0][:, :8].reshape(8, 4, 2)
    assert torch.equal(raw_q, expected_q)
    if not fused:
        norm = probe._load(tmp_path / "rank0000/block0/norm1.pt")["value"]
        assert torch.equal(modulated, norm + 3)
    assert all(
        not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()
    )
    adaln_input = probe._load(tmp_path / "rank0000/block0/adaln_input.pt")["value"]
    tables = control.blocks[0].adaln_proj(adaln_input)
    for name, table in zip(
        ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"),
        tables,
    ):
        assert torch.equal(
            probe._load(tmp_path / f"rank0000/block0/adaln/{name}.pt")["value"], table
        )
    capture.events.remove("block0/adaln/gate_msa")
    with pytest.raises(ValueError, match="missing=.*block0/adaln/gate_msa"):
        capture.receipt()


def test_hook_cleanup_after_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "runtime_record", fixture_runtime)
    model = Model()
    writer = probe.TraceWriter(tmp_path, 0, 1)
    with pytest.raises(RuntimeError, match="test failure"):
        with probe.FirstForwardCapture(model, writer, kind="reference"):
            raise RuntimeError("test failure")
    assert all(
        not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()
    )
    assert not (tmp_path / "rank0000/complete.json").exists()


def record(value, *axes):
    return dict(value=value, axes=[dict(axis=a, start=s, total=t) for a, s, t in axes])


def test_different_up_and_tp_reconstruct_same_tensor():
    full = torch.arange(16 * 8 * 2).reshape(16, 8, 2)
    reference = [
        record(part, (0, rank * 2, 16)) for rank, part in enumerate(full.chunk(8))
    ]
    native = [
        record(
            full[u * 4 : (u + 1) * 4, t * 4 : (t + 1) * 4],
            (0, u * 4, 16),
            (1, t * 4, 8),
        )
        for t in range(2)
        for u in range(4)
    ]
    assert torch.equal(probe.assemble(reference), full)
    assert torch.equal(probe.assemble(native), full)


def test_missing_shard_and_disagreeing_replica_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        probe.assemble([record(torch.ones(2, 4), (0, 0, 4))])
    with pytest.raises(ValueError, match="replicas"):
        probe.assemble([record(torch.ones(4)), record(torch.zeros(4))])
    with pytest.raises(ValueError, match="replicas"):
        probe.assemble([record(torch.tensor([0.0])), record(torch.tensor([-0.0]))])
    assert probe.assemble([record(torch.empty(0, dtype=torch.long))]).shape == (0,)
    assert probe.assemble([record(torch.tensor(3))]).item() == 3


def test_actual_weight_capture_splits_qkv_and_gate_up(tmp_path):
    model = Model()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.arange(p.numel()).reshape(p.shape))
    reference = probe.TraceWriter(tmp_path / "reference", 0, 1)
    probe.FirstForwardCapture(model, reference, kind="reference").weights()
    names = set(reference.files)
    pieces = {name: [] for name in names}
    for tp in range(2):
        native = copy.deepcopy(model)
        native.blocks[0].attn.num_heads = 2
        for name, value in list(native.blocks[0].named_parameters()):
            parent, _, leaf = name.rpartition(".")
            if name.startswith("attn.qkv_proj."):
                shard = value.reshape(3, 4, 2, -1)[:, tp * 2 : (tp + 1) * 2].reshape(
                    -1, 6
                )
            elif name == "mlp.fc1.weight":
                shard = torch.cat([part.chunk(2)[tp] for part in value.chunk(2)])
            elif name in ("attn.out_proj.weight", "mlp.fc2.weight"):
                shard = value.chunk(2, dim=1)[tp]
            elif name.startswith("adaln_proj.linear."):
                shard = value.chunk(2)[tp]
            else:
                shard = value
            setattr(
                native.blocks[0].get_submodule(parent),
                leaf,
                nn.Parameter(shard.clone()),
            )
        writer = probe.TraceWriter(tmp_path / "native", tp, 2)
        probe.FirstForwardCapture(
            native, writer, kind="native", tp_rank=tp, tp_size=2
        ).weights()
        assert set(writer.files) == names
        for name in names:
            pieces[name].append(probe._load(writer.path / f"{name}.pt"))
    for name in names:
        full = probe._load(reference.path / f"{name}.pt")["value"]
        assert torch.equal(probe.assemble(pieces[name]), full), name


def test_comparator_reports_finite_nonzero_error_without_tolerance(tmp_path):
    for kind, delta in (("reference", 0), ("native", 0.25)):
        writer = probe.TraceWriter(tmp_path / kind, 0, 1)
        for name in probe.REQUIRED_STAGES:
            writer.save(
                name,
                record(torch.arange(4.0) + (delta if name == "block0/raw_q" else 0)),
            )
        writer.finish(dict(probe="bootstrap-d0-block0/v1"), dict(kind=kind, up_rank=0))
    report = tmp_path / "report.json"
    assert probe.compare(tmp_path / "reference", tmp_path / "native", report) == 1
    result = json.loads(report.read_text())["results"]["block0/raw_q"]
    assert result["finite"] and result["max_abs"] == 0.25 and not result["exact"]
    with pytest.raises(FileExistsError):
        probe.compare(tmp_path / "reference", tmp_path / "native", report)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "first_forward_capture.py"),
            "compare",
            str(tmp_path / "reference"),
            str(tmp_path / "native"),
            "--report",
            str(tmp_path / "cli-report.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["exact"] is False


def test_missing_stage_is_not_numerical_success(tmp_path):
    for kind in ("reference", "native"):
        writer = probe.TraceWriter(tmp_path / kind, 0, 1)
        for name in probe.REQUIRED_STAGES - {"block0/raw_q"}:
            writer.save(name, record(torch.zeros(1)))
        writer.finish(dict(probe="bootstrap-d0-block0/v1"), dict(kind=kind, up_rank=0))
    with pytest.raises(ValueError, match="incomplete"):
        probe.compare(
            tmp_path / "reference", tmp_path / "native", tmp_path / "report.json"
        )


def test_capture_help_needs_no_native_or_reference_imports():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "first_forward_capture.py"),
            "capture",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--artifact" in result.stdout and "--bundle" in result.stdout
