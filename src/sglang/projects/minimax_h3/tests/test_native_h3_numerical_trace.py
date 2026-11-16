# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/native_h3/numerical_trace.py"
)
_SPEC = importlib.util.spec_from_file_location("h3_numerical_trace", _PATH)
trace = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(trace)


def make_run(path, kind, world, *, offset=0, omit=None):
    identity = dict(heads=4, head_dim=2, layers=2, artifact_sha256="fixture", recipe={})
    for rank in range(world):
        writer = trace.TraceWriter(path, rank, world)
        if rank == 0:
            writer.save("inputs/static", torch.zeros(1))
            writer.save("inputs/condition", "prompt")
            writer.save("inputs/image", torch.zeros(1))
            writer.save("encoder/condition", torch.zeros(1))
            writer.save("encoder/image", torch.zeros(1))
            for i in range(4):
                writer.save(f"inputs/source{i:04d}", dict(eos=i == 3))
                writer.save(f"inputs/proxy{i:04d}", dict(eos=i == 3))
                writer.save(f"encoder/proxy{i:04d}", torch.zeros(1))
            for window in range(3):
                for step in range(4):
                    name = f"window{window:04d}/d{step}"
                    if name != omit:
                        writer.save(
                            name,
                            dict(
                                xt=torch.arange(4.0),
                                sigma=torch.tensor(0.5),
                                x0=torch.arange(4.0) + offset,
                            ),
                        )
                start, stop = (0, 9, 14)[window], (9, 14, 17)[window]
                writer.save(
                    f"window{window:04d}/output",
                    dict(
                        plan=dict(
                            index=window,
                            bootstrap=window == 0,
                            eos=window == 2,
                            current=[start, stop],
                        ),
                        video=torch.zeros(1, stop - start, 2, 2),
                    ),
                )
        for window, writes in ((1, 4), (2, 2)):
            for write in range(writes):
                for layer in range(2):
                    value = torch.arange(16.0).reshape(2, 4, 2)[
                        :, rank * (4 // world) : (rank + 1) * (4 // world)
                    ]
                    writer.save(
                        f"window{window:04d}/write{write}/layer{layer:02d}",
                        dict(owners=[-1, 0], row_start=0, key=value, value=value + 1),
                        head_start=rank * (4 // world),
                    )
        writer.finish(identity, dict(kind=kind))


def test_compare_different_head_partitions(tmp_path):
    make_run(tmp_path / "reference", "independent-reference", 2)
    make_run(tmp_path / "native", "native", 4)
    result = trace.compare_runs(tmp_path / "reference", tmp_path / "native")
    assert result["exact"]
    assert any("heads3:4/key" in name for name in result["results"])


def test_numerical_error_is_reported_without_tolerance(tmp_path):
    make_run(tmp_path / "reference", "independent-reference", 1)
    make_run(tmp_path / "native", "native", 1, offset=0.25)
    result = trace.compare_runs(tmp_path / "reference", tmp_path / "native")
    assert not result["exact"]
    assert result["results"]["window0000/d0/x0"]["max_abs"] == 0.25


def test_incomplete_denoise_cannot_pass(tmp_path):
    make_run(tmp_path / "reference", "independent-reference", 1, omit="window0000/d3")
    make_run(tmp_path / "native", "native", 1, omit="window0000/d3")
    with pytest.raises(ValueError, match="incomplete four-step"):
        trace.compare_runs(tmp_path / "reference", tmp_path / "native")


def test_interrupted_rank_and_changed_payload_fail(tmp_path):
    writer = trace.TraceWriter(tmp_path, 0, 1)
    writer.save("value", torch.zeros(1))
    with pytest.raises(FileNotFoundError):
        trace.load_run(tmp_path)
    writer.finish(dict(heads=4), dict(kind="native"))
    torch.save(torch.ones(1), writer.path / "value.pt")
    with pytest.raises(ValueError, match="hash mismatch"):
        trace.load_run(tmp_path)
    with pytest.raises(FileExistsError):
        trace.TraceWriter(tmp_path, 0, 1)


def test_nonfinite_signed_zero_scalar_and_dtype():
    assert trace.difference(torch.tensor(1.0), torch.tensor(1.0))["exact"]
    assert not trace.difference(torch.tensor(-0.0), torch.tensor(0.0))["exact"]
    assert not trace.difference(torch.ones(1), torch.ones(1).bfloat16())["exact"]
    result = trace.difference(torch.tensor(float("nan")), torch.tensor(float("nan")))
    assert not result["exact"] and not result["finite"]
    json.dumps(result, allow_nan=False)


def test_kv_owner_order_and_head_holes_fail(tmp_path):
    make_run(tmp_path / "reference", "independent-reference", 1)
    make_run(tmp_path / "native", "native", 1)
    root = tmp_path / "native/rank0000"
    manifest = json.loads((root / "complete.json").read_text())
    path = root / "window0001/write0/layer00.pt"
    value = torch.load(path, weights_only=True)
    value["owners"] = [0, -1]
    torch.save(value, path)
    manifest["files"]["window0001/write0/layer00"]["sha256"] = trace.digest(path)
    (root / "complete.json").write_text(json.dumps(manifest))
    assert not trace.compare_runs(tmp_path / "reference", tmp_path / "native")["exact"]
    manifest["files"]["window0001/write0/layer00"]["head_start"] = 1
    (root / "complete.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid KV head partition"):
        trace.compare_runs(tmp_path / "reference", tmp_path / "native")


@pytest.mark.parametrize(
    "prefix, message",
    [("encoder/", "incomplete encoder"), ("window0002/write1/", "incomplete C/H")],
)
def test_shared_omissions_cannot_pass(tmp_path, prefix, message):
    for name, kind in (("reference", "independent-reference"), ("native", "native")):
        make_run(tmp_path / name, kind, 1)
        root = tmp_path / name / "rank0000"
        manifest = json.loads((root / "complete.json").read_text())
        for event in list(manifest["files"]):
            if event.startswith(prefix):
                (root / f"{event}.pt").unlink()
                del manifest["files"][event]
        (root / "complete.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        trace.compare_runs(tmp_path / "reference", tmp_path / "native")


def test_readonly_signature_ignores_arena_tail_and_detects_history_write():
    key, value = torch.arange(24.0).reshape(3, 4, 2), torch.zeros(3, 4, 2)
    before = trace.committed_signature([-1, 0], {0: (key, value)})
    key[2].fill_(99)
    assert (
        trace.committed_signature([-1, 0], {0: (key.clone(), value.clone())}) == before
    )
    value[0].fill_(1)
    assert trace.committed_signature([-1, 0], {0: (key, value)}) != before


def test_append_ledger_proves_retained_bytes_and_rejects_mutation():
    ledger = trace.CommitLedger()
    owners = [-1, -1, 0, 0]
    key = torch.arange(16.0).reshape(4, 2, 2)
    ledger.commit(owners, {0: (key, key + 1)}, 0)
    key = torch.cat((key, torch.ones(2, 2, 2)))
    owners += [1, 1]
    ledger.commit(owners, {0: (key, key + 1)}, 4)
    key = torch.cat((key[[0, 1, 4, 5]], torch.zeros(1, 2, 2)))
    owners = [-1, -1, 1, 1, 2]
    ledger.commit(owners, {0: (key, key + 1)}, 4)
    key = torch.cat((key, torch.ones(1, 2, 2)))
    key[0].fill_(99)
    with pytest.raises(ValueError, match="historical KV bytes"):
        ledger.commit(owners + [2], {0: (key, key + 1)}, 5)


@pytest.mark.parametrize("offset, code", [(0, 0), (0.25, 1)])
def test_cli_exit_and_report(tmp_path, offset, code):
    make_run(tmp_path / "reference", "independent-reference", 1)
    make_run(tmp_path / "native", "native", 1, offset=offset)
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_PATH),
            str(tmp_path / "reference"),
            str(tmp_path / "native"),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == code, result.stderr
    assert json.loads(report.read_text())["exact"] is (code == 0)
