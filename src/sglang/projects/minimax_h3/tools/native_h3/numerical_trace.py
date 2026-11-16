# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch

STEP600_MERGED_SHA256 = (
    "131703bc616164df65c1290da1497c153a56d3905c6d025225252ca86ed5c802"
)


def tensor_digest(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy()
    ).hexdigest()


class CommitLedger:
    def __init__(self):
        self.blocks = []

    def commit(self, owners, layers, row_start):
        retained_owners = set(owners[:row_start])
        retained = [block for block in self.blocks if block[0] in retained_owners]
        if [owner for owner, count, _ in retained for _ in range(count)] != list(
            owners[:row_start]
        ):
            raise ValueError("retained KV owner ledger differs")
        cursor = 0
        for _, count, hashes in retained:
            for layer, pair in layers.items():
                actual = tuple(
                    tensor_digest(value[cursor : cursor + count]) for value in pair
                )
                if actual != hashes[layer]:
                    raise ValueError(
                        "C/H commit or retention changed historical KV bytes"
                    )
            cursor += count
        appended = []
        for owner, group in itertools.groupby(owners[row_start:]):
            count = sum(1 for _ in group)
            hashes = {
                layer: tuple(
                    tensor_digest(value[cursor : cursor + count]) for value in pair
                )
                for layer, pair in layers.items()
            }
            appended.append((owner, count, hashes))
            cursor += count
        if not appended:
            raise ValueError("empty C/H append")
        self.blocks = retained + appended


def committed_signature(owners, layers):
    return tuple(owners), tuple(
        (
            layer,
            str(value.dtype),
            tuple(value[: len(owners)].shape),
            tensor_digest(value[: len(owners)]),
        )
        for layer, pair in layers.items()
        for value in pair
    )


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu(item) for item in value]
    return value


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def plan_record(plan):
    return {
        "index": plan.index,
        "bootstrap": plan.bootstrap,
        "eos": plan.eos,
        **{
            name: [getattr(plan, name).start, getattr(plan, name).stop]
            for name in ("sink", "dynamic_recent", "current")
        },
    }


class TraceWriter:
    def __init__(self, output, rank, world_size):
        self.root = Path(output)
        self.path = self.root / f"rank{rank:04d}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.rank, self.world_size = rank, world_size
        self.files = {}

    def save(self, name, value, *, head_start=None):
        path = self.path / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            torch.save(cpu(value), handle)
        self.files[name] = {"sha256": digest(path), "head_start": head_start}

    def finish(self, identity, runtime):
        with (self.path / "complete.json").open("x", encoding="utf-8") as handle:
            json.dump(
                dict(
                    schema="h3-numerical/v1",
                    rank=self.rank,
                    world_size=self.world_size,
                    identity=identity,
                    runtime=runtime,
                    files=self.files,
                ),
                handle,
                indent=2,
            )


def load_run(root):
    root = Path(root)
    paths = sorted(root.glob("rank*"))
    if not paths:
        raise ValueError(f"empty trace: {root}")
    ranks = [json.loads((path / "complete.json").read_text()) for path in paths]
    world = ranks[0]["world_size"]
    if len(ranks) != world or [rank["rank"] for rank in ranks] != list(range(world)):
        raise ValueError("incomplete or duplicate rank set")
    events = {}
    for path, rank in zip(paths, ranks):
        if (
            rank["schema"] != "h3-numerical/v1"
            or rank["world_size"] != world
            or rank["identity"] != ranks[0]["identity"]
        ):
            raise ValueError("rank identity mismatch")
        if set(rank["files"]) != {
            str(p.relative_to(path).with_suffix("")).replace("\\", "/")
            for p in path.rglob("*.pt")
        }:
            raise ValueError("unregistered or missing trace file")
        for name, info in rank["files"].items():
            file = path / f"{name}.pt"
            if digest(file) != info["sha256"]:
                raise ValueError(f"trace hash mismatch: {file}")
            events.setdefault(name, []).append((file, info["head_start"]))
    if not events:
        raise ValueError("trace contains no events")
    return ranks, events


def difference(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return dict(
            exact=False,
            left_shape=list(left.shape),
            right_shape=list(right.shape),
            left_dtype=str(left.dtype),
            right_dtype=str(right.dtype),
        )
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    delta = (left.double() - right.double()).abs()
    exact = finite and torch.equal(
        left.contiguous().reshape(-1).view(torch.uint8),
        right.contiguous().reshape(-1).view(torch.uint8),
    )
    return dict(
        exact=exact,
        finite=finite,
        shape=list(left.shape),
        dtype=str(left.dtype),
        max_abs=(float(delta.max()) if delta.numel() else 0.0) if finite else None,
        mean_abs=(float(delta.mean()) if delta.numel() else 0.0) if finite else None,
        rms=(float(delta.square().mean().sqrt()) if delta.numel() else 0.0)
        if finite
        else None,
    )


def compare_values(left, right, prefix=""):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return {prefix: difference(left, right)}
    if (
        isinstance(left, dict)
        and isinstance(right, dict)
        and left.keys() == right.keys()
    ):
        return {
            name: result
            for key in left
            for name, result in compare_values(
                left[key], right[key], f"{prefix}/{key}"
            ).items()
        }
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return {
            name: result
            for i, (a, b) in enumerate(zip(left, right))
            for name, result in compare_values(a, b, f"{prefix}/{i}").items()
        }
    return {prefix: {"exact": type(left) is type(right) and left == right}}


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def validate_events(identity, events):
    sources = sorted(name for name in events if name.startswith("inputs/source"))
    if (
        sources != [f"inputs/source{i:04d}" for i in range(len(sources))]
        or len(sources) < 4
    ):
        raise ValueError("expected three normal source packets and EOS")
    required = {
        "inputs/static",
        "inputs/condition",
        "inputs/image",
        "encoder/condition",
        "encoder/image",
    }
    for i, source in enumerate(sources):
        eos = i == len(sources) - 1
        proxy = f"inputs/proxy{i:04d}"
        required.update((proxy, proxy.replace("inputs/", "encoder/")))
        if (
            proxy not in events
            or _load(events[source][0][0])["eos"] is not eos
            or _load(events[proxy][0][0])["eos"] is not eos
        ):
            raise ValueError("source and encoder EOS ledgers differ")
    if not required <= events.keys():
        raise ValueError("incomplete encoder or input capture")
    windows = sorted(
        name.rsplit("/", 1)[0] for name in events if name.endswith("/output")
    )
    if len(windows) < 3 or windows != [f"window{i:04d}" for i in range(len(windows))]:
        raise ValueError("trace must contain bootstrap, ordinary continuation and EOS")
    stop = 0
    for index, window in enumerate(windows):
        output = _load(events[f"{window}/output"][0][0])
        plan = output["plan"]
        if (
            plan["index"] != index
            or plan["bootstrap"] is not (index == 0)
            or plan["eos"] is not (index == len(windows) - 1)
            or plan["current"][0] != stop
            or plan["current"][1] <= stop
            or output["video"].shape[1] != plan["current"][1] - stop
        ):
            raise ValueError("invalid window output ledger")
        stop = plan["current"][1]
        expected = {f"{window}/d{i}" for i in range(4)}
        if {name for name in events if name.startswith(f"{window}/d")} != expected:
            raise ValueError(f"incomplete four-step window: {window}")
        writes = 0 if index == 0 else (4 if index == 1 else 2)
        expected = {
            f"{window}/write{write}/layer{layer:02d}"
            for write in range(writes)
            for layer in range(identity["layers"])
        }
        if {name for name in events if name.startswith(f"{window}/write")} != expected:
            raise ValueError(f"incomplete C/H commits: {window}")
    if stop != (len(sources) - 1) * 5 + 2:
        raise ValueError("source and generated latent counts differ")


def compare_runs(reference, native):
    refs, a = load_run(reference)
    candidates, b = load_run(native)
    if refs[0]["runtime"]["kind"] != "independent-reference":
        raise ValueError("oracle must be the independent reference capture")
    if candidates[0]["runtime"]["kind"] != "native":
        raise ValueError("candidate must be native replay")
    if refs[0]["identity"] != candidates[0]["identity"]:
        raise ValueError("reference and native corpus identities differ")
    for ranks, events in ((refs, a), (candidates, b)):
        validate_events(ranks[0]["identity"], events)
    results = {}
    for name in sorted(a.keys() | b.keys()):
        if name not in a or name not in b:
            results[name] = {"exact": False, "reason": "missing event"}
            continue
        left, right = a[name], b[name]
        if left[0][1] is None:
            if len(left) != 1 or len(right) != 1 or right[0][1] is not None:
                raise ValueError(f"replicated event must be recorded once: {name}")
            results.update(compare_values(_load(left[0][0]), _load(right[0][0]), name))
            continue
        for shards in (left, right):
            stop = 0
            for path, start in sorted(shards, key=lambda item: item[1]):
                value = _load(path)
                if (
                    start != stop
                    or value["key"].ndim != 3
                    or value["key"].shape != value["value"].shape
                    or value["key"].shape[0]
                    != len(value["owners"]) - value["row_start"]
                    or not 0 <= value["row_start"] < len(value["owners"])
                    or value["key"].shape[1] == 0
                    or value["key"].shape[2] != refs[0]["identity"]["head_dim"]
                ):
                    raise ValueError(f"invalid KV head partition: {name}")
                stop += value["key"].shape[1]
            if stop != refs[0]["identity"]["heads"]:
                raise ValueError(f"incomplete KV heads: {name}")
        for lp, ls in left:
            lv = _load(lp)
            for rp, rs in right:
                rv = _load(rp)
                start, stop = (
                    max(ls, rs),
                    min(ls + lv["key"].shape[1], rs + rv["key"].shape[1]),
                )
                if start >= stop:
                    continue
                if lv["owners"] != rv["owners"]:
                    results[f"{name}/heads{start}:{stop}/owners"] = {
                        "exact": False,
                        "reason": "KV owner order differs; values not compared",
                    }
                    continue
                la = {
                    "row_start": lv["row_start"],
                    "key": lv["key"][:, start - ls : stop - ls],
                    "value": lv["value"][:, start - ls : stop - ls],
                }
                rb = {
                    "row_start": rv["row_start"],
                    "key": rv["key"][:, start - rs : stop - rs],
                    "value": rv["value"][:, start - rs : stop - rs],
                }
                results.update(compare_values(la, rb, f"{name}/heads{start}:{stop}"))
    return dict(
        exact=bool(results) and all(item["exact"] for item in results.values()),
        results=results,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Strict CPU comparison; no tolerance or GPU execution"
    )
    parser.add_argument("reference")
    parser.add_argument("native")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = compare_runs(args.reference, args.native)
    with Path(args.report).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({"exact": report["exact"], "comparisons": len(report["results"])}))
    raise SystemExit(0 if report["exact"] else 1)


if __name__ == "__main__":
    main()
