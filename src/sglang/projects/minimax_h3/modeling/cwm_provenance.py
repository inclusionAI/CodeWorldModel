"""Content identity for frozen CWM teachers, carried alongside every DCP.

Hashing is rank-zero-only and cached by filesystem change stamps. A stamp is
an invalidation key, never the identity: changing a file triggers a new SHA256
over its bytes. This avoids rereading a 33B teacher on every restart. The
persistence hooks run before model construction and know the resolved run
directory, including launch-time config overrides.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

IDENTITY_FILE = "cwm_teacher_identity.json"


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        yield


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, sort_keys=True, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _stamp(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _file_identity(path, cache_dir):
    key = hashlib.sha256(str(path).encode()).hexdigest()
    cached_path = cache_dir / (key + ".json")
    with _lock(cached_path):
        before = _stamp(path)
        try:
            cached = json.loads(cached_path.read_text())
            if cached["stamp"] == before and cached["path"] == str(path):
                if len(cached["sha256"]) == 64:
                    return cached
        except (OSError, ValueError, KeyError, TypeError):
            pass
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if before != _stamp(path):
            raise ValueError(f"teacher changed while hashing: {path}")
        value = dict(path=str(path), stamp=before, sha256=digest.hexdigest())
        _write_json(cached_path, value)
        return value


def source_identity(path, cache_dir):
    """Hash a single runtime export or all shards named by a safetensors index."""
    path, cache_dir = Path(path).expanduser().resolve(), Path(cache_dir).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"expected a teacher weight file, not a DCP directory: {path}")
    files = [_file_identity(path, cache_dir)]
    if path.name.endswith(".safetensors.index.json"):
        index = json.loads(path.read_text())
        shards = sorted(set(index["weight_map"].values()))
        if not shards:
            raise ValueError(f"empty teacher index: {path}")
        files += [_file_identity((path.parent / shard).resolve(), cache_dir) for shard in shards]
    # Include the index itself: tensor-to-shard mapping affects loaded weights.
    digest = (files[0]["sha256"] if len(files) == 1 else hashlib.sha256(
        json.dumps([file["sha256"] for file in files], separators=(",", ":")).encode()).hexdigest())
    if any(file["stamp"] != _stamp(Path(file["path"])) for file in files):
        raise ValueError("teacher changed while collecting its shard identities")
    return dict(schema="cwm_teacher_identity_v1", source=str(path), sha256=digest, files=files)


def _rank_zero(call):
    # Lazy import keeps configuration generation independent of CUDA/PyTorch.
    import torch
    import torch.distributed as dist

    distributed = dist.is_available() and dist.is_initialized()
    result = [None]
    if not distributed or dist.get_rank() == 0:
        try:
            result[0] = dict(value=call())
        except Exception as exc:
            result[0] = dict(error=f"{type(exc).__name__}: {exc}")
    if distributed:
        device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
        dist.broadcast_object_list(result, src=0, device=device)
    if "error" in result[0]:
        raise ValueError("CWM teacher provenance check failed: " + result[0]["error"])
    return result[0]["value"]


def _same(record, current, location):
    if record.get("schema") != current["schema"] or record.get("sha256") != current["sha256"]:
        raise ValueError(f"frozen teacher differs from {location}; use the original teacher or start a new run")


class CWMTeacherProvenance:
    """Persistence plugin: bind teacher bytes before loading and verify DCP resume."""

    def __init__(self, config):
        self.cache_dir = config.get("hash_cache")
        self.identity = None

    def after_run_dir_ready(self, *, config, run_dir, **kwargs):
        run_dir = Path(run_dir)
        cache = self.cache_dir or run_dir.parent.parent / ".cwm_weight_hashes"
        self.cache_dir = Path(cache)
        source = config.models.tea_model.weight.path

        def bind():
            current = source_identity(source, self.cache_dir)
            expected = config.get("lineage", {}).get("teacher_sha256")
            if expected is not None and expected != current["sha256"]:
                raise ValueError("teacher bytes differ from the generated configuration's SHA256")
            destination = run_dir / IDENTITY_FILE
            with _lock(destination):
                if destination.exists():
                    _same(json.loads(destination.read_text()), current, destination)
                else:
                    _write_json(destination, current)
            return current

        self.identity = _rank_zero(bind)
        config.setdefault("lineage", {})["teacher_sha256"] = self.identity["sha256"]

    def before_checkpoint_saved(self, *, checkpoint_dir, **kwargs):
        def save():
            if self.identity is None:
                raise ValueError("teacher identity was not initialized")
            # Usually only stat calls; a changed source is rehashed before save.
            current = source_identity(self.identity["source"], self.cache_dir)
            _same(current, self.identity, "the loaded teacher")
            _write_json(Path(checkpoint_dir) / IDENTITY_FILE, self.identity)
        _rank_zero(save)

    def before_checkpoint_loaded(self, *, checkpoint_dir, **kwargs):
        def verify():
            path = Path(checkpoint_dir) / IDENTITY_FILE
            if not path.is_file():
                raise ValueError(f"{path} missing; this checkpoint cannot prove its frozen teacher identity")
            _same(json.loads(path.read_text()), self.identity, path)
        _rank_zero(verify)
