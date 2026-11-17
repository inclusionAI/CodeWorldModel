# SPDX-License-Identifier: Apache-2.0
"""Real CPU process/RPC checks; no native model or GPU execution is mocked."""

import json
import multiprocessing as mp
import pickle
import socket
import struct
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=180, suite="base-a-test-cpu")

# Import the stdlib-only host without activating SGLang's GPU runtime on Windows.
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[4]
        / "python/sglang/multimodal_gen/runtime/managers"
    ),
)
from minimax_h3_causal_host import (
    H3CausalHost,
    _PrefetchedSource,
    _validate_causal_parallelism,
)


@dataclass
class _RGB:
    rgb: bytes


@dataclass(frozen=True)
class _Stamp:
    source_frame_id: str
    pts_ms: float
    duration_ms: float


@dataclass(frozen=True)
class _Packet:
    rgb: bytes
    frames: tuple
    width: int = 64
    height: int = 64


@dataclass(frozen=True)
class _Source:
    input_seq: int
    packet: object
    eos: bool = False


class _CPUComponents:
    def synchronize(self):
        pass


class _CPUBridge:
    def __init__(self, context, directory):
        self.context, self.directory = context, Path(directory)
        self.components = _CPUComponents()
        self.identity, self.index = None, 0
        self.computed = None

    def _record(self, method):
        with (self.directory / f"rank-{self.context.rank}.jsonl").open("a") as stream:
            stream.write(json.dumps([method, self.identity]) + "\n")

    def _stage(self, name):
        if self.context.rank == 0:
            time.sleep(0.025)
        (self.directory / f"{self.index}-{name}-{self.context.rank}").touch()
        self.components.synchronize()
        assert all(
            (self.directory / f"{self.index}-{name}-{rank}").exists()
            for rank in range(self.context.world_size)
        ), "rank crossed an unfinished GPU stage"

    def open(self, identity, request):
        assert self.identity is None
        self.identity = identity
        self._record("open")
        self._stage("condition")

    def append(self, identity, packet, cancelled, emit):
        assert identity == self.identity
        self._record("append")
        if packet == "crash" and self.context.rank == 1:
            raise RuntimeError("intentional rank failure")
        if packet == "exit" and self.context.rank == 1:
            import os

            os._exit(7)
        if packet == "stall":
            time.sleep(5)
        if packet == "wait_cancel":
            assert cancelled.wait(timeout=5)
        if cancelled.is_set():
            raise RuntimeError("session cancelled")
        for name in ("proxy", "dit", "decode"):
            self._stage(name)
        emit(_RGB(f"rank-{self.context.rank}:{packet}".encode()))
        self.index += 1

    def finish(self, identity, cancelled, emit):
        assert identity == self.identity
        self._record("finish")
        self._stage("eos")
        emit(_RGB(f"rank-{self.context.rank}:eos".encode()))

    def compute(self, identity, source, cancelled):
        assert identity == self.identity and self.computed is None
        assert not cancelled.is_set()
        self._record('compute')
        self._stage('dit')
        self.computed = source

    def deliver(self, identity, next_source, cancelled, emit):
        assert identity == self.identity and self.computed is not None
        self._record('deliver')
        kind = 'decode' if self.context.rank < 2 else 'encode' if self.context.rank < 4 else 'idle'
        (self.directory / f'{self.index}-{kind}-{self.context.rank}').touch()
        self._stage('codec_join')
        if self.context.world_size == 8:
            assert all((self.directory / f'{self.index}-{kind}-{rank}').exists()
                for kind, ranks in (('decode', (0, 1)), ('encode', (2, 3))) for rank in ranks)
        if cancelled.is_set():
            raise RuntimeError('session cancelled')
        packet = self.computed.packet
        emit(
            _RGB(
                packet.rgb
                if hasattr(packet, "rgb")
                else f"rank-{self.context.rank}:{packet}".encode()
            )
        )
        self.computed = None
        self.index += 1

    def close(self, identity):
        assert identity == self.identity
        self._record("close")
        self.identity = None
        self.computed = None
        self.index += 1


def build_cpu_bridge(context, *, directory, fail_rank=None):
    if context.rank == fail_rank:
        raise RuntimeError("intentional factory failure")
    assert context.server_args is None
    return _CPUBridge(context, directory)


class _LookaheadCPUBridge(_CPUBridge):
    def __init__(self, context, directory):
        super().__init__(context, directory)
        self.prepared, self.tail, self.output_rgb = None, None, None

    def compute(self, identity, source, cancelled):
        if type(source.input_seq) is not int or type(source.eos) is not bool:
            raise ValueError("stale or invalid native H3 source identity")
        if self.prepared is not None and self.prepared != source:
            raise ValueError(
                "prepared H3 proxy does not match the current source identity"
            )
        self.prepared = None
        super().compute(identity, source, cancelled)
        packet = source.packet if source.packet is not None else self.tail
        if len(packet.frames) == 5 and not source.eos:
            self.tail, self.output_rgb = packet, None
        else:
            self.output_rgb = packet.rgb

    def deliver(self, identity, next_source, cancelled, emit):
        super().deliver(
            identity,
            next_source,
            cancelled,
            lambda packet: (
                emit(_RGB(self.output_rgb)) if self.output_rgb is not None else None
            ),
        )
        if (
            next_source is not None
            and next_source.packet is not None
            and (len(next_source.packet.frames) != 5 or next_source.eos)
        ):
            self.prepared = next_source

    def close(self, identity):
        super().close(identity)
        self.prepared, self.tail, self.output_rgb = None, None, None


def build_lookahead_cpu_bridge(context, *, directory):
    return _LookaheadCPUBridge(context, directory)


class _TensorCPUBridge(_LookaheadCPUBridge):
    def deliver_cpu(self, identity, next_source, cancelled):
        import weakref

        import torch

        source = self.computed
        packet = source.packet if source.packet is not None else self.tail
        self.deliver(identity, next_source, cancelled, lambda packet: None)
        if self.context.rank != self.context.output_rank or self.output_rgb is None:
            return ()
        rgb = torch.empty((len(packet.frames), packet.height, packet.width, 3),
            dtype=torch.uint8).share_memory_()
        rgb.copy_(torch.frombuffer(bytearray(self.output_rgb), dtype=torch.uint8).reshape(rgb.shape))
        weakref.finalize(rgb, (self.directory / f'released-{source.input_seq}').touch)
        return ({'rgb': rgb, 'frames': tuple(packet.frames)},)


def build_tensor_cpu_bridge(context, *, directory):
    return _TensorCPUBridge(context, directory)


def _host(directory, **kwargs):
    return H3CausalHost(
        f"{__name__}:build_cpu_bridge",
        factory_kwargs={"directory": str(directory)},
        world_size=kwargs.pop("world_size", 2),
        timeout=kwargs.pop("timeout", 10),
        **kwargs,
    )


def test_all_ranks_follow_the_same_session_order_and_stage_barriers(tmp_path):
    with _host(tmp_path, world_size=8, output_rank=7) as host:
        workers = tuple(host._processes)
        assert len({worker.pid for worker in workers}) == 8
        host.open("one", {"prompt": "drive"})
        assert host.append("one", "a") == (_RGB(b"rank-7:a"),)
        assert host.append("one", "b") == (_RGB(b"rank-7:b"),)
        assert host.finish("one") == (_RGB(b"rank-7:eos"),)
        with pytest.raises(RuntimeError, match="has ended"):
            host.append("one", "late")
        host.close("one")
        host.open("two", {})
        host.close("two")
    expected = [
        [name, identity]
        for identity, names in (
            ("one", ("open", "append", "append", "finish", "close")),
            ("two", ("open", "close")),
        )
        for name in names
    ]
    for rank in range(8):
        assert [
            json.loads(line)
            for line in (tmp_path / f"rank-{rank}.jsonl").read_text().splitlines()
        ] == expected
    assert all(not worker.is_alive() for worker in workers)


def test_host_rejects_a_second_session_and_stale_calls_without_dispatch(tmp_path):
    with _host(tmp_path) as host:
        with pytest.raises(ValueError, match="stale"):
            host.append("missing", "a")
        host.open("one", {})
        with pytest.raises(ValueError, match="one non-null"):
            host.open("two", {})
        with pytest.raises(ValueError, match="stale"):
            host.close("two")
        assert host.append("one", "a") == (_RGB(b"rank-0:a"),)
        host.close("one")


@pytest.mark.parametrize(
    "packet, error", [("crash", "failed"), ("exit", "disconnected")]
)
def test_rank_failure_stops_every_worker_and_prevents_reuse(tmp_path, packet, error):
    host = _host(tmp_path)
    workers = tuple(host._processes)
    try:
        host.open("one", {})
        with pytest.raises(RuntimeError, match=error):
            host.append("one", packet)
        assert all(not worker.is_alive() for worker in workers)
        with pytest.raises(RuntimeError, match="shut down"):
            host.finish("one")
    finally:
        host.shutdown()


def test_timeout_terminates_stalled_workers(tmp_path):
    host = _host(tmp_path)
    workers = tuple(host._processes)
    try:
        host.open("one", {})
        host.timeout = 0.1
        with pytest.raises(TimeoutError, match="timed out"):
            host.append("one", "stall")
        assert all(not worker.is_alive() for worker in workers)
    finally:
        host.shutdown()


def test_cancel_reaches_a_running_call_without_taking_its_lock(tmp_path):
    import threading

    host = _host(tmp_path)
    try:
        host.open("one", {})
        timer = threading.Timer(0.1, host.cancel, args=("one",))
        timer.start()
        try:
            with pytest.raises(RuntimeError, match="cancelled"):
                host.append("one", "wait_cancel")
        finally:
            timer.join()
        assert all(not worker.is_alive() for worker in host._processes)
    finally:
        host.shutdown()


def test_old_cancellation_cannot_cancel_a_reopened_session(tmp_path, monkeypatch):
    import threading

    with _host(tmp_path, world_size=1) as host:
        host.open("old", {})
        setting, release = threading.Event(), threading.Event()
        closed, reopened = threading.Event(), threading.Event()
        errors = []
        cancelled, call = host._cancelled, host._call

        class GatedCancellation:
            def set(self):
                setting.set()
                assert release.wait(10)
                cancelled.set()

            def __getattr__(self, name):
                return getattr(cancelled, name)

        def observed_call(method, *args):
            result = call(method, *args)
            if method == "close":
                closed.set()
            return result

        def cancel_old():
            try:
                host.cancel("old")
            except BaseException as error:
                errors.append(error)

        def reopen():
            try:
                host.close("old")
                host.open("new", {})
                reopened.set()
            except BaseException as error:
                errors.append(error)

        monkeypatch.setattr(host, "_cancelled", GatedCancellation())
        monkeypatch.setattr(host, "_call", observed_call)
        cancelling = threading.Thread(target=cancel_old)
        reopening = threading.Thread(target=reopen)
        cancelling.start()
        try:
            assert setting.wait(10)
            reopening.start()
            assert closed.wait(10)
            reopened.wait(1)
        finally:
            release.set()
            cancelling.join(10)
            if reopening.ident is not None:
                reopening.join(10)
        assert not cancelling.is_alive() and not reopening.is_alive()
        assert not errors
        assert host.append("new", "a") == (_RGB(b"rank-0:a"),)
        with pytest.raises(ValueError, match="stale"):
            host.cancel("old")


def test_factory_failure_cleans_spawned_children(tmp_path):
    previous = {child.pid for child in mp.active_children()}
    with pytest.raises(RuntimeError, match="factory failure|BrokenBarrierError"):
        H3CausalHost(
            f"{__name__}:build_cpu_bridge",
            world_size=2,
            timeout=10,
            factory_kwargs={"directory": str(tmp_path), "fail_rank": 1},
        )
    assert {child.pid for child in mp.active_children()} <= previous


def _pipe_host(connection):
    host = H3CausalHost.__new__(H3CausalHost)
    host.world_size, host.output_rank, host.timeout = 1, 0, 0.05
    host._connections, host._processes = [connection], []
    host._barrier = mp.get_context("spawn").Barrier(1)
    host._sequence, host._closed = 0, False
    return host


def _hold_unread_pipe(connection):
    time.sleep(20)


def test_deadline_bounds_large_send_to_a_peer_that_is_not_reading():
    context = mp.get_context("spawn")
    parent, unread_peer = context.Pipe()
    process = context.Process(target=_hold_unread_pipe, args=(unread_peer,))
    process.start()
    unread_peer.close()
    host = _pipe_host(parent)
    host._processes = [process]
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="transport timed out"):
            host._call("append", "one", b"x" * (8 * 1024 * 1024))
        assert time.monotonic() - started < 2
        assert host._closed
    finally:
        host._stop()


def test_deadline_bounds_receiving_a_header_without_the_complete_body():
    from multiprocessing.connection import Connection

    reader, peer = socket.socketpair()
    connection = Connection(reader.detach())
    host = _pipe_host(connection)
    try:
        peer.sendall(struct.pack("!i", 10000) + b"partial")
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="transport timed out"):
            host._bounded_io(connection.recv)
        assert time.monotonic() - started < 2
    finally:
        peer.close()
        host._stop()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"world_size": 0},
        {"output_rank": 8},
        {"timeout": float("nan")},
        {"timeout": 0},
    ],
)
def test_invalid_launch_arguments_do_not_create_workers(kwargs):
    with pytest.raises(ValueError):
        H3CausalHost("unused:factory", **kwargs)


def _native_args(tp=1, ulysses=1, **kwargs):
    values = dict(
        num_gpus=tp * ulysses,
        nnodes=1,
        tp_size=tp,
        sp_degree=ulysses,
        ulysses_degree=ulysses,
        ring_degree=1,
        kv_gather_degree=1,
        dp_size=1,
        enable_cfg_parallel=False,
        cfg_parallel_degree=1,
        encoder_parallel="auto",
        attention_backend=None,
        component_attention_backends=None,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("tp,ulysses", [(1, 1), (1, 4), (2, 2), (4, 1), (2, 4)])
def test_native_configuration_accepts_tp_and_ulysses_without_a_fixed_world(tp, ulysses):
    _validate_causal_parallelism(_native_args(tp, ulysses))


@pytest.mark.parametrize(
    "backend,encoder",
    [
        ("fa", "auto"),
        ("torch_sdpa", "replicate"),
        ("torch_cudnn_sdpa", "fold"),
        ("dynamic_cudnn_sdpa", "auto"),
    ],
)
def test_native_dense_backend_and_encoder_policies_are_preserved(backend, encoder):
    args = _native_args(2, 2, attention_backend=backend, encoder_parallel=encoder)
    _validate_causal_parallelism(args)
    assert args.attention_backend == backend and args.encoder_parallel == encoder


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"tp_size": 0}, "tp_size"),
        ({"num_gpus": 8}, "num_gpus"),
        ({"dp_size": 2}, "DP1"),
        ({"enable_cfg_parallel": True}, "CFG1"),
        ({"cfg_parallel_degree": 2}, "CFG1"),
        ({"ring_degree": 2}, "ring"),
        ({"kv_gather_degree": 2}, "KV gather"),
        ({"sp_degree": 4}, "sp_degree"),
        ({"encoder_parallel": "dp"}, "encoder_parallel"),
        ({"attention_backend": "sage_attn"}, "dense"),
        (
            {"component_attention_backends": {"transformer": "cube_sparse_attn"}},
            "dense",
        ),
    ],
)
def test_unsupported_native_layouts_fail_before_spawning(changes, error):
    with pytest.raises(ValueError, match=error):
        H3CausalHost("unused:factory", server_args=_native_args(2, 2, **changes))


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"world_size": 8}, "world_size"),
        ({"world_size": True}, "world_size"),
        ({"server_args": _native_args(2, 2, nnodes=2)}, "one node"),
    ],
)
def test_native_host_size_and_node_mismatch_fail_before_spawning(kwargs, error):
    args = {"server_args": _native_args(2, 2), **kwargs}
    with pytest.raises(ValueError, match=error):
        H3CausalHost("unused:factory", **args)


def test_cpu_factory_default_is_one_worker(tmp_path):
    with H3CausalHost(
        f"{__name__}:build_cpu_bridge",
        factory_kwargs={"directory": str(tmp_path)},
        timeout=10,
    ) as host:
        assert host.world_size == len(host._processes) == 1
        host.open("one", {})
        assert host.append("one", "a") == (_RGB(b"rank-0:a"),)
        host.close("one")


def test_staged_commands_keep_all_rank_dit_and_codec_boundaries_and_eos(tmp_path):
    with _host(tmp_path, world_size=8) as host:
        host.open('one', {})
        first = SimpleNamespace(input_seq=0, packet='a', eos=False)
        last = SimpleNamespace(input_seq=1, packet='tail', eos=True)
        host.compute('one', first)
        with pytest.raises(RuntimeError, match='awaiting delivery'):
            host.compute('one', last)
        with pytest.raises(RuntimeError, match='awaiting delivery'):
            host.append('one', 'unexpected')
        assert host.deliver('one', last) == (_RGB(b'rank-0:a'),)
        with pytest.raises(RuntimeError, match='requires a computed'):
            host.deliver('one')
        host.compute('one', last)
        assert host.deliver('one') == (_RGB(b'rank-0:tail'),)
        with pytest.raises(RuntimeError, match='has ended'):
            host.compute('one', first)
        host.close('one')
        host.open('two', {})
        host.compute('two', first)
        host.close('two')
    for rank in range(8):
        records = [json.loads(line)[0] for line in
            (tmp_path / f'rank-{rank}.jsonl').read_text().splitlines()]
        assert records == ['open', 'compute', 'deliver', 'compute', 'deliver',
                           'close', 'open', 'compute', 'close']


def _source(input_seq, *, frames=17, eos=False):
    packet = _Packet(
        bytes([input_seq + 1]) * (64 * 64 * frames * 3),
        tuple(
            _Stamp(f"{input_seq}-{index}", index * 40.0, 40.0)
            for index in range(frames)
        ),
    )
    return _Source(input_seq, packet, eos)


def _observe_pipe_bytes(host, monkeypatch, compute_refs=None):
    sizes = [[] for _ in host._connections]
    for connection, sent in zip(host._connections, sizes):
        send = connection._send_bytes

        def observed(buffer, send=send, sent=sent):
            sent.append(len(buffer))
            if compute_refs is not None and len(buffer) < 256:
                _, method, _, payload = pickle.loads(buffer)
                if method == "compute":
                    compute_refs.append((payload.input_seq, payload.eos))
            return send(buffer)

        monkeypatch.setattr(connection, "_send_bytes", observed)
    return sizes


def _lookahead_host(directory, **kwargs):
    return H3CausalHost(
        f"{__name__}:build_lookahead_cpu_bridge",
        factory_kwargs={"directory": str(directory)},
        timeout=10,
        **kwargs,
    )


def test_delivered_sources_use_small_compute_payloads_and_preserve_rgb_and_tail(
    tmp_path, monkeypatch
):
    first, second, tail = _source(0), _source(1), _source(2, frames=5)
    eos = _Source(3, None, True)
    with _lookahead_host(tmp_path, world_size=8) as host:
        host.open("one", {})
        compute_refs = []
        sizes = _observe_pipe_bytes(host, monkeypatch, compute_refs)
        host.compute("one", first)
        assert host.deliver("one", second) == (_RGB(first.packet.rgb),)
        assert host._prefetched_source.packet.rgb is second.packet.rgb
        with pytest.raises(ValueError, match="stale"):
            host.compute("old", second)
        host.compute("one", deepcopy(second))
        assert host.deliver("one", tail) == (_RGB(second.packet.rgb),)
        host.compute("one", tail)
        assert host.deliver("one", eos) == ()
        host.compute("one", eos)
        assert host.deliver("one") == (_RGB(tail.packet.rgb),)
        assert host._prefetched_source is None
        assert compute_refs == [(1, False)] * 8 + [(2, False)] * 8 + [(3, True)] * 8
        for sent in sizes:
            assert len(sent) == 8
            assert sent[0] > len(first.packet.rgb)
            assert sent[1] > len(second.packet.rgb)
            assert sent[3] > len(tail.packet.rgb)
            assert all(sent[index] < 256 for index in (2, 4, 6))
        host.close("one")
        host.open("two", {})
        host.compute("two", second)
        assert all(sent[-1] > len(second.packet.rgb) for sent in sizes)
        host.close("two")
    expected = [
        "open",
        "compute",
        "deliver",
        "compute",
        "deliver",
        "compute",
        "deliver",
        "compute",
        "deliver",
        "close",
        "open",
        "compute",
        "close",
    ]
    for rank in range(8):
        records = [
            json.loads(line)[0]
            for line in (tmp_path / f"rank-{rank}.jsonl").read_text().splitlines()
        ]
        assert records == expected


@pytest.mark.parametrize("changed", ["rgb", "stamp", "sequence", "eos", "geometry"])
def test_changed_lookahead_is_sent_in_full_and_keeps_bridge_mismatch_failure(
    tmp_path, monkeypatch, changed
):
    first, next_source = _source(0), _source(1)
    if changed == "rgb":
        changed_source = replace(
            next_source,
            packet=replace(next_source.packet, rgb=b"z" * len(next_source.packet.rgb)),
        )
    elif changed == "stamp":
        stamps = (
            replace(next_source.packet.frames[0], source_frame_id="changed"),
            *next_source.packet.frames[1:],
        )
        changed_source = replace(
            next_source, packet=replace(next_source.packet, frames=stamps)
        )
    elif changed == "sequence":
        changed_source = replace(next_source, input_seq=2)
    elif changed == "eos":
        changed_source = replace(next_source, eos=True)
    else:
        changed_source = replace(
            next_source, packet=replace(next_source.packet, width=32)
        )
    with _lookahead_host(tmp_path, world_size=2) as host:
        host.open("one", {})
        host.compute("one", first)
        host.deliver("one", next_source)
        sizes = _observe_pipe_bytes(host, monkeypatch)
        with pytest.raises(RuntimeError, match="prepared H3 proxy does not match"):
            host.compute("one", changed_source)
        assert all(sent[0] > len(changed_source.packet.rgb) for sent in sizes)
        assert host._prefetched_source is None
        assert all(not worker.is_alive() for worker in host._processes)


@pytest.mark.parametrize("changed", ["rgb", "stamp"])
def test_mutating_a_delivered_source_does_not_reuse_its_old_payload(
    tmp_path, monkeypatch, changed
):
    first = _source(0)
    next_source = SimpleNamespace(**vars(_source(1)))
    next_source.packet = SimpleNamespace(**vars(next_source.packet))
    next_source.packet.frames = [
        SimpleNamespace(**vars(stamp)) for stamp in next_source.packet.frames
    ]
    with _lookahead_host(tmp_path, world_size=1) as host:
        host.open("one", {})
        host.compute("one", first)
        host.deliver("one", next_source)
        if changed == "rgb":
            next_source.packet.rgb = b"z" * len(next_source.packet.rgb)
        else:
            next_source.packet.frames[0].source_frame_id = "mutated"
        sizes = _observe_pipe_bytes(host, monkeypatch)
        with pytest.raises(RuntimeError, match="prepared H3 proxy does not match"):
            host.compute("one", next_source)
        assert sizes[0][0] > len(next_source.packet.rgb)


@pytest.mark.parametrize(
    "changes", [{"input_seq": True}, {"input_seq": 1.0}, {"eos": 0}]
)
def test_equal_but_invalid_source_types_reach_original_bridge_validation(
    tmp_path, monkeypatch, changes
):
    first, next_source = _source(0), _source(1)
    changed_source = replace(next_source, **changes)
    assert changed_source == next_source
    with _lookahead_host(tmp_path, world_size=1) as host:
        host.open("one", {})
        host.compute("one", first)
        host.deliver("one", next_source)
        sizes = _observe_pipe_bytes(host, monkeypatch)
        with pytest.raises(
            RuntimeError, match="stale or invalid native H3 source identity"
        ):
            host.compute("one", changed_source)
        assert sizes[0][0] > len(next_source.packet.rgb)
        assert host._prefetched_source is None


@pytest.mark.parametrize("invalid", ["missing", "sequence", "identity"])
def test_invalid_prefetched_reference_fails_the_whole_host(tmp_path, invalid):
    first, next_source = _source(0), _source(1)
    with _lookahead_host(tmp_path, world_size=2) as host:
        host.open("one", {})
        host.compute("one", first)
        host.deliver("one", next_source)
        if invalid == "missing":
            host.compute("one", next_source)
            host.deliver("one")
        identity = "old" if invalid == "identity" else "one"
        sequence = 2 if invalid == "sequence" else 1
        with pytest.raises(
            RuntimeError, match="prefetched source reference does not match"
        ):
            host._call("compute", identity, _PrefetchedSource(sequence, False))
        assert host._prefetched_source is None
        assert all(not worker.is_alive() for worker in host._processes)


def test_ordinary_commands_and_cancel_clear_cached_lookahead(tmp_path, monkeypatch):
    first, next_source = _source(0), _source(1)
    with _host(tmp_path, world_size=1) as host:
        host.open("one", {})
        sizes = _observe_pipe_bytes(host, monkeypatch)
        host.compute("one", first)
        host.deliver("one", next_source)
        host.append("one", "ordinary")
        assert host._prefetched_source is None
        host.compute("one", next_source)
        assert sizes[0][-1] > len(next_source.packet.rgb)
        host.deliver("one", first)
        host.finish("one")
        assert host._prefetched_source is None
        host.close("one")
        host.open("two", {})
        host.compute("two", first)
        host.deliver("two", next_source)
        host.cancel("two")
        assert host._prefetched_source is None
        with pytest.raises(RuntimeError):
            host.compute("two", next_source)
        assert sizes[0][-1] > len(next_source.packet.rgb)


def test_cpu_delivery_uses_descriptors_and_survives_next_compute_and_worker_teardown(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip('torch')
    first, second = _source(0), _source(1)
    first = replace(first, packet=replace(first.packet, width=768, height=512,
        rgb=b'\x01' * (17 * 512 * 768 * 3)))
    received_sizes = []
    with H3CausalHost(f'{__name__}:build_tensor_cpu_bridge', world_size=2, timeout=30,
            factory_kwargs={'directory': str(tmp_path)}) as host:
        host.open('one', {})
        for connection in host._connections:
            receive = connection._recv_bytes
            def observed(maxsize=None, receive=receive):
                frame = receive(maxsize)
                received_sizes.append(len(frame.getbuffer()))
                return frame
            monkeypatch.setattr(connection, '_recv_bytes', observed)
        host.compute('one', first)
        batch = host.deliver_cpu('one', second)
        assert len(batch) == 1 and isinstance(batch[0]['frames'], tuple)
        tensor = batch[0]['rgb']
        assert tensor.is_shared() and tensor.is_contiguous()
        assert tuple(tensor.shape) == (17, 512, 768, 3)
        assert torch.all(tensor == 1)
        assert batch[0]['frames'] == first.packet.frames
        assert len(first.packet.rgb) > 16 * 1024 * 1024
        assert max(received_sizes) < 8192
        host.compute('one', deepcopy(second))
        assert (tmp_path / 'released-0').exists()
        following = host.deliver_cpu('one')
        assert following[0]['frames'] == second.packet.frames
        assert torch.all(following[0]['rgb'] == 2)
        assert torch.all(tensor == 1)
        for rank in range(2):
            assert (tmp_path / f'0-codec_join-{rank}').exists()
        host.close('one')
        host.open('two', {})
        host.compute('two', second)
        host.deliver_cpu('two')
        host.close('two')
    assert torch.all(tensor == 1)
    assert torch.all(following[0]['rgb'] == 2)
    assert batch[0]['frames'] == first.packet.frames


def test_cpu_delivery_cancelled_at_rank_join_discards_host(tmp_path):
    pytest.importorskip('torch')
    host = H3CausalHost(f'{__name__}:build_tensor_cpu_bridge', world_size=2, timeout=30,
        factory_kwargs={'directory': str(tmp_path)})
    try:
        host.open('one', {})
        host.compute('one', _source(0))
        host.cancel('one')
        with pytest.raises(RuntimeError, match='cancelled'):
            host.deliver_cpu('one', _source(1))
        assert host._prefetched_source is None
        assert all(not worker.is_alive() for worker in host._processes)
    finally:
        host.shutdown()
