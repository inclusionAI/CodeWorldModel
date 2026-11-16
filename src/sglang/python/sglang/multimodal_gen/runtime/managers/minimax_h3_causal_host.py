# SPDX-License-Identifier: Apache-2.0
"""One local, synchronous rank group hosting one causal H3 media bridge."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import pickle
import socket
import threading
import time
import traceback
from copy import deepcopy
from multiprocessing.connection import wait

import msgspec


class H3WorkerContext(msgspec.Struct, frozen=True):
    rank: int
    world_size: int
    output_rank: int
    server_args: object = None


class _PrefetchedSource(msgspec.Struct, frozen=True):
    input_seq: int
    eos: bool


def _factory(symbol):
    module, separator, name = symbol.partition(":")
    if not separator or not module or not name or "." in name:
        raise ValueError("worker_factory must be an importable module:function")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise TypeError("worker_factory must resolve to a callable")
    return factory


def _validate_causal_parallelism(args):
    for name in ("num_gpus", "tp_size", "sp_degree", "ulysses_degree"):
        value = getattr(args, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        (args.dp_size or 1) != 1
        or (args.cfg_parallel_degree or 1) != 1
        or args.enable_cfg_parallel
    ):
        raise ValueError("causal H3 hosts one replica with DP1 and CFG1")
    if args.ring_degree != 1:
        raise ValueError("causal H3 does not support ring KV rotation")
    if args.kv_gather_degree != 1:
        raise ValueError("causal H3 requires Ulysses; KV gather is not implemented")
    if args.sp_degree != args.ulysses_degree:
        raise ValueError("causal H3 sp_degree must equal ulysses_degree")
    if args.num_gpus != args.tp_size * args.sp_degree:
        raise ValueError("causal H3 num_gpus must equal tp_size * sp_degree")
    if args.encoder_parallel not in ("auto", "fold", "replicate"):
        raise ValueError("causal H3 encoder_parallel requires auto, fold or replicate")
    backend = (args.component_attention_backends or {}).get(
        "transformer", args.attention_backend
    )
    if backend is not None and backend.upper() not in (
        "FA",
        "TORCH_SDPA",
        "TORCH_CUDNN_SDPA",
        "DYNAMIC_CUDNN_SDPA",
    ):
        raise ValueError("causal H3 requires a dense FlashAttention or SDPA backend")


def _initialize_native(payload, rank, world_size, master_port):
    # Match worker_bootstrap: platform and hooks precede the runtime object graph.
    from sglang.multimodal_gen.runtime.utils.process import kill_itself_when_parent_died

    kill_itself_when_parent_died()
    from sglang.multimodal_gen.runtime.platforms import initialize_current_platform

    initialize_current_platform()
    from sglang.multimodal_gen.runtime.platforms.plugins import (
        apply_plugin_hooks,
        load_plugins,
    )

    load_plugins()
    apply_plugin_hooks()
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )

    args = pickle.loads(payload)
    if not isinstance(args, ServerArgs):
        raise TypeError("server_args must contain native SGLang ServerArgs")
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
        LOCAL_RANK=str(rank),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    set_global_server_args(args)
    from sglang.multimodal_gen.runtime.distributed import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=args.tp_size,
        sp_size=args.sp_degree,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        dp_size=1,
        cfg_degree=1,
        distributed_init_method=f"tcp://127.0.0.1:{master_port}",
        dist_timeout=args.dist_timeout,
    )
    from sglang.srt.runtime_context import get_context, publish
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    if get_context()._server_args is None:
        publish(
            SrtServerArgs(model_path="dummy", tp_size=args.tp_size),
            role="diffusion_gpu_worker",
        )
    return args


class _StageComponents:
    def __init__(self, components, barrier, timeout):
        self._components, self._barrier, self._timeout = components, barrier, timeout

    def __getattr__(self, name):
        return getattr(self._components, name)

    def synchronize(self):
        self._components.synchronize()
        self._barrier.wait(timeout=self._timeout)


def _worker_main(
    connection,
    barrier,
    cancelled,
    rank,
    world_size,
    output_rank,
    factory_symbol,
    factory_payload,
    server_payload,
    master_port,
    timeout,
):
    bridge = None
    identity = None
    prefetched_source = None
    sequence = -1
    try:
        args = (
            None
            if server_payload is None
            else _initialize_native(
                server_payload,
                rank,
                world_size,
                master_port,
            )
        )
        context = H3WorkerContext(rank, world_size, output_rank, args)
        bridge = _factory(factory_symbol)(context, **pickle.loads(factory_payload))
        bridge.components = _StageComponents(bridge.components, barrier, timeout)
        bridge.components.synchronize()
        connection.send((-1, True, None))
        expected = 0
        while True:
            sequence, method, requested_identity, payload = connection.recv()
            if sequence != expected:
                raise RuntimeError("H3 worker command sequence diverged")
            expected += 1
            packets = []

            def emit(packet, packets=packets):
                if rank == output_rank:
                    packets.append(packet)

            if method == "shutdown":
                prefetched_source = None
                if bridge is not None and identity is not None:
                    bridge.close(identity)
                bridge.components.synchronize()
                connection.send((sequence, True, None))
                break
            if method == "open":
                prefetched_source = None
                bridge.open(requested_identity, payload)
                identity = requested_identity
            elif method == "append":
                prefetched_source = None
                bridge.append(requested_identity, payload, cancelled, emit)
            elif method == "finish":
                prefetched_source = None
                bridge.finish(requested_identity, cancelled, emit)
            elif method == "compute":
                if isinstance(payload, _PrefetchedSource):
                    if (
                        requested_identity != identity
                        or prefetched_source is None
                        or prefetched_source.input_seq != payload.input_seq
                        or prefetched_source.eos != payload.eos
                    ):
                        raise RuntimeError("H3 prefetched source reference does not match")
                    payload = prefetched_source
                prefetched_source = None
                bridge.compute(requested_identity, payload, cancelled)
            elif method == "deliver":
                prefetched_source = None
                bridge.deliver(requested_identity, payload, cancelled, emit)
                prefetched_source = payload
            elif method == "deliver_cpu":
                # Registers tensor reductions so the pipe sends shared-memory handles.
                import torch.multiprocessing  # noqa: F401

                prefetched_source = None
                batch = bridge.deliver_cpu(requested_identity, payload, cancelled)
                if rank == output_rank:
                    packets.extend(batch)
                del batch
                prefetched_source = payload
            elif method == "close":
                prefetched_source = None
                bridge.close(requested_identity)
                identity = None
            else:
                raise ValueError(f"unknown H3 worker command {method!r}")
            bridge.components.synchronize()
            connection.send(
                (sequence, True, tuple(packets) if rank == output_rank else None)
            )
            del packets, emit
    except BaseException:
        barrier.abort()
        try:
            connection.send((sequence, False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        prefetched_source = None
        connection.close()


class H3CausalHost:
    """Trusted local RPC. Factory/configuration and packet objects are pickled.

    A rank failure invalidates the complete host. Construct another host rather
    than reusing possibly divergent KV or decoder state. CPU factories may omit
    ServerArgs and own their initialization; native factories receive ready groups.
    """

    def __init__(
        self,
        worker_factory,
        *,
        factory_kwargs=None,
        server_args=None,
        world_size=None,
        output_rank=0,
        timeout=300.0,
    ):
        if world_size is not None and world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if server_args is not None:
            _validate_causal_parallelism(server_args)
            if server_args.nnodes != 1:
                raise ValueError("H3 host requires one node")
            if world_size is not None and world_size != server_args.num_gpus:
                raise ValueError("world_size must match ServerArgs.num_gpus")
            world_size = server_args.num_gpus
        elif world_size is None:
            world_size = 1
        if not 0 <= output_rank < world_size:
            raise ValueError("output_rank must identify a worker")
        if not 0 < timeout < float("inf"):
            raise ValueError("timeout must be finite and positive")
        factory_payload = pickle.dumps(dict(factory_kwargs or {}))
        server_payload = None if server_args is None else pickle.dumps(server_args)
        self.world_size, self.output_rank, self.timeout = (
            world_size,
            output_rank,
            timeout,
        )
        self._lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._sequence, self._identity, self._ended = 0, None, False
        self._computed_source = None
        self._prefetched_source = None
        self._closed = False
        self._io_thread = None
        self._connections, self._processes = [], []
        context = mp.get_context("spawn")
        self._barrier = context.Barrier(world_size)
        self._cancelled = context.Event()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            master_port = reservation.getsockname()[1]
        try:
            for rank in range(world_size):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker_main,
                    name=f"h3-causal-rank-{rank}",
                    args=(
                        child,
                        self._barrier,
                        self._cancelled,
                        rank,
                        world_size,
                        output_rank,
                        worker_factory,
                        factory_payload,
                        server_payload,
                        master_port,
                        timeout,
                    ),
                )
                self._connections.append(parent)
                self._processes.append(process)
                process.start()
                child.close()
            self._bounded_io(lambda: self._receive(-1))
        except BaseException:
            self._stop()
            raise

    def _receive(self, sequence):
        deadline = time.monotonic() + self.timeout
        output = ()
        pending = dict(zip(self._connections, range(self.world_size)))
        while pending:
            ready = wait(pending, timeout=max(0, deadline - time.monotonic()))
            if not ready:
                raise TimeoutError(f"H3 workers {list(pending.values())} timed out")
            for connection in ready:
                rank = pending.pop(connection)
                try:
                    received, success, result = connection.recv()
                except (EOFError, OSError) as error:
                    raise RuntimeError(f"H3 worker {rank} disconnected") from error
                if received != sequence or not success:
                    raise RuntimeError(
                        f"H3 worker {rank} failed at command {sequence}: {result}"
                    )
                if rank == self.output_rank and result is not None:
                    output = result
        return output

    def _call(self, method, identity=None, payload=None):
        if self._closed:
            raise RuntimeError("H3 host is shut down")
        sequence = self._sequence
        self._sequence += 1

        def exchange():
            for connection in self._connections:
                connection.send((sequence, method, identity, payload))
            return self._receive(sequence)

        try:
            return self._bounded_io(exchange)
        except BaseException:
            self._stop()
            raise

    def _bounded_io(self, action):
        result = []

        def run():
            try:
                result.append((True, action()))
            except BaseException as error:  # noqa: BLE001 -- report on the calling thread.
                result.append((False, error.with_traceback(None)))

        # Pipe readiness does not bound a full frame transfer or its unpickling.
        io = threading.Thread(target=run, name="h3-causal-rpc", daemon=True)
        self._io_thread = io
        io.start()
        io.join(timeout=self.timeout)
        if io.is_alive():
            raise TimeoutError("H3 worker transport timed out")
        self._io_thread = None
        success, value = result[0]
        if not success:
            raise value
        return value

    def open(self, session_identity, request):
        with self._lock:
            with self._session_lock:
                if session_identity is None or self._identity is not None:
                    raise ValueError("H3 host requires one non-null session identity")
                self._cancelled.clear()
                self._prefetched_source = None
            self._call("open", session_identity, request)
            with self._session_lock:
                self._identity, self._ended = session_identity, False

    def _current(self, identity):
        if self._identity is None or identity != self._identity:
            raise ValueError("unknown or stale H3 session identity")
        if self._ended:
            raise RuntimeError("H3 session has ended; close and reopen")

    def append(self, session_identity, packet):
        with self._lock:
            self._current(session_identity)
            if self._computed_source is not None:
                raise RuntimeError("H3 source is awaiting delivery")
            self._prefetched_source = None
            return self._call("append", session_identity, packet)

    def finish(self, session_identity):
        with self._lock:
            self._current(session_identity)
            if self._computed_source is not None:
                raise RuntimeError("H3 source is awaiting delivery")
            self._prefetched_source = None
            result = self._call("finish", session_identity)
            self._ended = True
            return result

    def compute(self, session_identity, source):
        with self._lock:
            self._current(session_identity)
            if self._computed_source is not None:
                raise RuntimeError("H3 source is awaiting delivery")
            prefetched, self._prefetched_source = self._prefetched_source, None
            payload = (
                _PrefetchedSource(source.input_seq, source.eos)
                if (
                    prefetched is not None
                    # Exact types: True == 1, so equality alone could swap identities.
                    and type(source.input_seq) is int
                    and type(source.eos) is bool
                    and type(prefetched.input_seq) is int
                    and type(prefetched.eos) is bool
                    and prefetched == source
                )
                else source
            )
            self._call("compute", session_identity, payload)
            self._computed_source = source

    def deliver(self, session_identity, next_source=None):
        return self._deliver("deliver", session_identity, next_source)

    def deliver_cpu(self, session_identity, next_source=None):
        return self._deliver("deliver_cpu", session_identity, next_source)

    def _deliver(self, method, session_identity, next_source):
        with self._lock:
            self._current(session_identity)
            if self._computed_source is None:
                raise RuntimeError("H3 delivery requires a computed source")
            # Snapshot metadata against caller mutation; deepcopy retains immutable RGB bytes.
            prefetched = deepcopy(next_source)
            result = self._call(method, session_identity, prefetched)
            self._ended = self._computed_source.eos
            self._computed_source = None
            with self._session_lock:
                self._prefetched_source = (
                    None if self._cancelled.is_set() or self._ended else prefetched
                )
            return result

    def cancel(self, session_identity):
        with self._session_lock:
            if self._identity is None or session_identity != self._identity:
                raise ValueError("unknown or stale H3 session identity")
            self._cancelled.set()
            self._prefetched_source = None

    def close(self, session_identity):
        with self._lock:
            if self._identity is None or session_identity != self._identity:
                raise ValueError("unknown or stale H3 session identity")
            self._call("close", session_identity)
            with self._session_lock:
                self._identity, self._ended = None, False
                self._computed_source = None
                self._prefetched_source = None

    def _stop(self):
        self._closed = True
        self._prefetched_source = None
        self._barrier.abort()
        for process in self._processes:
            if process.pid is not None:
                process.join(timeout=0.2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
        io = self._io_thread
        if io is not None:
            io.join(timeout=0.2)
        for connection in self._connections:
            connection.close()

    def shutdown(self):
        with self._lock:
            if not self._closed:
                try:
                    self._call("shutdown")
                finally:
                    self._stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
