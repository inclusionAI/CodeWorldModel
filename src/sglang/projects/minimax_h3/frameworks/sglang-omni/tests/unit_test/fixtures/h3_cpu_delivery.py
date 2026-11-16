# SPDX-License-Identifier: Apache-2.0
"""Real two-process H3 route with CPU-only native work and delivery gates."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path

import torch

from sglang_omni.comm.data_ref import DataRef, TransportKind
from sglang_omni.comm.stage_io import StageMessageSender
from sglang_omni.models.minimax_h3.config import H3_STAGE, MiniMaxH3PipelineConfig
from sglang_omni.models.minimax_h3.delivery import H3CPUFrameBatch, H3RGBDeliveryHooks
from sglang_omni.models.minimax_h3.session import (
    H3MediaBridge,
    H3NextSource,
    H3OpenRequest,
    H3RGBPacket,
    H3RuntimeLimits,
    H3SessionHooks,
    H3SourceInput,
)
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, construct_stage
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SessionIdentity,
    TimedChunk,
    find_session_operation,
)
from sglang_omni.relay.base import Relay
from sglang_omni.scheduling.session import SessionContext
from tests.unit_test.fixtures.h3_session import CPUEchoBridge

TIMEOUT_SECONDS = 20
# Importing Torch from a shared filesystem can exceed the operation deadline.
STARTUP_TIMEOUT_SECONDS = 120
RUNTIME_FACTORY = f"{__name__}.create_cpu_bridge"


@dataclass(kw_only=True)
class DeliveryProbe:
    packing_entered: Event
    packing_release: Event
    compute_entered: Event
    compute_release: Event
    next_compute: Event
    cancelled: Event
    relay_entered: Event
    relay_release: Event
    events: Queue[tuple[str, int]]
    block_packing: bool = False
    block_compute: bool = False
    block_relay: bool = False
    fail_delivery: bool = False


class CPUCarrierBridge(CPUEchoBridge):
    probe: DeliveryProbe | None = None

    def open(self, session_identity: SessionIdentity, request: H3OpenRequest) -> None:
        super().open(session_identity, request)
        if self.probe is not None:
            self.probe.events.put(("open-source", session_identity.open_index))
        else:
            pass

    def close(self, session_identity: SessionIdentity) -> None:
        if self.probe is not None:
            self.probe.events.put(("close-source", session_identity.open_index))
        else:
            pass
        super().close(session_identity)

    def advance_cpu(
        self,
        session_identity: SessionIdentity,
        source: H3SourceInput,
        cancelled: threading.Event,
        peek_next: H3NextSource | None = None,
    ) -> H3CPUFrameBatch:
        if self.probe is not None:
            self.probe.events.put(("compute", source.input_seq))
            if source.input_seq == 1:
                self.probe.next_compute.set()
            elif source.input_seq == 0 and self.probe.block_compute:
                self.probe.compute_entered.set()
                while not self.probe.compute_release.wait(0.01):
                    if cancelled.is_set():
                        self.probe.cancelled.set()
                        return ()
                    else:
                        pass
            else:
                pass
        else:
            pass
        packets: list[H3RGBPacket] = []
        self.advance(session_identity, source, cancelled, packets.append, peek_next)
        return tuple(
            {
                "rgb": torch.frombuffer(bytearray(packet.rgb), dtype=torch.uint8)
                .reshape(len(packet.frames), packet.height, packet.width, 3)
                .clone()
                .share_memory_(),
                "frames": packet.frames,
                "orchestration": {"clock": "native_worker_monotonic", "stages": [
                    {"stage": "render", "dispatch_ns": 1000, "complete_ns": 2000, "wall_ms": .001}
                ]},
            }
            for packet in packets
        )


def create_cpu_bridge(*, model_path: str, limits: H3RuntimeLimits) -> H3MediaBridge:
    return CPUCarrierBridge(limits)


class ProbeDeliveryHooks(H3RGBDeliveryHooks):
    def __init__(self, limits: H3RuntimeLimits, probe: DeliveryProbe) -> None:
        super().__init__(limits)
        self.probe = probe

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        operation = find_session_operation(payload.request.metadata)
        assert operation is not None and operation.chunk is not None
        assert operation.chunk.payload is None
        assert chunk.payload is None
        self.probe.events.put(("delivery", chunk.seq))
        if chunk.seq == 0 and self.probe.block_packing:
            self.probe.packing_entered.set()
            while not self.probe.packing_release.wait(0.01):
                if context.cancelled.is_set():
                    self.probe.cancelled.set()
                    break
                else:
                    pass
        else:
            pass
        if self.probe.fail_delivery:
            raise RuntimeError("CPU RGB packing failed")
        else:
            return super().append(chunk, payload, context)

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.probe.events.put(("open-delivery", session_identity.open_index))
        super().open(session_identity, request)

    def close(self, session_identity: SessionIdentity) -> None:
        self.probe.events.put(("close-delivery", session_identity.open_index))
        super().close(session_identity)


def stage_worker(spec: StageLaunchConfig, ready: Event, probe: DeliveryProbe) -> None:
    async def run() -> None:
        stage = construct_stage(spec, logging.getLogger(__name__))
        hooks = stage.scheduler.session_hooks
        if isinstance(hooks, H3SessionHooks):
            assert isinstance(hooks.bridge, CPUCarrierBridge)
            hooks.bridge.probe = probe
        else:
            assert isinstance(hooks, H3RGBDeliveryHooks)
            stage.scheduler.session_hooks = ProbeDeliveryHooks(hooks.limits, probe)
        original_read = stage.comm.read_payload

        async def read_payload(
            *, relay: Relay, request_id: str, data_ref: DataRef
        ) -> StagePayload:
            assert data_ref.transport is TransportKind.SHM
            payload = await original_read(
                relay=relay, request_id=request_id, data_ref=data_ref
            )
            operation = find_session_operation(payload.request.metadata)
            assert operation is not None and operation.chunk is not None
            probe.events.put(("shm", operation.chunk.seq))
            return payload

        stage.comm.read_payload = read_payload
        original_send = stage.comm.send_payload

        async def send_payload(
            *,
            relay: Relay,
            control_plane: StageMessageSender,
            request_id: str,
            payload: StagePayload,
            transport: TransportKind,
            from_stage: str,
            to_stage: str,
            target_endpoint: str,
            replica_bindings: dict[str, int] | None = None,
        ) -> DataRef:
            operation = find_session_operation(payload.request.metadata)
            if (
                probe.block_relay
                and operation is not None
                and operation.operation == "append"
                and operation.chunk.seq == 0
            ):
                probe.relay_entered.set()
                assert await asyncio.to_thread(
                    probe.relay_release.wait, TIMEOUT_SECONDS
                )
            else:
                pass
            return await original_send(
                relay=relay,
                control_plane=control_plane,
                request_id=request_id,
                payload=payload,
                transport=transport,
                from_stage=from_stage,
                to_stage=to_stage,
                target_endpoint=target_endpoint,
                replica_bindings=replica_bindings,
            )

        stage.comm.send_payload = send_payload
        await stage.start()
        ready.set()
        await stage.run()
        assert not stage.scheduler.open_sessions

    asyncio.run(run())


@asynccontextmanager
async def cpu_delivery_pipeline(
    directory: Path,
    *,
    block_packing: bool = False,
    block_compute: bool = False,
    block_relay: bool = False,
    fail_delivery: bool = False,
    max_output_chunks: int = 16,
) -> AsyncIterator[tuple[Coordinator, MiniMaxH3PipelineConfig, DeliveryProbe]]:
    config = MiniMaxH3PipelineConfig(
        model_path="cpu-delivery-test",
        runtime_factory=RUNTIME_FACTORY,
        codec_overlap=True,
        cpu_delivery=True,
        max_width=2,
        max_height=2,
        max_state_bytes=1024,
        max_output_chunks=max_output_chunks,
        operation_timeout_seconds=TIMEOUT_SECONDS,
    )
    context = multiprocessing.get_context("spawn")
    probe = DeliveryProbe(
        packing_entered=context.Event(),
        packing_release=context.Event(),
        compute_entered=context.Event(),
        compute_release=context.Event(),
        next_compute=context.Event(),
        cancelled=context.Event(),
        relay_entered=context.Event(),
        relay_release=context.Event(),
        events=context.Queue(),
        block_packing=block_packing,
        block_compute=block_compute,
        block_relay=block_relay,
        fail_delivery=fail_delivery,
    )
    endpoints = {
        stage.name: f"ipc://{directory}/{stage.name}" for stage in config.stages
    }
    completion = f"ipc://{directory}/done"
    abort = f"ipc://{directory}/abort"
    coordinator = Coordinator(
        completion,
        abort,
        H3_STAGE,
        config.terminal_stages,
        session_graph=config.session_graph(),
    )
    await coordinator.start()
    completion_loop = asyncio.create_task(coordinator.run_completion_loop())
    processes: list[SpawnProcess] = []
    try:
        for stage in config.stages:
            ready = context.Event()
            spec = StageLaunchConfig(
                stage_name=stage.name,
                factory=stage.factory_path,
                factory_kwargs=config.stage_factory_kwargs(stage.name),
                factory_arg_defaults={"model_path": config.model_path},
                next_stages=stage.next,
                is_terminal=stage.terminal,
                recv_endpoint=endpoints[stage.name],
                coordinator_endpoint=completion,
                abort_endpoint=abort,
                stage_endpoints=endpoints,
                comm_config={"slot_size_mb": 1, "credits": 2},
                allow_child_processes=stage.allow_child_processes,
            )
            process = context.Process(target=stage_worker, args=(spec, ready, probe))
            process.start()
            processes.append(process)
            assert await asyncio.to_thread(ready.wait, STARTUP_TIMEOUT_SECONDS)
            coordinator.register_stage(stage.name, endpoints[stage.name])
        yield coordinator, config, probe
    finally:
        probe.packing_release.set()
        probe.compute_release.set()
        probe.relay_release.set()
        await coordinator.shutdown_stages()
        await coordinator.stop()
        completion_loop.cancel()
        await asyncio.gather(completion_loop, return_exceptions=True)
        for process in processes:
            await asyncio.to_thread(process.join, TIMEOUT_SECONDS)
            if process.is_alive():
                process.kill()
                process.join()
            else:
                pass
            assert process.exitcode == 0
        probe.events.close()
