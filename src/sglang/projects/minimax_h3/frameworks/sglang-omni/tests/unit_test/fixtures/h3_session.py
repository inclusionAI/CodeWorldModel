# SPDX-License-Identifier: Apache-2.0
"""CPU protocol backend with real Omni scheduling and ZMQ session routing."""

from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import uvicorn
import zmq
from fastapi import FastAPI

from sglang_omni.client.client import Client
from sglang_omni.models.minimax_h3.config import H3_STAGE, MiniMaxH3PipelineConfig
from sglang_omni.models.minimax_h3.session import (
    H3MediaBridge,
    H3NextSource,
    H3OpenRequest,
    H3PacketEmitter,
    H3RGBPacket,
    H3RuntimeLimits,
    H3SourceInput,
)
from sglang_omni.models.minimax_h3.stages import create_h3_session_executor
from sglang_omni.pipeline.control_plane import StageControlPlane
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto.messages import CompleteMessage, StreamMessage, SubmitMessage
from sglang_omni.proto.request import StagePayload
from sglang_omni.proto.session import (
    ResourceUsage,
    SessionIdentity,
    SessionOperation,
    find_session_operation,
)
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.serve.session_wire import register_session_wire

CPU_RUNTIME_FACTORY = "tests.unit_test.fixtures.h3_session.create_cpu_test_bridge"
TIMEOUT_SECONDS = 10


@dataclass(kw_only=True)
class CPUState:
    last_end_ms: float = 0
    is_cancelled: bool = False
    is_finished: bool = False
    pending_packets: list[H3RGBPacket] = field(default_factory=list)


class CPUEchoBridge:
    def __init__(self, limits: H3RuntimeLimits) -> None:
        self.limits = limits
        self.states: dict[SessionIdentity, CPUState] = {}
        self.entered = threading.Event()
        self.release = threading.Event()
        self.should_block = False
        self.should_wait_for_cancel = False
        self.fail_open = False
        self.buffer_until_frames = 0
        self.closed: list[SessionIdentity] = []
        self.codec_overlap = limits.codec_overlap
        self.lookahead_sources: list[H3SourceInput | None] = []
        self.completed_sources: list[int] = []
        self.stage_trace: list[tuple[str, int, int | None]] = []
        self.received_operations: list[SessionOperation] = []

    def open(self, session_identity: SessionIdentity, request: H3OpenRequest) -> None:
        self.states[session_identity] = CPUState()
        if self.fail_open:
            raise RuntimeError("CPU provider failed after allocation")
        else:
            pass

    def append(
        self,
        session_identity: SessionIdentity,
        packet: H3RGBPacket,
        cancelled: threading.Event,
        emit: H3PacketEmitter,
    ) -> None:
        state = self.states[session_identity]
        if state.is_cancelled or state.is_finished:
            raise ValueError("CPU test session cannot accept more input")
        else:
            pass
        if self.should_wait_for_cancel:
            self.entered.set()
            assert cancelled.wait(TIMEOUT_SECONDS), "preempt never reached compute"
        elif self.should_block:
            self.entered.set()
            assert self.release.wait(TIMEOUT_SECONDS), "test did not release compute"
        else:
            pass
        if cancelled.is_set():
            state.is_cancelled = True
        elif packet.frames[0].pts_ms < state.last_end_ms:
            raise ValueError("CPU source frame timeline moved backwards")
        else:
            state.last_end_ms = packet.frames[-1].pts_ms + packet.frames[-1].duration_ms
            state.pending_packets.append(packet)
            if (
                sum(len(pending.frames) for pending in state.pending_packets)
                >= self.buffer_until_frames
            ):
                emit(
                    H3RGBPacket(
                        width=packet.width,
                        height=packet.height,
                        frames=tuple(
                            frame
                            for pending in state.pending_packets
                            for frame in pending.frames
                        ),
                        rgb=b"".join(pending.rgb for pending in state.pending_packets),
                    )
                )
                state.pending_packets.clear()
            else:
                pass

    def finish(
        self,
        session_identity: SessionIdentity,
        cancelled: threading.Event,
        emit: H3PacketEmitter,
    ) -> None:
        state = self.states[session_identity]
        if state.is_cancelled or cancelled.is_set():
            state.is_cancelled = True
        else:
            state.is_finished = True

    def advance(
        self,
        session_identity: SessionIdentity,
        source: H3SourceInput,
        cancelled: threading.Event,
        emit: H3PacketEmitter,
        peek_next: H3NextSource | None = None,
    ) -> None:
        self.stage_trace.append(("compute", source.input_seq, None))
        if self.should_wait_for_cancel:
            self.entered.set()
            assert cancelled.wait(TIMEOUT_SECONDS), "preempt never reached compute"
        elif self.should_block:
            self.entered.set()
            assert self.release.wait(TIMEOUT_SECONDS), "test did not release compute"
        else:
            pass
        next_source = peek_next() if peek_next is not None else None
        self.lookahead_sources.append(next_source)
        self.stage_trace.append(
            (
                "deliver",
                source.input_seq,
                None if next_source is None else next_source.input_seq,
            )
        )
        if source.packet is not None:
            self.append(session_identity, source.packet, cancelled, emit)
        else:
            pass
        if source.eos:
            self.finish(session_identity, cancelled, emit)
        else:
            pass
        self.completed_sources.append(source.input_seq)

    def cancel(self, session_identity: SessionIdentity) -> None:
        self.states[session_identity].is_cancelled = True

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)
        self.closed.append(session_identity)

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        return ResourceUsage(bytes=len(self.states) * 64)


def create_cpu_test_bridge(
    *, model_path: str, limits: H3RuntimeLimits
) -> H3MediaBridge:
    return CPUEchoBridge(limits)


def cpu_config() -> MiniMaxH3PipelineConfig:
    return MiniMaxH3PipelineConfig(
        model_path="cpu-protocol-test",
        runtime_factory=CPU_RUNTIME_FACTORY,
        max_width=2,
        max_height=2,
        max_state_bytes=1024,
        max_output_chunks=32,
        operation_timeout_seconds=TIMEOUT_SECONDS,
    )


@asynccontextmanager
async def cpu_pipeline(
    config: MiniMaxH3PipelineConfig,
) -> AsyncIterator[tuple[Coordinator, CPUEchoBridge]]:
    """Use real framework components without the Linux-only Stage platform imports."""
    scheduler = create_h3_session_executor(
        config.model_path, **config.stage_factory_kwargs(H3_STAGE)
    )
    bridge = scheduler.session_hooks.bridge
    assert isinstance(bridge, CPUEchoBridge)
    coordinator = Coordinator(
        "tcp://127.0.0.1:*",
        "tcp://127.0.0.1:*",
        H3_STAGE,
        [H3_STAGE],
        session_graph=config.session_graph(),
    )
    await coordinator.start()
    coordinator_plane = coordinator.control_plane
    stage_plane = StageControlPlane(
        H3_STAGE,
        "tcp://127.0.0.1:*",
        coordinator_plane.completion_socket.socket.getsockopt_string(zmq.LAST_ENDPOINT),
        coordinator_plane.abort_socket.socket.getsockopt_string(zmq.LAST_ENDPOINT),
    )
    await stage_plane.start()
    coordinator.register_stage(
        H3_STAGE, stage_plane.recv_socket.socket.getsockopt_string(zmq.LAST_ENDPOINT)
    )
    scheduler_thread = threading.Thread(target=scheduler.start, daemon=True)
    scheduler_thread.start()

    async def receive_operations() -> None:
        while True:
            message = await stage_plane.recv()
            assert isinstance(message, SubmitMessage)
            operation = find_session_operation(message.data.request.metadata)
            if operation is not None:
                bridge.received_operations.append(operation)
                for request_id in operation.preempted_request_ids:
                    scheduler.abort(request_id)
            else:
                pass
            scheduler.inbox.put(
                IncomingMessage(message.request_id, "new_request", message.data)
            )

    async def receive_aborts() -> None:
        while True:
            message = await stage_plane.recv_abort()
            scheduler.abort(message.request_id)

    async def forward_outputs() -> None:
        while True:
            message = await asyncio.to_thread(scheduler.outbox.get)
            if message.type == "wakeup":
                return
            elif message.type == "stream":
                await stage_plane.send_stream(
                    StreamMessage(message.request_id, H3_STAGE, message.data)
                )
            elif message.type == "result":
                assert isinstance(message.data, StagePayload)
                await stage_plane.send_complete(
                    CompleteMessage(
                        message.request_id, H3_STAGE, True, result=message.data.data
                    )
                )
            elif message.type == "error":
                await stage_plane.send_complete(
                    CompleteMessage(
                        message.request_id, H3_STAGE, False, error=str(message.data)
                    )
                )
            else:
                pass

    tasks = [
        asyncio.create_task(coordinator.run_completion_loop()),
        asyncio.create_task(receive_operations()),
        asyncio.create_task(receive_aborts()),
        asyncio.create_task(forward_outputs()),
    ]
    try:
        yield coordinator, bridge
    finally:
        bridge.release.set()
        await coordinator.stop()
        scheduler.stop()
        scheduler.outbox.put(OutgoingMessage("", "wakeup"))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.to_thread(scheduler_thread.join, TIMEOUT_SECONDS)
        assert not scheduler_thread.is_alive()
        stage_plane.close()


@asynccontextmanager
async def cpu_session_server(
    coordinator: Coordinator, config: MiniMaxH3PipelineConfig
) -> AsyncIterator[str]:
    app = FastAPI()
    app.state.client = Client(coordinator)
    register_session_wire(app, config.session_wire_limits())
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        task = asyncio.create_task(server.serve(sockets=[listener]))

        async def ready() -> None:
            while not server.started:
                if task.done():
                    await task
                else:
                    await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(ready(), TIMEOUT_SECONDS)
            yield f"ws://127.0.0.1:{port}/v1/session"
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, TIMEOUT_SECONDS)
