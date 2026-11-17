# SPDX-License-Identifier: Apache-2.0
"""Bounded append lookahead through real coordinator, scheduler and wire paths."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import msgpack
import pytest
from websockets.asyncio.client import connect

from sglang_omni.admission import QueueFullError
from sglang_omni.models.minimax_h3.config import H3_STAGE
from sglang_omni.models.minimax_h3.session import (
    H3_CANCEL,
    H3_RGB24,
    H3FrameStamp,
    H3RGBPacket,
    H3SessionHooks,
)
from sglang_omni.models.minimax_h3.stages import create_h3_session_executor
from sglang_omni.pipeline.sessions import Session
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import (
    SessionIdentity,
    SessionOperationKind,
    TimedChunk,
    wire_size,
)
from sglang_omni.proto.streams import SESSION_CONTROL
from tests.unit_test.fixtures.h3_session import (
    TIMEOUT_SECONDS,
    cpu_config,
    cpu_pipeline,
    cpu_session_server,
)


def source_chunk(
    sequence: int, *, start: int | None = None, frames: int = 1
) -> TimedChunk:
    first_frame = sequence if start is None else start
    packet = H3RGBPacket(
        width=2,
        height=2,
        frames=tuple(
            H3FrameStamp(
                source_frame_id=f"source-{frame}", pts_ms=frame * 40.0, duration_ms=40.0
            )
            for frame in range(first_frame, first_frame + frames)
        ),
        rgb=bytes([sequence]) * (12 * frames),
    )
    return TimedChunk(
        "video",
        packet.frames[0].pts_ms,
        packet.duration_ms,
        sequence,
        packet.wire_payload(),
        H3_RGB24,
        turn=sequence + 7,
    )


def conditioning() -> OmniRequest:
    return OmniRequest(
        None,
        params={
            "prompt": "lookahead",
            "first_image": source_chunk(0).payload,
            "seed": 7,
        },
    )


@pytest.fixture
def arrivals(monkeypatch: pytest.MonkeyPatch) -> dict[int, threading.Event]:
    arrived = {sequence: threading.Event() for sequence in range(4)}
    original = H3SessionHooks.on_append_arrival

    def observe(
        self: H3SessionHooks,
        identity: SessionIdentity,
        request_id: str,
        chunk: TimedChunk,
    ) -> None:
        try:
            original(self, identity, request_id, chunk)
        finally:
            arrived[chunk.seq].set()

    monkeypatch.setattr(H3SessionHooks, "on_append_arrival", observe)
    return arrived


@pytest.mark.asyncio
async def test_late_arrival_refills_two_slots_and_bytes_stay_charged(arrivals) -> None:
    config = cpu_config().model_copy(
        update={"codec_overlap": True, "max_pending_chunks": 3}
    )
    async with cpu_pipeline(config) as (coordinator, bridge):
        bridge.should_block = True
        identity = await coordinator.open_session(
            conditioning(), stages=[H3_STAGE], limits=config.session_wire_limits()
        )
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, source_chunk(0))
        assert await asyncio.to_thread(bridge.entered.wait, TIMEOUT_SECONDS)
        await coordinator.append_session(identity, source_chunk(1))
        assert await asyncio.to_thread(arrivals[1].wait, TIMEOUT_SECONDS)
        await coordinator.append_session(identity, source_chunk(2))
        session = coordinator.get_session(identity)
        assert [chunk.seq for chunk in session.submitted_appends.values()] == [0, 1]
        assert session.pending_count == 3
        assert session.pending_bytes == sum(
            wire_size(source_chunk(sequence).to_dict()) for sequence in range(3)
        )
        assert not arrivals[2].is_set()
        assert bridge.completed_sources == []
        bridge.release.set()
        received = [
            await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS) for _ in range(6)
        ]
        for kind in ("data", "input_done"):
            assert [output.input_seq for output in received if output.kind == kind] == [
                0,
                1,
                2,
            ]
        for sequence in range(3):
            assert [
                output.kind for output in received if output.input_seq == sequence
            ] == ["data", "input_done"]
        assert bridge.lookahead_sources[0].input_seq == 1
        assert bridge.stage_trace[:3] == [
            ("compute", 0, None),
            ("deliver", 0, 1),
            ("compute", 1, None),
        ]
        assert session.pending_count == session.pending_bytes == 0
        await outputs.aclose()
        assert not session.submitted_appends
        assert not coordinator.completion_futures


@pytest.mark.asyncio
async def test_ordered_control_fences_later_arrival(arrivals, monkeypatch) -> None:
    config = cpu_config().model_copy(update={"codec_overlap": True})
    control_seen = threading.Event()

    def control(
        self: H3SessionHooks, identity: SessionIdentity, event: TimedChunk
    ) -> None:
        assert not arrivals[1].is_set()
        control_seen.set()

    monkeypatch.setattr(H3SessionHooks, "control", control)
    async with cpu_pipeline(config) as (coordinator, bridge):
        bridge.should_block = True
        identity = await coordinator.open_session(
            conditioning(), stages=[H3_STAGE], limits=config.session_wire_limits()
        )
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, source_chunk(0))
        assert await asyncio.to_thread(bridge.entered.wait, TIMEOUT_SECONDS)
        event = TimedChunk("control", 0, 0, 0, {"type": H3_CANCEL}, SESSION_CONTROL)
        controlled = asyncio.create_task(coordinator.control_session(identity, event))
        await asyncio.sleep(0)
        assert coordinator.get_session(identity).control_barriers
        await coordinator.append_session(identity, source_chunk(1))
        assert not arrivals[1].is_set()
        bridge.release.set()
        await asyncio.wait_for(controlled, TIMEOUT_SECONDS)
        received = [
            await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS) for _ in range(4)
        ]
        assert control_seen.is_set()
        assert arrivals[1].is_set()
        assert bridge.lookahead_sources[0] is None
        assert [
            output.input_seq for output in received if output.kind == "input_done"
        ] == [0, 1]
        await outputs.aclose()


@pytest.mark.asyncio
async def test_completions_remain_fifo_when_later_request_finishes_first(
    monkeypatch,
) -> None:
    config = cpu_config().model_copy(update={"codec_overlap": True})
    async with cpu_pipeline(config) as (coordinator, bridge):
        oldest_finished, later_finished, release = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )
        original = coordinator.finish_session_operation

        async def finish(
            session: Session, operation: SessionOperationKind, request_id: str
        ) -> None:
            chunk = session.submitted_appends.get(request_id)
            await original(session, operation, request_id)
            if operation == "append" and chunk.seq == 0:
                oldest_finished.set()
                await release.wait()
            elif operation == "append":
                later_finished.set()
            else:
                pass

        monkeypatch.setattr(coordinator, "finish_session_operation", finish)
        identity = await coordinator.open_session(
            conditioning(), stages=[H3_STAGE], limits=config.session_wire_limits()
        )
        try:
            await coordinator.append_session(identity, source_chunk(0))
            await coordinator.append_session(identity, source_chunk(1))
            await asyncio.wait_for(oldest_finished.wait(), TIMEOUT_SECONDS)
            await asyncio.wait_for(later_finished.wait(), TIMEOUT_SECONDS)
            session = coordinator.get_session(identity)
            assert bridge.completed_sources == [0, 1]
            assert session.pending_count == 2
            assert [output.kind for output, _ in session.outputs] == ["data", "data"]
            release.set()
            outputs = coordinator.session_outputs(identity)
            received = [
                await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS)
                for _ in range(4)
            ]
            assert [
                output.input_seq for output in received if output.kind == "input_done"
            ] == [0, 1]
            assert session.pending_count == session.pending_bytes == 0
            await outputs.aclose()
        finally:
            release.set()


@pytest.mark.asyncio
async def test_close_wakes_and_releases_both_submitted_appends(arrivals) -> None:
    config = cpu_config().model_copy(update={"codec_overlap": True})
    async with cpu_pipeline(config) as (coordinator, bridge):
        bridge.should_wait_for_cancel = True
        identity = await coordinator.open_session(
            conditioning(), stages=[H3_STAGE], limits=config.session_wire_limits()
        )
        await coordinator.append_session(identity, source_chunk(0))
        assert await asyncio.to_thread(bridge.entered.wait, TIMEOUT_SECONDS)
        await coordinator.append_session(identity, source_chunk(1))
        assert await asyncio.to_thread(arrivals[1].wait, TIMEOUT_SECONDS)
        session = coordinator.get_session(identity)
        await asyncio.wait_for(coordinator.close_session(identity), TIMEOUT_SECONDS)
        assert session.is_closed
        assert session.pending_count == session.pending_bytes == 0
        assert not session.submitted_appends
        assert not coordinator.completion_futures
        assert not bridge.states


@pytest.mark.asyncio
async def test_absent_output_consumer_keeps_overflow_bounded(arrivals) -> None:
    config = cpu_config().model_copy(
        update={"codec_overlap": True, "max_output_chunks": 1}
    )
    async with cpu_pipeline(config) as (coordinator, bridge):
        bridge.should_block = True
        identity = await coordinator.open_session(
            conditioning(), stages=[H3_STAGE], limits=config.session_wire_limits()
        )
        await coordinator.append_session(identity, source_chunk(0))
        assert await asyncio.to_thread(bridge.entered.wait, TIMEOUT_SECONDS)
        await coordinator.append_session(identity, source_chunk(1))
        assert await asyncio.to_thread(arrivals[1].wait, TIMEOUT_SECONDS)
        session = coordinator.get_session(identity)
        bridge.release.set()
        await asyncio.wait_for(session.pump, TIMEOUT_SECONDS)
        assert isinstance(session.error, QueueFullError)
        await coordinator.close_session(identity)
        assert not session.outputs
        assert not coordinator.completion_futures
        assert not bridge.states


@pytest.mark.asyncio
async def test_preempt_names_current_and_queued_lookahead(arrivals) -> None:
    config = cpu_config().model_copy(update={"codec_overlap": True})
    async with cpu_pipeline(config) as (coordinator, bridge):
        bridge.should_wait_for_cancel = True
        identity = await coordinator.open_session(
            conditioning(), stages=[H3_STAGE], limits=config.session_wire_limits()
        )
        await coordinator.append_session(identity, source_chunk(0))
        assert await asyncio.to_thread(bridge.entered.wait, TIMEOUT_SECONDS)
        await coordinator.append_session(identity, source_chunk(1))
        assert await asyncio.to_thread(arrivals[1].wait, TIMEOUT_SECONDS)
        submitted = set(coordinator.get_session(identity).submitted_appends)
        assert len(submitted) == 2
        event = TimedChunk("control", 0, 0, 0, {"type": H3_CANCEL}, SESSION_CONTROL)
        await asyncio.wait_for(
            coordinator.control_session(identity, event, should_preempt=True),
            TIMEOUT_SECONDS,
        )
        operation = next(
            operation
            for operation in bridge.received_operations
            if operation.operation == "control"
        )
        assert set(operation.preempted_request_ids) == submitted
        outputs = coordinator.session_outputs(identity)
        received = [
            await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS) for _ in range(2)
        ]
        assert [(output.kind, output.input_seq) for output in received] == [
            ("input_done", 0),
            ("input_done", 1),
        ]
        assert bridge.completed_sources == [0]
        assert all(state.is_cancelled for state in bridge.states.values())
        await outputs.aclose()
        assert not coordinator.completion_futures


@pytest.mark.asyncio
async def test_malformed_lookahead_fails_accepted_session_and_reopens(arrivals) -> None:
    config = cpu_config().model_copy(update={"codec_overlap": True})
    async with cpu_pipeline(config) as (coordinator, bridge):
        bridge.should_block = True
        identity = await coordinator.open_session(
            conditioning(),
            stages=[H3_STAGE],
            limits=config.session_wire_limits(),
            session_id="lookahead",
        )
        await coordinator.append_session(identity, source_chunk(0))
        assert await asyncio.to_thread(bridge.entered.wait, TIMEOUT_SECONDS)
        malformed = replace(
            source_chunk(1), payload=source_chunk(1).payload | {"rgb": b"wrong"}
        )
        assert await coordinator.append_session(identity, malformed) == 1
        assert await asyncio.to_thread(arrivals[1].wait, TIMEOUT_SECONDS)
        bridge.release.set()
        outputs = coordinator.session_outputs(identity)
        with pytest.raises(RuntimeError, match="RGB byte count"):
            await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS)
        assert not bridge.states
        reopened = await coordinator.open_session(
            conditioning(),
            stages=[H3_STAGE],
            limits=config.session_wire_limits(),
            session_id="lookahead",
        )
        assert reopened.open_index > identity.open_index
        await coordinator.append_session(reopened, source_chunk(0))
        outputs = coordinator.session_outputs(reopened)
        assert (await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS)).input_seq == 0
        await outputs.aclose()


def test_arrival_handoff_bounds_identity_and_rgb_ownership() -> None:
    config = cpu_config().model_copy(update={"codec_overlap": True})
    scheduler = create_h3_session_executor(
        config.model_path, **config.stage_factory_kwargs(H3_STAGE)
    )
    hooks = scheduler.session_hooks
    assert isinstance(hooks, H3SessionHooks)
    identity = SessionIdentity("bounded", 1)
    hooks.open(identity, conditioning())
    first = source_chunk(0)
    hooks.on_append_arrival(identity, "first", first)
    hooks.on_append_arrival(identity, "second", source_chunk(1))
    assert hooks.input_handoffs[identity].inputs[0].packet.rgb is first.payload["rgb"]
    with pytest.raises(ValueError, match="lookahead is full"):
        hooks.on_append_arrival(identity, "third", source_chunk(2))
    hooks.close(identity)
    reopened = SessionIdentity("bounded", 2)
    hooks.open(reopened, conditioning())
    with pytest.raises(ValueError, match="stale"):
        hooks.on_append_arrival(identity, "stale", source_chunk(0))
    hooks.on_append_arrival(reopened, "fresh", source_chunk(0))
    hooks.close(reopened)
    assert not hooks.input_handoffs


@pytest.mark.asyncio
async def test_wire_next_packet_arrives_during_compute_with_original_origin(
    arrivals,
) -> None:
    config = cpu_config().model_copy(update={"codec_overlap": True})
    async with cpu_pipeline(config) as (coordinator, bridge):
        async with cpu_session_server(coordinator, config) as endpoint:
            async with connect(endpoint) as websocket:
                await websocket.send(
                    msgpack.packb({"op": "open", "params": conditioning().params})
                )
                assert (
                    msgpack.unpackb(await websocket.recv(), raw=False)["op"] == "opened"
                )
                bridge.should_block = True
                for sequence, start, frames in ((0, 0, 17), (1, 17, 5)):
                    await websocket.send(
                        msgpack.packb(
                            {
                                "op": "append",
                                "chunk": source_chunk(
                                    sequence, start=start, frames=frames
                                ).to_dict(),
                            }
                        )
                    )
                    assert msgpack.unpackb(await websocket.recv(), raw=False) == {
                        "op": "accepted",
                        "seq": sequence,
                    }
                    if sequence == 0:
                        assert await asyncio.to_thread(
                            bridge.entered.wait, TIMEOUT_SECONDS
                        )
                    else:
                        assert await asyncio.to_thread(
                            arrivals[1].wait, TIMEOUT_SECONDS
                        )
                assert bridge.completed_sources == []
                bridge.release.set()
                received = [
                    msgpack.unpackb(
                        await asyncio.wait_for(websocket.recv(), TIMEOUT_SECONDS),
                        raw=False,
                    )["chunk"]
                    for _ in range(4)
                ]
                assert [
                    (output["input_seq"], output["turn"])
                    for output in received
                    if output["kind"] == "data"
                ] == [(0, 7), (1, 8)]
                assert [
                    output["input_seq"]
                    for output in received
                    if output["kind"] == "input_done"
                ] == [0, 1]
                for sequence in range(2):
                    assert [
                        output["kind"]
                        for output in received
                        if output["input_seq"] == sequence
                    ] == ["data", "input_done"]
                assert not any(output["eos"] for output in received)
                assert [
                    output["payload"]["frames"][0]["source_frame_id"]
                    for output in received
                    if output["kind"] == "data"
                ] == ["source-0", "source-17"]
                assert bridge.lookahead_sources[0].input_seq == 1
                assert not bridge.lookahead_sources[0].eos
                assert bridge.completed_sources == [0, 1]
                end = TimedChunk("video", 880, 0, 2, None, H3_RGB24, True, turn=9)
                await websocket.send(
                    msgpack.packb({"op": "append", "chunk": end.to_dict()})
                )
                ended = [
                    msgpack.unpackb(
                        await asyncio.wait_for(websocket.recv(), TIMEOUT_SECONDS),
                        raw=False,
                    )
                    for _ in range(3)
                ]
                assert {"op": "accepted", "seq": 2} in ended
                assert [
                    (message["chunk"]["kind"], message["chunk"]["input_seq"])
                    for message in ended
                    if message["op"] == "output"
                ] == [("data", 2), ("input_done", 2)]
                assert bridge.completed_sources == [0, 1, 2]
                await websocket.send(msgpack.packb({"op": "close"}))
                assert msgpack.unpackb(await websocket.recv(), raw=False) == {
                    "op": "closed"
                }
                assert not bridge.states
