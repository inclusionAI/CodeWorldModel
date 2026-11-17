# SPDX-License-Identifier: Apache-2.0
"""H3 protocol acceptance through real Omni session and router components."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import msgpack
import pytest
from pydantic import ValidationError
from websockets.asyncio.client import connect

from sglang_omni.admission import QueueFullError
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.config.schema import SessionDataEdgeConfig
from sglang_omni.models.minimax_h3.config import H3_STAGE, MiniMaxH3PipelineConfig
from sglang_omni.models.minimax_h3.session import (
    H3_CANCEL,
    H3_RGB24,
    H3FrameStamp,
    H3RGBPacket,
)
from sglang_omni.models.minimax_h3.stages import create_h3_session_executor
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import SESSION_CONTROL
from tests.unit_test.fixtures.h3_session import (
    TIMEOUT_SECONDS,
    cpu_config,
    cpu_pipeline,
    cpu_session_server,
)


def packet(sequence: int) -> H3RGBPacket:
    return H3RGBPacket(
        width=2,
        height=2,
        frames=(
            H3FrameStamp(
                source_frame_id=f"source-{sequence}",
                pts_ms=sequence * 40.0,
                duration_ms=40.0,
            ),
        ),
        rgb=bytes([sequence % 256]) * 12,
    )


def request() -> OmniRequest:
    return OmniRequest(
        None,
        params={"prompt": "test", "first_image": packet(0).wire_payload(), "seed": 7},
    )


def chunk(sequence: int, *, eos: bool = False) -> TimedChunk:
    video = packet(sequence)
    return TimedChunk(
        "video",
        video.frames[0].pts_ms,
        video.duration_ms,
        sequence,
        video.wire_payload(),
        H3_RGB24,
        eos,
    )


@pytest.mark.asyncio
async def test_real_sessions_preserve_source_order_eos_and_reopen() -> None:
    config = cpu_config()
    async with cpu_pipeline(config) as (coordinator, bridge):
        identity = await coordinator.open_session(
            request(), limits=config.session_wire_limits(), session_id="h3"
        )
        outputs = coordinator.session_outputs(identity)
        for sequence in range(2):
            await coordinator.append_session(
                identity, chunk(sequence, eos=sequence == 1)
            )
        received = [
            await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS) for _ in range(5)
        ]
        video = [
            output
            for output in received
            if output.kind == "data" and output.payload is not None
        ]
        assert [output.input_seq for output in video] == [0, 1]
        assert [output.payload["frames"][0]["source_frame_id"] for output in video] == [
            "source-0",
            "source-1",
        ]
        assert [output.payload["rgb"] for output in video] == [
            packet(0).rgb,
            packet(1).rgb,
        ]
        assert [(output.t_start_ms, output.duration_ms) for output in video] == [
            (0, 40),
            (40, 40),
        ]
        assert received[-1].kind == "input_done" and received[-1].eos
        assert sum(output.eos for output in received if output.kind == "data") == 1
        with pytest.raises(ValueError, match="EOS"):
            await coordinator.append_session(identity, chunk(2))
        await outputs.aclose()
        await coordinator.close_session(identity)
        assert not bridge.states
        reopened = await coordinator.open_session(
            request(), limits=config.session_wire_limits(), session_id="h3"
        )
        assert reopened.open_index > identity.open_index
        await coordinator.append_session(reopened, chunk(0))
        new_outputs = coordinator.session_outputs(reopened)
        assert (
            await asyncio.wait_for(anext(new_outputs), TIMEOUT_SECONDS)
        ).input_seq == 0
        await new_outputs.aclose()
        await coordinator.close_session(reopened)
        assert not bridge.states


@pytest.mark.asyncio
async def test_backpressure_rejects_before_acceptance_and_retry_loses_no_frames() -> (
    None
):
    config = cpu_config()
    async with cpu_pipeline(config) as (coordinator, bridge):
        bridge.should_block = True
        identity = await coordinator.open_session(
            request(), limits=config.session_wire_limits()
        )
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, chunk(0))
        assert await asyncio.to_thread(bridge.entered.wait, TIMEOUT_SECONDS)
        accepted_count = 1
        for sequence in range(1, 20):
            try:
                await coordinator.append_session(identity, chunk(sequence))
            except QueueFullError:
                break
            else:
                accepted_count += 1
        assert accepted_count < 20
        bridge.release.set()
        received = [
            await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS)
            for _ in range(accepted_count * 2)
        ]
        assert [
            output.input_seq for output in received if output.kind == "data"
        ] == list(range(accepted_count))
        await coordinator.append_session(identity, chunk(accepted_count, eos=True))
        retried = [
            await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS) for _ in range(3)
        ]
        assert (
            retried[0].payload["frames"][0]["source_frame_id"]
            == f"source-{accepted_count}"
        )
        assert retried[-1].input_seq == accepted_count and retried[-1].eos
        await outputs.aclose()
        await coordinator.close_session(identity)


@pytest.mark.asyncio
async def test_preempt_control_reaches_inflight_compute_and_reset_requires_reopen() -> (
    None
):
    config = cpu_config()
    async with cpu_pipeline(config) as (coordinator, bridge):
        bridge.should_wait_for_cancel = True
        identity = await coordinator.open_session(
            request(), limits=config.session_wire_limits()
        )
        await coordinator.append_session(identity, chunk(0))
        assert await asyncio.to_thread(bridge.entered.wait, TIMEOUT_SECONDS)
        cancel = TimedChunk("control", 0, 0, 0, {"type": H3_CANCEL}, SESSION_CONTROL)
        await asyncio.wait_for(
            coordinator.control_session(identity, cancel), TIMEOUT_SECONDS
        )
        assert all(state.is_cancelled for state in bridge.states.values())
        with pytest.raises(ValueError, match="no node takes"):
            await coordinator.control_session(
                identity, replace(cancel, payload={"type": "h3.reset"})
            )
        outputs = coordinator.session_outputs(identity)
        assert (
            await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS)
        ).kind == "input_done"
        await outputs.aclose()
        await coordinator.close_session(identity)
        assert not bridge.states


@pytest.mark.asyncio
async def test_empty_eos_finishes_and_invalid_rgb_closes_owned_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = cpu_config()
    async with cpu_pipeline(config) as (coordinator, bridge):
        identity = await coordinator.open_session(
            request(), limits=config.session_wire_limits()
        )
        await coordinator.append_session(
            identity, TimedChunk("video", 0, 0, 0, None, H3_RGB24, True)
        )
        outputs = coordinator.session_outputs(identity)
        assert (await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS)).eos
        assert (
            await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS)
        ).kind == "input_done"
        await outputs.aclose()
        await coordinator.close_session(identity)
        identity = await coordinator.open_session(
            request(), limits=config.session_wire_limits()
        )
        malformed = chunk(0).payload | {"rgb": b"wrong"}
        await coordinator.append_session(identity, replace(chunk(0), payload=malformed))
        outputs = coordinator.session_outputs(identity)
        with pytest.raises(RuntimeError, match="closed|RGB byte count"):
            await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS)
        assert "RGB byte count" in caplog.text
        assert not bridge.states


def test_config_round_trip_forbids_dropping_history_and_missing_provider() -> None:
    config = cpu_config()
    restored = MiniMaxH3PipelineConfig.model_validate(config.model_dump())
    assert restored.stage_factory_kwargs(H3_STAGE) == config.stage_factory_kwargs(
        H3_STAGE
    )
    document = config.model_dump()
    document["session_nodes"][H3_STAGE]["inputs"][0] = SessionDataEdgeConfig(
        source="input", overflow="drop_oldest"
    ).model_dump()
    with pytest.raises(ValidationError, match="blocking"):
        MiniMaxH3PipelineConfig.model_validate(document)
    document = config.model_dump()
    document.pop("runtime_factory")
    with pytest.raises(ValidationError, match="runtime_factory"):
        MiniMaxH3PipelineConfig.model_validate(document)
    settings = config.stage_factory_kwargs(H3_STAGE) | {
        "runtime_factory": "h3_provider_missing.create"
    }
    with pytest.raises(ModuleNotFoundError, match="h3_provider_missing"):
        create_h3_session_executor(config.model_path, **settings)


@pytest.mark.parametrize("invalid_part", ["cancel", "publish", "factory"])
def test_config_requires_causal_session_routes(invalid_part: str) -> None:
    document = cpu_config().model_dump()
    if invalid_part == "cancel":
        document["session_nodes"][H3_STAGE]["inputs"].pop()
    elif invalid_part == "publish":
        document["session_publish"] = {}
    else:
        document["stages"][0]["factory_path"] = "another.factory"
    with pytest.raises(ValidationError, match="H3 requires"):
        MiniMaxH3PipelineConfig.model_validate(document)


def test_real_factory_resolution_forwards_gpu_placement() -> None:
    config = cpu_config()
    settings = resolve_stage_factory_args(config.stages[0], config)
    assert settings["gpu_id"] == 0
    scheduler = create_h3_session_executor(**settings, runtime_gpu_ids=[0, 1])
    assert scheduler.session_hooks.limits.gpu_id == 0
    assert scheduler.session_hooks.limits.runtime_gpu_ids == (0, 1)


@pytest.mark.asyncio
async def test_failed_open_releases_partial_state_and_allows_new_session() -> None:
    config = cpu_config()
    async with cpu_pipeline(config) as (coordinator, bridge):
        oversized = packet(0).wire_payload() | {"width": 3, "rgb": bytes(18)}
        invalid = request()
        invalid.params["first_image"] = oversized
        with pytest.raises(RuntimeError, match="geometry or frame budget"):
            await coordinator.open_session(invalid, limits=config.session_wire_limits())
        assert not bridge.states and not bridge.closed
        bridge.fail_open = True
        with pytest.raises(RuntimeError, match="CPU provider failed"):
            await coordinator.open_session(
                request(), limits=config.session_wire_limits()
            )
        assert not bridge.states
        assert len(bridge.closed) == 1
        bridge.fail_open = False
        identity = await coordinator.open_session(
            request(), limits=config.session_wire_limits()
        )
        await coordinator.close_session(identity)
        assert not bridge.states


@pytest.mark.asyncio
async def test_session_wire_uses_real_client_for_accept_output_control_and_close() -> (
    None
):
    config = cpu_config()
    async with cpu_pipeline(config) as (coordinator, bridge):
        async with cpu_session_server(coordinator, config) as endpoint:
            async with connect(endpoint) as websocket:
                await websocket.send(
                    msgpack.packb({"op": "open", "params": request().params})
                )
                opened = msgpack.unpackb(await websocket.recv(), raw=False)
                assert opened["op"] == "opened"
                bridge.should_block = True
                for sequence in range(2):
                    await websocket.send(
                        msgpack.packb(
                            {"op": "append", "chunk": chunk(sequence).to_dict()}
                        )
                    )
                    assert msgpack.unpackb(await websocket.recv(), raw=False) == {
                        "op": "accepted",
                        "seq": sequence,
                    }
                assert await asyncio.to_thread(bridge.entered.wait, TIMEOUT_SECONDS)
                bridge.release.set()
                received = [
                    msgpack.unpackb(
                        await asyncio.wait_for(websocket.recv(), TIMEOUT_SECONDS),
                        raw=False,
                    )
                    for _ in range(4)
                ]
                video = [
                    message["chunk"]
                    for message in received
                    if message["chunk"]["kind"] == "data"
                ]
                assert [output["input_seq"] for output in video] == [0, 1]
                assert [
                    output["payload"]["frames"][0]["source_frame_id"]
                    for output in video
                ] == ["source-0", "source-1"]
                end = TimedChunk("video", 80, 0, 2, None, H3_RGB24, True)
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
                assert (
                    sum(
                        message["op"] == "output"
                        and message["chunk"]["kind"] == "data"
                        and message["chunk"]["eos"]
                        for message in ended
                    )
                    == 1
                )
                await websocket.send(msgpack.packb({"op": "close"}))
                assert msgpack.unpackb(await websocket.recv(), raw=False) == {
                    "op": "closed"
                }
                assert not bridge.states
            async with connect(endpoint) as websocket:
                await websocket.send(
                    msgpack.packb({"op": "open", "params": request().params})
                )
                reopened = msgpack.unpackb(await websocket.recv(), raw=False)
                assert (
                    reopened["session"]["open_index"] > opened["session"]["open_index"]
                )
                cancel = TimedChunk(
                    "control", 0, 0, 0, {"type": H3_CANCEL}, SESSION_CONTROL
                )
                await websocket.send(
                    msgpack.packb({"op": "control", "chunk": cancel.to_dict()})
                )
                assert msgpack.unpackb(await websocket.recv(), raw=False) == {
                    "op": "controlled",
                    "seq": 0,
                }
                await websocket.send(msgpack.packb({"op": "close"}))
                assert msgpack.unpackb(await websocket.recv(), raw=False) == {
                    "op": "closed"
                }
                assert not bridge.states
