# SPDX-License-Identifier: Apache-2.0
"""CPU delivery preserves append ownership across the real SHM stage route."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pydantic import ValidationError

from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.minimax_h3.config import (
    H3_RGB_DELIVERY_STAGE,
    H3_STAGE,
    MiniMaxH3PipelineConfig,
)
from sglang_omni.models.minimax_h3.delivery import (
    H3RGBDeliveryHooks,
    create_h3_rgb_delivery_executor,
)
from sglang_omni.models.minimax_h3.session import (
    H3_CANCEL,
    H3_RGB24,
    H3FrameStamp,
    H3RGBPacket,
    H3RuntimeLimits,
)
from sglang_omni.models.minimax_h3.stages import create_h3_session_executor
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    SessionIdentity,
    SessionOperation,
    TimedChunk,
)
from sglang_omni.proto.streams import SESSION_CONTROL
from sglang_omni.scheduling.session import SessionContext
from tests.unit_test.fixtures.h3_cpu_delivery import (
    RUNTIME_FACTORY,
    TIMEOUT_SECONDS,
    cpu_delivery_pipeline,
)


def source_chunk(sequence: int, *, eos: bool = False) -> TimedChunk:
    packet = H3RGBPacket(
        width=2,
        height=2,
        frames=(
            H3FrameStamp(
                source_frame_id=f"source-{sequence}",
                pts_ms=sequence * 40.0,
                duration_ms=40.0,
            ),
        ),
        rgb=bytes([sequence]) * 12,
    )
    return TimedChunk(
        "video",
        sequence * 40.0,
        40.0,
        sequence,
        packet.wire_payload(),
        H3_RGB24,
        eos=eos,
        turn=sequence + 7,
    )


def conditioning() -> OmniRequest:
    return OmniRequest(
        None,
        params={
            "prompt": "delivery",
            "first_image": source_chunk(0).payload,
            "seed": 7,
        },
    )


def config() -> MiniMaxH3PipelineConfig:
    return MiniMaxH3PipelineConfig(
        model_path="cpu-delivery-test",
        runtime_factory=RUNTIME_FACTORY,
        max_state_bytes=1024,
        max_width=2,
        max_height=2,
        cpu_delivery=True,
        codec_overlap=True,
    )


def test_candidate_config_round_trip_and_stage_specific_factory_arguments() -> None:
    settings = config()
    restored = MiniMaxH3PipelineConfig.model_validate(settings.model_dump())
    assert restored.session_graph().nodes[H3_STAGE].stages == (
        H3_STAGE,
        H3_RGB_DELIVERY_STAGE,
    )
    gpu_arguments = resolve_stage_factory_args(restored.stages[0], restored)
    gpu_scheduler = create_h3_session_executor(**gpu_arguments)
    assert gpu_scheduler.session_hooks.limits.cpu_delivery
    cpu_arguments = resolve_stage_factory_args(restored.stages[1], restored)
    assert set(cpu_arguments) == {
        "max_width",
        "max_height",
        "max_output_frames",
        "max_state_bytes",
    }
    cpu_scheduler = create_h3_rgb_delivery_executor(**cpu_arguments)
    assert cpu_scheduler.session_hooks.limits.gpu_id is None
    baseline = MiniMaxH3PipelineConfig(
        model_path="legacy", runtime_factory=RUNTIME_FACTORY, max_state_bytes=1024
    )
    assert not baseline.cpu_delivery
    assert baseline.session_graph().nodes[H3_STAGE].stages == (H3_STAGE,)


@pytest.mark.parametrize(
    "invalid", ["overlap", "gpu", "runtime_gpu", "process", "route"]
)
def test_candidate_rejects_wrong_cpu_placement_or_route(invalid: str) -> None:
    document = config().model_dump()
    if invalid == "overlap":
        document["codec_overlap"] = False
    elif invalid == "gpu":
        document["stages"][1]["gpu"] = 0
    elif invalid == "runtime_gpu":
        document["stages"][1]["runtime_gpu_ids"] = [0]
    elif invalid == "process":
        document["stages"][1]["process"] = H3_STAGE
    else:
        document["session_nodes"][H3_STAGE]["stages"] = [H3_STAGE]
    with pytest.raises(ValidationError):
        MiniMaxH3PipelineConfig.model_validate(document)


def test_gpu_handoff_keeps_accepted_request_and_lookahead_rgb_intact() -> None:
    settings = config()
    scheduler = create_h3_session_executor(
        settings.model_path, **settings.stage_factory_kwargs(H3_STAGE)
    )
    hooks = scheduler.session_hooks
    identity = SessionIdentity("copy", 1)
    first = source_chunk(0)
    operation = SessionOperation(
        "append", identity, (H3_STAGE, H3_RGB_DELIVERY_STAGE), first
    )
    operation_fields = operation.to_dict()
    request = conditioning()
    request.metadata = {
        SESSION_METADATA_KEY: operation_fields,
        "alias": operation_fields,
    }
    accepted_rgb = first.payload["rgb"]
    payload = StagePayload("request", request, None)
    outputs: list[TimedChunk] = []
    hooks.open(identity, request)
    try:
        hooks.on_append_arrival(identity, "request", first)
        hooks.on_append_arrival(identity, "next", source_chunk(1))
        result = hooks.append(
            first,
            payload,
            SessionContext(
                session_identity=identity,
                cancelled=threading.Event(),
                emit=outputs.append,
            ),
        )
        assert not outputs
        assert result.request is not request
        assert result.request.metadata[SESSION_METADATA_KEY]["chunk"]["payload"] is None
        assert result.request.metadata["alias"] is not operation_fields
        assert operation_fields["chunk"]["payload"]["rgb"] == accepted_rgb
        assert first.payload["rgb"] == accepted_rgb
        assert (
            hooks.bridge.lookahead_sources[0].packet.rgb
            == source_chunk(1).payload["rgb"]
        )
        assert result.data[0]["frames"][0].source_frame_id == "source-0"
        assert result.data[0]["rgb"].device.type == "cpu"
    finally:
        hooks.close(identity)


def test_cpu_delivery_rejects_frame_budget_before_emitting() -> None:
    limits = H3RuntimeLimits(
        max_width=2,
        max_height=2,
        max_input_frames=1,
        max_output_frames=1,
        max_state_bytes=1024,
    )
    hooks = H3RGBDeliveryHooks(limits)
    rgb = torch.zeros((1, 2, 2, 3), dtype=torch.uint8)
    frames = H3RGBPacket.model_validate(source_chunk(0).payload).frames
    batch = ({"rgb": rgb, "frames": frames},) * 2
    outputs: list[TimedChunk] = []
    with pytest.raises(ValueError, match="H3"):
        hooks.append(
            source_chunk(0),
            StagePayload("invalid", conditioning(), batch),
            SessionContext(
                session_identity=SessionIdentity("invalid", 1),
                cancelled=threading.Event(),
                emit=outputs.append,
            ),
        )
    assert not outputs


@pytest.mark.parametrize('with_trace', [False, True])
def test_cpu_delivery_preserves_optional_orchestration_and_rgb(with_trace):
    limits = H3RuntimeLimits(max_width=2, max_height=2, max_input_frames=1,
        max_output_frames=1, max_state_bytes=1024)
    chunk = source_chunk(0)
    source = H3RGBPacket.model_validate(chunk.payload)
    rgb = torch.arange(12, dtype=torch.uint8).reshape(1, 2, 2, 3)
    packet = {'rgb': rgb, 'frames': source.frames}
    trace = {'clock': 'native_worker_monotonic', 'stages': [
        {'stage': 'codec', 'dispatch_ns': 1000, 'complete_ns': 2000, 'wall_ms': .001}]}
    if with_trace:
        packet['orchestration'] = trace
    outputs = []
    payload = StagePayload('trace', conditioning(), (packet,))
    H3RGBDeliveryHooks(limits).append(chunk, payload, SessionContext(
        session_identity=SessionIdentity('trace', 1), cancelled=threading.Event(), emit=outputs.append))
    assert payload.data is None
    assert len(outputs) == 1
    assert outputs[0].payload['rgb'] == bytes(range(12))
    assert outputs[0].payload['frames'] == [frame.model_dump() for frame in source.frames]
    if with_trace:
        assert outputs[0].payload['orchestration'] == trace
    else:
        assert 'orchestration' not in outputs[0].payload


@pytest.mark.skipif(sys.platform != "linux", reason="Real Stage fixture requires Linux")
@pytest.mark.asyncio
async def test_real_shm_route_starts_next_compute_before_prior_cpu_input_done(
    tmp_path: Path,
) -> None:
    async with cpu_delivery_pipeline(tmp_path, block_packing=True) as (
        coordinator,
        settings,
        probe,
    ):
        identity = await coordinator.open_session(
            conditioning(), limits=settings.session_wire_limits(), session_id="overlap"
        )
        outputs = coordinator.session_outputs(identity)
        first_output = asyncio.create_task(anext(outputs))
        await coordinator.append_session(identity, source_chunk(0))
        assert await asyncio.to_thread(probe.packing_entered.wait, TIMEOUT_SECONDS)
        await coordinator.append_session(identity, source_chunk(1))
        assert await asyncio.to_thread(probe.next_compute.wait, TIMEOUT_SECONDS)
        assert not first_output.done()
        node_identity = coordinator.graphs[identity.id].nodes[H3_STAGE]
        session = coordinator.get_session(node_identity)
        assert len(session.submitted_appends) == 2
        probe.packing_release.set()
        received = [await asyncio.wait_for(first_output, TIMEOUT_SECONDS)]
        received += [
            await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS) for _ in range(3)
        ]
        video = [output for output in received if output.kind == "data"]
        assert [output.input_seq for output in video] == [0, 1]
        assert [
            output.input_seq for output in received if output.kind == "input_done"
        ] == [0, 1]
        for sequence in range(2):
            assert [
                output.kind for output in received if output.input_seq == sequence
            ] == ["data", "input_done"]
        for sequence, output in enumerate(video):
            assert output.turn == sequence + 7
            assert (
                output.payload["frames"][0]["source_frame_id"] == f"source-{sequence}"
            )
            assert output.payload["rgb"] == source_chunk(sequence).payload["rgb"]
            assert output.payload["orchestration"]["stages"][0]["stage"] == "render"
        await coordinator.append_session(
            identity, TimedChunk("video", 80, 0, 2, None, H3_RGB24, True, turn=9)
        )
        ended = [
            await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS) for _ in range(2)
        ]
        assert [(output.kind, output.eos, output.input_seq) for output in ended] == [
            ("data", True, 2),
            ("input_done", True, 2),
        ]
        await coordinator.close_session(identity)
        await outputs.aclose()
        events: list[tuple[str, int]] = []
        while ("close-source", node_identity.open_index) not in events:
            events.append(
                await asyncio.to_thread(probe.events.get, True, TIMEOUT_SECONDS)
            )
        assert ("open-source", node_identity.open_index) in events
        assert ("open-delivery", node_identity.open_index) in events
        assert [sequence for name, sequence in events if name == "shm"] == [0, 1, 2]
        assert events.index(
            ("close-delivery", node_identity.open_index)
        ) < events.index(("close-source", node_identity.open_index))


@pytest.mark.skipif(sys.platform != "linux", reason="Real Stage fixture requires Linux")
@pytest.mark.asyncio
async def test_stalled_real_route_send_still_admits_next_compute(
    tmp_path: Path,
) -> None:
    async with cpu_delivery_pipeline(tmp_path, block_relay=True) as (
        coordinator,
        settings,
        probe,
    ):
        identity = await coordinator.open_session(
            conditioning(), limits=settings.session_wire_limits()
        )
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, source_chunk(0))
        assert await asyncio.to_thread(probe.relay_entered.wait, TIMEOUT_SECONDS)
        await coordinator.append_session(identity, source_chunk(1, eos=True))
        assert await asyncio.to_thread(probe.next_compute.wait, TIMEOUT_SECONDS)
        probe.relay_release.set()
        received = [
            await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS) for _ in range(5)
        ]
        assert [
            output.input_seq for output in received if output.kind == "input_done"
        ] == [0, 1]
        assert sum(output.eos for output in received if output.kind == "data") == 1
        await outputs.aclose()


@pytest.mark.skipif(sys.platform != "linux", reason="Real Stage fixture requires Linux")
@pytest.mark.asyncio
@pytest.mark.parametrize("before_handoff", [True, False])
async def test_cancel_and_reopen_fence_old_cpu_batch(
    tmp_path: Path, before_handoff: bool
) -> None:
    async with cpu_delivery_pipeline(
        tmp_path, block_compute=before_handoff, block_packing=not before_handoff
    ) as (coordinator, settings, probe):
        identity = await coordinator.open_session(
            conditioning(), limits=settings.session_wire_limits(), session_id="reopen"
        )
        await coordinator.append_session(identity, source_chunk(0, eos=True))
        entered = probe.compute_entered if before_handoff else probe.packing_entered
        assert await asyncio.to_thread(entered.wait, TIMEOUT_SECONDS)
        cancel = TimedChunk("control", 0, 0, 0, {"type": H3_CANCEL}, SESSION_CONTROL)
        await asyncio.wait_for(
            coordinator.control_session(identity, cancel), TIMEOUT_SECONDS
        )
        assert probe.cancelled.is_set()
        outputs = coordinator.session_outputs(identity)
        receipt = await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS)
        assert receipt.kind == "input_done"
        await coordinator.close_session(identity)
        await outputs.aclose()
        probe.compute_release.set()
        probe.packing_release.set()
        reopened = await coordinator.open_session(
            conditioning(), limits=settings.session_wire_limits(), session_id="reopen"
        )
        assert reopened.open_index > identity.open_index
        reopened_outputs = coordinator.session_outputs(reopened)
        await coordinator.append_session(reopened, source_chunk(0, eos=True))
        received = [
            await asyncio.wait_for(anext(reopened_outputs), TIMEOUT_SECONDS)
            for _ in range(3)
        ]
        assert [output.kind for output in received] == ["data", "data", "input_done"]
        assert all(output.session_identity == reopened for output in received)
        assert sum(output.eos for output in received if output.kind == "data") == 1
        await reopened_outputs.aclose()


@pytest.mark.skipif(sys.platform != "linux", reason="Real Stage fixture requires Linux")
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["packing", "overflow"])
async def test_cpu_failure_and_slow_output_fail_without_input_done(
    tmp_path: Path, failure: str
) -> None:
    async with cpu_delivery_pipeline(
        tmp_path,
        fail_delivery=failure == "packing",
        max_output_chunks=1 if failure == "overflow" else 16,
    ) as (coordinator, settings, probe):
        identity = await coordinator.open_session(
            conditioning(), limits=settings.session_wire_limits()
        )
        graph = coordinator.graphs[identity.id]
        outputs = coordinator.graph_outputs(graph)
        await coordinator.append_session(identity, source_chunk(0, eos=True))
        received = []

        async def wait_for_failure() -> None:
            while graph.error is None:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_failure(), TIMEOUT_SECONDS)

        async def drain_outputs() -> None:
            async for output in outputs:
                received.append(output)

        with pytest.raises(RuntimeError, match="closed|packing|output|queue"):
            await asyncio.wait_for(drain_outputs(), TIMEOUT_SECONDS)
        assert not any(output.kind == "input_done" for output in received)
        await outputs.aclose()
