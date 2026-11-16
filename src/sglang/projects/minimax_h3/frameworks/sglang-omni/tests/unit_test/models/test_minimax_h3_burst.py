# SPDX-License-Identifier: Apache-2.0
"""Buffered source packets through real H3 hooks and Omni session routing."""

import asyncio

import pytest

from sglang_omni.models.minimax_h3.session import H3_RGB24, H3FrameStamp, H3RGBPacket
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import TimedChunk
from tests.unit_test.fixtures.h3_session import (
    TIMEOUT_SECONDS,
    cpu_config,
    cpu_pipeline,
)


@pytest.mark.asyncio
async def test_buffered_output_burst_preserves_sources_across_input_packets() -> None:
    config = cpu_config()
    assert config.max_output_frames == 34
    packets = [
        H3RGBPacket(
            width=2,
            height=2,
            frames=tuple(
                H3FrameStamp(
                    source_frame_id=f"source-{sequence}",
                    pts_ms=sequence * 40.0,
                    duration_ms=40.0,
                )
                for sequence in range(start, start + 17)
            ),
            rgb=bytes([start]) * (17 * 2 * 2 * 3),
        )
        for start in (0, 17)
    ]
    conditioning = OmniRequest(
        None,
        params={
            "prompt": "buffered bootstrap",
            "seed": 7,
            "first_image": packets[0]
            .model_copy(
                update={"frames": packets[0].frames[:1], "rgb": packets[0].rgb[:12]}
            )
            .wire_payload(),
        },
    )
    async with cpu_pipeline(config) as (coordinator, bridge):
        bridge.buffer_until_frames = 34
        identity = await coordinator.open_session(
            conditioning, limits=config.session_wire_limits()
        )
        outputs = coordinator.session_outputs(identity)
        for sequence, packet in enumerate(packets):
            await coordinator.append_session(
                identity,
                TimedChunk(
                    "video",
                    packet.frames[0].pts_ms,
                    packet.duration_ms,
                    sequence,
                    packet.wire_payload(),
                    H3_RGB24,
                    eos=sequence == 1,
                ),
            )
        received = [
            await asyncio.wait_for(anext(outputs), TIMEOUT_SECONDS) for _ in range(4)
        ]
        video = next(
            output
            for output in received
            if output.kind == "data" and output.payload is not None
        )
        decoded = H3RGBPacket.model_validate(video.payload)
        assert video.input_seq == 1
        assert decoded.frames == packets[0].frames + packets[1].frames
        assert decoded.rgb == packets[0].rgb + packets[1].rgb
        assert len(decoded.frames) == 34
        assert (video.t_start_ms, video.duration_ms) == (0, 1360)
        assert sum(output.eos for output in received if output.kind == "data") == 1
        await outputs.aclose()
        await coordinator.close_session(identity)
        assert not bridge.states
