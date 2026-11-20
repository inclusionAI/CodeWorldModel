# SPDX-License-Identifier: Apache-2.0
"""Terminal CPU RGB delivery on the original H3 append lifetime."""

from sglang_omni.models.minimax_h3.payload_types import (
    H3CPUFrameBatch,
    H3CPUFramePacket,
    H3CPUOutputBridge,
    H3RGBPacket,
)
from sglang_omni.models.minimax_h3.session import (
    H3RuntimeLimits,
    rgb_eos_chunk,
    rgb_packet_chunk,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import (
    SessionContext,
    SessionHooks,
    SessionScheduler,
)

__all__ = [
    "H3CPUFrameBatch",
    "H3CPUFramePacket",
    "H3CPUOutputBridge",
    "H3RGBDeliveryHooks",
    "create_h3_rgb_delivery_executor",
]


class H3RGBDeliveryHooks(SessionHooks):
    def __init__(self, limits: H3RuntimeLimits) -> None:
        self.limits = limits

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        pass

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        batch: H3CPUFrameBatch = payload.data
        total_frames = sum(len(packet["frames"]) for packet in batch)
        if total_frames > self.limits.max_output_frames:
            raise ValueError("H3 output exceeds the per-append frame budget")
        else:
            pass
        for packet in batch:
            if context.cancelled.is_set():
                break
            else:
                rgb = packet["rgb"]
                packed = H3RGBPacket(
                    width=rgb.shape[2],
                    height=rgb.shape[1],
                    frames=packet["frames"],
                    rgb=rgb.numpy().tobytes(),
                    orchestration=packet.get("orchestration"),
                )
                context.emit(rgb_packet_chunk(packed, chunk.turn))
        if chunk.eos and not context.cancelled.is_set():
            context.emit(rgb_eos_chunk(chunk))
        else:
            pass
        payload.data = None
        return payload

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        pass

    def close(self, session_identity: SessionIdentity) -> None:
        pass


def create_h3_rgb_delivery_executor(
    *, max_width: int, max_height: int, max_output_frames: int, max_state_bytes: int
) -> SessionScheduler:
    limits = H3RuntimeLimits(
        max_width=max_width,
        max_height=max_height,
        max_input_frames=1,
        max_output_frames=max_output_frames,
        max_state_bytes=max_state_bytes,
    )
    return SessionScheduler(
        H3RGBDeliveryHooks(limits),
        max_open_sessions=1,
        max_concurrency=1,
        max_state_bytes_per_session=max_state_bytes,
    )
