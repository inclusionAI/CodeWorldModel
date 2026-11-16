# SPDX-License-Identifier: Apache-2.0
"""Raw RGB boundaries between Omni sessions and the native H3 media bridge."""

from __future__ import annotations

import threading
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sglang_omni.models.minimax_h3.payload_types import RGB24_CHANNELS as RGB24_CHANNELS
from sglang_omni.models.minimax_h3.payload_types import (
    H3CPUFrameBatch,
    H3CPUOutputBridge,
)
from sglang_omni.models.minimax_h3.payload_types import H3FrameStamp as H3FrameStamp
from sglang_omni.models.minimax_h3.payload_types import (
    H3NextSource,
    H3RGBPacket,
    H3SourceInput,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    ResourceUsage,
    SessionIdentity,
    TimedChunk,
    find_session_operation,
)
from sglang_omni.scheduling.session import SessionContext, SessionHooks

H3_RGB24 = "h3.rgb24"
H3_CANCEL = "h3.cancel"


class H3OpenRequest(BaseModel):
    """Static conditioning for a fresh causal timeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str = Field(max_length=16384, strict=True)
    first_image: H3RGBPacket
    seed: int = Field(ge=0, lt=2**63, strict=True)

    @model_validator(mode="after")
    def validate_first_image(self) -> H3OpenRequest:
        if len(self.first_image.frames) != 1:
            raise ValueError("first_image must contain exactly one frame")
        else:
            return self


@dataclass(frozen=True, kw_only=True)
class H3RuntimeLimits:
    max_width: int
    max_height: int
    max_input_frames: int
    max_output_frames: int
    max_state_bytes: int
    gpu_id: int | None = None
    runtime_gpu_ids: tuple[int, ...] = ()
    codec_overlap: bool = False
    cpu_delivery: bool = False

    def check_packet(self, packet: H3RGBPacket, *, output: bool = False) -> None:
        max_frames = self.max_output_frames if output else self.max_input_frames
        if (
            packet.width > self.max_width
            or packet.height > self.max_height
            or len(packet.frames) > max_frames
        ):
            raise ValueError(
                "H3 RGB packet exceeds deployment geometry or frame budget"
            )
        else:
            pass


class H3PacketEmitter(Protocol):
    def __call__(self, packet: H3RGBPacket) -> None: ...


class H3MediaBridge(Protocol):
    """Compute owns causal state, codecs, bounded allocations and GPU ordering."""

    def open(
        self, session_identity: SessionIdentity, request: H3OpenRequest
    ) -> None: ...

    def append(
        self,
        session_identity: SessionIdentity,
        packet: H3RGBPacket,
        cancelled: threading.Event,
        emit: H3PacketEmitter,
    ) -> None: ...

    def finish(
        self,
        session_identity: SessionIdentity,
        cancelled: threading.Event,
        emit: H3PacketEmitter,
    ) -> None: ...

    def cancel(self, session_identity: SessionIdentity) -> None: ...

    def close(self, session_identity: SessionIdentity) -> None: ...

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage: ...


@runtime_checkable
class H3OverlapMediaBridge(H3MediaBridge, Protocol):
    codec_overlap: bool

    def advance(
        self,
        session_identity: SessionIdentity,
        source: H3SourceInput,
        cancelled: threading.Event,
        emit: H3PacketEmitter,
        peek_next: H3NextSource | None = None,
    ) -> None: ...


@dataclass(kw_only=True)
class H3InputHandoff:
    inputs: deque[H3SourceInput] = field(default_factory=deque)


def rgb_packet_chunk(packet: H3RGBPacket, turn: int | None) -> TimedChunk:
    return TimedChunk(
        modality="video",
        t_start_ms=packet.frames[0].pts_ms,
        duration_ms=packet.duration_ms,
        seq=0,
        payload=packet.wire_payload(),
        format=H3_RGB24,
        turn=turn,
    )


def rgb_eos_chunk(chunk: TimedChunk) -> TimedChunk:
    return TimedChunk(
        modality="video",
        t_start_ms=chunk.t_start_ms + chunk.duration_ms,
        duration_ms=0,
        seq=0,
        payload=None,
        format=H3_RGB24,
        eos=True,
        turn=chunk.turn,
    )


class H3SessionHooks(SessionHooks):
    def __init__(self, bridge: H3MediaBridge, limits: H3RuntimeLimits) -> None:
        if limits.cpu_delivery and not isinstance(bridge, H3CPUOutputBridge):
            raise TypeError("H3 cpu_delivery requires a bridge with advance_cpu")
        else:
            pass
        if limits.codec_overlap and (
            not isinstance(bridge, H3OverlapMediaBridge) or not bridge.codec_overlap
        ):
            raise TypeError(
                "H3 codec_overlap requires a bridge with native advance support"
            )
        else:
            pass
        self.bridge = bridge
        self.limits = limits
        self.input_handoffs: dict[SessionIdentity, H3InputHandoff] = {}
        self.handoff_lock = threading.Lock()

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        conditioning = H3OpenRequest.model_validate(request.params)
        self.limits.check_packet(conditioning.first_image)
        try:
            self.bridge.open(session_identity, conditioning)
            if self.limits.codec_overlap:
                with self.handoff_lock:
                    self.input_handoffs[session_identity] = H3InputHandoff()
            else:
                pass
        except BaseException:
            self.bridge.close(session_identity)
            raise

    def source_input(self, chunk: TimedChunk) -> H3SourceInput:
        if chunk.modality != "video" or chunk.format != H3_RGB24:
            raise ValueError(f"H3 takes video chunks in {H3_RGB24} format")
        elif chunk.payload is None:
            if not chunk.eos or chunk.duration_ms != 0:
                raise ValueError("empty H3 input must be a zero-duration EOS")
            else:
                return H3SourceInput(input_seq=chunk.seq, packet=None, eos=True)
        else:
            packet = H3RGBPacket.model_validate(chunk.payload)
            self.limits.check_packet(packet)
            if (
                packet.frames[0].pts_ms != chunk.t_start_ms
                or packet.duration_ms != chunk.duration_ms
            ):
                raise ValueError("chunk timing must match the RGB frame timeline")
            else:
                return H3SourceInput(input_seq=chunk.seq, packet=packet, eos=chunk.eos)

    def on_append_arrival(
        self, session_identity: SessionIdentity, request_id: str, chunk: TimedChunk
    ) -> None:
        if self.limits.codec_overlap:
            source = self.source_input(chunk)
            with self.handoff_lock:
                handoff = self.input_handoffs.get(session_identity)
                if handoff is None:
                    raise ValueError("unknown or stale H3 session arrival")
                elif len(handoff.inputs) >= 2:
                    raise ValueError("H3 append lookahead is full")
                else:
                    handoff.inputs.append(source)
        else:
            pass

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        if self.limits.codec_overlap:
            with self.handoff_lock:
                handoff = self.input_handoffs.get(context.session_identity)
                if context.cancelled.is_set():
                    if handoff is not None:
                        handoff.inputs.clear()
                    else:
                        pass
                    payload.data = None
                    return payload
                elif (
                    handoff is None
                    or not handoff.inputs
                    or handoff.inputs[0].input_seq != chunk.seq
                ):
                    raise ValueError("H3 append has no matching arrived input")
                else:
                    source = handoff.inputs[0]
        else:
            source = self.source_input(chunk)

        emitted_frames = 0

        def emit_packet(packet: H3RGBPacket) -> None:
            nonlocal emitted_frames
            self.limits.check_packet(packet, output=True)
            emitted_frames += len(packet.frames)
            if emitted_frames > self.limits.max_output_frames:
                raise ValueError("H3 output exceeds the per-append frame budget")
            else:
                pass
            context.emit(rgb_packet_chunk(packet, chunk.turn))

        if self.limits.codec_overlap:

            def peek_next() -> H3SourceInput | None:
                with self.handoff_lock:
                    if context.cancelled.is_set():
                        handoff.inputs.clear()
                        return None
                    elif len(handoff.inputs) > 1:
                        return handoff.inputs[1]
                    else:
                        return None

            try:
                if self.limits.cpu_delivery:
                    batch: H3CPUFrameBatch = self.bridge.advance_cpu(
                        context.session_identity, source, context.cancelled, peek_next
                    )
                    operation = find_session_operation(payload.request.metadata)
                    metadata = deepcopy(payload.request.metadata)
                    metadata[SESSION_METADATA_KEY] = replace(
                        operation, chunk=replace(operation.chunk, payload=None)
                    ).to_dict()
                    payload.request = replace(payload.request, metadata=metadata)
                    payload.data = batch
                else:
                    self.bridge.advance(
                        context.session_identity,
                        source,
                        context.cancelled,
                        emit_packet,
                        peek_next,
                    )
            except BaseException:
                with self.handoff_lock:
                    handoff.inputs.clear()
                raise
            finally:
                with self.handoff_lock:
                    if context.cancelled.is_set():
                        handoff.inputs.clear()
                    elif handoff.inputs and handoff.inputs[0] is source:
                        handoff.inputs.popleft()
                    else:
                        pass
        else:
            if source.packet is not None:
                self.bridge.append(
                    context.session_identity,
                    source.packet,
                    context.cancelled,
                    emit_packet,
                )
            else:
                pass
            if source.eos:
                self.bridge.finish(
                    context.session_identity, context.cancelled, emit_packet
                )
            else:
                pass
        if self.limits.cpu_delivery:
            return payload
        elif source.eos:
            context.emit(rgb_eos_chunk(chunk))
        else:
            pass
        payload.data = None
        return payload

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        with self.handoff_lock:
            handoff = self.input_handoffs.get(session_identity)
            if handoff is not None:
                handoff.inputs.clear()
            else:
                pass
        self.bridge.cancel(session_identity)

    def close(self, session_identity: SessionIdentity) -> None:
        with self.handoff_lock:
            self.input_handoffs.pop(session_identity, None)
        self.bridge.close(session_identity)

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        return self.bridge.usage(session_identity)
