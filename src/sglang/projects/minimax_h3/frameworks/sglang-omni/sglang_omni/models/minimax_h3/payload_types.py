# SPDX-License-Identifier: Apache-2.0
"""RGB packets and owned CPU output shared by the native H3 boundaries."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import NotRequired, Protocol, TypedDict, runtime_checkable

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sglang_omni.proto.session import SessionIdentity

RGB24_CHANNELS = 3


class H3FrameStamp(BaseModel):
    """Source identity and presentation time of one frame."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_frame_id: str = Field(min_length=1, max_length=512)
    pts_ms: float = Field(ge=0, allow_inf_nan=False)
    duration_ms: float = Field(gt=0, allow_inf_nan=False)


class H3RGBPacket(BaseModel):
    """Packed contiguous uint8 RGB frames in frame, height, width, channel order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    frames: tuple[H3FrameStamp, ...] = Field(min_length=1)
    rgb: bytes = Field(strict=True)
    orchestration: dict[str, object] | None = None

    @model_validator(mode="after")
    def validate_frames(self) -> H3RGBPacket:
        if (
            len(self.rgb)
            != self.width * self.height * len(self.frames) * RGB24_CHANNELS
        ):
            raise ValueError("RGB byte count does not match packet geometry")
        elif any(
            following.pts_ms < previous.pts_ms + previous.duration_ms
            for previous, following in zip(self.frames, self.frames[1:])
        ):
            raise ValueError("frame PTS overlaps or moves backwards")
        else:
            return self

    @property
    def duration_ms(self) -> float:
        return (
            self.frames[-1].pts_ms + self.frames[-1].duration_ms - self.frames[0].pts_ms
        )

    def wire_payload(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "frames": [frame.model_dump() for frame in self.frames],
            "rgb": self.rgb,
            **({"orchestration": self.orchestration} if self.orchestration is not None else {}),
        }


@dataclass(frozen=True, kw_only=True)
class H3SourceInput:
    input_seq: int
    packet: H3RGBPacket | None
    eos: bool


class H3NextSource(Protocol):
    def __call__(self) -> H3SourceInput | None: ...


class H3CPUFramePacket(TypedDict):
    """Owned contiguous CPU uint8 THWC storage, immutable after handoff."""

    rgb: torch.Tensor
    frames: tuple[H3FrameStamp, ...]
    orchestration: NotRequired[dict[str, object] | None]


H3CPUFrameBatch = tuple[H3CPUFramePacket, ...]


@runtime_checkable
class H3CPUOutputBridge(Protocol):
    """Return readable CPU RGB after producing GPU work and D2H finish; delivery owns it until consumed or cancelled."""

    def advance_cpu(
        self,
        session_identity: SessionIdentity,
        source: H3SourceInput,
        cancelled: threading.Event,
        peek_next: H3NextSource | None = None,
    ) -> H3CPUFrameBatch: ...
