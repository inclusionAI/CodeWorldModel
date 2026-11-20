# SPDX-License-Identifier: Apache-2.0
"""Construct the H3 stage with an explicitly installed native media provider."""

from sglang_omni.models.minimax_h3.session import (
    H3MediaBridge,
    H3RuntimeLimits,
    H3SessionHooks,
)
from sglang_omni.scheduling.session import SessionScheduler
from sglang_omni.utils.imports import import_string


def create_h3_session_executor(
    model_path: str,
    *,
    runtime_factory: str,
    max_width: int,
    max_height: int,
    max_input_frames: int,
    max_output_frames: int,
    max_state_bytes: int,
    gpu_id: int | None = None,
    runtime_gpu_ids: list[int] | None = None,
    codec_overlap: bool = False,
    cpu_delivery: bool = False,
) -> SessionScheduler:
    limits = H3RuntimeLimits(
        max_width=max_width,
        max_height=max_height,
        max_input_frames=max_input_frames,
        max_output_frames=max_output_frames,
        max_state_bytes=max_state_bytes,
        gpu_id=gpu_id,
        runtime_gpu_ids=tuple(runtime_gpu_ids or ()),
        codec_overlap=codec_overlap,
        cpu_delivery=cpu_delivery,
    )
    bridge: H3MediaBridge = import_string(runtime_factory)(
        model_path=model_path, limits=limits
    )
    return SessionScheduler(
        H3SessionHooks(bridge, limits),
        max_open_sessions=1,
        max_concurrency=1,
        max_state_bytes_per_session=max_state_bytes,
    )
