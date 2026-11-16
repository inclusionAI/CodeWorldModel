# SPDX-License-Identifier: Apache-2.0
"""Causal H3 over upstream realtime sessions: one request opens or advances a stream.

``req.condition_inputs["h3"]`` carries the request. ``block_idx == 0`` holds an
``H3RealtimeOpen`` and emits nothing. The native adapter can either use the
compatibility ``H3RealtimeAdvance`` request or split a source into
``H3RealtimeCompute`` and ``H3RealtimeCodecRequest``. The split form leaves the
GPU compute completion boundary before codec/D2H work, while preserving the
same single-session state machine and collectives.
"""

from __future__ import annotations

import hashlib
import math
import os
import time

import msgspec
import numpy as np
import torch

from sglang.multimodal_gen.configs.sample.minimax_h3_causal import H3FixedHistoryConfig
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.causal_latent import (
    H3FixedHistorySession,
    H3GeneratedSpan,
    H3SourceLatents,
    H3StaticCondition,
)
from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState
from .shared_rgb import H3SharedRGB, copy_rgb_to_host

H3_CONDITION_KEY = "h3"


class H3RealtimeOpen(msgspec.Struct, frozen=True):
    # The prompt rides on Req.prompt like other realtime pipelines; Req.batch_size reads it.
    # HxWx3 uint8.
    first_image: np.ndarray
    seed: int


class H3RealtimeSource(msgspec.Struct, frozen=True):
    input_seq: int
    # Tx HxWx3 uint8: 17 frames, or the final 5; None only for an EOS after a held tail.
    frames: np.ndarray | None
    eos: bool


class H3RealtimeAdvance(msgspec.Struct, frozen=True):
    source: H3RealtimeSource
    # Lookahead the encoder pair prepares during this advance's decode.
    next_source: H3RealtimeSource | None = None


class H3RealtimeCompute(msgspec.Struct, frozen=True):
    """Finish this rank's model work; keep generated latents in the session."""

    source: H3RealtimeSource
    ack_ids: tuple[str, ...] = ()


class H3RealtimeCodecRequest(msgspec.Struct, frozen=True):
    """Finish local codec/D2H work; return CPU RGB on the output rank."""

    input_seq: int
    next_source: H3RealtimeSource | None = None
    ack_ids: tuple[str, ...] = ()
    shared_cpu_output: bool = False
    trace: bool = False


class H3RealtimeAck(msgspec.Struct, frozen=True):
    """Release RGB relay blocks after the bridge has consumed them."""

    ack_ids: tuple[str, ...] = ()


class _PreparedProxy(msgspec.Struct, frozen=True):
    source: H3RealtimeSource
    proxy: torch.Tensor


class _ComputedSource(msgspec.Struct, frozen=True):
    source: H3RealtimeSource
    spans: tuple[H3GeneratedSpan, ...]


class H3RealtimeState(BaseRealtimeState):
    """Per-rank session; phase moves unopened -> open -> ended, or to failed."""

    def __init__(self) -> None:
        self.phase = "unopened"
        self.components = None
        self.codec = None
        self.latent: H3FixedHistorySession | None = None
        self.seed = 0
        self.target_shape: tuple[int, ...] = ()
        # Counts frames only; Omni keeps the per-frame stamps.
        self.pending_frames = 0
        self.tail: np.ndarray | None = None
        self.proxy_geometry: tuple[int, int] | None = None
        self.input_seq = -1
        self.source_clip = 0
        self.prepared: _PreparedProxy | None = None
        self.computed: _ComputedSource | None = None
        self.stages: list[dict] = []

    def dispose(self) -> None:
        if self.latent is not None:
            self.components.synchronize()
            self.latent.reset()
        if self.codec is not None:
            self.codec.begin_session()
        self.latent = self.prepared = self.computed = self.tail = None
        self.phase = "released"


def combine_seed(*parts) -> int:
    # Same derivation as raven.common.seed.combine_seed; sessions must reproduce its noise.
    digest = hashlib.sha256("_".join(str(part) for part in parts).encode()).hexdigest()
    return int(digest, 16) % (2**63)


def _generator(seed: int, device: torch.device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _check_rgb(value, *, ndim: int, name: str) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.ndim != ndim
        or value.shape[-1] != 3
    ):
        raise ValueError(f"{name} must be a uint8 ndarray with {ndim} dims ending in RGB")


def _check_source(source, name: str) -> None:
    if (
        not isinstance(source, H3RealtimeSource)
        or type(source.input_seq) is not int
        or type(source.eos) is not bool
    ):
        raise ValueError(f"{name} must be an H3RealtimeSource with int input_seq and bool eos")
    if source.frames is not None:
        _check_rgb(source.frames, ndim=4, name=f"{name}.frames")


def _check_ack_ids(values: tuple[str, ...]) -> None:
    if any(type(value) is not str or not value for value in values):
        raise ValueError("H3 acknowledgement ids must be nonempty strings")


def _parse_request(req):
    if req.session is None:
        raise ValueError("H3 realtime requests require a realtime_session_id")
    if not isinstance(req.prompt, str):
        raise ValueError("H3 realtime requests require a str Req.prompt")
    request = req.condition_inputs.get(H3_CONDITION_KEY)
    if req.block_idx == 0:
        if not isinstance(request, H3RealtimeOpen) or type(request.seed) is not int:
            raise ValueError("H3 block 0 requires an H3RealtimeOpen with int seed")
        _check_rgb(request.first_image, ndim=3, name="first_image")
        return request
    if isinstance(request, H3RealtimeAdvance):
        _check_source(request.source, "source")
        if request.next_source is not None:
            _check_source(request.next_source, "next_source")
    elif isinstance(request, H3RealtimeCompute):
        _check_source(request.source, "source")
        _check_ack_ids(request.ack_ids)
    elif isinstance(request, H3RealtimeCodecRequest):
        if type(request.input_seq) is not int:
            raise ValueError("codec request input_seq must be an int")
        if type(request.shared_cpu_output) is not bool:
            raise ValueError("shared_cpu_output must be a bool")
        if type(request.trace) is not bool:
            raise ValueError("trace must be a bool")
        if request.next_source is not None:
            _check_source(request.next_source, "next_source")
        _check_ack_ids(request.ack_ids)
    elif isinstance(request, H3RealtimeAck):
        _check_ack_ids(request.ack_ids)
    else:
        raise ValueError(
            "H3 blocks after 0 require H3RealtimeAdvance, H3RealtimeCompute, "
            "H3RealtimeCodecRequest, or H3RealtimeAck"
        )
    return request


def _same_source(left: H3RealtimeSource, right: H3RealtimeSource) -> bool:
    if (
        left.input_seq != right.input_seq
        or left.eos != right.eos
        or (left.frames is None) != (right.frames is None)
    ):
        return False
    return left.frames is None or (
        left.frames.shape == right.frames.shape
        and np.array_equal(left.frames, right.frames)
    )


class H3RealtimeStage(PipelineStage):
    def __init__(self, *, components, codec, recipe: dict, codec_overlap: bool,
                 fuse_first_denoise: bool, cwm_silence_path=None, cwm_proxy_encode_size=(768, 1344),
                 cwm_proxy_resize="lanczos"):
        super().__init__()
        self.components, self.codec = components, codec
        self.recipe = recipe
        self.codec_overlap = codec_overlap
        self.fuse_first_denoise = fuse_first_denoise
        self.cwm_silence_path = cwm_silence_path
        if cwm_silence_path is not None:
            components.proxy_encode_size = None if cwm_proxy_encode_size == (256, 448) else cwm_proxy_encode_size
            components.proxy_resize = cwm_proxy_resize
        self._rgb_shm_enabled = os.environ.get("H3_RGB_SHM_RELAY") == "1"
        self._rgb_shm_writer = None
        config = H3FixedHistoryConfig.from_recipe(recipe)
        # Bootstrap can hold back the largest window plus one packet and the tail.
        self.pending_limit = 17 * (math.ceil(max(
            config.bootstrap_latents, *config.current_latent_schedule) / 5) + 1) + 5

    @torch.inference_mode()
    def forward(self, batch, server_args):
        request = _parse_request(batch)
        if isinstance(request, H3RealtimeAck):
            self._ack_rgb(request.ack_ids)
            return batch
        state = batch.session.get_or_create_state(H3RealtimeState)
        if batch.block_idx == 0:
            self._open(state, batch.prompt, request)
            return batch
        if state.phase != "open":
            raise RuntimeError(f"H3 session is {state.phase}; release and reopen")
        if isinstance(request, H3RealtimeAdvance):
            self._compute(state, request.source)
            rgb = self._deliver(state, request.next_source)
            if self.codec.is_output:
                batch.output = rgb
        elif isinstance(request, H3RealtimeCompute):
            self._ack_rgb(request.ack_ids)
            self._compute(state, request.source)
        else:
            self._ack_rgb(request.ack_ids)
            started = time.perf_counter_ns()
            rgb = self._deliver(state, request.next_source, request.input_seq,
                                shared_cpu_output=request.shared_cpu_output)
            if self.codec.is_output:
                if request.trace:
                    self._stage_time(state, 'codec', started)
                    batch.output = dict(_type='H3RealtimeOutput', rgb=rgb,
                        orchestration=dict(clock='native_worker_monotonic', stages=state.stages))
                else:
                    batch.output = rgb
        return batch

    def _ack_rgb(self, object_ids: tuple[str, ...]) -> None:
        if object_ids and self._rgb_shm_writer is not None:
            self._rgb_shm_writer.ack(object_ids)

    def _ready(self) -> None:
        # Local device completion; the next distributed request orders ranks.
        self.components.synchronize()

    def _open(self, state: H3RealtimeState, prompt: str, request: H3RealtimeOpen) -> None:
        state.phase = "failed"
        state.components, state.codec = self.components, self.codec
        self.codec.begin_session()
        try:
            features, tags = self.components.condition(prompt, request.first_image)
            picture = self.components.encode_image(request.first_image)
            generator = _generator(
                combine_seed(request.seed, "first_frame", "static"), torch.device("cpu")
            )
            static = H3StaticCondition(features, tags, picture,
                torch.randn(picture.shape, generator=generator).to(picture.device))
            # CWM runs the fixed-history H3 stream; its recipe only changes the
            # frame-aligned RoPE coordinates in causal_latent._pack().
            latent = H3FixedHistorySession(self.components.model, static,
                recipe=self.recipe, fuse_first_denoise=self.fuse_first_denoise)
            self._ready()
        except BaseException:
            self._ready()
            self.codec.begin_session()
            raise
        state.latent, state.seed = latent, request.seed
        state.target_shape = tuple(picture.shape)
        state.phase = "open"

    def _validate_source(self, state: H3RealtimeState, source: H3RealtimeSource) -> None:
        # Omni's frame stamps no longer reach the GPU, so input_seq orders sources here.
        if source.input_seq != state.input_seq + 1:
            raise ValueError("H3 source input_seq must follow the previous source")
        if source.frames is None:
            if not source.eos or state.tail is None:
                raise ValueError(
                    "H3 EOS requires the final 5 source frames; no implicit padding"
                )
            return
        count = len(source.frames)
        if count not in (5, 17) or state.tail is not None:
            raise ValueError("H3 requires 17-frame packets followed by one final 5-frame tail")
        if state.proxy_geometry not in (None, source.frames.shape[1:3]):
            raise ValueError("proxy geometry must stay fixed for a session")
        if source.eos and count != 5:
            raise ValueError("H3 EOS requires exactly the final 5 source frames")

    def _accept(self, state: H3RealtimeState, frames: np.ndarray) -> None:
        if state.pending_frames + len(frames) > self.pending_limit:
            raise ValueError("pending source ledger exceeds the native window bound")
        state.proxy_geometry = frames.shape[1:3]
        state.pending_frames += len(frames)

    def _compute(self, state: H3RealtimeState, source: H3RealtimeSource):
        state.stages = []
        if state.computed is not None:
            raise RuntimeError("H3 source is awaiting codec delivery")
        if state.prepared is None:
            self._validate_source(state, source)
        elif not _same_source(state.prepared.source, source):
            state.phase = "failed"
            raise ValueError("prepared H3 proxy does not match the current source identity")
        if source.frames is not None:
            self._accept(state, source.frames)
            frames = source.frames
        else:
            frames, state.tail = state.tail, None
        try:
            if len(frames) == 5 and not source.eos:
                state.tail = frames
                spans = ()
            else:
                spans = self._compute_accepted(state, source, frames)
            state.input_seq = source.input_seq
            state.computed = _ComputedSource(source, tuple(spans))
            return spans
        except BaseException:
            state.phase = "failed"
            raise

    def _compute_accepted(self, state, source, frames):
        started = time.perf_counter_ns()
        if state.prepared is None:
            proxy = self.components.encode_proxy(frames, eos=source.eos)
        else:
            proxy, state.prepared = state.prepared.proxy, None
        reference_rng = _generator(
            combine_seed(state.seed, "proxy", state.source_clip), torch.device("cpu")
        )
        reference_noise = torch.randn(proxy.shape, generator=reference_rng).to(proxy.device)
        device = torch.device(self.components.device)
        # raven RandomState(seed).fork("cwm_rgb", 0, clip) on the worker device.
        rng = _generator(combine_seed(state.seed, "cwm_rgb", 0, state.source_clip), device)
        shape = (state.target_shape[0], 2 if source.eos else 5, *state.target_shape[2:])
        latents = H3SourceLatents((proxy,), (reference_noise,),
            torch.randn(shape, device=device, generator=rng),
            torch.randn(shape, device=device, generator=rng))
        self._ready()
        self._stage_time(state, 'proxy_prepare', started)
        started = time.perf_counter_ns()
        spans = state.latent.finish(latents) if source.eos else state.latent.append(latents)
        self._ready()
        self._stage_time(state, 'render', started)
        state.source_clip += 1
        return spans

    def _stage_time(self, state, stage, started):
        # Observe existing GPU-completion boundaries without introducing synchronization.
        if self.codec.is_output:
            ended = time.perf_counter_ns()
            state.stages.append(dict(stage=stage, dispatch_ns=started, complete_ns=ended,
                                     wall_ms=(ended - started) / 1e6))

    def _deliver(self, state, next_source, input_seq=None, *, shared_cpu_output=False):
        computed = state.computed
        if computed is None:
            raise RuntimeError("H3 codec delivery requires a computed source")
        source, spans = computed.source, computed.spans
        if input_seq is not None and input_seq != source.input_seq:
            raise ValueError("H3 codec request does not match the computed source")
        try:
            if next_source is not None:
                if not self.codec_overlap or source.eos or state.prepared is not None:
                    raise ValueError("H3 lookahead is unavailable at this delivery boundary")
                self._validate_source(state, next_source)
            if self.codec_overlap:
                outputs = self._deliver_overlapped(state, spans, next_source,
                                                   shared_cpu_output=shared_cpu_output)
            else:
                outputs = self._deliver_sequential(state, spans,
                                                   shared_cpu_output=shared_cpu_output)
            if source.eos:
                if state.pending_frames:
                    raise RuntimeError("decoder EOS did not cover every accepted source frame")
                state.phase = "ended"
            state.computed = None
            if not self.codec.is_output:
                return None
            if shared_cpu_output:
                return H3SharedRGB.from_tensors(outputs)
            if self._rgb_shm_enabled and outputs:
                if self._rgb_shm_writer is None:
                    from inference.h3_shm_transport import H3ShmOutputWriter

                    self._rgb_shm_writer = H3ShmOutputWriter()
                rgb = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
                return self._rgb_shm_writer.put(
                    rgb,
                    request_id=f"h3-rgb-{id(state)}",
                    chunk_id=state.source_clip,
                )
            return self._rgb_output(state, outputs)
        except BaseException:
            state.phase = "failed"
            raise

    def _deliver_sequential(self, state, spans, *, shared_cpu_output=False) -> list[torch.Tensor]:
        outputs = []
        for span in spans:
            decoded = self.codec.decode(span.video, eos=span.plan.eos)
            cpu = copy_rgb_to_host(decoded.rgb, shared=shared_cpu_output)
            count = decoded.frame_count
            del decoded
            self._ready()
            self._retire(state, count, cpu, outputs)
        return outputs

    def _deliver_overlapped(self, state, spans, next_source, *, shared_cpu_output=False) -> list[torch.Tensor]:
        prepare = (
            next_source is not None
            and next_source.frames is not None
            and (len(next_source.frames) == 17 or next_source.eos)
        )
        local_proxy = None
        if prepare:
            local_proxy = self.components.encode_proxy_local(
                next_source.frames, eos=next_source.eos
            )
        local_counts, cpus = [], []
        for span in spans:
            decoded = self.codec.decode_local(span.video, eos=span.plan.eos)
            local_counts.append(decoded.frame_count)
            cpus.append(copy_rgb_to_host(decoded.rgb, shared=shared_cpu_output))
            del decoded
        self._ready()
        counts = self.codec.publish_counts(local_counts, span_count=len(spans))
        if prepare:
            proxy = self.components.broadcast_proxy(
                local_proxy, rgb_shape=next_source.frames.shape, eos=next_source.eos
            )
            state.prepared = _PreparedProxy(next_source, proxy)
        # The synchronous collective is ordered on the current CUDA stream; the
        # next scheduler request preserves that stream order, so a second full
        # device synchronize would only wait for work already sequenced here.
        outputs = []
        for count, cpu in zip(counts, cpus):
            self._retire(state, count, cpu, outputs)
        return outputs

    def _retire(self, state, count: int, cpu, outputs: list) -> None:
        # Counts are published to every rank first, so all ranks fail here together.
        if not 0 <= count <= state.pending_frames:
            raise RuntimeError("decoder produced more frames than the accepted source")
        state.pending_frames -= count
        if cpu is not None and count:
            outputs.append(cpu)

    def _rgb_output(self, state, outputs: list[torch.Tensor]) -> np.ndarray:
        if not outputs:
            # The causal decoder upsamples each latent pixel to 16x16 RGB.
            height, width = state.target_shape[2] * 16, state.target_shape[3] * 16
            return np.empty((0, height, width, 3), dtype=np.uint8)
        rgb = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
        return rgb.numpy()
