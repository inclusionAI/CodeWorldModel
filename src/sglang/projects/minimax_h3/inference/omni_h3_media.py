import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from sglang_omni.models.minimax_h3.delivery import H3CPUFrameBatch
from sglang_omni.models.minimax_h3.session import H3RGBPacket, H3SourceInput
from sglang_omni.proto.session import ResourceUsage


def _native_types():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.causal_latent import (
        H3FixedHistorySession, H3SourceLatents, H3StaticCondition,
    )
    return H3FixedHistorySession, H3SourceLatents, H3StaticCondition


def _pixels(packet):
    return np.frombuffer(packet.rgb, dtype=np.uint8).reshape(
        len(packet.frames), packet.height, packet.width, 3)


@dataclass
class _ComputedSource:
    source: H3SourceInput
    spans: tuple


@dataclass
class _PreparedProxy:
    source: H3SourceInput
    proxy: torch.Tensor


@dataclass
class _MediaState:
    identity: object
    seed: int
    latent: object
    target_shape: tuple
    frames: deque = field(default_factory=deque)
    tail: H3RGBPacket | None = None
    proxy_geometry: tuple | None = None
    next_pts_ms: float = 0.0
    source_clip: int = 0
    input_seq: int = -1
    computed: _ComputedSource | None = None
    prepared: _PreparedProxy | None = None
    phase: str = "open"


class NativeH3MediaBridge:
    """One session owns source stamps, decoder state and synchronous GPU completion.

    Components are the native SGLang loader result. The host must execute each
    call on all compute ranks in the same order; only the output rank publishes.
    """

    def __init__(self, components, codec, *, recipe, limits, state_reservation_bytes,
                 fuse_first_denoise=False):
        self.components, self.codec = components, codec
        self.recipe, self.limits = recipe, limits
        self.fuse_first_denoise = fuse_first_denoise
        self.codec_overlap = getattr(limits, 'codec_overlap', False)
        self.state_reservation_bytes = state_reservation_bytes
        self.pending_limit = 17 * (math.ceil(max(
            recipe['bootstrap_latents'], *recipe['current_latent_schedule']) / 5) + 1) + 5
        self.state = None

    def _current(self, identity):
        if self.state is None or self.state.identity != identity:
            raise ValueError("unknown or stale H3 session identity")
        if self.state.phase != "open":
            raise RuntimeError(f"H3 session is {self.state.phase}; close and reopen")
        return self.state

    def _ready(self):
        self.components.synchronize()

    @torch.inference_mode()
    def open(self, session_identity, request):
        from frameworks.raven.common.seed import combine_seed
        if self.state is not None:
            raise RuntimeError("native H3 bridge already owns a session")
        session_type, _, static_type = _native_types()
        image = _pixels(request.first_image)[0]
        self.codec.begin_session()
        try:
            features, tags = self.components.condition(request.prompt, image)
            picture = self.components.encode_image(image)
            generator = torch.Generator(device='cpu').manual_seed(
                combine_seed(request.seed, 'first_frame', 'static'))
            static = static_type(features, tags, picture,
                torch.randn(picture.shape, generator=generator).to(picture.device))
            latent = session_type(self.components.model, static, recipe=self.recipe,
                fuse_first_denoise=self.fuse_first_denoise)
            self._ready()
        except BaseException:
            self._ready()
            self.codec.begin_session()
            raise
        self.state = _MediaState(session_identity, request.seed, latent, tuple(picture.shape))

    def _validate_packet(self, state, packet):
        # Omni's session hooks already parsed the packet and applied the deployment limits.
        count = len(packet.frames)
        if count not in (5, 17) or state.tail is not None:
            raise ValueError("H3 requires 17-frame packets followed by one final 5-frame tail")
        geometry = (packet.width, packet.height)
        if state.proxy_geometry not in (None, geometry):
            raise ValueError("proxy geometry must stay fixed for a session")
        if packet.frames[0].pts_ms < state.next_pts_ms:
            raise ValueError("source time overlaps the preceding packet")

    def _accept(self, state, packet):
        if len(state.frames) + len(packet.frames) > self.pending_limit:
            raise ValueError("pending source ledger exceeds the native window bound")
        geometry = (packet.width, packet.height)
        state.proxy_geometry = geometry
        state.next_pts_ms = packet.frames[-1].pts_ms + packet.frames[-1].duration_ms
        state.frames.extend(packet.frames)

    @torch.inference_mode()
    def append(self, session_identity, packet, cancelled, emit):
        state = self._current(session_identity)
        self.compute(session_identity, H3SourceInput(
            input_seq=state.input_seq + 1, packet=packet, eos=False), cancelled)
        self.deliver(session_identity, None, cancelled, emit)

    @torch.inference_mode()
    def finish(self, session_identity, cancelled, emit):
        state = self._current(session_identity)
        self.compute(session_identity, H3SourceInput(
            input_seq=state.input_seq + 1, packet=None, eos=True), cancelled)
        self.deliver(session_identity, None, cancelled, emit)

    def _check_cancel(self, state, cancelled):
        if cancelled.is_set():
            state.phase = "cancelled"
            raise RuntimeError("H3 session was cancelled; close and reopen")

    def _validate_source(self, state, source):
        if source.packet is None:
            if not source.eos or state.tail is None:
                raise ValueError("H3 EOS requires the final 5 source frames; no implicit padding")
        else:
            self._validate_packet(state, source.packet)
            if source.eos and len(source.packet.frames) != 5:
                raise ValueError("H3 EOS requires exactly the final 5 source frames")

    @torch.inference_mode()
    def compute(self, session_identity, source, cancelled):
        state = self._current(session_identity)
        self._check_cancel(state, cancelled)
        if state.computed is not None:
            raise RuntimeError("H3 source is awaiting delivery")
        if state.prepared is None:
            self._validate_source(state, source)
        elif state.prepared.source != source:
            state.phase = 'failed'
            raise ValueError("prepared H3 proxy does not match the current source identity")
        if source.packet is not None:
            self._accept(state, source.packet)
            packet = source.packet
        else:
            packet, state.tail = state.tail, None
        try:
            if len(packet.frames) == 5 and not source.eos:
                state.tail = packet
                spans = ()
            else:
                spans = self._compute_accepted(state, source, packet, cancelled)
            state.input_seq = source.input_seq
            state.computed = _ComputedSource(source, tuple(spans))
        except BaseException:
            state.phase = "cancelled" if cancelled.is_set() else "failed"
            raise

    def _compute_accepted(self, state, source, packet, cancelled):
        from frameworks.raven.common.seed import RandomState, combine_seed
        _, source_type, _ = _native_types()
        if state.prepared is None:
            proxy = self.components.encode_proxy(_pixels(packet), eos=source.eos)
        else:
            proxy, state.prepared = state.prepared.proxy, None
        ref_rng = torch.Generator(device='cpu').manual_seed(
            combine_seed(state.seed, 'proxy', state.source_clip))
        reference_noise = torch.randn(proxy.shape, generator=ref_rng).to(proxy.device)
        rng = RandomState(state.seed).fork('cwm_rgb', 0, state.source_clip)
        device = torch.device(self.components.device)
        generator = rng.torch_cuda_generator if device.type == 'cuda' else rng.torch_generator
        shape = (state.target_shape[0], 2 if source.eos else 5, *state.target_shape[2:])
        latents = source_type((proxy,), (reference_noise,),
            torch.randn(shape, device=device, generator=generator),
            torch.randn(shape, device=device, generator=generator))
        self._ready()
        self._check_cancel(state, cancelled)
        spans = state.latent.finish(latents) if source.eos else state.latent.append(latents)
        self._ready()
        self._check_cancel(state, cancelled)
        state.source_clip += 1
        return spans

    @torch.inference_mode()
    def deliver(self, session_identity, next_source, cancelled, emit):
        self._deliver(session_identity, next_source, cancelled, emit, cpu_output=False)

    @torch.inference_mode()
    def deliver_cpu(self, session_identity, next_source, cancelled) -> H3CPUFrameBatch:
        packets = []
        self._deliver(session_identity, next_source, cancelled, packets.append, cpu_output=True)
        return tuple(packets)

    def _deliver(self, session_identity, next_source, cancelled, emit, *, cpu_output):
        state = self._current(session_identity)
        computed = state.computed
        if computed is None:
            raise RuntimeError("H3 delivery requires a computed source")
        try:
            if next_source is not None:
                if not self.codec_overlap or computed.source.eos or state.prepared is not None:
                    raise ValueError("H3 lookahead is unavailable at this delivery boundary")
                self._validate_source(state, next_source)
            if self.codec_overlap:
                self._deliver_overlapped(state, computed.spans, next_source, cancelled, emit,
                    cpu_output=cpu_output)
            else:
                output_frames = 0
                for span in computed.spans:
                    decoded = self.codec.decode(span.video, eos=span.plan.eos)
                    output_frames += decoded.frame_count
                    if cpu_output and output_frames > self.limits.max_output_frames:
                        raise ValueError("H3 RGB batch exceeds deployment frame budget")
                    cpu = self._copy_rgb(decoded, cpu_output=cpu_output)
                    count = decoded.frame_count
                    del decoded
                    self._ready()
                    self._check_cancel(state, cancelled)
                    self._retire(state, count, cpu, emit, cpu_output=cpu_output)
            if computed.source.eos:
                if state.frames:
                    raise RuntimeError("decoder EOS did not cover every accepted source frame")
                state.phase = 'ended'
            state.computed = None
        except BaseException:
            state.phase = "cancelled" if cancelled.is_set() else "failed"
            raise

    def _copy_rgb(self, decoded, *, cpu_output):
        rgb = decoded.rgb
        if rgb is None:
            return None
        if not cpu_output:
            return rgb.cpu().contiguous()
        if rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise RuntimeError("causal decoder must produce uint8 THWC RGB")
        if (not 0 < rgb.shape[1] <= self.limits.max_height
                or not 0 < rgb.shape[2] <= self.limits.max_width):
            raise ValueError("H3 RGB batch exceeds deployment geometry budget")
        cpu = torch.empty(rgb.shape, dtype=torch.uint8, device='cpu').share_memory_()
        cpu.copy_(rgb, non_blocking=False)
        return cpu

    def _deliver_overlapped(self, state, spans, next_source, cancelled, emit, *, cpu_output):
        prepare = (next_source is not None and next_source.packet is not None
                   and (len(next_source.packet.frames) == 17 or next_source.eos))
        local_proxy = None
        if prepare:
            local_proxy = self.components.encode_proxy_local(
                _pixels(next_source.packet), eos=next_source.eos)
        local_counts, cpus = [], []
        for span in spans:
            decoded = self.codec.decode_local(span.video, eos=span.plan.eos)
            local_counts.append(decoded.frame_count)
            cpus.append(self._copy_rgb(decoded, cpu_output=cpu_output))
            del decoded
        self._ready()
        counts = self.codec.publish_counts(local_counts, span_count=len(spans))
        if prepare:
            proxy = self.components.broadcast_proxy(local_proxy,
                rgb_shape=(len(next_source.packet.frames), next_source.packet.height,
                           next_source.packet.width, 3), eos=next_source.eos)
            state.prepared = _PreparedProxy(next_source, proxy)
        self._ready()
        self._check_cancel(state, cancelled)
        if cpu_output and sum(counts) > self.limits.max_output_frames:
            raise ValueError("H3 RGB batch exceeds deployment frame budget")
        for index, count in enumerate(counts):
            cpu, cpus[index] = cpus[index], None
            self._retire(state, count, cpu, emit, cpu_output=cpu_output)

    def _retire(self, state, count, cpu, emit, *, cpu_output):
        # Counts are published to every rank before retirement, so all ranks fail here together.
        if not 0 <= count <= len(state.frames):
            raise RuntimeError("decoder produced more frames than the accepted source")
        if cpu is None:
            for _ in range(count):
                state.frames.popleft()
        elif len(cpu) != count:
            raise RuntimeError("decoder frame metadata does not match RGB")
        elif not cpu_output:
            self._emit(state, cpu, emit)
        elif count:
            emit({"rgb": cpu, "frames": tuple(state.frames.popleft() for _ in range(count))})

    def _emit(self, state, rgb, emit):
        if rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise RuntimeError("causal decoder must produce CPU uint8 THWC RGB")
        for start in range(0, len(rgb), self.limits.max_output_frames):
            part = rgb[start:start + self.limits.max_output_frames]
            packet = H3RGBPacket(width=part.shape[2], height=part.shape[1],
                frames=tuple(state.frames.popleft() for _ in range(len(part))),
                rgb=part.numpy().tobytes())
            emit(packet)

    def cancel(self, session_identity):
        if self.state is not None and self.state.identity == session_identity:
            self.state.phase = "cancelled"

    def close(self, session_identity):
        if self.state is not None and self.state.identity == session_identity:
            self._ready()
            self.state.latent.reset()
            self.codec.begin_session()
            self.state = None

    def usage(self, session_identity):
        if self.state is None or self.state.identity != session_identity:
            return ResourceUsage()
        return ResourceUsage(bytes=self.state_reservation_bytes, slots={'causal_stream': 1})
