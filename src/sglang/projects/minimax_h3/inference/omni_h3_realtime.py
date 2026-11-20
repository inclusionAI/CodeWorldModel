"""Omni H3 media bridge on SGLang's upstream realtime sessions (ADR 0001, U4).

Omni keeps frame stamps. Block 0 opens; each source then uses a compute request
and a codec request, allowing lookahead to arrive during model computation.
"""
import atexit
import multiprocessing
import multiprocessing.connection
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from sglang_omni.models.minimax_h3.delivery import H3CPUFrameBatch
from sglang_omni.models.minimax_h3.session import H3RGBPacket
from sglang_omni.proto.session import ResourceUsage

from .omni_h3_assets import read_native_bundle
from .omni_h3_runtime import check_deployment

PIPELINE = "MiniMaxH3CausalRealtimePipeline"


def pipeline_config(manifest, *, overlap):
    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_causal import (
        H3CausalDecoderOptions,
        H3CodecParallel,
        MiniMaxH3CausalPipelineConfig,
    )

    codec, dit = manifest["codec"], manifest.get("dit", {})
    if codec["encoder"] not in ("h3", "tae") or codec["decoder"] != "causal":
        raise ValueError("realtime H3 requires an H3 or TAE proxy encoder and the causal decoder")
    parallel = manifest.get("codec_parallel")
    return MiniMaxH3CausalPipelineConfig(
        recipe=manifest["recipe"],
        decoder=H3CausalDecoderOptions(
            decoder_config=codec["decoder_config"],
            decoder_weights=codec["decoder_weights"],
            window=codec["window"],
            backend=codec["backend"],
            compile=codec["compile"],
            cuda_graph=codec["cuda_graph"],
            cudnn_sdpa=codec.get("suffix_sdpa_backend", "default") == "cudnn",
            spatial_attention_backend=codec.get("spatial_attention_backend", "fa2"),
            compact_suffix_kv=codec.get("compact_suffix_kv", False),
            suffix_attention_backend="fa3" if codec.get("suffix_sdpa_backend") == "fa3" else "sdpa",
            compile_mode=codec.get("compile_mode", "default"),
            tp_fused_post=codec.get("tp_fused_post", False),
        ),
        codec_parallel=None if parallel is None else H3CodecParallel(
            tuple(parallel["encoder_ranks"]), tuple(parallel["decoder_ranks"])),
        codec_overlap=overlap,
        fuse_first_denoise=dit.get("fuse_first_denoise", False),
        lossless_norm=dit.get("norm_quality", "exact") == "lossless",
        cwm_silence_path=manifest.get('cwm', {}).get('silence_path'),
        cwm_proxy_encode_size=(manifest.get('cwm', {}).get('proxy_encode_height', 768),
                                 manifest.get('cwm', {}).get('proxy_encode_width', 1344)),
        cwm_proxy_resize=manifest.get('cwm', {}).get('proxy_resize', 'lanczos'),
        tae_weights=codec.get('tae_weights') if codec['encoder'] == 'tae' else None,
    )


def _realtime_types():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.realtime import (
        H3RealtimeAck, H3RealtimeCodecRequest, H3RealtimeCompute,
        H3RealtimeOpen, H3RealtimeSource,
    )
    return (
        H3RealtimeOpen, H3RealtimeSource, H3RealtimeCompute,
        H3RealtimeCodecRequest, H3RealtimeAck,
    )


def _pixels(packet):
    return np.frombuffer(packet.rgb, dtype=np.uint8).reshape(
        len(packet.frames), packet.height, packet.width, 3)


@dataclass
class _Session:
    identity: object
    id: str
    prompt: str
    block_idx: int = 0
    # Native input_seq counts sources from 0; Omni chunk seq numbers need not.
    input_seq: int = -1
    frames: deque = field(default_factory=deque)
    geometry: tuple | None = None
    next_pts_ms: float = 0.0
    phase: str = "open"
    pending_rgb_acks: list[str] = field(default_factory=list)


class NativeH3RealtimeBridge:
    """One Omni session at a time over the native scheduler client.

    Each call waits on the native workers as well, so a dead rank fails the
    call instead of leaving it blocked in a collective.
    """

    def __init__(self, generator, client, reservation, *, codec_overlap, max_output_frames):
        self.generator, self.client = generator, client
        self.workers = list(generator.local_scheduler_process or ())
        self.reservation = reservation
        self.codec_overlap = codec_overlap
        self.max_output_frames = max_output_frames
        self.session = None

    def _call(self, payload):
        reader, writer = multiprocessing.Pipe(duplex=False)
        outcome = {}

        def run():
            try:
                outcome["result"] = self.client.forward(payload)
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                writer.close()

        threading.Thread(target=run, name="h3-native-forward", daemon=True).start()
        sentinels = {process.sentinel: process for process in self.workers}
        try:
            ready = multiprocessing.connection.wait([reader, *sentinels])
        finally:
            reader.close()
        if reader not in ready:
            process = sentinels[ready[0]]
            raise RuntimeError(f"native H3 worker {process.pid} exited with code {process.exitcode}")
        if "error" in outcome:
            raise outcome["error"]
        result = outcome["result"]
        if result.error is not None:
            raise RuntimeError(result.error)
        return result.output

    def _send(self, session, request):
        from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

        block_idx = session.block_idx
        session.block_idx += 1
        return self._call(Req(
            sampling_params=SamplingParams(request_id=f"{session.id}-{block_idx}",
                                           prompt=session.prompt, save_output=False),
            realtime_session_id=session.id, block_idx=block_idx,
            condition_inputs={"h3": request}))

    def _current(self, identity):
        session = self.session
        if session is None or session.identity != identity:
            raise ValueError("unknown or stale H3 session identity")
        if session.phase != "open":
            raise RuntimeError(f"H3 session is {session.phase}; close and reopen")
        return session

    def open(self, identity, request):
        if self.session is not None:
            raise ValueError("native H3 bridge already owns a session")
        open_type, _, _, _, _ = _realtime_types()
        self.session = session = _Session(identity, f"h3-{uuid.uuid4().hex}", request.prompt)
        try:
            self._send(session, open_type(_pixels(request.first_image)[0], request.seed))
        except BaseException:
            session.phase = "failed"
            raise

    def _source(self, session, packet, eos, offset):
        _, source_type, _, _, _ = _realtime_types()
        return source_type(session.input_seq + offset,
                           None if packet is None else _pixels(packet), eos)

    def _accept(self, session, packet):
        # Stamps are Omni-only, so their order and the fixed geometry are checked here.
        geometry = (packet.width, packet.height)
        if session.geometry not in (None, geometry):
            raise ValueError("proxy geometry must stay fixed for a session")
        if packet.frames[0].pts_ms < session.next_pts_ms:
            raise ValueError("source time overlaps the preceding packet")
        session.geometry = geometry
        session.next_pts_ms = packet.frames[-1].pts_ms + packet.frames[-1].duration_ms
        session.frames.extend(packet.frames)

    def _check_cancel(self, session, cancelled):
        if cancelled.is_set():
            session.phase = "cancelled"
            raise RuntimeError("H3 session was cancelled; close and reopen")

    def _advance(self, identity, packet, eos, cancelled, emit, peek_next=None,
                 *, cpu_output=False):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.shared_rgb import H3SharedRGB

        session = self._current(identity)
        _, _, compute_type, codec_type, ack_type = _realtime_types()
        try:
            self._check_cancel(session, cancelled)
            source = self._source(session, packet, eos, 1)
            if packet is not None:
                self._accept(session, packet)
            compute_acks = tuple(session.pending_rgb_acks)
            session.pending_rgb_acks.clear()
            self._send(session, compute_type(source, ack_ids=compute_acks))
            self._check_cancel(session, cancelled)
            # Inputs arriving during DiT can still enter this codec phase.
            lookahead = None if eos or peek_next is None else peek_next()
            following = None if lookahead is None else self._source(
                session, lookahead.packet, lookahead.eos, 2)
            rgb = self._send(session, codec_type(
                source.input_seq, following, shared_cpu_output=cpu_output, trace=True))
            orchestration = None
            if isinstance(rgb, dict) and rgb.get('_type') == 'H3RealtimeOutput':
                orchestration, rgb = rgb['orchestration'], rgb['rgb']
            # Receive storage before observing cancellation so transferred FDs
            # are claimed and released even when the output is discarded.
            if isinstance(rgb, H3SharedRGB):
                rgb = rgb.materialize()
            if isinstance(rgb, dict) and rgb.get("_type") == "H3OmniStreamRef":
                if not hasattr(self, "_rgb_shm_reader"):
                    from .h3_shm_transport import H3ShmOutputReader

                    self._rgb_shm_reader = H3ShmOutputReader()
                rgb, object_id, _ = self._rgb_shm_reader.read(rgb)
                session.pending_rgb_acks.append(object_id)
            session.input_seq += 1
            self._check_cancel(session, cancelled)
            if cpu_output:
                result = self._retire_cpu(session, rgb, orchestration)
            else:
                self._retire(session, rgb, emit, orchestration)
                result = None
            if eos:
                if session.pending_rgb_acks:
                    # Ack after materializing the final RGB, even though the
                    # realtime stage has already entered its EOS state.
                    self._send(session, ack_type(tuple(session.pending_rgb_acks)))
                    session.pending_rgb_acks.clear()
                if session.frames:
                    raise RuntimeError("decoder EOS did not cover every accepted source frame")
                session.phase = "ended"
            return result
        except BaseException:
            if session.phase == "open":
                session.phase = "failed"
            raise

    def _retire(self, session, rgb, emit, orchestration=None):
        if rgb is None:
            raise RuntimeError("output rank returned no RGB for an H3 advance")
        if len(rgb) > len(session.frames):
            raise RuntimeError("decoder produced more frames than the accepted source")
        for start in range(0, len(rgb), self.max_output_frames):
            part = rgb[start:start + self.max_output_frames]
            emit(H3RGBPacket(width=part.shape[2], height=part.shape[1],
                frames=tuple(session.frames.popleft() for _ in range(len(part))),
                rgb=np.ascontiguousarray(part).tobytes(), orchestration=orchestration))

    def _retire_cpu(self, session, rgb, orchestration=None) -> H3CPUFrameBatch:
        if rgb is None:
            raise RuntimeError("output rank returned no RGB for an H3 codec request")
        shared = isinstance(rgb, tuple)
        buffers = rgb if shared else (rgb,)
        if sum(len(buffer) for buffer in buffers) > len(session.frames):
            raise RuntimeError("decoder produced more frames than the accepted source")
        packets = []
        for buffer in buffers:
            for start in range(0, len(buffer), self.max_output_frames):
                part = buffer[start:start + self.max_output_frames]
                if shared:
                    cpu = part
                else:
                    part = np.ascontiguousarray(part)
                    cpu = torch.empty(part.shape, dtype=torch.uint8).share_memory_()
                    cpu.copy_(torch.from_numpy(part))
                packets.append({
                    "rgb": cpu,
                    "orchestration": orchestration,
                    "frames": tuple(
                        session.frames.popleft() for _ in range(len(part))
                    ),
                })
        return tuple(packets)

    def append(self, identity, packet, cancelled, emit):
        self._advance(identity, packet, False, cancelled, emit)

    def finish(self, identity, cancelled, emit):
        self._advance(identity, None, True, cancelled, emit)

    def advance(self, identity, source, cancelled, emit, peek_next=None):
        if not self.codec_overlap:
            raise ValueError("native H3 codec overlap is not enabled")
        self._advance(identity, source.packet, source.eos, cancelled, emit,
                      peek_next)

    def advance_cpu(self, identity, source, cancelled, peek_next=None) -> H3CPUFrameBatch:
        if not self.codec_overlap:
            raise ValueError("native H3 codec overlap is not enabled")
        return self._advance(identity, source.packet, source.eos, cancelled, None,
                             peek_next, cpu_output=True)

    def cancel(self, identity):
        if self.session is not None and self.session.identity == identity:
            self.session.phase = "cancelled"

    def close(self, identity):
        from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
            ReleaseRealtimeSessionReq,
        )

        session = self.session
        if session is None or session.identity != identity:
            return
        self.session = None
        if session.pending_rgb_acks:
            _, _, _, _, ack_type = _realtime_types()
            self._send(session, ack_type(tuple(session.pending_rgb_acks)))
            session.pending_rgb_acks.clear()
        if session.block_idx:
            self._call(ReleaseRealtimeSessionReq(session.id))

    def shutdown(self):
        generator, self.generator = self.generator, None
        reader = getattr(self, "_rgb_shm_reader", None)
        if reader is not None:
            reader.close()
            self._rgb_shm_reader = None
        if generator is not None:
            generator.shutdown()

    def usage(self, identity):
        return ResourceUsage(bytes=self.reservation if self.session is not None
                             and self.session.identity == identity else 0)



def create_runtime(*, model_path, limits):
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
    from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    manifest = read_native_bundle(model_path)
    check_deployment(manifest, limits, realtime=True)
    overlap = getattr(limits, "codec_overlap", False)
    generator = DiffGenerator.from_server_args(ServerArgs(**manifest["server_args"],
        pipeline_class_name=PIPELINE, pipeline_config=pipeline_config(manifest, overlap=overlap)))
    bridge = NativeH3RealtimeBridge(generator, sync_scheduler_client,
        manifest["state_reservation_bytes"], codec_overlap=overlap,
        max_output_frames=limits.max_output_frames)
    atexit.register(bridge.shutdown)
    return bridge
