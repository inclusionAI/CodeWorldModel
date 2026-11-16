# SPDX-License-Identifier: Apache-2.0
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_causal import (
    H3CausalDecoderOptions,
    H3CodecParallel,
    MiniMaxH3CausalPipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    realtime,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.realtime import (
    H3RealtimeAdvance, H3RealtimeCodecRequest, H3RealtimeCompute,
    H3RealtimeOpen,
    H3RealtimeSource,
    H3RealtimeStage,
    combine_seed,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.realtime_codec import (
    H3RealtimeCodec,
)
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSessionCache
from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

RECIPE = json.loads((Path(__file__).parent / "h3_step600_recipe.json").read_text())
# Fixture latents are 2x2, so decoded RGB is 32x32.
RGB = (32, 32, 3)


@pytest.fixture(autouse=True)
def global_server_args():
    # PipelineStage reads the worker's global server args at construction.
    previous = server_args_module._global_server_args
    server_args_module.set_global_server_args(SimpleNamespace(comfyui_mode=False))
    yield
    server_args_module.set_global_server_args(previous)


class FakeSession:
    def __init__(self, model, static, *, recipe, fuse_first_denoise=False):
        self.static, self.fuse_first_denoise = static, fuse_first_denoise
        self.sources = []

    def append(self, source):
        self.sources.append(source)
        return () if len(self.sources) == 1 else self._span(32, False)

    def finish(self, source):
        self.sources.append(source)
        return self._span(7 if len(self.sources) == 3 else 5, True)

    def _span(self, count, eos):
        return (SimpleNamespace(video=count, plan=SimpleNamespace(eos=eos)),)

    def reset(self):
        self.sources.clear()


class FakeComponents:
    device = "cpu"
    model = object()

    def __init__(self, trace):
        self.trace = trace

    def condition(self, prompt, image):
        return torch.zeros(2, 4), torch.ones(2, dtype=torch.long)

    def encode_image(self, image):
        return torch.zeros(1, 1, 2, 2)

    def encode_proxy(self, frames, *, eos):
        return torch.full((1, 2 if eos else 5, 2, 2), float(frames[0, 0, 0, 0]))

    def encode_proxy_local(self, frames, *, eos):
        self.trace.append(("encode_local", len(frames), eos))
        return self.encode_proxy(frames, eos=eos)

    def broadcast_proxy(self, local, *, rgb_shape, eos):
        self.trace.append(("publish_proxy", rgb_shape[0], eos))
        return local

    def synchronize(self):
        self.trace.append(("join",))


class FakeDecoder:
    def __init__(self, trace):
        self.trace, self.extra, self.emitted, self.sessions = trace, 0, 0, 0

    def begin_session(self):
        self.sessions += 1

    def decode(self, count, *, eos):
        self.trace.append(("decode", count, eos))
        count += self.extra
        if not count:
            return None
        start, self.emitted = self.emitted, self.emitted + count
        return torch.arange(start, start + count, dtype=torch.uint8)[:, None, None, None].expand(
            count, *RGB).contiguous()


class FakeWorld:
    def __init__(self, trace, rank=0, published=()):
        self.trace, self.rank, self.ranks = trace, rank, list(range(8))
        self.published = list(published)

    def broadcast(self, tensor, *, src):
        if self.published:
            tensor = torch.tensor(self.published.pop(0))
        self.trace.append(("publish_counts", tuple(tensor.tolist())))
        return tensor


class Rank:
    def __init__(self, monkeypatch, *, overlap=False, rank=0, decoder=True, published=()):
        monkeypatch.setattr(realtime, "H3FixedHistorySession", FakeSession)
        self.trace = []
        self.decoder = FakeDecoder(self.trace) if decoder else None
        self.codec = H3RealtimeCodec(self.decoder, FakeWorld(self.trace, rank, published),
            output_rank=0, device="cpu")
        self.stage = H3RealtimeStage(components=FakeComponents(self.trace), codec=self.codec,
            recipe=RECIPE, codec_overlap=overlap, fuse_first_denoise=False)
        self.cache = RealtimeSessionCache(max_sessions=1)
        self.req = None

    def send(self, block_idx, request, session_id="session"):
        req = Req(sampling_params=SamplingParams(prompt="road"),
            realtime_session_id=session_id, block_idx=block_idx,
            condition_inputs={"h3": request})
        # The pipeline executor reads batch_size before every stage.
        assert req.batch_size == 1
        # Scheduler broadcast_pyobj pickles each request to every rank.
        self.req = req = pickle.loads(pickle.dumps(req))
        self.cache.attach(req)
        return self.stage.forward(req, None).output

    def open(self, seed=73, session_id="session"):
        return self.send(0, H3RealtimeOpen(np.zeros((2, 2, 3), np.uint8), seed),
            session_id)

    def advance(self, source, next_source=None):
        return self.send(source.input_seq + 1, H3RealtimeAdvance(source, next_source))

    def staged_advance(self, source, next_source=None):
        self.send(source.input_seq * 2 + 1, H3RealtimeCompute(source))
        return self.send(source.input_seq * 2 + 2,
                         H3RealtimeCodecRequest(source.input_seq, next_source))

    @property
    def state(self):
        return self.req.session.get_state(realtime.H3RealtimeState)


def frames(start, count):
    return (np.arange(start, start + count, dtype=np.uint8)[:, None, None, None]
            * np.ones((1, 2, 2, 3), np.uint8))


def test_stage_timing_uses_existing_boundaries_and_keeps_untraced_output(monkeypatch):
    rank = Rank(monkeypatch, overlap=True)
    rank.open()
    source = H3RealtimeSource(0, frames(0, 17), False)
    rank.send(1, H3RealtimeCompute(source))
    reply = rank.send(2, H3RealtimeCodecRequest(0, trace=True))
    assert reply['_type'] == 'H3RealtimeOutput'
    trace = reply['orchestration']
    assert trace['clock'] == 'native_worker_monotonic'
    assert [s['stage'] for s in trace['stages']] == ['proxy_prepare', 'render', 'codec']
    assert all(s['complete_ns'] >= s['dispatch_ns'] and
               s['wall_ms'] == (s['complete_ns'] - s['dispatch_ns']) / 1e6
               for s in trace['stages'])
    measured_calls = rank.trace.copy()
    baseline = Rank(monkeypatch, overlap=True)
    baseline.open()
    rgb = baseline.staged_advance(source)
    assert not isinstance(rgb, dict)
    assert rank.trace == measured_calls == baseline.trace, 'timing adds no model calls or synchronization'


def stream(explicit_eos):
    sources = [H3RealtimeSource(0, frames(0, 17), False),
               H3RealtimeSource(1, frames(17, 17), False),
               H3RealtimeSource(2, frames(34, 5), explicit_eos)]
    if not explicit_eos:
        sources.append(H3RealtimeSource(3, None, True))
    return sources


def run(rank, sources, *, lookahead):
    outputs = []
    for index, source in enumerate(sources):
        following = sources[index + 1] if lookahead and index + 1 < len(sources) else None
        outputs.append(rank.advance(source, following))
    return outputs


def noises(rank):
    latent = rank.state.latent
    return [latent.static.first_frame_noise] + [
        tensor for source in latent.sources
        for tensor in (source.reference_noise[0], source.target_noise, source.clean_noise)]


@pytest.mark.parametrize("explicit_eos", [False, True])
def test_overlap_and_lookahead_match_sequential_frames_noise_and_eos(monkeypatch, explicit_eos):
    serial = Rank(monkeypatch)
    assert serial.open() is None
    expected = run(serial, stream(explicit_eos), lookahead=False)
    assert [len(rgb) for rgb in expected] == ([0, 32, 7] if explicit_eos else [0, 32, 0, 7])
    assert all(rgb.dtype == np.uint8 and rgb.shape[1:] == RGB for rgb in expected)
    assert [row[2] for row in serial.trace if row[0] == "decode"] == [False, True]
    with pytest.raises(RuntimeError, match="ended"):
        serial.advance(H3RealtimeSource(4, None, True))

    overlapped = Rank(monkeypatch, overlap=True)
    overlapped.open()
    actual = run(overlapped, stream(explicit_eos), lookahead=True)
    assert [rgb.tobytes() for rgb in actual] == [rgb.tobytes() for rgb in expected]
    assert overlapped.state.phase == "ended"
    # The held 5-frame tail is never prepared; only 17-frame or EOS lookahead encodes.
    assert [row for row in overlapped.trace if row[0] == "encode_local"] == (
        [("encode_local", 17, False), ("encode_local", 5, True)] if explicit_eos
        else [("encode_local", 17, False)])
    serial_noise = noises(serial)
    assert all(torch.equal(a, b) for a, b in zip(noises(overlapped), serial_noise, strict=True))


@pytest.mark.parametrize("overlap,explicit_eos", [(False, True), (True, False), (True, True)])
def test_split_compute_and_codec_preserves_advance_output_and_state(monkeypatch, overlap, explicit_eos):
    serial = Rank(monkeypatch, overlap=overlap)
    serial.open()
    sources = stream(explicit_eos)
    expected = run(serial, sources, lookahead=overlap)

    staged = Rank(monkeypatch, overlap=overlap)
    staged.open()
    actual = []
    for index, source in enumerate(sources):
        following = sources[index + 1] if overlap and index + 1 < len(sources) else None
        actual.append(staged.staged_advance(source, following))

    assert [rgb.tobytes() for rgb in actual] == [rgb.tobytes() for rgb in expected]
    assert staged.state.phase == "ended"
    assert staged.state.computed is None
    assert staged.trace == serial.trace
    assert all(torch.equal(a, b) for a, b in zip(noises(staged), noises(serial), strict=True))


def test_split_codec_requires_the_matching_computed_source(monkeypatch):
    rank = Rank(monkeypatch)
    rank.open()
    source = H3RealtimeSource(0, frames(0, 17), False)
    with pytest.raises(RuntimeError, match="requires a computed"):
        rank.send(1, H3RealtimeCodecRequest(0))
    rank.send(1, H3RealtimeCompute(source))
    with pytest.raises(RuntimeError, match="awaiting codec"):
        rank.send(2, H3RealtimeCompute(H3RealtimeSource(1, frames(17, 17), False)))
    with pytest.raises(ValueError, match="does not match"):
        rank.send(2, H3RealtimeCodecRequest(1))
    assert rank.state.phase == "open"
    rank.send(3, H3RealtimeCodecRequest(0))
    rank.send(4, H3RealtimeCompute(H3RealtimeSource(1, frames(17, 17), False)))
    assert rank.cache.release("session")
    rank.open()
    assert rank.state.computed is None and rank.state.prepared is None


@pytest.mark.parametrize("overlap", [False, True])
def test_shared_cpu_delivery_matches_default_and_owns_output(monkeypatch, overlap):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.shared_rgb import H3SharedRGB

    baseline = Rank(monkeypatch, overlap=overlap)
    baseline.open()
    sources = stream(True)
    expected = run(baseline, sources, lookahead=overlap)
    candidate = Rank(monkeypatch, overlap=overlap)
    candidate.open()
    actual = []
    for index, source in enumerate(sources):
        following = sources[index + 1] if overlap and index + 1 < len(sources) else None
        candidate.send(index * 2 + 1, H3RealtimeCompute(source))
        reply = candidate.send(index * 2 + 2, H3RealtimeCodecRequest(
            source.input_seq, following, shared_cpu_output=True))
        assert isinstance(reply, H3SharedRGB)
        buffers = reply.materialize()
        assert all(buffer.is_shared() for buffer in buffers)
        actual.append(b''.join(buffer.numpy().tobytes() for buffer in buffers))
    assert actual == [rgb.tobytes() for rgb in expected]
    assert candidate.trace == baseline.trace
    assert candidate.state.phase == 'ended'


def test_seed_derivation_matches_raven_literals():
    # raven.common.seed: combine_seed and RandomState(73).fork("cwm_rgb", 0, 1).seed.
    assert combine_seed(2026, "first_frame", "static") == 8708076201935202333
    assert combine_seed(73, "cwm_rgb", 0, 1) == 254678466011478084


def test_release_resets_streams_and_reopen_reproduces_noise(monkeypatch):
    rank = Rank(monkeypatch)
    rank.open()
    rank.advance(H3RealtimeSource(0, frames(0, 17), False))
    first = [tensor.clone() for tensor in noises(rank)]
    latent, sessions = rank.state.latent, rank.decoder.sessions
    assert rank.cache.release("session")
    assert latent.sources == [] and rank.decoder.sessions == sessions + 1
    with pytest.raises(ValueError, match="Missing realtime session"):
        rank.advance(H3RealtimeSource(1, frames(17, 17), False))
    rank.open()
    rank.advance(H3RealtimeSource(0, frames(0, 17), False))
    assert all(torch.equal(a, b) for a, b in zip(noises(rank), first, strict=True))


def test_invalid_sources_are_rejected_without_poisoning_the_session(monkeypatch):
    rank = Rank(monkeypatch)
    rank.open()
    with pytest.raises(ValueError, match="17-frame"):
        rank.advance(H3RealtimeSource(0, frames(0, 6), False))
    with pytest.raises(ValueError, match="final 5"):
        rank.advance(H3RealtimeSource(0, None, True))
    with pytest.raises(ValueError, match="input_seq"):
        rank.advance(H3RealtimeSource(1, frames(0, 17), False))
    with pytest.raises(ValueError, match="H3RealtimeSource"):
        rank.send(1, H3RealtimeAdvance(H3RealtimeSource(True, frames(0, 17), False)))
    rank.advance(H3RealtimeSource(0, frames(0, 17), False))
    rank.advance(H3RealtimeSource(1, frames(17, 17), False))
    rank.advance(H3RealtimeSource(2, frames(34, 5), False))
    with pytest.raises(ValueError, match="final 5-frame tail"):
        rank.advance(H3RealtimeSource(3, frames(39, 17), False))
    assert rank.state.phase == "open"
    assert len(rank.advance(H3RealtimeSource(3, None, True))) == 7
    assert rank.state.phase == "ended"


@pytest.mark.parametrize("changed", ["sequence", "frames", "eos"])
def test_prepared_proxy_requires_the_exact_next_source(monkeypatch, changed):
    rank = Rank(monkeypatch, overlap=True)
    rank.open()
    queued = H3RealtimeSource(1, frames(17, 17), False)
    rank.advance(H3RealtimeSource(0, frames(0, 17), False), queued)
    altered = H3RealtimeSource(2 if changed == "sequence" else 1,
        frames(18 if changed == "frames" else 17, 17), changed == "eos")
    with pytest.raises(ValueError, match="match"):
        rank.send(1, H3RealtimeAdvance(altered))
    with pytest.raises(RuntimeError, match="failed"):
        rank.advance(queued)
    rank.open()
    assert rank.state.prepared is None
    rank.advance(H3RealtimeSource(0, frames(0, 17), False))


def test_lookahead_requires_codec_overlap_and_fails_the_session(monkeypatch):
    rank = Rank(monkeypatch)
    rank.open()
    with pytest.raises(ValueError, match="lookahead"):
        rank.advance(H3RealtimeSource(0, frames(0, 17), False),
                     H3RealtimeSource(1, frames(17, 17), False))
    assert rank.state.phase == "failed"


def test_decoder_overcount_fails_every_rank_the_same_way(monkeypatch):
    rank = Rank(monkeypatch)
    rank.open()
    rank.advance(H3RealtimeSource(0, frames(0, 17), False))
    rank.decoder.extra = 3
    with pytest.raises(RuntimeError, match="more frames"):
        rank.advance(H3RealtimeSource(1, frames(17, 17), False))
    assert rank.state.phase == "failed"


@pytest.mark.parametrize("rank_index, decoder", [(1, True), (5, False)])
def test_nonoutput_ranks_return_nothing_and_retire_published_counts(
        monkeypatch, rank_index, decoder):
    rank = Rank(monkeypatch, rank=rank_index, decoder=decoder, published=[[32], [7]])
    rank.open()
    assert run(rank, stream(False), lookahead=False) == [None] * 4
    assert rank.state.phase == "ended" and rank.state.pending_frames == 0


def test_overlap_encodes_next_while_decoding_then_publishes_after_one_join(monkeypatch):
    rank = Rank(monkeypatch, overlap=True)
    rank.open()
    rank.advance(H3RealtimeSource(0, frames(0, 17), False))
    rank.state.latent.append = lambda source: tuple(
        SimpleNamespace(video=count, plan=SimpleNamespace(eos=False)) for count in (0, 6, 11))
    rank.trace.clear()
    rgb = rank.advance(H3RealtimeSource(1, frames(17, 17), False),
                       H3RealtimeSource(2, frames(34, 17), False))
    assert rank.trace[2:] == [("encode_local", 17, False), ("decode", 0, False),
        ("decode", 6, False), ("decode", 11, False), ("join",),
        ("publish_counts", (0, 6, 11)), ("publish_proxy", 17, False)]
    assert len(rgb) == 17 and rank.state.pending_frames == 17


@pytest.mark.parametrize("suffix_backend", ["sdpa", "fa3"])
def test_pipeline_loads_groups_norm_and_decoder_only_on_decoder_members(monkeypatch, suffix_backend):
    from sglang.kernels.ops import diffusion
    from sglang.multimodal_gen.registry import (
        get_pipeline_class,
        get_pipeline_config_classes,
    )
    from sglang.multimodal_gen.runtime.distributed import parallel_state
    from sglang.multimodal_gen.runtime.loader import minimax_h3_causal as loader
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
        causal_streaming,
    )
    from sglang.multimodal_gen.runtime.pipelines import (
        minimax_h3_causal_realtime_pipeline as pipeline,
    )

    name = "MiniMaxH3CausalRealtimePipeline"
    assert get_pipeline_class(name) is pipeline.MiniMaxH3CausalRealtimePipeline
    assert get_pipeline_config_classes(name)[0] is MiniMaxH3CausalPipelineConfig
    config = MiniMaxH3CausalPipelineConfig(recipe=RECIPE, codec_overlap=True,
        codec_parallel=H3CodecParallel((2, 3)), lossless_norm=True,
        decoder=H3CausalDecoderOptions("d.yaml", "d.safetensors", backend="fast",
            compile=True, compile_mode="max-autotune-no-cudagraphs", tp_fused_post=True,
            cudnn_sdpa=True, suffix_attention_backend=suffix_backend))
    assert config.transformer_class_override == "MiniMaxH3CausalDiTModel"
    events, decoders = [], []
    for rank in range(8):
        coordinator = SimpleNamespace(rank=rank)
        groups = (loader.H3CodecGroup(coordinator, (2, 3)), loader.H3CodecGroup(coordinator))
        def bind(**kwargs):
            events.append(("bind", kwargs))
            return groups
        def load(args, *, encoder_group):
            events.append(("load", encoder_group is groups[0]))
            return SimpleNamespace(model=f"model{rank}", device="cpu")
        def decoder(**kwargs):
            events.append(("decoder",))
            decoders.append(kwargs)
            return SimpleNamespace(begin_session=lambda: None)
        monkeypatch.setattr(loader, "bind_h3_codec_groups", bind)
        monkeypatch.setattr(loader, "load_h3_causal_components", load)
        monkeypatch.setattr(diffusion, "mount_minimax_h3_norm_modulate",
            lambda model: events.append(("mount", model)) or True)
        monkeypatch.setattr(causal_streaming, "H3CausalVideoDecoder", decoder)
        monkeypatch.setattr(parallel_state, "get_world_group",
            lambda: SimpleNamespace(rank=rank, ranks=list(range(8))))
        events.clear()
        modules = pipeline.load_h3_realtime_modules(SimpleNamespace(pipeline_config=config))
        assert events[:3] == [
            ("bind", dict(encoder_ranks=(2, 3), decoder_ranks=(0, 1), output_rank=0)),
            ("load", True), ("mount", f"model{rank}")]
        assert events[3:] == ([("decoder",)] if rank < 2 else [])
        assert modules["h3_codec"].is_output == (rank == 0)
    assert [(d["tp_group"].rank, d["cudnn_sdpa"], d["backend"]) for d in decoders] == [
        (0, True, "fast"), (1, True, "fast")]
    assert all(d["suffix_attention_backend"] == suffix_backend for d in decoders)
    assert all(d["compile_mode"] == "max-autotune-no-cudagraphs" and d["tp_fused_post"] for d in decoders)


@pytest.mark.parametrize("encoder", [None, (0, 1), (2, 3), (4, 5)])
def test_overlap_config_requires_the_disjoint_e2_dtp2_placement(encoder):
    placement = None if encoder is None else H3CodecParallel(encoder)
    MiniMaxH3CausalPipelineConfig(recipe=RECIPE, codec_parallel=placement)
    if encoder == (2, 3):
        MiniMaxH3CausalPipelineConfig(recipe=RECIPE, codec_parallel=placement,
                                      codec_overlap=True)
    else:
        with pytest.raises(ValueError, match="requires E2 ranks 2/3"):
            MiniMaxH3CausalPipelineConfig(recipe=RECIPE, codec_parallel=placement,
                                          codec_overlap=True)


def test_overlap_config_accepts_disjoint_e4_dtp4():
    placement = H3CodecParallel((2, 3, 6, 7), (0, 1, 4, 5))
    config = MiniMaxH3CausalPipelineConfig(recipe=RECIPE, codec_parallel=placement, codec_overlap=True)
    assert config.codec_parallel == placement
    with pytest.raises(ValueError, match='requires E2'):
        MiniMaxH3CausalPipelineConfig(recipe=RECIPE,
            codec_parallel=H3CodecParallel((2, 3, 6, 7), (0, 1)), codec_overlap=True)
