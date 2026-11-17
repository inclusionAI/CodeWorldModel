import multiprocessing
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from inference.omni_h3_realtime import NativeH3RealtimeBridge
from sglang.multimodal_gen.runtime.entrypoints.control_requests import ReleaseRealtimeSessionReq
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.realtime import (
    H3RealtimeCodecRequest, H3RealtimeCompute, H3RealtimeOpen,
)
from sglang_omni.models.minimax_h3.session import (
    H3FrameStamp, H3OpenRequest, H3RGBPacket, H3SourceInput,
)


def test_decoder_compile_options_reach_native_pipeline_and_old_bundle_defaults():
    import json
    from pathlib import Path
    from inference.omni_h3_realtime import pipeline_config
    recipe = json.loads((Path(__file__).parents[1] / 'configs/recipes/omni-h3-longlive.json').read_text())
    codec = dict(encoder='h3', decoder='causal', decoder_config='decoder.json',
        decoder_weights='decoder.safetensors', window=16, backend='fast',
        compile=True, cuda_graph=True, compile_mode='max-autotune-no-cudagraphs', tp_fused_post=True)
    manifest = dict(recipe=recipe, codec=codec,
        codec_parallel=dict(encoder_ranks=[2, 3], decoder_ranks=[0, 1, 4, 5]))
    options = pipeline_config(manifest, overlap=True).decoder
    assert options.compile_mode == 'max-autotune-no-cudagraphs' and options.tp_fused_post
    del codec['compile_mode'], codec['tp_fused_post']
    options = pipeline_config(manifest, overlap=True).decoder
    assert options.compile_mode == 'default' and not options.tp_fused_post


def test_tae_proxy_encoder_reaches_the_shared_pipeline_config():
    import json
    from pathlib import Path
    from inference.omni_h3_realtime import pipeline_config

    recipe = json.loads((Path(__file__).parents[1] / 'configs/recipes/omni-h3-longlive.json').read_text())
    manifest = dict(recipe=recipe, codec=dict(
        encoder='tae', tae_weights='/assets/tae.pth', decoder='causal',
        decoder_config='decoder.json', decoder_weights='decoder.safetensors',
        window=16, backend='fast', compile=False, cuda_graph=False),
        codec_parallel=dict(encoder_ranks=[2, 3], decoder_ranks=[0, 1]))
    config = pipeline_config(manifest, overlap=False)
    assert config.tae_weights == '/assets/tae.pth'


def packet(start, count):
    return H3RGBPacket(width=2, height=2,
        rgb=np.repeat(np.arange(start, start + count, dtype=np.uint8), 12).tobytes(),
        frames=tuple(H3FrameStamp(source_frame_id=f'frame-{i}', pts_ms=float(i * 40),
                                  duration_ms=40.0) for i in range(start, start + count)))


OPEN = H3OpenRequest(prompt='road', seed=7, first_image=packet(0, 1))


class Client:
    """Stands in for the scheduler: decodes `counts[k]` frames on the k-th advance."""

    def __init__(self, counts):
        self.counts, self.requests = list(counts), []

    def forward(self, req):
        self.requests.append(req)
        if isinstance(req, ReleaseRealtimeSessionReq):
            return SimpleNamespace(error=None, output=None)
        if req.block_idx == 0:
            return SimpleNamespace(error=None, output=None)
        # The scheduler test double may load the SGLang submodule through a
        # different checkout path; protocol name is the stable boundary here.
        if type(req.condition_inputs['h3']).__name__ == H3RealtimeCompute.__name__:
            return SimpleNamespace(error=None, output=None)
        if type(req.condition_inputs['h3']).__name__ != H3RealtimeCodecRequest.__name__:
            return SimpleNamespace(error=None, output=None)
        return SimpleNamespace(error=None, output=np.zeros((self.counts.pop(0), 2, 2, 3), np.uint8))


def bridge(client, workers=(), *, overlap=False):
    return NativeH3RealtimeBridge(SimpleNamespace(local_scheduler_process=list(workers)),
        client, 128, codec_overlap=overlap, max_output_frames=8)


def test_append_finish_retire_stamps_in_order_and_release_on_close():
    client = Client([0, 13, 9])
    native = bridge(client)
    native.open('a', OPEN)
    emitted, cancelled = [], threading.Event()
    native.append('a', packet(0, 17), cancelled, emitted.append)
    native.append('a', packet(17, 5), cancelled, emitted.append)
    native.finish('a', cancelled, emitted.append)
    native.close('a')

    stamps = [frame.source_frame_id for chunk in emitted for frame in chunk.frames]
    assert stamps == [f'frame-{i}' for i in range(22)]
    assert [len(chunk.frames) for chunk in emitted] == [8, 5, 8, 1]
    *reqs, release = client.requests
    assert [req.block_idx for req in reqs] == list(range(7))
    # Req.batch_size reads the prompt, so every request carries it.
    assert all(req.prompt == 'road' and req.batch_size == 1 for req in reqs)
    assert type(reqs[0].condition_inputs['h3']).__name__ == H3RealtimeOpen.__name__
    sources = [req.condition_inputs['h3'].source for req in reqs[1:]
               if type(req.condition_inputs['h3']).__name__ == H3RealtimeCompute.__name__]
    assert [(s.input_seq, None if s.frames is None else len(s.frames), s.eos) for s in sources] == [
        (0, 17, False), (1, 5, False), (2, None, True)]
    assert release.session_id == reqs[0].realtime_session_id
    assert native.session is None


def test_advance_carries_lookahead_with_native_sequence_numbers():
    client = Client([0, 22])
    native = bridge(client, overlap=True)
    native.open('a', OPEN)
    first = H3SourceInput(input_seq=40, packet=packet(0, 17), eos=False)
    last = H3SourceInput(input_seq=41, packet=packet(17, 5), eos=True)
    emitted, cancelled = [], threading.Event()
    def late_arrival():
        assert type(client.requests[-1].condition_inputs['h3']).__name__ == H3RealtimeCompute.__name__
        return last

    native.advance('a', first, cancelled, emitted.append, late_arrival)
    native.advance('a', last, cancelled, emitted.append, lambda: pytest.fail('peek after EOS'))
    stages = [req.condition_inputs['h3'] for req in client.requests[1:]]
    computes = [request for request in stages if type(request).__name__ == H3RealtimeCompute.__name__]
    codecs = [request for request in stages if type(request).__name__ == H3RealtimeCodecRequest.__name__]
    assert [request.source.input_seq for request in computes] == [0, 1]
    assert codecs[0].input_seq == 0 and codecs[0].next_source.input_seq == 1
    assert codecs[0].next_source.eos and codecs[1].next_source is None
    assert np.array_equal(codecs[0].next_source.frames, computes[1].source.frames)
    assert sum(len(chunk.frames) for chunk in emitted) == 22
    assert native.session.phase == 'ended'


def test_cpu_delivery_owns_rgb_and_retires_eos_stamps():
    client = Client([0, 22])
    native = bridge(client, overlap=True)
    native.open('a', OPEN)
    first = H3SourceInput(input_seq=0, packet=packet(0, 17), eos=False)
    last = H3SourceInput(input_seq=1, packet=packet(17, 5), eos=True)
    assert native.advance_cpu('a', first, threading.Event()) == ()
    source_rgb = np.ones((22, 2, 2, 3), dtype=np.uint8)
    forward = client.forward

    def output(req):
        result = forward(req)
        if type(req.condition_inputs['h3']).__name__ == H3RealtimeCodecRequest.__name__:
            assert req.condition_inputs['h3'].trace
            result.output = dict(_type='H3RealtimeOutput', rgb=source_rgb,
                orchestration=dict(clock='native_worker_monotonic', stages=[
                    dict(stage='codec', dispatch_ns=1000, complete_ns=2000, wall_ms=.001)]))
        return result

    client.forward = output
    result = native.advance_cpu('a', last, threading.Event())
    source_rgb.fill(0)
    assert all(item['rgb'].is_shared() and item['rgb'].sum().item() > 0 for item in result)
    assert all(item['orchestration']['stages'][0]['stage'] == 'codec' for item in result)
    assert [stamp.source_frame_id for item in result for stamp in item['frames']] == [
        f'frame-{i}' for i in range(22)]
    assert native.session.phase == 'ended' and not native.session.frames


def test_shared_cpu_reply_reuses_storage_and_survives_session_close():
    import torch
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.shared_rgb import H3SharedRGB

    client = Client([0, 22])
    native = bridge(client, overlap=True)
    native.open('a', OPEN)
    native.advance_cpu('a', H3SourceInput(input_seq=0, packet=packet(0, 17), eos=False), threading.Event())
    buffers = [torch.full((13, 2, 2, 3), 7, dtype=torch.uint8).share_memory_(),
               torch.full((9, 2, 2, 3), 8, dtype=torch.uint8).share_memory_()]
    forward = client.forward

    def output(req):
        result = forward(req)
        if not isinstance(req, ReleaseRealtimeSessionReq) and isinstance(req.condition_inputs['h3'], H3RealtimeCodecRequest):
            assert req.condition_inputs['h3'].shared_cpu_output
            assert req.condition_inputs['h3'].trace
            result.output = dict(_type='H3RealtimeOutput', rgb=H3SharedRGB.from_tensors(buffers),
                orchestration=dict(clock='native_worker_monotonic', stages=[]))
        return result

    client.forward = output
    result = native.advance_cpu('a', H3SourceInput(input_seq=1, packet=packet(17, 5), eos=True), threading.Event())
    assert result[0]['rgb'].untyped_storage().data_ptr() == buffers[0].untyped_storage().data_ptr()
    assert [len(item['frames']) for item in result] == [8, 5, 8, 1]
    assert torch.equal(torch.cat([item['rgb'] for item in result]), torch.cat(buffers))
    assert all(item['orchestration']['clock'] == 'native_worker_monotonic' for item in result)
    native.close('a')
    assert result[0]['rgb'][0, 0, 0, 0] == 7


def test_cancel_after_codec_still_materializes_shared_reply(monkeypatch):
    import torch
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.shared_rgb import H3SharedRGB

    native = bridge(Client([13]), overlap=True)
    native.open('a', OPEN)
    cancelled = threading.Event()
    forward = native.client.forward
    consumed = []
    materialize = H3SharedRGB.materialize

    def read(ref):
        result = materialize(ref)
        consumed.append(len(result[0]))
        return result

    monkeypatch.setattr(H3SharedRGB, 'materialize', read)

    def output(req):
        result = forward(req)
        if isinstance(req.condition_inputs['h3'], H3RealtimeCodecRequest):
            result.output = H3SharedRGB.from_tensors([
                torch.zeros((13, 2, 2, 3), dtype=torch.uint8).share_memory_()])
            cancelled.set()
        return result

    native.client.forward = output
    with pytest.raises(RuntimeError, match='cancelled'):
        native.advance_cpu('a', H3SourceInput(input_seq=0, packet=packet(0, 17), eos=False), cancelled)
    assert consumed == [13] and native.session.phase == 'cancelled'


def test_shm_close_ack_releases_session():
    native = bridge(Client([]))
    native.open('a', OPEN)
    native.session.pending_rgb_acks.append('pending-output')
    native.close('a')
    assert native.client.requests[-2].condition_inputs['h3'].ack_ids == ('pending-output',)
    assert isinstance(native.client.requests[-1], ReleaseRealtimeSessionReq)


def test_cancel_drops_output_and_session_needs_reopen():
    client = Client([13])
    native = bridge(client)
    native.open('a', OPEN)
    cancelled, emitted = threading.Event(), []

    def forward(req):
        cancelled.set()
        return Client.forward(client, req)

    client.forward = forward
    with pytest.raises(RuntimeError, match='cancelled'):
        native.append('a', packet(0, 17), cancelled, emitted.append)
    assert emitted == []
    assert len(client.requests) == 2  # open + compute; cancelled before codec.
    with pytest.raises(RuntimeError, match='cancelled; close and reopen'):
        native.append('a', packet(17, 17), threading.Event(), emitted.append)


def test_out_of_order_stamps_and_extra_frames_fail_the_session():
    native = bridge(Client([0, 30]))
    native.open('a', OPEN)
    native.append('a', packet(17, 17), threading.Event(), [].append)
    with pytest.raises(ValueError, match='overlaps'):
        native.append('a', packet(0, 17), threading.Event(), [].append)
    native = bridge(Client([30]))
    native.open('a', OPEN)
    with pytest.raises(RuntimeError, match='more frames'):
        native.append('a', packet(0, 17), threading.Event(), [].append)
    assert native.session.phase == 'failed'


def test_dead_worker_fails_a_blocked_call():
    reader, writer = multiprocessing.Pipe(duplex=False)
    worker = SimpleNamespace(sentinel=reader, pid=99, exitcode=-9)
    blocked = threading.Event()
    client = SimpleNamespace(forward=lambda req: blocked.wait())
    native = bridge(client, [worker])
    threading.Timer(0.2, writer.close).start()
    with pytest.raises(RuntimeError, match='worker 99 exited'):
        native.open('a', OPEN)
    blocked.set()


def test_realtime_trace_reaches_split_rgb_wire_packets():
    client = Client([0, 22])
    native = bridge(client, overlap=True)
    native.open('trace', OPEN)
    trace = dict(clock='native_worker_monotonic', stages=[
        dict(stage='render', dispatch_ns=1000, complete_ns=3000, wall_ms=.002)])
    forward = client.forward

    def output(req):
        result = forward(req)
        if (not isinstance(req, ReleaseRealtimeSessionReq)
                and type(req.condition_inputs['h3']).__name__ == H3RealtimeCodecRequest.__name__):
            assert req.condition_inputs['h3'].trace
            result.output = dict(_type='H3RealtimeOutput', rgb=result.output, orchestration=trace)
        return result

    client.forward = output
    emitted = []
    native.advance('trace', H3SourceInput(input_seq=0, packet=packet(0, 17), eos=False),
        threading.Event(), emitted.append)
    native.advance('trace', H3SourceInput(input_seq=1, packet=packet(17, 5), eos=True),
        threading.Event(), emitted.append)
    assert [len(part.frames) for part in emitted] == [8, 8, 6]
    assert all(part.wire_payload()['orchestration'] == trace for part in emitted)
    native.close('trace')
