from dataclasses import dataclass, replace
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from inference.omni_h3_media import NativeH3MediaBridge
from inference.omni_h3_codec import NativeDecodeBatch
from sglang_omni.models.minimax_h3.session import (
    H3FrameStamp, H3OpenRequest, H3RGBPacket, H3RuntimeLimits, H3SourceInput,
)
from sglang_omni.proto.session import SessionIdentity


def packet(start, count):
    return H3RGBPacket(width=2, height=2, rgb=bytes(12 * count), frames=tuple(
        H3FrameStamp(source_frame_id=f'frame-{i}', pts_ms=float(i * 40), duration_ms=40.0)
        for i in range(start, start + count)))


@dataclass
class Static:
    prompt_embeds: object
    token_tags: object
    first_frame: object
    first_frame_noise: object


@dataclass
class Source:
    references: object
    reference_noise: object
    target_noise: object
    clean_noise: object


class FixtureSession:
    def __init__(self, model, static, recipe, *, fuse_first_denoise=False):
        self.static = static
        self.fuse_first_denoise = fuse_first_denoise
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


class FixtureComponents:
    device = 'cpu'
    model = object()

    def condition(self, prompt, image):
        return torch.zeros(2, 4), torch.ones(2, dtype=torch.long)

    def encode_image(self, image):
        return torch.zeros(1, 1, 2, 2)

    def encode_proxy(self, frames, *, eos):
        return torch.zeros(1, 2 if eos else 5, 2, 2)

    def synchronize(self):
        pass


class FixtureCodec:
    def begin_session(self):
        self.calls = []

    def decode(self, frames, *, eos):
        self.calls.append(eos)
        return NativeDecodeBatch(frames, torch.zeros(frames, 2, 2, 3, dtype=torch.uint8))


@pytest.fixture
def bridge():
    with patch('inference.omni_h3_media._native_types',
               return_value=(FixtureSession, Source, Static)):
        value = NativeH3MediaBridge(FixtureComponents(), FixtureCodec(),
            recipe={'bootstrap_latents': 9, 'current_latent_schedule': [5]},
            limits=H3RuntimeLimits(max_width=2, max_height=2, max_input_frames=17,
                max_output_frames=17, max_state_bytes=1 << 20),
            state_reservation_bytes=1 << 20)
        yield value


def opened(bridge, index=1):
    identity = SessionIdentity('sample', index)
    bridge.open(identity, H3OpenRequest(prompt='road', seed=73, first_image=packet(0, 1)))
    return identity


@pytest.mark.parametrize('fusion', [False, True])
def test_media_open_passes_dit_execution_choice_to_native_session(bridge, fusion):
    bridge.fuse_first_denoise = fusion
    opened(bridge)
    assert bridge.state.latent.fuse_first_denoise is fusion


def test_seed_namespace_collision_does_not_break_open_or_append(bridge, monkeypatch, tmp_path):
    import sys

    for name in tuple(sys.modules):
        if name == 'common' or name.startswith('common.'):
            monkeypatch.delitem(sys.modules, name)
    (tmp_path / 'common.py').write_text('from . import collision_fixture_dependency\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    identity = opened(bridge)
    bridge.append(identity, packet(0, 17), threading.Event(), lambda packet: None)
    assert len(bridge.state.latent.sources) == 1
    assert torch.isfinite(bridge.state.latent.sources[0].target_noise).all()


def test_bootstrap_delayed_frames_keep_their_sources_and_eos(bridge):
    identity, event, outputs = opened(bridge), threading.Event(), []
    bridge.append(identity, packet(0, 17), event, outputs.append)
    assert outputs == []
    bridge.append(identity, packet(17, 17), event, outputs.append)
    assert [len(p.frames) for p in outputs] == [17, 15]
    bridge.append(identity, packet(34, 5), event, outputs.append)
    bridge.finish(identity, event, outputs.append)
    assert [frame.source_frame_id for p in outputs for frame in p.frames] == [
        f'frame-{i}' for i in range(39)]
    assert bridge.codec.calls == [False, True]
    assert not bridge.state.frames
    with pytest.raises(RuntimeError, match='ended'):
        bridge.finish(identity, event, outputs.append)


def test_close_reopen_reproduces_noise_and_rejects_stale_identity(bridge):
    identity, event = opened(bridge), threading.Event()
    initial = bridge.state.latent.static.first_frame_noise.clone()
    bridge.append(identity, packet(0, 17), event, lambda packet: None)
    expected = bridge.state.latent.sources[0].target_noise.clone()
    bridge.close(identity)
    assert bridge.usage(identity).bytes == 0
    fresh = opened(bridge, 2)
    with pytest.raises(ValueError, match='stale'):
        bridge.append(identity, packet(0, 17), event, lambda packet: None)
    bridge.append(fresh, packet(0, 17), event, lambda packet: None)
    assert torch.equal(initial, bridge.state.latent.static.first_frame_noise)
    assert torch.equal(expected, bridge.state.latent.sources[0].target_noise)


def test_cancel_after_native_forward_never_publishes_or_reuses_history(bridge):
    identity, event, outputs = opened(bridge), threading.Event(), []
    bridge.append(identity, packet(0, 17), event, outputs.append)
    advance = bridge.state.latent.append
    def cancel_after_forward(source):
        result = advance(source)
        event.set()
        return result
    bridge.state.latent.append = cancel_after_forward
    with pytest.raises(RuntimeError, match='cancelled'):
        bridge.append(identity, packet(17, 17), event, outputs.append)
    assert outputs == []
    event.clear()
    with pytest.raises(RuntimeError, match='cancelled'):
        bridge.append(identity, packet(34, 17), event, outputs.append)
    bridge.close(identity)
    opened(bridge, 2)


def test_invalid_tail_and_source_order_do_not_mutate_accepted_frames(bridge):
    identity, event = opened(bridge), threading.Event()
    with pytest.raises(ValueError, match='17-frame'):
        bridge.append(identity, packet(0, 6), event, lambda packet: None)
    assert not bridge.state.frames
    bridge.append(identity, packet(0, 17), event, lambda packet: None)
    with pytest.raises(ValueError, match='overlaps'):
        bridge.append(identity, packet(16, 17), event, lambda packet: None)
    assert len(bridge.state.frames) == 17
    with pytest.raises(ValueError, match='final 5'):
        bridge.finish(identity, event, lambda packet: None)
    bridge.append(identity, packet(17, 5), event, lambda packet: None)
    with pytest.raises(ValueError, match='final 5-frame tail'):
        bridge.append(identity, packet(22, 17), event, lambda packet: None)


def test_delivery_failure_poisons_the_session_until_close(bridge):
    identity, event = opened(bridge), threading.Event()
    bridge.append(identity, packet(0, 17), event, lambda packet: None)
    def fail(packet):
        raise OSError('receiver disconnected')
    with pytest.raises(OSError, match='disconnected'):
        bridge.append(identity, packet(17, 17), event, fail)
    assert bridge.state.phase == 'failed'
    bridge.close(identity)
    bridge.close(identity)
    opened(bridge, 2)


def test_nonoutput_rank_retires_frame_metadata_through_eos_and_reopen(bridge):
    bridge.codec.decode = lambda count, eos: NativeDecodeBatch(count, None)
    for index in (1, 2):
        identity, event, outputs = opened(bridge, index), threading.Event(), []
        bridge.append(identity, packet(0, 17), event, outputs.append)
        bridge.append(identity, packet(17, 17), event, outputs.append)
        assert len(bridge.state.frames) == 2
        bridge.append(identity, packet(34, 5), event, outputs.append)
        bridge.finish(identity, event, outputs.append)
        assert not bridge.state.frames and outputs == []
        bridge.close(identity)


def test_nonoutput_rank_rejects_frame_count_beyond_source(bridge):
    identity, event = opened(bridge), threading.Event()
    bridge.codec.decode = lambda count, eos: NativeDecodeBatch(35, None)
    bridge.append(identity, packet(0, 17), event, lambda packet: None)
    with pytest.raises(RuntimeError, match='more frames'):
        bridge.append(identity, packet(17, 17), event, lambda packet: None)
    assert bridge.state.phase == 'failed'


def overlap_fixture(bridge):
    trace = []
    bridge.codec_overlap = True
    encode, decode = bridge.components.encode_proxy, bridge.codec.decode
    def encode_local(frames, *, eos):
        trace.append(('encode_local', len(frames), eos))
        return encode(frames, eos=eos)
    def broadcast(local, *, rgb_shape, eos):
        trace.append(('publish_proxy', rgb_shape[0], eos))
        return local
    def decode_local(video, *, eos):
        trace.append(('decode_local', video, eos))
        return decode(video, eos=eos)
    def counts(local, *, span_count):
        assert span_count == len(local)
        trace.append(('publish_counts', tuple(local)))
        return tuple(local)
    bridge.components.encode_proxy_local = encode_local
    bridge.components.broadcast_proxy = broadcast
    bridge.components.synchronize = lambda: trace.append(('join',))
    bridge.codec.decode_local = decode_local
    bridge.codec.publish_counts = counts
    return trace


@pytest.mark.parametrize('explicit_eos', [False, True])
def test_staged_prefetch_preserves_ledger_noise_and_complete_rgb(bridge, explicit_eos):
    identity, event, serial = opened(bridge), threading.Event(), []
    bridge.append(identity, packet(0, 17), event, serial.append)
    bridge.append(identity, packet(17, 17), event, serial.append)
    bridge.append(identity, packet(34, 5), event, serial.append)
    bridge.finish(identity, event, serial.append)
    noises = [(s.reference_noise[0].clone(), s.target_noise.clone(), s.clean_noise.clone())
              for s in bridge.state.latent.sources]
    bridge.close(identity)
    identity, outputs = opened(bridge, 2), []
    trace = overlap_fixture(bridge)
    first = H3SourceInput(input_seq=0, packet=packet(0, 17), eos=False)
    second = H3SourceInput(input_seq=1, packet=packet(17, 17), eos=False)
    tail = H3SourceInput(input_seq=2, packet=packet(34, 5), eos=explicit_eos)
    bridge.compute(identity, first, event)
    trace.clear()
    bridge.deliver(identity, second, event, outputs.append)
    assert len(bridge.state.frames) == 17
    assert bridge.state.source_clip == 1 and len(bridge.state.latent.sources) == 1
    assert bridge.state.prepared.source == second
    assert trace == [('encode_local', 17, False), ('join',), ('publish_counts', ()),
                     ('publish_proxy', 17, False), ('join',)]
    bridge.compute(identity, second, event)
    assert bridge.state.prepared is None and bridge.state.source_clip == 2
    bridge.deliver(identity, tail, event, outputs.append)
    bridge.compute(identity, tail, event)
    bridge.deliver(identity, None, event, outputs.append)
    if not explicit_eos:
        assert bridge.state.tail is not None and bridge.state.source_clip == 2
        assert not any(row == ('encode_local', 5, False) for row in trace)
        bridge.compute(identity, H3SourceInput(input_seq=3, packet=None, eos=True), event)
        bridge.deliver(identity, None, event, outputs.append)
    assert outputs == serial
    assert not bridge.state.frames and bridge.state.phase == 'ended'
    assert bridge.state.source_clip == 3
    for source, expected in zip(bridge.state.latent.sources, noises):
        assert all(torch.equal(actual, previous) for actual, previous in zip(
            (source.reference_noise[0], source.target_noise, source.clean_noise), expected))


@pytest.mark.parametrize('changed', ['sequence', 'packet', 'eos'])
def test_prepared_proxy_requires_the_exact_next_source_and_is_cleared_on_reopen(bridge, changed):
    identity, event = opened(bridge), threading.Event()
    overlap_fixture(bridge)
    source = H3SourceInput(input_seq=1, packet=packet(17, 17), eos=False)
    bridge.compute(identity, H3SourceInput(input_seq=0, packet=packet(0, 17), eos=False), event)
    bridge.deliver(identity, source, event, lambda packet: None)
    altered = H3SourceInput(input_seq=2 if changed == 'sequence' else 1,
        packet=packet(18 if changed == 'packet' else 17, 17), eos=changed == 'eos')
    with pytest.raises(ValueError, match='match|final 5'):
        bridge.compute(identity, altered, event)
    assert len(bridge.state.frames) == 17
    bridge.close(identity)
    opened(bridge, 2)
    assert bridge.state.prepared is None and bridge.state.computed is None


def test_multiple_spans_and_zero_decode_publish_after_local_join(bridge):
    identity, event, outputs = opened(bridge), threading.Event(), []
    trace = overlap_fixture(bridge)
    bridge.state.latent.append = lambda source: tuple(
        SimpleNamespace(video=count, plan=SimpleNamespace(eos=False)) for count in (0, 6, 11))
    bridge.compute(identity, H3SourceInput(input_seq=0, packet=packet(0, 17), eos=False), event)
    with pytest.raises(RuntimeError, match='awaiting delivery'):
        bridge.compute(identity, H3SourceInput(input_seq=1, packet=packet(17, 17), eos=False), event)
    trace.clear()
    bridge.deliver(identity, H3SourceInput(input_seq=1, packet=packet(17, 17), eos=False), event, outputs.append)
    assert trace == [('encode_local', 17, False), ('decode_local', 0, False),
        ('decode_local', 6, False), ('decode_local', 11, False), ('join',),
        ('publish_counts', (0, 6, 11)), ('publish_proxy', 17, False), ('join',)]
    assert [len(value.frames) for value in outputs] == [6, 11]
    assert not bridge.state.frames


def test_cancel_during_codec_finishes_publication_and_never_emits(bridge):
    identity, event, outputs = opened(bridge), threading.Event(), []
    trace = overlap_fixture(bridge)
    bridge.compute(identity, H3SourceInput(input_seq=0, packet=packet(0, 17), eos=False), event)
    local_encode = bridge.components.encode_proxy_local
    def cancel_in_codec(frames, *, eos):
        event.set()
        return local_encode(frames, eos=eos)
    bridge.components.encode_proxy_local = cancel_in_codec
    with pytest.raises(RuntimeError, match='cancelled'):
        bridge.deliver(identity, H3SourceInput(input_seq=1, packet=packet(17, 17), eos=False), event, outputs.append)
    assert ('publish_counts', ()) in trace and ('publish_proxy', 17, False) in trace
    assert not outputs and bridge.state.phase == 'cancelled'


def test_no_next_completes_immediately_and_rejects_duplicate_delivery(bridge):
    identity, event = opened(bridge), threading.Event()
    trace = overlap_fixture(bridge)
    source = H3SourceInput(input_seq=0, packet=packet(0, 17), eos=False)
    bridge.compute(identity, source, event)
    trace.clear()
    bridge.deliver(identity, None, event, lambda packet: None)
    assert trace == [('join',), ('publish_counts', ()), ('join',)]
    assert bridge.state.prepared is None
    with pytest.raises(RuntimeError, match='requires a computed'):
        bridge.deliver(identity, None, event, lambda packet: None)
    # Omni enforces contiguous input_seq; a replayed packet still fails the source timeline.
    with pytest.raises(ValueError, match='overlaps'):
        bridge.compute(identity, source, event)


@pytest.mark.parametrize('explicit_eos', [False, True])
@pytest.mark.parametrize('overlap', [False, True])
def test_cpu_delivery_matches_bytes_and_owns_retired_frames(bridge, explicit_eos, overlap):
    bridge.limits = replace(bridge.limits, max_output_frames=40)
    event, expected, retained = threading.Event(), [], []
    sources = [
        H3SourceInput(input_seq=0, packet=packet(0, 17), eos=False),
        H3SourceInput(input_seq=1, packet=packet(17, 17), eos=False),
        H3SourceInput(input_seq=2, packet=packet(34, 5), eos=explicit_eos),
    ]
    if not explicit_eos:
        sources.append(H3SourceInput(input_seq=3, packet=None, eos=True))
    for index, cpu_output in enumerate((False, True), 1):
        identity = opened(bridge, index)
        if overlap:
            overlap_fixture(bridge)
        for source_index, source in enumerate(sources):
            bridge.compute(identity, source, event)
            next_source = (sources[source_index + 1]
                if overlap and source_index + 1 < len(sources) else None)
            if cpu_output:
                batch = bridge.deliver_cpu(identity, next_source, event)
                assert isinstance(batch, tuple)
                for value in batch:
                    rgb = value['rgb']
                    assert rgb.device.type == 'cpu' and rgb.dtype == torch.uint8
                    assert rgb.is_contiguous() and rgb.is_shared()
                    assert isinstance(value['frames'], tuple)
                    assert len(value['frames']) == len(rgb)
                retained.extend(batch)
                assert bridge.state.computed is None
            else:
                bridge.deliver(identity, next_source, event, expected.append)
        assert not bridge.state.frames and bridge.state.phase == 'ended'
        bridge.close(identity)
    actual = [H3RGBPacket(width=value['rgb'].shape[2], height=value['rgb'].shape[1],
        frames=value['frames'], rgb=value['rgb'].numpy().tobytes()) for value in retained]
    assert actual == expected
    assert [stamp.source_frame_id for value in retained for stamp in value['frames']] == [
        f'frame-{index}' for index in range(39)]


def test_cpu_delivery_copies_noncontiguous_decoder_storage_and_all_spans(bridge):
    identity, event = opened(bridge), threading.Event()
    trace = overlap_fixture(bridge)
    bridge.state.latent.append = lambda source: tuple(
        SimpleNamespace(video=count, plan=SimpleNamespace(eos=False)) for count in (0, 6, 11))
    decoded = []
    def decode(count, *, eos):
        rgb = torch.arange(count * 12, dtype=torch.uint8).reshape(count, 2, 2, 3).transpose(1, 2)
        decoded.append(rgb)
        return NativeDecodeBatch(count, rgb)
    bridge.codec.decode_local = decode
    bridge.compute(identity, H3SourceInput(input_seq=0, packet=packet(0, 17), eos=False), event)
    trace.clear()
    batch = bridge.deliver_cpu(identity, None, event)
    assert [len(value['frames']) for value in batch] == [6, 11]
    assert trace == [('join',), ('publish_counts', (0, 6, 11)), ('join',)]
    assert not bridge.state.frames
    expected = [value.clone() for value in decoded[1:]]
    with torch.inference_mode():
        for value in decoded:
            value.zero_()
    assert all(torch.equal(value['rgb'], original) for value, original in zip(batch, expected))
    assert all(value['rgb'].is_contiguous() and value['rgb'].is_shared() for value in batch)


def test_cpu_delivery_checks_total_frame_budget_before_retirement(bridge):
    bridge.limits = replace(bridge.limits, max_output_frames=10)
    identity, event = opened(bridge), threading.Event()
    overlap_fixture(bridge)
    bridge.state.latent.append = lambda source: tuple(
        SimpleNamespace(video=6, plan=SimpleNamespace(eos=False)) for _ in range(2))
    bridge.compute(identity, H3SourceInput(input_seq=0, packet=packet(0, 17), eos=False), event)
    with pytest.raises(ValueError, match='frame budget'):
        bridge.deliver_cpu(identity, None, event)
    assert len(bridge.state.frames) == 17 and bridge.state.phase == 'failed'


@pytest.mark.parametrize('invalid', ['negative', 'dtype', 'shape', 'geometry', 'count'])
def test_cpu_delivery_rejects_invalid_decoder_output(bridge, invalid):
    identity, event = opened(bridge), threading.Event()
    overlap_fixture(bridge)
    bridge.state.latent.append = lambda source: (
        SimpleNamespace(video=6, plan=SimpleNamespace(eos=False)),)
    rgb = torch.zeros(6, 2, 2, 3, dtype=torch.uint8)
    count = 6
    if invalid == 'negative':
        count, rgb = -1, None
    elif invalid == 'dtype':
        rgb = rgb.float()
    elif invalid == 'shape':
        rgb = rgb[..., :2]
    elif invalid == 'geometry':
        rgb = torch.zeros(6, 3, 2, 3, dtype=torch.uint8)
    elif invalid == 'count':
        count = 5
    bridge.codec.decode_local = lambda *args, **kwargs: NativeDecodeBatch(count, rgb)
    bridge.compute(identity, H3SourceInput(input_seq=0, packet=packet(0, 17), eos=False), event)
    with pytest.raises((ValueError, RuntimeError)):
        bridge.deliver_cpu(identity, None, event)
    assert len(bridge.state.frames) == 17 and bridge.state.phase == 'failed'


def test_cancel_during_cpu_delivery_finishes_joins_without_publishing(bridge):
    identity, event = opened(bridge), threading.Event()
    trace = overlap_fixture(bridge)
    bridge.state.latent.append = lambda source: (
        SimpleNamespace(video=17, plan=SimpleNamespace(eos=False)),)
    bridge.compute(identity, H3SourceInput(input_seq=0, packet=packet(0, 17), eos=False), event)
    decode = bridge.codec.decode_local
    def cancelling_decode(*args, **kwargs):
        result = decode(*args, **kwargs)
        event.set()
        return result
    bridge.codec.decode_local = cancelling_decode
    trace.clear()
    with pytest.raises(RuntimeError, match='cancelled'):
        bridge.deliver_cpu(identity,
            H3SourceInput(input_seq=1, packet=packet(17, 17), eos=False), event)
    assert trace[-3:] == [('publish_counts', (17,)), ('publish_proxy', 17, False), ('join',)]
    assert len(bridge.state.frames) == 17 and bridge.state.phase == 'cancelled'
