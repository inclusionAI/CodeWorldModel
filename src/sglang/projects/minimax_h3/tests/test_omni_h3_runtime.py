import threading
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference.omni_h3_runtime import NativeH3GroupBridge, check_deployment


class HostFixture:
    def __init__(self):
        self.stopped = False
        self.identity = None

    def open(self, identity, request):
        self.identity = identity

    def append(self, identity, packet):
        return (packet,)

    def cancel(self, identity):
        pass

    def close(self, identity):
        self.identity = None

    def shutdown(self):
        self.stopped = True


def test_cancelled_output_is_not_published_and_reopen_recreates_failed_group():
    hosts = []
    def factory():
        hosts.append(HostFixture())
        return hosts[-1]
    bridge = NativeH3GroupBridge(factory, 128)
    bridge.open('first', object())
    cancelled = threading.Event()
    cancelled.set()
    outputs = []
    with pytest.raises(RuntimeError, match='cancelled'):
        bridge.append('first', object(), cancelled, outputs.append)
    assert not outputs and hosts[0].stopped
    bridge.close('first')
    bridge.open('second', object())
    assert len(hosts) == 2 and bridge.usage('second').bytes == 128
    bridge.close('second')
    assert bridge.usage('second').bytes == 0
    bridge.shutdown()


def test_receiver_failure_discards_whole_group():
    host = HostFixture()
    bridge = NativeH3GroupBridge(lambda: host, 128)
    bridge.open('session', object())
    def fail(packet):
        raise OSError('closed receiver')
    with pytest.raises(OSError, match='closed receiver'):
        bridge.append('session', object(), threading.Event(), fail)
    assert host.stopped and bridge.usage('session').bytes == 0


def test_staged_driver_observes_late_arrival_only_after_compute_then_delivers_synchronously():
    host, trace, outputs = HostFixture(), [], []
    current = SimpleNamespace(input_seq=0, packet=object(), eos=False)
    next_source = SimpleNamespace(input_seq=1, packet=object(), eos=False)
    ready = []
    def compute(identity, source):
        assert identity == 'session' and source is current
        trace.append('compute')
        ready.append(next_source)
    def peek():
        trace.append('peek')
        return ready[0]
    def deliver(identity, source):
        assert identity == 'session' and source is next_source
        trace.append('deliver')
        return ('rgb',)
    host.compute, host.deliver = compute, deliver
    bridge = NativeH3GroupBridge(lambda: host, 128, codec_overlap=True)
    bridge.open('session', object())
    bridge.advance('session', current, threading.Event(), outputs.append, peek)
    assert trace == ['compute', 'peek', 'deliver'] and outputs == ['rgb']
    assert bridge.codec_overlap


def test_staged_driver_discards_group_when_lookahead_or_delivery_fails():
    host = HostFixture()
    host.compute = lambda *args: None
    bridge = NativeH3GroupBridge(lambda: host, 128, codec_overlap=True)
    bridge.open('session', object())
    def invalid_next():
        raise ValueError('invalid queued source')
    with pytest.raises(ValueError, match='invalid queued'):
        bridge.advance('session', SimpleNamespace(eos=False), threading.Event(),
            lambda packet: None, invalid_next)
    assert host.stopped and bridge.identity is None


@pytest.mark.parametrize('eos', [False, True])
def test_cpu_driver_returns_completed_batch_and_keeps_lookahead_identity(eos):
    host, trace = HostFixture(), []
    source = SimpleNamespace(eos=eos)
    next_source = object()
    batch = ({'rgb': object(), 'frames': ()},)
    host.compute = lambda identity, current: trace.append(('compute', identity, current))
    def peek():
        trace.append(('peek',))
        return next_source
    def deliver(identity, queued):
        trace.append(('deliver_cpu', identity, queued))
        return batch
    host.deliver_cpu = deliver
    bridge = NativeH3GroupBridge(lambda: host, 128, codec_overlap=True)
    bridge.open('session', object())
    assert bridge.advance_cpu('session', source, threading.Event(), peek) is batch
    assert trace == ([('compute', 'session', source)]
        + ([] if eos else [('peek',)])
        + [('deliver_cpu', 'session', None if eos else next_source)])
    assert not host.stopped


@pytest.mark.parametrize('cancel_at', ['before_compute', 'after_delivery'])
def test_cpu_driver_cancellation_discards_whole_group(cancel_at):
    host, event, calls = HostFixture(), threading.Event(), []
    host.compute = lambda *args: calls.append('compute')
    def deliver(*args):
        calls.append('deliver_cpu')
        event.set()
        return ()
    host.deliver_cpu = deliver
    bridge = NativeH3GroupBridge(lambda: host, 128, codec_overlap=True)
    bridge.open('session', object())
    if cancel_at == 'before_compute':
        event.set()
    with pytest.raises(RuntimeError, match='cancelled'):
        bridge.advance_cpu('session', SimpleNamespace(eos=False), event)
    assert calls == ([] if cancel_at == 'before_compute' else ['compute', 'deliver_cpu'])
    assert host.stopped and bridge.identity is None


def test_cpu_driver_transport_failure_recreates_group_on_reopen():
    hosts = []
    def factory():
        host = HostFixture()
        host.compute = lambda *args: None
        def fail(*args):
            raise OSError('descriptor transfer failed')
        host.deliver_cpu = fail
        hosts.append(host)
        return host
    bridge = NativeH3GroupBridge(factory, 128, codec_overlap=True)
    bridge.open('first', object())
    with pytest.raises(OSError, match='descriptor transfer'):
        bridge.advance_cpu('first', SimpleNamespace(eos=False), threading.Event())
    assert hosts[0].stopped
    bridge.open('second', object())
    assert len(hosts) == 2 and bridge.usage('second').bytes == 128
    bridge.shutdown()


@pytest.mark.parametrize('decoder', [[0, 1], [0, 1, 4, 5]])
@pytest.mark.parametrize('encoder', [None, [0, 1], [2, 3], [4, 5]])
def test_overlap_opt_in_requires_the_disjoint_e2_decoder_placement(encoder, decoder):
    manifest = {'server_args': {'num_gpus': 8}, 'state_reservation_bytes': 128}
    if encoder is not None:
        manifest['codec_parallel'] = {'encoder_ranks': encoder, 'decoder_ranks': decoder}

    def limits(overlap):
        return SimpleNamespace(codec_overlap=overlap, runtime_gpu_ids=tuple(range(8)),
                               max_state_bytes=128)

    check_deployment(manifest, limits(False))
    if encoder == [2, 3]:
        check_deployment(manifest, limits(True))
    else:
        with pytest.raises(ValueError, match='requires E2 ranks 2/3'):
            check_deployment(manifest, limits(True))


@pytest.mark.parametrize('overlap', [False, True])
def test_four_encoder_ranks_are_realtime_only(overlap):
    manifest = {'server_args': {'num_gpus': 8}, 'state_reservation_bytes': 128,
        'codec_parallel': {'encoder_ranks': [2, 3, 6, 7], 'decoder_ranks': [0, 1, 4, 5]}}
    limits = SimpleNamespace(codec_overlap=overlap, runtime_gpu_ids=tuple(range(8)), max_state_bytes=128)
    check_deployment(manifest, limits, realtime=True)
    with pytest.raises(ValueError, match='realtime'):
        check_deployment(manifest, limits)


@pytest.mark.parametrize('encoder_ranks', [None, [0, 1], [2, 3]])
@pytest.mark.parametrize('fusion', [False, True])
@pytest.mark.parametrize('suffix_sdpa_backend', [None, 'default', 'cudnn'])
@pytest.mark.parametrize('norm_quality, marked', [(None, True), ('exact', True),
    ('lossless', True), ('lossless', False)])
def test_production_factory_selects_norm_before_decoder_and_preserves_component_members(
        tmp_path, monkeypatch, encoder_ranks, fusion, suffix_sdpa_backend, norm_quality, marked):
    import torch
    from sglang.kernels.ops import diffusion
    from sglang.multimodal_gen.runtime.loader import minimax_h3_causal as loader
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root.parent))
    runtime = importlib.import_module(f'{root.name}.inference.omni_h3_runtime')
    model_runtime = importlib.import_module(f'{root.name}.modeling.inference_runtime')
    parallel = encoder_ranks is not None
    manifest = dict(schema='h3_omni_native/v1', state_reservation_bytes=128,
        codec=dict(encoder='h3', decoder='causal', backend='fast', compile=False, cuda_graph=False),
        server_args={'num_gpus': 8},
        recipe={'bootstrap_latents': 9, 'current_latent_schedule': [5]})
    if suffix_sdpa_backend is not None:
        manifest['codec']['suffix_sdpa_backend'] = suffix_sdpa_backend
    if fusion:
        manifest['dit'] = {'fuse_first_denoise': True}
    if norm_quality is not None:
        manifest.setdefault('dit', {})['norm_quality'] = norm_quality
    if parallel:
        manifest['codec_parallel'] = dict(encoder_ranks=encoder_ranks, decoder_ranks=[0, 1])
    (tmp_path / 'native-runtime.json').write_text(json.dumps(manifest))
    mount_calls, models = [], []
    official_mount = diffusion.mount_minimax_h3_norm_modulate
    def mount(model):
        assert model is models[-1]
        mount_calls.append(model)
        return official_mount(model)
    monkeypatch.setattr(diffusion, 'mount_minimax_h3_norm_modulate', mount)
    def backend_flags():
        return tuple(getattr(torch.backends.cuda, f'{name}_sdp_enabled')()
            for name in ('flash', 'mem_efficient', 'math', 'cudnn'))
    initial_flags = backend_flags()
    decoder_flags = (False, False, False, True) if suffix_sdpa_backend == 'cudnn' else initial_flags
    def decode(latent, *, eos=False):
        assert backend_flags() == decoder_flags
        if eos:
            raise RuntimeError('decoder failure')
        return None
    loaded = []
    def codec(*args, **kwargs):
        assert backend_flags() == initial_flags
        assert diffusion.minimax_h3_norm_modulate_active(models[-1]) == (norm_quality == 'lossless')
        loaded.append(kwargs)
        return SimpleNamespace(decode=decode)
    monkeypatch.setattr(model_runtime, 'InferenceCodec', codec)
    for rank in range(8):
        coordinator = SimpleNamespace(rank=rank, ranks=[rank // 2 * 2, rank // 2 * 2 + 1])
        active_encoder_group = loader.H3CodecGroup(coordinator, tuple(encoder_ranks or [0, 1]))
        decoder_group = loader.H3CodecGroup(coordinator)
        def bind(*, output_rank, **members):
            assert parallel and output_rank == 0
            assert members == manifest['codec_parallel']
            return active_encoder_group, decoder_group
        monkeypatch.setattr(loader, 'bind_h3_codec_groups', bind)
        def dit():
            assert backend_flags() == initial_flags
        model = torch.nn.Module()
        model.forward = dit
        if marked:
            diffusion.mark_minimax_h3_norm_modulate_site(model)
        assert not diffusion.minimax_h3_norm_modulate_active(model)
        models.append(model)
        def components(args, *, encoder_group):
            assert backend_flags() == initial_flags
            assert encoder_group is (active_encoder_group if parallel else None)
            return SimpleNamespace(device='cpu', model=model)
        monkeypatch.setattr(loader, 'load_h3_causal_components', components)
        def broadcast(value, *, src):
            assert backend_flags() == initial_flags
            return value
        monkeypatch.setattr(parallel_state, 'get_world_group', lambda: SimpleNamespace(
            rank=rank, ranks=list(range(8)), broadcast=broadcast))
        context = SimpleNamespace(server_args=None, output_rank=0)
        limits = SimpleNamespace(max_state_bytes=128)
        if norm_quality == 'lossless' and not marked:
            with pytest.raises(RuntimeError, match='lossless H3 norm.*could not be mounted'):
                runtime.build_worker(context, model_path=tmp_path, limits=limits)
            assert not loaded
            continue
        bridge = runtime.build_worker(context, model_path=tmp_path, limits=limits)
        assert diffusion.minimax_h3_norm_modulate_active(bridge.components.model) == (norm_quality == 'lossless')
        assert (bridge.codec.local_codec is not None) == (not parallel or rank < 2)
        assert bridge.fuse_first_denoise is fusion
        assert backend_flags() == initial_flags
        bridge.components.model()
        assert bridge.codec.decode('latent').frame_count == 0
        assert backend_flags() == initial_flags
        bridge.components.model()
        if bridge.codec.local_codec is not None:
            with pytest.raises(RuntimeError, match='decoder failure'):
                bridge.codec.decode('latent', eos=True)
            assert backend_flags() == initial_flags
            bridge.components.model()
            assert bridge.codec.decode('latent').frame_count == 0
    assert mount_calls == (models if norm_quality == 'lossless' else [])
    assert len(loaded) == (0 if not marked else (2 if parallel else 8))
    assert all(call['load_encoder'] is False for call in loaded)
    assert all((call['decoder_tp_group'] is not None) == parallel for call in loaded)
