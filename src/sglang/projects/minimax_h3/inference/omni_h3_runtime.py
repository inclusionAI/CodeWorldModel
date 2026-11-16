import atexit

from sglang_omni.models.minimax_h3.delivery import H3CPUFrameBatch
from sglang_omni.proto.session import ResourceUsage

from .omni_h3_assets import read_native_bundle
from .omni_h3_codec import NativeH3Codec
from .omni_h3_media import NativeH3MediaBridge


def check_deployment(manifest, limits, *, realtime=False):
    placement = manifest.get('codec_parallel') or {}
    encoder = tuple(placement.get('encoder_ranks', ()))
    decoder = tuple(placement.get('decoder_ranks', ()))
    if len(encoder) == 4 and not realtime:
        raise ValueError('four-rank encoder requires the native realtime backend')
    if getattr(limits, 'codec_overlap', False):
        supported = (((2, 3), (0, 1)), ((2, 3), (0, 1, 4, 5)))
        if realtime:
            supported += (((2, 3, 6, 7), (0, 1, 4, 5)),)
        if (manifest['server_args']['num_gpus'] != 8
                or (encoder, decoder) not in supported):
            raise ValueError('native codec overlap requires E2 ranks 2/3 and DTP2 or DTP4; realtime also supports E4 ranks 2/3/6/7 with DTP4 ranks 0/1/4/5')
    if tuple(limits.runtime_gpu_ids) != tuple(range(manifest['server_args']['num_gpus'])):
        raise ValueError('Omni must assign the native group contiguous local GPU IDs starting at zero')
    if not 0 < manifest['state_reservation_bytes'] <= limits.max_state_bytes:
        raise ValueError('native state reservation exceeds the Omni deployment budget')


def build_worker(context, *, model_path, limits):
    from sglang.multimodal_gen.runtime.loader.minimax_h3_causal import (
        bind_h3_codec_groups, load_h3_causal_components,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_group
    from ..modeling.inference_runtime import InferenceCodec

    # create_runtime already checked overlap placement and the state budget for this bundle and limits.
    manifest = read_native_bundle(model_path)
    if manifest['codec'].get('suffix_sdpa_backend') == 'fa3':
        raise ValueError('FA3 suffix attention requires the native realtime backend')
    if (manifest['codec'].get('compile_mode', 'default') != 'default'
            or manifest['codec'].get('tp_fused_post', False)):
        raise ValueError('decoder compile tuning requires the native realtime backend')
    encoder_group = decoder_group = None
    if manifest.get('codec_parallel') is not None:
        encoder_group, decoder_group = bind_h3_codec_groups(
            **manifest['codec_parallel'], output_rank=context.output_rank)
    components = load_h3_causal_components(context.server_args, encoder_group=encoder_group)
    if manifest.get('dit', {}).get('norm_quality', 'exact') == 'lossless':
        from sglang.kernels.ops.diffusion import mount_minimax_h3_norm_modulate
        if not mount_minimax_h3_norm_modulate(components.model):
            raise RuntimeError('requested lossless H3 norm modulation could not be mounted')
    local_codec = None
    if decoder_group is None or decoder_group.is_member:
        local_codec = InferenceCodec(None, manifest['codec'], device=components.device,
            load_encoder=False, decoder_tp_group=(
                None if decoder_group is None else decoder_group.coordinator))
        if manifest['codec'].get('suffix_sdpa_backend', 'default') == 'cudnn':
            from torch.nn.attention import SDPBackend, sdpa_kernel
            local_codec.decode = sdpa_kernel(SDPBackend.CUDNN_ATTENTION)(local_codec.decode)
    codec = NativeH3Codec(local_codec, get_world_group(), output_rank=context.output_rank,
        device=components.device, decoder_group=decoder_group)
    return NativeH3MediaBridge(components, codec, recipe=manifest['recipe'], limits=limits,
        state_reservation_bytes=manifest['state_reservation_bytes'],
        fuse_first_denoise=manifest.get('dit', {}).get('fuse_first_denoise', False))


class NativeH3GroupBridge:
    def __init__(self, host_factory, reservation, *, codec_overlap=False):
        self.host_factory, self.reservation, self.identity = host_factory, reservation, None
        self.codec_overlap = codec_overlap
        self.host = host_factory()

    def open(self, identity, request):
        if self.identity is not None:
            raise ValueError('native H3 group already owns a session')
        if self.host is None:
            self.host = self.host_factory()
        try:
            self.host.open(identity, request)
            self.identity = identity
        except BaseException:
            self.shutdown()
            raise

    def _check_cancel(self, identity, cancelled):
        if cancelled.is_set():
            self.host.cancel(identity)
            raise RuntimeError('H3 append cancelled at the completed rank-group boundary')

    def _emit(self, identity, cancelled, packets, emit):
        self._check_cancel(identity, cancelled)
        for packet in packets:
            emit(packet)

    def append(self, identity, packet, cancelled, emit):
        if identity != self.identity:
            raise ValueError('unknown or stale H3 session identity')
        try:
            self._emit(identity, cancelled, (), emit)
            self._emit(identity, cancelled, self.host.append(identity, packet), emit)
        except BaseException:
            self.shutdown()
            raise

    def finish(self, identity, cancelled, emit):
        if identity != self.identity:
            raise ValueError('unknown or stale H3 session identity')
        try:
            self._emit(identity, cancelled, (), emit)
            self._emit(identity, cancelled, self.host.finish(identity), emit)
        except BaseException:
            self.shutdown()
            raise

    def advance(self, identity, source, cancelled, emit, peek_next=None):
        if identity != self.identity:
            raise ValueError('unknown or stale H3 session identity')
        if not self.codec_overlap:
            raise ValueError('native H3 codec overlap is not enabled')
        try:
            self._emit(identity, cancelled, (), emit)
            self.host.compute(identity, source)
            next_source = None if peek_next is None or source.eos else peek_next()
            self._emit(identity, cancelled, self.host.deliver(identity, next_source), emit)
        except BaseException:
            self.shutdown()
            raise

    def advance_cpu(self, identity, source, cancelled, peek_next=None) -> H3CPUFrameBatch:
        if identity != self.identity:
            raise ValueError('unknown or stale H3 session identity')
        if not self.codec_overlap:
            raise ValueError('native H3 codec overlap is not enabled')
        try:
            self._check_cancel(identity, cancelled)
            self.host.compute(identity, source)
            next_source = None if peek_next is None or source.eos else peek_next()
            batch = self.host.deliver_cpu(identity, next_source)
            self._check_cancel(identity, cancelled)
            return batch
        except BaseException:
            self.shutdown()
            raise

    def cancel(self, identity):
        if self.identity == identity:
            self.host.cancel(identity)

    def close(self, identity):
        if self.identity == identity:
            try:
                self.host.close(identity)
            except BaseException:
                self.shutdown()
                raise
            finally:
                self.identity = None

    def shutdown(self):
        if self.host is not None:
            self.host.shutdown()
            self.host = None
        self.identity = None

    def usage(self, identity):
        return ResourceUsage(bytes=self.reservation if self.identity == identity else 0)


def create_runtime(*, model_path, limits):
    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import MiniMaxH3PipelineConfig
    from sglang.multimodal_gen.runtime.managers.minimax_h3_causal_host import H3CausalHost
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    manifest = read_native_bundle(model_path)
    check_deployment(manifest, limits)
    settings = manifest['server_args']
    world_size = settings['num_gpus']
    reservation = manifest['state_reservation_bytes']
    args = ServerArgs(**settings, pipeline_config=MiniMaxH3PipelineConfig())
    def make_host():
        return H3CausalHost(worker_factory=f'{__name__}:build_worker',
            factory_kwargs=dict(model_path=model_path, limits=limits), server_args=args,
            world_size=world_size, timeout=600)
    bridge = NativeH3GroupBridge(make_host, reservation,
        codec_overlap=getattr(limits, 'codec_overlap', False))
    atexit.register(bridge.shutdown)
    return bridge
