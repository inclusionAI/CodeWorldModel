"""One resident worker group, one session, and ordered RGB delivery boundaries."""
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import re
import time

from .artifact import validate_bundle, verify_bundle_assets, sha256
from ..modeling.ref_recipe import BIDIR_RECIPE, FIXED_RECIPE, STREAMING_TEXT_RECIPES
from .streaming_text import MAX_TEXT_ROWS, validate_prompt
from .raw_recipe import RAW_PROTOCOLS, ALL_STREAMING_TEXT, RawLatentRecipe, parse_recipe, validate_online_guidance
from .serving_stages import CpuOutput, PrefetchedProxy, serialize_rgb
from .settings import backbone_dit_options


class SessionError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise SessionError(f'{name} must be an integer >= {minimum}')
    return value


def load_artifacts(paths):
    artifacts = {}
    identity = None
    for path in paths:
        bundle = json.loads(Path(path).read_text())
        recipe = validate_bundle(bundle)
        validate_online_guidance(recipe)
        if recipe.protocol not in (FIXED_RECIPE, BIDIR_RECIPE, *STREAMING_TEXT_RECIPES, *RAW_PROTOCOLS):
            raise ValueError('serving accepts only fixed or bidir static-image and streaming-text recipes')
        current = deepcopy({key: bundle[key] for key in ('model', 'weights', 'encoders', 'deployment')})
        current['encoders']['recipe'].pop('encoder_code', None)
        if identity is not None and current != identity:
            raise ValueError('all served artifacts must have identical resident assets and deployment')
        identity = current
        artifacts[sha256(path)] = (bundle, recipe)
    if not artifacts:
        raise ValueError('at least one artifact is required')
    verify_bundle_assets(next(iter(artifacts.values()))[0])
    return artifacts


def runtime_config(bundle, options, output):
    from common.config import CfgNode
    from .artifact import runtime_backbone
    model = runtime_backbone(bundle)
    model.update(runtime=dict(training=False, requires_grad=False, param_dtype='bfloat16'),
        placement=dict(module='projects.minimax_h3.modeling.inference_runtime', class_name='ResidentInferencePlacement'))
    return CfgNode(dict(models=dict(backbone=model), distributed=deepcopy(bundle['deployment']['distributed']),
        engine=dict(up_size=dict(train=bundle['deployment']['active_up_size'])), seed=2026, logging={},
        persistence=dict(run_dir=str(output)),
        inference=dict(checkpoint=bundle['encoders']['checkpoint'], runtime=options)))


def settings_receipt(settings):
    receipt = deepcopy(settings)
    receipt['details_file'] = 'effective-settings.json'
    receipt['artifacts'] = [{key: artifact[key] for key in ('path', 'recipe', 'weights', 'deployment', 'units') if key in artifact}
                            for artifact in receipt['artifacts']]
    return receipt


@dataclass
class Request:
    op: str
    metadata: dict
    pixels: bytes = b''
    received_ns: int = field(default_factory=time.perf_counter_ns)


@dataclass
class Reply:
    metadata: dict
    pixels: bytes | memoryview | CpuOutput = b''
    status: int = 200


@dataclass(frozen=True)
class PromptSnapshot:
    revision: int
    text: str
    emitted_ns: int

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) != {'revision', 'text', 'emitted_ns'}:
            raise SessionError('prompt requires exactly revision, text and emitted_ns')
        revision = integer(value['revision'], 'prompt revision', 1)
        if revision >= 2**63:
            raise SessionError('prompt revision must fit signed int64')
        integer(value['emitted_ns'], 'prompt emitted_ns')
        try:
            validate_prompt(value['text'])
        except ValueError as exc:
            raise SessionError(str(exc)) from exc
        return cls(**value)


@dataclass
class Session:
    session_id: str
    artifact_id: str
    recipe: object
    next_chunk: int = 0
    next_source: int | None = None
    last_capture: int = -1
    accepted: int = 0
    emitted: int = 0
    output_id: int = 0
    pending: deque = field(default_factory=deque)
    prompt: PromptSnapshot | None = None


class SessionController:
    """CPU validation and provenance; model state belongs to ResidentBackend."""
    def __init__(self, artifacts, *, settings=None):
        self.artifacts = artifacts
        self.settings = settings
        self.active = None

    def prepare(self, request, *, commit=True):
        m = request.metadata
        if request.op == 'shutdown':
            if m or request.pixels:
                raise SessionError('shutdown takes no payload')
            return {}
        sid = m.get('session_id')
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', sid):
            raise SessionError('invalid session_id')
        if request.op == 'start':
            if self.active is not None:
                raise SessionError('a session is already active', 409)
            if not isinstance(m.get('artifact_id'), str) or m['artifact_id'] not in self.artifacts:
                raise SessionError('unknown artifact_id')
            bundle, recipe = self.artifacts[m['artifact_id']]
            fields = {'session_id', 'artifact_id', 'seed', 'condition_seed', 'prompt'}
            raw_fields = {'sink_latents', 'recent_latents', 'bootstrap_latents',
                          'current_latent_schedule', 'segment_boundaries'}
            raw = isinstance(recipe, RawLatentRecipe)
            optional = raw_fields if raw else {'window_size'}
            if not fields <= m.keys() or set(m)-fields-optional:
                raise SessionError('start fields differ from the protocol')
            if 'window_size' in m:
                integer(m['window_size'], 'window_size')
            integer(m['seed'], 'seed')
            integer(m['condition_seed'], 'condition_seed')
            if max(m['seed'], m['condition_seed']) >= 2**63:
                raise SessionError('seed must fit signed int64')
            try:
                recipe = replace(recipe, **{key:m[key] for key in optional if key in m})
            except ValueError as exc:
                raise SessionError(str(exc)) from exc
            spec = bundle['encoders']['recipe']
            if m['prompt'] != spec['system_prompt']:
                raise SessionError('prompt differs from the frozen artifact prompt')
            if len(request.pixels) != spec['height'] * spec['width'] * 3:
                raise SessionError('first RGB payload differs from artifact geometry')
            self.active = Session(sid, m['artifact_id'], recipe)
            return dict(m, recipe=recipe.to_dict())
        session = self.active
        if session is None or sid != session.session_id:
            raise SessionError('unknown active session', 409)
        if request.op in ('cancel', 'expire'):
            if set(m) - {'session_id', 'reason'} or request.pixels:
                raise SessionError('cancel takes only session_id and reason')
            if not isinstance(m.get('reason', ''), str) or len(m.get('reason', '')) > 1024:
                raise SessionError('invalid cancel reason')
            return m
        if request.op != 'chunk':
            raise SessionError('unknown operation')
        streaming = session.recipe.protocol in ALL_STREAMING_TEXT
        fields = {'session_id', 'chunk_id', 'first_source_frame', 'capture_ns', 'assembled_ns', 'eos'}
        if set(m) - ({'prompt'} if streaming else set()) != fields:
            raise SessionError('chunk fields differ from the protocol')
        prompt = session.prompt
        if streaming:
            if 'prompt' in m:
                prompt = PromptSnapshot.parse(m['prompt'])
                if session.prompt is not None and (prompt.revision < session.prompt.revision
                        or prompt.revision == session.prompt.revision and prompt != session.prompt):
                    raise SessionError('prompt revision must increase or repeat the identical snapshot')
            if prompt is None:
                raise SessionError('chunk 0 requires a prompt snapshot')
        integer(m['chunk_id'], 'chunk_id')
        integer(m['first_source_frame'], 'first_source_frame')
        integer(m['assembled_ns'], 'assembled_ns')
        if type(m['eos']) is not bool:
            raise SessionError('eos must be boolean')
        count = 5 if m['eos'] else 17
        stamps = m['capture_ns']
        if not isinstance(stamps, list) or len(stamps) != count:
            raise SessionError(f'expected {count} capture timestamps')
        for stamp in stamps:
            integer(stamp, 'capture_ns')
        if stamps[0] < session.last_capture or any(a > b for a, b in zip(stamps, stamps[1:])):
            raise SessionError('capture timestamps must be monotonic')
        if m['assembled_ns'] < stamps[-1]:
            raise SessionError('assembled_ns precedes capture')
        if m['chunk_id'] != session.next_chunk or (session.next_source is not None
                and m['first_source_frame'] != session.next_source):
            raise SessionError('out-of-order chunk or source frame', 409)
        spec = self.artifacts[session.artifact_id][0]['encoders']['recipe']
        if len(request.pixels) != count * spec['proxy_height'] * spec['proxy_width'] * 3:
            raise SessionError(f'expected {count} RGB proxy frames at artifact geometry')
        provenance = dict(prompt_revision=prompt.revision) if streaming else {}
        if commit:
            session.pending.extend(dict(chunk_id=m['chunk_id'], source_frame=m['first_source_frame'] + i,
                                        capture_ns=stamp, **provenance) for i, stamp in enumerate(stamps))
            session.prompt = prompt
            session.next_chunk += 1
            session.next_source = m['first_source_frame'] + count
            session.last_capture = stamps[-1]
            session.accepted += count
        return dict(m, prompt=asdict(prompt)) if streaming else m

    def finish(self, request, pixels, timings):
        session = self.active
        m = dict(session_id=request.metadata.get('session_id'), status='accepted', **timings)
        if request.op == 'start':
            m.update(recipe=session.recipe.to_dict(), artifact_id=session.artifact_id)
            if self.settings is not None:
                snapshot = deepcopy(self.settings)
                snapshot['scope'] = 'session'
                snapshot['session'] = dict(artifact_id=session.artifact_id,
                    recipe=session.recipe.to_dict(), seed=request.metadata['seed'],
                    condition_seed=request.metadata['condition_seed'])
                m['effective_settings'] = snapshot
        elif request.op == 'chunk':
            if isinstance(pixels, CpuOutput) and (pixels.key.session_id != session.session_id
                    or pixels.key.chunk_id != request.metadata['chunk_id']):
                raise SessionError('output identity differs from the accepted work', 500)
            spec = self.artifacts[session.artifact_id][0]['encoders']['recipe']
            size = spec['height'] * spec['width'] * 3
            if len(pixels) % size:
                raise SessionError('decoder returned an incomplete RGB frame', 500)
            count = len(pixels) // size
            if count > len(session.pending):
                raise SessionError('decoder produced more frames than the accepted source', 500)
            if session.recipe.protocol in ALL_STREAMING_TEXT:
                applied = timings.get('submitted_prompt_encoding' if isinstance(session.recipe, RawLatentRecipe) else 'applied_prompt')
                expected_sha = hashlib.sha256(session.prompt.text.encode('utf-8')).hexdigest()
                if (not isinstance(applied, dict) or set(applied) != {'text', 'sha256', 'first_chunk', 'rows'}
                        or applied['text'] != session.prompt.text or applied['sha256'] != expected_sha
                        or type(applied['first_chunk']) is not int
                        or not 0 <= applied['first_chunk'] <= request.metadata['chunk_id']
                        or type(applied['rows']) is not int or not 1 <= applied['rows'] <= MAX_TEXT_ROWS):
                    raise SessionError('driver applied prompt differs from the submitted text', 500)
                m['submitted_prompt'] = asdict(session.prompt)
                for source in session.pending:
                    if source['chunk_id'] == request.metadata['chunk_id']:
                        source.update(prompt_sha256=applied['sha256'],
                            prompt_first_chunk=applied['first_chunk'], prompt_rows=applied['rows'])
            sources = [session.pending.popleft() for _ in range(count)]
            m.update(chunk_id=request.metadata['chunk_id'], output_id=session.output_id,
                     output_frame_start=session.emitted, frame_count=count, source_frames=sources,
                     height=spec['height'], width=spec['width'], available_after_chunk=request.metadata['chunk_id'])
            session.emitted += count
            session.output_id += 1
            pending_bound = 17
            if isinstance(session.recipe, RawLatentRecipe):
                pending_bound += ((max(session.recipe.bootstrap_latents, *session.recipe.current_latent_schedule)+4)//5)*17
            if len(session.pending) > pending_bound:
                raise SessionError('decoder exceeded the bounded pending frame ledger', 500)
            if request.metadata['eos']:
                if session.pending or session.accepted != session.emitted:
                    raise SessionError('EOS output count does not equal accepted source frames', 500)
                m['status'] = 'complete'
            m.update(total_frames=session.emitted, accepted_frames=session.accepted)
        elif request.op in ('cancel', 'expire'):
            m['status'] = 'cancelled'
        elif request.op == 'shutdown':
            m['status'] = 'shutdown'
        return Reply(m, pixels)


class Distributed:
    def __init__(self):
        import torch.distributed as dist
        self.dist = dist
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def broadcast(self, value, *, src=0):
        import torch
        values = [value]
        if self.dist.is_initialized():
            self.dist.broadcast_object_list(values, src=src, device=torch.device('cpu'))
        return values[0]

    def barrier(self):
        if self.dist.is_initialized():
            self.dist.barrier()

    def gather(self, value):
        if not self.dist.is_initialized():
            return [value]
        values = [None] * self.dist.get_world_size()
        self.dist.all_gather_object(values, value)
        return values

    def rank0(self, function):
        return self.call_on(0, function)

    def call_on(self, rank, function):
        result = None
        if self.rank == rank:
            try:
                result = (True, function())
            except (ValueError, SessionError) as exc:
                result = (False, str(exc))
        success, value = self.broadcast(result, src=rank)
        if not success:
            raise SessionError(value, 500)
        return value


class _ChunkInput:
    def __init__(self):
        self.item = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.item is None:
            raise RuntimeError('recipe requested an input before push')
        item, self.item = self.item, None
        return item


class ResidentBackend:
    def __init__(self, config, artifacts, distributed, *, vae_rank=0, decoder_rank=0):
        import torch.distributed as dist
        from torch.distributed.device_mesh import DeviceMesh
        from common.engine import BaseEngine
        from common.distributed.ops import get_device
        from ..modeling.cwm_encoders import FrozenCWMEncoders
        from ..modeling.qwen_inference import FrozenQwenEncoder
        from ..modeling.inference_runtime import InferenceCodec, StageProfiler
        self.group = distributed
        self.vae_rank, self.decoder_rank = vae_rank, decoder_rank
        self.artifacts = artifacts
        self.config = config
        self.outputs = self.inputs = None
        self.prefetched_proxy = None
        self.static_document = self.first_latent = None
        self.prepared_text = self.submitted_text = self.encoded_text = self.text_document = None
        self.result = None
        self.cpu_output = None
        self.qwen_fsdp = config.inference.runtime.get('qwen_fsdp', False)
        self.qwen_encoder = None
        mesh = None
        if self.qwen_fsdp:
            if not dist.is_initialized() or dist.get_world_size() < 2:
                raise ValueError('qwen_fsdp requires at least two initialized distributed ranks')
            qwen_group = dist.new_group(ranks=[0, 1], backend='nccl')
            if self.group.rank < 2:
                mesh = DeviceMesh.from_group(qwen_group, 'cuda', mesh=[0, 1])
        self.profiler = StageProfiler(config.inference.runtime.profile)
        self.backbone = BaseEngine(config, None).models['backbone']
        self.backbone.eval()
        dit_options = backbone_dit_options(config.inference.runtime)
        if any(dit_options.values()):
            from ..modeling.inference_optimizations import configure_inference
            configure_inference(self.backbone, **dit_options)
        def load():
            self.encoders = FrozenCWMEncoders(config.inference.checkpoint, device=get_device(),
                posterior='mean', load_text_encoder=not self.qwen_fsdp and vae_rank == 0)
            self.codec = InferenceCodec(self.encoders, config.inference.runtime.codec,
                device=get_device(), load_decoder=vae_rank == decoder_rank)
        self.group.call_on(vae_rank, load)
        if decoder_rank != vae_rank:
            def load_decoder():
                self.codec = InferenceCodec(None, config.inference.runtime.codec,
                    device=get_device(), load_encoder=False)
            self.group.call_on(decoder_rank, load_decoder)
        if self.qwen_fsdp:
            if self.group.rank < 2:
                self.qwen_encoder = FrozenQwenEncoder(config.inference.checkpoint, device=get_device(), mesh=mesh)
        elif self.group.rank == 0:
            self.qwen_encoder = (self.encoders if vae_rank == 0 else
                FrozenQwenEncoder(config.inference.checkpoint, device=get_device()))
        self.weight_load_count = 1
        self.residency = self.group.gather(self.memory())
        present = [row for row in self.residency if row['qwen']['mode'] != 'absent']
        mode = 'fsdp2' if self.qwen_fsdp else 'single'
        ranks = [0, 1] if self.qwen_fsdp else [0]
        if [row['rank'] for row in present] != ranks or any(row['qwen']['mode'] != mode for row in present):
            raise RuntimeError('observed Qwen parameter placement differs from the serving topology')
        self.qwen_placement = dict(mode=mode, ranks=[row['rank'] for row in present],
            local_parameter_bytes=[row['qwen']['local_parameter_bytes'] for row in self.residency])

    def memory(self):
        import torch
        from torch.distributed.tensor import DTensor
        parameters = list(self.qwen_encoder.qwen.parameters()) if self.qwen_encoder is not None else []
        sharded = sum(isinstance(p, DTensor) for p in parameters)
        local = [p.to_local() if isinstance(p, DTensor) else p for p in parameters]
        mode = ('absent' if not parameters else 'fsdp2' if sharded == len(parameters)
                else 'single' if not sharded else 'mixed')
        qwen = dict(mode=mode, parameter_count=len(parameters), sharded_parameter_count=sharded,
                    local_parameter_numel=sum(p.numel() for p in local),
                    local_parameter_bytes=sum(p.numel() * p.element_size() for p in local))
        return dict(rank=self.group.rank, allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    weight_load_count=self.weight_load_count, qwen=qwen,
                    codec=dict(encoder_loaded=hasattr(self, 'encoders'),
                        decoder_loaded=getattr(getattr(self, 'codec', None), 'decoder', None) is not None),
                    qwen_local_parameter_bytes=qwen['local_parameter_bytes'])

    def _prepare_text(self, text):
        import torch
        from ..modeling.cwm_encoders import qwen_inputs
        try:
            with self.profiler.stage('text_prepare', cuda=False):
                validate_prompt(text)
                prepared = (self.qwen_encoder.prepare_condition(text) if self.qwen_fsdp else
                            qwen_inputs(self.qwen_encoder.processor, text))
                kwargs, tags = prepared
                ids = kwargs['input_ids']
                context = self.qwen_encoder.qwen.model.config.text_config.max_position_embeddings
                limit = min(MAX_TEXT_ROWS, context)
                if ids.device.type != 'cpu' or ids.ndim != 2 or ids.shape[0] != 1:
                    raise ValueError('text inputs must have one CPU token sequence')
                rows = ids.shape[1]
                if not 1 <= rows <= limit:
                    raise ValueError(f'text tokenizer rows {rows} exceed supported range 1..{limit}')
                if (tags.device.type != 'cpu' or tags.shape != (rows,)
                        or tags.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                        or not (tags == 1).all() or (kwargs['mm_token_type_ids'] != 0).any()):
                    raise ValueError('dynamic text inputs must have aligned all-text rows')
                return prepared
        except Exception as exc:
            # CPU preparation must reject on every rank before any model collective.
            raise SessionError(f'text preparation failed: {exc}') from exc

    def _encode_text(self, text):
        from .streaming_text import own_document
        from ..modeling.qwen_inference import encode_prepared
        document = None
        if text != self.submitted_text:
            raise RuntimeError('driver text differs from the prepared instruction')
        if self.qwen_fsdp:
            if self.qwen_encoder is not None:
                try:
                    with self.profiler.stage('text_qwen'):
                        document = self.qwen_encoder.encode_condition(self.prepared_text)
                except Exception as exc:
                    raise RuntimeError('Qwen collective failed; worker restart required') from exc
            with self.profiler.stage('text_broadcast', cuda=False):
                document = self.group.rank0(lambda: own_document(document, static=self.static_document))
        else:
            if self.group.rank == 0:
                try:
                    with self.profiler.stage('text_qwen'):
                        document = encode_prepared(self.qwen_encoder.qwen, self.qwen_encoder.device, self.prepared_text)
                except (ValueError, SessionError) as exc:
                    document = exc
            def publish():
                if isinstance(document, Exception):
                    raise document
                return own_document(document, static=self.static_document)
            with self.profiler.stage('text_broadcast', cuda=False):
                document = self.group.rank0(publish)
        self.dynamic_text_forwards += 1
        return document

    def begin(self, op, metadata):
        self.op, self.metadata = op, metadata
        self.timings = {}
        self.cpu_output = None
        self.result = self.prepared_text = None
        self.profiler.records.clear()
        self.profiler.origin = time.perf_counter()
        if op == 'start':
            self.recipe = parse_recipe(metadata['recipe'])
            self.streaming_text = self.recipe.protocol in ALL_STREAMING_TEXT
            self.dynamic_text_forwards = 0
            self.submitted_text = self.encoded_text = self.text_document = None
            self.static_document = self.first_latent = None
            self.spec = self.artifacts[metadata['artifact_id']][0]['encoders']['recipe']

    def static_condition(self, payload):
        import numpy as np
        def first_image():
            return np.frombuffer(payload, np.uint8).reshape(self.spec['height'], self.spec['width'], 3)
        if self.qwen_fsdp:
            def prepare():
                try:
                    with self.profiler.stage('static_qwen_prepare', cuda=False):
                        return self.qwen_encoder.prepare_condition(self.metadata['prompt'], images=[first_image()])
                except Exception as exc:
                    raise SessionError(f'static preparation failed: {exc}') from exc
            prepared = self.group.rank0(prepare)
            document = None
            if self.qwen_encoder is not None:
                try:
                    with self.profiler.stage('static_qwen'):
                        document = self.qwen_encoder.encode_condition(prepared)
                except Exception as exc:
                    raise RuntimeError('Qwen collective failed; worker restart required') from exc
            self.static_document = self.group.rank0(lambda: document)
        else:
            def condition():
                with self.profiler.stage('static_qwen'):
                    return self.qwen_encoder.condition(self.metadata['prompt'], images=[first_image()])
            self.static_document = self.group.rank0(condition)

    def first_image(self, payload):
        import numpy as np
        if self.group.rank in (self.vae_rank, self.decoder_rank):
            self.codec.begin_session()
        def encode():
            first = np.frombuffer(payload, np.uint8).reshape(self.spec['height'], self.spec['width'], 3)
            with self.profiler.stage('static_image_encode'):
                return self.encoders.image(first)
        self.first_latent = self.group.call_on(self.vae_rank, encode)

    def open_recipe(self):
        if (getattr(self.backbone, '_h3_inference', {}).get('delayed_hc', False)
                and (not isinstance(self.recipe, RawLatentRecipe) or self.recipe.bidirectional)):
            raise ValueError('delayed_hc requires a fixed-history v2 recipe')
        if isinstance(self.recipe, RawLatentRecipe):
            import torch
            from .raw_latent import RawLatentSession
            if self.config.inference.runtime.codec.decoder != 'causal':
                raise ValueError('raw-latent serving requires the causal decoder')
            self.inputs = _ChunkInput()
            with torch.no_grad():
                self.outputs = RawLatentSession(self.backbone, self.recipe, self.static_document,
                    seed=self.metadata['seed'], condition_seed=self.metadata['condition_seed'],
                    first_frame=self.first_latent, profiler=self.profiler,
                    encode_text=self._lookup_text if self.streaming_text else None,
                    fuse_first_denoise=getattr(self.config.inference.runtime, 'dit', {}).get('fuse_first_denoise', False))
            self.first_latent = None
            return
        from ..inference.recipes.ref_fixed_history import RefFixedHistory
        from ..inference.recipes.ref_bidir_history import RefBidirHistory
        self.inputs = _ChunkInput()
        drivers = {FIXED_RECIPE: RefFixedHistory, BIDIR_RECIPE: RefBidirHistory}
        options = {}
        if self.streaming_text:
            from .recipes.ref_streaming_text import RefFixedHistoryStreamingText, RefBidirHistoryStreamingText
            drivers.update(zip(STREAMING_TEXT_RECIPES,
                (RefFixedHistoryStreamingText, RefBidirHistoryStreamingText)))
            options['encode_text'] = self._lookup_text
        driver = drivers[self.recipe.protocol](self.recipe)
        self.outputs = driver.stream_latents(self.backbone, self.static_document, self.inputs,
            seed=self.metadata['seed'], condition_seed=self.metadata['condition_seed'], first_frame=self.first_latent,
            video_shape=(24, self.spec['height']//16, self.spec['width']//16),
            conditioning_mode=self.recipe.conditioning_mode, profiler=self.profiler, **options)
        self.first_latent = None

    def text_condition(self):
        if not self.streaming_text:
            return
        forwards_before = self.dynamic_text_forwards
        self.submitted_text = self.metadata['prompt']['text']
        if self.submitted_text != self.encoded_text:
            self.prepared_text = self.group.rank0(lambda: self._prepare_text(self.submitted_text))
            self.text_document = self._encode_text(self.submitted_text)
            self.encoded_text = self.submitted_text
            self.prepared_text = None
        self.timings['dynamic_text_forwards'] = self.dynamic_text_forwards - forwards_before
        self.timings['session_dynamic_text_forwards'] = self.dynamic_text_forwards

    def _lookup_text(self, text):
        if text != self.encoded_text or self.text_document is None:
            raise RuntimeError('recipe requested text before text_condition completed')
        return self.text_document

    def _encode_proxy(self, payload, metadata, timings, profiler):
        import numpy as np
        timings['encode_start_ns'] = time.perf_counter_ns()
        frames = np.frombuffer(payload, np.uint8).reshape(-1, self.spec['proxy_height'], self.spec['proxy_width'], 3)
        with profiler.stage('proxy_encode'):
            latent = self.codec.encode(frames, eos=metadata['eos'])
        timings['encode_submit_ns'] = time.perf_counter_ns()
        return dict(references=[dict(kind='video', role='semantic', video=latent)], eos=metadata['eos'])

    def proxy_encode(self, payload):
        prefetched = self.prefetched_proxy
        if prefetched is not None:
            if (prefetched.key.session_id != self.metadata['session_id']
                    or prefetched.key.chunk_id != self.metadata['chunk_id']):
                raise RuntimeError('prefetched input differs from the current chunk')
            self.inputs.item, self.prefetched_proxy = prefetched.item, None
            self.timings.update(prefetched.timings)
            if prefetched.profile is not None:
                self.timings['codec_prefetch_profile'] = prefetched.profile
        else:
            self.inputs.item = self.group.call_on(self.vae_rank,
                lambda: self._encode_proxy(payload, self.metadata, self.timings, self.profiler))
        if self.streaming_text:
            self.inputs.item['prompt'] = self.submitted_text
            if isinstance(self.recipe, RawLatentRecipe):
                self.inputs.item['prompt_revision'] = self.metadata['prompt']['revision']

    def decode_with_prefetch(self, key, metadata, payload):
        from ..modeling.inference_runtime import StageProfiler
        if self.prefetched_proxy is not None:
            raise RuntimeError('proxy prefetch slot is occupied')
        timings, item, error, profile = {}, None, None, None
        if self.group.rank == self.vae_rank:
            profiler = StageProfiler(self.profiler.enabled)
            try:
                item = self._encode_proxy(payload, metadata, timings, profiler)
                if profiler.enabled:
                    profile = dict(profiler.report(), origin_ns=int(profiler.origin * 1e9),
                                   clock='worker_local_monotonic')
            except (ValueError, SessionError) as exc:
                error = exc
        self.decode()
        def publish():
            if error is not None:
                raise error
            return item
        item = self.group.call_on(self.vae_rank, publish)
        self.prefetched_proxy = PrefetchedProxy(key, item, timings, profile)

    def render(self):
        import torch
        self.timings['render_start_ns'] = time.perf_counter_ns()
        with torch.no_grad(), self.profiler.stage('recipe'):
            result = (self.outputs.push(self.inputs.item) if isinstance(self.recipe, RawLatentRecipe)
                      else next(self.outputs))
        if isinstance(self.recipe, RawLatentRecipe):
            self.inputs.item = None
            self.timings['windows'] = []
            for span in result:
                sources = [dict(source, prompt=None if source['prompt'] is None else
                                {key:value for key,value in source['prompt'].items() if key != 'text'})
                           for source in span.sources]
                self.timings['windows'].append(dict(plan=span.plan.record(), sources=sources, **span.metrics))
            if self.streaming_text:
                self.timings['submitted_prompt_encoding'] = asdict(self.outputs.applied)
        elif self.streaming_text:
            self.timings['applied_prompt'] = asdict(result.prompt)
            result = result.video
        self.result = result
        self.timings['render_submit_ns'] = time.perf_counter_ns()

    def decode(self):
        import torch
        from ..modeling.inference_runtime import OutputSink
        def decode():
            raw = isinstance(getattr(self, 'recipe', None), RawLatentRecipe)
            values = [(span.video, span.plan.eos) for span in self.result] if raw else [(self.result, self.metadata['eos'])]
            if any(not torch.isfinite(value).all() for value, _ in values):
                raise ValueError('generated latent contains NaN or Inf')
            self.timings['decode_start_ns'] = time.perf_counter_ns()
            frames = []
            for i, (value, eos) in enumerate(values):
                started = time.perf_counter_ns()
                with self.profiler.stage('decode'):
                    rgb = self.codec.decode(value, eos=eos)
                if rgb is not None:
                    frames.append(rgb)
                if raw:
                    self.timings['windows'][i]['decode_submit_ns'] = time.perf_counter_ns()-started
                    self.timings['windows'][i]['decoded_frames'] = 0 if rgb is None else len(rgb)
            rgb = torch.cat(frames) if frames else None
            ready = torch.cuda.Event() if values and values[0][0].is_cuda else None
            if ready is not None:
                ready.record()
            def receive(key, cpu):
                self.timings['cpu_ready_ns'] = time.perf_counter_ns()
                self.cpu_output = cpu
            with torch.no_grad():
                sink = OutputSink(receive, profiler=self.profiler)
                sink.submit(None, rgb)
                sink.close()
            if ready is not None:
                ready.synchronize()
            self.timings['rgb_ready_upper_bound_ns'] = time.perf_counter_ns()
        self.group.call_on(self.decoder_rank, decode)

    def collect(self, *, defer_output=False):
        profiles = self.group.gather(self.profiler.report()) if self.profiler.enabled else None
        memory = self.group.gather(self.memory())
        rank_timings = self.group.gather(dict(rank=self.group.rank, **self.timings))
        if self.group.rank == 0:
            if profiles is not None:
                self.timings['profile'] = profiles[0]
                self.timings['rank_profiles'] = profiles
            self.timings['gpu_memory'] = memory
            self.timings['rank_timings'] = rank_timings
            if isinstance(getattr(self, 'recipe', None), RawLatentRecipe) and self.op == 'chunk':
                self.timings['windows'] = rank_timings[self.decoder_rank]['windows']
                self.timings['rank_timings'] = []
                for row in rank_timings:
                    compact = {key:value for key,value in row.items()
                               if key not in ('windows', 'submitted_prompt_encoding')}
                    graph_windows = [dict(window_index=w['plan']['index'], branches=w['dit_cuda_graph'])
                                     for w in row.get('windows', []) if w.get('dit_cuda_graph')]
                    if graph_windows:
                        compact['dit_cuda_graph_windows'] = graph_windows
                    self.timings['rank_timings'].append(compact)
        cpu, self.cpu_output = self.cpu_output, None
        self.result = self.prepared_text = self.first_latent = None
        return (cpu if defer_output else serialize_rgb(cpu)), self.timings

    def execute(self, op, metadata, payload):
        from .serving_stages import STAGES, STAGE_SPECS
        self.begin(op, metadata)
        for stage in STAGES.get(op, ('collect',)):
            method = getattr(self, stage)
            result = method(payload) if STAGE_SPECS[stage].payload_owner is not None else method()
        return result

    def close_session(self):
        if self.outputs is not None and not isinstance(getattr(self, 'recipe', None), RawLatentRecipe):
            self.outputs.close()
        self.outputs = self.inputs = None
        self.prefetched_proxy = None
        self.prepared_text = self.static_document = self.submitted_text = None
        self.encoded_text = self.text_document = self.first_latent = self.result = None
        self.cpu_output = None
        self.metadata = None
        self.timings = {}
        self.profiler.records.clear()
        if self.group.rank in (self.vae_rank, self.decoder_rank):
            self.codec.begin_session()


class Worker:
    def __init__(self, backend, group, controller=None, mailbox=None, *, idle_timeout=300, occupier=None):
        self.backend, self.group = backend, group
        self.controller, self.mailbox = controller, mailbox
        self.idle_timeout = idle_timeout
        self.occupier = occupier

    def pending_proxy(self):
        request = self.mailbox.peek_pending_chunk()
        if request is not None:
            try:
                return self.controller.prepare(request, commit=False), request.pixels
            except SessionError:
                pass

    def close_session(self):
        failure = None
        try:
            self.backend.close_session()
        except Exception as exc:
            if not getattr(self.backend, 'failed', False):
                raise
            failure = exc
        if self.group.rank == 0:
            self.controller.active = None
        self.group.barrier()
        if self.occupier is not None:
            if getattr(self.backend, 'failed', False):
                self.occupier.restore_after_shutdown()
            else:
                self.occupier.resume()
        if self.group.rank == 0:
            self.mailbox.session_closed()
        return failure

    def serve(self):
        while True:
            job = envelope = None
            if self.group.rank == 0:
                job = self.mailbox.take(self.idle_timeout if self.controller.active else None,
                    self.controller.active.session_id if self.controller.active else None)
                try:
                    self.mailbox.before_prepare(job, self.controller)
                    checked = self.controller.prepare(job.request)
                    self.mailbox.prepared(job)
                    envelope = dict(op=job.request.op, metadata=checked)
                except SessionError as exc:
                    envelope = dict(op='reject', error=str(exc), status=exc.status)
            envelope = self.group.broadcast(envelope)
            op = envelope['op']
            result = None
            terminal = op in ('cancel', 'expire', 'shutdown')
            try:
                if op == 'reject':
                    if self.group.rank == 0:
                        result = Reply(dict(status='error', error=envelope['error']), status=envelope['status'])
                else:
                    if op == 'start' and self.occupier is not None:
                        self.occupier.pause()
                    started = time.perf_counter_ns()
                    options = (dict(lookahead=self.pending_proxy)
                               if getattr(self.backend, 'codec_overlap', False) else {})
                    if job is not None and job.media is not None and getattr(self.backend, 'deferred_output', False):
                        options['defer_output'] = op == 'chunk'
                    pixels, timings = self.backend.execute(op, envelope['metadata'],
                        job.request.pixels if self.group.rank == 0 else b'', **options)
                    def finish():
                        nonlocal result
                        timings.update(process_start_ns=started, response_ready_ns=time.perf_counter_ns(),
                            request_complete_ns=job.request.received_ns, admitted_ns=job.admitted_ns)
                        result = self.controller.finish(job.request, pixels, timings)
                        return result.metadata['status'] in ('complete', 'cancelled', 'shutdown')
                    terminal = self.group.rank0(finish)
            except SessionError as exc:
                terminal = True
                if self.group.rank == 0:
                    result = Reply(dict(status='error', error=str(exc)), status=exc.status)
            except Exception as exc:
                if not getattr(self.backend, 'failed', False):
                    raise
                terminal = True
                if self.group.rank == 0:
                    result = Reply(dict(status='error', error=str(exc)), status=503)
            if terminal:
                failure = self.close_session()
                if failure is not None and self.group.rank == 0:
                    result = Reply(dict(status='error', error=str(failure)), status=503)
            if self.group.rank == 0 and self.occupier is not None:
                result.metadata['gpu_occupier'] = 'paused' if self.occupier.paused else 'running'
            delivery = None
            if self.group.rank == 0:
                delivery = self.mailbox.deliver(job, result)
            delivered = self.group.broadcast(delivery)
            if not terminal and not delivered and op != 'reject':
                self.close_session()
            result = pixels = None
            self.group.barrier()
            if self.group.rank == 0:
                self.mailbox.release(job)
            if op == 'shutdown':
                return
