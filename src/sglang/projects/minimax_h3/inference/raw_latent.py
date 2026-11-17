"""Raw-latent sessions with isolated prefix prefill and bounded dynamic history."""
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass
import hashlib
import time

import torch
from common.distributed.ops import get_device
from common.seed import RandomState, combine_seed

from .latent_windows import LatentPlanner
from .model import RefModel
from .raw_recipe import RawLatentRecipe
from .streaming_text import AppliedPrompt, own_document, validate_prompt
from ..modeling.causal_layout import CausalChunk, CausalLayout
from ..modeling.packing import _video_t_span
from ..modeling.ref_condition import GroupedKVCache, video_time


@dataclass(frozen=True)
class SourceLatent:
    position: int
    source_clip: int
    source_stop: int
    references: tuple
    reference_eps: tuple
    noise: torch.Tensor
    eps: torch.Tensor
    video: torch.Tensor | None
    document: dict | None
    prompt: dict | None


@dataclass(frozen=True)
class GeneratedSpan:
    video: torch.Tensor
    plan: object
    sources: list[dict]
    metrics: dict


class LatentKVCache(GroupedKVCache):
    """Each retained row carries its absolute latent or source-document identity."""
    def __init__(self, layers, *, padding=False):
        super().__init__(layers, 1, windows=[None], padding=padding)
        self.supports_inference_arena = False
        self.owners = []

    @contextmanager
    def denoise_graph_window(self, *, eligible):
        if hasattr(self, '_h3_dit_graph_window'):
            raise RuntimeError('denoise graph window is already active')
        self._h3_dit_graph_window = object()
        self._h3_dit_graph_eligible = eligible
        try:
            yield
        finally:
            del self._h3_dit_graph_window
            self._h3_dit_graph_eligible = False
            graphs = getattr(self, '_h3_dit_graphs', None)
            if graphs is not None:
                graphs.end_window()

    def retain(self, start):
        if hasattr(self, '_h3_dit_graph_window'):
            raise RuntimeError('cannot retain during a denoise graph window')
        keep = [i for i, (position, _) in enumerate(self.owners) if position < 0 or position >= start]
        if len(keep) == len(self.owners):
            return
        indices = {}
        for cache in (self.key_cache, self.value_cache):
            for layer, value in cache.items():
                if value is not None:
                    if value.device not in indices:
                        indices[value.device] = torch.tensor(keep, dtype=torch.long, device=value.device)
                    cache[layer] = value.index_select(0, indices[value.device])
        self.owners = [self.owners[i] for i in keep]
        self.kvlens[0] = len(keep)
        self.groups[0] = [(-1, len(keep))]

    def fork(self):
        """Share a snapshot; both caches acquire private arenas on later writes."""
        if hasattr(self, '_h3_dit_graph_window'):
            raise RuntimeError('cannot fork during a denoise graph window')
        graphs = self.__dict__.pop('_h3_dit_graphs', None)
        if graphs is not None:
            graphs.close()
        cache = LatentKVCache(len(self.key_cache), padding=self.batch_size > 1)
        cache.supports_inference_arena = self.supports_inference_arena
        cache.key_cache = dict(self.key_cache)
        cache.value_cache = dict(self.value_cache)
        cache.kvlens = list(self.kvlens)
        cache.groups = [list(group) for group in self.groups]
        cache.owners = list(self.owners)
        self.__dict__.pop('_h3_kv_arenas', None)
        return cache

    @property
    def documents(self):
        return {source for _, source in self.owners if source is not None}


@dataclass
class RawInputs:
    samples: list
    layouts: list
    seqlens: torch.Tensor
    reference_eps: list
    geometry: dict
    batch_size: int = 1


class RawLatentModel(RefModel):
    """Supply inference geometry to the shared packed H3 forward ABI."""
    def _ref_geometry(self, inputs, selection=None, native=False):
        return inputs.geometry

    def inputs(self, static, picture, picture_eps, frames, *, current_start=None,
               include_static=False, include_conditions=True, include_video=True,
               prefix=False, documents=(), drop_proxy=False, history_before=None):
        if history_before is not None and (current_start not in (None, history_before) or include_static
                or prefix or not include_conditions or not include_video):
            raise ValueError('history_before requires H/C packing with optional current noise at the same boundary')
        device = get_device()
        base = picture if not frames else frames[0].noise
        channels, _, height, width = base.shape
        shape = (channels, len(frames), height, width)
        current_offset = next((i for i, f in enumerate(frames) if f.position == current_start), len(frames))
        current_count = len(frames)-current_offset
        chunk = CausalChunk(0, None, 0, current_count*height*width//4,
                            current_offset, len(frames), 0, 0)
        layout = CausalLayout(len(static['prompt_embeds']), (chunk,),
            torch.empty(0, dtype=torch.long, device=device), shape, (2,32,0))
        refs, ref_eps = [dict(video=picture)], [picture_eps]
        reference_count = len(frames[0].references) if frames else 0
        for j in range(reference_count):
            refs.append(dict(video=torch.cat([f.references[j] for f in frames], 1)))
            ref_eps.append(torch.cat([f.reference_eps[j] for f in frames], 1))
        positions, tags, text, img_ids, text_ids, output, sources, owners = [], [], [], [], [], [], [], []
        cursor = 0

        def push(kind, pos, *, ref=None, start=0, stop=None, document=None, owner=-1, source=None):
            nonlocal cursor
            n = len(pos)
            ids = torch.arange(cursor, cursor+n, device=device)
            positions.append(pos)
            tags.append(torch.zeros(n, dtype=torch.long, device=device) if document is None else document['text_token_tags'])
            if document is not None:
                text.append(document['prompt_embeds'])
                text_ids.append(ids)
            else:
                img_ids.append(ids)
            if kind == 'noise':
                output.append(ids)
            sources.append((0, kind, 0, ref, start, stop, n, document is not None))
            owners.extend([(owner, source)]*n)
            cursor += n

        def visual(latent, origin, absolute):
            pos = self._visual_positions(latent.shape, origin)
            pos[:, 0] = video_time(absolute, absolute+1, origin)[0]
            return pos

        if include_static:
            pos = torch.zeros(len(static['prompt_embeds']), 3, dtype=torch.float64)
            pos[:, 0] = torch.arange(len(pos))
            push('static', pos, document=static)
            push('static', self._visual_positions(picture.shape, layout.text_len), ref=0, stop=1)
        origin = layout.text_len+1.
        span = _video_t_span(self.recipe.rope_reference_latents)
        seen = set(documents)
        for i, frame in enumerate(frames):
            owner = -1 if prefix else frame.position
            if include_conditions and (history_before is None or frame.position >= history_before):
                if not prefix and frame.document is not None and frame.source_clip not in seen:
                    seen.add(frame.source_clip)
                    pos = torch.zeros(len(frame.document['prompt_embeds']), 3, dtype=torch.float64)
                    pos[:, 0] = video_time(frame.source_clip*5, frame.source_clip*5+1, origin)[0]
                    push('condition', pos, document=frame.document, owner=frame.source_stop-1, source=frame.source_clip)
                if not drop_proxy:
                    for j, latent in enumerate(frame.references):
                        push('condition', visual(latent, origin+j*span, frame.position),
                             ref=j+1, start=i, stop=i+1, owner=owner)
            if include_video and (history_before is None or frame.position < history_before):
                kind = 'noise' if current_start is not None and frame.position >= current_start else 'history'
                push(kind, visual(frame.noise, origin+reference_count*span, frame.position),
                     start=i, stop=i+1, owner=owner)
        hc_rows = None
        if history_before is not None:
            h = sum(source[6] for source in sources if source[1] == 'history')
            if 0 < h < cursor:
                hc_rows = (h, cursor-h)
        denoise_rows = 0
        if history_before is not None and current_start is not None:
            for i, frame in enumerate(frames):
                if frame.position >= current_start:
                    push('noise', visual(frame.noise, origin+reference_count*span, frame.position),
                         start=i, stop=i+1, owner=frame.position)
            denoise_rows = sum(source[6] for source in sources if source[1] == 'noise')
            if hc_rows is None or not denoise_rows:
                raise ValueError('H/C/D pack requires nonempty history, condition and denoise roles')
        roles = hc_rows or (cursor,)
        if denoise_rows:
            roles += (denoise_rows,)
        padding = tuple(self._pad_rows(n) for n in roles)
        pad = sum(padding)
        if pad:
            positions.append(torch.zeros(pad, 3, dtype=torch.float64))
            tags.append(torch.full((pad,), -1, dtype=torch.long, device=device))
        if not text:
            text = [static['prompt_embeds'][:0]]
        cat = lambda values: torch.cat(values).to(device) if values else torch.empty(0, dtype=torch.long, device=device)
        geometry = dict(position_ids=cat(positions), token_tags=cat(tags),
            img_pos=cat(img_ids), text_pos=cat(text_ids), output_pos=cat(output),
            prompt_embeds=torch.cat(text), refiner_lengths=[len(doc) for doc in text],
            sample_lens=[cursor]+([pad] if self._sp_size()>1 else []), sources=sources, pad=pad,
            owners=owners, hc_rows=hc_rows, denoise_rows=denoise_rows, hc_padding=padding)
        sample = dict(references=refs)
        return RawInputs([sample], [layout], torch.tensor([cursor], dtype=torch.int32, device=device),
                         [ref_eps], geometry)

    def forward(self, backbone, inputs, frames, timestep, cache, *, store=False, cache_segments=None):
        empty = inputs.samples[0]['references'][0]['video'][:, :0]
        current = torch.cat([f.noise if f.video is None else f.video for f in frames], 1) if frames else empty
        eps = torch.cat([f.eps for f in frames], 1) if frames else empty
        if not inputs.geometry['sample_lens'][0]:
            return None
        output = self._ref_forward(backbone, inputs, video_xts=[current], video_context=[current],
            video_eps=[eps], video_timesteps=[timestep.reshape(1)],
            selection=('history' if store and not inputs.geometry['denoise_rows'] else 'noise', 0),
            cache=cache, store_kv=store,
            cache_segments=cache_segments)
        if store and cache_segments is None:
            cache.owners.extend(inputs.geometry['owners'])
        return output


class RawLatentSession:
    def __init__(self, backbone, recipe, static_condition, *, first_frame, seed=2026,
                 condition_seed=None, encode_text=None, profiler=None, fuse_first_denoise=False):
        if recipe.protocol == 'cwm_causal/v1':
            raise ValueError('CWM causal requires the native SGLang realtime session')
        if not isinstance(recipe, RawLatentRecipe) or backbone.training:
            raise ValueError('raw session requires a v2 recipe and eval-mode backbone')
        self.delayed_hc = getattr(backbone, '_h3_inference', {}).get('delayed_hc', False)
        if self.delayed_hc and recipe.bidirectional:
            raise ValueError('delayed_hc requires a fixed-history v2 recipe')
        if type(fuse_first_denoise) is not bool:
            raise ValueError('fuse_first_denoise must be a boolean')
        if fuse_first_denoise:
            from ..modeling.transformer.model import get_tp_world_size
            if (not self.delayed_hc or recipe.streaming_text or recipe.bidirectional
                    or recipe.guidance['mode'] != 'plain' or recipe.guidance['scale'] != 1
                    or torch.is_grad_enabled() or get_tp_world_size() != 1
                    or getattr(backbone, '_h3_inference', {}).get('cuda_graph', False)):
                raise ValueError('first-denoise fusion requires fixed static plain delayed-HC, no-grad, TP1 and graph-off')
        self.fuse_first_denoise = fuse_first_denoise
        self.pending_history = ()
        self.recipe, self.backbone = recipe, backbone
        self.model = RawLatentModel()
        self.model.recipe = recipe
        self.static = self.model._to_device(own_document(static_condition))
        self.picture = self._latent(first_frame, 1)
        self.condition_seed = seed if condition_seed is None else condition_seed
        generator = torch.Generator(device='cpu').manual_seed(combine_seed(self.condition_seed, 'first_frame', 'static'))
        self.picture_eps = torch.randn(self.picture.shape, generator=generator).to(get_device())
        self.stream_rng = RandomState(seed)
        self.encode_text, self.profiler = encode_text, profiler
        self.planner = LatentPlanner(recipe)
        self.frames, self.prefix, self.fixed = {}, {}, {}
        self.source_clip = self.available = 0
        self.signature = self.text = self.document = self.applied = self.weak_static = None
        self.last_submitted_prompt = None
        from common.diffusion import build_diffusion
        diffusion = build_diffusion(recipe.to_dict()['diffusion'])
        self.timesteps, self.sampler = diffusion['sampling_timesteps'], diffusion['sampler']
        if fuse_first_denoise and getattr(self.timesteps, 'dynamic_shift', False):
            raise ValueError('first-denoise fusion requires a static sampling grid')

    @staticmethod
    def _latent(value, count):
        if (not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[1] != count
                or min(value.shape) < 1 or value.shape[2] % 2 or value.shape[3] % 2
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError('expected a finite CTHW latent at the declared source length and even spatial size')
        return value.detach().clone().to(get_device())

    def _stage(self, name):
        return self.profiler.stage(name) if self.profiler else nullcontext()

    def _cache(self):
        cache = LatentKVCache(self.model._num_layers(self.backbone), padding=self.model._sp_size()>1)
        cache.supports_inference_arena = not self.recipe.bidirectional
        return cache

    def _branches(self):
        return ('positive', 'negative') if self.recipe.guidance['scale'] != 1 else ('positive',)

    def _graph_window(self, plan, branch):
        recipe = self.recipe
        return (not recipe.bidirectional and not recipe.streaming_text and not self.delayed_hc
                and recipe.guidance['mode'] == 'plain' and recipe.guidance['scale'] == 1
                and branch == 'positive' and not plan.bootstrap
                and len(recipe.current_latent_schedule) == 1
                and plan.current.count == recipe.current_latent_schedule[0]
                and plan.dynamic_recent.count == recipe.recent_latents)

    def _inputs(self, frames, branch, **kwargs):
        static = self.weak_static if branch == 'negative' else self.static
        return self.model.inputs(static, self.picture, self.picture_eps, frames,
            drop_proxy=branch == 'negative' and self.recipe.guidance['drop'] in ('proxy', 'prompt_proxy'),
            **kwargs)

    def _history_condition(self, current, branch, start, zero, *, first_timestep=None):
        from ..modeling.transformer.causal_model import CachedAttentionSegments
        cache = self.fixed[branch]
        frames = self.pending_history + tuple(current)
        with self._stage('condition_inputs'):
            retained = [i for i, (p, _) in enumerate(cache.owners) if p < 0 or p >= start]
            documents = {cache.owners[i][1] for i in retained if cache.owners[i][1] is not None}
            inputs = self._inputs(frames, branch, history_before=current[0].position,
                                  current_start=current[0].position if first_timestep is not None else None,
                                  documents=documents)
        with self._stage('cache_routing'):
            g = inputs.geometry
            old = len(cache.owners)
            h = sum(source[6] for source in g['sources'] if source[1] == 'history')
            denoise = g['denoise_rows']
            live = len(g['owners'])-denoise
            owners = cache.owners + g['owners']
            commit = retained + [old+i for i, (p, _) in enumerate(g['owners'][:live]) if p < 0 or p >= start]
            keys, q_lens, k_lens = list(range(old+h)), [h], [old+h]
            if live > h:
                keys.extend(commit)
                q_lens.append(live-h)
                k_lens.append(len(commit))
            if denoise:
                keys.extend(commit)
                keys.extend(range(old+live, old+live+denoise))
                q_lens.append(denoise)
                k_lens.append(len(commit)+denoise)
            pad_start = old+live+denoise
            for pad in g['hc_padding']:
                if pad:
                    keys.extend(range(pad_start, pad_start+pad))
                    q_lens.append(pad)
                    k_lens.append(pad)
                    pad_start += pad
            device = inputs.seqlens.device
            route = CachedAttentionSegments(
                torch.tensor(keys, dtype=torch.long, device=device),
                torch.tensor(q_lens, dtype=torch.int32, device=device),
                torch.tensor(k_lens, dtype=torch.int32, device=device),
                torch.tensor(commit, dtype=torch.long, device=device), hc_rows=g['hc_rows'],
                denoise_rows=denoise)
            updated = cache.fork()
        with self._stage('clean_condition'):
            output = self.model.forward(self.backbone, inputs, frames,
                zero if first_timestep is None else first_timestep, updated,
                store=True, cache_segments=route)
        updated.owners = [owners[i] for i in commit]
        updated.kvlens = [len(commit)] + [0]*(updated.batch_size-1)
        updated.groups = [[(-1, len(commit))]] + [[] for _ in range(updated.batch_size-1)]
        self.fixed[branch] = updated
        return output

    @torch.no_grad()
    def push(self, item):
        if self.planner.ended:
            raise ValueError('received a chunk after EOS')
        fields = {'references', 'eos', 'guidance_conditions', 'prompt', 'prompt_revision'}
        if not isinstance(item, dict) or set(item)-fields or not item.get('references'):
            raise ValueError('raw source requires references and optional EOS, guidance or prompt')
        eos = item.get('eos', False)
        if type(eos) is not bool:
            raise ValueError('eos must be boolean')
        count = 2 if eos else 5
        refs = item['references']
        if (not isinstance(refs, (list, tuple)) or any(not isinstance(r, dict)
                or set(r) != {'kind', 'role', 'video'} or r['kind'] != 'video'
                or not isinstance(r['role'], str) or not r['role'] for r in refs)):
            raise ValueError('source references require video kind, role and video')
        videos = tuple(self._latent(r['video'], count) for r in refs)
        signature = [(r['role'], tuple(v.shape[:1])+tuple(v.shape[2:])) for r,v in zip(refs, videos)]
        if len({r['role'] for r in refs}) != len(refs) or self.signature is not None and signature != self.signature:
            raise ValueError('reference roles or resolution changed within a stream')
        if any(v.shape[0] != self.picture.shape[0] for v in videos):
            raise ValueError('reference channels differ from the first image')
        text, document, applied = self.text, self.document, self.applied
        submitted = self.last_submitted_prompt
        weak_static = self.weak_static
        if self.recipe.streaming_text:
            text = item.get('prompt', self.text)
            validate_prompt(text)
            revision = item.get('prompt_revision')
            if revision is not None and (type(revision) is not int or not 1 <= revision < 2**63):
                raise ValueError('prompt_revision must be a positive signed int64')
            if text != self.text:
                if not callable(self.encode_text):
                    raise ValueError('streaming text requires encode_text')
                document = self.model._to_device(own_document(self.encode_text(text), static=self.static))
                applied = AppliedPrompt(text, hashlib.sha256(text.encode()).hexdigest(),
                                        self.source_clip, len(document['prompt_embeds']))
            submitted = dict(asdict(applied), source_clip=self.source_clip, revision=revision)
        elif 'prompt' in item or 'prompt_revision' in item:
            raise ValueError('fixed-text raw recipe does not accept dynamic prompts')
        if self.recipe.guidance['scale'] != 1:
            drop = self.recipe.guidance['drop']
            weak = item.get('guidance_conditions', {}).get(drop)
            if weak is None and drop != 'proxy':
                raise ValueError('prompt guidance needs an independently encoded weak document')
            weak_document = self.static if weak is None else self.model._to_device(own_document(weak['student_condition']['static']))
            if self.weak_static is not None and any(not torch.equal(weak_document[k], self.weak_static[k]) for k in weak_document):
                raise ValueError('weak static Qwen condition changed within a stream')
            weak_static = weak_document
        noises = []
        for ref, video in zip(refs, videos):
            generator = torch.Generator(device='cpu').manual_seed(combine_seed(self.condition_seed, ref['role'], self.source_clip))
            noises.append(torch.randn(video.shape, generator=generator).to(get_device()))
        rng = self.stream_rng.fork('cwm_rgb', 0, self.source_clip)
        shape = (self.picture.shape[0], count, *self.picture.shape[2:])
        noise = torch.randn(shape, device=get_device(), generator=self.model._generator(rng))
        eps = torch.randn(shape, device=get_device(), generator=self.model._generator(rng))
        self.signature, self.text, self.document, self.applied = signature, text, document, applied
        self.weak_static, self.last_submitted_prompt = weak_static, submitted
        for j in range(count):
            position = self.available+j
            self.frames[position] = SourceLatent(position, self.source_clip, self.available+count,
                tuple(v[:, j:j+1].clone() for v in videos), tuple(v[:, j:j+1].clone() for v in noises),
                noise[:, j:j+1].clone(), eps[:, j:j+1].clone(), None, self.document, self.last_submitted_prompt)
        self.available += count
        self.source_clip += 1
        outputs = []
        for plan in self.planner.ready(self.available, eos=eos):
            with self._stage('raw_generate'):
                outputs.append(self._generate(plan))
            keep_start = max(self.recipe.sink_latents, plan.current.stop-self.recipe.recent_latents)
            self.frames = {p:f for p,f in self.frames.items() if p < self.recipe.sink_latents or p >= keep_start}
        return outputs

    def _generate(self, plan):
        from dataclasses import replace
        started = time.perf_counter_ns()
        current = [self.frames[i] for i in range(plan.current.start, plan.current.stop)]
        history = [self.frames[i] for i in range(plan.dynamic_recent.start, plan.dynamic_recent.stop)]
        metrics = dict(nfe=0, prefix_prefill=0, prefix_cache_hits=0, history_prefill=0,
                       prefix_bank_policy='branch_only_zero_timestep', position_policy=self.recipe.position_policy)
        zero = current[0].noise.new_zeros(())
        fuse_first = self.fuse_first_denoise and bool(self.pending_history) and not plan.bootstrap
        if not plan.bootstrap:
            for branch in self._branches():
                if branch not in self.prefix:
                    sink = [self.frames[i] for i in range(plan.sink.stop)]
                    with self._stage('prefix_inputs'):
                        cache = self._cache()
                        inputs = self._inputs(sink, branch, include_static=True, prefix=True)
                    with self._stage('prefix_prefill'):
                        self.model.forward(self.backbone, inputs, sink, zero, cache, store=True)
                    self.prefix[branch] = cache
                    metrics['prefix_prefill'] += 1
                    if not self.recipe.bidirectional:
                        self.fixed[branch] = cache.fork()
                        if history:
                            with self._stage('history_inputs'):
                                inputs = self._inputs(history, branch)
                            with self._stage('history_prefill'):
                                self.model.forward(self.backbone, inputs, history, zero, self.fixed[branch], store=True)
                            metrics['history_prefill'] += 1
                else:
                    metrics['prefix_cache_hits'] += 1
                if not self.recipe.bidirectional:
                    if self.pending_history:
                        if not fuse_first:
                            self._history_condition(current, branch, plan.dynamic_recent.start, zero)
                        continue
                    with self._stage('cache_retain'):
                        self.fixed[branch].retain(plan.dynamic_recent.start)
                    cache = self.fixed[branch]
                    with self._stage('condition_inputs'):
                        inputs = self._inputs(current, branch, include_video=False, documents=cache.documents)
                    with self._stage('condition_prefill'):
                        self.model.forward(self.backbone, inputs, current, zero, cache, store=True)
            if not fuse_first:
                self.pending_history = ()
        contexts = {}
        for branch in self._branches():
            with self._stage('denoise_inputs'):
                if plan.bootstrap:
                    frames, cache = current, self._cache()
                    inputs = self._inputs(frames, branch, current_start=plan.current.start, include_static=True)
                elif self.recipe.bidirectional:
                    frames, cache = history+current, self.prefix[branch]
                    inputs = self._inputs(frames, branch, current_start=plan.current.start)
                else:
                    frames, cache = current, self.fixed[branch]
                    inputs = self._inputs(frames, branch, current_start=plan.current.start, include_conditions=False)
            contexts[branch] = frames, cache, inputs
        with self._stage('sampling_setup'):
            self.timesteps.set_timesteps(seqlen=contexts['positive'][2].seqlens, device=get_device())
            grid = self.timesteps.timesteps
            if grid.ndim != 1 or getattr(self.timesteps, 'dynamic_shift', False):
                raise ValueError('raw windows require a static length-independent sampling grid')
            xt = torch.cat([f.noise for f in current], 1)
            rng = self.stream_rng.fork('raw_window', plan.index)
        with self._stage('denoise'), ExitStack() as scopes:
            for branch, (_, cache, _) in contexts.items():
                if not fuse_first:
                    scopes.enter_context(cache.denoise_graph_window(eligible=self._graph_window(plan, branch)))
            for step, t in enumerate(grid):
                predictions = {}
                for branch, (frames, cache, inputs) in contexts.items():
                    live = [replace(f, video=xt[:, f.position-plan.current.start:f.position-plan.current.start+1])
                            if f.position >= plan.current.start else f for f in frames]
                    if fuse_first and step == 0:
                        predictions[branch] = self._history_condition(live, branch, plan.dynamic_recent.start,
                            zero, first_timestep=t)[0][0]
                        cache = self.fixed[branch]
                        contexts[branch] = frames, cache, inputs
                        self.pending_history = ()
                    else:
                        predictions[branch] = self.model.forward(self.backbone, inputs, live, t, cache)[0][0]
                    metrics['nfe'] += 1
                pred = predictions['positive']
                if 'negative' in predictions:
                    weak = predictions['negative']
                    pred = weak.float()+self.recipe.guidance['scale']*(pred.float()-weak.float())
                ts = t.expand(1)
                xt = self.sampler.step_to(pred=[pred], x_t=[xt], t=ts,
                    s=self.timesteps.get_next_timesteps(ts), rng=[rng], seqlens=contexts['positive'][2].seqlens)[0]
        with self._stage('latent_commit'):
            for i, f in enumerate(current):
                self.frames[f.position] = replace(f, video=xt[:, i:i+1].detach().clone())
        if not plan.bootstrap and not self.recipe.bidirectional:
            clean = tuple(self.frames[f.position] for f in current)
            if self.delayed_hc and not plan.eos:
                self.pending_history = clean
            else:
                for branch in self._branches():
                    with self._stage('clean_inputs'):
                        inputs = self._inputs(clean, branch, include_conditions=False)
                    with self._stage('clean_commit'):
                        self.model.forward(self.backbone, inputs, clean, zero, self.fixed[branch], store=True)
        sources = []
        for frame in current:
            if sources and sources[-1]['source_clip'] == frame.source_clip:
                sources[-1]['latent_stop'] = frame.position+1
            else:
                sources.append(dict(source_clip=frame.source_clip, latent_start=frame.position,
                                    latent_stop=frame.position+1, prompt=frame.prompt))
        metrics['elapsed_ns'] = time.perf_counter_ns()-started
        metrics['prefix_rows'] = {k:v.kvlens[0] for k,v in self.prefix.items()}
        metrics['retained_rows'] = {k:v.kvlens[0] for k,v in self.fixed.items()}
        metrics['dit_cuda_graph'] = {branch: cache._h3_dit_graphs.stats
            for branch, (_, cache, _) in contexts.items() if hasattr(cache, '_h3_dit_graphs')}
        return GeneratedSpan(xt, plan, sources, metrics)


class RawLatentDriver:
    def __init__(self, recipe):
        self.recipe = recipe

    def stream_latents(self, backbone, static_condition, chunks, *, first_frame, seed=2026,
                       condition_seed=None, encode_text=None, profiler=None, **kwargs):
        session = RawLatentSession(backbone, self.recipe, static_condition, first_frame=first_frame,
            seed=seed, condition_seed=condition_seed, encode_text=encode_text, profiler=profiler)
        for item in chunks:
            yield from session.push(item)
        if not session.planner.ended:
            raise ValueError('stream ended without EOS')
