"""CWM's two views of Ref2VA: native full context and online chunk conditions.

Only geometry/conditioning lives here. TF, TSCD and DMD retain their existing
algorithms. A proxy is observed context, never a denoising target. Cached Qwen
rows retain their modality tags but enter through text_pos, not img_pos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from common.distributed.ops import get_device
from common.seed import combine_seed
from projects.minimax_h3.data.ref_latent import (DEFAULT_WINDOW, DEFAULT_ROLE_SPAN, STATIC_CONDITIONING_MODE,
    causal_layout, conditioning_mode)
from .causal_layout import CausalChunk as _Chunk, CausalLayout as _Layout
from projects.minimax_h3.modeling.packing import (
    minimax_h3_packed_sequence_ref2va_blocks, _axis_from_sqrt_area, _video_t_grid, _video_t_span,
)
from utils.naive_cache import NaiveCache
from .inference_optimizations import inference_enabled, inference_geometry

_CHUNK_TIME = _video_t_span(5)
_FRAME_TIMES = _video_t_grid(5, 0.)


def video_time(start, stop, origin=0.):
    """H3 time grid for only the requested rows, independent of total length."""
    frames = torch.arange(start,stop)
    return origin + (frames//5).double()*_CHUNK_TIME + _FRAME_TIMES[frames%5]


@dataclass
class RefInputs:
    samples: list[dict]
    layouts: list[_Layout]
    seqlens: torch.Tensor
    window_sizes: list[int | None]
    reference_eps: list[list[torch.Tensor]]
    role_time_span: float
    geometry: dict = field(default_factory=dict)

    @property
    def batch_size(self):
        return len(self.samples)

    @property
    def clean_latents(self):
        return ([s['video'] for s in self.samples], [s['audio'] for s in self.samples])


@dataclass(frozen=True)
class Block:
    sample: int
    kind: str
    chunk: int
    start: int
    stop: int


def visible(q: Block, k: Block, window: int | None) -> bool:
    """The same dependency graph used by packed TF and GroupedKVCache."""
    if q.sample != k.sample:
        return False
    if q.kind == 'static':
        return k.kind == 'static'
    if k.kind == 'static':
        return True
    if window is not None and k.chunk < q.chunk - window:
        return False
    if k.chunk > q.chunk:
        return False
    if k.kind == 'noise':
        return q.kind == 'noise' and k.chunk == q.chunk
    if k.kind == 'history' and k.chunk == q.chunk:
        return q.kind == 'history'
    return True


class GroupedKVCache(NaiveCache):
    """Pin static text; evict C_j and H_j together before the next condition.

    ``window`` counts previous time groups, excluding the current group. Padding
    is an isolated extra sample discarded on every write. No noisy KV is stored.
    """
    def __init__(self, num_layers, batch_size, *, windows, padding=False):
        self.real_batch = batch_size
        super().__init__(num_layers, batch_size + int(padding),
                         sink=[0] * (batch_size + int(padding)),
                         window_size=[None] * (batch_size + int(padding)))
        self.windows = windows
        self.groups = [[] for _ in range(self.batch_size)]
        self.group = -1
        self.supports_inference_arena = self.real_batch == 1

    def begin_chunk(self, chunk):
        self.group = chunk
        spans, cursor = [], 0
        for b in range(self.batch_size):
            kept = []
            for group, size in self.groups[b]:
                window = self.windows[b] if b < self.real_batch else 0
                if group == -1 or window is None or group >= chunk - window:
                    spans.append((cursor, cursor + size))
                    kept.append((group, size))
                cursor += size
            self.groups[b] = kept
            self.kvlens[b] = sum(size for _, size in kept)
        arenas = getattr(self, "_h3_kv_arenas", {}) if not torch.is_grad_enabled() else {}
        if arenas and sum(b - a for a, b in spans) == cursor:
            return  # No eviction: keep the clean views into the inference arena.
        for layer in self.key_cache:
            for index, cache in enumerate((self.key_cache, self.value_cache)):
                value = cache[layer]
                if value is not None:
                    compact = torch.cat([value[a:b] for a, b in spans], 0)
                    arena = arenas.get(layer)
                    if (arena is not None and len(arena[index]) >= len(compact)
                            and arena[index].dtype == compact.dtype
                            and arena[index].device == compact.device
                            and arena[index].shape[1:] == compact.shape[1:]):
                        arena[index][:len(compact)].copy_(compact)
                        cache[layer] = arena[index][:len(compact)]
                    else:
                        cache[layer] = compact
                    del compact

    def update_kvlens(self, new_sample_lens):
        if len(new_sample_lens) != self.batch_size:
            raise ValueError('cache/sample topology changed')
        self._input_lens = [old + new for old, new in zip(self.kvlens, new_sample_lens)]
        for b, size in enumerate(new_sample_lens):
            if b >= self.real_batch:
                continue
            self.kvlens[b] += size
            if self.groups[b] and self.groups[b][-1][0] == self.group:
                self.groups[b][-1] = (self.group, self.groups[b][-1][1] + size)
            else:
                self.groups[b].append((self.group, size))

    def update_kvcache(self, layer_idx, new_keys, new_values, new_kvlens=None):
        # The padding sample is last and has no retained past.
        length = sum(self._input_lens[:self.real_batch])
        self.key_cache[layer_idx] = new_keys[:length]
        self.value_cache[layer_idx] = new_values[:length]


class CWMRefConditionMixin:
    def _condition_options(self):
        args = self.config.data.args
        return dict(args, causal=args.get('stage') != 'sft')

    def _build_inputs(self, batch, prompt_embeds=None, *, streaming=False):
        from projects.minimax_h3.data.ref_latent import validate_sample, packed_length
        samples = self._to_device(batch['ref_samples'])
        cfg = self._condition_options()
        modes = {conditioning_mode(s['metadata']) for s in samples}
        if len(modes) != 1:
            raise ValueError('one batch must have one conditioning mode')
        mode = next(iter(modes))
        if cfg.get('conditioning_mode',mode) != mode:
            raise ValueError('sample/config conditioning mode mismatch')
        window = cfg.get('window_size',DEFAULT_WINDOW)
        span = cfg.get('rope_reference_latents',DEFAULT_ROLE_SPAN)
        if cfg['causal']:
            causal_layout(window,span)
        layouts, reference_eps = [], []
        for sample in samples:
            validate_sample(sample, check_values=False, streaming=streaming)
            video, audio = sample['video'], sample['audio']
            frame_rows = video.shape[2] * video.shape[3] // 4
            chunks = tuple(_Chunk(0, 0 if end < video.shape[1] else None,
                                  0, (end - start) * frame_rows, start, end, 0, 0)
                           for start, end in sample['metadata']['chunk_latent_ranges'])
            layouts.append(_Layout(len(sample['student_condition']['static']['prompt_embeds']),
                                   chunks, torch.empty(0, dtype=torch.long, device=video.device),
                                   tuple(video.shape), tuple(audio.shape)))
            seed = sample['metadata']['condition_seed']
            generator = torch.Generator(device='cpu')
            if not cfg['causal']:
                generator.manual_seed(seed)
                reference_eps.append([torch.randn(ref['video'].shape,generator=generator).to(video.device)
                                      for ref in sample['references']])
            else:
                # A chunk's condition noise must not depend on future length
                # or whether it arrived alone or in an offline training clip.
                # A static image reference draws one noise, independent of chunks.
                first = sample['metadata'].get('stream_chunk_start',0)
                def noise(ref):
                    if ref['kind'] == 'image':
                        return torch.randn(ref['video'].shape,
                            generator=generator.manual_seed(combine_seed(seed,ref['role'],'static')))
                    return torch.cat([torch.randn((ref['video'].shape[0],b-a,*ref['video'].shape[2:]),
                        generator=generator.manual_seed(combine_seed(seed,ref['role'],first+k)))
                        for k,(a,b) in enumerate(sample['metadata']['chunk_latent_ranges'])],1)
                reference_eps.append([noise(ref).to(video.device) for ref in sample['references']])
        return RefInputs(samples, layouts,
                         torch.tensor([packed_length(s, 'tf') for s in samples],
                                      dtype=torch.int32, device=get_device()),
                         [window] * len(samples), reference_eps, _video_t_span(span))

    @staticmethod
    def _visual_positions(shape, origin, start=0, stop=None, *, chunk_start=0):
        _, t, h, w = shape
        stop = t if stop is None else stop
        y, x = torch.meshgrid(_axis_from_sqrt_area(h, 2, (h*w)**.5),
                              _axis_from_sqrt_area(w, 2, (h*w)**.5), indexing='ij')
        pos = torch.empty(stop-start, h*w//4, 3, dtype=torch.float64)
        pos[:, :, 0] = video_time(5*chunk_start+start,5*chunk_start+stop,float(origin))[:,None]
        pos[:, :, 1] = y.reshape(-1)
        pos[:, :, 2] = x.reshape(-1)
        return pos.flatten(0, 1)

    def _ref_geometry(self, inputs, selection=None, native=False):
        key = (native, selection)
        if key in inputs.geometry:
            return inputs.geometry[key]
        device = inputs.seqlens.device
        positions, tags, img_pos, text_pos, output_pos, blocks, text = [], [], [], [], [], [], []
        sample_lens, sources = [], []
        offset = 0
        for i, (sample, layout) in enumerate(zip(inputs.samples, inputs.layouts)):
            begin = offset
            def push(kind, chunk, pos, cond=None, ref=None, start=0, stop=None):
                nonlocal offset
                n = len(pos)
                blocks.append(Block(i, kind, chunk, offset, offset+n))
                positions.append(pos)
                tags.append(cond['text_token_tags'].cpu() if cond is not None else torch.zeros(n, dtype=torch.long))
                ids = torch.arange(offset, offset+n)
                if cond is not None:
                    text_pos.append(ids)
                    text.append(cond['prompt_embeds'])
                else:
                    img_pos.append(ids)
                if kind == 'noise':
                    output_pos.append(ids)
                sources.append((i, kind, chunk, ref, start, stop, n, cond is not None))
                offset += n
            if native:
                cond = sample['teacher_condition']
                refs = sample['references']
                native_pack = minimax_h3_packed_sequence_ref2va_blocks(
                    text_len=len(cond['prompt_embeds']), latent_t=layout.latent_shape[1],
                    latent_h=layout.latent_shape[2], latent_w=layout.latent_shape[3], audio_t=0,
                    ref_blocks=[dict(kind='image', latent_h=r['video'].shape[2], latent_w=r['video'].shape[3])
                                if r['kind'] == 'image' else
                                dict(kind='video', ref_audio_t=0, latent_t=r['video'].shape[1],
                                     latent_h=r['video'].shape[2], latent_w=r['video'].shape[3]) for r in refs],
                    seq_len=len(cond['prompt_embeds']) + sum(r['video'].numel()//r['video'].shape[0]//4 for r in refs)
                    + layout.latent_shape[1]*layout.latent_shape[2]*layout.latent_shape[3]//4)
                pos = native_pack['img_position_ids']
                cursor = len(cond['prompt_embeds'])
                push('static', -1, pos[:cursor], cond=cond)
                for j, ref in enumerate(refs):
                    n = ref['video'].numel()//ref['video'].shape[0]//4
                    push('condition', -1, pos[cursor:cursor+n], ref=j, stop=ref['video'].shape[1])
                    cursor += n
                push('noise', -1, pos[cursor:], stop=layout.latent_shape[1])
            else:
                first = sample['metadata'].get('stream_chunk_start',0)
                refs = sample['references']
                images = [j for j, r in enumerate(refs) if r['kind'] == 'image']
                videos = [j for j, r in enumerate(refs) if r['kind'] != 'image']
                # As in the native packer, each image reference takes one integer
                # time slot after the text; the video roles follow it.
                origins = [float(layout.text_len) + len(images)]
                for _ in videos:
                    origins.append(origins[-1] + inputs.role_time_span)
                cond = sample['student_condition']['static']
                if selection is None or selection[0] == 'static':
                    pos = torch.zeros(len(cond['prompt_embeds']), 3, dtype=torch.float64)
                    pos[:, 0] = torch.arange(len(pos))
                    push('static', -1, pos, cond=cond)
                    for slot, j in enumerate(images):
                        push('static', -1, self._visual_positions(refs[j]['video'].shape,
                             layout.text_len + slot), ref=j, stop=1)
                for k, chunk in enumerate(layout.chunks):
                    a, b = chunk.video_start, chunk.video_stop
                    window = selection is not None and selection[0] == 'window'
                    in_window = window and max(0, selection[1]-inputs.window_sizes[i]) <= k <= selection[1]
                    if selection is None or selection == ('condition', k) or in_window:
                        if sample['student_condition']['chunks']:
                            cond = sample['student_condition']['chunks'][k]
                            pos = torch.zeros(len(cond['prompt_embeds']), 3, dtype=torch.float64)
                            pos[:, 0] = video_time(5*(first+k),5*(first+k)+1,origins[0])
                            push('condition', k, pos, cond=cond)
                        for slot, j in enumerate(videos):
                            push('condition', k, self._visual_positions(refs[j]['video'].shape,
                                 origins[slot], a, b, chunk_start=first), ref=j, start=a, stop=b)
                    for kind in ('noise', 'history'):
                        if selection is None:
                            if kind == 'history' and k == len(layout.chunks)-1:
                                continue
                        elif window:
                            if not in_window or kind != ('noise' if k == selection[1] else 'history'):
                                continue
                        elif selection != (kind, k):
                            continue
                        push(kind, k, self._visual_positions(layout.latent_shape,origins[-1],a,b,
                                                           chunk_start=first),start=a,stop=b)
            sample_lens.append(offset-begin)
        pad = self._pad_rows(offset)
        if pad:
            positions.append(torch.zeros(pad, 3, dtype=torch.float64))
            tags.append(torch.full((pad,), -1, dtype=torch.long))
            blocks.append(Block(inputs.batch_size, 'static', -1, offset, offset+pad))
        # Cached forwards always carry an extra padding sample under SP, even
        # when this particular query happens to be aligned.
        if self._sp_size() > 1:
            sample_lens.append(pad)
        cat = lambda xs: torch.cat(xs).to(device) if xs else torch.empty(0, dtype=torch.long, device=device)
        # RGB-only cached calls have no text projection/refiner work. The H3
        # _embed ABI supports a zero live-text document explicitly.
        if not text:
            text = [inputs.samples[0]['student_condition']['static']['prompt_embeds'][:0]]
        lengths = [len(t) for t in text]
        geometry = dict(position_ids=cat(positions), token_tags=cat(tags), img_pos=cat(img_pos),
                        text_pos=cat(text_pos), output_pos=cat(output_pos), blocks=blocks,
                        prompt_embeds=torch.cat(text), refiner_lengths=lengths,
                        sample_lens=sample_lens, sources=sources, pad=pad)
        if selection is None and not native:
            q, k = [], []
            allowed = torch.tensor([[visible(a, b, inputs.window_sizes[a.sample] if a.sample < inputs.batch_size else None)
                                     for b in blocks] for a in blocks], dtype=torch.bool)
            for qi, qb in enumerate(blocks):
                spans = []
                for ki, kb in enumerate(blocks):
                    if bool(allowed[qi, ki]):
                        if spans and spans[-1][1] == kb.start:
                            spans[-1][1] = kb.stop
                        else:
                            spans.append([kb.start, kb.stop])
                q.extend([[qb.start, qb.stop]] * len(spans)); k.extend(spans)
            geometry.update(q_ranges=torch.tensor(q, dtype=torch.int32, device=device),
                            k_ranges=torch.tensor(k, dtype=torch.int32, device=device),
                            attn_type_map=torch.zeros(len(q), dtype=torch.int32, device=device),
                            attn_workloads=[sum((b-a)*(d-c) for (a,b),(c,d) in zip(q,k))])
            from utils.flex_attn import FLEX_FLASH_ATTN_AVAILABLE, _FUSION_KNOBS
            geometry['attention_mask'] = None
            if not FLEX_FLASH_ATTN_AVAILABLE:
                from torch.nn.attention.flex_attention import create_block_mask
                import torch._inductor.config as inductor_config
                allowed = allowed.to(device)
                row_block = torch.cat([torch.full((b.stop-b.start,), i, dtype=torch.long)
                                       for i,b in enumerate(blocks)]).to(device)
                n = len(row_block)
                def mask(batch, head, query, key):
                    return (query<n) & (key<n) & allowed[row_block[query.clamp(max=n-1)], row_block[key.clamp(max=n-1)]]
                with inductor_config.patch(_FUSION_KNOBS):
                    geometry['attention_mask'] = create_block_mask(mask, 1, 1, n, n, device=str(device), _compile=True)
        inputs.geometry[key] = geometry
        return geometry

    def _ref_forward(self, model, inputs, *, video_xts, video_context=None,
                     video_eps=None, video_timesteps, selection=None, native=False,
                     cache=None, store_kv=False, cache_segments=None):
        if selection is not None and selection[0] == 'window':
            if native or cache is None or store_kv:
                raise ValueError('window selection requires a read-only static cache')
            if any(any(group != -1 for group, _ in groups) for groups in cache.groups):
                raise ValueError('window selection requires a static-only cache')
        g = self._ref_geometry(inputs, selection, native)
        # A proxy-dropped branch has no per-chunk condition rows to prefill.
        if selection is not None and selection[0] not in ('noise', 'window') and not any(g['sample_lens'][:inputs.batch_size]):
            return None
        device = inputs.seqlens.device
        dim = inputs.layouts[0].latent_shape[0] * 4
        xs, eps, times = [], [], []
        for i, kind, chunk, ref, start, stop, n, is_text in g['sources']:
            zero = video_xts[i].new_zeros
            if is_text:
                x, e = zero((n,dim)), zero((n,dim))
            elif ref is not None:
                x = self._video_rows(inputs.samples[i]['references'][ref]['video'], start, stop)
                e = self._video_rows(inputs.reference_eps[i][ref], start, stop)
            elif kind == 'history':
                x = self._video_rows(video_context[i], start, stop)
                e = self._video_rows(video_eps[i], start, stop)
            else:
                x = self._video_rows(video_xts[i], start, stop)
                e = torch.zeros_like(x)
            t = video_timesteps[i] if native else video_timesteps[i][max(chunk, 0)]
            if kind != 'noise' and not (native and is_text):
                t = t.new_zeros(())
            xs.append(x); eps.append(e); times.append(t.float().expand(n))
        if g['pad']:
            xs.append(xs[0].new_zeros((g['pad'],dim)))
            eps.append(xs[0].new_zeros((g['pad'],dim)))
            times.append(torch.zeros(g['pad'], device=device))
        x = torch.cat(xs).unsqueeze(0).to(getattr(model, 'param_dtype', torch.bfloat16))
        e = torch.cat(eps).unsqueeze(0).to(x.dtype)
        audio = x.new_zeros((1,x.shape[1],inputs.layouts[0].audio_shape[1]))
        if inference_enabled(model, 'metadata'):
            # Sort one time per source, then expand the inverse to token rows.
            nonempty = [t for t in times if t.numel()]
            unique, source_inverse = torch.unique(torch.stack([t[0] for t in nonempty]), return_inverse=True)
            lengths = torch.tensor([t.numel() for t in nonempty], device=device)
            inverse = torch.repeat_interleave(source_inverse, lengths, output_size=x.shape[1])
        else:
            unique, inverse = torch.unique(torch.cat(times), return_inverse=True)
        prepared = inference_geometry(model, g) if not native else {}
        if not prepared:
            cu = torch.tensor([0]+g['refiner_lengths'], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
            prepared = dict(img_position_ids=g['position_ids'].unsqueeze(0),
                update_mask=torch.ones(len(g['output_pos']), dtype=torch.bool, device=device),
                audio_pos_info={'position_ids':torch.empty(0, dtype=torch.long, device=device)},
                refiner_packed_seq_params={'cu_seqlens_q':cu, 'max_seqlen_q':max(g['refiner_lengths'])})
        kwargs = dict(x=x, audio_x=audio, eps=e, audio_eps=audio,
                      unique_timesteps=unique, inverse_indices=inverse, token_tags=g['token_tags'],
                      prompt_embeds=g['prompt_embeds'], img_pos_info={'position_ids':g['img_pos']},
                      text_pos_info={'position_ids':g['text_pos']},
                      img_pos_for_infer_output_info={'position_ids':g['output_pos']}, **prepared)
        if native:
            lens = [v for v in g['sample_lens'] if v]
            cu = torch.tensor([0]+lens, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
            kwargs['packed_seq_params'] = {'cu_seqlens_q':cu, 'max_seqlen_q':max(lens)}
        else:
            kwargs['sample_lens'] = prepared.get('sample_lens', g['sample_lens'])
            if cache is not None:
                kwargs.update(past_key_values=cache, update_past_key_values=store_kv)
                if cache_segments is not None:
                    kwargs['cache_segments'] = cache_segments
            else:
                kwargs.update({k:g[k] for k in ('q_ranges','k_ranges','attn_type_map','attn_workloads','attention_mask')})
        if (not native and selection is not None and selection[0] not in ('noise', 'window')
                and cache is not None and store_kv and not model.training and not torch.is_grad_enabled()):
            kwargs['cache_only'] = True
        result = model(**kwargs)
        if selection is not None and selection[0] not in ('noise', 'window'):
            return None
        output, _ = result
        videos, cursor = [], 0
        for layout in inputs.layouts:
            if selection is None:
                n = layout.latent_shape[1]*layout.latent_shape[2]*layout.latent_shape[3]//4
                videos.append(self._video_from_rows(output[cursor:cursor+n], layout))
            else:
                from dataclasses import replace
                chunk = layout.chunks[selection[1]]
                n = chunk.video_rows
                short = replace(layout, latent_shape=(layout.latent_shape[0],chunk.video_stop-chunk.video_start,*layout.latent_shape[2:]))
                videos.append(self._video_from_rows(output[cursor:cursor+n], short))
            cursor += n
        return videos, [v.new_empty(layout.audio_shape) for v,layout in zip(videos,inputs.layouts)]

    def _causal_packed_forward(self, model, inputs, **kwargs):
        return self._ref_forward(model, inputs, **{k:v for k,v in kwargs.items() if k.startswith('video_')})

    def _bidirectional_forward(self, model, inputs, **kwargs):
        return self._ref_forward(model, inputs, native=True,
                                 **{k:v for k,v in kwargs.items() if k.startswith('video_')})
