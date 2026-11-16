"""Static S KV plus a live bidirectional C/H/current window at every NFE."""
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from common.distributed.ops import get_device
from common.seed import RandomState

from projects.minimax_h3.data.ref_latent import SCHEMA
from projects.minimax_h3.inference.recipes.ref_fixed_history import RefFixedHistory
from projects.minimax_h3.modeling.ref_condition import GroupedKVCache
from projects.minimax_h3.modeling.ref_recipe import BIDIR_RECIPE, STATIC_CONDITIONING_MODE


@dataclass(frozen=True)
class _HistoryChunk:
    references: list[dict]
    video: torch.Tensor
    eps: torch.Tensor
    condition: dict | None = None


class RefBidirHistory(RefFixedHistory):
    PROTOCOL = BIDIR_RECIPE

    def _window_inputs(self, current, history):
        if not history:
            return current
        sample = current.samples[0]
        refs = [sample['references'][0]]
        for j, ref in enumerate(sample['references'][1:]):
            refs.append(dict(ref, video=torch.cat([h.references[j]['video'] for h in history]+[ref['video']], 1)))
        video = torch.cat([h.video for h in history]+[sample['video']], 1)
        frames = 17*len(history)+sample['metadata']['num_frames']
        metadata = dict(sample['metadata'], num_frames=frames,
            stream_chunk_start=sample['metadata']['stream_chunk_start']-len(history),
            chunk_raw_ranges=[[a, min(a+17, frames)] for a in range(0, frames, 17)],
            chunk_latent_ranges=[[a, min(a+5, video.shape[1])] for a in range(0, video.shape[1], 5)])
        condition = sample['student_condition']
        if condition['chunks']:
            condition = dict(condition, chunks=[h.condition for h in history]+condition['chunks'])
        return self._build_inputs({'ref_samples': [dict(sample, video=video, references=refs,
                                                       student_condition=condition, metadata=metadata)]},
                                  streaming=True)

    @torch.no_grad()
    def stream_latents(self, backbone, static_condition, chunks, *, seed=2026,
                       condition_seed=None, video_shape=None, first_frame=None,
                       conditioning_mode=STATIC_CONDITIONING_MODE, profiler=None):
        """Yield current RGB only; all retained C/H query current N at each step."""
        if getattr(backbone, '_h3_inference', {}).get('delayed_hc', False):
            raise ValueError('delayed_hc requires a fixed-history v2 recipe')
        mode = self._stream_conditioning_mode(conditioning_mode)
        if backbone.training:
            raise ValueError('stream_latents requires a causal eval-mode backbone')
        if first_frame is None:
            raise ValueError('static-image conditioning requires a first_frame image reference')
        static_condition = {key: value.detach().clone() for key, value in static_condition.items()}
        picture = dict(kind='image', role='first_frame', video=first_frame.detach().clone())
        history = deque(maxlen=self.recipe.window_size)
        cache, weak_cache, weak_static, signature, ended = None, None, None, None, False
        stream_rng = RandomState(seed)
        for k, item in enumerate(chunks):
            if ended:
                raise ValueError('received a chunk after EOS')
            if not item['references']:
                raise ValueError('at least one semantic reference is required')
            if mode == STATIC_CONDITIONING_MODE and 'condition' in item:
                raise ValueError('static-image streaming chunks must omit Qwen condition')
            refs = [picture]+[dict(ref, video=ref['video'].detach().clone()) for ref in item['references']]
            current_signature = [(r['role'], r['video'].shape[0], *r['video'].shape[2:]) for r in refs]
            if signature is None:
                signature = current_signature
            if current_signature != signature:
                raise ValueError('reference roles or resolution changed within a stream')
            ended = bool(item.get('eos', False))
            frames, latents = (5, 2) if ended else (17, 5)
            c, h, w = video_shape or (refs[0]['video'].shape[0], *refs[0]['video'].shape[2:])
            sample = dict(video=torch.zeros(c, latents, h, w), audio=torch.empty(2, 32, 0), references=refs,
                guidance_conditions=item.get('guidance_conditions', {}), teacher_condition=static_condition,
                student_condition=dict(static=static_condition,
                    chunks=[] if mode == STATIC_CONDITIONING_MODE else [item['condition']]),
                metadata=dict(schema=SCHEMA, fps=24, normalized=True, num_frames=frames,
                    conditioning_mode=mode, condition_seed=seed if condition_seed is None else condition_seed,
                    stream_chunk_start=k, chunk_raw_ranges=[[0, frames]], chunk_latent_ranges=[[0, latents]]))
            with profiler.stage('stream_inputs') if profiler else nullcontext():
                current = self._build_inputs({'ref_samples': [sample]}, streaming=True)
                inputs = self._window_inputs(current, history)
            weak = self._sampling_inputs(inputs)
            if weak is not None:
                document = weak.samples[0]['student_condition']['static']
                if weak_static is not None and any(document[key].dtype != weak_static[key].dtype
                        or not torch.equal(document[key], weak_static[key]) for key in weak_static):
                    raise ValueError('weak static Qwen condition changed within a stream')
                if weak_static is None:
                    weak_static = {key: document[key].detach().clone() for key in ('prompt_embeds', 'text_token_tags')}
            rng = stream_rng.fork('cwm_rgb', 0, k)
            shape = current.samples[0]['video'].shape
            xt = torch.randn(shape, device=get_device(), generator=self._generator(rng))
            eps = torch.randn(shape, device=get_device(), generator=self._generator(rng))
            result = torch.cat([entry.video for entry in history]+[torch.zeros_like(xt)], 1)
            context_eps = torch.cat([entry.eps for entry in history]+[eps], 1)
            times = result.new_zeros(len(history)+1)
            args = dict(video_xts=[result], video_context=[result], video_eps=[context_eps],
                        video_timesteps=[times], weak=weak)
            if cache is None:
                cache = GroupedKVCache(self._num_layers(backbone), 1, windows=inputs.window_sizes,
                                       padding=self._sp_size()>1)
                weak_cache = (GroupedKVCache(self._num_layers(backbone), 1, windows=inputs.window_sizes,
                              padding=self._sp_size()>1) if weak is not None else None)
                with profiler.stage('static_prefill') if profiler else nullcontext():
                    self._sampling_forward(backbone, inputs, selection=('static', -1), store_kv=True,
                                           cache=cache, weak_cache=weak_cache, **args)
            with profiler.stage('denoise') if profiler else nullcontext():
                for t in self._rollout_grid(current):
                    result[:, -latents:] = xt
                    times[-1] = t
                    prediction, _ = self._sampling_forward(backbone, inputs, selection=('window', len(history)),
                                                          cache=cache, weak_cache=weak_cache, **args)
                    ts = t.expand(1)
                    xt = self.sampler.step_to(pred=prediction, x_t=[xt], t=ts,
                        s=self.sampling_timesteps.get_next_timesteps(ts), rng=[rng], seqlens=current.seqlens)[0]
            # Consumers and sources may reuse their buffers after yielding.
            if not ended:
                history.append(_HistoryChunk(current.samples[0]['references'][1:], xt.detach().clone(), eps,
                    None if mode == STATIC_CONDITIONING_MODE else
                    {key: value.detach().clone() for key, value in item['condition'].items()}))
            del current, inputs, weak, args, result, context_eps, sample, item, refs
            yield xt
