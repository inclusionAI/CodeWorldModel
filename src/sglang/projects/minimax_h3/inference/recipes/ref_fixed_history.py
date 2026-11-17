"""Independent fixed recent-history Ref recipe. No training rollout calls."""
from contextlib import nullcontext

import torch
from common.distributed.ops import get_device
from common.seed import RandomState
from projects.minimax_h3.data.ref_latent import (LEGACY_CONDITIONING_MODE, STATIC_CONDITIONING_MODE)
from projects.minimax_h3.modeling.ref_condition import GroupedKVCache
from projects.minimax_h3.inference.model import RefModel
from projects.minimax_h3.modeling.ref_recipe import FIXED_RECIPE


class RefFixedHistory(RefModel):
    PROTOCOL = FIXED_RECIPE

    def __init__(self, recipe):
        if recipe.protocol != self.PROTOCOL:
            raise ValueError(f'{type(self).__name__} requires protocol {self.PROTOCOL}')
        from common.diffusion import build_diffusion
        self.recipe = recipe
        self.guidance = recipe.guidance
        diffusion = build_diffusion(recipe.to_dict()['diffusion'])
        self.sampling_timesteps = diffusion['sampling_timesteps']
        self.sampler = diffusion['sampler']

    @torch.no_grad()
    def _rollout_grid(self, inputs):
        timesteps = self.sampling_timesteps
        if getattr(timesteps,'dynamic_shift',False):
            raise ValueError('CWM rollout requires length-independent sampling timesteps')
        timesteps.set_timesteps(seqlen=inputs.seqlens,device=get_device())
        # Keep scheduler precision: its index(t) uses exact equality. DiT
        # conditioning is cast to float32 separately by _ref_forward.
        grid = timesteps.timesteps
        if grid.ndim != 1:
            raise ValueError('CWM rollout requires a static sampling grid')
        return grid

    def _rollout_chunk(self, backbone, inputs, k, args, grid, streams,
                       *, profiler=None):
        timesteps = self.sampling_timesteps
        sampler = self.sampler
        result,times = args['video_xts'],args['video_timesteps']
        rngs,xts = [],[]
        for i,(sample,layout,rng) in enumerate(zip(inputs.samples,inputs.layouts,streams)):
            c = layout.chunks[k]
            r = rng.fork('cwm_rgb',i,sample['metadata'].get('stream_chunk_start',0)+k)
            rngs.append(r)
            shape = (layout.latent_shape[0],c.video_stop-c.video_start,*layout.latent_shape[2:])
            xts.append(torch.randn(shape,device=get_device(),generator=self._generator(r)))
            args['video_eps'][i][:,c.video_start:c.video_stop] = torch.randn(
                shape,device=get_device(),generator=self._generator(r))
        with profiler.stage('condition_prefill') if profiler else nullcontext():
            self._sampling_forward(backbone,inputs,selection=('condition',k),store_kv=True,**args)
        with profiler.stage('denoise') if profiler else nullcontext():
            for t in grid:
                for i,l in enumerate(inputs.layouts):
                    c = l.chunks[k]
                    result[i][:,c.video_start:c.video_stop] = xts[i]
                    times[i][k] = t
                prediction,_ = self._sampling_forward(backbone,inputs,selection=('noise',k),**args)
                ts = t.expand(inputs.batch_size)
                xts = sampler.step_to(pred=prediction,x_t=xts,t=ts,
                    s=timesteps.get_next_timesteps(ts),rng=rngs,seqlens=inputs.seqlens)
        for i,l in enumerate(inputs.layouts):
            c = l.chunks[k]
            result[i][:,c.video_start:c.video_stop] = xts[i]
            times[i][k] = 0
        with profiler.stage('clean_commit') if profiler else nullcontext():
            self._sampling_forward(backbone,inputs,selection=('history',k),store_kv=True,**args)

    @torch.no_grad()
    def stream_latents(self, backbone, static_condition, chunks, *, seed=2026,
                       condition_seed=None, video_shape=None, first_frame=None,
                       conditioning_mode=LEGACY_CONDITIONING_MODE, profiler=None):
        """Yield normalized RGB chunks from an iterator; never prefetch future input.

        Each item has ordered proxy ``references`` and optional ``eos`` (5 raw/2
        latent frames; otherwise 17/5). Legacy mode also requires an independent
        Qwen ``condition`` per chunk. ``static_image_qwen_proxy_vae_v1`` omits it:
        Qwen encodes only the initial image and prompt, once. ``first_frame``
        is the stream's normalized [C,1,H,W] first RGB latent from
        ``FrozenCWMEncoders.image``; pass it exactly when the cache recipe included
        it, with ``static_condition`` encoded on the same picture. video_shape is
        optional (C,H,W), defaulting to the first reference's spatial shape. All
        SP/FSDP ranks must call this with identical inputs and an eval-mode
        backbone. Only static text, the picture and the finite KV window survive
        between iterations. The artifact already resolves the serving grid independently of training stage.
        """
        from projects.minimax_h3.data.ref_latent import SCHEMA
        if getattr(backbone, '_h3_inference', {}).get('delayed_hc', False):
            raise ValueError('delayed_hc requires a fixed-history v2 recipe')
        mode = self._stream_conditioning_mode(conditioning_mode)
        if backbone.training:
            raise ValueError('stream_latents requires a causal eval-mode backbone')
        cache,weak_cache,weak_static,signature,ended = None,None,None,None,False
        streams = [RandomState(seed)]
        picture = [] if first_frame is None else [dict(kind='image',role='first_frame',video=first_frame)]
        for k,item in enumerate(chunks):
            if ended:
                raise ValueError('received a chunk after EOS')
            if not item['references']:
                raise ValueError('at least one semantic reference is required')
            if mode == STATIC_CONDITIONING_MODE and 'condition' in item:
                raise ValueError('static-image streaming chunks must omit Qwen condition')
            refs = picture + list(item['references'])
            current = [(r['role'],r['video'].shape[0],*r['video'].shape[2:]) for r in refs]
            if signature is None:
                signature = current
            if current != signature:
                raise ValueError('reference roles or resolution changed within a stream')
            ended = bool(item.get('eos',False))
            frames,latents = (5,2) if ended else (17,5)
            c,h,w = video_shape or (refs[0]['video'].shape[0],*refs[0]['video'].shape[2:])
            sample = dict(video=torch.zeros(c,latents,h,w),audio=torch.empty(2,32,0),references=refs,
                guidance_conditions=item.get('guidance_conditions',{}),
                teacher_condition=static_condition,
                student_condition=dict(static=static_condition,
                    chunks=[] if mode == STATIC_CONDITIONING_MODE else [item['condition']]),
                metadata=dict(schema=SCHEMA,fps=24,normalized=True,num_frames=frames,conditioning_mode=mode,
                    condition_seed=seed if condition_seed is None else condition_seed,stream_chunk_start=k,
                    chunk_raw_ranges=[[0,frames]],chunk_latent_ranges=[[0,latents]]))
            with profiler.stage('stream_inputs') if profiler else nullcontext():
                inputs = self._build_inputs({'ref_samples':[sample]},streaming=True)
            weak = self._sampling_inputs(inputs)
            if weak is not None:
                current = weak.samples[0]['student_condition']['static']
                if weak_static is not None and any(current[key].dtype != weak_static[key].dtype
                        or not torch.equal(current[key],weak_static[key])
                        for key in ('prompt_embeds','text_token_tags')):
                    raise ValueError('weak static Qwen condition changed within a stream')
                if weak_static is None:
                    weak_static = {key:current[key].detach().clone()
                                   for key in ('prompt_embeds','text_token_tags')}
            result = torch.zeros_like(inputs.samples[0]['video'])
            args = dict(video_xts=[result],video_context=[result],video_eps=[torch.zeros_like(result)],
                        video_timesteps=[result.new_zeros(1)])
            if cache is None:
                cache = GroupedKVCache(self._num_layers(backbone),1,windows=inputs.window_sizes,
                                       padding=self._sp_size()>1)
                weak_cache = (GroupedKVCache(self._num_layers(backbone),1,windows=inputs.window_sizes,
                              padding=self._sp_size()>1) if weak is not None else None)
                with profiler.stage('static_prefill') if profiler else nullcontext():
                    self._sampling_forward(backbone,inputs,selection=('static',-1),store_kv=True,
                                           cache=cache,weak=weak,weak_cache=weak_cache,**args)
            cache.begin_chunk(k)
            if weak_cache is not None:
                weak_cache.begin_chunk(k)
            args['cache'] = cache
            args.update(weak=weak,weak_cache=weak_cache)
            self._rollout_chunk(backbone,inputs,0,args,self._rollout_grid(inputs),streams,
                                profiler=profiler)
            del inputs,args,sample,item,refs
            yield result
