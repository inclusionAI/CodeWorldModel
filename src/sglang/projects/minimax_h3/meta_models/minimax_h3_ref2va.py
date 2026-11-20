"""CWM Ref2VA SFT -> causal TF -> TSCD -> DMD on RAVEN's existing engines."""
from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext

import torch

from common.distributed.ops import all_reduce_sum, get_device, get_rank, get_world_size
from common.distributed.unified_parallel import SPDistForward
from common.meter import get_running_average_meter
from common.phase import ExecutionPhase, execution_phase
from common.seed import RandomState, local_seed, yield_seed
from projects.minimax_h3.meta_models.causal_minimax_h3_base import _RolloutX0s
from projects.minimax_h3.meta_models.causal_minimax_h3_tf import CausalMiniMaxH3TF
from projects.minimax_h3.meta_models.causal_minimax_h3_tscd import CausalMiniMaxH3TSCD
from projects.minimax_h3.meta_models.causal_minimax_h3_dmd import CausalMiniMaxH3DMD
from projects.minimax_h3.modeling.ref_condition import CWMRefConditionMixin, GroupedKVCache
from projects.minimax_h3.modeling.ref_guidance import RefGuidanceMixin
from projects.minimax_h3.modeling.sampling import EulerSampler
from projects.minimax_h3.modeling.training_timesteps import RefSFTTrainingTimesteps
from projects.minimax_h3.data.ref_latent import (LEGACY_CONDITIONING_MODE, STATIC_CONDITIONING_MODE,
    conditioning_mode as get_conditioning_mode)


class _RefTraining(RefGuidanceMixin, CWMRefConditionMixin):
    def __init__(self, config):
        if 'inference_recipe' in config:
            from projects.minimax_h3.modeling.ref_recipe import RefRecipe
            self.inference_recipe = RefRecipe.from_training(config)
        super().__init__(config)

    @execution_phase(ExecutionPhase.PREPARE)
    def prepare_inputs(self, ctx):
        ctx['inputs'] = self._build_inputs(ctx['batch'])
        ctx['clean_latents'] = ctx['inputs'].clean_latents
        ctx['neg_inputs'] = None
        return ctx

    def sync_inputs(self, ctx):
        if self._sp_size() <= 1:
            yield ctx
            return
        sync = SPDistForward(name='cwm_ref_inputs', comm_shape=True, device=get_device())
        for batch in sync(self._to_device(ctx['batch'])):
            sub = dict(ctx, batch=batch)
            # Synchronize cached Qwen outputs, never run frozen encoders inside
            # the training loop or broadcast a pickled GPU dataclass.
            yield self.prepare_inputs(sub)

    def _loss_fn(self, video_preds, audio_preds, *, inputs, video_targets=None,
                 audio_targets=None, key_prefix):
        if any(a.numel() for a in audio_preds):
            raise ValueError('CWM pipeline expects absent audio')
        alpha = (self.chunk_wise_weighting or {}).get(key_prefix)
        weights = None if alpha is None else [self._chunk_weights(l,float(alpha),get_device()) for l in inputs.layouts]
        total, count = self._reduce_modality(video_preds, video_targets, inputs=inputs,
                                              modality='video', weights=weights)
        all_reduce_sum(count)
        if count.item() <= 0:
            raise ValueError('batch has no supervised RGB elements')
        loss = total * get_world_size() / count
        get_running_average_meter().put_scalar(f'running/{key_prefix}/video', loss.detach().item())
        return loss

    def _dmd_terms(self, x0s, fakes, reals, modality, inputs):
        # Do not even evaluate mean(empty) in the inherited DMD normalizer.
        if modality == 'audio':
            return [torch.zeros_like(x) for x in x0s]
        return super()._dmd_terms(x0s, fakes, reals, modality, inputs)

    @torch.no_grad()
    def _rollout_grid(self, inputs, diffusion=None):
        timesteps = (diffusion or {}).get('sampling_timesteps',self.sampling_timesteps)
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
                       trajectory=(), trajectory_x0_chunks=None, *, diffusion=None, profiler=None):
        """One shared condition/denoise/clean-commit path for DMD and live input."""
        timesteps = (diffusion or {}).get('sampling_timesteps',self.sampling_timesteps)
        sampler = (diffusion or {}).get('sampler',self.sampler)
        result,times = args['video_xts'],args['video_timesteps']
        audio = [v.new_empty(l.audio_shape) for v,l in zip(result,inputs.layouts)]
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
            for step,t in enumerate(grid):
                for i,l in enumerate(inputs.layouts):
                    c = l.chunks[k]
                    result[i][:,c.video_start:c.video_stop] = xts[i]
                    times[i][k] = t
                    if trajectory:
                        trajectory[step][0][i][:,c.video_start:c.video_stop] = xts[i]
                prediction,_ = self._sampling_forward(backbone,inputs,selection=('noise',k),**args)
                if trajectory_x0_chunks is not None:
                    trajectory_x0_chunks.append(([p.detach().clone() for p in prediction],audio))
                ts = t.expand(inputs.batch_size)
                xts = sampler.step_to(pred=prediction,x_t=xts,t=ts,
                    s=timesteps.get_next_timesteps(ts),rng=rngs,seqlens=inputs.seqlens)
        for i,l in enumerate(inputs.layouts):
            c = l.chunks[k]
            result[i][:,c.video_start:c.video_stop] = xts[i]
            times[i][k] = 0
        # Noisy KV never becomes history; commit a clean rerun exactly once.
        with profiler.stage('clean_commit') if profiler else nullcontext():
            self._sampling_forward(backbone,inputs,selection=('history',k),store_kv=True,**args)

    @torch.no_grad()
    def _rollout_latents(self, backbone, inputs, rng, *, keep_trajectory,
                         rngs=None, trajectory_x0_chunks=None):
        device = get_device()
        streams = rngs if rngs is not None else [rng]*inputs.batch_size
        result = [torch.zeros(l.latent_shape,device=device) for l in inputs.layouts]
        video_eps = [torch.zeros_like(v) for v in result]
        audio = [v.new_empty(l.audio_shape) for v,l in zip(result,inputs.layouts)]
        grid = self._rollout_grid(inputs)
        counts = {len(l.chunks) for l in inputs.layouts}
        if len(counts) != 1:
            raise ValueError('rollout pack must have one temporal bucket')
        trajectory = [([torch.zeros_like(v) for v in result],audio) for _ in grid] if keep_trajectory else []
        cache = GroupedKVCache(self._num_layers(backbone),inputs.batch_size,
                               windows=inputs.window_sizes,padding=self._sp_size()>1)
        weak = self._sampling_inputs(inputs)
        weak_cache = (GroupedKVCache(self._num_layers(backbone),inputs.batch_size,
                       windows=inputs.window_sizes,padding=self._sp_size()>1) if weak is not None else None)
        times = [torch.zeros(len(l.chunks),device=device) for l in inputs.layouts]
        args = dict(video_xts=result,video_context=result,video_eps=video_eps,
                    video_timesteps=times,cache=cache,weak=weak,weak_cache=weak_cache)
        self._sampling_forward(backbone,inputs,selection=('static',-1),store_kv=True,**args)
        for k in range(next(iter(counts))):
            cache.begin_chunk(k)
            if weak_cache is not None:
                weak_cache.begin_chunk(k)
            self._rollout_chunk(backbone,inputs,k,args,grid,streams,trajectory,trajectory_x0_chunks)
        return _RolloutX0s(result,audio,video_eps,audio),trajectory

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
        between iterations. TSCD uses its validation diffusion (the few-step
        inference grid).
        """
        from projects.minimax_h3.data.ref_latent import SCHEMA
        mode = get_conditioning_mode({'conditioning_mode': conditioning_mode})
        if isinstance(self,MiniMaxH3RefSFT) or backbone.training:
            raise ValueError('stream_latents requires a causal eval-mode backbone')
        cache,weak_cache,weak_static,signature,ended = None,None,None,None,False
        streams = [RandomState(seed)]
        diffusion = getattr(self,'validation_diffusion',None)
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
            self._rollout_chunk(backbone,inputs,0,args,self._rollout_grid(inputs,diffusion),streams,
                                diffusion=diffusion,profiler=profiler)
            # Do not retain every previous input/geometry in a live session.
            del inputs,args,sample,item,refs
            yield result

    @torch.no_grad()
    def _sample_native(self, backbone, inputs, rng):
        weak = self._sampling_inputs(inputs)
        video,audio = self._sample_noises(inputs,rng)
        self.sampling_timesteps.set_timesteps(seqlen=inputs.seqlens,device=get_device())
        for t in self.sampling_timesteps.timesteps:
            ts = t.expand(inputs.batch_size)
            pred,_ = self._sampling_forward(backbone,inputs,weak=weak,native=True,
                                            video_xts=video,audio_xts=audio,
                                            video_timesteps=ts,audio_timesteps=ts)
            video = self.sampler.step_to(pred=pred,x_t=video,t=ts,
                s=self.sampling_timesteps.get_next_timesteps(ts),rng=rng,seqlens=inputs.seqlens)
        return video

    def _save_validation(self, path, value):
        torch.save(value, path)

    @torch.no_grad()
    def validate(self, ctx):
        from projects.minimax_h3.data.ref_latent import read_manifest, load_entry
        cfg = self.config.validation
        path,entries,_ = read_manifest(cfg.manifest)
        root = Path(cfg.get('output_dir',str(Path(self.config.persistence.output_dir)/'cwm_validation')))
        step = int(ctx.get('step',ctx.get('iter',0)))
        root = root / str(step)
        if get_rank() == 0:
            root.mkdir(parents=True,exist_ok=True)
        name = 'backbone_ema' if cfg.get('validate_ema',False) else 'backbone'
        backbone = ctx['models'][name]
        was_training = backbone.training
        backbone.eval()
        old = self.sampling_timesteps,self.audio_sampling_timesteps,self.sampler
        if hasattr(self,'validation_diffusion'):
            self.sampling_timesteps = self.validation_diffusion['sampling_timesteps']
            self.audio_sampling_timesteps = self.validation_diffusion['audio_sampling_timesteps']
            self.sampler = self.validation_diffusion['sampler']
        try:
            for i,entry in enumerate(entries[:int(cfg.get('num_samples',2))]):
                inputs = self._build_inputs({'ref_samples':[load_entry(path,entry)]})
                rng = RandomState(seed=int(cfg.get('seed',2026))+i)
                video = (self._sample_native(backbone,inputs,rng) if isinstance(self,MiniMaxH3RefSFT)
                         else self._rollout_latents(backbone,inputs,rng,keep_trajectory=False)[0].video)
                if get_rank() == 0:
                    self._save_validation(root/f'{entry["sample_id"]}.pt',
                                          dict(video=video[0].cpu(),metadata=inputs.samples[0]['metadata'],
                                               guidance=dict(getattr(self,'guidance',{'mode':'plain'}),
                                                             scale=self._guidance_scale())))
        finally:
            self.sampling_timesteps,self.audio_sampling_timesteps,self.sampler = old
            backbone.train(was_training)
        return ctx


class MiniMaxH3RefTF(_RefTraining, CausalMiniMaxH3TF):
    """Chunk FM with GT RGB history; shared source for TSCD and rollout.

    ``objective=gaussian_velocity`` (default) uses the SFT Gaussian weighting
    at each chunk's continuous sigma. Explicit ``objective=x0`` retains the
    historical unweighted x0 MSE.

    Optional ``meta_model.resampling`` is Resampling Forcing (arXiv:2512.15702,
    Alg. 1): after ``warmup_steps`` optimizer steps of plain TF, the RGB history
    is the online model's own detached resampling of GT -- every history chunk
    noised to a per-sample t_s (logit-normal, then ``shift``; paper 0.6) and
    denoised to 0 in ``steps`` Euler steps (default 20; explicit ``steps=1``
    retains the paper's setting), conditioned on the chunks resampled before
    it. Noisy inputs and x0 targets stay GT.
    """

    def __init__(self, config):
        super().__init__(config)
        rf = config.meta_model.get('resampling')
        self.resampling = None
        if rf is not None:
            warmup,shift,steps = int(rf['warmup_steps']),float(rf.get('shift',0.6)),int(rf.get('steps',20))
            if warmup < 0 or not shift > 0 or steps < 1:
                raise ValueError('resampling needs warmup_steps >= 0, shift > 0 and steps >= 1')
            self.resampling = dict(warmup_steps=warmup,shift=shift,steps=steps)
            # Resampling's ts-to-zero Euler trajectory has its own step count.
            self._resampler = EulerSampler(schedule=self.schedule)
            self._ga_steps = int(config.engine.ga_steps)

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def forward(self, ctx):
        """Reads: models, inputs, noisy_latents, clean_latents, context_eps,
        train_timesteps, and iter/rng when resampling. Writes: pred."""
        backbone,inputs = ctx['models']['backbone'],ctx['inputs']
        history,eps = ctx['clean_latents'][0],ctx['context_eps'][0]
        # iter counts micro-steps; warmup is in optimizer steps, like the LR schedule.
        if self.resampling is not None and ctx['iter']//self._ga_steps >= self.resampling['warmup_steps']:
            history = self._resample_history(backbone,inputs,history,eps,ctx['rng'])
        ctx['pred'] = self._guidance_forward(ctx,self._causal_packed_forward,video_xts=ctx['noisy_latents'][0],
            video_context=history,video_eps=eps,video_timesteps=ctx['train_timesteps'][0])
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def compute_loss(self, ctx):
        objective = self.config.meta_model.get('objective','gaussian_velocity')
        if objective == 'x0':
            return super().compute_loss(ctx)
        if objective != 'gaussian_velocity':
            raise ValueError('TF objective must be gaussian_velocity or x0')
        if not hasattr(self,'_gaussian_loss_timesteps'):
            # Match the default SFT recipe, independent of the continuous TF
            # sampler's clamp or the distribution of this batch's draws.
            self._gaussian_loss_timesteps = RefSFTTrainingTimesteps(shift=12.0)
        terms = []
        for pred,target,sigma,layout in zip(ctx['pred'][0],ctx['clean_latents'][0],
                ctx['train_timesteps'][0],ctx['inputs'].layouts,strict=True):
            sigma = sigma.float()
            if sigma.ndim != 1 or len(sigma) != len(layout.chunks):
                raise ValueError('TF requires one sigma per RGB chunk')
            if not torch.isfinite(sigma).all() or ((sigma <= 0) | (sigma > 1)).any():
                raise ValueError('velocity TF requires finite sigma in (0, 1]')
            weight = self._gaussian_loss_timesteps.continuous_loss_weight(sigma)
            # x0=xt+sigma*v, so the velocity residual is (pred_x0-x0)/sigma.
            # Apply sqrt(weight) before squaring to avoid an unnecessary large
            # intermediate near zero; all arithmetic stays FP32 without a clamp.
            scale = torch.cat([s.expand(c.video_stop-c.video_start)
                for s,c in zip(weight.sqrt()/sigma,layout.chunks,strict=True)])
            loss = ((pred.float()-target.detach().float()) * scale[None,:,None,None]).square()
            if not torch.isfinite(loss).all():
                raise FloatingPointError('TF Gaussian velocity loss is not finite in FP32')
            terms.append(loss)
        ctx['loss'] = self._loss_fn(terms,ctx['pred'][1],inputs=ctx['inputs'],key_prefix='train_losses')
        return ctx

    @execution_phase(ExecutionPhase.ROLLOUT)
    @torch.no_grad()
    def _resample_history(self, backbone, inputs, clean, eps, rng):
        """Alg. 1 (a): autoregressive self-resampling of the history, no grad.

        ``_rollout_chunk``'s cache protocol, but each chunk starts from GT noised
        to t_s instead of pure noise. History KV is committed with the context
        ``eps`` the packed forward reuses. The last chunk is never history.
        """
        rf = self.resampling
        if len({len(l.chunks) for l in inputs.layouts}) != 1:
            raise ValueError('resampling pack must have one temporal bucket')
        # Eq. 6-7: logit(t_s) ~ N(0, 1), then the timestep shift.
        u = torch.randn(inputs.batch_size,device=get_device(),dtype=torch.float64,
                        generator=self._generator(rng)).sigmoid()
        t_s = (rf['shift']*u/(1+(rf['shift']-1)*u)).float()
        grid = t_s[:,None]*torch.linspace(1,0,rf['steps']+1,device=t_s.device)
        noise = self._sample_noises(inputs,rng)[0]
        history = [x.to(torch.float32,copy=True) for x in clean]
        times = [x.new_zeros(len(l.chunks)) for x,l in zip(history,inputs.layouts)]
        cache = GroupedKVCache(self._num_layers(backbone),inputs.batch_size,
                               windows=inputs.window_sizes,padding=self._sp_size()>1)
        args = dict(video_xts=history,video_context=history,video_eps=eps,video_timesteps=times,cache=cache)
        self._ref_forward(backbone,inputs,selection=('static',-1),store_kv=True,**args)
        for k in range(len(inputs.layouts[0].chunks)-1):
            spans = [self._chunk_slice(l.chunks[k],'video') for l in inputs.layouts]
            xts = self.schedule.forward([x[s] for x,s in zip(clean,spans)],[n[s] for n,s in zip(noise,spans)],t_s)
            cache.begin_chunk(k)
            self._ref_forward(backbone,inputs,selection=('condition',k),store_kv=True,**args)
            for step in range(rf['steps']):
                for h,x,s,time,g in zip(history,xts,spans,times,grid):
                    h[s],time[k] = x,g[step]
                prediction,_ = self._ref_forward(backbone,inputs,selection=('noise',k),**args)
                xts = self._resampler.step_to(pred=prediction,x_t=xts,t=grid[:,step],s=grid[:,step+1])
            for h,x,s in zip(history,xts,spans):
                h[s] = x
            # Noisy KV never becomes history; commit the resampled chunk once.
            self._ref_forward(backbone,inputs,selection=('history',k),store_kv=True,**args)
        meter = get_running_average_meter()
        for h,x,l,t in zip(history,clean,inputs.layouts,t_s):
            stop = l.chunks[-1].video_start
            meter.put_scalar('running/resampling/t_s',t)
            meter.put_scalar('running/resampling/history_mse',(h[:,:stop]-x[:,:stop]).square().mean() if stop else None)
        return history


class MiniMaxH3RefTSCD(_RefTraining, CausalMiniMaxH3TSCD):
    """Frozen causal TF teacher, EMA target and a few-step causal student."""


class MiniMaxH3RefDMD(_RefTraining, CausalMiniMaxH3DMD):
    """Causal rollout generator; full-context SFT real/fake score models."""


class MiniMaxH3RefSFT(_RefTraining, CausalMiniMaxH3TF):
    """Native bidirectional Ref2VA FM, supervising target RGB only."""
    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def sample_timesteps(self, ctx):
        rng,inputs = ctx['rng'],ctx['inputs']
        with local_seed(rng.seed % 2**31):
            pair = self.training_timesteps.sample_pair((inputs.batch_size,),inputs.seqlens,get_device())
        rng.seed = yield_seed(rng.seed)
        ctx['train_timesteps'] = tuple(t.float() for t in pair)
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def add_noise(self, ctx):
        noises = self._sample_noises(ctx['inputs'],ctx['rng'])
        ctx['noisy_latents'] = tuple(self.schedule.forward(x,n,t) for x,n,t in
                                     zip(ctx['clean_latents'],noises,ctx['train_timesteps']))
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def forward(self, ctx):
        ctx['pred'] = self._guidance_forward(ctx,self._bidirectional_forward,
            video_xts=ctx['noisy_latents'][0],audio_xts=ctx['noisy_latents'][1],
            video_timesteps=ctx['train_timesteps'][0],audio_timesteps=ctx['train_timesteps'][1])
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def compute_loss(self, ctx):
        objective = self.config.meta_model.get('objective','velocity')
        if objective not in ('velocity','x0'):
            raise ValueError('SFT objective must be velocity or x0')
        terms = []
        for pred,target,t in zip(ctx['pred'][0],ctx['clean_latents'][0],ctx['train_timesteps'][0]):
            loss = (pred.float()-target.float()).square()
            # H3 v=x0-eps, wrapper x0=xt+sigma*v. Do not silently replace
            # velocity matching by the timestep-weighted x0 objective.
            if objective == 'velocity':
                if t.item() <= 0:
                    raise ValueError('velocity SFT requires strictly positive sigma')
                loss = loss/t.square()
            if hasattr(self.training_timesteps, 'loss_weight'):
                loss = loss * self.training_timesteps.loss_weight(t)
            terms.append(loss)
        ctx['loss'] = self._loss_fn(terms,ctx['pred'][1],inputs=ctx['inputs'],key_prefix='sft_losses')
        return ctx
