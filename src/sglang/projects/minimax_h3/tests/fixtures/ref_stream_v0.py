"""Frozen 3fe5602 causal stream/utility bodies, AST-loaded only by tests.

External names are supplied by ref_inference_support. Do not use as runtime code.
"""
class CausalMiniMaxH3Base:

    @classmethod
    def _to_device(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(get_device())
        if isinstance(value, list):
            return [cls._to_device(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._to_device(item) for key, item in value.items()}
        return value

    @staticmethod
    def _video_rows(latent: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        return minimax_h3_patchify_video_latent(latent[:, start:stop].unsqueeze(0), patch_size=_PATCH_SIZE)

    @staticmethod
    def _video_from_rows(rows: torch.Tensor, layout: _Layout) -> torch.Tensor:
        latent_t, patch_h, patch_w = layout.video_patch_grid
        return minimax_h3_unpatchify_video_tokens(rows, latent_shape=(latent_t, patch_h, patch_w, layout.latent_shape[0]), patch_size=_PATCH_SIZE)[0]

    @staticmethod
    def _num_layers(model: Any) -> int:
        assert hasattr(model, 'dit'), 'MiniMax-H3 model nodes must be MiniMaxH3X0Model wrappers; the raw DiT returns velocity and would silently flip the x0 sign'
        return len(model.dit.blocks)

    @staticmethod
    def _sp_size() -> int:
        if is_unified_parallel_initialized():
            return get_unified_parallel_world_size()
        return 1

    @classmethod
    def _pad_rows(cls, rows: int) -> int:
        alignment = cls._sp_size()
        if alignment <= 1:
            return 0
        remainder = rows % alignment
        return alignment - remainder if remainder else alignment

    @staticmethod
    def _generator(rng: Any) -> torch.Generator:
        device = get_device()
        return rng.torch_cuda_generator if device.type == 'cuda' else rng.torch_generator

class _RefTraining:

    @torch.no_grad()
    def _rollout_grid(self, inputs, diffusion=None):
        timesteps = (diffusion or {}).get('sampling_timesteps', self.sampling_timesteps)
        if getattr(timesteps, 'dynamic_shift', False):
            raise ValueError('CWM rollout requires length-independent sampling timesteps')
        timesteps.set_timesteps(seqlen=inputs.seqlens, device=get_device())
        grid = timesteps.timesteps
        if grid.ndim != 1:
            raise ValueError('CWM rollout requires a static sampling grid')
        return grid

    def _rollout_chunk(self, backbone, inputs, k, args, grid, streams, trajectory=(), trajectory_x0_chunks=None, *, diffusion=None, profiler=None):
        timesteps = (diffusion or {}).get('sampling_timesteps', self.sampling_timesteps)
        sampler = (diffusion or {}).get('sampler', self.sampler)
        result, times = (args['video_xts'], args['video_timesteps'])
        audio = [v.new_empty(l.audio_shape) for v, l in zip(result, inputs.layouts)]
        rngs, xts = ([], [])
        for i, (sample, layout, rng) in enumerate(zip(inputs.samples, inputs.layouts, streams)):
            c = layout.chunks[k]
            r = rng.fork('cwm_rgb', i, sample['metadata'].get('stream_chunk_start', 0) + k)
            rngs.append(r)
            shape = (layout.latent_shape[0], c.video_stop - c.video_start, *layout.latent_shape[2:])
            xts.append(torch.randn(shape, device=get_device(), generator=self._generator(r)))
            args['video_eps'][i][:, c.video_start:c.video_stop] = torch.randn(shape, device=get_device(), generator=self._generator(r))
        with profiler.stage('condition_prefill') if profiler else nullcontext():
            self._sampling_forward(backbone, inputs, selection=('condition', k), store_kv=True, **args)
        with profiler.stage('denoise') if profiler else nullcontext():
            for step, t in enumerate(grid):
                for i, l in enumerate(inputs.layouts):
                    c = l.chunks[k]
                    result[i][:, c.video_start:c.video_stop] = xts[i]
                    times[i][k] = t
                    if trajectory:
                        trajectory[step][0][i][:, c.video_start:c.video_stop] = xts[i]
                prediction, _ = self._sampling_forward(backbone, inputs, selection=('noise', k), **args)
                if trajectory_x0_chunks is not None:
                    trajectory_x0_chunks.append(([p.detach().clone() for p in prediction], audio))
                ts = t.expand(inputs.batch_size)
                xts = sampler.step_to(pred=prediction, x_t=xts, t=ts, s=timesteps.get_next_timesteps(ts), rng=rngs, seqlens=inputs.seqlens)
        for i, l in enumerate(inputs.layouts):
            c = l.chunks[k]
            result[i][:, c.video_start:c.video_stop] = xts[i]
            times[i][k] = 0
        with profiler.stage('clean_commit') if profiler else nullcontext():
            self._sampling_forward(backbone, inputs, selection=('history', k), store_kv=True, **args)

    @torch.no_grad()
    def stream_latents(self, backbone, static_condition, chunks, *, seed=2026, condition_seed=None, video_shape=None, first_frame=None, conditioning_mode=LEGACY_CONDITIONING_MODE, profiler=None):
        from projects.minimax_h3.data.ref_latent import SCHEMA
        mode = get_conditioning_mode({'conditioning_mode': conditioning_mode})
        if isinstance(self, MiniMaxH3RefSFT) or backbone.training:
            raise ValueError('stream_latents requires a causal eval-mode backbone')
        cache, weak_cache, weak_static, signature, ended = (None, None, None, None, False)
        streams = [RandomState(seed)]
        diffusion = getattr(self, 'validation_diffusion', None)
        picture = [] if first_frame is None else [dict(kind='image', role='first_frame', video=first_frame)]
        for k, item in enumerate(chunks):
            if ended:
                raise ValueError('received a chunk after EOS')
            if not item['references']:
                raise ValueError('at least one semantic reference is required')
            if mode == STATIC_CONDITIONING_MODE and 'condition' in item:
                raise ValueError('static-image streaming chunks must omit Qwen condition')
            refs = picture + list(item['references'])
            current = [(r['role'], r['video'].shape[0], *r['video'].shape[2:]) for r in refs]
            if signature is None:
                signature = current
            if current != signature:
                raise ValueError('reference roles or resolution changed within a stream')
            ended = bool(item.get('eos', False))
            frames, latents = (5, 2) if ended else (17, 5)
            c, h, w = video_shape or (refs[0]['video'].shape[0], *refs[0]['video'].shape[2:])
            sample = dict(video=torch.zeros(c, latents, h, w), audio=torch.empty(2, 32, 0), references=refs, guidance_conditions=item.get('guidance_conditions', {}), teacher_condition=static_condition, student_condition=dict(static=static_condition, chunks=[] if mode == STATIC_CONDITIONING_MODE else [item['condition']]), metadata=dict(schema=SCHEMA, fps=24, normalized=True, num_frames=frames, conditioning_mode=mode, condition_seed=seed if condition_seed is None else condition_seed, stream_chunk_start=k, chunk_raw_ranges=[[0, frames]], chunk_latent_ranges=[[0, latents]]))
            with profiler.stage('stream_inputs') if profiler else nullcontext():
                inputs = self._build_inputs({'ref_samples': [sample]}, streaming=True)
            weak = self._sampling_inputs(inputs)
            if weak is not None:
                current = weak.samples[0]['student_condition']['static']
                if weak_static is not None and any((current[key].dtype != weak_static[key].dtype or not torch.equal(current[key], weak_static[key]) for key in ('prompt_embeds', 'text_token_tags'))):
                    raise ValueError('weak static Qwen condition changed within a stream')
                if weak_static is None:
                    weak_static = {key: current[key].detach().clone() for key in ('prompt_embeds', 'text_token_tags')}
            result = torch.zeros_like(inputs.samples[0]['video'])
            args = dict(video_xts=[result], video_context=[result], video_eps=[torch.zeros_like(result)], video_timesteps=[result.new_zeros(1)])
            if cache is None:
                cache = GroupedKVCache(self._num_layers(backbone), 1, windows=inputs.window_sizes, padding=self._sp_size() > 1)
                weak_cache = GroupedKVCache(self._num_layers(backbone), 1, windows=inputs.window_sizes, padding=self._sp_size() > 1) if weak is not None else None
                with profiler.stage('static_prefill') if profiler else nullcontext():
                    self._sampling_forward(backbone, inputs, selection=('static', -1), store_kv=True, cache=cache, weak=weak, weak_cache=weak_cache, **args)
            cache.begin_chunk(k)
            if weak_cache is not None:
                weak_cache.begin_chunk(k)
            args['cache'] = cache
            args.update(weak=weak, weak_cache=weak_cache)
            self._rollout_chunk(backbone, inputs, 0, args, self._rollout_grid(inputs, diffusion), streams, diffusion=diffusion, profiler=profiler)
            del inputs, args, sample, item, refs
            yield result
