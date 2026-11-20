# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import numpy as np
import torch
from msgspec.structs import replace
from sglang.multimodal_gen.configs.sample.minimax_h3_causal import H3FixedHistoryConfig
from sglang.multimodal_gen.runtime.layers.kvcache.minimax_h3_causal import (
    H3CausalForward,
    H3CausalFusedForward,
    H3CausalKVCache,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_geometry import (
    _axis_from_sqrt_area,
    _video_t_grid,
    _video_t_span,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpatchify_video_tokens,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3_causal import (
        MiniMaxH3CausalDiTModel,
    )


class H3LatentRange(msgspec.Struct, frozen=True):
    start: int
    stop: int

    @property
    def count(self) -> int:
        return self.stop - self.start


class H3WindowPlan(msgspec.Struct, frozen=True):
    index: int
    bootstrap: bool
    sink: H3LatentRange
    dynamic_recent: H3LatentRange
    current: H3LatentRange
    eos: bool


class H3WindowPlanner:
    def __init__(self, config: H3FixedHistoryConfig, *, frame_aligned_rope: bool = False):
        self.config = config
        self.frame_aligned_rope = frame_aligned_rope
        self.position = self.index = self.phase = self.available = 0
        self.nominal_stop = config.bootstrap_latents
        self.ended = False

    def ready(self, available: int, *, eos: bool = False) -> tuple[H3WindowPlan, ...]:
        if self.ended:
            raise ValueError("source received after EOS")
        if (
            type(available) is not int
            or available < self.available
            or type(eos) is not bool
        ):
            raise ValueError("availability must be monotonic and EOS must be boolean")
        self.available = available
        result = []
        while self.position < available:
            stop = next(
                (
                    b
                    for b in self.config.segment_boundaries
                    if self.position < b < self.nominal_stop
                ),
                self.nominal_stop,
            )
            if stop > available:
                if not eos:
                    break
                stop = available
            start = self.position
            sink_stop = min(self.config.sink_latents, start)
            result.append(
                H3WindowPlan(
                    self.index,
                    start == 0,
                    H3LatentRange(0, sink_stop),
                    H3LatentRange(
                        max(sink_stop, start - self.config.recent_latents), start
                    ),
                    H3LatentRange(start, stop),
                    eos and stop == available,
                )
            )
            self.position, self.index = stop, self.index + 1
            if stop == self.nominal_stop:
                schedule = self.config.current_latent_schedule
                self.nominal_stop += schedule[self.phase % len(schedule)]
                self.phase += 1
            elif self.frame_aligned_rope and stop == self.config.rope_reference_latents // 5 * 5:
                self.nominal_stop = stop + self.config.current_latent_schedule[self.phase % len(self.config.current_latent_schedule)]
        self.ended = eos
        return tuple(result)


def _latent(value: torch.Tensor, name: str, count: int | None = None) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 4
        or not torch.is_floating_point(value)
        or min(value.shape) <= 0
        or value.shape[2] % 2
        or value.shape[3] % 2
        or (count is not None and value.shape[1] != count)
    ):
        raise ValueError(
            f"{name} must be floating CTHW with even H/W and the expected T"
        )


class H3StaticCondition(msgspec.Struct, frozen=True):
    prompt_embeds: torch.Tensor
    token_tags: torch.Tensor
    first_frame: torch.Tensor
    first_frame_noise: torch.Tensor

    def __post_init__(self):
        _latent(self.first_frame, "first_frame", 1)
        _latent(self.first_frame_noise, "first_frame_noise", 1)
        if self.first_frame.shape != self.first_frame_noise.shape:
            raise ValueError("first_frame and noise shapes must match")
        if (
            self.prompt_embeds.ndim != 2
            or min(self.prompt_embeds.shape) <= 0
            or not torch.is_floating_point(self.prompt_embeds)
            or self.token_tags.shape != (self.prompt_embeds.shape[0],)
            or self.token_tags.dtype != torch.long
            or not bool(((self.token_tags == 0) | (self.token_tags == 1)).all())
        ):
            raise ValueError(
                "static features require [L,D] Qwen rows and [L] text/image tags"
            )


class H3SourceLatents(msgspec.Struct, frozen=True):
    references: tuple[torch.Tensor, ...]
    reference_noise: tuple[torch.Tensor, ...]
    target_noise: torch.Tensor
    clean_noise: torch.Tensor

    def validate(self, count: int) -> None:
        if not self.references or len(self.references) != len(self.reference_noise):
            raise ValueError("every proxy role requires matching reference noise")
        for name, value in (
            ("target_noise", self.target_noise),
            ("clean_noise", self.clean_noise),
            *(("reference", value) for value in self.references),
            *(("reference_noise", value) for value in self.reference_noise),
        ):
            _latent(value, name, count)
        if self.target_noise.shape != self.clean_noise.shape or any(
            value.shape != noise.shape
            for value, noise in zip(self.references, self.reference_noise)
        ):
            raise ValueError(
                "each latent and its clean-conditioning noise must have matching shapes"
            )


class H3GeneratedSpan(msgspec.Struct, frozen=True):
    video: torch.Tensor
    plan: H3WindowPlan


class _Frame(msgspec.Struct, frozen=True):
    position: int
    references: tuple[torch.Tensor, ...]
    reference_noise: tuple[torch.Tensor, ...]
    noise: torch.Tensor
    clean_noise: torch.Tensor
    video: torch.Tensor | None = None


def h3_absolute_video_time(positions: torch.Tensor, origin: float) -> torch.Tensor:
    positions = positions.cpu()
    return (
        origin
        + (positions // 5).double() * _video_t_span(5)
        + _video_t_grid(5, 0.0)[positions % 5]
    )


def h3_cwm_static_shift(plan: H3WindowPlan, reference_latents: int) -> float:
    if plan.eos and plan.current.stop <= reference_latents:
        return 0.0
    last_full_chunk = reference_latents // 5 - 1
    return max(0, plan.current.start // 5 - last_full_chunk) * _video_t_span(5)


def h3_fixed_euler_step(state, velocity, sigma, next_sigma):
    h3_time = 1.0 - sigma.float()
    packed_state = state.to(torch.bfloat16).float()
    x0 = packed_state + (1 - h3_time.expand_as(packed_state)) * velocity
    t, s = sigma.reshape((1,) * state.ndim), next_sigma.reshape((1,) * state.ndim)
    epsilon = (state - (1 - t) * x0) / t
    return ((1 - s) * x0 + s * epsilon).to(state.dtype)


class H3FixedHistorySession:
    def __init__(
        self,
        model: MiniMaxH3CausalDiTModel,
        static: H3StaticCondition,
        *,
        recipe: dict,
        fuse_first_denoise: bool = False,
    ):
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_ulysses_ctx,
        )

        self.model = model
        if type(fuse_first_denoise) is not bool:
            raise ValueError("fuse_first_denoise must be a boolean")
        self.fuse_first_denoise = fuse_first_denoise
        self.config = H3FixedHistoryConfig.from_recipe(recipe)
        self.frame_aligned_rope = recipe.get("position_policy") == "cwm_frame_aligned_v1"
        self.device = next(model.parameters()).device
        self.ulysses_world_size = get_ulysses_ctx()[0]
        if model.training or tuple(model.arch.patch_size) != (1, 2, 2):
            raise ValueError("fixed H3 requires an eval model with patch size (1,2,2)")
        if (
            static.prompt_embeds.shape[1] != model.arch.text_dim
            or static.first_frame.shape[0] != model.arch.latents_dim
        ):
            raise ValueError(
                "static feature dimensions do not match the native H3 model"
            )
        self.static = H3StaticCondition(
            **{
                name: getattr(static, name).detach().to(self.device).clone()
                for name in (
                    "prompt_embeds",
                    "token_tags",
                    "first_frame",
                    "first_frame_noise",
                )
            }
        )
        grid = np.arange(1.0, 0.0, -1.0 / self.config.num_sampling_steps)
        grid = self.config.shift * grid / (1 + (self.config.shift - 1) * grid)
        self.sigmas = torch.cat(
            (torch.from_numpy(grid), torch.zeros(1, dtype=torch.float64))
        ).to(self.device)
        self.reset()

    @torch.no_grad()
    def reset(self) -> None:
        self._prepare_adaln_plans()
        self.planner = H3WindowPlanner(self.config, frame_aligned_rope=self.frame_aligned_rope)
        self.cache = H3CausalKVCache(len(self.model.blocks))
        self.static_rope_shift = 0.0
        self.frames: dict[int, _Frame] = {}
        self.pending_history: tuple[_Frame, ...] = ()
        self.available = 0
        self.source_shapes = None
        self.state = "open"

    def _prepare_adaln_plans(self) -> None:
        cache = self.model.adaln_cache
        if cache is None:
            self._adaln_slots = None
            self._adaln_generation = None
            return
        clean = torch.tensor([0.999], dtype=torch.float32, device=self.device)
        denoise = [(1.0 - sigma.float()).reshape(1) for sigma in self.sigmas[:-1]]
        plans = [clean, *(torch.unique(torch.cat((time, clean))) for time in denoise)]
        denoise_start = len(plans)
        if self.ulysses_world_size == 1:
            plans.extend(denoise)
        fused_indices = frozenset()
        if self.fuse_first_denoise:
            fused_indices = frozenset({len(plans)})
            plans.append(plans[1])
        slots = self.model.prepare_adaln_plans(plans, fused_plan_indices=fused_indices)
        steps = len(denoise)
        self._adaln_slots = (
            slots[0],
            tuple(slots[1 : 1 + steps].unbind()),
            tuple(slots[denoise_start : denoise_start + steps].unbind())
            if self.ulysses_world_size == 1
            else tuple(slots[1 : 1 + steps].unbind()),
            slots[-1] if fused_indices else None,
        )
        self._adaln_generation = cache.generation

    def append(self, source: H3SourceLatents) -> tuple[H3GeneratedSpan, ...]:
        return self._push(source, eos=False)

    def finish(self, source: H3SourceLatents) -> tuple[H3GeneratedSpan, ...]:
        """End the stream without materializing clean KV for a future window."""
        return self._push(source, eos=True)

    @torch.no_grad()
    def _push(self, source: H3SourceLatents, *, eos: bool):
        if self.state != "open":
            raise RuntimeError(
                f"cannot submit source to {self.state} H3 session; reset first"
            )
        count = 2 if eos else 5
        source.validate(count)
        values = (*source.references, source.target_noise)
        shapes = tuple(
            (value.shape[0], value.shape[2], value.shape[3]) for value in values
        )
        if any(value.shape[0] != self.model.arch.latents_dim for value in values):
            raise ValueError("source latent channels do not match native H3")
        if self.source_shapes is not None and self.source_shapes != shapes:
            raise ValueError(
                "proxy role count and latent resolutions must remain fixed"
            )
        refs, eps = (
            tuple(v.detach().to(self.device).clone() for v in source.references),
            tuple(v.detach().to(self.device).clone() for v in source.reference_noise),
        )
        noise, clean_noise = (
            v.detach().to(self.device).clone()
            for v in (source.target_noise, source.clean_noise)
        )
        try:
            self.source_shapes = shapes
            for i in range(count):
                position = self.available + i
                self.frames[position] = _Frame(
                    position,
                    tuple(v[:, i : i + 1] for v in refs),
                    tuple(v[:, i : i + 1] for v in eps),
                    noise[:, i : i + 1],
                    clean_noise[:, i : i + 1],
                )
            self.available += count
            outputs = []
            for plan in self.planner.ready(self.available, eos=eos):
                outputs.append(self._generate(plan))
                recent_start = max(
                    self.config.sink_latents,
                    plan.current.stop - self.config.recent_latents,
                )
                self.frames = {
                    p: frame
                    for p, frame in self.frames.items()
                    if p < self.config.sink_latents or p >= recent_start
                }
            if eos:
                self.state = "ended"
            return tuple(outputs)
        except BaseException:
            self.state = "failed"
            raise

    def _range(self, value: H3LatentRange) -> list[_Frame]:
        return [self.frames[p] for p in range(value.start, value.stop)]

    def _generate(self, plan: H3WindowPlan) -> H3GeneratedSpan:
        cache = self.model.adaln_cache
        if cache is not None and self._adaln_generation != cache.generation:
            self._prepare_adaln_plans()
        current = self._range(plan.current)
        fuse = not plan.bootstrap and len(self.pending_history) == len(current) == 5
        if plan.bootstrap:
            cache = H3CausalKVCache(len(self.model.blocks))
        else:
            cache = self.cache.fork()
            shift = h3_cwm_static_shift(plan, self.config.rope_reference_latents) if self.frame_aligned_rope else 0.0
            if not cache.owners:
                self.static_rope_shift = shift
                self._forward(
                    self._range(plan.sink), cache, static=True, prefix=True, write=True
                )
                history = self._range(plan.dynamic_recent)
                if history:
                    self._forward(history, cache, write=True)
            elif shift != self.static_rope_shift:
                self.static_rope_shift = shift
                prefix = H3CausalKVCache(len(self.model.blocks))
                self._forward(self._range(plan.sink), prefix, static=True, prefix=True, write=True)
                cache.replace_prefix(prefix)
            if not fuse:
                if self.pending_history:
                    self._forward(
                        self.pending_history,
                        cache,
                        conditions=False,
                        write=True,
                        recent_start=plan.dynamic_recent.start,
                    )
                else:
                    cache.retain(plan.dynamic_recent.start)
                self._forward(current, cache, video=False, write=True)
        xt = torch.cat([frame.noise for frame in current], dim=1)
        for step_index, (sigma, next_sigma) in enumerate(
            zip(self.sigmas[:-1], self.sigmas[1:])
        ):
            live = [
                replace(frame, video=xt[:, i : i + 1])
                for i, frame in enumerate(current)
            ]
            velocity = self._forward(
                live,
                cache,
                static=plan.bootstrap,
                conditions=plan.bootstrap,
                denoise=True,
                sigma=sigma,
                step_index=step_index,
                history=self.pending_history if fuse and step_index == 0 else (),
                recent_start=plan.dynamic_recent.start,
            )
            xt = h3_fixed_euler_step(xt, velocity, sigma, next_sigma)
        if not bool(torch.isfinite(xt).all()):
            raise RuntimeError("native H3 produced nonfinite generated latents")
        clean = [
            replace(frame, video=xt[:, i : i + 1].detach().clone())
            for i, frame in enumerate(current)
        ]
        if not plan.bootstrap:
            if self.fuse_first_denoise and not plan.eos:
                self.pending_history = tuple(clean)
            else:
                # finish() rejects further input; no consumer needs terminal clean KV.
                if not plan.eos:
                    self._forward(clean, cache, conditions=False, write=True)
                self.pending_history = ()
            self.cache = cache
        self.frames.update((frame.position, frame) for frame in clean)
        return H3GeneratedSpan(xt, plan)

    def _forward(
        self,
        frames,
        cache,
        *,
        static=False,
        conditions=True,
        video=True,
        prefix=False,
        write=False,
        denoise=False,
        sigma=None,
        step_index=0,
        history=(),
        recent_start=None,
    ):
        packed, owners, target_shape = self._pack(
            frames,
            static=static,
            conditions=conditions,
            video=video,
            prefix=prefix,
            denoise=denoise,
            sigma=sigma,
            history=history,
        )
        if history:
            history_stop = sum(
                f.noise.shape[2] * f.noise.shape[3] // 4 for f in history
            )
            condition_rows = sum(
                value.shape[2] * value.shape[3] // 4
                for frame in frames
                for value in frame.references
            )
            causal_context = H3CausalFusedForward(
                cache, owners, history_stop, history_stop + condition_rows, recent_start
            )
        else:
            causal_context = H3CausalForward(
                cache, owners, write, recent_start if write else None
            )
        if self._adaln_slots is not None:
            clean, mixed, denoise_only, fused = self._adaln_slots
            slot = clean
            if denoise:
                has_clean = static or conditions or self.ulysses_world_size > 1
                slot = (mixed if has_clean else denoise_only)[step_index]
            packed["adaln_cache_slot"] = fused if history else slot
        with set_forward_context(current_timestep=step_index, attn_metadata=None):
            result, _ = self.model(causal_context=causal_context, **packed)
        if not denoise:
            return None
        return minimax_h3_unpatchify_video_tokens(
            result,
            latent_shape=target_shape,
            patch_size=(1, 2, 2),
        )[0]

    def _pack(
        self, frames, *, static, conditions, video, prefix, denoise, sigma, history=()
    ):
        rows, positions, tags, times, owners = [], [], [], [], []
        image_rows, text_rows, outputs = [], [], []
        cursor = 0
        text_len = self.static.prompt_embeds.shape[0]
        span = _video_t_span(self.config.rope_reference_latents)
        reference_count = (
            len(frames[0].references) if frames else len(self.source_shapes) - 1
        )

        def add(value, epsilon, origin, absolute, owner, *, target=False):
            nonlocal cursor
            value = minimax_h3_patchify_video_latent(
                value[None], patch_size=(1, 2, 2)
            ).to(torch.bfloat16)
            epsilon = minimax_h3_patchify_video_latent(
                epsilon[None], patch_size=(1, 2, 2)
            ).to(torch.bfloat16)
            n = value.shape[0]
            time = (
                1.0 - sigma.float()
                if target
                else value.new_tensor(0.999, dtype=torch.float32)
            )
            shaped_time = time.expand(n, 1)
            rows.append(
                value.float()
                if target
                else shaped_time * value + (1 - shaped_time) * epsilon
            )
            times.append(time.expand(n))
            tags.append(torch.zeros(n, dtype=torch.long, device=self.device))
            ids = torch.arange(cursor, cursor + n, device=self.device)
            image_rows.append(ids)
            if target:
                outputs.append(ids)
            owners.extend([owner] * n)
            height, width = absolute[1:]
            hh, ww = torch.meshgrid(
                _axis_from_sqrt_area(height, 2, (height * width) ** 0.5),
                _axis_from_sqrt_area(width, 2, (height * width) ** 0.5),
                indexing="ij",
            )
            pos = torch.empty(n, 3, dtype=torch.float64)
            pos[:, 0] = h3_absolute_video_time(
                torch.tensor([absolute[0]]), float(origin)
            )[0]
            pos[:, 1], pos[:, 2] = hh.flatten(), ww.flatten()
            positions.append(pos)
            cursor += n

        if static:
            rows.append(
                torch.zeros(
                    text_len, self.model.arch.latents_dim * 4, device=self.device
                )
            )
            times.append(torch.full((text_len,), 0.999, device=self.device))
            tags.append(self.static.token_tags)
            text_rows.append(torch.arange(text_len, device=self.device))
            pos = torch.zeros(text_len, 3, dtype=torch.float64)
            pos[:, 0] = torch.arange(text_len, dtype=torch.float64) + self.static_rope_shift
            positions.append(pos)
            owners.extend([-1] * text_len)
            cursor += text_len
            picture = self.static.first_frame
            add(
                picture,
                self.static.first_frame_noise,
                text_len + self.static_rope_shift,
                (0, *picture.shape[2:]),
                -1,
            )

        def add_frames(values, *, conditions, video, denoise=False):
            for frame in values:
                owner = -1 if prefix else frame.position
                if conditions:
                    for role, (value, epsilon) in enumerate(
                        zip(frame.references, frame.reference_noise)
                    ):
                        add(
                            value,
                            epsilon,
                            text_len if self.frame_aligned_rope else text_len + 1.0 + role * span,
                            (frame.position, *value.shape[2:]),
                            owner,
                        )
                if video:
                    value = frame.video if frame.video is not None else frame.noise
                    add(
                        value,
                        frame.clean_noise,
                        text_len if self.frame_aligned_rope else text_len + 1.0 + reference_count * span,
                        (frame.position, *value.shape[2:]),
                        owner,
                        target=denoise,
                    )

        if history:
            add_frames(history, conditions=False, video=True)
            add_frames(frames, conditions=True, video=False)
            add_frames(frames, conditions=False, video=True, denoise=True)
        else:
            add_frames(frames, conditions=conditions, video=video, denoise=denoise)
        used = cursor
        world_size = self.ulysses_world_size
        pad = world_size - used % world_size if world_size > 1 else 0
        if pad:
            rows.append(rows[0].new_zeros(pad, rows[0].shape[1]))
            times.append(torch.full((pad,), 0.999, device=self.device))
            tags.append(torch.full((pad,), -1, device=self.device, dtype=torch.long))
            positions.append(torch.zeros(pad, 3, dtype=torch.float64))
        cat_ids = lambda parts: (
            torch.cat(parts)
            if parts
            else torch.empty(0, device=self.device, dtype=torch.long)
        )
        unique, inverse = torch.unique(torch.cat(times), return_inverse=True)
        cu = [0, used] + ([used + pad] if pad else [])
        packed = dict(
            x=torch.cat(rows)[None],
            audio_x=rows[0].new_zeros(1, used + pad, self.model.arch.audio_latents_dim),
            img_position_ids=torch.cat(positions)[None].to(self.device),
            unique_timesteps=unique,
            inverse_indices=inverse,
            token_tags=torch.cat(tags),
            prompt_embeds=self.static.prompt_embeds
            if static
            else self.static.prompt_embeds[:0],
            img_pos_info={"position_ids": cat_ids(image_rows)},
            text_pos_info={"position_ids": cat_ids(text_rows)},
            audio_pos_info={"position_ids": cat_ids([])},
            img_pos_for_infer_output_info={"position_ids": cat_ids(outputs)},
            update_mask=torch.ones(
                sum(ids.numel() for ids in outputs),
                device=self.device,
                dtype=torch.bool,
            ),
            packed_seq_params={
                "cu_seqlens_q": torch.tensor(cu, device=self.device, dtype=torch.int32),
                "cu_seqlens_q_host": tuple(cu),
                "max_seqlen_q": used,
            },
            refiner_packed_seq_params={
                "cu_seqlens_q": torch.tensor(
                    [0, text_len if static else 0],
                    device=self.device,
                    dtype=torch.int32,
                ),
                "max_seqlen_q": text_len if static else 0,
            },
        )
        shape = (
            (
                len(frames),
                frames[0].noise.shape[2] // 2,
                frames[0].noise.shape[3] // 2,
                self.model.arch.latents_dim,
            )
            if frames
            else None
        )
        return packed, tuple(owners), shape
