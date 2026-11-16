# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import msgspec


class H3FixedHistoryConfig(msgspec.Struct, frozen=True):
    rope_reference_latents: int
    num_sampling_steps: int
    shift: float
    sink_latents: int = 2
    recent_latents: int = 10
    bootstrap_latents: int = 9
    current_latent_schedule: tuple[int, ...] = (3, 2)
    segment_boundaries: tuple[int, ...] = ()

    def __post_init__(self):
        for name in (
            "rope_reference_latents",
            "num_sampling_steps",
            "bootstrap_latents",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("sink_latents", "recent_latents"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.sink_latents > self.bootstrap_latents:
            raise ValueError("sink_latents must not exceed bootstrap_latents")
        if (
            isinstance(self.shift, bool)
            or not math.isfinite(self.shift)
            or self.shift <= 0
        ):
            raise ValueError("shift must be finite and positive")
        if not self.current_latent_schedule or any(
            type(n) is not int or n < 1 for n in self.current_latent_schedule
        ):
            raise ValueError("current_latent_schedule requires positive integers")
        if any(
            type(n) is not int or n < self.bootstrap_latents
            for n in self.segment_boundaries
        ) or list(self.segment_boundaries) != sorted(set(self.segment_boundaries)):
            raise ValueError(
                "segment_boundaries must be increasing positions after bootstrap"
            )

    @classmethod
    def from_recipe(cls, recipe: dict) -> H3FixedHistoryConfig:
        if recipe.get('protocol') == 'cwm_causal/v1':
            expected = dict(_FIXED_RECIPE, protocol='cwm_causal/v1', position_policy='cwm_frame_aligned_v1',
                rope_reference_latents=47, sink_latents=1, recent_latents=12, bootstrap_latents=9,
                current_latent_schedule=[5], segment_boundaries=[45, 70], diffusion=_diffusion(4, 12.0))
            if recipe == expected:
                return cls(47, 4, 12., 1, 12, 9, (5,), (45, 70))
            raise ValueError('CWM requires fixed H3 history with aligned47 RoPE')
        if set(recipe) - _RECIPE_FIELDS:
            raise ValueError("unsupported fixed-history recipe fields")
        if (
            any(recipe.get(key) != value for key, value in _FIXED_RECIPE.items())
            or recipe.get("position_policy", "absolute_native_v1")
            != "absolute_native_v1"
        ):
            raise ValueError(
                "native causal H3 requires fixed v2, absolute positions and plain scale 1"
            )
        try:
            diffusion = recipe["diffusion"]
            steps = diffusion["sampling_timesteps"]["args"]
            if diffusion != _diffusion(steps["num_sampling_steps"], steps["shift"]):
                raise ValueError(
                    "native causal H3 requires static trailing timesteps and deterministic Euler"
                )
            return cls(
                rope_reference_latents=recipe["rope_reference_latents"],
                num_sampling_steps=steps["num_sampling_steps"],
                shift=steps["shift"],
                **{
                    key: recipe[key]
                    for key in ("sink_latents", "recent_latents", "bootstrap_latents")
                    if key in recipe
                },
                **{
                    key: tuple(recipe[key])
                    for key in ("current_latent_schedule", "segment_boundaries")
                    if key in recipe
                },
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid fixed-history v2 recipe") from exc


_RECIPE_FIELDS = {
    "protocol",
    "conditioning_mode",
    "position_policy",
    "rope_reference_latents",
    "diffusion",
    "guidance",
    "sink_latents",
    "recent_latents",
    "bootstrap_latents",
    "current_latent_schedule",
    "segment_boundaries",
}
_FIXED_RECIPE = {
    "protocol": "ref_fixed_recent_history/v2",
    "conditioning_mode": "static_image_qwen_proxy_vae_v1",
    "guidance": {"mode": "plain", "drop": "proxy", "scale": 1.0},
}


def _diffusion(num_sampling_steps, shift) -> dict:
    return {
        "schedule": {
            "module": "common.diffusion.schedule.lerp",
            "class_name": "LinearInterpolationSchedule",
            "args": {"T": 1.0, "pred_type": "x_0"},
        },
        "sampling_timesteps": {
            "module": "common.diffusion.timestep.sampling.trailing",
            "class_name": "TrailingSamplingTimesteps",
            "args": {"T": 1.0, "num_sampling_steps": num_sampling_steps, "shift": shift},
        },
        "sampler": {
            "module": "projects.minimax_h3.modeling.sampling",
            "class_name": "EulerSampler",
            "args": {},
        },
    }
