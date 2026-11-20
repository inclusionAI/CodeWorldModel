# SPDX-License-Identifier: Apache-2.0
"""Training timesteps that emit a paired (video, audio) draw for MiniMax-H3.

H3 advances ONE step index through two grids that carry different shifts, so
along its own trajectory ``sigma_video != sigma_audio`` everywhere except the
endpoints. Training has to land on those same pairs. Two independent draws miss
them, and so does a single shared sigma -- the shared sigma is just as far off
the curve as the independent draw is. What lands on it is one draw from the base
distribution pushed through two shifts, which is what ``sample_pair`` does.

Owning the pair here rather than in a meta model is what makes the invariant
structural. A two-node config can only state "these differ in nothing but shift"
by convention, and a convention is exactly what a later edit breaks silently.
Here the audio side IS the video object with one field replaced, so there is no
second set of distribution parameters that can drift.

The ``_sample_raw`` bodies restate the draw from the matching class in
``common/diffusion/timestep/training``. That duplication is deliberate: the
distribution lives inside ``sample()``, which returns an already-shifted value,
and a shifted value cannot be un-shifted once ``postprocess_sample`` has clamped
it. Keep these two lines in step with common if the base classes ever change.
"""

from __future__ import annotations

import copy
import math
from typing import Sequence

import torch

from common.diffusion.timestep.training.logitnormal import LogitNormalTrainingTimesteps
from common.diffusion.timestep.training.uniform import UniformTrainingTimesteps


class _PairedTimestepsMixin:
    """Adds an audio twin that differs from ``self`` in ``shift`` alone."""

    def __init__(self, *, audio_shift: float, **kwargs) -> None:
        super().__init__(**kwargs)
        # dynamic_shift derives the shift from seqlen inside postprocess_sample
        # and ignores self.shift, so the twin would silently be handed the video
        # shift and the pair would collapse onto one sigma.
        assert not self.dynamic_shift, (
            "paired timesteps require a static shift: dynamic_shift computes the "
            "shift from seqlen and would give video and audio the same one"
        )
        # A shallow copy, not a second construction: everything except shift --
        # T, scaling, clamps, and the distribution object itself -- is shared by
        # identity rather than by matching arguments.
        self._audio = copy.copy(self)
        self._audio.shift = float(audio_shift)

    def _sample_raw(self, size: Sequence[int]) -> torch.Tensor:
        raise NotImplementedError

    def sample_pair(
        self,
        size: Sequence[int],
        seqlens: Sequence[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One draw from the base distribution, shifted two ways.

        The caller owns the RNG: seed the global stream around this call the way
        the rest of the phase does, exactly as it would for ``sample``.
        """
        raw = self._sample_raw(size)
        return (
            self.postprocess_sample(raw.clone(), seqlens, device),
            self._audio.postprocess_sample(raw.clone(), seqlens, device),
        )


class PairedUniformTrainingTimesteps(_PairedTimestepsMixin, UniformTrainingTimesteps):
    """Uniform on [0, 1), then each modality's shift."""

    def _sample_raw(self, size: Sequence[int]) -> torch.Tensor:
        return torch.rand(size, dtype=torch.float64)


class PairedLogitNormalTrainingTimesteps(_PairedTimestepsMixin, LogitNormalTrainingTimesteps):
    """sigmoid(N(loc, scale)), then each modality's shift."""

    def _sample_raw(self, size: Sequence[int]) -> torch.Tensor:
        return self.dist.sample(size)[..., 0].to(torch.float64)


class RefSFTTrainingTimesteps:
    """DiffSynth H3's 1000-point training grid and normalized loss weights.

    Preserve the float32 operation order of ``set_timesteps_minimax_h3`` and
    ``set_training_weight``. The pure-noise endpoint has zero loss weight; it
    must not be clamped away. CWM has no audio, so both ABI entries share sigma.
    """

    def __init__(self, T: float = 1.0, shift: float = 12.0):
        if T != 1.0 or not math.isfinite(shift) or shift <= 0:
            raise ValueError("Ref SFT requires T=1.0 and a finite positive shift")
        self.T, self.shift, self.dynamic_shift = T, shift, False
        self.num_train_timesteps = 1000
        base = torch.linspace(1.0, 0.0, 1001, dtype=torch.float32)[:-1]
        self.sigmas = shift * base / (1 + (shift - 1) * base)
        x = self.sigmas * 1000
        y = torch.exp(-2 * ((x - 1000 / 2) / 1000) ** 2)
        self._weight_min = y.min()
        shifted = y - self._weight_min
        self._weight_scale = 1000 / shifted.sum()
        self.weights = shifted * self._weight_scale
        self._device_grids = {}

    def sample(self, size: Sequence[int], seqlens: Sequence[int], device: torch.device):
        # Caller supplies RAVEN's local_seed context, including on resume.
        indices = torch.randint(0, self.num_train_timesteps, size)
        return self.sigmas[indices].to(device)

    def sample_pair(self, size: Sequence[int], seqlens: Sequence[int], device: torch.device):
        sigma = self.sample(size, seqlens, device)
        return sigma, sigma

    def loss_weight(self, sigma: torch.Tensor):
        """Nearest-grid lookup, matching DiffSynth for scalar or batched sigma."""
        key = str(sigma.device)
        if key not in self._device_grids:
            self._device_grids[key] = self.sigmas.to(sigma.device), self.weights.to(sigma.device)
        grid, weights = self._device_grids[key]
        index = (grid - sigma.float().unsqueeze(-1)).abs().argmin(dim=-1)
        return weights[index]

    def continuous_loss_weight(self, sigma: torch.Tensor):
        """The same normalized Gaussian at an unquantized continuous sigma.

        The grid includes sigma=1, so its minimum is exp(-1/2). The identity
        exp(-2*(sigma-1/2)**2)-exp(-1/2)
        = exp(-1/2)*expm1(2*sigma*(1-sigma)) avoids cancellation near zero.
        Normalization remains the mean on this object's original 1000-point
        shifted grid; it is never recomputed from a minibatch or TF draws.
        """
        sigma = sigma.float()
        if not torch.isfinite(sigma).all() or ((sigma < 0) | (sigma > 1)).any():
            raise ValueError('Gaussian loss weights require finite sigma in [0, 1]')
        return (self._weight_min.item() * torch.expm1(2 * sigma * (1 - sigma))
                * self._weight_scale.item())


__all__ = [
    "PairedUniformTrainingTimesteps",
    "PairedLogitNormalTrainingTimesteps",
    "RefSFTTrainingTimesteps",
]
