"""Shared causal geometry, independent of training algorithms."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CausalChunk:
    """One time chunk of a sample, as laid out by the causal dataset.

    ``noise_start``/``clean_start`` are sample-local row offsets of the split
    that carries the noisy copy and (for every chunk but the last) the clean
    copy. The two splits are row-identical in structure: audio rows first, then
    video rows.
    """

    noise_start: int
    clean_start: int | None
    audio_rows: int
    video_rows: int
    video_start: int
    video_stop: int
    audio_start: int
    audio_stop: int

    @property
    def rows(self) -> int:
        return self.audio_rows + self.video_rows


@dataclass(frozen=True)
class CausalLayout:
    """Everything about one sample's geometry, derived from the dataset batch."""

    text_len: int
    chunks: tuple[CausalChunk, ...]
    # Chunk-order audio row k lands at native row audio_row_perm[k]. Video needs
    # no such map: concatenating chunks in order already reproduces the native
    # frame-major order, whereas audio chunks are [L_c | R_c] and native is
    # [L(all) | R(all)].
    audio_row_perm: torch.Tensor
    latent_shape: tuple[int, int, int, int]
    audio_shape: tuple[int, int, int]

    @property
    def video_patch_grid(self) -> tuple[int, int, int]:
        _, latent_t, latent_h, latent_w = self.latent_shape
        return latent_t, latent_h // 2, latent_w // 2
