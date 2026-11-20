# SPDX-License-Identifier: Apache-2.0
"""Causal decoder participation and output ownership across the H3 rank group."""

from __future__ import annotations

import msgspec
import torch


class H3DecodedBatch(msgspec.Struct, frozen=True):
    frame_count: int
    # GPU THWC uint8 on the output rank only.
    rgb: torch.Tensor | None


class H3RealtimeCodec:
    """``decoder`` is None on ranks outside the decoder group."""

    def __init__(self, decoder, world_group, *, output_rank, device, decoder_group=None):
        self.decoder, self.world_group = decoder, world_group
        self.output_rank = output_rank
        self.decoder_group = decoder_group
        self.is_output = world_group.rank == output_rank
        if self.is_output and decoder is None:
            raise ValueError("H3 output rank must participate in decoding")
        self._count = torch.zeros(1, dtype=torch.int64, device=device)

    def begin_session(self) -> None:
        if self.decoder is not None:
            self.decoder.begin_session()

    def decode_local(self, latent, *, eos: bool) -> H3DecodedBatch:
        rgb = None if self.decoder is None else self.decoder.decode(latent, eos=eos)
        local_count = 0 if rgb is None else len(rgb)
        return H3DecodedBatch(local_count, rgb if self.is_output else None)

    def publish_counts(self, local_counts, *, span_count: int) -> tuple[int, ...]:
        if len(local_counts) != span_count:
            raise ValueError("decoder count vector does not match generated spans")
        if not span_count:
            return ()
        counts = self._count.new_tensor(
            local_counts if self.is_output else [0] * span_count
        )
        published = tuple(
            int(count)
            for count in self.world_group.broadcast(
                counts, src=self.world_group.ranks.index(self.output_rank)
            ).tolist()
        )
        if self.decoder is not None and tuple(local_counts) != published:
            raise RuntimeError("decoder participants disagree on output frame count")
        return published

    def decode(self, latent, *, eos: bool) -> H3DecodedBatch:
        decoded = self.decode_local(latent, eos=eos)
        counts = self.publish_counts((decoded.frame_count,), span_count=1)
        return H3DecodedBatch(counts[0], decoded.rgb)
