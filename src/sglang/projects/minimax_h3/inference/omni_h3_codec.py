"""Native codec participation and output ownership at the rank-group boundary."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NativeDecodeBatch:
    frame_count: int
    rgb: torch.Tensor | None


class NativeH3Codec:
    def __init__(self, local_codec, world_group, *, output_rank, device, decoder_group=None):
        self.local_codec, self.world_group = local_codec, world_group
        self.output_rank = output_rank
        self.decoder_group = decoder_group
        self.is_output = world_group.rank == output_rank
        if self.is_output and local_codec is None:
            raise ValueError("native codec output owner must participate in decoding")
        self._count = torch.zeros(1, dtype=torch.int64, device=device)

    def begin_session(self):
        if self.local_codec is not None:
            self.local_codec.begin_session()

    def decode_local(self, latent, *, eos=False):
        rgb = None if self.local_codec is None else self.local_codec.decode(latent, eos=eos)
        local_count = 0 if rgb is None else len(rgb)
        return NativeDecodeBatch(local_count, rgb if self.is_output else None)

    def publish_counts(self, local_counts, *, span_count):
        if len(local_counts) != span_count:
            raise ValueError("decoder count vector does not match generated spans")
        if not span_count:
            return ()
        counts = self._count.new_tensor(local_counts if self.is_output else [0] * span_count)
        published = tuple(int(count) for count in self.world_group.broadcast(counts,
            src=self.world_group.ranks.index(self.output_rank)).tolist())
        if self.local_codec is not None and tuple(local_counts) != published:
            raise RuntimeError("decoder participants disagree on output frame count")
        return published

    def decode(self, latent, *, eos=False):
        decoded = self.decode_local(latent, eos=eos)
        counts = self.publish_counts((decoded.frame_count,), span_count=1)
        return NativeDecodeBatch(counts[0], decoded.rgb)
