"""Standalone role shards for H/C cache writes and optional first denoise."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HCLayout:
    local_rows: tuple[int, ...]
    input_indexes: torch.Tensor
    to_attention_order: torch.Tensor
    from_attention_order: torch.Tensor

    @classmethod
    def prepare(cls, segments, *, world_size, rank, device):
        roles = segments.hc_rows
        if len(roles) != 2:
            raise ValueError('hc_rows must contain history and condition counts')
        denoise = getattr(segments, 'denoise_rows', 0)
        if denoise < 0:
            raise ValueError('denoise_rows must be nonnegative')
        if denoise:
            roles += (denoise,)
        lengths = segments.query_lens.tolist()
        count = len(roles)
        if min(roles) <= 0 or lengths[:count] != list(roles) or len(lengths) not in (count, 2*count):
            raise ValueError('H/C layout requires live role segments and optional separate padding')
        padding = lengths[count:] if len(lengths) == 2*count else [0]*count
        if world_size < 1 or not 0 <= rank < world_size or min(padding) < 0:
            raise ValueError('invalid H/C parallel layout')
        if world_size > 1 and padding != [world_size-n % world_size for n in roles]:
            raise ValueError('H/C layout requires the standalone padding of each role')
        if any((n+p) % world_size for n, p in zip(roles, padding)):
            raise ValueError('H/C padded roles must divide the parallel world size')
        local_rows = tuple((n+p)//world_size for n, p in zip(roles, padding))
        role_indexes = []
        live_start, pad_start = 0, sum(roles)
        for rows, pad in zip(roles, padding):
            role_indexes.append(list(range(live_start, live_start+rows)) + list(range(pad_start, pad_start+pad)))
            live_start += rows
            pad_start += pad
        rank_order = []
        for r in range(world_size):
            for indexes, rows in zip(role_indexes, local_rows):
                rank_order.extend(indexes[r*rows:(r+1)*rows])
        rank_order = torch.tensor(rank_order, device=device, dtype=torch.long)
        width = sum(local_rows)
        return cls(local_rows, rank_order[rank*width:(rank+1)*width],
                   torch.argsort(rank_order), rank_order)
