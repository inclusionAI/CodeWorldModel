"""Per-forward metadata for fully visible, packed cached attention."""
from dataclasses import dataclass
from itertools import accumulate

import torch

from utils import flash_attn


@dataclass(frozen=True)
class PackedAttentionMetadata:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    q_bounds: tuple[int, ...]
    k_bounds: tuple[int, ...]

    @classmethod
    def prepare(cls, query_lens, past_lens, *, query_len, device):
        if (not query_lens or len(query_lens) != len(past_lens)
                or any(n < 0 for n in (*query_lens, *past_lens))
                or sum(query_lens) != query_len):
            raise ValueError("packed attention lengths must cover the query and matching history documents")
        key_lens = tuple(q + p for q, p in zip(query_lens, past_lens))
        return cls.prepare_lengths(query_lens, key_lens, query_len=query_len,
                                   key_len=sum(key_lens), device=device)

    @classmethod
    def prepare_lengths(cls, query_lens, key_lens, *, query_len, key_len, device):
        if (not query_lens or len(query_lens) != len(key_lens)
                or any(n < 0 for n in (*query_lens, *key_lens))
                or sum(query_lens) != query_len or sum(key_lens) != key_len):
            raise ValueError("packed attention lengths must cover the query and key documents")
        q_bounds = tuple(accumulate(query_lens, initial=0))
        k_bounds = tuple(accumulate(key_lens, initial=0))
        if max(q_bounds[-1], k_bounds[-1]) > torch.iinfo(torch.int32).max:
            raise ValueError("packed attention lengths exceed int32 capacity")
        return cls(
            torch.tensor(q_bounds, dtype=torch.int32, device=device),
            torch.tensor(k_bounds, dtype=torch.int32, device=device),
            max(query_lens), max(key_lens), q_bounds, k_bounds,
        )


def cached_attention(wrapper, q, k, v, *, q_lens, k_lens, softmax_scale,
                     metadata: PackedAttentionMetadata | None):
    if metadata is None:
        return wrapper(q, k, v, q_lens=q_lens, k_lens=k_lens,
                       softmax_scale=softmax_scale, causal=False)

    # shortcut: use the pinned Raven helpers; recheck their contract on dependency upgrades.
    if not (flash_attn.FLASH_ATTN_2_AVAILABLE or flash_attn.FLASH_ATTN_3_AVAILABLE):
        return flash_attn._sdpa_varlen(
            q, k, v, q_bounds=metadata.q_bounds, k_bounds=metadata.k_bounds,
            dropout_p=0., softmax_scale=softmax_scale, causal=False,
        )

    assert q.device.type == 'cuda' and q.size(-1) <= 256
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    q, k, v = (x if x.dtype in half_dtypes else x.to(torch.bfloat16) for x in (q, k, v))
    kwargs = dict(
        q=q.to(v.dtype), k=k.to(v.dtype), v=v,
        cu_seqlens_q=metadata.cu_seqlens_q, cu_seqlens_k=metadata.cu_seqlens_k,
        max_seqlen_q=metadata.max_seqlen_q, max_seqlen_k=metadata.max_seqlen_k,
        softmax_scale=softmax_scale, causal=False, deterministic=False,
    )
    if flash_attn.FLASH_ATTN_3_AVAILABLE:
        kwargs.update(version=3, seqused_q=None, seqused_k=None)
    else:
        kwargs.update(version=2, dropout_p=0., window_size=(-1, -1))
    return wrapper._flash_attn_impl(**kwargs).to(out_dtype)
