import torch
import triton
import triton.language as tl


@triton.jit
def _collect_segment_kv_kernel(OLD_K, OLD_V, NEW_K, NEW_V, INDEXES, ROW_ORDER,
                               OUT_K, OUT_V, OLD_LEN, HEADS: tl.constexpr,
                               WIDTH: tl.constexpr, TOTAL,
                               OK0, OK1, OK2, OV0, OV1, OV2,
                               NK0, NK1, NK2, NV0, NV1, NV2,
                               HAS_OLD: tl.constexpr, HAS_ORDER: tl.constexpr,
                               BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offset // (HEADS * WIDTH)
    head = offset // WIDTH % HEADS
    channel = offset % WIDTH
    live = offset < TOTAL
    logical = tl.load(INDEXES + row, live, other=0)
    old = live & (logical < OLD_LEN)
    new_row = logical - OLD_LEN
    if HAS_ORDER:
        new_row = tl.load(ROW_ORDER + new_row, live & ~old, other=0)
    if HAS_OLD:
        old_k = tl.load(OLD_K + logical*OK0 + head*OK1 + channel*OK2, old, other=0)
        old_v = tl.load(OLD_V + logical*OV0 + head*OV1 + channel*OV2, old, other=0)
    new_k = tl.load(NEW_K + new_row*NK0 + head*NK1 + channel*NK2, live & ~old, other=0)
    new_v = tl.load(NEW_V + new_row*NV0 + head*NV1 + channel*NV2, live & ~old, other=0)
    if HAS_OLD:
        new_k = tl.where(old, old_k, new_k)
        new_v = tl.where(old, old_v, new_v)
    tl.store(OUT_K + offset, new_k, live)
    tl.store(OUT_V + offset, new_v, live)


def collect_segment_kv(old_k, old_v, new_k, new_v, indexes, old_len, new_row_order):
    out_k = torch.empty((len(indexes), *new_k.shape[1:]), dtype=new_k.dtype, device=new_k.device)
    out_v = torch.empty_like(out_k)
    if not indexes.numel():
        return out_k, out_v
    old_k = old_k if old_len else new_k
    old_v = old_v if old_len else new_v
    with torch.cuda.device(new_k.device):
        _collect_segment_kv_kernel[(triton.cdiv(out_k.numel(), 256),)](
            old_k, old_v, new_k, new_v, indexes,
            new_row_order if new_row_order is not None else indexes, out_k, out_v,
            old_len, new_k.shape[1], new_k.shape[2], out_k.numel(),
            *old_k.stride(), *old_v.stride(), *new_k.stride(), *new_v.stride(),
            bool(old_len), new_row_order is not None, 256)
    return out_k, out_v
