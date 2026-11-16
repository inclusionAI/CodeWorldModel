import torch
import triton
import triton.language as tl


@triton.jit
def _rope_qk_kernel(Q, K, CACHE, Q_OUT, K_OUT, TOTAL, N: tl.constexpr, BLOCK: tl.constexpr):
    if tl.program_id(1) == 0:
        source, output = Q, Q_OUT
    else:
        source, output = K, K_OUT
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = index % 128
    row = index // (N * 128)
    live = index < TOTAL
    rotary = live & (channel < 96)
    value = tl.load(source + index, live, other=0)
    partner = index + tl.where(channel < 48, 48, -48)
    other = tl.load(source + partner, rotary, other=0)
    # Flip the sign bit because Triton unary minus lowers to zero subtraction.
    negative = (other.to(tl.uint16, bitcast=True) ^ 0x8000).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    rotated = tl.where(channel < 48, negative, other).to(tl.float32)
    cosine = tl.load(CACHE + row * 96 + channel % 48, rotary, other=0).to(tl.float32)
    sine = tl.load(CACHE + row * 96 + channel % 48 + 48, rotary, other=0).to(tl.float32)
    # Eager materializes both BF16 products before adding them.
    first = (value.to(tl.float32) * cosine).to(tl.bfloat16, fp_downcast_rounding='rtne').to(tl.float32)
    second = (rotated * sine).to(tl.bfloat16, fp_downcast_rounding='rtne').to(tl.float32)
    result = (first + second).to(tl.bfloat16, fp_downcast_rounding='rtne')
    tl.store(output + index, tl.where(channel < 96, result, value), live)


def apply_rope_qk(q, k, cos_sin_cache):
    with torch.cuda.device(q.device):
        q_out, k_out = torch.empty_like(q), torch.empty_like(k)
        _rope_qk_kernel[(triton.cdiv(q.numel(), 256), 2)](
            q, k, cos_sin_cache, q_out, k_out, q.numel(), q.shape[1],
            BLOCK=256, num_warps=4, enable_fp_fusion=False,
        )
    return q_out, k_out
