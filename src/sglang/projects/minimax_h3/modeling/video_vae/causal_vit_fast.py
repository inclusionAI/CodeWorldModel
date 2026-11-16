"""Inference-only packed FA2 decoder, adapted from Sana's causal_vit_stream.

The reference owns all checkpoint parameters. A lazily built BF16 Linear copy
is shared by streams; their fixed-shape geometry, ring KV and graphs are private.
Norms, residuals, input/output projections retain the reference FP32 recipe.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from .causal_vit_decoder import OUT_CH, PATCH, PATCH_T
from .vit_utils import _apply_rotary_pos_emb_impl

BF16 = torch.bfloat16


def _pre(blk, x, suffix, rope, suffix_rope, permutation, heads, dh):
    q, k, v = blk.attn.to_qkv(blk.norm1(x).to(BF16)).view(-1, heads, 3 * dh).chunk(3, -1)
    qs, ks, vs = blk.attn.to_qkv(blk.norm1(suffix).to(BF16)).view(-1, heads, 3 * dh).chunk(3, -1)
    if blk.attn.norm_q is not None:
        q, qs = blk.attn.norm_q(q.float()).to(BF16), blk.attn.norm_q(qs.float()).to(BF16)
        k, ks = blk.attn.norm_k(k.float()).to(BF16), blk.attn.norm_k(ks.float()).to(BF16)
    q = _apply_rotary_pos_emb_impl(q[None], rope)[0]
    k = _apply_rotary_pos_emb_impl(k[None], rope)[0]
    qs = _apply_rotary_pos_emb_impl(qs[None], suffix_rope)[0]
    ks = _apply_rotary_pos_emb_impl(ks[None], suffix_rope)[0]
    return q[permutation], k[permutation], v[permutation], qs, ks, vs


def _post(blk, x, suffix, output, suffix_output, inverse):
    x = x + blk.attn.to_out(output.flatten(1)[inverse]).float() * blk.scale1
    suffix = suffix + blk.attn.to_out(suffix_output).float() * blk.scale1
    x = x + blk.ff(blk.norm2(x).to(BF16)).float() * blk.scale2
    suffix = suffix + blk.ff(blk.norm2(suffix).to(BF16)).float() * blk.scale2
    return x, suffix


class FastCausalViT:
    """Deployment weights, deliberately outside the module/state_dict tree."""

    @torch.inference_mode(False)
    @torch.no_grad()
    def __init__(self, ref, *, compile=False, cuda_graph=False, tp_group=None):
        from flash_attn import flash_attn_varlen_func

        if any(t.is_inference() for t in (*ref.parameters(), *ref.buffers())):
            raise ValueError("Fast decoder weights/buffers need version counters; reload outside inference_mode()")
        if ref.register_tokens.device.type != "cuda":
            raise ValueError("The fast decoder requires CUDA")
        if any(p.dtype != torch.float32 for p in ref.parameters()):
            raise ValueError("Keep reference VAE parameters FP32; fast Linear copies use BF16")
        if ref.cache_latents % ref.chunk:
            raise ValueError("The KV ring requires cache_latents divisible by chunk")
        self.ref, self.cuda_graph, self.attention = ref, cuda_graph, flash_attn_varlen_func
        self.tp_group = tp_group
        self.heads = ref.heads if tp_group is None else ref.heads // tp_group.world_size
        post = _post
        if tp_group is None:
            self.blocks = copy.deepcopy(ref.blocks).eval().requires_grad_(False)
            for module in self.blocks.modules():
                if isinstance(module, nn.Linear):
                    module.to(BF16)
        else:
            from .causal_vit_tp import make_tp_blocks, tp_post
            self.blocks = make_tp_blocks(ref, tp_group)
            post = tp_post
        self.pre = torch.compile(_pre, dynamic=False) if compile else _pre
        self.post = torch.compile(post, dynamic=False) if compile else post

    def capture_context(self):
        return nullcontext() if self.tp_group is None else self.tp_group.graph_capture()


class FastDecodeState:
    """One batch-one fixed-resolution stream; reset preserves allocated buffers."""

    @torch.inference_mode(False)
    @torch.no_grad()
    def __init__(self, backend, height, width):
        self.backend = backend
        r = backend.ref
        self.height, self.width = height, width
        self.slots, self.suffix_count = r.cache_latents + r.chunk, r.register_tokens.shape[1] + 1
        self.shifts = [r.window // 2 if r.shift and i % 2 else 0 for i in range(len(r.blocks))]
        dev = r.register_tokens.device
        self.geometry = {s: self._geometry(s, dev) for s in set(self.shifts)}
        self.k = [torch.zeros(self.geometry[s]["total"], backend.heads, r.dh, dtype=BF16, device=dev) for s in self.shifts]
        self.v = [torch.zeros_like(k) for k in self.k]
        self.input = torch.empty(1, r.chunk, r.latents_mean.numel(), height, width, device=dev)
        self.rope = {s: r._ids(0, r.chunk, height, width, s, torch.float32, dev) for s in self.geometry}
        self.suffix_rope = r._suffix_ids(0, self.suffix_count, torch.float32, dev)
        self.graph, self.output = None, None
        self.reset()

    def _geometry(self, shift, device):
        r, H, W, S, slots = self.backend.ref, self.height, self.width, self.suffix_count, self.slots
        ws, C = r.window, r.chunk
        hp, wp = ((H + shift + ws - 1) // ws) * ws, ((W + shift + ws - 1) // ws) * ws
        yy = torch.arange(hp).view(hp // ws, 1, ws, 1) - shift
        xx = torch.arange(wp).view(1, wp // ws, 1, ws) - shift
        yy, xx = torch.broadcast_tensors(yy, xx)
        valid = ((yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)).flatten(2).flatten(0, 1)
        flat = (yy * W + xx).flatten(2).flatten(0, 1)
        windows = [f[m] for f, m in zip(flat, valid) if m.any()]
        sizes = torch.tensor([len(w) for w in windows])
        lengths = S + slots * sizes
        starts = lengths.cumsum(0) - lengths
        permutation = torch.cat([t * H * W + w for w in windows for t in range(C)])
        inverse = permutation.argsort()
        write = {a: torch.cat([start + S + (a + t) * len(w) + torch.arange(len(w))
                              for start, w in zip(starts, windows) for t in range(C)])
                 for a in range(0, slots, C)}
        suffix_index = (starts[:, None] + torch.arange(S)).flatten()
        # Each startup prefix has its own compact FA2 layout (FA2 has no seqused_k).
        stages = {}
        for filled in range(C, slots + 1, C):
            keep = torch.zeros(int(lengths.sum()), dtype=torch.bool)
            indices = []
            for wi, (start, n) in enumerate(zip(starts, sizes)):
                indices.append(start + torch.arange(S + filled * n))
                keep[start + S:start + S + filled * n] = True
                if wi == 0:
                    keep[start:start + S] = True  # Count global suffix exactly once.
            segment = S + filled * sizes
            stages[filled] = dict(
                index=torch.cat(indices).to(device),
                cu=torch.cat((torch.zeros(1, dtype=torch.long), segment.cumsum(0))).to(device, torch.int32),
                max=int(segment.max()), mask=keep.to(device)[None, None, None, :],
            )
        return dict(
            total=int(lengths.sum()), count=len(windows), permutation=permutation.to(device),
            inverse=inverse.to(device), write={a: v.to(device) for a, v in write.items()},
            write_current=torch.empty_like(permutation, device=device), suffix_index=suffix_index.to(device),
            cu_q=torch.cat((torch.zeros(1, dtype=torch.long), (C * sizes).cumsum(0))).to(device, torch.int32),
            max_q=int(C * sizes.max()), stages=stages,
        )

    def reset(self):
        self.t = 0
        for k, v in zip(self.k, self.v):
            k.zero_()
            v.zero_()

    @torch.no_grad()
    def _step(self, filled):
        r, C, H, W = self.backend.ref, self.backend.ref.chunk, self.height, self.width
        # Do not inherit caller autocast for FP32 projections or residuals.
        with torch.autocast("cuda", enabled=False):
            raw = self.input * r.latents_std.view(1, 1, -1, 1, 1) + r.latents_mean.view(1, 1, -1, 1, 1)
            with torch.autocast("cuda", dtype=BF16):
                raw = r.post_quant_conv(raw.transpose(1, 2))
            x = r.x_embedder(raw.permute(0, 2, 3, 4, 1).float()).to(BF16).float().reshape(C * H * W, -1)
            suffix = torch.cat((r.register_tokens[0], r.register_tokens.new_zeros(1, x.shape[-1]))).to(BF16).float()
            for li, block in enumerate(self.backend.blocks):
                shift = self.shifts[li]
                g = self.geometry[shift]
                stage = g["stages"][filled]
                q, k, v, qs, ks, vs = self.backend.pre(
                    block, x, suffix, self.rope[shift], self.suffix_rope, g["permutation"], self.backend.heads, r.dh)
                kp, vp = self.k[li], self.v[li]
                kp.index_copy_(0, g["write_current"], k)
                vp.index_copy_(0, g["write_current"], v)
                kp.index_copy_(0, g["suffix_index"], ks.repeat(g["count"], 1, 1))
                vp.index_copy_(0, g["suffix_index"], vs.repeat(g["count"], 1, 1))
                ka, va = (kp, vp) if filled == self.slots else (kp[stage["index"]], vp[stage["index"]])
                output = self.backend.attention(q, ka, va, g["cu_q"], stage["cu"], g["max_q"], stage["max"])
                suffix_output = F.scaled_dot_product_attention(
                    qs.transpose(0, 1)[None], kp.transpose(0, 1)[None], vp.transpose(0, 1)[None],
                    attn_mask=stage["mask"],
                )[0].transpose(0, 1).reshape(self.suffix_count, -1)
                x, suffix = self.backend.post(block, x, suffix, output, suffix_output, g["inverse"])
            out = r.proj_out(r.norm_out(x))
            out = out.view(C, H, W, OUT_CH, PATCH_T, PATCH, PATCH).permute(3, 0, 4, 1, 5, 2, 6)
            return out.reshape(1, OUT_CH, C * PATCH_T, H * PATCH, W * PATCH)

    @torch.no_grad()
    def decode_chunk(self, z):
        r = self.backend.ref
        if tuple(z.shape) != tuple(self.input.shape) or z.device != self.input.device:
            raise ValueError("Fast decoder streams require batch one and fixed resolution/device")
        self.input.copy_(z)
        for shift, g in self.geometry.items():
            g["write_current"].copy_(g["write"][self.t % self.slots])
            for target, source in zip(self.rope[shift], r._ids(
                    self.t, r.chunk, self.height, self.width, shift, torch.float32, z.device)):
                target.copy_(source)
        for target, source in zip(self.suffix_rope, r._suffix_ids(self.t, self.suffix_count, torch.float32, z.device)):
            target.copy_(source)
        filled = min(self.t + r.chunk, self.slots)
        if not self.backend.cuda_graph or self.t < self.slots:
            output = self._step(filled)
        else:
            if self.graph is None:
                torch.cuda.synchronize(z.device)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.device(z.device), self.backend.capture_context() as capture:
                    stream = None if capture is None else capture.stream
                    with torch.cuda.graph(self.graph, stream=stream):
                        self.output = self._step(self.slots)
            self.graph.replay()
            output = self.output
        keep = [i for i in range(r.chunk * PATCH_T)
                if not ((self.t + i // PATCH_T) % r.block_latents == 0 and i % PATCH_T < r.frames_to_trim)]
        self.t += r.chunk
        # Never expose graph-owned output: subsequent replay must not alter published frames.
        return output[:, :, keep] if len(keep) < r.chunk * PATCH_T else output.clone()
