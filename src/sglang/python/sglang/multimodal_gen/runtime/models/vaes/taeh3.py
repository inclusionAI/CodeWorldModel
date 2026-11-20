# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025 Ollin Boer Bohan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Encoder-only port of madebyollin/taehv's H3 network (62f7591f).

Native 17/5-frame packets reset temporal memory independently, following
NVlabs/Sana StreamRender-H3. Output is already normalized H3 diffusion latent.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _conv(in_channels, out_channels, **kwargs):
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, **kwargs)


class _MemoryBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(channels * 2, channels), nn.ReLU(inplace=True),
            _conv(channels, channels), nn.ReLU(inplace=True),
            _conv(channels, channels),
        )
        self.skip = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, tensor, past):
        return self.act(self.conv(torch.cat([tensor, past], dim=1)) + self.skip(tensor))


class _TemporalPool(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels * stride, channels, 1, bias=False)

    def forward(self, tensor):
        _, channels, height, width = tensor.shape
        return self.conv(tensor.reshape(-1, self.stride * channels, height, width))


class MiniMaxH3TAEEncoder(nn.Module):
    """NTCHW RGB in [0,1] -> NTCHW normalized 24-channel H3 latent."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            _conv(12, 64), nn.ReLU(inplace=True), _TemporalPool(64, 2),
            _conv(64, 64, stride=2, bias=False),
            _MemoryBlock(64), _MemoryBlock(64), _MemoryBlock(64),
            _TemporalPool(64, 2), _conv(64, 64, stride=2, bias=False),
            _MemoryBlock(64), _MemoryBlock(64), _MemoryBlock(64),
            _TemporalPool(64, 1), _conv(64, 64, stride=2, bias=False),
            _MemoryBlock(64), _MemoryBlock(64), _MemoryBlock(64), _conv(64, 24),
        )

    @classmethod
    def from_weights(cls, path, *, device):
        path = Path(path)
        if path.suffix in (".safetensors", ".safetensor"):
            from safetensors.torch import load_file
            state = load_file(str(path))
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls()
        # The released file includes a tiny decoder; the causal decoder stays separate.
        model.load_state_dict({k: v for k, v in state.items() if k.startswith("encoder.")},
                              strict=True, assign=True)
        return model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    def encode_video(self, pixels):
        if (pixels.ndim != 5 or pixels.shape[2] != 3
                or pixels.shape[1] not in (5, 17)
                or min(pixels.shape[0], *pixels.shape[-2:]) <= 0
                or any(size % 16 for size in pixels.shape[-2:])):
            raise ValueError("TAE proxy requires NTCHW RGB, 17 or 5 frames, and H/W divisible by 16")
        batch, frames = pixels.shape[:2]
        count = 2 if frames == 5 else 5
        if frames == 5:
            pixels = torch.cat((pixels, pixels[:, -1:].expand(-1, 12, -1, -1, -1)), dim=1)
        pixels = F.pixel_unshuffle(F.pad(pixels, (0, 0, 0, 0, 0, 0, 3, 0)), 2)
        tensor = pixels.flatten(0, 1)
        for block in self.encoder:
            if isinstance(block, _MemoryBlock):
                sequence = tensor.unflatten(0, (batch, -1))
                past = F.pad(sequence, (0, 0, 0, 0, 0, 0, 1, 0))[:, :sequence.shape[1]]
                tensor = block(tensor, past.reshape(tensor.shape))
            else:
                tensor = block(tensor)
        return tensor.unflatten(0, (batch, -1))[:, :count]
