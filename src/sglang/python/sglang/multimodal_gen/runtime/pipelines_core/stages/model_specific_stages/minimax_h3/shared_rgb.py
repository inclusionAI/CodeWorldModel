# SPDX-License-Identifier: Apache-2.0
"""Owned CPU RGB storage for local H3 scheduler replies."""

from multiprocessing.reduction import ForkingPickler

import msgspec
import torch


class H3SharedRGB(msgspec.Struct, frozen=True):
    """Small local-process handle; materialize once while the worker is alive.

    PyTorch's multiprocessing reducer owns the shared storage lifetime. No
    reusable slot or application ACK is needed: each output owns its storage.
    Only the trusted local scheduler may supply this pickle payload.
    """

    handle: bytes

    @classmethod
    def from_tensors(cls, tensors: list[torch.Tensor]) -> "H3SharedRGB":
        return cls(bytes(ForkingPickler.dumps(tuple(tensors))))

    def materialize(self) -> tuple[torch.Tensor, ...]:
        return ForkingPickler.loads(self.handle)


def copy_rgb_to_host(rgb: torch.Tensor | None, *, shared: bool):
    if rgb is None:
        return None
    if not shared:
        return rgb.cpu()
    # Copy directly into the owned shared destination, not via a second CPU copy.
    host = torch.empty(rgb.shape, dtype=rgb.dtype, device="cpu").share_memory_()
    host.copy_(rgb)
    return host
