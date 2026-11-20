# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch

_INTERP = 32
_T_GROUP = 5
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) * 1.0 / 2.0
    right = left + ratio * 1.0
    grid = np.linspace(left, right, dim // patch, endpoint=False) * _INTERP
    return torch.from_numpy(grid).to(torch.float64)


def _video_t_grid(n: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(n)],
        dtype=torch.float64,
    )
    return origin + torch.cat(
        [torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)]
    )


def _video_t_span(n: int) -> float:
    # Sequential summation preserves the reference origin at the last ulp.
    return sum(_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(n))
