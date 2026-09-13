"""Round-to-nearest (RTN): the grid every other method rounds onto.

Asymmetric min-max grid per group of `group_size` consecutive input columns of a row
(docs/concepts/quantization.md §2; Frantar 2022 §3 uses the same grid per row):

    scale = (max - min) / (2^bits - 1)
    zero  = round(-min / scale)
    q     = clamp(round(w / scale) + zero, 0, 2^bits - 1)
    w_hat = scale * (q - zero)

Every w_hat is within scale / 2 of w: the first test to write.
"""

import torch
from torch import Tensor


def grid(group: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    """Return (scale, zero) of the asymmetric grid for `group` [out, group_size], per row."""
    raise NotImplementedError


def quantize_groups(weight: Tensor, bits: int, group_size: int) -> Tensor:
    """Return `weight` [out, in] fake-quantized group by group along the input dimension."""
    # Reshape [out, in] -> [out, in // group_size, group_size], apply `grid` per group,
    # round, dequantize, reshape back. GPTQ and AWQ call this too.
    raise NotImplementedError


class RTN:
    """Round every weight to the nearest grid point. No calibration data."""

    def __init__(self, bits: int, group_size: int) -> None:
        self.bits = bits
        self.group_size = group_size

    def quantize(self, weight: Tensor) -> Tensor:
        """Return the fake-quantized weight [out, in], same dtype as the input."""
        raise NotImplementedError
