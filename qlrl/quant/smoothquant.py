"""SmoothQuant (Xiao 2023, Eq. 3-4): move activation outliers into the weights, then W8A8.

Per input channel j, activation max a_j = max_t |X[t, j]| and weight max
w_j = max_k |W[k, j]| (torch layout [out, in]: channel j is column j). With migration
strength alpha:

    s_j     = a_j^alpha / w_j^(1 - alpha)
    X_hat   = X diag(s)^-1          W_hat = W diag(s)          X_hat W_hat^T == X W^T

W_hat is rounded per output channel (symmetric, step = max|w| / (2^(b-1) - 1)); X_hat is
rounded per token at run time to `act_bits`. This project runs it as W4A8, so the weight
grid is the grouped RTN grid at 4 bits and only the activation side is 8-bit symmetric.
In fake quantization the forward pre-hook does the X / s and the rounding; nothing is
folded into the norm. Steps 1-6 in docs/papers/methods.md.
"""

import torch
from torch import Tensor


class SmoothQuant:
    """Per-channel smoothing, weight rounding, and per-token activation rounding."""

    def __init__(self, bits: int, group_size: int, act_bits: int, alpha: float) -> None:
        self.bits = bits
        self.group_size = group_size
        self.act_bits = act_bits
        self.alpha = alpha
        self.act_max: Tensor | None = None  # a_j, [in]
        self.scale: Tensor | None = None  # s, [in], set by `quantize`

    def add_batch(self, inputs: Tensor) -> None:
        """Update the running per-channel max |x| from `inputs` [tokens, in]."""
        raise NotImplementedError

    def quantize(self, weight: Tensor) -> Tensor:
        """Set `self.scale` from act_max and weight, return Q(W diag(s)) [out, in]."""
        raise NotImplementedError

    def quantize_input(self, x: Tensor) -> Tensor:
        """Return x / s rounded per token to `act_bits`; installed as a forward pre-hook."""
        raise NotImplementedError
