"""AWQ (Lin 2023, Eq. 4-5): scale salient input channels up before rounding.

For a scale vector s over input channels, W diag(s) is rounded and diag(s)^-1 is folded
into the activation, so W X == (W diag(s)) (diag(s)^-1 X). Scaling channel j by s_j > 1
leaves the group max, and so the step, nearly unchanged while its rounding error shrinks
by about 1/s_j (Eq. 2-3). Saliency is activation magnitude, not weight magnitude:

    s = s_X ** alpha,   s_X[j] = mean_t |X[t, j]|,   alpha on a grid in [0, 1]
    alpha* = argmin || Q(W diag(s)) (diag(s)^-1 X) - W X ||

In fake quantization the stored weight is Q(W diag(s*)) diag(s*)^-1 and nothing is folded
into the previous op. Steps 1-7 in docs/papers/methods.md.
"""

import torch
from torch import Tensor


class AWQ:
    """Activation-aware per-channel scaling followed by RTN."""

    def __init__(self, bits: int, group_size: int, n_grid: int) -> None:
        self.bits = bits
        self.group_size = group_size
        self.n_grid = n_grid  # number of alpha values searched in [0, 1]
        self.channel_mean_abs: Tensor | None = None  # s_X, [in]
        self.inputs: list[Tensor] = []  # a kept sample of X for the output-error search

    def add_batch(self, inputs: Tensor) -> None:
        """Update the running per-channel mean |x| from `inputs` [tokens, in]; keep a sample."""
        raise NotImplementedError

    def quantize(self, weight: Tensor) -> Tensor:
        """Return Q(W diag(s*)) diag(s*)^-1, shape [out, in], after the alpha grid search."""
        raise NotImplementedError
