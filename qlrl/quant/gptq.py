"""GPTQ (Frantar 2022, Algorithm 1): round one column at a time, compensate the rest.

Layer reconstruction loss  || W X - W_hat X ||^2  has Hessian  H = 2 X X^T  [in, in].
Rounding column j to q_j and applying the OBS update to the unrounded columns
(Frantar 2022, Eq. 3 with the OBS/OBQ derivation)

    err_j            = (W[:, j] - q_j) / [H^-1]_jj
    W[:, j+1:]      -= err_j * [H^-1]_{j, j+1:}

keeps the loss minimal given the columns already fixed. The paper's tricks: damping
H += damp * mean(diag H) * I; the upper Cholesky factor of H^-1 supplies the rows
[H^-1]_{j, j:}; columns are processed in blocks of `block_size` with a lazy batch update.
Act-order (from the released code, not the paper): permute columns so diag(H) is
decreasing, quantize in that order, un-permute at the end. Steps 1-7 in
docs/papers/methods.md.
"""

import torch
from torch import Tensor


class GPTQ:
    """Column-by-column quantizer with second-order error compensation."""

    def __init__(
        self, bits: int, group_size: int, block_size: int, damp: float, act_order: bool
    ) -> None:
        self.bits = bits
        self.group_size = group_size
        self.block_size = block_size
        self.damp = damp
        self.act_order = act_order
        self.hessian: Tensor | None = None  # [in, in], running 2/n * sum x x^T
        self.n_samples = 0

    def add_batch(self, inputs: Tensor) -> None:
        """Accumulate H = 2 X X^T from `inputs` [tokens, in]; keeps a running mean."""
        raise NotImplementedError

    def quantize(self, weight: Tensor) -> Tensor:
        """Return the fake-quantized weight [out, in] after the column walk."""
        # 1. (act_order) permutation from diag(H); permute weight columns and H.
        # 2. Damp H, invert, upper Cholesky factor U so that U[j, j:] plays [H^-1]_{j, j:}.
        # 3. For each block of columns: round (grid recomputed at each group start),
        #    error, update the rest of the block; after the block, update later columns.
        # 4. (act_order) un-permute the quantized columns.
        raise NotImplementedError
