'''
GPTQ (Frantar et al. 2023) as a Quantizer. Adrian's original, with the full roadmap, is
archive/_gptq.py; the math below is the same.

Columns are rounded one at a time on the RTN grid; each column's rounding error is pushed
onto the columns not yet rounded, weighted by the inverse Hessian H^-1, H = X X^T.

The model is quantized block by block (utils/blocks.py): a block's H is measured on the
outputs of the blocks before it, already quantized. So H cannot be collected in
calibrate(); calibrate() keeps the windows and quantize() does the rest.
'''
import torch
from transformers import PreTrainedModel

from qlrl.quant.base import Quantizer
from qlrl.quant.utils.blocks import quantize_by_block


class GPTQ(Quantizer):
    def calibrate(self, model: PreTrainedModel, input_ids: torch.Tensor) -> None:
        """Keep the windows [N, L]; H is measured block by block inside quantize()."""
        self.input_ids = input_ids

    def quantize(self, model: PreTrainedModel) -> PreTrainedModel:
        """Return a quantized copy of `model`, one block at a time."""
        return quantize_by_block(self, model, self.input_ids, add_to_hessian)

    def quantize_matrix(self, name: str, matrix: torch.Tensor) -> torch.Tensor:
        """Return GPTQ's Q for `matrix` [out, in], using this block's H for `name`."""
        return gptq_matrix(matrix, self.stats[name], self.bits, self.group_size)


def gptq_matrix(
        matrix: torch.Tensor, H: torch.Tensor, bits: int, group_size: int
    ) -> torch.Tensor:
    """Return Q [out, in] in float32, the GPTQ-rounded `matrix` [out, in]."""
    number_of_levels = 2 ** bits
    damping = 0.01 * torch.diag(H).mean()
    I = torch.eye(H.shape[0], device=H.device)
    L = torch.linalg.cholesky(H + damping * I)
    inv_H = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(inv_H, upper=True)
    # float32 for the GPTQ math even when the model is bf16; copy_ casts back.
    work = matrix.float().clone()
    Q = torch.zeros_like(work)

    for start in range(0, matrix.shape[1], group_size):
        group = work[:, start:start + group_size]
        group_min = group.amin(dim=-1, keepdim=True)
        group_max = group.amax(dim=-1, keepdim=True)
        spacer = (group_max - group_min) / (number_of_levels - 1)
        spacer = torch.where(spacer == 0, 1.0, spacer)

        end = start + group_size
        scaled_errors = torch.zeros(matrix.shape[0], group_size, device=work.device)
        for col_idx in range(start, end):
            column = work[:, col_idx:col_idx + 1]
            q = torch.round((column - group_min) / spacer)
            q = torch.clamp(q, 0, number_of_levels - 1)
            Q[:, col_idx:col_idx + 1] = q * spacer + group_min

            error = work[:, col_idx:col_idx + 1] - Q[:, col_idx:col_idx + 1]
            scaled_error = error / U[col_idx, col_idx]
            work[:, col_idx+1:end] -= scaled_error * U[col_idx, col_idx+1:end]
            scaled_errors[:, col_idx - start:col_idx - start + 1] = scaled_error

        work[:, end:] -= scaled_errors @ U[start:end, end:]

    return Q


def add_to_hessian(H: torch.Tensor | None, X: torch.Tensor) -> torch.Tensor:
    """Return H + X X^T [in, in]: H is a sum over tokens, so each window adds its own."""
    if H is None: return X @ X.T
    return H.add_(X @ X.T)
