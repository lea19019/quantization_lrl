'''
AWQ, Activation-aware Weight Quantization (Lin et al. 2024, MLSys), as a Quantizer.
Weight-only, on the RTN grid.

• The idea (Sec. 3.2):
  A few input channels carry large activations; the weight columns they multiply matter most.
  Scale column j by s_j before rounding, then divide it back:
      W_hat = Q(W * s) / s,   so   W_hat x = Q(W * s) (x / s)
  The rounding error of a column scaled up by s_j shrinks by about 1/s_j once divided back.
  s = mean|x_j| ^ alpha, alpha searched in {0, 0.05, ..., 0.95}. alpha = 0 is plain RTN.

• Roadmap (block by block, utils/blocks.py, so only one block's H's are in memory):
  1. Per linear of the block keep H = X X^T and mean|x_j| per input channel.
  2. For each linear and each alpha: build s, round W * s with the RTN grid, divide by s.
  3. Output error on the calibration inputs: ||(W - W_hat) X||^2 = sum((D @ H) * D),
     with D = W - W_hat (the same damage formula as in GPTQ).
  4. Keep the W_hat with the smallest error.

• Simplifications (fake quantization):
  - s is divided back into the weight instead of folded into the layer before. Same outputs,
    because activations stay in full precision; a real int4 kernel would need the fold.
  - Error per linear, not per attention/MLP block; no clipping search (paper, Sec. 3.3).
'''
import torch
from transformers import PreTrainedModel

from qlrl.quant.base import Quantizer
from qlrl.quant.utils.blocks import quantize_by_block


class AWQ(Quantizer):
    def calibrate(self, model: PreTrainedModel, input_ids: torch.Tensor) -> None:
        """Keep the windows [N, L]; H and mean|x| are measured block by block in quantize()."""
        self.input_ids = input_ids

    def quantize(self, model: PreTrainedModel) -> PreTrainedModel:
        """Return a quantized copy of `model`, one block at a time."""
        return quantize_by_block(self, model, self.input_ids, add_to_sums)

    def quantize_matrix(self, name: str, matrix: torch.Tensor) -> torch.Tensor:
        """Return the W_hat [out, in] with the smallest output error over the alpha grid."""
        H, abs_sum = self.stats[name]
        mean_abs = abs_sum / self.input_ids.numel()  # mean|x_j| over every calibration token
        matrix = matrix.float()  # float32 for the search even when the model is bf16
        best_error, best = float("inf"), matrix
        for alpha in [i / 20 for i in range(20)]:
            s = mean_abs.pow(alpha).clamp(min=1e-4)
            # Keep s centred around 1, as the reference AWQ code does.
            s = s / (s.max() * s.min()).sqrt()
            rounded = self.round(matrix * s) / s
            D = matrix - rounded
            error = ((D @ H) * D).sum().item()
            if error < best_error: best_error, best = error, rounded
        return best


def add_to_sums(
        sums: tuple[torch.Tensor, torch.Tensor] | None, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (H + X X^T [in, in], sum|x| [in]) after one window X [in, tokens]."""
    if sums is None: return X @ X.T, X.abs().sum(dim=1)
    H, abs_sum = sums
    return H.add_(X @ X.T), abs_sum.add_(X.abs().sum(dim=1))
