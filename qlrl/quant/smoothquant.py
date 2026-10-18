'''
SmoothQuant (Xiao et al. 2023, ICML) as a Quantizer, run as W4A8 (docs/project/design.md,
sec. 3). Weights: bits, groups of group_size, the RTN grid. Activations: act_bits, one
scale per token.

• The idea (Sec. 4, Eq. 3-4):
  Activations have a few channels ~100x larger than the rest, so an 8-bit grid sized by
  them leaves the other channels with one or two steps. For any positive s per channel j,
      Y = X W^T = (X / s) (W * s)^T
  Dividing channel j of X by s_j shrinks the outlier; multiplying column j of W by s_j
  moves that difficulty into the weights.
      s_j = max|X_j| ^ alpha / max|W_j| ^ (1 - alpha),   alpha = 0.5 (paper default)

• Roadmap:
  1. calibrate(): per linear keep max|x_j| per input channel.
  2. Smooth the linears whose input comes from a norm, as the paper does; the other
     linears keep s = 1.
  3. Round W * s with the RTN grid.
  4. Hook every linear so its input becomes Q8(x / s) while the model runs.

• Simplification: x / s happens in the hook instead of being folded into the norm's
  weight and bias. Same outputs. The hooks hold s and the 8-bit rounding, so this model is
  not saved to disk: save_pretrained would keep only the weights.
'''
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from qlrl.quant.base import Quantizer, linears
from qlrl.quant.utils.calibration import collect_inputs, run_windows

# Linears whose input comes from a norm: LayerNorm in GPT-2, RMSNorm in Llama.
NORM_FED = ("attn.c_attn", "mlp.c_fc", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")


class SmoothQuant(Quantizer):
    def __init__(self, bits: int, group_size: int, act_bits: int, alpha: float):
        super().__init__(bits, group_size)
        self.act_bits = act_bits
        self.alpha = alpha
        self.scales: dict[str, torch.Tensor] = {}

    def calibrate(self, model: PreTrainedModel, input_ids: torch.Tensor) -> None:
        """Keep { linear: max|x_j| [in] } over every token of the windows [N, L]."""
        self.act_max = collect_inputs(model, lambda: run_windows(model, input_ids), keep_max)

    def quantize(self, model: PreTrainedModel) -> PreTrainedModel:
        """Return a copy with smoothed, rounded weights and 8-bit inputs on every linear."""
        quantized_model = super().quantize(model)  # fills self.scales
        for name, module in linears(quantized_model):
            module.register_forward_pre_hook(make_quantize_input(self.scales[name], self.act_bits))
        return quantized_model

    def quantize_matrix(self, name: str, matrix: torch.Tensor) -> torch.Tensor:
        """Return round(W * s) [out, in] and remember s [in] for the input hook."""
        matrix = matrix.float()
        s = torch.ones(matrix.shape[1], device=matrix.device)
        if name.endswith(NORM_FED):
            w_max = matrix.abs().amax(dim=0).clamp(min=1e-5)  # max|W_j| per column
            s = (self.act_max[name].pow(self.alpha) / w_max.pow(1 - self.alpha)).clamp(min=1e-5)
        self.scales[name] = s
        return self.round(matrix * s)

    def save(
            self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, folder: str
        ) -> None:
        """Save nothing: s and the 8-bit rounding live in hooks, which save_pretrained drops."""


def keep_max(act_max: torch.Tensor | None, X: torch.Tensor) -> torch.Tensor:
    """Return the running max|x_j| [in] after one window X [in, tokens]."""
    window_max = X.abs().amax(dim=1)
    if act_max is None: return window_max
    return torch.maximum(act_max, window_max)


def quantize_activations(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Return x on a symmetric grid, one scale per token: its largest |value| is the top level."""
    levels = 2 ** (bits - 1) - 1  # 127 at 8 bits
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / levels
    return torch.clamp(torch.round(x / scale), -levels, levels) * scale


def make_quantize_input(s: torch.Tensor, bits: int):
    """Return a forward pre-hook that replaces a linear's input x with Q(x / s)."""
    def quantize_input(_module, args):
        x = args[0]
        return (quantize_activations(x / s, bits).to(x.dtype),) + args[1:]
    return quantize_input
