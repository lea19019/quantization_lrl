'''
SmoothQuant (Xiao et al. 2023, ICML), run as W4A8 for GPT-2 (docs/project/design.md, sec. 3).
Written by Claude Code on 2026-10-17 at Adrian's request, in the style of rtn.py and gptq.py.
Weights: 4 bits, groups of 128, the rtn.py grid. Activations: 8 bits, one scale per token.

• The idea (Sec. 4, Eq. 3-4):
  Activations have a few channels ~100x larger than the rest, so an 8-bit grid sized by
  them leaves the other channels with one or two steps. For any positive s per channel j,
      Y = X W^T = (X / s) (W * s)^T
  Dividing channel j of X by s_j shrinks the outlier; multiplying column j of W by s_j
  moves that difficulty into the weights.
      s_j = max|X_j| ^ alpha / max|W_j| ^ (1 - alpha),   alpha = 0.5 (paper default)

• Roadmap:
  1. Run calibration windows; per linear keep max|x_j| per input channel.
  2. Smooth the linears whose input comes from a LayerNorm (attn.c_attn, mlp.c_fc), as the
     paper does; the other linears keep s = 1.
  3. Round W * s with the RTN grid at 4 bits.
  4. Hook every linear so its input becomes Q8(x / s) while the model runs.

• Simplification: x / s happens in the hook instead of being folded into the LayerNorm's
  weight and bias. Same outputs. The hooks hold s and the 8-bit rounding, so this model is
  not saved to disk: save_pretrained would keep only the weights.
'''
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.pytorch_utils import Conv1D

from qlrl.calibration.c4 import load_c4, sample_windows
from qlrl.quant.utils import generate, round_to_groups


def quantize_activations(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    # Symmetric, one scale per token: each token's largest |value| maps to the top level.
    levels = 2 ** (bits - 1) - 1  # 127 at 8 bits
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / levels
    return torch.clamp(torch.round(x / scale), -levels, levels) * scale


def smoothquant(
        model: PreTrainedModel,
        act_max: dict[str, torch.Tensor],
        weight_bits: int = 4,
        act_bits: int = 8,
        alpha: float = 0.5
    ) -> PreTrainedModel:

    quantized_model = deepcopy(model)

    def make_quantize_input(s):
        def quantize_input(_module, args):
            x = args[0]
            return (quantize_activations(x / s, act_bits).to(x.dtype),) + args[1:]
        return quantize_input

    with torch.no_grad():
        for name, module in quantized_model.named_modules():
            # Quantize only linear layers: Conv1D in GPT-2, nn.Linear in Llama.
            if not isinstance(module, (torch.nn.Linear, Conv1D)): continue
            # lm_head stays in full precision (it is the same tensor as wte in GPT-2).
            if name == "lm_head": continue
            param = module.weight
            is_conv1d = isinstance(module, Conv1D)

            if is_conv1d: matrix = param.T
            else: matrix = param

            matrix = matrix.float()
            s = torch.ones(matrix.shape[1], device=matrix.device)
            # Inputs that come from a norm: LayerNorm in GPT-2, RMSNorm in Llama.
            if name.endswith(("attn.c_attn", "mlp.c_fc", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
                w_max = matrix.abs().amax(dim=0).clamp(min=1e-5)  # max|W_j| per column
                s = (act_max[name].pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)

            rounded = round_to_groups(matrix * s, weight_bits)
            if is_conv1d: param.copy_(rounded.T)
            else: param.copy_(rounded)

            module.register_forward_pre_hook(make_quantize_input(s))

    return quantized_model


def get_calibration(model: PreTrainedModel, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    # input_ids: [N, L]. Returns { layer: max|x_j| [in] } over every calibration token.
    act_max: dict[str, torch.Tensor] = {}
    handles = []

    def make_save_input(name):
        def save_input(_module, args):
            x = args[0].detach()
            X = x.reshape(-1, x.shape[-1]).T.float()  # [input_features, tokens]
            window_max = X.abs().amax(dim=1)
            if name not in act_max: act_max[name] = window_max
            else: act_max[name] = torch.maximum(act_max[name], window_max)
        return save_input

    for name, module in model.named_modules():
        # The same layers the quantizer rounds: every linear except lm_head (GPT-2 and Llama).
        if isinstance(module, (torch.nn.Linear, Conv1D)) and name != "lm_head":
            handle = module.register_forward_pre_hook(make_save_input(name))
            handles.append(handle)

    with torch.no_grad():
        for i in range(input_ids.shape[0]):
            model(input_ids[i:i + 1].to(model.device))  # one window at a time: [1, L]

    [handle.remove() for handle in handles]

    return act_max


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()

    # Small enough for the login node; the full size (128 x 1024) is a SLURM job.
    input_ids = sample_windows(tokenizer, load_c4(), n=16, length=512, seed=0)
    act_max = get_calibration(model, input_ids)
    quantized_model = smoothquant(model, act_max)

    text = "The capital of France is"
    print("Prompt:", repr(text))
    print("Full-precision continuation:", repr(generate(model, tokenizer, text, 20)))
    print("SmoothQuant W4A8 continuation:", repr(generate(quantized_model, tokenizer, text, 20)))


if __name__ == "__main__":
    main()
