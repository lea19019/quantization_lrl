'''
AWQ, Activation-aware Weight Quantization (Lin et al. 2024, MLSys), simple version for GPT-2.
Written by Claude Code on 2026-10-17 at Adrian's request, in the style of rtn.py and gptq.py.
Weight-only, groups of 128, the same grid as rtn.py.

• The idea (Sec. 3.2):
  A few input channels carry large activations; the weight columns they multiply matter most.
  Scale column j by s_j before rounding, then divide it back:
      W_hat = Q(W * s) / s,   so   W_hat x = Q(W * s) (x / s)
  The rounding error of a column scaled up by s_j shrinks by about 1/s_j once divided back.
  s = mean|x_j| ^ alpha, alpha searched in {0, 0.05, ..., 0.95}. alpha = 0 is plain RTN.

• Roadmap:
  1. Run calibration windows; per linear keep H = X X^T and mean|x_j| per input channel.
  2. For each linear and each alpha: build s, round W * s with the RTN grid, divide by s.
  3. Output error on the calibration inputs: ||(W - W_hat) X||^2 = sum((D @ H) * D),
     with D = W - W_hat (the same damage formula as in GPTQ).
  4. Keep the W_hat with the smallest error.

• Simplifications (fake quantization):
  - s is divided back into the weight instead of folded into the layer before. Same outputs,
    because activations stay in full precision; a real int4 kernel would need the fold.
  - Error per linear, not per attention/MLP block; no clipping search (paper, Sec. 3.3).
'''
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.pytorch_utils import Conv1D

from qlrl.calibration.c4 import load_c4, sample_windows
from qlrl.quant.utils import generate, load_or_create_quantized_model, round_to_groups


def awq(
        model: PreTrainedModel,
        stats: dict[str, tuple[torch.Tensor, torch.Tensor]],
        bits: int = 4
    ) -> PreTrainedModel:

    quantized_model = deepcopy(model)

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

            H, mean_abs = stats[name]
            matrix = matrix.float()  # float32 for the search even when the model is bf16
            best_error, best = float("inf"), matrix
            for alpha in [i / 20 for i in range(20)]:
                s = mean_abs.pow(alpha).clamp(min=1e-4)
                # Keep s centred around 1, as the reference AWQ code does.
                s = s / (s.max() * s.min()).sqrt()
                rounded = round_to_groups(matrix * s, bits) / s
                D = matrix - rounded
                error = ((D @ H) * D).sum().item()
                if error < best_error: best_error, best = error, rounded

            if is_conv1d: param.copy_(best.T)
            else: param.copy_(best)

    return quantized_model


def get_calibration(
        model: PreTrainedModel, input_ids: torch.Tensor
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    # input_ids: [N, L]. Returns { layer: (H [in, in], mean|x| [in]) }.
    H: dict[str, torch.Tensor] = {}
    abs_sum: dict[str, torch.Tensor] = {}
    handles = []

    def make_save_input(name):
        def save_input(_module, args):
            x = args[0].detach()
            X = x.reshape(-1, x.shape[-1]).T.float()  # [input_features, tokens]
            if name not in H:
                H[name] = X @ X.T
                abs_sum[name] = X.abs().sum(dim=1)
            else:
                H[name] += X @ X.T
                abs_sum[name] += X.abs().sum(dim=1)
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

    tokens = input_ids.numel()
    return {name: (H[name], abs_sum[name] / tokens) for name in H}


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()

    # Small enough for the login node; the full size (128 x 1024) is a SLURM job.
    input_ids = sample_windows(tokenizer, load_c4(), n=16, length=512, seed=0)
    stats = get_calibration(model, input_ids)

    output_folder = "models/gpt2-awq-4bit"
    quantized_model = load_or_create_quantized_model(
        model, tokenizer, output_folder, lambda m: awq(m, stats)
    )

    text = "The capital of France is"
    print("Prompt:", repr(text))
    print("Full-precision continuation:", repr(generate(model, tokenizer, text, 20)))
    print("AWQ 4-bit continuation:", repr(generate(quantized_model, tokenizer, text, 20)))


if __name__ == "__main__":
    main()
