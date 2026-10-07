'''
SmoothQuant W8A8 (Xiao et al. 2023, "SmoothQuant", ICML). Written by Claude Code on
2026-10-07 at Adrian's request. Llama and GPT-2 (see architectures.py).

• The problem
  To run a matrix multiplication Y = X W^T in int8, both the weights W and the activations X
  are rounded to 8 bits. Weights are easy: their values are spread evenly. Activations are
  hard: a few input channels (the outlier features of docs/papers/outliers.md) are ~100x
  larger than the rest. With one int8 scale per token, the step size is set by the outlier,
  so the normal channels fall into one or two grid steps and lose almost all information.

• The idea (Sec. 4, Eq. 3-4)
  For any positive vector s, one value per input channel j,
      Y = X W^T = (X diag(s)^-1) (W diag(s))^T.
  Dividing activation channel j by s_j and multiplying weight column j by s_j changes nothing
  in exact math, but it moves part of the outlier range from X (hard to quantize) into W
  (easy to quantize). SmoothQuant picks
      s_j = max|X_j|^alpha / max|W_j|^(1 - alpha)
  alpha = 0.5 splits the difficulty evenly; larger alpha pushes more of it into the weights.
  The paper uses alpha = 0.85 for Llama-2 (Table 7).

• Where s goes for free
  X here is the output of an RMSNorm, y = x / rms(x) * g. Dividing g by s divides the
  norm's output by s, so 1/s costs nothing at inference. Only the two inputs that come out
  of a norm are smoothed: q/k/v (after input_layernorm) and gate/up (after
  post_attention_layernorm), as in the reference code. In GPT-2 the same places are
  ln_1 -> c_attn and ln_2 -> c_fc; its LayerNorm also has a bias b, divided by s as well.

• Roadmap
  1. Run calibration text; record max|X_j| for the inputs of q_proj and gate_proj.
  2. For each decoder layer compute s for (q, k, v) and for (gate, up).
  3. Divide the RMSNorm weight by s and multiply the linear weight columns by s.
  4. Round every linear weight to int8: symmetric, one scale per output row.
  5. At inference, round every linear input to int8: symmetric, one scale per token
     (add_activation_quantization; hooks are not saved, so evaluation must call it).
  Not done here: the reference also quantizes the q/k/v outputs used by attention (BMM).

• Version 2: explicit input scales (option 3, input_scales.py). ROLLED BACK on 2026-10-07:
  worse results (below); its code is commented out, version 1 is the active code again.
  Version 1 (the roadmap above) folded 1/s into the RMSNorm, so
  only q/k/v and gate/up could be smoothed and they had to share one s. Now each linear's
  input is divided by its own s in a hook, so every linear is smoothed, including o_proj and
  down_proj (in Llama, down_proj reads the largest activation outliers), and any
  architecture works. Each s uses only that linear's own weights: max|W_j| over its rows.
  The scales are saved in input_scales.pt; perplexity.py attaches them before rounding x.

• Results, 2026-10-07 (WikiText-2 test perplexity, Llama-3.1-8B, 2048-token windows,
  qlrl/eval/perplexity.py --quantize-activations, job 14017057; C4 calibration 512 x 512, seed 0):
  - Full precision (bf16):                    6.240
  - W8A8, no smoothing (--no-smoothing):      6.365  (+0.125)
  - SmoothQuant W8A8, alpha = 0.85:           6.343  (+0.103)
  - Version 2, every linear smoothed (job 14017796): 6.381  (+0.141)
  GPT-2 (124M), 1024-token windows; full precision 30.221:
  - W8A8, no smoothing (job 14017573):         30.443
  - Version 1 (job 14017573):                  30.421
  - Version 2, every linear smoothed (job 14017798): 75.185
  Version 2 is worse because of the MLP's second matrix (down_proj, GPT-2 mlp.c_proj). On
  GPT-2 (8 windows, 64 calibration samples): smoothing c_attn + c_fc 33.92, + attn.c_proj
  33.84, + mlp.c_proj 81.35. With weights rounded but activations not, version 2 is fine
  (34.02), so the damage comes from rounding x / s.
'''
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
import argparse
from copy import deepcopy

import torch
from transformers import PreTrainedModel

from qlrl.data import load_c4_calibration_ids
from qlrl.quant.architectures import (
    divide_output_channels,
    get_weight,
    multiply_input_channels,
    set_weight,
    smoothing_pairs,
)
from qlrl.quant.gptq import _decoder_layers, _is_quantized_module
# from qlrl.quant.gptq import _is_quantized_module
# from qlrl.quant.input_scales import attach_input_scale, save_input_scales
from qlrl.quant.rtn import _load_model, compare_forward_passes, generate


def activation_maxima(
    model: PreTrainedModel, modules: list[torch.nn.Module], calibration_ids: torch.Tensor
) -> dict[torch.nn.Module, torch.Tensor]:
    """Return max|x_j| per input channel of each given module, over all calibration tokens."""
    maxima: dict[torch.nn.Module, torch.Tensor] = {}

    def record(module, args):
        x = args[0].reshape(-1, args[0].shape[-1])  # [tokens, in_features]
        channel_max = x.abs().amax(dim=0).float()
        if module in maxima:
            maxima[module] = torch.maximum(maxima[module], channel_max)
        else:
            maxima[module] = channel_max

    handles = [module.register_forward_pre_hook(record) for module in modules]
    device = next(model.parameters()).device
    with torch.no_grad():
        for ids in calibration_ids:
            model(ids.unsqueeze(0).to(device), use_cache=False)
    [handle.remove() for handle in handles]
    return maxima


def smooth(
    norm: torch.nn.Module, linears: list[torch.nn.Module], activation_max: torch.Tensor, alpha: float
) -> None:
    """Move part of the activation range into the weights: norm output / s, weight columns * s."""
    # max|W_j|: largest weight in input column j over every linear that reads this input.
    weights = [get_weight(linear).abs() for linear in linears]  # each [out, in]
    weight_max = torch.cat(weights, dim=0).amax(dim=0).float().clamp(min=1e-5)
    scales = (activation_max.pow(alpha) / weight_max.pow(1 - alpha)).clamp(min=1e-5)  # Eq. 4
    divide_output_channels(norm, scales)
    for linear in linears:
        multiply_input_channels(linear, scales)


# Version 2 (rolled back):
# def smoothing_scales(weight: torch.Tensor, activation_max: torch.Tensor, alpha: float) -> torch.Tensor:
#     """Return s_j = max|X_j|^alpha / max|W_j|^(1 - alpha) for one linear's [out, in] weight."""
#     # max|W_j|: largest weight in input column j of this linear (over all its output rows).
#     weight_max = weight.abs().amax(dim=0).float().clamp(min=1e-5)
#     return (activation_max.pow(alpha) / weight_max.pow(1 - alpha)).clamp(min=1e-5)  # Eq. 4


def quantize_weight_int8(weight: torch.Tensor) -> torch.Tensor:
    """Return a [out, in] weight rounded to a symmetric int8 grid, one scale per output row."""
    # Symmetric grid: scale * {-127, ..., 127}, centred on 0, so an int8 matmul needs no zero
    # point. scale = max|row| / 127 puts the row's largest weight exactly on the edge.
    w = weight.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127
    return torch.round(w / scale).clamp(-127, 127) * scale


def quantize_activation_int8(x: torch.Tensor) -> torch.Tensor:
    """Return x rounded to a symmetric int8 grid, one scale per token (dynamic, per-token)."""
    scale = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127
    return (torch.round(x.float() / scale).clamp(-127, 127) * scale).to(x.dtype)


def add_activation_quantization(model: PreTrainedModel) -> None:
    """Make every decoder linear round its input to int8 per token before multiplying."""
    for name, module in model.named_modules():
        if _is_quantized_module(name, module):
            module.register_forward_pre_hook(
                lambda _module, args: (quantize_activation_int8(args[0]),)
            )


def smoothquant(
    model: PreTrainedModel, calibration_ids: torch.Tensor, alpha: float, smoothing: bool = True
) -> PreTrainedModel:
    """Return the model smoothed (unless smoothing=False) with int8 weights, in place."""
    model_type = model.config.model_type
    layers, _ = _decoder_layers(model)
    with torch.no_grad():
        if smoothing:
            pairs = [pair for layer in layers for pair in smoothing_pairs(layer, model_type)]
            # All linears of a pair read the same input, so recording the first one is enough.
            first_linears = [linears[0] for _norm, linears in pairs]
            maxima = activation_maxima(model, first_linears, calibration_ids)
            for norm, linears in pairs:
                smooth(norm, linears, maxima[linears[0]], alpha)
        for name, module in model.named_modules():
            if _is_quantized_module(name, module):
                set_weight(module, quantize_weight_int8(get_weight(module)))
    return model


# Version 2 (rolled back):
# def smoothquant(
#     model: PreTrainedModel, calibration_ids: torch.Tensor, alpha: float, smoothing: bool = True
# ) -> tuple[PreTrainedModel, dict[str, torch.Tensor]]:
#     """Return the model with int8 weights, in place, and the input scale s of every linear."""
#     linears = {
#         name: module for name, module in model.named_modules() if _is_quantized_module(name, module)
#     }
#     scales: dict[str, torch.Tensor] = {}
#     with torch.no_grad():
#         if smoothing:
#             maxima = activation_maxima(model, list(linears.values()), calibration_ids)
#             for name, linear in linears.items():
#                 scales[name] = smoothing_scales(get_weight(linear), maxima[linear], alpha)
#                 # (x / s) (W s)^T = x W^T: weight columns times s, input divided by s.
#                 multiply_input_channels(linear, scales[name])
#                 attach_input_scale(linear, scales[name])
#         for linear in linears.values():
#             set_weight(linear, quantize_weight_int8(get_weight(linear)))
#     return model, scales


def _parse_args() -> argparse.Namespace:
    """Return command-line paths and SmoothQuant settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("output_folder")
    parser.add_argument("--alpha", type=float, default=0.85)
    # Plain W8A8 rounding without smoothing: the baseline SmoothQuant is compared against.
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--calibration-file", default="data/c4/c4-train.00000-of-01024.json.gz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model, tokenizer = _load_model(args.model_path, args.test)
    reference_model = deepcopy(model) if args.test else None
    # The paper calibrates on 512 sentences of 512 tokens (Pile); here C4.
    samples, seq_len = (2, 64) if args.test else (512, 512)
    # Never longer than the model can read (GPT-2: 1024 tokens).
    seq_len = min(seq_len, model.config.max_position_embeddings)
    calibration_ids = load_c4_calibration_ids(
        tokenizer, args.calibration_file, samples, seq_len, args.seed
    )
    smoothquant(model, calibration_ids, args.alpha, smoothing=not args.no_smoothing)
    # Version 2 (rolled back):
    # model, scales = smoothquant(
    #     model, calibration_ids, args.alpha, smoothing=not args.no_smoothing
    # )
    model.save_pretrained(args.output_folder)
    tokenizer.save_pretrained(args.output_folder)
    # Hooks are not saved by save_pretrained; the scales go in their own file.
    # save_input_scales(args.output_folder, scales)
    print("Saved SmoothQuant model to", args.output_folder)
    if reference_model is None:
        return

    add_activation_quantization(model)
    text = "The capital of France is"
    identical, mean_difference, max_difference = compare_forward_passes(
        reference_model, model, tokenizer, text
    )
    print("Logits identical:", identical)
    print("Mean absolute logit difference:", mean_difference)
    print("Maximum absolute logit difference:", max_difference)
    print("Full-precision continuation:", repr(generate(reference_model, tokenizer, text, 2)))
    print("W8A8 continuation:", repr(generate(model, tokenizer, text, 2)))


if __name__ == "__main__":
    main()
