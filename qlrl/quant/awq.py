'''
AWQ: Activation-aware Weight Quantization (Lin et al. 2024, MLSys). Written by Claude Code
on 2026-10-07 at Adrian's request. Llama and GPT-2 (see architectures.py, which also lists
what is not scaled or clipped in GPT-2). Weight-only, 4 bits, groups of 128.

• The problem
  Plain rounding (rtn.py) treats every weight the same. But the weight columns that multiply
  large activations (salient input channels, ~1% of them) carry most of the output: keeping
  only those in FP16 removes most of the 4-bit damage (Table 1). Mixed precision is awkward
  on hardware, so AWQ protects them a different way.

• Step A, scaling (Sec. 3.2)
  Multiply weight column j by s_j > 1 before rounding and divide activation channel j by s_j:
      y = Q(w * s) * (x / s)
  Rounding moves any weight by at most half a grid step. If the group's max (so the step) does
  not change much, that same absolute error is now relative to a weight s times larger, and
  dividing by s afterwards shrinks the error on the salient channel by about 1/s.
  Scaling too much raises the group max and hurts the other weights, so AWQ searches:
      s = s_x ^ alpha,   s_x = mean |x_j| over calibration tokens            (Eq. 4-5)
      alpha* = argmin over alpha in {0, 0.05, ..., 0.95} of || block(X) - block_q(X) ||^2
  block is what reads x: the whole attention for q/k/v, the whole MLP for gate/up, and the
  linear alone for down. Measuring after the softmax / SiLU is what the next layer sees.
  Like SmoothQuant, 1/s is folded into whatever produces x (an RMSNorm or a previous linear),
  so it costs nothing at inference. Unlike SmoothQuant, the activations stay in 16 bits.

• Step B, clipping (reference code, auto_clip; not in the paper's text)
  The grid of a group spans its min to its max, so one extreme weight stretches the step for
  the other 127. Clamping the group to [-c, c] with c < max|w| makes the step smaller: the
  extreme weight gets a large error, every other weight a smaller one. For each output row and
  group, try c = max|w| * (1, 0.95, ..., 0.55) and keep the c whose rounded weights give the
  lowest error on that group's part of the output, sum_j x_j w_j over the group's 128 inputs.
  q and k are not clipped: their error passes through the attention softmax, which a linear
  output does not measure.

• Roadmap, one decoder layer at a time
  1. Run the calibration inputs through the layer; save the inputs of q, o, gate and down and
     the layer's output (the next layer's input, from the unquantized layer, as the reference).
  2. Step A for each group of linears reading the same input; fold s in. Divide the saved
     input by s too, since that is what the scaled linears now receive.
  3. Step B for v, o, gate, up and down; clamp their weights.
  After every layer, round all weights with rtn.py's grid (4-bit, groups of 128).

• Version 2: explicit input scales (option 3, input_scales.py). ROLLED BACK on 2026-10-07
  together with SmoothQuant's version 2; its code is commented out, version 1 is the active
  code again. Version 1 (the roadmap above) folded 1/s into the producer, so it
  needed the per-architecture map in architectures.py and could not scale every matrix
  (Llama's o_proj with grouped-query attention, GPT-2's c_proj after GELU or attention).
  Now each linear's input is divided by its own s in a hook, so:
  - every linear in every layer gets its own s, in any architecture;
  - step A measures the error at the output of the whole decoder layer: for each linear in
    turn, put Q(W s) / s in that linear only, run the layer, compare with the unquantized
    layer output. That needs no map of which attention or MLP block a linear belongs to;
  - step B (clipping) is unchanged and skips the matrices that produce q and k
    (NOT_CLIPPED, matched by name).
  The scales are saved in input_scales.pt; perplexity.py attaches them after loading.

• Results, 2026-10-07 (WikiText-2 test perplexity, Llama-3.1-8B, 2048-token windows,
  qlrl/eval/perplexity.py; 4-bit, groups of 128; C4 calibration 128 x 512, seed 0):
  - Full precision (bf16):                                        6.240
  - RTN:                                                          6.910  (+0.67)
  - AWQ, step A only, error on linear outputs (job 14017056):     6.638  (+0.40)
  - AWQ, step A with block error + step B clipping (job 14017217): 6.620  (+0.38)
  - Same, seeds 1 and 2 (job 14017566):                           6.612, 6.618
    Three seeds: mean 6.617, standard deviation 0.004             (+0.38)
  - Version 2 (input scales, layer-output error), seeds 0, 1, 2 (job 14017795):
    6.606, 6.593, 6.599; mean 6.599, standard deviation 0.007     (+0.36)
  GPT-2 (124M), 1024-token windows, seed 0; full precision 30.221, RTN 34.761, GPTQ 32.168:
  - Version 1 (job 14017572):                                     33.498 (+3.28)
  - Version 2 (job 14017797):                                     33.092 (+2.87)
'''
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
import argparse
from copy import deepcopy

import torch
from transformers import PreTrainedModel

from qlrl.data import load_c4_calibration_ids
from qlrl.quant.architectures import (
    awq_clip_targets,
    awq_scale_groups,
    divide_output_channels,
    get_weight,
    multiply_input_channels,
    set_weight,
)
# from qlrl.quant.gptq import _is_quantized_module
# from qlrl.quant.input_scales import attach_input_scale, save_input_scales
from qlrl.quant.gptq import _decoder_layers, _first_layer_inputs
from qlrl.quant.rtn import _load_model, compare_forward_passes, generate, rtn


def fake_quantize(weight: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    """Return a [out, in] weight rounded to rtn.py's grid: min/max per group of input columns."""
    rows, columns = weight.shape
    group_size = min(group_size, columns)
    groups = weight.reshape(rows, columns // group_size, group_size)
    group_min = groups.amin(dim=2, keepdim=True)
    spacing = (groups.amax(dim=2, keepdim=True) - group_min) / (2**bits - 1)
    safe_spacing = torch.where(spacing == 0, torch.ones_like(spacing), spacing)
    q = torch.clamp(torch.round((groups - group_min) / safe_spacing), 0, 2**bits - 1)
    return (q * spacing + group_min).reshape(rows, columns)


def _module_inputs(
    layer: torch.nn.Module, modules: list[torch.nn.Module], inputs: list[torch.Tensor], args: tuple, kwargs: dict
) -> tuple[dict[torch.nn.Module, torch.Tensor], list[torch.Tensor]]:
    """Return each module's calibration input, [samples, seq, in], and the layer's outputs."""
    saved: dict[torch.nn.Module, list[torch.Tensor]] = {module: [] for module in modules}

    def save(module, hook_args):
        saved[module].append(hook_args[0])

    handles = [module.register_forward_pre_hook(save) for module in modules]
    outputs = [layer(hidden, *args, **kwargs) for hidden in inputs]
    [handle.remove() for handle in handles]
    return {module: torch.cat(values) for module, values in saved.items()}, outputs


def search_scales(linears: list[torch.nn.Module], block, x: torch.Tensor, bits: int, grid: int = 20) -> torch.Tensor:
    """Return the per-input-channel scales s = s_x^alpha with the lowest block output error."""
    originals = [get_weight(linear).clone() for linear in linears]  # each [out, in]
    reference = block(x)  # full-precision block output
    activation_mean = x.reshape(-1, x.shape[-1]).abs().float().mean(dim=0)  # s_x, [in]
    best_error, best_scales = float("inf"), torch.ones_like(activation_mean)
    for step in range(grid):
        alpha = step / grid
        scales = activation_mean.pow(alpha).clamp(min=1e-4)
        # Centre the scales around 1 so the weights keep roughly their overall size.
        scales = scales / (scales.max() * scales.min()).sqrt()
        # Put Q(W s) / s in the linears: block(x) then computes (x / s) Q(W s)^T.
        for linear, weight in zip(linears, originals):
            set_weight(linear, fake_quantize(weight.float() * scales, bits) / scales)
        error = (reference - block(x)).float().pow(2).mean().item()
        if error < best_error:
            best_error, best_scales = error, scales
    for linear, weight in zip(linears, originals):
        set_weight(linear, weight)
    return best_scales


# Version 2 (rolled back):
# def search_scales(linear: torch.nn.Module, x: torch.Tensor, bits: int, layer_error, grid: int = 20) -> torch.Tensor:
#     """Return s = s_x^alpha for one linear with the lowest error at the decoder layer's output."""
#     # layer_error() runs the whole decoder layer and returns its mean squared difference
#     # from the unquantized layer output (built in _awq_layer).
#     original = get_weight(linear).clone()  # [out, in]
#     activation_mean = x.reshape(-1, x.shape[-1]).abs().float().mean(dim=0)  # s_x, [in]
#     best_error, best_scales = float("inf"), torch.ones_like(activation_mean)
#     for step in range(grid):
#         alpha = step / grid
#         scales = activation_mean.pow(alpha).clamp(min=1e-4)
#         # Centre the scales around 1 so the weights keep roughly their overall size.
#         scales = scales / (scales.max() * scales.min()).sqrt()
#         # Q(W s) / s in the linear: the layer then computes (x / s) Q(W s)^T for this linear.
#         set_weight(linear, fake_quantize(original.float() * scales, bits) / scales)
#         error = layer_error()
#         if error < best_error:
#             best_error, best_scales = error, scales
#     set_weight(linear, original)
#     return best_scales


def search_clipping(weight: torch.Tensor, x: torch.Tensor, bits: int) -> torch.Tensor:
    """Return the best clipping value c per output row and group, shape [rows, groups, 1]."""
    group_size = min(128, weight.shape[1])
    tokens = x.reshape(-1, x.shape[-1])
    tokens = tokens[:: max(1, tokens.shape[0] // 512)].float()  # 512 tokens, as the reference
    tokens = tokens.reshape(1, tokens.shape[0], -1, group_size)  # [1, tokens, groups, group]
    best_values = []
    for start in range(0, weight.shape[0], 32):  # 32 output rows at a time to bound memory
        w = weight[start : start + 32].float()
        rows = w.shape[0]
        w = w.reshape(rows, 1, -1, group_size)  # [rows, 1, groups, group]
        # Each group's part of each output: sum over its input channels.
        reference = (tokens * w).sum(dim=-1)  # [rows, tokens, groups]
        largest = w.abs().amax(dim=-1, keepdim=True)  # max|w| per row and group
        best, best_error = largest.clone(), torch.full_like(largest, float("inf"))
        for step in range(10):  # c = max|w| * (1, 0.95, ..., 0.55)
            limit = largest * (1 - step / 20)
            clipped = torch.clamp(w, -limit, limit)
            quantized = fake_quantize(clipped.reshape(rows, -1), bits).reshape(w.shape)
            error = ((tokens * quantized).sum(dim=-1) - reference).pow(2).mean(dim=1)
            error = error.reshape(best_error.shape)
            better = error < best_error
            best_error[better], best[better] = error[better], limit[better]
        best_values.append(best.reshape(rows, -1, 1))
    return torch.cat(best_values)


def apply_clipping(linear: torch.nn.Module, limit: torch.Tensor) -> None:
    """Clamp every group of the weight to [-c, c] with its own c, shape [rows, groups, 1]."""
    weight = get_weight(linear).float()
    rows, columns = weight.shape
    groups = weight.reshape(rows, limit.shape[1], -1)
    set_weight(linear, torch.clamp(groups, -limit, limit).reshape(rows, columns))


def _awq_layer(layer: torch.nn.Module, model_type: str, inputs: list, args: tuple, kwargs: dict, bits: int) -> list:
    """Apply step A (scaling) and step B (clipping) to one decoder layer; return its outputs."""
    groups = awq_scale_groups(layer, model_type, args, kwargs)
    clip_targets = awq_clip_targets(layer, model_type)
    # Save the input of every matrix that is scaled or clipped.
    readers = [linear for _producer, linears, _block in groups for linear in linears]
    features, outputs = _module_inputs(layer, readers + clip_targets, inputs, args, kwargs)
    for producer, linears, block in groups:
        scales = search_scales(linears, block, features[linears[0]], bits)
        divide_output_channels(producer, scales)
        for linear in linears:
            multiply_input_channels(linear, scales)
            # The scaled matrix now receives x / s; clipping must be measured on that.
            features[linear] = features[linear] / scales.to(features[linear].dtype)
    for linear in clip_targets:
        apply_clipping(linear, search_clipping(get_weight(linear), features[linear], bits))
    return outputs


# Version 2 (rolled back):
# # Matrices that produce q and k (Llama; GPT-2 computes q, k and v in one c_attn): not clipped,
# # because their error passes through the attention softmax. Unknown names are clipped.
# NOT_CLIPPED = ("q_proj", "k_proj", "c_attn")
#
#
# def _run_layer(layer: torch.nn.Module, x: torch.Tensor, args: tuple, kwargs: dict) -> torch.Tensor:
#     """Return a decoder layer's output hidden states (some models return a tuple)."""
#     output = layer(x, *args, **kwargs)
#     return output[0] if isinstance(output, tuple) else output
#
#
# def _awq_layer(
#     layer: torch.nn.Module, layer_name: str, inputs: list, args: tuple, kwargs: dict, bits: int
# ) -> tuple[list, dict[str, torch.Tensor]]:
#     """Scale and clip every linear of one decoder layer; return its outputs and the scales."""
#     linears = {
#         f"{layer_name}.{name}": module
#         for name, module in layer.named_modules()
#         if _is_quantized_module(f"{layer_name}.{name}", module)
#     }
#     features, outputs = _module_inputs(layer, list(linears.values()), inputs, args, kwargs)
#     # All samples in one batch: [samples, seq, hidden]. Reference = unquantized layer output.
#     layer_input = torch.cat(inputs)
#     reference = _run_layer(layer, layer_input, args, kwargs)
#
#     def layer_error():
#         output = _run_layer(layer, layer_input, args, kwargs)
#         return (reference - output).float().pow(2).mean().item()
#
#     scales: dict[str, torch.Tensor] = {}
#     for name, linear in linears.items():  # step A, one linear at a time
#         scales[name] = search_scales(linear, features[linear], bits, layer_error)
#         # (x / s) (W s)^T = x W^T: weight columns times s now, input divided by s in a hook.
#         multiply_input_channels(linear, scales[name])
#         attach_input_scale(linear, scales[name])
#         # The linear now receives x / s; clipping must be measured on that.
#         features[linear] = features[linear] / scales[name].to(features[linear].dtype)
#     for name, linear in linears.items():  # step B
#         if not name.endswith(NOT_CLIPPED):
#             apply_clipping(linear, search_clipping(get_weight(linear), features[linear], bits))
#     return outputs, scales


def awq(model: PreTrainedModel, calibration_ids: torch.Tensor, bits: int = 4) -> PreTrainedModel:
    """Return the model with AWQ scaling and clipping applied and every decoder weight rounded."""
    layers, _ = _decoder_layers(model)
    inputs, args, kwargs = _first_layer_inputs(model, calibration_ids)
    with torch.no_grad():
        for index, layer in enumerate(layers):
            # The next layer's inputs come from this layer before clipping, as in the reference.
            inputs = _awq_layer(layer, model.config.model_type, inputs, args, kwargs, bits)
            print(f"AWQ: scaled and clipped layer {index + 1}/{len(layers)}", flush=True)
    # rtn() returns a rounded copy for GPT-2 and rounds Llama in place; use what it returns.
    return rtn(model, bits=bits)


# Version 2 (rolled back):
# def awq(
#     model: PreTrainedModel, calibration_ids: torch.Tensor, bits: int = 4
# ) -> tuple[PreTrainedModel, dict[str, torch.Tensor]]:
#     """Return the AWQ-quantized model and the input scale s of every linear."""
#     layers, prefix = _decoder_layers(model)
#     inputs, args, kwargs = _first_layer_inputs(model, calibration_ids)
#     scales: dict[str, torch.Tensor] = {}
#     with torch.no_grad():
#         for index, layer in enumerate(layers):
#             # The next layer's inputs come from this layer before any change, as in the reference.
#             inputs, layer_scales = _awq_layer(layer, f"{prefix}.{index}", inputs, args, kwargs, bits)
#             scales.update(layer_scales)
#             print(f"AWQ: scaled and clipped layer {index + 1}/{len(layers)}", flush=True)
#     # rtn() returns a rounded copy for GPT-2 and rounds Llama in place; use what it returns.
#     # A copy keeps the input-scale hooks, which point to the same scale tensors.
#     return rtn(model, bits=bits), scales


def _parse_args() -> argparse.Namespace:
    """Return command-line paths and AWQ settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("output_folder")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--calibration-file", default="data/c4/c4-train.00000-of-01024.json.gz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model, tokenizer = _load_model(args.model_path, args.test)
    reference_model = deepcopy(model) if args.test else None
    # The reference code calibrates on 128 windows of 512 tokens (Pile); here C4.
    samples, seq_len = (2, 64) if args.test else (128, 512)
    # Never longer than the model can read (GPT-2: 1024 tokens).
    seq_len = min(seq_len, model.config.max_position_embeddings)
    calibration_ids = load_c4_calibration_ids(
        tokenizer, args.calibration_file, samples, seq_len, args.seed
    )
    model = awq(model, calibration_ids, bits=args.bits)
    # Version 2 (rolled back):
    # model, scales = awq(model, calibration_ids, bits=args.bits)
    model.save_pretrained(args.output_folder)
    tokenizer.save_pretrained(args.output_folder)
    # Hooks are not saved by save_pretrained; the scales go in their own file.
    # save_input_scales(args.output_folder, scales)
    print("Saved AWQ model to", args.output_folder)
    if reference_model is None:
        return

    text = "The capital of France is"
    identical, mean_difference, max_difference = compare_forward_passes(
        reference_model, model, tokenizer, text
    )
    print("Logits identical:", identical)
    print("Mean absolute logit difference:", mean_difference)
    print("Maximum absolute logit difference:", max_difference)
    print("Full-precision continuation:", repr(generate(reference_model, tokenizer, text, 2)))
    print("AWQ 4-bit continuation:", repr(generate(model, tokenizer, text, 2)))


if __name__ == "__main__":
    main()
