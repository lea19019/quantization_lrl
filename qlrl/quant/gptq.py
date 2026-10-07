'''
By: Adrian Castillo - Sep 26, 2026

• GPTQ roadmap:
  1. Copy the model and collect calibration inputs for each weight matrix.
  2. For one matrix W, compute inverse-Hessian information from its inputs X.
  3. Quantize one input column at a time using the RTN grid.
  4. Freeze that quantized column in Q.
  5. Update the remaining floating-point columns of W to compensate for its rounding error.
  6. Repeat until every column is frozen, then replace W with Q.
  7. Later: add group size 128, block updates, and Cholesky for speed/stability.

• GPTQ algorithm for one weight matrix W:

  1. Collect calibration inputs X for this layer.
  2. Compute H_inverse = inverse(X @ X.T + damping * I).
  3. Set work = copy(W) and Q = empty matrix.
  4. For each input column j:
     a. Round work[:, j] with the RTN quantization grid; save it in Q[:, j].
     b. error = work[:, j] - Q[:, j].
     c. Divide error by H_inverse[j, j].
     d. Subtract that scaled error from work[:, j:] using H_inverse[j, j:],
        so later columns compensate for the rounding error.
     e. Update the remaining H_inverse for the next column.
  5. Replace W with Q.

• Results, 2026-10-07 (WikiText-2 test perplexity, Llama-3.1-8B, 2048-token windows,
  qlrl/eval/perplexity.py, job 14016949; 4-bit, groups of 128):
  - Full precision (bf16):                              6.240
  - RTN:                                                6.910  (+0.67)
  - GPTQ, 128 x 2048 WikiText-2 train tokens, seed 0:   6.576  (+0.34)
  - GPTQ, 6-token calibration ("The capital of France is"): 7.206  (+0.97)
  - GPTQ, 128 x 2048 C4 tokens, seed 0 (job 14017056):  6.642  (+0.40)
  - Same, seeds 1 and 2 (job 14017566):                 6.679, 6.662
    Three seeds: mean 6.661, standard deviation 0.019   (+0.42)
'''
import argparse
from copy import deepcopy
from pathlib import Path
from typing import cast

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.pytorch_utils import Conv1D

from qlrl.data import load_c4_calibration_ids
from qlrl.quant.architectures import decoder_layers, is_quantized_module


def _is_quantized_module(module_name: str, module: torch.nn.Module) -> bool:
    # gpt2_weight = module_name.startswith("transformer.h.") and isinstance(module, Conv1D)
    # llama_weight = module_name.startswith("model.layers.") and isinstance(
    #     module, torch.nn.Linear
    # )
    # return gpt2_weight or llama_weight
    # Any architecture: every linear inside the repeated layers (see architectures.py).
    return is_quantized_module(module_name, module)

def forward_pass(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, text: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits [1, seq, vocab] in float32, input_ids [1, seq]) for one string."""
    device = next(model.parameters()).device
    tokenized = tokenizer(text, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in tokenized.items()}
    with torch.no_grad():  # inference only, no autograd graph
        out = model(**inputs)
    return out.logits.float(), inputs["input_ids"]

def gold_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return the log-prob of the true next token at each position, shape [seq - 1]."""
    # Position t predicts token t+1, so drop the last logit row and the first token.
    logprobs = torch.log_softmax(logits[0, :-1], dim=-1)  # [seq - 1, vocab]
    targets = input_ids[0, 1:]  # [seq - 1]
    return logprobs[torch.arange(targets.shape[0]), targets]

def perplexity(gold: torch.Tensor) -> float:
    """Return exp(-mean log-prob): the effective number of tokens the model hesitated between."""
    return torch.exp(-gold.mean()).item()

def generate(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, text: str, max_new_tokens: int
) -> str:
    """Return the greedy continuation of `text`, without the prompt."""
    device = next(model.parameters()).device
    tokenized = tokenizer(text, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in tokenized.items()}
    with torch.no_grad():
        ids = model.generate(  # type: ignore[attr-defined]  # transformers 5.x stub bug
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    # decode() is typed `str | list[str]`; a 1-D id tensor always gives a str.
    return cast(str, tokenizer.decode(ids[0, inputs["input_ids"].shape[1] :]))

def load_or_create_quantized_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_folder: str,
    # calibrated_inputs: dict[str, torch.Tensor],
    calibration_ids: torch.Tensor,
    bits: int = 4,
) -> PreTrainedModel:
    """Load a saved GPTQ model, or create and save it when it is absent."""
    # model_path = Path(output_folder) / "model.safetensors"
    model_path = Path(output_folder)
    single_file = (model_path / "model.safetensors").exists()
    sharded_file = (model_path / "model.safetensors.index.json").exists()
    if single_file or sharded_file:
        # return AutoModelForCausalLM.from_pretrained(
        #     output_folder, dtype=torch.float32
        # ).eval()
        return AutoModelForCausalLM.from_pretrained(
            output_folder, dtype=model.dtype
        ).eval()

    # quantized_model = gptq(model, calibrated_inputs, bits=bits)
    # quantized_model = gptq(model, bits=bits)
    quantized_model = gptq_by_layer(model, calibration_ids, bits=bits)
    quantized_model.save_pretrained(output_folder)
    tokenizer.save_pretrained(output_folder)
    return quantized_model

def compare_forward_passes(
    model: PreTrainedModel,
    quantized_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> tuple[bool, float, float]:
    """Return whether logits match, plus their mean and maximum absolute differences."""
    logits, input_ids = forward_pass(model, tokenizer, text)
    quantized_logits, quantized_ids = forward_pass(quantized_model, tokenizer, text)

    if not torch.equal(input_ids, quantized_ids):
        raise ValueError("The models received different token IDs.")

    difference = torch.abs(logits - quantized_logits)
    return torch.equal(logits, quantized_logits), difference.mean().item(), difference.max().item()


def gptq(
        model: PreTrainedModel,
        # calibrated_inputs: dict[str, torch.Tensor],
        hessians: dict[str, torch.Tensor],
        bits: int = 4
    ) -> PreTrainedModel:
    # quantized_model = deepcopy(model)
    # quantized_model = model if model.config.model_type == "llama" else deepcopy(model)
    # In place: gptq_by_layer makes the copy once, then calls this once per decoder layer.
    quantized_model = model
    number_of_levels = 2 ** bits

    with torch.no_grad():
        for name, param in quantized_model.named_parameters():
            if param.ndim != 2: continue
            # Code to check if it's Conv1D or whatever other format
            module_name = name.rsplit(".", 1)[0]
            module = quantized_model.get_submodule(module_name)
            # if not ( name.startswith("transformer.h.") and isinstance(module, Conv1D) ): continue
            if not _is_quantized_module(module_name, module):
                continue
            # Only the matrices of the decoder layer being quantized right now.
            if module_name not in hessians:
                continue
            is_conv1d = isinstance(module, Conv1D)
            # matrix = param.T
            matrix = param.T if is_conv1d else param
            matrix = matrix.float()
            # X = calibrated_inputs[module_name]
            # X = calibrated_inputs[module_name].float()
            # Float64: with few calibration tokens H is badly conditioned and the
            # float32 Cholesky of its inverse fails (job 14016398).
            # X = calibrated_inputs[module_name].double()

            # Calculate the Hessian
            # hessian = 2 * (X @ X.T)
            # Already summed over all calibration tokens by _layer_hessians.
            hessian = hessians[module_name].double()
            # Get the Damping
            average_diagonal = hessian.diagonal().mean()
            damping = 0.01 * average_diagonal
            # Get the inverse
            identity = torch.eye(
                hessian.shape[0],
                dtype=hessian.dtype,
                device=hessian.device,
            )

            damped_hessian = hessian + damping * identity
            # inverse_hessian = torch.linalg.inv(damped_hessian)
            lower = torch.linalg.cholesky(damped_hessian)
            inverse_hessian = torch.cholesky_inverse(lower)

            # inverse_factor = torch.linalg.cholesky(
            #     inverse_hessian,
            #     upper=True,
            # )
            # Back to float32 so it can multiply the float32 weights below.
            inverse_factor = torch.linalg.cholesky(inverse_hessian, upper=True).float()

            work = matrix.clone()
            Q = torch.empty_like(matrix)

            # -------- My Implementation
            # mins = torch.empty_like(matrix)
            # spacings = torch.empty_like(matrix)
            # for row_index in range(work.shape[0]):
            #     row = matrix[row_index]
            #     row_mins = row.clone()
            #     row_spas = row.clone()
            #     groups = torch.split(row,128)
            #     for group_number, group in enumerate(groups):
            #         group_min = group.min()
            #         if group.max() == group.min():
            #             spacer = 0
            #         else:
            #             spacer = (group.max() - group_min) / (number_of_levels - 1)
            #         min_vals = group.clone().fill_(group_min)
            #         spa_vals = group.clone().fill_(spacer)
            #         start = group_number * 128
            #         end = start + group.numel()
            #         row_mins[start:end] = min_vals
            #         row_spas[start:end] = spa_vals
            #     mins[row_index] = row_mins
            #     spacings[row_index] = row_spas

            # # Loop over the columns
            # for col_index in range(work.shape[1]):
            #     col = work[:,col_index]
            #     rounded_col = col.clone()

            #     # Start of RTN
            #     for row, val in enumerate(col):
            #         group_min = mins[row, col_index]
            #         spacer = spacings[row, col_index]
            #         if spacer == 0:
            #             rounded_val = group_min
            #         else :
            #             q = torch.round((val - group_min) / spacer)
            #             q = torch.clamp(q, 0, number_of_levels - 1)
            #             rounded_val = q * spacer + group_min
            #         rounded_col[row] = rounded_val
            #     # End of RTN

            #     # a. Round work[:, j] with the RTN quantization grid; save it in Q[:, j].
            #     Q[:,col_index].copy_(rounded_col) # Here we're freezing the values

            #     #  b. error = work[:, j] - Q[:, j].
            #     error = work[:, col_index] - Q[:, col_index] # is how much the weights in the current column moved during rounding.

            #     # c. Divide error by H_inverse[j, j].
            #     scaled_err = error / inverse_hessian[col_index, col_index] # Still not super clear
                
            #     # d. Subtract that scaled error from work[:, j:] using H_inverse[j, j:],
            #     #     so later columns compensate for the rounding error.
            #     update_vals = torch.outer( # I know we got to get the scale values to subtract to the working matrix, but woldn't be able to actually explain why with my won words.
            #         scaled_err,
            #         inverse_hessian[col_index, col_index:],
            #     )
            #     work[:,col_index:] -= update_vals # Same here, I know what mechanically is happening but not semantically

            #     # e. Update the remaining H_inverse for the next column.
            #     j = col_index

            #     diagonal = inverse_hessian[j, j]
            #     current_column = inverse_hessian[j + 1 :, j]
            #     current_row = inverse_hessian[j, j + 1 :]
            #     remaining_block = inverse_hessian[j + 1 :, j + 1 :]

            #     correction = torch.outer(current_column, current_row) / diagonal
            #     inverse_hessian[j + 1 :, j + 1 :] = remaining_block - correction
            # ---------- End of implementation

            group_size = 128
            rows, columns = matrix.shape

            # group_count = (columns + group_size - 1) // group_size
            # padded_width = group_count * group_size
            # padding = padded_width - columns

            # for_mins = torch.nn.functional.pad(
            #     matrix, (0, padding), value=float("inf")
            # )
            # for_maxs = torch.nn.functional.pad(
            #     matrix, (0, padding), value=float("-inf")
            # )

            # for_mins = for_mins.reshape(rows, group_count, group_size)
            # for_maxs = for_maxs.reshape(rows, group_count, group_size)

            # group_mins = for_mins.amin(dim=2)
            # group_maxs = for_maxs.amax(dim=2)

            # group_spacings = (
            #     (group_maxs - group_mins)
            #     / (number_of_levels - 1)
            # )

            block_size = 128
            # Each group must fit inside one block, so its columns are all in block_work.
            assert block_size % group_size == 0

            for block_start in range(0, work.shape[1], block_size):
                block_end = min(block_start + block_size, work.shape[1])

                # This block already contains compensation from earlier blocks.
                block_work = work[:, block_start:block_end].clone()
                block_quantized = torch.empty_like(block_work)
                block_errors = torch.empty_like(block_work)

                # The part of the inverse factor describing this block.
                block_factor = inverse_factor[
                    block_start:block_end,
                    block_start:block_end,
                ]

                for local_col in range(block_work.shape[1]):
                    global_col = block_start + local_col
                    col = block_work[:, local_col]

                    # Each of these is now a vector containing one value per output row.
                    # group_index = global_col // group_size
                    # group_min = group_mins[:, group_index]
                    # spacing = group_spacings[:, group_index]

                    # At a group's first column, build its grid from the current weights,
                    # which already carry the compensation from earlier columns, as in the
                    # reference GPTQ code. Taking min/max of the original weights instead
                    # leaves compensated weights outside the grid, where clamping clips them.
                    if global_col % group_size == 0:
                        group = block_work[:, local_col : local_col + group_size]
                        group_min = group.amin(dim=1)
                        spacing = (group.amax(dim=1) - group_min) / (number_of_levels - 1)


                    # Avoid dividing by zero for constant groups.
                    constant_group = spacing == 0
                    safe_spacing = torch.where(
                        constant_group,
                        torch.ones_like(spacing),
                        spacing,
                    )

                    # Vectorized RTN: quantize every row in this column simultaneously.
                    q = torch.round((col - group_min) / safe_spacing)
                    q = torch.clamp(q, 0, number_of_levels - 1)
                    rounded_col = q * spacing + group_min
                    rounded_col = torch.where(
                        constant_group,
                        group_min,
                        rounded_col,
                    )

                    block_quantized[:, local_col] = rounded_col

                    # GPTQ compensation inside the current block.
                    diagonal = block_factor[local_col, local_col]
                    scaled_error = (col - rounded_col) / diagonal

                    block_work[:, local_col:] -= (
                        scaled_error.unsqueeze(1)
                        * block_factor[local_col, local_col:].unsqueeze(0)
                    )

                    assert torch.allclose(
                        block_work[:, local_col],
                        rounded_col,
                        atol=1e-5,
                        rtol=1e-5,
                    )

                    block_errors[:, local_col] = scaled_error

                Q[:, block_start:block_end] = block_quantized

                # Apply all errors from this block to later columns in one matrix
                # multiplication instead of updating them separately per column.
                if block_end < work.shape[1]:
                    work[:, block_end:] -= (
                        block_errors
                        @ inverse_factor[block_start:block_end, block_end:]
                    )
            # param.copy_(Q.T)
            quantized_weights = Q.T if is_conv1d else Q
            param.copy_(quantized_weights.to(param.dtype))

            del hessian, damped_hessian, lower, inverse_hessian, inverse_factor
            # del work, Q, matrix, X
            del work, Q, matrix
            
    return quantized_model


def get_calibration(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, text: str) -> dict[str, torch.Tensor]:
    saved_inputs: dict[str, torch.Tensor] = {}
    handles = []

    def make_save_input(name):
        def save_input(_module, args):
            _x = args[0]  # input passed into this layer
            x = _x.detach().clone()
            # Turn [batch, sequence_length, input_features] 
            # into [input_features, calibration_tokens]
            X = x.reshape(-1, x.shape[-1]).T 
            saved_inputs[name] = X
        return save_input
    
    for name, module in model.named_modules():
        # if name.startswith("transformer.h.") and isinstance(module, Conv1D):
        if _is_quantized_module(name, module):
            handle = module.register_forward_pre_hook(make_save_input(name))
            handles.append(handle)

    # tokenized_text = tokenizer(text, return_tensors="pt")
    tokenized_text = tokenizer(text, return_tensors="pt")
    input_device = next(model.parameters()).device
    tokenized_text = {
        name: tensor.to(input_device) for name, tensor in tokenized_text.items()
    }
    with torch.no_grad():
        model(**tokenized_text)

    [handle.remove() for handle in handles]

    return saved_inputs


def load_calibration_ids(
    tokenizer: PreTrainedTokenizerBase, parquet_path: str, samples: int, seq_len: int, seed: int
) -> torch.Tensor:
    """Return `samples` random windows of `seq_len` token ids from a text parquet, [samples, seq_len]."""
    # GPTQ (Frantar et al. 2023, Sec. 5) calibrates on 128 random 2048-token segments.
    lines = load_dataset("parquet", data_files=parquet_path, split="train")["text"]
    ids = tokenizer("".join(lines), return_tensors="pt")["input_ids"][0]
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - seq_len, (samples,), generator=generator)
    return torch.stack([ids[start : start + seq_len] for start in starts.tolist()])


def _decoder_layers(model: PreTrainedModel) -> tuple[torch.nn.ModuleList, str]:
    """Return the decoder layers and their name prefix, e.g. 'model.layers'."""
    # if model.config.model_type == "llama":
    #     return model.model.layers, "model.layers"
    # return model.transformer.h, "transformer.h"
    # Any architecture: the model's longest nn.ModuleList (see architectures.py).
    return decoder_layers(model)


class _StopForward(Exception):
    """Raised by a hook to stop the forward pass once the first layer's inputs are saved."""


def _first_layer_inputs(
    model: PreTrainedModel, calibration_ids: torch.Tensor
) -> tuple[list[torch.Tensor], tuple, dict]:
    """Return the hidden states entering layer 0 (one per sample) and its other arguments."""
    layers, _ = _decoder_layers(model)
    inputs: list[torch.Tensor] = []
    extra: dict = {}

    def save_and_stop(_module, args, kwargs):
        inputs.append(args[0])
        # Mask, position ids and rotary embeddings: the same for every sample of equal length.
        extra["args"], extra["kwargs"] = args[1:], kwargs
        raise _StopForward

    handle = layers[0].register_forward_pre_hook(save_and_stop, with_kwargs=True)
    device = next(model.parameters()).device
    with torch.no_grad():
        for ids in calibration_ids:
            try:
                model(ids.unsqueeze(0).to(device), use_cache=False)
            except _StopForward:
                pass
    handle.remove()
    return inputs, extra["args"], extra["kwargs"]


def _layer_hessians(
    layer: torch.nn.Module, layer_name: str, inputs: list[torch.Tensor], args: tuple, kwargs: dict
) -> dict[str, torch.Tensor]:
    """Return H = 2/N * sum of x x^T over all N calibration tokens, for every matrix in a layer."""
    hessians: dict[str, torch.Tensor] = {}
    token_count = 0
    handles = []

    def make_accumulate(name):
        def accumulate(_module, hook_args):
            x = hook_args[0].reshape(-1, hook_args[0].shape[-1]).float()  # [tokens, features]
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
            hessians[name] += 2 * (x.T @ x)
        return accumulate

    for name, module in layer.named_modules():
        full_name = f"{layer_name}.{name}"
        if _is_quantized_module(full_name, module):
            handles.append(module.register_forward_pre_hook(make_accumulate(full_name)))
    with torch.no_grad():
        for hidden in inputs:
            layer(hidden, *args, **kwargs)
            token_count += hidden.shape[0] * hidden.shape[1]
    [handle.remove() for handle in handles]
    return {name: hessian / token_count for name, hessian in hessians.items()}


def gptq_by_layer(
    model: PreTrainedModel, calibration_ids: torch.Tensor, bits: int = 4
) -> PreTrainedModel:
    """Return the model with every decoder layer quantized by GPTQ, one layer at a time."""
    # One layer at a time keeps one layer's Hessians in memory instead of the whole model's,
    # and calibrates each layer on the outputs of the already-quantized layers before it,
    # so later layers also compensate for earlier layers' rounding (GPTQ paper, Sec. 4).
    quantized_model = model if model.config.model_type == "llama" else deepcopy(model)
    layers, prefix = _decoder_layers(quantized_model)
    inputs, args, kwargs = _first_layer_inputs(quantized_model, calibration_ids)
    for index, layer in enumerate(layers):
        hessians = _layer_hessians(layer, f"{prefix}.{index}", inputs, args, kwargs)
        gptq(quantized_model, hessians, bits=bits)
        del hessians
        with torch.no_grad():
            inputs = [layer(hidden, *args, **kwargs) for hidden in inputs]
        print(f"GPTQ: quantized layer {index + 1}/{len(layers)}", flush=True)
    return quantized_model

def _load_model(model_path: str, test: bool) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the requested model, or a tiny Llama with the same tokenizer for a test run."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if test:
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        config.hidden_size = 64
        config.intermediate_size = 128
        config.num_hidden_layers = 1
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_config(config, dtype=dtype).to(device)
        return model.eval(), tokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Full Llama quantization requires a CUDA GPU with bfloat16 support.")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    return model.eval(), tokenizer


def _parse_args() -> argparse.Namespace:
    """Return command-line paths and quantization settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("output_folder")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--calibration-file",
        # default="data/wikitext-2-raw-v1/wikitext-2-raw-v1/train-00000-of-00001.parquet",
        # C4 as in the GPTQ paper, so calibration text is not from the WikiText test corpus.
        default="data/c4/c4-train.00000-of-01024.json.gz",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model, tokenizer = _load_model(args.model_path, args.test)
    reference_model = deepcopy(model) if args.test else None
    text = "The capital of France is"

    # Type { layer: [ input_features, calibration_tokens ] }
    # calibrated_inputs = get_calibration(model, tokenizer, text)
    # quantized_model = gptq(model, calibrated_inputs)
    samples, seq_len = (2, 64) if args.test else (128, 2048)
    # Never longer than the model can read (GPT-2: 1024 tokens, Llama-3.1: 131072).
    seq_len = min(seq_len, model.config.max_position_embeddings)
    # calibration_ids = load_calibration_ids(
    #     tokenizer, args.calibration_file, samples, seq_len, args.seed
    # )
    calibration_ids = load_c4_calibration_ids(
        tokenizer, args.calibration_file, samples, seq_len, args.seed
    )

    quantized_model = load_or_create_quantized_model(
        model,
        tokenizer,
        args.output_folder,
        # calibrated_inputs,
        calibration_ids,
        bits=args.bits,
    )
    if reference_model is None:
        print("Saved GPTQ model to", args.output_folder)
        return

    identical, mean_difference, max_difference = compare_forward_passes(
        reference_model, quantized_model, tokenizer, text
    )
    print("Logits identical:", identical)
    print("Mean absolute logit difference:", mean_difference)
    print("Maximum absolute logit difference:", max_difference)
    print("Prompt:", repr(text))
    max_new_tokens = 2 if args.test else 20
    print(
        "Full-precision continuation:",
        repr(generate(reference_model, tokenizer, text, max_new_tokens)),
    )
    print(
        "GPTQ 4-bit continuation:",
        repr(generate(quantized_model, tokenizer, text, max_new_tokens)),
    )

if __name__ == "__main__":
    main()
