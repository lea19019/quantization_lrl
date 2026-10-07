'''
By: Adrian Castillo - Sep 14, 2026

Still to validate:
- A constant group stays unchanged.
- Every rounded value lies on its group's 16-value grid at 4 bits.
- A non-constant weight moves by at most half a grid spacing.
- The original model stays unchanged after quantization.
- The quantized model produces finite logits.


1. Select a weight matrix to quantize.
2. Take one row.
3. Split it into 128-weight groups.
4. For each group, find its smallest and largest values.
5. Build that group’s allowed-value grid from those endpoints and the chosen bit width.
6. Replace every weight with its closest grid value.
7. Put the rows back into a matrix of exactly the same shape.

• Results, 2026-10-07 (WikiText-2 test perplexity, Llama-3.1-8B, 2048-token windows,
  qlrl/eval/perplexity.py, job 14016949):
  - Full precision (bf16):  6.240
  - RTN 4-bit, groups of 128: 6.910  (+0.67)
'''
import argparse
from copy import deepcopy
from pathlib import Path
from typing import cast

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.pytorch_utils import Conv1D

from qlrl.quant.architectures import is_quantized_module


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


def rtn( model: PreTrainedModel, bits: int = 4 ) -> PreTrainedModel:
    from copy import deepcopy
    # from transformers.pytorch_utils import Conv1D

    # quantized_model = deepcopy(model)
    quantized_model = model if model.config.model_type == "llama" else deepcopy(model)
    number_of_levels = 2 ** bits
    # names = "\n".join(name for name, _ in model.named_parameters())
    # print(names)
    with torch.no_grad():
        for name, param in quantized_model.named_parameters():
            if param.ndim != 2:
                continue
            # if "weight" not in name:
            #     continue
            # Code to check if it's Conv1D or whatever other format
            module_name = name.rsplit(".", 1)[0]
            module = quantized_model.get_submodule(module_name)

            if not _is_quantized_module(module_name, module):
                continue

            is_conv1d = isinstance(module, Conv1D)

            if is_conv1d:
                matrix = param.T
            else:
                matrix = param
            matrix = matrix.float()

            # rounded_matrix = matrix.clone()

            # Code to transpose the weight
            # This stays the same
            # for row_index in range(matrix.shape[0]):
            #     row = matrix[row_index]
            #     rounded_row = row.clone()
    
            #     groups = torch.split(row, 128)
            #     for group_number, group in enumerate(groups):
            #         group_min = group.min()
            #         group_max = group.max()

            #         if group_min == group_max: continue

            #         # Calculate the space we're going to have in between each value in the grid
            #         spacer = (group_max - group_min) / (number_of_levels - 1)
            #         #
            #         q = torch.round((group - group_min) / spacer)
            #         q = torch.clamp(q, 0, number_of_levels - 1)
            #         rounded_group = q * spacer + group_min
            #         '''
            #         Brute Force Approach
            #         grid = []
            #
            #         for i in range(number_of_levels):
            #             grid_val = group_min + i*spacer
            #             grid.append(grid_val)
            #
            #         rounded_group = torch.empty_like(group)
            #
            #         for index, p in enumerate(group):
            #             idx = round((p-group_min)/spacer)
            #             closest_val = grid[idx]
            #             # BRUTE FORCE APPROACH
            #             # curr_dif = float("inf")
            #             # closest_val = 0
            #             # for val_grid in grid:
            #             #     # Get the absolute difference between p and val_grid
            #             #     absolute_dif = torch.abs(p - val_grid)
            #             #     if absolute_dif < curr_dif:
            #             #         curr_dif = absolute_dif
            #             #         closest_val = val_grid
            #             # # Replace p with closest_val, don't know how to do it.
            #
            #             rounded_group[index] = closest_val
            #         '''
            #         start = group_number * 128
            #         end = start + group.numel()
            #         rounded_row[start:end] = rounded_group
            #     rounded_matrix[row_index].copy_(rounded_row)

            group_size = 128
            rows, columns = matrix.shape
            group_count = (columns + group_size - 1) // group_size
            padded_width = group_count * group_size
            padding = padded_width - columns

            padded_matrix = torch.nn.functional.pad(matrix, (0, padding))
            for_mins = torch.nn.functional.pad(matrix, (0, padding), value=float("inf"))
            for_maxs = torch.nn.functional.pad(matrix, (0, padding), value=float("-inf"))

            grouped = padded_matrix.reshape(rows, group_count, group_size)
            group_mins = for_mins.reshape(rows, group_count, group_size).amin(dim=2, keepdim=True)
            group_maxs = for_maxs.reshape(rows, group_count, group_size).amax(dim=2, keepdim=True)
            spacings = (group_maxs - group_mins) / (number_of_levels - 1)
            safe_spacings = torch.where(spacings == 0, torch.ones_like(spacings), spacings)

            q = torch.round((grouped - group_mins) / safe_spacings)
            q = torch.clamp(q, 0, number_of_levels - 1)
            rounded_groups = torch.where(spacings == 0, group_mins, q * spacings + group_mins)
            rounded_matrix = rounded_groups.reshape(rows, padded_width)[:, :columns]
            # Perform operations on param.data here
            # Transpose back the Conv1D back
            if is_conv1d: param.copy_(rounded_matrix.T)
            else: param.copy_(rounded_matrix)
    return quantized_model


def load_or_create_quantized_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_folder: str,
    bits: int = 4,
) -> PreTrainedModel:
    """Load a saved RTN model, or create and save it when it is absent."""
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

    quantized_model = rtn(model, bits=bits)
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model, tokenizer = _load_model(args.model_path, args.test)
    reference_model = deepcopy(model) if args.test else None
    quantized_model = load_or_create_quantized_model(
        model, tokenizer, args.output_folder, bits=args.bits
    )
    if reference_model is None:
        print("Saved RTN model to", args.output_folder)
        return
    text = "The capital of France is"

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
        "RTN 4-bit continuation:",
        repr(generate(quantized_model, tokenizer, text, max_new_tokens)),
    )

    # logits, input_ids = forward_pass(model, tokenizer, text) # What is exactly logits and input_ids?
    # print(tokenizer.convert_ids_to_tokens(input_ids[0]))  # the subword pieces

    # probs = torch.softmax(logits[0, -1], dim=-1)  # scores for the token after "is"
    # top = torch.topk(probs, 5)
    # for p, i in zip(top.values, top.indices):
    #     print(f"{p:.3f}  {tokenizer.decode(i)!r}")

    # gold = gold_logprobs(logits, input_ids)
    # print(gold, "ppl =", perplexity(gold))
    # print(repr(generate(model, tokenizer, text, max_new_tokens=10)))


if __name__ == "__main__":
    main()
