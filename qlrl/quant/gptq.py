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
'''
from typing import cast
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.pytorch_utils import Conv1D
from copy import deepcopy

def forward_pass(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, text: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits [1, seq, vocab] in float32, input_ids [1, seq]) for one string."""
    inputs = tokenizer(text, return_tensors="pt")
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
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(  # type: ignore[attr-defined]  # transformers 5.x stub bug
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    # decode() is typed `str | list[str]`; a 1-D id tensor always gives a str.
    return cast(str, tokenizer.decode(ids[0, inputs["input_ids"].shape[1] :]))

# def load_or_create_quantized_model(
#     model: PreTrainedModel,
#     tokenizer: PreTrainedTokenizerBase,
#     output_folder: str,
#     bits: int = 4,
# ) -> PreTrainedModel:
#     """Load a saved GPTQ model, or create and save it when it is absent."""
#     model_path = Path(output_folder) / "model.safetensors"
#     if model_path.exists():
#         return AutoModelForCausalLM.from_pretrained(
#             output_folder, dtype=torch.float32
#         ).eval()

#     quantized_model = gptq(model, bits=bits)
#     quantized_model.save_pretrained(output_folder)
#     tokenizer.save_pretrained(output_folder)
#     return quantized_model
# def compare_forward_passes(
#     model: PreTrainedModel,
#     quantized_model: PreTrainedModel,
#     tokenizer: PreTrainedTokenizerBase,
#     text: str,
# ) -> tuple[bool, float, float]:
#     """Return whether logits match, plus their mean and maximum absolute differences."""
#     logits, input_ids = forward_pass(model, tokenizer, text)
#     quantized_logits, quantized_ids = forward_pass(quantized_model, tokenizer, text)

#     if not torch.equal(input_ids, quantized_ids):
#         raise ValueError("The models received different token IDs.")

#     difference = torch.abs(logits - quantized_logits)
#     return torch.equal(logits, quantized_logits), difference.mean().item(), difference.max().item()


def gptq( 
        model: PreTrainedModel, 
        calibrated_inputs: dict[str, torch.Tensor],  
        bits: int = 4 
    ) -> PreTrainedModel:
    quantized_model = deepcopy(model)
    number_of_levels = 2 ** bits

    with torch.no_grad():
        for name, param in quantized_model.named_parameters():
            if param.ndim != 2: continue
            # Code to check if it's Conv1D or whatever other format
            module_name = name.rsplit(".", 1)[0]
            module = quantized_model.get_submodule(module_name)
            if not ( name.startswith("transformer.h.") and isinstance(module, Conv1D) ): continue
            matrix = param.T
            rounded_matrix = matrix.clone()
            X = calibrated_inputs[module_name]
            inverse_hessian = 2 * (X @ X.T)

            # The rows here are the columns, so we're good
            for row_index in range(matrix.shape[0]):
                row = matrix[row_index]
                rounded_row = row.clone()

                groups = torch.split(row, 128)
                for group_number, group in enumerate(groups):
                    group_min = group.min()
                    group_max = group.max()
                    if group_min == group_max: continue
                    # Calculate the space we're going to have in between each value in the grid
                    spacer = (group_max - group_min) / (number_of_levels - 1)
                    q = torch.round((group - group_min) / spacer)
                    q = torch.clamp(q, 0, number_of_levels - 1)
                    rounded_group = q * spacer + group_min

                    start = group_number * 128
                    end = start + group.numel()
                    # Step 3?
                    rounding_error = row[start:end] - rounded_group
                    #-----------------------
                    rounded_row[start:end] = rounded_group
                rounded_matrix[row_index].copy_(rounded_row)

            # Perform operations on param.data here
            # Transpose back the Conv1D back
            param.copy_(rounded_matrix.T)
            
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
        if name.startswith("transformer.h.") and isinstance(module, Conv1D):
            handle = module.register_forward_pre_hook(make_save_input(name))
            handles.append(handle)

    tokenized_text = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        model(**tokenized_text)

    [handle.remove() for handle in handles]

    return saved_inputs

def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    text = "The capital of France is"

    # Type { layer: [ input_features, calibration_tokens ] }
    calibrated_inputs = get_calibration(model, tokenizer, text)
    quantized_model = gptq(model, calibrated_inputs)

    # output_folder = "models/gpt2-gptq-4bit"
    # quantized_model = load_or_create_quantized_model(model, tokenizer, output_folder)

    # identical, mean_difference, max_difference = compare_forward_passes(
    #     model, quantized_model, tokenizer, text
    # )
    # print("Logits identical:", identical)
    # print("Mean absolute logit difference:", mean_difference)
    # print("Maximum absolute logit difference:", max_difference)
    # print("Prompt:", repr(text))
    # print("Full-precision continuation:", repr(generate(model, tokenizer, text, 20)))
    # print(
    #     "GPTQ 4-bit continuation:",
    #     repr(generate(quantized_model, tokenizer, text, 20)),
    # )

if __name__ == "__main__":
    main()
