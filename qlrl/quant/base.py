'''
Quantizer: the methods every quantization class in this package has.

    quantizer.calibrate(model, input_ids)       read calibration windows [N, L]
    quantizer.quantize(model) -> model          a quantized copy; `model` is left untouched
    quantizer.save(model, tokenizer, folder)

The default quantize() rounds each linear on its own with quantize_matrix(). A method that
needs more than that (GPTQ, SmoothQuant) overrides quantize().

The helpers below the class are shared: which modules are linears, the Conv1D transpose,
and the RTN grid every method rounds onto.
'''
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.pytorch_utils import Conv1D


class Quantizer:
    def __init__(self, bits: int, group_size: int):
        self.bits = bits
        self.group_size = group_size
        # Per-linear calibration statistics, for methods that need them (GPTQ, AWQ).
        self.stats: dict[str, Any] = {}

    def calibrate(self, model: PreTrainedModel, input_ids: torch.Tensor) -> None:
        """Collect what quantize() needs from windows [N, L]. Default: nothing (RTN)."""

    def quantize(self, model: PreTrainedModel) -> PreTrainedModel:
        """Return a copy of `model` with every linear replaced by quantize_matrix()."""
        quantized_model = deepcopy(model)
        with torch.no_grad():
            for name, module in linears(quantized_model):
                set_matrix(module, self.quantize_matrix(name, get_matrix(module)))
        return quantized_model

    def quantize_matrix(self, name: str, matrix: torch.Tensor) -> torch.Tensor:
        """Return the quantized version of one weight matrix [out, in]."""
        raise NotImplementedError

    def save(
            self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, folder: str
        ) -> None:
        """Write the quantized model and its tokenizer to `folder`."""
        model.save_pretrained(folder)
        tokenizer.save_pretrained(folder)

    def round(self, matrix: torch.Tensor) -> torch.Tensor:
        """Return `matrix` [out, in] on the RTN grid: groups of group_size, 2^bits levels."""
        return round_to_groups(matrix, self.bits, self.group_size)


def linears(model: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Linear | Conv1D]]:
    """Yield (name, module) for every linear that gets quantized."""
    for name, module in model.named_modules():
        # Conv1D in GPT-2, nn.Linear in Llama. Embeddings and norms are other types.
        if not isinstance(module, (torch.nn.Linear, Conv1D)): continue
        # lm_head stays in full precision. In GPT-2 it is the same tensor as wte, so
        # quantizing it would quantize the embeddings too.
        if name == "lm_head": continue
        yield name, module


def get_matrix(module: torch.nn.Linear | Conv1D) -> torch.Tensor:
    """Return the weight as [out, in]. GPT-2's Conv1D stores it transposed, [in, out]."""
    if isinstance(module, Conv1D): return module.weight.T
    return module.weight


def set_matrix(module: torch.nn.Linear | Conv1D, matrix: torch.Tensor) -> None:
    """Write `matrix` [out, in] into the weight; copy_ casts back to the model's dtype."""
    if isinstance(module, Conv1D): module.weight.copy_(matrix.T)
    else: module.weight.copy_(matrix)


def round_to_groups(matrix: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Return `matrix` [out, in] with each row's groups of `group_size` on their own grid.

    A group's grid has 2^bits evenly spaced values from its min to its max; every weight
    moves to the closest one.
    """
    number_of_levels = 2 ** bits
    G = matrix.reshape(matrix.shape[0], matrix.shape[-1] // group_size, group_size)
    group_min = G.amin(dim=-1, keepdim=True)
    group_max = G.amax(dim=-1, keepdim=True)
    spacer = (group_max - group_min) / (number_of_levels - 1)
    spacer = torch.where(spacer == 0, 1.0, spacer)  # a constant group stays as it is
    q = torch.clamp(torch.round((G - group_min) / spacer), 0, number_of_levels - 1)
    return (q * spacer + group_min).reshape(matrix.shape)
