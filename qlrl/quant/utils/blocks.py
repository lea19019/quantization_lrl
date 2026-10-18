'''
Block-by-block quantization, used by GPTQ and AWQ.

Each method needs a statistic of what every linear receives (H for GPTQ, H and sum|x| for
AWQ). Measuring all of them in one pass keeps every linear's statistic in memory at once:
for Llama 3.1 8B, ~39 GB of H's. Going one decoder block at a time keeps only one block's
(~1.2 GB):
  1. Catch what block 0 receives for every calibration window.
  2. Measure each linear's statistic in this block on those inputs.
  3. Round the block's linears with quantizer.quantize_matrix(); free the statistics.
  4. Run the inputs through the quantized block; they are the next block's inputs.
So each block is measured on the outputs of blocks already quantized, as in GPTQ (Sec. 4).
'''
from collections.abc import Callable
from copy import deepcopy
from functools import partial

import torch
from transformers import PreTrainedModel

from qlrl.quant.base import Quantizer, get_matrix, linears, set_matrix
from qlrl.quant.utils.calibration import Statistic, collect_inputs


def quantize_by_block(
        quantizer: Quantizer,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        update: Callable[[Statistic | None, torch.Tensor], Statistic],
    ) -> PreTrainedModel:
    """Return a quantized copy of `model`, one block at a time (steps 1-4 above).

    While a block is rounded, quantizer.stats holds { linear in the block: statistic },
    built by `update` as in collect_inputs; quantize_matrix() reads it.
    """
    quantized_model = deepcopy(model)
    inputs, args, kwargs = catch_block_inputs(quantized_model, input_ids)
    blocks = get_blocks(quantized_model)
    name = type(quantizer).__name__

    with torch.no_grad():
        for index, block in enumerate(blocks):
            run = partial(run_block, block, inputs, args, kwargs)
            quantizer.stats = collect_inputs(block, run, update)
            for linear_name, module in linears(block):
                set_matrix(module, quantizer.quantize_matrix(linear_name, get_matrix(module)))
            quantizer.stats = {}  # free this block's statistics before the next block
            inputs = run_block(block, inputs, args, kwargs)
            print(f"{name}: quantized block {index + 1}/{len(blocks)}", flush=True)

    return quantized_model


def get_blocks(model: PreTrainedModel) -> list[torch.nn.Module]:
    """Return the decoder blocks. GPT-2: model.transformer.h. Llama: model.model.layers."""
    name = "transformer.h" if hasattr(model, "transformer") else "model.layers"
    return list(model.get_submodule(name).children())


def catch_block_inputs(
        model: PreTrainedModel, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, tuple, dict]:
    """Return block 0's hidden states [N, L, hidden] and its other arguments.

    The other arguments are the causal mask and Llama's rotary cos/sin. Every window has
    length L, so they are the same for all windows.
    """
    inputs, other_args, kwargs = [], [], {}

    def catch(_module, args, block_kwargs):
        inputs.append(args[0])
        other_args[:] = args[1:]
        kwargs.update(block_kwargs)

    handle = get_blocks(model)[0].register_forward_pre_hook(catch, with_kwargs=True)
    with torch.no_grad():
        for i in range(input_ids.shape[0]):
            # use_cache=False: otherwise a key/value cache is passed in and keeps growing.
            model(input_ids[i:i + 1].to(model.device), use_cache=False)
    handle.remove()
    return torch.cat(inputs), tuple(other_args), kwargs


def run_block(
        block: torch.nn.Module, inputs: torch.Tensor, args: tuple, kwargs: dict
    ) -> torch.Tensor:
    """Return the block's outputs [N, L, hidden] for inputs [N, L, hidden], one window at a time."""
    outputs = []
    with torch.no_grad():
        for i in range(inputs.shape[0]):
            out = block(inputs[i:i + 1], *args, **kwargs)
            outputs.append(out[0] if isinstance(out, tuple) else out)
    return torch.cat(outputs)
