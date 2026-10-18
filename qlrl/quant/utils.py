'''
Helpers shared by every quantization script: run a model, score it, generate, and cache
a quantized model on disk. Moved out of rtn.py by Claude Code on 2026-10-15 at Adrian's
request.
'''
from pathlib import Path
from typing import Callable, cast

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase


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


def load_or_create_quantized_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_folder: str,
    quantize: Callable[[PreTrainedModel], PreTrainedModel],
) -> PreTrainedModel:
    """Load a saved quantized model, or create it with `quantize` and save it when absent."""
    model_path = Path(output_folder) / "model.safetensors"
    if model_path.exists():
        return AutoModelForCausalLM.from_pretrained(
            output_folder, dtype=torch.float32
        ).eval()

    quantized_model = quantize(model)
    quantized_model.save_pretrained(output_folder)
    tokenizer.save_pretrained(output_folder)
    return quantized_model


def round_to_groups(matrix: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    """Round each row's groups of `group_size` weights to their own min-max grid (as rtn.py)."""
    number_of_levels = 2 ** bits
    G = matrix.reshape(matrix.shape[0], matrix.shape[-1] // group_size, group_size)
    group_min = G.amin(dim=-1, keepdim=True)
    group_max = G.amax(dim=-1, keepdim=True)
    spacer = (group_max - group_min) / (number_of_levels - 1)
    spacer = torch.where(spacer == 0, 1.0, spacer)
    q = torch.clamp(torch.round((G - group_min) / spacer), 0, number_of_levels - 1)
    return (q * spacer + group_min).reshape(matrix.shape)
