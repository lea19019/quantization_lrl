'''
Helpers used by archive/_rtn.py and archive/_gptq.py: run a model on one string, generate,
and cache a quantized model on disk. Moved out of rtn.py by Claude Code on 2026-10-15 at Adrian's
request.
'''
from collections.abc import Callable
from pathlib import Path
from typing import cast

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
