'''
WikiText-2 perplexity: the standard check that a quantizer works.

Protocol (GPTQ paper, Frantar et al. 2023, Sec. 5; most quantization papers follow it):
  1. Join the WikiText-2 *test* split into one text and tokenize it once.
  2. Cut the token stream into non-overlapping windows of 2048 tokens; drop the remainder.
  3. In each window, every position predicts the next token; keep log p(true next token).
  4. Perplexity = exp(-mean log-prob over all windows).
A good 4-bit quantizer raises perplexity only a little above the full-precision model.
'''
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
import argparse

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from qlrl.quant.input_scales import load_input_scales
from qlrl.quant.rtn import _load_model, gold_logprobs, perplexity
from qlrl.quant.smoothquant import add_activation_quantization


def load_test_ids(tokenizer: PreTrainedTokenizerBase, parquet_path: str) -> torch.Tensor:
    """Return the whole test split as one stream of token ids, shape [tokens]."""
    lines = load_dataset("parquet", data_files=parquet_path, split="train")["text"]
    # "\n\n".join matches the reference GPTQ evaluation, so numbers compare with papers.
    return tokenizer("\n\n".join(lines), return_tensors="pt")["input_ids"][0]


def window_logprobs(
    model: PreTrainedModel, ids: torch.Tensor, seq_len: int, max_windows: int | None = None
) -> torch.Tensor:
    """Return log p(true next token) at every position of every full window, concatenated."""
    window_count = ids.numel() // seq_len
    if max_windows is not None:
        window_count = min(window_count, max_windows)
    device = next(model.parameters()).device
    gold = []
    with torch.no_grad():
        for index in range(window_count):
            window = ids[index * seq_len : (index + 1) * seq_len].unsqueeze(0).to(device)
            logits = model(window, use_cache=False).logits.float()  # [1, seq_len, vocab]
            gold.append(gold_logprobs(logits, window).cpu())  # [seq_len - 1]
    return torch.cat(gold)


def _parse_args() -> argparse.Namespace:
    """Return the model paths to evaluate and the evaluation settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_paths", nargs="+")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--test-file",
        default="data/wikitext-2-raw-v1/wikitext-2-raw-v1/test-00000-of-00001.parquet",
    )
    parser.add_argument("--test", action="store_true")
    # W8A8 models (smoothquant.py): also round every linear input to int8 per token.
    parser.add_argument("--quantize-activations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seq_len, max_windows = (64, 2) if args.test else (args.seq_len, None)
    for model_path in args.model_paths:
        model, tokenizer = _load_model(model_path, args.test)
        # SmoothQuant / AWQ (version 2) divide each linear's input by s in a hook; hooks are
        # not saved with the weights, so attach them again from input_scales.pt if present.
        # This must come before activation rounding: x is divided by s, then rounded.
        load_input_scales(model, model_path)
        if args.quantize_activations:
            add_activation_quantization(model)
        ids = load_test_ids(tokenizer, args.test_file)
        # Never longer than the model can read (GPT-2: 1024); the printout shows what was used.
        model_seq_len = min(seq_len, model.config.max_position_embeddings)
        gold = window_logprobs(model, ids, model_seq_len, max_windows)
        print(
            f"{model_path}: perplexity {perplexity(gold):.3f} "
            f"({gold.numel()} predicted tokens, windows of {model_seq_len})",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
