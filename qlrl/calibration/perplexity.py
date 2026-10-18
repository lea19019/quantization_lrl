'''
WikiText-2 perplexity
Validation only: checks that a quantizer works (docs/project/evaluation.md).

• Roadmap:
  1. Load the test split from data/wikitext-2-raw-v1/ ("text" column).
  2. Join all lines into one string with "\n\n" between them.
  3. Tokenize it once into [1, total_tokens].
  4. Cut into non-overlapping windows of L tokens; drop the leftover.
  5. For each window, run the model and keep gold_logprobs() from utils.py.
  6. Concatenate all windows' log-probs and compute perplexity() from utils.py.
  7. Do it for full precision, RTN and GPTQ with the same windows.
  8. Save one results folder per model with a manifest.json.

• Sizes:
  - L = 1024 for GPT-2, 2048 for Llama 3.1 8B.
  - --test: 2 windows of 64 tokens.

• Still to validate:
  - Full precision < GPTQ < RTN.
  - The three models see exactly the same windows.
  - Only the log-probs are kept per window, never the full logits.
  - Every perplexity is finite.
'''

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers import PreTrainedTokenizerBase

from qlrl.quant.utils import gold_logprobs, perplexity

WIKITEXT_TEST = "data/wikitext-2-raw-v1/wikitext-2-raw-v1/test-00000-of-00001.parquet"


def load_wikitext2(tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    # One parquet file loads as a split called "train"; it is the WikiText-2 test split.
    lines = load_dataset("parquet", data_files=WIKITEXT_TEST, split="train")["text"]
    return tokenizer("\n\n".join(lines), return_tensors="pt").input_ids  # [1, total_tokens]


def wikitext2_perplexity(
        model: PreTrainedModel, ids: torch.Tensor, length: int, max_windows: int | None = None
    ) -> tuple[float, int]:
    # Non-overlapping windows of `length` tokens; `//` drops the leftover at the end.
    number_of_windows = ids.shape[1] // length
    if max_windows: number_of_windows = min(number_of_windows, max_windows)

    gold = []
    with torch.no_grad():
        for i in range(number_of_windows):
            window = ids[:, i * length:(i + 1) * length].to(model.device)  # [1, length]
            logits = model(window, use_cache=False).logits.float()  # bf16 logits lose precision
            gold.append(gold_logprobs(logits, window))  # keep [length - 1], drop the logits
    return perplexity(torch.cat(gold)), number_of_windows


def main() -> None:
    # GPT-2: float32, length 1024. Llama 3.1 8B: its tokenizer, torch.bfloat16, length 2048.
    model_folder, tokenizer_name, dtype, length = "models/gpt2-gptq-4bit", "gpt2", torch.float32, 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_folder, dtype=dtype, device_map=device).eval()
    ids = load_wikitext2(tokenizer)

    # 4 windows keep it quick on the login node; None runs all 280 (about 16 minutes).
    ppl, windows = wikitext2_perplexity(model, ids, length, max_windows=4)
    print(f"GPTQ perplexity {ppl:.3f} over {windows} windows of {length}")


if __name__ == "__main__":
    main()
