'''
C4 calibration for GPTQ. 
• Roadmap:
  1. Load data/c4/c4-train.00000-of-01024.json.gz; each row has a "text" field.
  2. Seed a random generator so the same seed always gives the same windows.
  3. Pick a random document and tokenize it; skip it if it has L tokens or fewer.
  4. Take L consecutive tokens from a random start position.
  5. Repeat until there are N windows; stack them into [N, L] token ids.
  6. In get_calibration, run the windows one at a time through the model.
  7. In the hook, add H[name] += X @ X.T instead of saving X.
  8. Return the H's; gptq() uses them directly instead of computing X @ X.T.

• Sizes:
  - N = 128 windows (GPTQ paper).
  - L = 1024 for GPT-2 (its maximum), 2048 for Llama 3.1 8B.
  - Five seeds, one per calibration draw (design.md).
  - --test: N = 2, L = 64.

• Still to validate:
  - The same seed gives identical windows; a different seed gives different ones.
  - Every window has exactly L tokens.
  - H added up window by window equals X @ X.T of all windows at once.
  - Calibration text never comes from the evaluation data.
'''

import random

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

C4_FILE = "data/c4/c4-train.00000-of-01024.json.gz"


def load_c4() -> Dataset:
    return load_dataset("json", data_files=C4_FILE, split="train")


def sample_windows(
    tokenizer: PreTrainedTokenizerBase, documents: Dataset, n: int, length: int, seed: int
) -> torch.Tensor:
    rng = random.Random(seed)
    windows = []
    while len(windows) < n:
        text = documents[rng.randrange(len(documents))]["text"]
        ids = tokenizer(text, return_tensors="pt").input_ids  # [1, tokens]
        if ids.shape[1] <= length: continue
        start = rng.randint(0, ids.shape[1] - length - 1)
        windows.append(ids[:, start:start + length])
    return torch.cat(windows)  # [n, length]

