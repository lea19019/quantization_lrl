'''
Calibration: C4 windows, and the hooks that record what each linear receives on them.
GPTQ and AWQ use both; RTN uses neither.

• Roadmap:
  1. Load data/c4/c4-train.00000-of-01024.json.gz; each row has a "text" field.
  2. Seed a random generator so the same seed always gives the same windows.
  3. Pick a random document and tokenize it; skip it if it has L tokens or fewer.
  4. Take L consecutive tokens from a random start position.
  5. Repeat until there are N windows; stack them into [N, L] token ids.
  6. collect_inputs: run the windows one at a time, with a hook on every linear.
  7. In the hook, fold X into a running statistic instead of saving X:
     H += X X^T (GPTQ, AWQ), sum|x| (AWQ).

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
from collections.abc import Callable
from typing import TypeVar

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from qlrl.quant.base import linears

C4_FILE = "data/c4/c4-train.00000-of-01024.json.gz"

# The statistic a method keeps per linear: H or (H, sum|x|).
Statistic = TypeVar("Statistic")


def load_c4() -> Dataset:
    """Return the C4 documents of C4_FILE; each row has a "text" field."""
    return load_dataset("json", data_files=C4_FILE, split="train")


def sample_windows(
    tokenizer: PreTrainedTokenizerBase, documents: Dataset, n: int, length: int, seed: int
) -> torch.Tensor:
    """Return n random windows of `length` token ids, [n, length], drawn with `seed`."""
    rng = random.Random(seed)
    windows = []
    while len(windows) < n:
        text = documents[rng.randrange(len(documents))]["text"]
        ids = tokenizer(text, return_tensors="pt").input_ids  # [1, tokens]
        if ids.shape[1] <= length: continue
        start = rng.randint(0, ids.shape[1] - length - 1)
        windows.append(ids[:, start:start + length])
    return torch.cat(windows)  # [n, length]


def collect_inputs(
        model: torch.nn.Module,
        run: Callable[[], object],
        update: Callable[[Statistic | None, torch.Tensor], Statistic],
    ) -> dict[str, Statistic]:
    """Call run() with a hook on every linear of `model`; return { linear name: statistic }.

    update(old, X) folds one window's inputs X [in, tokens] into the statistic; old is
    None on the first window. Only the statistic is kept, never X itself.
    """
    stats: dict[str, Statistic] = {}
    handles = []

    def make_save_input(name):
        def save_input(_module, args):
            x = args[0].detach()
            X = x.reshape(-1, x.shape[-1]).T.float()  # [input_features, tokens]
            stats[name] = update(stats.get(name), X)
        return save_input

    for name, module in linears(model):
        handles.append(module.register_forward_pre_hook(make_save_input(name)))
    run()
    for handle in handles: handle.remove()
    return stats


def run_windows(model: PreTrainedModel, input_ids: torch.Tensor) -> None:
    """Run windows [N, L] through `model` one at a time; the outputs are not kept."""
    with torch.no_grad():
        for i in range(input_ids.shape[0]):
            # use_cache=False: otherwise a key/value cache is built and kept for nothing.
            model(input_ids[i:i + 1].to(model.device), use_cache=False)
