'''
Calibration data shared by the quantizers.

C4 (Raffel et al. 2020) is a large cleaned web crawl. GPTQ, AWQ and SmoothQuant all calibrate
on generic web or Pile text so that the statistics they collect (Hessians, activation sizes)
describe the language in general, not the test set they are evaluated on.
'''
import random

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def load_c4_calibration_ids(
    tokenizer: PreTrainedTokenizerBase, json_path: str, samples: int, seq_len: int, seed: int
) -> torch.Tensor:
    """Return `samples` random windows of `seq_len` token ids from C4, shape [samples, seq_len]."""
    # Same sampling as the reference GPTQ code (datautils.get_c4): pick a random document
    # that is longer than one window, then a random window inside it.
    documents = load_dataset("json", data_files=json_path, split="train")["text"]
    generator = random.Random(seed)
    windows = []
    while len(windows) < samples:
        text = documents[generator.randrange(len(documents))]
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        if ids.numel() <= seq_len:
            continue
        start = generator.randrange(ids.numel() - seq_len)
        windows.append(ids[start : start + seq_len])
    return torch.stack(windows)
