'''
The token files read as training windows.

    data = TokenData("/.../pretrain_tokens", seq_len=2048, seed=0)
    data.batch([0, 1, 2])     # [3, seq_len + 1] ids: positions 0..2 of the shuffled order
    data.valid["yor"]         # Yoruba's validation ids, one flat array

Every train.bin is joined into one stream, cut into windows of seq_len + 1 tokens (the input and
its next-token target), and shuffled once by the seed. Each token is seen once, in proportion to
the data. A window may span two documents or two files; <s> and </s> mark the boundaries.
'''
import json
from pathlib import Path

import numpy as np
import torch


class TokenData:
    def __init__(self, folder: str, seq_len: int, seed: int):
        manifests = sorted(Path(folder).glob("*/*/manifest.json"))
        self.size = seq_len + 1
        self.langs = [json.loads(path.read_text())["lang"] for path in manifests]
        self.arrays = [np.memmap(path.parent / "train.bin", dtype=np.uint16, mode="r")
                       for path in manifests]
        self.starts = np.cumsum([0] + [len(array) for array in self.arrays])  # in the stream
        self.windows = int(self.starts[-1]) // self.size
        self.order = np.random.default_rng(seed).permutation(self.windows)

        valid: dict[str, list[np.ndarray]] = {}
        for path, lang in zip(manifests, self.langs):
            valid.setdefault(lang, []).append(np.fromfile(path.parent / "valid.bin", np.uint16))
        self.valid = {lang: np.concatenate(parts) for lang, parts in valid.items()}
        self.valid = {lang: ids for lang, ids in self.valid.items() if len(ids) > 1}

    def tokens(self) -> dict[str, int]:
        """Return the train tokens of each language."""
        counts: dict[str, int] = {}
        for array, lang in zip(self.arrays, self.langs):
            counts[lang] = counts.get(lang, 0) + len(array)
        return counts

    def window(self, k: int) -> np.ndarray:
        """Return window k of the stream, reading from two or more files if it spans them."""
        start, stop = k * self.size, (k + 1) * self.size
        file = int(np.searchsorted(self.starts, start, side="right")) - 1
        pieces = []
        while start < stop:
            end = min(stop, int(self.starts[file + 1]))
            offset = int(self.starts[file])
            pieces.append(self.arrays[file][start - offset:end - offset])
            start, file = end, file + 1
        return np.concatenate(pieces)

    def batch(self, positions) -> torch.Tensor:
        """Return the windows at these positions of the shuffled order: [batch, seq_len + 1]."""
        rows = [self.window(int(self.order[p])).astype(np.int64) for p in positions]
        return torch.from_numpy(np.stack(rows))
