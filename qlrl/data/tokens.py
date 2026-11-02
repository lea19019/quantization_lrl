'''
Token files of the from-scratch model: every corpus encoded once, then read as training windows.

    python -m qlrl.data.tokens configs/data/tokenizer.yaml <index> [--test]     one corpus

    data = TokenData("/.../pretrain_tokens", seq_len=2048, seed=0)
    data.batch(order_indices)      # [batch, seq_len + 1] token ids, in the seeded order
    data.valid["yor"]              # Yoruba's validation token ids, one flat array

Each document is <s> tokens </s>, all documents of a corpus one after another in
<tokens>/<source>/<folder>/train.bin (uint16: a 65,536-token vocabulary fits exactly), its
validation text in valid.bin, counts and language in manifest.json. Free-text corpora validate
on their valid.parquet; parallel-only text on the target side of <dev_splits>/<lang>/dev.jsonl
(the church English side has none: English validates on Wikipedia).

Training reads every train.bin as one stream cut into windows of seq_len + 1 tokens (input and
next-token target), shuffled once by the seed: every token is seen once, in proportion to the
data (the natural mix, pretraining_model.md §3). A window may span two documents or corpora; the
<s> and </s> tokens mark where one ends.
'''
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from tokenizers import Tokenizer

BATCH_DOCS = 10_000  # documents encoded per call (the Rust tokenizer spreads them over the cores)


def corpora(config: dict) -> list[dict]:
    """Return every corpus as {source, folder, lang, path}, sorted: the Slurm array index."""
    root, found = Path(config["freetext"]), []
    for source in ("fineweb", "parallel", "wikipedia"):
        names = config.get(source)  # parallel folders are named by their language
        for path in sorted(p for p in (root / source).iterdir() if p.is_dir()):
            lang = names[path.name] if names else path.name
            found.append({"source": source, "folder": path.name, "lang": lang, "path": path})
    return found


def texts(path: Path, limit: int | None = None):
    """Yield lists of documents from a parquet file's text column, at most limit in all."""
    seen, size = 0, min(BATCH_DOCS, limit or BATCH_DOCS)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=size, columns=["text"]):
        docs = batch.column("text").to_pylist()
        if limit is not None:
            docs = docs[: limit - seen]
        seen += len(docs)
        yield docs
        if limit is not None and seen >= limit:
            return


def valid_texts(corpus: dict, config: dict, limit: int | None):
    """Yield the corpus's validation documents (valid.parquet, else its dev targets)."""
    if (corpus["path"] / "valid.parquet").exists():
        yield from texts(corpus["path"] / "valid.parquet", limit)
        return
    dev = Path(config["dev_splits"]) / corpus["lang"] / "dev.jsonl"
    if corpus["source"] == "parallel" and corpus["lang"] != "eng" and dev.exists():
        with open(dev, encoding="utf-8") as handle:
            yield [json.loads(line)["tgt"] for line in handle][:limit]


def encode(batches, tokenizer: Tokenizer, path: Path) -> dict:
    """Write <s> ids </s> of every document to path as uint16; return documents and tokens."""
    eos = tokenizer.token_to_id("</s>")
    docs = tokens = 0
    with open(path, "wb") as handle:
        for batch in batches:
            for encoding in tokenizer.encode_batch(batch):  # the post-processor adds <s>
                ids = np.array(encoding.ids + [eos], dtype=np.uint16)
                ids.tofile(handle)
                tokens += len(ids)
            docs += len(batch)
    return {"docs": docs, "tokens": tokens}


def main() -> None:
    """Encode the corpus at the given array index into <tokens>/<source>/<folder>/."""
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    index = int(next(arg for arg in sys.argv[2:] if arg.isdigit()))
    test = "--test" in sys.argv
    limit = config["test"]["docs"] if test else None
    name = config["name"] + ("_test" if test else "")
    tokenizer = Tokenizer.from_file(str(Path(config["output"]) / name / "tokenizer.json"))
    assert tokenizer.get_vocab_size() <= 2**16, "ids must fit in uint16"
    corpus = corpora(config)[index]
    root = Path(config["tokens"] + ("_test" if test else ""))
    folder = root / corpus["source"] / corpus["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").unlink(missing_ok=True)  # written last: present means complete
    counts = {"train": encode(texts(corpus["path"] / "train.parquet", limit), tokenizer,
                              folder / "train.bin"),
              "valid": encode(valid_texts(corpus, config, limit), tokenizer, folder / "valid.bin")}
    manifest = {"source": corpus["source"], "folder": corpus["folder"], "lang": corpus["lang"],
                "tokenizer": name, "counts": counts}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


class TokenData:
    def __init__(self, folder: str, seq_len: int, seed: int):
        manifests = sorted(Path(folder).glob("*/*/manifest.json"))
        self.seq_len = seq_len
        self.arrays = [np.memmap(path.parent / "train.bin", dtype=np.uint16, mode="r")
                       for path in manifests]
        self.offsets = np.cumsum([0] + [len(array) for array in self.arrays])  # stream starts
        self.windows = int(self.offsets[-1]) // (seq_len + 1)
        self.order = np.random.default_rng(seed).permutation(self.windows)
        self.langs = [json.loads(path.read_text())["lang"] for path in manifests]
        self.valid = self.read_valid(manifests)

    def read_valid(self, manifests: list[Path]) -> dict[str, np.ndarray]:
        """Return each language's validation ids, its corpora's valid.bin joined."""
        parts: dict[str, list[np.ndarray]] = {}
        for path, lang in zip(manifests, self.langs):
            parts.setdefault(lang, []).append(np.fromfile(path.parent / "valid.bin", np.uint16))
        return {lang: np.concatenate(arrays) for lang, arrays in parts.items()
                if sum(len(a) for a in arrays) > 1}

    def tokens(self) -> dict[str, int]:
        """Return the train tokens of each language: its share of the stream."""
        counts: dict[str, int] = {}
        for array, lang in zip(self.arrays, self.langs):
            counts[lang] = counts.get(lang, 0) + len(array)
        return counts

    def window(self, k: int) -> np.ndarray:
        """Return window k of the stream, seq_len + 1 ids, read across files if it spans two."""
        start, stop = k * (self.seq_len + 1), (k + 1) * (self.seq_len + 1)
        first = int(np.searchsorted(self.offsets, start, side="right")) - 1
        pieces = []
        while start < stop:
            begin = start - int(self.offsets[first])
            end = min(stop, int(self.offsets[first + 1])) - int(self.offsets[first])
            pieces.append(self.arrays[first][begin:end])
            start += end - begin
            first += 1
        return np.concatenate(pieces)

    def batch(self, positions: list[int]) -> torch.Tensor:
        """Return the windows at these positions of the shuffled order: [batch, seq_len + 1]."""
        rows = [self.window(int(self.order[p])).astype(np.int64) for p in positions]
        return torch.from_numpy(np.stack(rows))


if __name__ == "__main__":
    main()
