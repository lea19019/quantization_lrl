'''
The tokenizer of the from-scratch model, and the token files it writes.

    train_tokenizer(config, test)          learn the BPE merges, save models/<name>/
    encode_corpus(config, index, test)     write one corpus as token ids

Byte-level BPE (Sennrich 2016) learned from a sample of the free text in which language l is
drawn with probability ∝ p_l^alpha, p_l its share of all characters (Conneau 2020). Text is
NFC-normalised and split by the GPT-4o pattern before the merges. The tokenizer adds <s> in
front of every text.

Token files: every document as <s> ids </s>, one after another, in
<tokens>/<source>/<folder>/train.bin and valid.bin (uint16: 65,536 ids fit exactly), with
manifest.json. Validation text is valid.parquet, or for parallel-only corpora the target side
of <dev_splits>/<lang>/dev.jsonl.
'''
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

BATCH_DOCS = 10_000  # documents read and encoded at a time


# ---- Reading the free text ----

def corpora(config: dict) -> list[dict]:
    """Return every corpus as {source, folder, lang, path}, sorted: the Slurm array index."""
    root, found = Path(config["freetext"]), []
    for source in ("fineweb", "parallel", "wikipedia"):
        languages = config.get(source)  # parallel folders are already named by language
        for path in sorted(p for p in (root / source).iterdir() if p.is_dir()):
            lang = languages[path.name] if languages else path.name
            found.append({"source": source, "folder": path.name, "lang": lang, "path": path})
    return found


def texts(path: Path, limit: int | None):
    """Yield the documents of a parquet text column in lists, at most limit documents."""
    seen = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_DOCS, columns=["text"]):
        docs = batch.column("text").to_pylist()
        if limit is not None:
            docs = docs[: limit - seen]
        seen += len(docs)
        yield docs
        if limit is not None and seen >= limit:
            return


def valid_texts(corpus: dict, config: dict, limit: int | None):
    """Yield the corpus's validation documents: valid.parquet, else its dev targets."""
    if (corpus["path"] / "valid.parquet").exists():
        yield from texts(corpus["path"] / "valid.parquet", limit)
        return
    dev = Path(config["dev_splits"]) / corpus["lang"] / "dev.jsonl"
    if corpus["source"] == "parallel" and corpus["lang"] != "eng" and dev.exists():
        lines = dev.read_text(encoding="utf-8").splitlines()
        yield [json.loads(line)["tgt"] for line in lines][:limit]


# ---- Training the tokenizer ----

def sampling_rates(chars: dict[str, int], alpha: float, sample_chars: float) -> dict[str, float]:
    """Return the fraction of each language's documents kept in the sample.

    Language l gets sample_chars * p_l^alpha / sum_k p_k^alpha characters, p_l = chars_l / total;
    a language with less text than that is kept whole (rate 1).
    """
    total = sum(chars.values())
    weights = {lang: (count / total) ** alpha for lang, count in chars.items()}
    norm = sum(weights.values())
    rates = {}
    for lang, count in chars.items():
        quota = sample_chars * weights[lang] / norm
        rates[lang] = min(1.0, quota / count) if count else 0.0
    return rates


def build(config: dict, vocab_size: int) -> tuple[Tokenizer, trainers.BpeTrainer]:
    """Return an empty byte-level BPE tokenizer and its trainer."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(config["pattern"]), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),  # bytes as symbols
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=config["special_tokens"], show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())  # all 256 bytes, seen or not
    return tokenizer, trainer


def save(tokenizer: Tokenizer, folder: Path, manifest: dict) -> None:
    """Add <s> in front of every text; write the Hugging Face files and manifest.json."""
    bos, eos, pad = manifest["config"]["special_tokens"]
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{bos} $A", special_tokens=[(bos, tokenizer.token_to_id(bos))])
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token=bos, eos_token=eos,
                                      pad_token=pad)
    wrapped.save_pretrained(folder)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"saved {folder}", flush=True)


def train_tokenizer(config: dict, test: bool) -> None:
    """Count the characters of each language, sample them with p^alpha, learn the merges."""
    limit = config["test"]["docs"] if test else None
    settings = config["test"] if test else config
    found, chars = corpora(config), {}
    for corpus in found:
        count = sum(len(doc) for docs in texts(corpus["path"] / "train.parquet", limit)
                    for doc in docs)
        chars[corpus["lang"]] = chars.get(corpus["lang"], 0) + count
        print(corpus["source"], corpus["folder"], corpus["lang"], count, flush=True)
    rates = sampling_rates(chars, config["alpha"], settings["sample_chars"])

    def sample():  # each document kept with its language's rate, by a seeded draw
        for i, corpus in enumerate(found):
            rng = np.random.default_rng([config["seed"], i])
            for docs in texts(corpus["path"] / "train.parquet", limit):
                keep = rng.random(len(docs)) < rates[corpus["lang"]]
                yield [doc for doc, kept in zip(docs, keep) if kept]

    tokenizer, trainer = build(config, settings["vocab_size"])
    tokenizer.train_from_iterator(sample(), trainer)
    manifest = {"test": test, "git_sha": git_sha(), "vocab_size": tokenizer.get_vocab_size(),
                "chars": chars, "rates": rates,
                "sampled_chars": {lang: round(chars[lang] * rates[lang]) for lang in chars},
                "config": config}
    save(tokenizer, Path(config["output"]) / (config["name"] + ("_test" if test else "")),
         manifest)


# ---- Writing the token files ----

def encode(batches, tokenizer: Tokenizer, path: Path) -> dict:
    """Write <s> ids </s> of every document to path as uint16; return documents and tokens."""
    eos = tokenizer.token_to_id("</s>")
    docs = tokens = 0
    with open(path, "wb") as file:
        for batch in batches:
            for encoding in tokenizer.encode_batch(batch):  # <s> comes from the tokenizer
                ids = np.array(encoding.ids + [eos], dtype=np.uint16)
                ids.tofile(file)
                tokens += len(ids)
            docs += len(batch)
    return {"docs": docs, "tokens": tokens}


def encode_corpus(config: dict, index: int, test: bool) -> None:
    """Encode the corpus at this index into <tokens>/<source>/<folder>/."""
    suffix = "_test" if test else ""
    limit = config["test"]["docs"] if test else None
    tokenizer = Tokenizer.from_file(str(Path(config["output"]) / (config["name"] + suffix)
                                        / "tokenizer.json"))
    assert tokenizer.get_vocab_size() <= 2**16, "ids must fit in uint16"
    corpus = corpora(config)[index]
    folder = Path(config["tokens"] + suffix) / corpus["source"] / corpus["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").unlink(missing_ok=True)  # written last: present means complete
    counts = {
        "train": encode(texts(corpus["path"] / "train.parquet", limit), tokenizer,
                        folder / "train.bin"),
        "valid": encode(valid_texts(corpus, config, limit), tokenizer, folder / "valid.bin"),
    }
    manifest = {"source": corpus["source"], "folder": corpus["folder"], "lang": corpus["lang"],
                "tokenizer": config["name"] + suffix, "counts": counts}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=False).stdout.strip()
