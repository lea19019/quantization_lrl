'''
Tokenizer of the from-scratch model:
    python -m qlrl.data.tokenizer configs/data/tokenizer.yaml [--test]

Byte-level BPE (Sennrich 2016 merges over UTF-8 bytes, so no text is ever unknown) learned
from a sample of the free text in which each language l is drawn with probability
q_l ∝ p_l^alpha, p_l its share of all characters (Conneau 2020, alpha = 0.3). alpha = 1 keeps
the natural shares and spends the vocabulary on the large languages; alpha = 0 gives every
language the same; 0.3 flattens the shares so small languages get their own merges without
filling the vocabulary with words the model rarely trains on (Land 2024). Sizes are counted in
characters, not bytes: a Burmese character takes three UTF-8 bytes (Arnett 2024).

Text is NFC-normalised, split by the GPT-4o pattern, then merged. The tokenizer adds <s> in
front of every text, as the evaluation prompts expect. Writes <output>/<name>/ (Hugging Face
files, loadable by AutoTokenizer) and manifest.json with the characters and sample per language.
'''
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
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

from qlrl.data.tokens import BATCH_DOCS, corpora, texts


def count_chars(path: Path, limit: int | None) -> int:
    """Return the characters of a parquet text column, over at most limit documents."""
    if limit is not None:
        return sum(len(doc) for docs in texts(path, limit) for doc in docs)
    batches = pq.ParquetFile(path).iter_batches(batch_size=BATCH_DOCS, columns=["text"])
    return sum(pc.sum(pc.utf8_length(batch.column("text"))).as_py() or 0 for batch in batches)


def sampling_rates(chars: dict[str, int], alpha: float, sample_chars: float) -> dict[str, float]:
    """Return the fraction of each language's documents kept in the sample.

    Language l gets sample_chars * p_l^alpha / sum_k p_k^alpha characters, p_l = chars_l / total.
    A language with less text than its quota is kept whole (rate 1).
    """
    total = sum(chars.values())
    weights = {lang: (count / total) ** alpha for lang, count in chars.items()}
    norm = sum(weights.values())
    return {lang: min(1.0, sample_chars * weights[lang] / norm / chars[lang]) if chars[lang] else 0
            for lang in chars}


def sample(found: list[dict], rates: dict[str, float], seed: int, limit: int | None):
    """Yield the sampled documents: each kept with its language's rate, by a seeded draw."""
    for i, corpus in enumerate(found):
        rng = np.random.default_rng([seed, i])
        for docs in texts(corpus["path"] / "train.parquet", limit):
            keep = rng.random(len(docs)) < rates[corpus["lang"]]
            yield [doc for doc, kept in zip(docs, keep) if kept]


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
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes, seen or not
    )
    return tokenizer, trainer


def save(tokenizer: Tokenizer, folder: Path, manifest: dict) -> None:
    """Add <s> in front of every text, write the Hugging Face files and manifest.json."""
    bos, eos, pad = manifest["config"]["special_tokens"]
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{bos} $A", special_tokens=[(bos, tokenizer.token_to_id(bos))])
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token=bos, eos_token=eos,
                                      pad_token=pad)
    wrapped.save_pretrained(folder)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"saved {folder}", flush=True)


def main() -> None:
    """Count characters per language, sample with p^alpha, learn the merges, save."""
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    test = "--test" in sys.argv
    limit = config["test"]["docs"] if test else None
    sample_chars = config["test"]["sample_chars"] if test else config["sample_chars"]
    vocab_size = config["test"]["vocab_size"] if test else config["vocab_size"]
    found = corpora(config)
    chars: dict[str, int] = {}
    for corpus in found:
        count = count_chars(corpus["path"] / "train.parquet", limit)
        chars[corpus["lang"]] = chars.get(corpus["lang"], 0) + count
        print(corpus["source"], corpus["folder"], corpus["lang"], count, flush=True)
    rates = sampling_rates(chars, config["alpha"], sample_chars)
    tokenizer, trainer = build(config, vocab_size)
    tokenizer.train_from_iterator(sample(found, rates, config["seed"], limit), trainer)
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=False).stdout.strip()
    manifest = {"test": test, "git_sha": git_sha, "vocab_size": tokenizer.get_vocab_size(),
                "chars": chars, "rates": rates,
                "sampled_chars": {lang: round(chars[lang] * rates[lang]) for lang in chars},
                "config": config}
    save(tokenizer, Path(config["output"]) / (config["name"] + ("_test" if test else "")),
         manifest)


if __name__ == "__main__":
    main()
