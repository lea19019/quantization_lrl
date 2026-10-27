'''
Clean the free text used for pretraining (Wikipedia and FineWeb-2), one corpus per run:
    python -m qlrl.data.freetext configs/data/freetext.yaml <index> [--test]
In a Slurm array the task index picks the corpus from the config's `corpora` list.

A seeded sample of at most `max_docs` documents per corpus is cleaned (6.4M English articles
are far more than any share will use). Two passes over that sample:
1. Count the skeletons of all documents (text_filters.skeleton, over the `frequent_words`
   found in the most documents) to find the templates: a
   skeleton seen `min_count` times or more. Only the first `keep` documents of each stay.
2. Per document: drop it if its site is blocked (FineWeb-2) or it is a surplus template copy;
   otherwise drop its non-prose (Wikipedia), other-language and contaminated lines
   (text_filters.Cleaner). A seeded `fraction` of the documents left, lowered so that about
   `max_docs` are chosen (the number left is known from pass 1), becomes the validation slice.

Writes <output>/<source>/<code>/{train,valid}.parquet (columns id, text) and manifest.json.
--test: first TEST_DOCS documents, into <output>_test.
'''
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
import hashlib
import json
import subprocess
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from qlrl.data.text_filters import WORD, Cleaner, blocked, script_chars, script_of, skeleton

TEST_DOCS = 2000
SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string())])


def sampled(doc_id: str, config: dict, salt: str, fraction: float) -> bool:
    """Return True for a seeded `fraction` of documents; the hash is stable across runs."""
    digest = hashlib.sha256(f"{config['seed']}:{salt}:{doc_id}".encode()).hexdigest()
    return int(digest, 16) % 10**6 < fraction * 10**6


def corpus_files(config: dict, source: str, code: str) -> tuple[list[Path], int]:
    """Return the parquet files of one corpus and its number of documents."""
    if source == "wikipedia":
        files = sorted(Path(config["wikipedia"]).glob(f"20231101.{code}/*.parquet"))
    else:
        files = sorted(Path(config["fineweb"]).glob(f"{code}/*/*.parquet"))
    return files, sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def documents(config: dict, source: str, code: str, limit: int | None):
    """Yield (id, text, url) for a seeded sample of at most `max_docs` documents of one corpus.

    url is "" for Wikipedia. The sample is by hash, so it spreads over the whole corpus.
    """
    files, total = corpus_files(config, source, code)
    columns = ["id", "text"] + (["url"] if source == "fineweb" else [])
    rows = (row for path in files
            for batch in pq.ParquetFile(path).iter_batches(1000, columns=columns)
            for row in batch.to_pylist())
    fraction = min(1.0, config["max_docs"] / max(total, 1))
    kept = (row for row in rows if sampled(str(row["id"]), config, "sample", fraction))
    for row in islice(kept, limit):
        yield str(row["id"]), row["text"], row.get("url", "")


def is_blocked(url: str, config: dict) -> bool:
    """Return True if the document comes from a blocked site (FineWeb-2 only has urls)."""
    return bool(url) and blocked(url, config["blocked_sites"], config["blocked_url_words"])


def find_templates(config: dict, source: str, code: str, cleaner: Cleaner, limit, keep) -> tuple:
    """Return (frequent words, set of template skeletons): the first pass over the corpus.

    Only the documents `keep(text)` accepts (all, or those of the corpus's script) count.
    """
    settings = config["templates"]
    sample = (doc for doc in documents(config, source, code, None) if keep(doc[1]))
    sample = islice(sample, min(limit or 10**12, settings["vocabulary_docs"]))
    # Document frequency, not word count: a genus name repeated in its 300 species stubs is
    # common by count but not spread over documents, and would split the template by genus.
    words = Counter(word for _, text, _ in sample for word in set(WORD.findall(text.lower())))
    frequent = {word for word, _ in words.most_common(settings["frequent_words"])}
    prose = (cleaner.prose(text) for _, text, url in documents(config, source, code, limit)
             if keep(text) and not is_blocked(url, config))
    counts = Counter(skeleton(text, frequent) for text in prose if text.strip())
    cleaner.counts.clear()  # the second pass counts the removed lines again
    templates = {key for key, n in counts.items() if n >= settings["min_count"]}
    kept = sum(min(n, settings["keep"]) if key in templates else n for key, n in counts.items())
    return frequent, templates, kept


def validation_fraction(config: dict, kept: int) -> float:
    """Return `fraction`, lowered so that about `max_docs` of the `kept` documents are chosen."""
    settings = config["validation"]
    return min(settings["fraction"], settings["max_docs"] / max(kept, 1))


def in_validation(doc_id: str, config: dict, fraction: float) -> bool:
    """Return True for a seeded `fraction` of documents: spread over the whole sample."""
    return sampled(doc_id, config, "valid", fraction)


class Writer:
    """Writes the train and valid parquet files of one corpus in chunks."""

    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.folder = folder  # written to .tmp files, renamed only when complete
        self.writers = {split: pq.ParquetWriter(folder / f"{split}.parquet.tmp", SCHEMA)
                        for split in ("train", "valid")}
        self.buffers = {"train": [], "valid": []}

    def add(self, split: str, doc_id: str, text: str) -> None:
        """Queue one document; write the split's queue every 10,000 documents."""
        self.buffers[split].append({"id": doc_id, "text": text})
        if len(self.buffers[split]) >= 10000:
            self.writers[split].write_table(pa.Table.from_pylist(self.buffers[split], SCHEMA))
            self.buffers[split] = []

    def close(self) -> None:
        """Write what is left and close both files."""
        for split, writer in self.writers.items():
            writer.write_table(pa.Table.from_pylist(self.buffers[split], SCHEMA))
            writer.close()
            (self.folder / f"{split}.parquet.tmp").rename(self.folder / f"{split}.parquet")


def run(config: dict, corpus: list, folder: Path, limit: int | None) -> dict:
    """Clean one corpus and write its train and valid files; return the counts.

    corpus is [source, code, labels] or [source, code, labels, script]: then only the
    documents written in that script (zh: hans or hant) are kept.
    """
    source, code, labels, script = (corpus + [None])[:4]
    chars = script_chars(config["scripts"]) if script else None
    ratio = config["scripts"]["ratio"] if script else None
    found = (lambda text: script_of(text, chars, ratio)) if script else (lambda text: None)
    cleaner = Cleaner(config, source == "wikipedia", labels)
    frequent, templates, kept = find_templates(config, source, code, cleaner, limit,
                                               lambda text: found(text) == script)
    writer, copies = Writer(folder), Counter()
    counts, fraction = cleaner.counts, validation_fraction(config, kept)
    for doc_id, text, url in documents(config, source, code, limit):
        counts["docs_in"] += 1
        if found(text) != script:
            counts["docs_other_script" if found(text) else "docs_unclear_script"] += 1
            continue
        text = cleaner.prose(text) if not is_blocked(url, config) else None
        if text is not None and not text.strip():  # only headings, lists or tables
            counts["docs_empty"] += 1
            continue
        key = skeleton(text, frequent) if text is not None else None
        copies[key] += key in templates
        if key is None or copies[key] > config["templates"]["keep"]:
            counts["docs_blocked_site" if key is None else "docs_template"] += 1
            continue
        text = cleaner.clean(text)  # prose() again inside drops and counts nothing more
        valid = in_validation(doc_id, config, fraction)
        split = "empty" if not text else "valid" if valid else "train"
        counts[f"docs_{split}"] += 1
        counts[f"chars_{split}"] += len(text)
        if text:
            writer.add(split, doc_id, text)
    writer.close()
    counts["templates"] = len(templates)
    return dict(counts)


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    index = int(next(arg for arg in sys.argv[2:] if arg.isdigit()))
    test = "--test" in sys.argv
    if test:
        config["output"] += "_test"
    corpus = config["corpora"][index]
    source, code, labels = corpus[:3]
    folder = Path(config["output"]) / source / "_".join([code] + corpus[3:])  # zh_hans, zh_hant
    path = folder / "manifest.json"
    for name in ("manifest.json", "train.parquet", "valid.parquet"):
        (folder / name).unlink(missing_ok=True)  # a task that dies leaves nothing stale behind
    counts = run(config, corpus, folder, TEST_DOCS if test else None)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    manifest = {"source": source, "code": code, "labels": labels, "script": corpus[3:],
                "counts": counts,
                "test": test, "git_sha": git_sha, "config": config}
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(source, code, json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
