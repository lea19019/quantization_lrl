'''
One loader per raw corpus. Each returns (english, target) pairs, line i of one side
aligned with line i of the other.

The external corpora pair their language with Spanish, not English. Their English side is
a machine translation of the Spanish side (translate.py), stored in <translated>/<name>.en;
the target side stays human-written. spanish_sides() lists those Spanish files and the
names their translations are stored under.
'''
import json
from itertools import islice
from pathlib import Path

import pyarrow.parquet as pq

SPLITS = ("train", "dev", "test")
FLORES_SPLITS = {"dev": "dev", "test": "devtest"}  # FLORES+ has no train split
KICHWA = "kichwa-spanish/lexicon/train-00000-of-00001.parquet"


def read_lines(path: str | Path, limit: int | None = None) -> list[str]:
    """Return the lines of a text file, without the newline; at most `limit` lines."""
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in islice(handle, limit)]


def church(root: str, folder: str, limit: int | None = None) -> list[tuple[str, str]]:
    """Return a church language's pairs. filtered/ is the lab's cleaned version."""
    english = read_lines(f"{root}/{folder}/filtered/src.txt", limit)
    target = read_lines(f"{root}/{folder}/filtered/tgt.txt", limit)
    return pair(english, target, f"church {folder}")


def flores(external: str, code: str, split: str) -> list[tuple[str, str]]:
    """Return FLORES+ pairs: English and `code` are human translations of the same text."""
    folder = f"{external}/flores_plus/{FLORES_SPLITS[split]}"
    sides = []
    for language in ("eng_Latn", code):
        lines = read_lines(f"{folder}/{language}.jsonl")
        sides.append([json.loads(line)["text"] for line in lines])
    return pair(sides[0], sides[1], f"flores {code} {split}")


def americasnlp(
        external: str, translated: str, folder: str, extension: str, split: str
    ) -> list[tuple[str, str]]:
    """Return AmericasNLP 2021 pairs. The test targets were released in test_data/."""
    data = f"{external}/americasnlp2021"
    target_path = f"{data}/data/{folder}-spanish/{split}.{extension}"
    if split == "test":
        target_path = f"{data}/test_data/test.{extension}"
    english = read_lines(f"{translated}/{folder}-{split}.en")
    return pair(english, read_lines(target_path), f"americasnlp {folder} {split}")


def mayanv(external: str, translated: str, language: str, split: str) -> list[tuple[str, str]]:
    """Return MayanV pairs; its own train, dev and test splits."""
    target = read_lines(f"{external}/mayanv/MayanV/{language}/{split}/data.{language}")
    english = read_lines(f"{translated}/mayanv-{language}-{split}.en")
    return pair(english, target, f"mayanv {language} {split}")


def kichwa(external: str, translated: str) -> list[tuple[str, str]]:
    """Return the Napo Kichwa lexicon: headwords and short example sentences, train only."""
    target = kichwa_column(external, "kichwa")
    return pair(read_lines(f"{translated}/kichwa-lexicon.en"), target, "kichwa lexicon")


def kichwa_column(external: str, column: str) -> list[str]:
    """Return one column ("kichwa" or "spanish") of the lexicon, whitespace collapsed."""
    rows = pq.read_table(f"{external}/{KICHWA}", columns=[column]).column(column).to_pylist()
    return [" ".join(row.split()) for row in rows]


def spanish_sides(external: str, languages: dict) -> dict[str, list[str]]:
    """Return {name of the English file: Spanish lines} for every Spanish-X corpus."""
    sides = {}
    for spec in languages.values():
        if "americasnlp" in spec:
            folder = spec["americasnlp"][0]
            for split in SPLITS:
                path = f"{external}/americasnlp2021/data/{folder}-spanish/{split}.es"
                sides[f"{folder}-{split}"] = read_lines(path)
        if "mayanv" in spec:
            language = spec["mayanv"]
            for split in SPLITS:
                path = f"{external}/mayanv/MayanV/{language}/{split}/data.es"
                sides[f"mayanv-{language}-{split}"] = read_lines(path)
        if spec.get("kichwa"):
            sides["kichwa-lexicon"] = kichwa_column(external, "spanish")
    return sides


def pair(english: list[str], target: list[str], name: str) -> list[tuple[str, str]]:
    """Zip the two sides; they must have the same number of lines."""
    assert len(english) == len(target), f"{name}: {len(english)} vs {len(target)} lines"
    return list(zip(english, target))
