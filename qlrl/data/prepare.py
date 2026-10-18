'''
Entry point for data preparation: python -m qlrl.data.prepare configs/data.yaml [--test]

1. Only if the config has `translate:`, translate the Spanish sides into English.
2. Every language's dev and test: church held-out pairs, and FLORES+, AmericasNLP and
   MayanV dev and test as published.
3. Every language's train: the rest of the church data, plus the external train sets
   cleaned by the lab tool.
4. Drop every train pair that matches a dev or test sentence of any language (split.py).
   That is why step 2 runs for all languages before step 3.
5. Write <output>/<lang>/{train,dev,test}.jsonl and <output>/manifest.json.

A pair with an empty side is dropped (blank lines; long lines left untranslated).
--test: first TEST_LINES lines of each corpus, TEST_HELD_OUT held out, into <output>_test.
'''
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

from qlrl.data.cleaning import clean
from qlrl.data.sources import americasnlp, church, flores, kichwa, mayanv, spanish_sides
from qlrl.data.split import decontaminate, held_out_sentences, hold_out

TEST_LINES = 2000
TEST_HELD_OUT = 20


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    test = "--test" in sys.argv
    if test:
        config["output"] += "_test"
        config["cleaning"] += "_test"
        config |= {"held_out": TEST_HELD_OUT, "limit": TEST_LINES}
    if config.get("translate"):
        from qlrl.data.translate import translate  # needs torch and a GPU; imported only here
        translate(config, test)

    languages = config["languages"]
    evaluation = {lang: evaluation_rows(config, lang, spec) for lang, spec in languages.items()}
    held_out = [row for splits in evaluation.values() for rows in splits.values() for row in rows]
    spanish = spanish_sides(config["external"], languages)
    spanish_held_out = [line for name, lines in spanish.items()
                        if name.endswith(("-dev", "-test")) for line in lines]
    seen = held_out_sentences(held_out, spanish_held_out)

    summary = {}
    for lang, spec in languages.items():
        train = training_rows(config, lang, spec)
        splits = {"train": decontaminate(train, seen)} | evaluation[lang]
        summary[lang] = {}
        for split, rows in splits.items():
            write_jsonl(f"{config['output']}/{lang}/{split}.jsonl", rows)
            summary[lang][split] = dict(Counter(row["source"] for row in rows))
        summary[lang]["removed_from_train"] = len(train) - len(splits["train"])
        print(lang, summary[lang], flush=True)
    write_manifest(config, summary)


def church_split(config: dict, spec: dict) -> tuple[list, list, list]:
    """Return the (train, dev, test) pairs of one church language."""
    pairs = church(config["church"], spec["church"], config.get("limit"))
    return hold_out(pairs, config["held_out"], config["seed"])


def evaluation_rows(config: dict, lang: str, spec: dict) -> dict[str, list[dict]]:
    """Return {"dev": rows, "test": rows} from every corpus the language has."""
    external, translated = config["external"], config["translated"]
    splits = {"dev": [], "test": []}
    if "church" in spec:
        _, dev, test = church_split(config, spec)
        splits["dev"] += rows(dev, lang, "church")
        splits["test"] += rows(test, lang, "church")
    for split in splits:
        if "flores" in spec:
            splits[split] += rows(flores(external, spec["flores"], split), lang, "flores")
        if "americasnlp" in spec:
            folder, extension = spec["americasnlp"]
            pairs = americasnlp(external, translated, folder, extension, split)
            splits[split] += rows(pairs, lang, "americasnlp")
        if "mayanv" in spec:
            pairs = mayanv(external, translated, spec["mayanv"], split)
            splits[split] += rows(pairs, lang, "mayanv")
    return splits


def training_rows(config: dict, lang: str, spec: dict) -> list[dict]:
    """Return the church train rows and the cleaned external train rows of one language."""
    external, translated = config["external"], config["translated"]
    train = []
    if "church" in spec:
        train += rows(church_split(config, spec)[0], lang, "church")
    corpora = {}
    if "americasnlp" in spec:
        folder, extension = spec["americasnlp"]
        corpora["americasnlp"] = americasnlp(external, translated, folder, extension, "train")
    if "mayanv" in spec:
        corpora["mayanv"] = mayanv(external, translated, spec["mayanv"], "train")
    if spec.get("kichwa"):
        corpora["kichwa"] = kichwa(external, translated)
    for source, pairs in corpora.items():
        pairs = [(en, tgt) for en, tgt in pairs if en and tgt][: config.get("limit")]
        folder = f"{config['cleaning']}/{lang}-{source}"
        cleaned = clean(pairs, folder, config["cleaner"], spec.get("cleaner", lang))
        train += rows(cleaned, lang, source)
    return train


def rows(pairs: list[tuple[str, str]], lang: str, source: str) -> list[dict]:
    """Return one row per pair with both sides non-empty."""
    return [
        {"en": en, "tgt": tgt, "lang": lang, "source": source}
        for en, tgt in pairs if en.strip() and tgt.strip()
    ]


def write_jsonl(path: str, rows: list[dict]) -> None:
    """Write one JSON object per line."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def write_manifest(config: dict, summary: dict) -> None:
    """Write the config, the git commit and the row counts per language, split and source."""
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    manifest = {"config": config, "git_sha": git_sha, "rows": summary}
    path = Path(config["output"]) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
