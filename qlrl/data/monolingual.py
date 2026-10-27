'''
Free text from the parallel train splits, for pretraining:
    python -m qlrl.data.monolingual configs/data/freetext.yaml

Reads <splits>/<lang>/train.jsonl (written by prepare.py, so already decontaminated against
every dev and test set) and writes, next to the cleaned Wikipedia and FineWeb-2 text:
- <output>/parallel/<lang>/train.parquet: the target side of every train pair, one row each.
- <output>/parallel/eng/train.parquet: the English side of the church pairs, each line once.
  The same verse is the English side of up to 123 languages (58.8M lines, 3.24M distinct).
  The English sides of the Spanish-X corpora are machine translations and are left out.
Only train splits are read: dev and test stay out of pretraining. Rows keep the split's
order (shuffled by prepare.py), so there is no document context, one sentence per row.
'''
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string())])


def write(path: Path, texts: list[str], prefix: str) -> None:
    """Write one row per text, with ids <prefix>-<index>."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {"id": [f"{prefix}-{i}" for i in range(len(texts))], "text": texts}
    pq.write_table(pa.Table.from_pydict(rows, SCHEMA), path)


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    splits, output = Path(config["parallel_splits"]), Path(config["output"]) / "parallel"
    english, counts = {}, {}  # dict keeps the first-seen order of the distinct English lines
    for folder in sorted(path for path in splits.iterdir() if path.is_dir()):
        with open(folder / "train.jsonl", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        write(output / folder.name / "train.parquet", [row["tgt"] for row in rows], folder.name)
        english.update((row["en"], None) for row in rows if row["source"] == "church")
        counts[folder.name] = len(rows)
        print(folder.name, len(rows), flush=True)
    write(output / "eng" / "train.parquet", list(english), "eng")
    counts["eng"] = len(english)
    manifest = {"rows": counts, "parallel_splits": str(splits)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
