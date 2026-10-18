'''
ParallelData: what fine-tuning and evaluation use to read the prepared data.

    data = ParallelData("data/processed")
    train = data.split("train", languages=["yor", "zul"])           # a datasets.Dataset
    test = data.split("test", languages=["grn"], sources=["flores"])

Every row has four columns: en, tgt (the other language), lang and source (church, flores,
americasnlp, mayanv, kichwa). Turning a row into a prompt, tokenizing and mixing languages
are left to the caller, so the same data serves every model.

The files are written by `python -m qlrl.data.prepare configs/data.yaml`.
'''
from pathlib import Path

from datasets import Dataset, load_dataset


class ParallelData:
    def __init__(self, root: str = "data/processed"):
        self.root = Path(root)
        assert self.root.is_dir(), f"{root} not found: run python -m qlrl.data.prepare first"

    def languages(self) -> list[str]:
        """Return the codes of every prepared language."""
        return sorted(folder.name for folder in self.root.iterdir() if folder.is_dir())

    def split(
            self, name: str, languages: list[str] | None = None,
            sources: list[str] | None = None,
        ) -> Dataset:
        """Return one split ("train", "dev" or "test") of the given languages, all by default."""
        languages = languages or self.languages()
        files = [str(self.root / lang / f"{name}.jsonl") for lang in languages]
        # Hugging Face names a split built from plain files "train", whatever it holds.
        dataset = load_dataset("json", data_files=files, split="train")
        if sources:
            dataset = dataset.filter(lambda row: row["source"] in sources)
        return dataset
