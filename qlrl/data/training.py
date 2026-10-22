'''
TrainingData: ParallelData's pairs turned into fine-tuning batches.

    pairs = ParallelData("data/processed").split("train", languages, caps=caps, seed=0)
    data = TrainingData(pairs, tokenizer, config)
    data.stats                                  # pairs, sequences, dropped rows per language
    for batch in data.loader(): ...             # input_ids, attention_mask, labels
    for batch in data.loader("yor"): ...        # one language only (the dev loss)

Every row is an evaluation prompt followed by its answer, and only the answer is graded.
'''
# Written by Claude Code at Adrian's request, 2026-10-20.
import random
from collections import Counter

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from qlrl.eval.translation import get_prompts


class TrainingData:
    def __init__(self, pairs: Dataset, tokenizer, config: dict):
        self.tokenizer = tokenizer
        self.config = config
        self.rng = random.Random(config["seed"])
        self.rows, self.stats = self.build(pairs)

    def loader(self, lang: str | None = None) -> DataLoader:
        """Return shuffled batches of every row, or of one language's rows, padded on the right
        to the longest row; padding gets label -100 and attention mask 0 (Hugging Face's
        DataCollatorForSeq2Seq does the padding)."""
        rows = self.rows
        if lang is not None:
            langs = rows["lang"][:]
            rows = rows.select([i for i, row_lang in enumerate(langs) if row_lang == lang])
        return DataLoader(
            rows.remove_columns("lang"), batch_size=self.config["train"]["batch_size"],
            shuffle=True, generator=torch.Generator().manual_seed(self.config["seed"]),
            collate_fn=DataCollatorForSeq2Seq(self.tokenizer, label_pad_token_id=-100),
        )

    def build(self, pairs: Dataset) -> tuple[Dataset, dict]:
        """Return the rows (input_ids, labels, lang) and counts per language: each pair used
        once, shuffled, packed, too-long rows dropped."""
        # pairs["en"] is a lazy view that reads the table on every access, and pack slices it
        # once per sequence; [:] reads each column into a plain list once.
        english, other, langs = pairs["en"][:], pairs["tgt"][:], pairs["lang"][:]
        rows_of: dict[str, list[int]] = {}
        for i, lang in enumerate(langs):
            rows_of.setdefault(lang, []).append(i)
        texts: dict[str, list[str]] = {"context": [], "target": [], "lang": []}
        for lang, rows in rows_of.items():
            self.rng.shuffle(rows)  # random pairs share a sequence, not neighbouring lines
            chosen_english, chosen_other = [english[i] for i in rows], [other[i] for i in rows]
            for context, target in self.pack(chosen_english, chosen_other, lang):
                texts["context"].append(context)
                texts["target"].append(target)
                texts["lang"].append(lang)
        sequences = Dataset.from_dict(texts).map(
            tokenize, batched=True, fn_kwargs={"tokenizer": self.tokenizer},
            remove_columns=["context", "target"],
        )
        kept = sequences.filter(lambda row: len(row["input_ids"]) <= self.config["max_length"])
        built, left = Counter(sequences["lang"][:]), Counter(kept["lang"][:])
        stats = {lang: {"pairs": len(rows_of[lang]), "sequences": left[lang],
                        "dropped_too_long": built[lang] - left[lang]} for lang in rows_of}
        return kept, stats

    def pack(self, english: list[str], other: list[str], lang: str) -> list[tuple[str, str]]:
        """Return (context, target) texts, one per group of pairs_per_sequence pairs in one
        random direction.

        The first pairs are the examples and the last one is the sentence, written by the
        evaluation's own get_prompts, so a training row is exactly an evaluation prompt followed
        by its answer. A leftover group smaller than pairs_per_sequence is dropped.
        """
        size = self.config["pairs_per_sequence"]
        texts = []
        for start in range(0, len(english) - size + 1, size):
            group = list(zip(english[start:start + size], other[start:start + size]))
            src, tgt = self.rng.choice([("eng", lang), (lang, "eng")])
            if src != "eng":
                group = [(o, e) for e, o in group]
            *examples, (source, target) = group
            context = get_prompts(examples, [source], src, tgt)[0]
            texts.append((context, f"{target}</{tgt}>"))
        return texts


def tokenize(batch: dict, tokenizer) -> dict:
    """Return input_ids and labels of each row; only the target tokens are graded.

    The context is tokenized as the evaluation tokenizes its prompt (BOS first), the target on
    its own, as the model generates it after the open tag. Label -100 is ignored by the loss;
    the model shifts labels by one itself, so labels line up with input_ids.
    """
    contexts = tokenizer(batch["context"]).input_ids
    targets = tokenizer(batch["target"], add_special_tokens=False).input_ids
    return {
        "input_ids": [c + t for c, t in zip(contexts, targets)],
        "labels": [[-100] * len(c) + t for c, t in zip(contexts, targets)],
    }
