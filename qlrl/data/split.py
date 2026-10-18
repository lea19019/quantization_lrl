'''
Hold out dev and test sets, and keep every held-out sentence out of training.

hold_out(): church data has no published splits, so dev and test are drawn at random by
line (fixed seed). The metadata has no usable document unit, so the split cannot be by
document.

decontaminate(): a training pair is dropped if its English or its target side, normalized,
equals any held-out sentence of any language. The check spans all languages because the
same English sentence (a scripture verse, a manual line) appears in many of them: a
multilingual model trained on it in Zulu would have seen the reference for the Yoruba test.
Normalized means lowercased with only letters and single spaces left, so "29 And he went."
and "29. and he went" count as the same sentence.
'''
import random


def normalize(text: str) -> str:
    """Return the text lowercased, with only letters and single spaces left."""
    kept = "".join(char for char in text.lower() if char.isalpha() or char.isspace())
    return " ".join(kept.split())


def hold_out(pairs: list, size: int, seed: int) -> tuple[list, list, list]:
    """Return (train, dev, test): `size` random pairs each for dev and test, the rest train."""
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled[2 * size :], shuffled[:size], shuffled[size : 2 * size]


def held_out_sentences(rows: list[dict], extra: list[str]) -> set[str]:
    """Return the normalized form of both sides of every row, plus the `extra` sentences."""
    seen = {normalize(row["en"]) for row in rows} | {normalize(row["tgt"]) for row in rows}
    seen |= {normalize(sentence) for sentence in extra}
    seen.discard("")  # a line of only digits or punctuation must not match everything
    return seen


def decontaminate(rows: list[dict], seen: set[str]) -> list[dict]:
    """Return the rows whose English and target sides both differ from every held-out one."""
    return [
        row for row in rows
        if normalize(row["en"]) not in seen and normalize(row["tgt"]) not in seen
    ]
