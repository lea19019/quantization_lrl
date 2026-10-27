'''
Hold out dev and test sets, and keep every held-out sentence out of training.

hold_out(): church data has no published splits, so dev and test are drawn at random by
line (fixed seed). The metadata has no usable document unit, so the split cannot be by
document.

hold_out_shared() (configs with `split: shared`): one model trained on every language sees a
verse in all of them, so per-language random held-out sets would together hold out most
verses and decontamination would empty the train sets of small languages. Instead every
language holds out the same core verses, as FLORES does (the most widely translated, then a
seeded hash), plus a sample of its own text by the same hash; capped at a fraction of each
language's pairs so small languages keep data, and only pairs with a usable reference.

decontaminate(): a training pair is dropped if its English or its target side, normalized,
equals any held-out sentence of any language. The check spans all languages because the
same English sentence (a scripture verse, a manual line) appears in many of them: a
multilingual model trained on it in Zulu would have seen the reference for the Yoruba test.
Normalized means lowercased with only letters and single spaces left, so "29 And he went."
and "29. and he went" count as the same sentence. With `decontaminate_sentences` (pretrain.yaml)
a pair is also dropped if one sentence inside a side equals a held-out sentence.
'''
import hashlib
import random
import re

# Sentence ends; Chinese and Japanese put no space after a full stop.
SENTENCE_SPLIT = re.compile(r"(?<=[。！？])\s*|(?<=[.!?।።։۔؟။])\s+")


def normalize(text: str) -> str:
    """Return the text lowercased, with only letters and single spaces left."""
    kept = "".join(char for char in text.lower() if char.isalpha() or char.isspace())
    return " ".join(kept.split())


def hold_out(pairs: list, size: int, seed: int) -> tuple[list, list, list]:
    """Return (train, dev, test): `size` random pairs each for dev and test, the rest train."""
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled[2 * size :], shuffled[:size], shuffled[size : 2 * size]


def verse_rank(english: str, seed: int) -> int:
    """Return the seeded hash of a normalized English line: the same in every language."""
    return int(hashlib.sha256(f"{seed}:{english}".encode()).hexdigest(), 16)


def usable(pairs: list) -> set[int]:
    """Return the indices of pairs whose target can serve as a reference.

    Dropped: targets with no letters (a bare verse number, "40.") and pairs whose length ratio
    (target / English characters, normalized) is more than 2x off the language's median, which
    in the small church corpora are mostly misaligned verses. Relative to the median because
    the ratio depends on the script (Chinese is far shorter than English).
    """
    ratios = {}
    for index, (en, tgt) in enumerate(pairs):
        english, target = normalize(en), normalize(tgt)
        if english and target:
            ratios[index] = len(target) / len(english)
    if not ratios:
        return set()
    median = sorted(ratios.values())[len(ratios) // 2]
    return {index for index, ratio in ratios.items() if median / 2 <= ratio <= 2 * median}


def take(order: list, cap: int, splits: dict, held: set) -> None:
    """Add pairs from `order` ((rank, index), best first) to dev or test until both hold `cap`."""
    for rank, index in order:
        if len(splits[0]) >= cap and len(splits[1]) >= cap:
            return
        if index not in held and len(splits[rank & 1]) < cap:
            splits[rank & 1].append(index)
            held.add(index)


def hold_out_shared(pairs: list, size: int, fraction: float, seed: int, core: set[str],
                    protected: set[str]) -> tuple[list, list, list]:
    """Return (train, dev, test) with the held-out verses shared across languages.

    Of each language's usable pairs, at most `size` and `fraction` of its pairs go to dev and
    to test each: first the lines of the global `core` (core_lines) it has, then the rest by
    verse_rank, a sample of the language's own text, outside `protected` (the lines of the
    smallest languages): a line held out anywhere is removed from every train set by
    decontaminate(), and the smallest languages would lose 13% of their train to mid-size
    languages' held-out sets. So the smallest languages hold out only core lines. Not by
    coverage either, for the same reason. The hash's last bit picks dev or test,
    so a verse is in the same split in every language. Lines that do not fit, and other
    copies of a held-out line, stay in train; decontaminate() then drops them.
    """
    first = {}  # normalized English -> index of its first usable pair
    good = usable(pairs)
    for index, (en, _) in enumerate(pairs):
        if index in good:
            first.setdefault(normalize(en), index)
    lines = sorted((verse_rank(english, seed), index, english)
                   for english, index in first.items())
    limit = min(size, int(fraction * len(pairs)))
    splits, held = {0: [], 1: []}, set()
    take([(rank, index) for rank, index, english in lines if english in core], limit, splits, held)
    take([(rank, index) for rank, index, english in lines if english not in protected],
         limit, splits, held)
    train = [pair for index, pair in enumerate(pairs) if index not in held]
    random.Random(seed).shuffle(train)
    return train, [pairs[i] for i in splits[0]], [pairs[i] for i in splits[1]]


def core_lines(coverage: dict[str, int], size: int, seed: int) -> set[str]:
    """Return the shared core: the `size` dev and `size` test English lines (by verse_rank's
    last bit) that the most languages have, ties broken by verse_rank."""
    ranked = sorted((-count, verse_rank(english, seed), english)
                    for english, count in coverage.items() if english)
    core, per_split = set(), {0: 0, 1: 0}
    for _, rank, english in ranked:
        if per_split[rank & 1] < size:
            core.add(english)
            per_split[rank & 1] += 1
        if per_split[0] == per_split[1] == size:
            break
    return core


def coverage_of(languages, small: int) -> tuple[dict[str, int], set[str]]:
    """Return (coverage, protected): per normalized English line the number of languages whose
    pairs contain it, and the lines of the languages with fewer than `small` pairs."""
    counts, protected = {}, set()
    for pairs in languages:
        lines = {normalize(en) for en, _ in pairs}
        for english in lines:
            counts[english] = counts.get(english, 0) + 1
        if len(pairs) < small:
            protected |= lines
    return counts, protected


def held_out_sentences(rows: list[dict], extra: list[str]) -> set[str]:
    """Return the normalized form of both sides of every row, plus the `extra` sentences."""
    seen = {normalize(row["en"]) for row in rows} | {normalize(row["tgt"]) for row in rows}
    seen |= {normalize(sentence) for sentence in extra}
    seen.discard("")  # a line of only digits or punctuation must not match everything
    return seen


def pieces(text: str, sentences: bool) -> list[str]:
    """Return the text, and with `sentences` also each sentence in it (if there are several)."""
    parts = SENTENCE_SPLIT.split(text) if sentences else []
    return [text] + parts if len(parts) > 1 else [text]


def decontaminate(rows: list[dict], seen: set[str], sentences: bool = False) -> list[dict]:
    """Return the rows whose English and target sides both differ from every held-out one.

    With `sentences`, a side also counts as held out if one of its sentences is: a verse
    repeated inside a longer verse ("He said: 'Is it well?'") would otherwise leak.
    """
    return [
        row for row in rows
        if not any(normalize(piece) in seen
                   for side in (row["en"], row["tgt"]) for piece in pieces(side, sentences))
    ]
