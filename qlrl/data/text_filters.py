'''
The line- and document-level checks of free-text cleaning (used by freetext.py).

- blocked(): the url is on a blocked site (religious sites, whose verses could be church test
  sentences; machine-translated sites) or contains a blocked word.
- is_prose(): Wikipedia lines that are not prose (section headings, list items, link lists)
  are dropped. A line is prose if it ends a sentence or is at least `min_line_chars` long;
  the length rule keeps running text in scripts without sentence punctuation (Thai, Lao).
- Language ID: GlotLID (a fastText classifier over ~2,000 languages) labels each line of at
  least `min_chars`, reading its first `max_chars`. A line is dropped only if GlotLID is
  confident (>= threshold) it is a language outside the corpus's own labels. Unsure and
  shorter lines stay: GlotLID is weak on small languages, close varieties and a few words
  ("!" comes out French).
- contaminated(): the line, or a sentence in it, equals a dev or test sentence of any
  language after normalize(): split.normalize() (lowercase, letters and single spaces) done
  with one regex, ~10x faster; both sides of the comparison use it.
- script_of(): the Chinese Wikipedia mixes Simplified and Traditional articles, which are
  separate languages here; each article goes to the script whose characters it uses.
- skeleton(): bot-written articles share one text with names, numbers and rare words
  swapped; replacing every word outside the corpus's most common words with "_" maps all
  copies of a template to one skeleton.
'''
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import fasttext
from huggingface_hub import hf_hub_download

SENTENCE_END = tuple(".!?:;。！？।॥።։۔؟။")
CLOSING = "\"')]»”’"  # a sentence can end inside quotes or brackets: 'he said "yes."'

# Chinese and Japanese put no space after a full stop, so their stops split without one.
SENTENCE_SPLIT = re.compile(r"(?<=[。！？])\s*|(?<=[.!?।።։۔؟။])\s+")
# Scripts written without spaces between words (Thai, Lao, Burmese, Khmer, Japanese kana,
# Chinese characters): there each character counts as a word, or a whole sentence is one word.
UNSPACED = "\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
WORD = re.compile(rf"[{UNSPACED}]|[^\W{UNSPACED}]+")
NOT_LETTER = re.compile(r"[^\w\s]|[\d_]")
# \w minus decimal digits and "_" is str.isalpha() except for other numerals (², ½, Roman ⅷ,
# Bengali and Tamil number signs), which isalpha() rejects: dropped with this table.
NUMERALS = {code: None for code in range(sys.maxunicode + 1)
            if chr(code).isalnum() and not chr(code).isalpha() and not chr(code).isdecimal()}


def blocked(url: str, sites: list[str], words: list[str]) -> bool:
    """Return True if the url is on one of `sites` (or a subdomain) or contains one of `words`."""
    host = re.sub(r"^www\.", "", url.split("/")[2].lower()) if "//" in url else ""
    if any(host == site or host.endswith("." + site) for site in sites):
        return True
    return any(word in url.lower() for word in words)


def is_prose(line: str, min_chars: int) -> bool:
    """Return True if the line ends a sentence or is long enough to be running text."""
    line = line.strip()
    return line.rstrip(CLOSING).endswith(SENTENCE_END) or len(line) >= min_chars


def normalize(text: str) -> str:
    """Return the text lowercased, with only letters and single spaces left."""
    return " ".join(NOT_LETTER.sub("", text.lower()).translate(NUMERALS).split())


def contaminated(line: str, held_out: set[str]) -> bool:
    """Return True if the line, or any sentence in it, is a held-out sentence."""
    sentences = [line] + SENTENCE_SPLIT.split(line)
    return any(normalize(sentence) in held_out for sentence in sentences)


def skeleton(text: str, frequent: set[str]) -> str:
    """Return a hash of the text with every run of words outside `frequent` replaced by "_"."""
    words = [word if word in frequent else "_" for word in WORD.findall(text.lower())]
    words = [word for i, word in enumerate(words) if word != "_" or i == 0 or words[i - 1] != "_"]
    return hashlib.sha256(" ".join(words).encode()).hexdigest()


def script_chars(settings: dict) -> dict[str, set[str]]:
    """Return, per script (hans, hant), the Chinese characters found only in that script.

    Learned from the church Chinese S and Chinese T targets: a character seen at least
    `min_count` times in one and at most 1/100 as often in the other. Simplified and
    Traditional differ in many common characters (们/們, 这/這, 会/會).
    """
    counts = {}
    for script, folder in settings["folders"].items():
        with open(f"{settings['church']}/{folder}/filtered/tgt.txt", encoding="utf-8") as handle:
            counts[script] = Counter(char for line in handle for char in line
                                     if "\u4e00" <= char <= "\u9fff")
    chars = {}
    for script, mine in counts.items():
        other = sum((counts[name] for name in counts if name != script), Counter())
        chars[script] = {char for char, n in mine.items()
                         if n >= settings["min_count"] and other[char] * 100 <= n}
    return chars


def script_of(text: str, chars: dict[str, set[str]], ratio: float) -> str | None:
    """Return the script whose exclusive characters outnumber every other's `ratio` to 1."""
    found = {script: sum(char in own for char in text) for script, own in chars.items()}
    best = max(found, key=found.get)
    others = max(n for script, n in found.items() if script != best)
    return best if found[best] > 0 and found[best] >= ratio * others else None


def held_out_sentences(patterns: list[str]) -> set[str]:
    """Return the normalized sentences of every dev and test file (en, tgt and FLORES text)."""
    seen = set()
    for pattern in patterns:
        for path in Path().glob(pattern):
            with open(path, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            seen |= {normalize(row[key]) for row in rows for key in ("en", "tgt", "text")
                     if key in row}
    seen.discard("")  # a line of only digits or punctuation must not match everything
    return seen


class Cleaner:
    """Holds what the line-level steps need: GlotLID, the held-out set and the counts."""

    def __init__(self, config: dict, prose_only: bool, labels: list[str]):
        glotlid = config["glotlid"]
        self.model = fasttext.load_model(
            hf_hub_download(glotlid["repo"], "model.bin", revision=glotlid["revision"]))
        self.threshold, self.labels = glotlid["threshold"], set(labels)
        self.judged_chars, self.read_chars = glotlid["min_chars"], glotlid["max_chars"]
        self.held_out = held_out_sentences(config["held_out"])
        self.prose_only, self.min_chars = prose_only, config["min_line_chars"]
        self.counts = Counter()

    def prose(self, text: str) -> str:
        """Return the text without its non-prose lines (only when `prose_only`)."""
        if not self.prose_only:
            return text
        lines = text.split("\n")
        kept = [line for line in lines if not line.strip() or is_prose(line, self.min_chars)]
        self.counts["lines_not_prose"] += len(lines) - len(kept)
        return "\n".join(kept)

    def other_language(self, lines: list[str]) -> list[bool]:
        """Return, per line, whether GlotLID is confident it is not the corpus's language."""
        if not lines:
            return []
        labels, probabilities = self.model.predict([line[: self.read_chars] for line in lines])
        return [label[0].removeprefix("__label__") not in self.labels
                and probability[0] >= self.threshold
                for label, probability in zip(labels, probabilities, strict=True)]

    def clean(self, text: str) -> str:
        """Return the text without non-prose, other-language and contaminated lines."""
        lines = self.prose(text).split("\n")
        content = [line for line in lines if len(line.strip()) >= self.judged_chars]
        other = dict(zip(content, self.other_language(content), strict=True))
        kept = []
        for line in lines:
            if other.get(line):
                self.counts["lines_other_language"] += 1
            elif line.strip() and contaminated(line, self.held_out):
                self.counts["lines_contaminated"] += 1
            else:
                kept.append(line)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()  # collapse emptied paragraphs
