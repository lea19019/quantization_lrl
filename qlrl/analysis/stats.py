'''
Statistics of the analysis (docs/project/problem.md §3).

• Bootstrap over sentences (Efron & Tibshirani 1993). A test set is one draw of ~1,000
  sentences; drawing n of them with replacement, many times, shows how much a score would
  move under another draw. A resample is stored as counts[b, i], how many times sentence i
  was drawn, so any sum over sentences is counts @ values. Row 0 is all ones: the point
  estimate. Every model and both sides of a parallel test set use the same counts (paired),
  so "sentence i is hard" cancels in a difference.
• Damage of sentence i: d(i) = log p_full(i) - log p_quant(i), in nats, summed over its
  tokens (> 0: quantization hurt), averaged over seeds. Per byte, D = sum d / sum bytes,
  which does not depend on how the tokenizer cut the text. Gap: D(L) - D(English side).
• chrF++ is a corpus score of summed per-sentence n-gram statistics; sacrebleu keeps them
  per sentence and scores any sum (as its own paired bootstrap does). AfriCOMET is a mean.
• Floor: the score of shuffled references (right language and domain, wrong meaning). A
  drop is not read if the full-precision score is not k standard errors above the floor.
'''
import numpy as np
import sacrebleu

DIRECTIONS = ("eng→X", "X→eng")

CHRF = sacrebleu.CHRF(word_order=2)  # the settings of chrfPP_eval in qlrl/eval/scores.py


def resample_counts(n: int, samples: int, seed: int) -> np.ndarray:
    """Return counts [1 + samples, n]: row 0 all ones, then multinomial resamples of n."""
    rng = np.random.default_rng(seed)
    resamples = rng.multinomial(n, np.full(n, 1 / n), size=samples)
    return np.vstack([np.ones((1, n), dtype=resamples.dtype), resamples])


def interval(values: np.ndarray, level: float) -> tuple[float, float]:
    """Return the percentile interval at `level` (e.g. 0.95) of bootstrap values."""
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def spread(values: list[float]) -> float:
    """Return the standard deviation over seeds; NaN for a single run."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def shuffled(references: list[str], seed: int) -> list[str]:
    """Return the references in a seeded random order: the floor's meaningless outputs."""
    order = np.random.default_rng(seed).permutation(len(references))
    return [references[i] for i in order]


def chrf(counts: np.ndarray, hypotheses: list[str], references: list[str]) -> np.ndarray:
    """Return corpus chrF++ for each row of counts [1 + B, n]: [1 + B]."""
    statistics = np.array(CHRF._extract_corpus_statistics(hypotheses, [references]))
    return np.array([CHRF._compute_score_from_stats(list(row)).score
                     for row in counts @ statistics])


def damage(full: dict, quantized: list[dict], counts: np.ndarray, unit: str,
           level: float) -> dict:
    """Return D(L), D(eng) and the gap per `unit` ("bytes" or "words"), with intervals.

    full and each quantized run: { "lang" or "eng": { "logprob", "words", "bytes": [n] } }.
    """
    per_unit = {}
    for side in ("lang", "eng"):
        d = np.mean([full[side]["logprob"] - run[side]["logprob"] for run in quantized], axis=0)
        per_unit[side] = (counts @ d) / (counts @ full[side][unit])  # [1 + B]
    gap = per_unit["lang"] - per_unit["eng"]
    lang = full["lang"]
    seeds = [(lang["logprob"] - run["lang"]["logprob"]).sum() / lang[unit].sum()
             for run in quantized]
    row = {"FP NLL": -lang["logprob"].sum() / lang[unit].sum(),
           "damage": per_unit["lang"][0], "English damage": per_unit["eng"][0], "gap": gap[0]}
    row["damage low"], row["damage high"] = interval(per_unit["lang"][1:], level)
    row["gap low"], row["gap high"] = interval(gap[1:], level)
    row["seed sd"] = spread(seeds)
    return {key: float(value) for key, value in row.items()}


def drop(full: np.ndarray, quantized: list[np.ndarray], floor: float | None,
         settings: dict) -> dict:
    """Return full - quantized with its interval, seed spread, normalised drops and reading.

    full and each quantized run: scores [1 + B]. settings: level, floor_standard_errors.
    """
    runs = np.array(quantized)  # [seeds, 1 + B]
    loss = full - runs.mean(axis=0)
    row = {"full": full[0], "quantized": runs[:, 0].mean(), "drop": loss[0]}
    row["drop low"], row["drop high"] = interval(loss[1:], settings["level"])
    row["seed sd"] = spread(list(runs[:, 0]))
    margin = settings["floor_standard_errors"] * np.std(full[1:], ddof=1)
    above = floor is None or full[0] - floor > margin
    row["floor"] = float("nan") if floor is None else floor
    # The four normalisations of problem.md §3; skill only where the floor is known and cleared.
    row["relative %"] = 100 * loss[0] / full[0]
    known = floor is not None and above
    row["skill %"] = 100 * loss[0] / (full[0] - floor) if known else float("nan")
    row["headroom %"] = 100 * loss[0] / (100 - full[0])
    if not above: row["reading"] = "at floor"
    elif row["drop low"] <= 0 <= row["drop high"] or abs(loss[0]) <= row["seed sd"]:
        row["reading"] = "noise"
    else: row["reading"] = "loss" if loss[0] > 0 else "gain"
    return {key: value if isinstance(value, str) else float(value) for key, value in row.items()}


def scores(run: dict, direction: str, counts: np.ndarray) -> dict[str, np.ndarray]:
    """Return { metric: scores [1 + B] } of one direction: chrF++, and AfriCOMET if scored."""
    rows = run[direction]
    found = {"chrF++": chrf(counts, [row["hyp"] for row in rows], [row["ref"] for row in rows])}
    if "africomet" in rows[0]:
        found["AfriCOMET"] = 100 * (counts @ np.array([row["africomet"] for row in rows]))
        found["AfriCOMET"] /= len(rows)
    return found


def compare(full: dict, quantized: list[dict], counts: np.ndarray, floors: dict,
            settings: dict) -> tuple[dict, list[dict]]:
    """Return (damage per byte and word, translation drops) of one configuration."""
    found = {}
    for unit in ("bytes", "words"):
        row = damage(full, quantized, counts, unit, settings["level"])
        found |= {f"{key} per {unit[:-1]}": value for key, value in row.items()}
    drops = []
    for direction in DIRECTIONS:
        full_scores = scores(full, direction, counts)
        runs = [scores(run, direction, counts) for run in quantized]
        for metric, values in full_scores.items():
            row = drop(values, [run[metric] for run in runs], floors[direction][metric], settings)
            drops.append({"direction": direction, "metric": metric} | row)
    return found, drops


def floors_of(full: dict, lang: str, comet: dict, seed: int) -> dict:
    """Return { direction: { metric: floor } } of one language."""
    floors = {}
    for direction in DIRECTIONS:
        references = [row["ref"] for row in full[direction]]
        chrf_floor = chrf(np.ones((1, len(references))), shuffled(references, seed), references)
        floors[direction] = {"chrF++": float(chrf_floor[0]),
                             "AfriCOMET": comet.get(lang, {}).get(direction)}
    return floors
