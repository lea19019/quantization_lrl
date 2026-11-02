'''
Scores of every run for the report tables (qlrl.analysis.tables), read from scores.json
(qlrl.evaluation): mean over seeds, and the loss from full precision.
'''
import json
from pathlib import Path


def run_folders(model: str, method: str, bits: int, seeds: list[int]) -> list[str]:
    """Return the run folders of one model version: one, or one per seed."""
    if method == "fp":
        return [model]
    if method == "rtn":
        return [f"{model}-rtn-{bits}bit"]
    return [f"{model}-{method}-{bits}bit-seed{seed}" for seed in seeds]


def flatten(entry: dict, lang: str) -> dict:
    """Return {(where, metric): number} of one scores.json entry. where: "eng→X" or
    "X→eng" (translation), "eng" or "X" (each test sentence on its own)."""
    flat = {}
    for where, pair in (("eng→X", f"eng→{lang}"), ("X→eng", f"{lang}→eng")):
        for metric, found in entry.get(pair, {}).items():
            flat[(where, metric)] = found
    for where, side in (("eng", "eng"), ("X", lang)):
        for metric, found in entry["perplexity"][side].items():
            flat[(where, metric)] = found
    return flat


def mean_scores(config: dict, folders: list[str]) -> dict:
    """Return {(language, test set): {(where, metric): mean over the runs}}."""
    found = {}
    for folder in folders:
        for evaluation in config["evaluations"]:
            path = Path(evaluation) / folder / "scores.json"
            if not path.exists():
                continue
            for code, entry in json.loads(path.read_text()).items():
                cell = found.setdefault((code.lower(), entry["test set"]), {})
                for field, value in flatten(entry, code.lower()).items():
                    if value is not None:
                        cell.setdefault(field, []).append(value)
    return {key: {field: sum(values) / len(values) for field, values in cell.items()}
            for key, cell in found.items()}


def loss(config: dict, field: tuple, fp: float | None, quantized: float | None):
    """Return quantized minus FP in the "worse" direction, or None if either is missing."""
    if fp is None or quantized is None:
        return None
    return fp - quantized if field[1] in config["higher_is_better"] else quantized - fp


def largest(config: dict, scores: dict, keys: list, field: tuple, models: list) -> float:
    """Return the largest loss of one field over the languages, models and methods."""
    losses = [loss(config, field, scores[(model, "fp")].get(key, {}).get(field),
                   scores[(model, method)].get(key, {}).get(field))
              for key in keys for model in models for method in config["methods"]]
    return max([found for found in losses if found is not None], default=0.0)


def languages(config: dict, scores: dict, codes: list[str]) -> list[tuple[str, str]]:
    """Return the (language, test set) pairs of these languages evaluated at full precision."""
    fp = scores[(next(iter(config["models"])), "fp")]
    return [(lang, test) for lang in codes for test in config["test_sets"] if (lang, test) in fp]


def label(config: dict, key: tuple[str, str]) -> str:
    """Return "Language (test set)"."""
    return f"{config['languages'][key[0]]} ({config['test_sets'][key[1]]})"
