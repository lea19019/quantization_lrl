'''
Entry point for the analysis: python -m qlrl.analysis configs/analysis/all.yaml

Compares every finished quantized evaluation run with its full-precision run on the same
sentences (the math is in qlrl/analysis/stats.py), in every evaluation folder of the config. No
model is run, except AfriCOMET once for its floor when the floor file does not exist yet
(needs a GPU). Writes <output>/summary.md, one table per model and configuration with the
test sets of every evaluation folder, and damage.csv and translation.csv; then every
figure and the report tables (qlrl.analysis.plots).
'''
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from qlrl.analysis.plots import draw_figures
from qlrl.analysis.stats import DIRECTIONS, compare, floors_of, resample_counts, shuffled
from qlrl.analysis.tables import write_tables
from qlrl.eval.scores import afriCOMET_eval


def read_jsonl(path: Path) -> list[dict]:
    """Return one dict per line of a JSON-lines file."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_run(folder: Path, lang: str, source: str) -> dict:
    """Return a run's per-sentence log-probs ("lang", "eng") and translations (DIRECTIONS)."""
    run = {}
    for side, text in (("lang", lang), ("eng", "eng")):
        rows = read_jsonl(folder / "perplexity" / f"{lang}-{source}-{text}.jsonl")
        run[side] = {key: np.array([row[key] for row in rows], dtype=float)
                     for key in ("logprob", "words", "bytes")}
    for direction, (src, tgt) in zip(DIRECTIONS, (("eng", lang), (lang, "eng"))):
        run[direction] = read_jsonl(folder / "translation" / f"{lang}-{source}-{src}2{tgt}.jsonl")
    return run


def configurations(results: Path, spec: dict) -> dict[str, list[Path]]:
    """Return { "<method>-<bits>bit": finished run folders, one per seed } for one model."""
    found = {}
    for bits in spec["bits"]:
        for method, seeds in spec["methods"].items():
            name = f"{spec['full']}-{method}-{bits}bit"
            runs = [name if seed is None else f"{name}-seed{seed}" for seed in seeds]
            folders = [results / run for run in runs if (results / run / "scores.json").exists()]
            print(f"{name}: {len(folders)} of {len(runs)} runs finished")
            if folders: found[f"{method}-{bits}bit"] = folders
    return found


def comet_floors(path: Path, folder: Path, seed: int) -> dict:
    """Return { language: { direction: AfriCOMET of shuffled references } }; compute once."""
    if path.exists(): return json.loads(path.read_text())
    manifest = json.loads((folder / "manifest.json").read_text())["config"]
    floors = {}
    for lang in manifest["africomet"]["languages"]:
        run = load_run(folder, lang, manifest["test_sets"][lang])
        floors[lang] = {}
        for direction in DIRECTIONS:
            references = [row["ref"] for row in run[direction]]
            sources = [row["src"] for row in run[direction]]
            hypotheses = shuffled(references, seed)
            score = afriCOMET_eval(sources, hypotheses, references, manifest["africomet"])
            floors[lang][direction] = score["score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(floors, indent=2))
    return floors


def analyze_model(model: str, spec: dict, evaluation: dict,
                  settings: dict) -> tuple[list[dict], list[dict]]:
    """Return (damage rows, translation rows) of every configuration and language of one
    evaluation folder ({ "results", "africomet_floor" })."""
    results = Path(evaluation["results"])
    folder = results / spec["full"]
    configs = configurations(results, spec)
    comet = comet_floors(Path(evaluation["africomet_floor"]), folder, settings["seed"])
    evaluated = json.loads((folder / "manifest.json").read_text())["config"]
    test_sets = evaluated["test_sets"]
    damages, drops = [], []
    for lang in evaluated["languages"]:
        full = load_run(folder, lang, test_sets[lang])
        counts = resample_counts(len(full["lang"]["logprob"]), settings["samples"],
                                 settings["seed"])
        floors = floors_of(full, lang, comet, settings["seed"])
        for name, folders in configs.items():
            about = {"model": model, "config": name, "seeds": len(folders), "language": lang,
                     "test set": test_sets[lang]}
            quantized = [load_run(run, lang, test_sets[lang]) for run in folders]
            found, rows = compare(full, quantized, counts, floors, settings)
            damages.append(about | found)
            drops += [about | row for row in rows]
        print(f"{model} {lang}: done", flush=True)
    return damages, drops


def cell(row: dict) -> str:
    """Return a drop as shown in the summary: the number, (number) if noise, or floor."""
    if row["reading"] == "at floor": return "floor"
    return f"({row['drop']:.1f})" if row["reading"] == "noise" else f"{row['drop']:.1f}"


def table(damages: list[dict], drops: list[dict]) -> list[str]:
    """Return the markdown table of one model and configuration, a row per language."""
    columns = [(m, d) for m in ("chrF++", "AfriCOMET") for d in DIRECTIONS]
    lines = ["| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | "
             + " | ".join(f"{m} {d}" for m, d in columns) + " |", "|---" * (4 + 4) + "|"]
    for row in damages:
        shown = {(r["metric"], r["direction"]): cell(r) for r in drops
                 if (r["language"], r["test set"]) == (row["language"], row["test set"])}
        gap = (f"{row['gap per byte']:.3f} [{row['gap low per byte']:.3f}, "
               f"{row['gap high per byte']:.3f}]")
        lines.append(f"| {row['language'].upper()} | {row['test set']} | "
                     f"{row['FP NLL per byte']:.2f} | {gap} | "
                     + " | ".join(shown.get(c, "–") for c in columns) + " |")
    return lines


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    settings = config["bootstrap"]
    damages, drops = [], []
    for evaluation in config["evaluations"]:
        for model, spec in config["models"].items():
            if not (Path(evaluation["results"]) / spec["full"] / "scores.json").exists():
                print(f"{evaluation['results']} {model}: full-precision run not finished, skipped")
                continue
            model_damages, model_drops = analyze_model(model, spec, evaluation, settings)
            damages, drops = damages + model_damages, drops + model_drops
    folders = ", ".join(evaluation["results"] for evaluation in config["evaluations"])
    summary = [f"# Analysis of {folders}", "", LEGEND.format(**settings)]
    for model, name in dict.fromkeys((row["model"], row["config"]) for row in damages):
        rows = [row for row in damages if (row["model"], row["config"]) == (model, name)]
        shown = [row for row in drops if (row["model"], row["config"]) == (model, name)]
        summary += ["", f"## {model}, {name} ({rows[0]['seeds']} seeds)", ""]
        summary += table(rows, shown)
    for name, rows in (("damage", damages), ("translation", drops)):
        with (output / f"{name}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "summary.md").write_text("\n".join(summary) + "\n")
    draw_figures(config["figures"], output)
    write_tables(config["figures"] | config["tables"], Path(config["tables"]["output"]))


LEGEND = """Intervals: {level:.0%} paired percentile bootstrap, {samples} resamples of sentences.
Gap: damage per byte (nats) of the language's text minus that of the English side. Drops:
full precision minus quantized (mean over seeds); (number) is inside the interval or the
seed spread; "floor": the full-precision score is not clearly above shuffled references.
Every number, with intervals, is in damage.csv and translation.csv."""


if __name__ == "__main__":
    main()
