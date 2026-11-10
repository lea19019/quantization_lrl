'''
Tables of the Front 1 results:
    python -m qlrl.experiments.analyze configs/experiments/analyze.yaml

Reads every finished .jsonl in <experiments>/{patching,lens,noise,change,swap}; no model is run.
Writes to <output>, with one row per model and language, plus site, layer or bin where it applies:
    damage.csv       damage in nats, per byte and per word (these compare across languages)
    patching.csv     share of the damage recovered per site
    lens.csv         logit lens: full model's log p per token; full - quant per byte, per layer
    change.csv       activation change per layer and part: ||full|| and ||quant - full||
    noise.csv        noise vs flatness, and the damage at the flatness mix of the reference language
    noise_bins.csv   the same per flatness bin
    swap.csv         share of the damage per hybrid, and the share only appearing when combined
    swap_change.csv  residual disturbance per hybrid and layer
Sizes are root mean squares per token: sqrt(sum ||.||^2 / tokens). Intervals are percentile
bootstraps of sentences, as in qlrl/analysis/stats.py.
'''
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from qlrl.analysis.stats import resample_counts

FLATNESS_BINS = [0, .001, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]
CONFIDENT = .1  # flatness under this is a token the full model was sure about


def read_rows(path: Path) -> list[dict]:
    """Return one dict per line of a JSON-lines file."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def find_files(folder: Path) -> dict[str, dict[str, Path]]:
    """Return { quantized model: { language: finished .jsonl } }."""
    found = {}
    for path in sorted(folder.glob("*/*.jsonl")):
        found.setdefault(path.parent.name, {})[path.name[:3]] = path
    return found


def read_run(path: Path) -> list[dict]:
    """Return the rows of one model and language with the four experiments of run.py joined."""
    parts = [read_rows(path.parents[2] / name / path.parent.name / path.name)
             for name in ("patching", "lens", "noise", "change")]
    return [a | b | c | d for a, b, c, d in zip(*parts)]


def interval(values: np.ndarray, level: float) -> tuple[float, float]:
    """Return the percentile interval along the first axis; row 0 is the sample itself."""
    tail = (1 - level) / 2 * 100
    return np.percentile(values[1:], tail, axis=0), np.percentile(values[1:], 100 - tail, axis=0)


def rms(rows: list[dict], part, name: str) -> float:
    """Return sqrt(sum ||.||^2 / tokens) of `name`, with `part` picking a row's sizes."""
    return (sum(part(row)[name] ** 2 for row in rows) / sum(r["tokens"] for r in rows)) ** .5


def damage_row(rows: list[dict], damage: np.ndarray, counts: np.ndarray, level: float) -> dict:
    """Return the totals of one model and language."""
    size = {"bytes": sum(len(row["ref"].encode()) for row in rows),
            "words": sum(len(row["ref"].split()) for row in rows)}
    low, high = interval(counts @ damage, level)
    return {"sentences": len(rows), "tokens": sum(row["tokens"] for row in rows), **size,
            "full NLL per byte": -sum(row["full"] for row in rows) / size["bytes"],
            "damage": damage.sum(), "low": low, "high": high,
            "per byte": damage.sum() / size["bytes"], "low per byte": low / size["bytes"],
            "high per byte": high / size["bytes"], "per word": damage.sum() / size["words"],
            "sentences with negative damage": int((damage < 0).sum())}


def patching_rows(rows, damage, counts, level) -> list[dict]:
    """Return per site the share of the damage recovered, 1 - damage left / damage."""
    sites = list(rows[0]["patched"])
    left = np.array([[row["full"] - row["patched"][site] for site in sites] for row in rows])
    shares = 1 - (counts @ left) / (counts @ damage)[:, None]
    low, high = interval(shares, level)
    point = 1 - left.sum(axis=0) / damage.sum()
    return [{"site": site, "recovered": point[i], "low": low[i], "high": high[i]}
            for i, site in enumerate(sites)]


def lens_rows(rows: list[dict]) -> list[dict]:
    """Return per layer the full model's log p per token and the full - quant gap per byte."""
    full = np.array([row["lens"]["full"] for row in rows]).sum(axis=0)
    quant = np.array([row["lens"]["quant"] for row in rows]).sum(axis=0)
    tokens = sum(row["tokens"] for row in rows)
    size = sum(len(row["ref"].encode()) for row in rows)
    return [{"layer": layer, "full log p per token": full[layer] / tokens,
             "gap per byte": (full[layer] - quant[layer]) / size} for layer in range(len(full))]


def change_rows(rows: list[dict], part=lambda row: row["change"]) -> list[dict]:
    """Return per layer and part the sizes of ||full|| and ||quant - full|| and their ratio."""
    table = []
    for name in part(rows[0])["norm"]:
        layer, kind = name.split(".")
        norm = rms(rows, lambda row: part(row)["norm"], name)
        distance = rms(rows, lambda row: part(row)["distance"], name)
        table.append({"layer": int(layer[5:]), "part": kind, "norm": norm, "distance": distance,
                      "ratio": distance / norm})
    return table


def token_columns(rows: list[dict]) -> dict[str, np.ndarray]:
    """Return per token its sentence, sigma2, flatness, damage and bin of flatness."""
    get = lambda key: np.concatenate([row["noise"][key] for row in rows])
    flatness = get("flatness")
    return {"sentence": np.repeat(np.arange(len(rows)), [row["tokens"] for row in rows]),
            "sigma2": get("sigma2"), "flatness": flatness,
            "damage": get("gold full") - get("gold quant"),
            "bin": np.digitize(flatness, FLATNESS_BINS) - 1}


def noise_rows(rows, reference_mix, counts, level) -> tuple[dict, list[dict]]:
    """Return the noise vs flatness totals and the table per flatness bin."""
    c = token_columns(rows)
    predicted = .5 * c["sigma2"] * c["flatness"]  # expected rise in NLL, second-order expansion
    bins, table = len(FLATNESS_BINS) - 1, []
    per_sentence = np.zeros((len(rows), bins))  # damage summed per sentence and bin
    per_count = np.zeros((len(rows), bins))
    np.add.at(per_sentence, (c["sentence"], c["bin"]), c["damage"])
    np.add.at(per_count, (c["sentence"], c["bin"]), 1)
    rates, sigma2 = np.zeros(bins), np.zeros(bins)
    for index in range(bins):
        mask = c["bin"] == index
        if not mask.any():
            continue
        rates[index], sigma2[index] = c["damage"][mask].mean(), c["sigma2"][mask].mean()
        table.append({"flatness from": FLATNESS_BINS[index], "tokens": int(mask.sum()),
                      "share of tokens": mask.mean(), "sigma2": sigma2[index],
                      "damage per token": rates[index], "predicted": predicted[mask].mean()})
    # Direct standardization: the language's damage per bin, weighted by the reference's mix.
    sums, n = counts @ per_sentence, counts @ per_count
    resampled = np.divide(sums, n, out=np.zeros_like(sums), where=n > 0) @ reference_mix
    low, high = interval(resampled, level)
    confident = c["flatness"] < CONFIDENT
    return ({"sigma2": c["sigma2"].mean(), "flatness": c["flatness"].mean(),
             "damage per token": c["damage"].mean(), "predicted per token": predicted.mean(),
             "share predicted": predicted.sum() / c["damage"].sum(),
             "correlation": np.corrcoef(predicted, c["damage"])[0, 1],
             "share on confident tokens": c["damage"][confident].sum() / c["damage"].sum(),
             "matched damage per token": rates @ reference_mix, "matched low": low,
             "matched high": high, "matched sigma2": sigma2 @ reference_mix}, table)


def swap_tables(rows, damage, counts, level) -> tuple[list[dict], list[dict]]:
    """Return the share of the damage per hybrid and the residual disturbance per layer."""
    shares, disturbance, combined = [], [], np.zeros(len(rows))
    for name in rows[0]["hybrids"]:
        alone = np.array([row["full"] - row["hybrids"][name]["logprob"] for row in rows])
        combined = combined + alone
        low, high = interval((counts @ alone) / (counts @ damage), level)
        shares.append({"hybrid": name, "damage": alone.sum(), "share": alone.sum() / damage.sum(),
                       "low": low, "high": high})
        part = lambda row, n=name: row["hybrids"][n]["change"]
        disturbance += [{"hybrid": name, **row} for row in change_rows(rows, part)
                        if row["part"] == "residual"]
    low, high = interval(1 - (counts @ combined) / (counts @ damage), level)
    shares.append({"hybrid": "only when combined", "damage": damage.sum() - combined.sum(),
                   "share": 1 - combined.sum() / damage.sum(), "low": low, "high": high})
    return shares, disturbance


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a table, if it has rows."""
    if rows:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    output, level = Path(config["output"]), config["level"]
    output.mkdir(parents=True, exist_ok=True)
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=False).stdout.strip()
    (output / "manifest.json").write_text(json.dumps({"config": config, "git_sha": git_sha},
                                                     indent=2))
    tables = {name: [] for name in ("damage", "patching", "lens", "change", "noise", "noise_bins",
                                    "swap", "swap_change")}
    experiments = Path(config["experiments"])
    for model, files in find_files(experiments / "patching").items():
        mix = np.bincount(token_columns(read_run(files[config["reference"]]))["bin"],
                          minlength=len(FLATNESS_BINS) - 1).astype(float)
        mix /= mix.sum()
        for lang, path in files.items():
            rows, about = read_run(path), {"model": model, "language": lang}
            damage = np.array([row["full"] - row["quant"] for row in rows])
            counts = resample_counts(len(rows), config["samples"], config["seed"])
            noise, bins = noise_rows(rows, mix, counts, level)
            tables["damage"].append(about | damage_row(rows, damage, counts, level))
            sites = patching_rows(rows, damage, counts, level)
            tables["patching"] += [about | row for row in sites]
            tables["lens"] += [about | row for row in lens_rows(rows)]
            tables["change"] += [about | row for row in change_rows(rows)]
            tables["noise"].append(about | noise)
            tables["noise_bins"] += [about | row for row in bins]
            print(model, lang, flush=True)
    for model, files in find_files(experiments / "swap").items():
        for lang, path in files.items():
            rows, about = read_rows(path), {"model": model, "language": lang}
            damage = np.array([row["full"] - row["quant"] for row in rows])
            counts = resample_counts(len(rows), config["samples"], config["seed"])
            shares, disturbance = swap_tables(rows, damage, counts, level)
            tables["swap"] += [about | row for row in shares]
            tables["swap_change"] += [about | row for row in disturbance]
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)


if __name__ == "__main__":
    main()
