'''
Translation figures, drawn by qlrl.analysis.plots.figures from translation.csv (qlrl.analysis).

A drop is full precision minus quantized (mean over seeds), with a 95% paired bootstrap
interval. "At floor": the full-precision model does not beat shuffled references, so the drop
is not read. Every function takes drops = {(model, config, metric, direction):
{(language, test set): row}} and the language rows to show.
'''
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qlrl.analysis.plots.gap import OFFSETS
from qlrl.analysis.plots.style import (
    COLOURS,
    GRID,
    INK,
    MUTED,
    dots,
    grid,
    heatmap,
    language_axis,
    legend_above,
    style,
)

DROP = ("drop", "drop low", "drop high")
DIRECTIONS = ("eng→X", "X→eng")


def plot_drop(drops: dict, keys: list, metric: str, direction: str, config: dict):
    """Drop per language, a dot and interval per method, hollow at the floor: rows bit
    widths, columns models."""
    bits, models = config["bits"], config["models"]
    figure, axes = plt.subplots(len(bits), len(models), figsize=(11, 3 + 0.33 * len(keys)),
                                sharex="row", sharey=True)
    for row, width in enumerate(bits):
        for column, (model, model_name) in enumerate(models.items()):
            ax = axes[row][column]
            for colour, offset, (method, method_name) in zip(COLOURS, OFFSETS,
                                                             config["methods"].items()):
                found = drops.get((model, f"{method}-{width}bit", metric, direction), {})
                read = {key: r for key, r in found.items() if r["reading"] != "at floor"}
                floor = {key: r for key, r in found.items() if r["reading"] == "at floor"}
                dots(ax, keys, read, DROP, offset, {"color": colour, "label": method_name})
                dots(ax, keys, floor, DROP, offset, {"color": colour, "markerfacecolor": "none"})
            ax.axvline(0, color=MUTED, linewidth=0.8)
            language_axis(ax, keys, config)
            style(ax, f"{model_name}, {width}-bit", f"{metric} drop (points)")
    handles, names = axes[0][0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, marker="o", markerfacecolor="none",
                              linestyle=""))
    figure.suptitle(f"{metric} drop, {direction}", x=0.01, ha="left", fontsize=10, color=INK)
    legend_above(figure, handles, names + ["at floor (not read)"])
    return figure


def plot_scores(drops: dict, keys: list, metric: str, direction: str, config: dict):
    """Full-precision score joined to the quantized one (method "slope_method"), floor as a
    gray tick: rows bit widths, columns models."""
    bits, models = config["bits"], config["models"]
    figure, axes = plt.subplots(len(bits), len(models), figsize=(11, 3 + 0.33 * len(keys)),
                                sharex=True, sharey=True)
    for row, width in enumerate(bits):
        for column, (model, model_name) in enumerate(models.items()):
            ax = axes[row][column]
            found = drops.get((model, f"{config['slope_method']}-{width}bit", metric,
                               direction), {})
            for index, key in enumerate(keys):
                if key not in found:
                    continue
                full, quantized = found[key]["full"], found[key]["quantized"]
                ax.plot(found[key]["floor"], index, "|", color=MUTED, markersize=9)
                ax.plot([full, quantized], [index, index], color=GRID, linewidth=2, zorder=1)
                ax.plot(full, index, "o", color=COLOURS[0], markersize=5)
                ax.plot(quantized, index, "o", color=COLOURS[1], markersize=5)
            language_axis(ax, keys, config)
            style(ax, f"{model_name}, {width}-bit", metric)
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="") for c in COLOURS[:2]]
    handles.append(plt.Line2D([], [], color=MUTED, marker="|", markersize=9, linestyle=""))
    method = config["methods"][config["slope_method"]]
    figure.suptitle(f"{metric}, {direction}", x=0.01, ha="left", fontsize=10, color=INK)
    legend_above(figure, handles, ["Full precision", f"{method}, mean over seeds", "Floor"])
    return figure


def plot_drop_heatmap(drops: dict, keys: list, config: dict):
    """chrF++ drop of every language under every model and method, gray at the floor: rows
    directions, columns bit widths."""
    matrices = {}
    for direction in DIRECTIONS:
        for bits in config["bits"]:
            columns = [drops.get((model, f"{method}-{bits}bit", "chrF++", direction), {})
                       for model in config["models"] for method in config["methods"]]
            matrices[(direction, bits)] = grid(columns, keys, DROP[0])
    limit = max(np.nanmax(np.abs(matrix)) for matrix in matrices.values())
    figure, axes = plt.subplots(len(DIRECTIONS), len(config["bits"]), figsize=(13, 15),
                                sharex=True, sharey=True)
    for ax, ((direction, bits), matrix) in zip(axes.flatten(), matrices.items()):
        image = heatmap(ax, matrix, keys, limit, config)
        ax.set_title(f"{direction}, {bits}-bit", loc="left", fontsize=10, color=INK)
    bar = figure.colorbar(image, ax=axes, shrink=0.5, pad=0.02)
    bar.set_label("chrF++ drop (points); gray: at floor", fontsize=8, color=MUTED)
    return figure


def off_target(config: dict, runs: list[str], direction: str) -> dict:
    """Return {(language, test set): off-target %, mean over the runs}, read from scores.json
    (qlrl.evaluation) of the runs in every evaluation folder."""
    found = {}
    for run in runs:
        for evaluation in config["evaluations"]:
            path = Path(evaluation) / run / "scores.json"
            if not path.exists():
                continue
            for code, entry in json.loads(path.read_text()).items():
                lang = code.lower()
                pair = f"eng→{lang}" if direction == "eng→X" else f"{lang}→eng"
                key = (lang, entry["test set"])
                found.setdefault(key, []).append(entry[pair]["off-target %"])
    return {key: sum(values) / len(values) for key, values in found.items()}


def plot_off_target(keys: list, config: dict):
    """Share of outputs in the wrong language (GlotLID), at full precision and at each bit
    width (method "slope_method", mean over seeds): rows directions, columns models."""
    method, seeds = config["slope_method"], config["seeds"]
    figure, axes = plt.subplots(len(DIRECTIONS), len(config["models"]),
                                figsize=(11, 3 + 0.5 * len(keys)), sharex=True, sharey=True)
    names = ["Full precision"] + [f"{bits}-bit" for bits in config["bits"]]
    for row, direction in enumerate(DIRECTIONS):
        for column, (model, model_name) in enumerate(config["models"].items()):
            ax = axes[row][column]
            runs = [[model]] + [[f"{model}-{method}-{bits}bit-seed{seed}" for seed in seeds]
                                for bits in config["bits"]]
            for colour, offset, run, label in zip(COLOURS, OFFSETS, runs, names):
                found = off_target(config, run, direction)
                shown = [(index, found[key]) for index, key in enumerate(keys) if key in found]
                ax.plot([share for _, share in shown], [index + offset for index, _ in shown],
                        "o", color=colour, markersize=4, label=label)
            language_axis(ax, keys, config)
            style(ax, f"{model_name}, {direction}", "Off-target (% of outputs)")
    legend_above(figure, *axes[0][0].get_legend_handles_labels())
    return figure
