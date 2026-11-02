'''
Figures of the likelihood gap, drawn by qlrl.analysis.plots.figures from damage.csv (qlrl.analysis).

The gap is the damage per byte of a language's text minus that of its English side (nats per
byte, mean over seeds), with a 95% paired bootstrap interval. Every function takes
gaps = {(model, config): {(language, test set): row}} and the language rows to show.
'''
import matplotlib.pyplot as plt
import numpy as np

from qlrl.analysis.plots.style import (
    COLOURS,
    INK,
    MUTED,
    dots,
    end_labels,
    grid,
    heatmap,
    language_axis,
    legend_above,
    style,
)

GAP = ("gap per byte", "gap low per byte", "gap high per byte")
OFFSETS = (-0.25, 0.0, 0.25)  # one row per language, the methods side by side in it


def group_colour(lang: str, config: dict) -> str:
    """Return the colour of the language's group (high-resource, African, Latin American)."""
    for colour, members in zip(COLOURS, config["groups"].values()):
        if lang in members:
            return colour
    raise KeyError(lang)


def plot_gap(gaps: dict, keys: list, config: dict):
    """Gap per language, a dot and interval per method: rows bit widths, columns models."""
    bits, models = config["bits"], config["models"]
    figure, axes = plt.subplots(len(bits), len(models), figsize=(11, 10), sharex="row",
                                sharey=True)
    for row, width in enumerate(bits):
        for column, (model, model_name) in enumerate(models.items()):
            ax = axes[row][column]
            for colour, offset, (method, method_name) in zip(COLOURS, OFFSETS,
                                                             config["methods"].items()):
                found = gaps.get((model, f"{method}-{width}bit"), {})
                dots(ax, keys, found, GAP, offset, {"color": colour, "label": method_name})
            ax.axvline(0, color=MUTED, linewidth=0.8)
            language_axis(ax, keys, config)
            style(ax, f"{model_name}, {width}-bit", "Gap (nats per byte)")
    legend_above(figure, *axes[0][0].get_legend_handles_labels())
    return figure


def plot_gap_heatmap(gaps: dict, keys: list, config: dict):
    """Gap of every language under every model and method, one panel per bit width."""
    matrices = []
    for bits in config["bits"]:
        columns = [gaps.get((model, f"{method}-{bits}bit"), {})
                   for model in config["models"] for method in config["methods"]]
        matrices.append(grid(columns, keys, GAP[0]))
    limit = max(np.nanmax(np.abs(matrix)) for matrix in matrices)
    figure, axes = plt.subplots(1, len(matrices), figsize=(13, 8), sharey=True)
    for ax, bits, matrix in zip(axes, config["bits"], matrices):
        image = heatmap(ax, matrix, keys, limit, config)
        ax.set_title(f"{bits}-bit", loc="left", fontsize=10, color=INK)
    bar = figure.colorbar(image, ax=axes, shrink=0.6, pad=0.02)
    bar.set_label("Gap (nats per byte)", fontsize=8, color=MUTED)
    return figure


def plot_bits_slope(gaps: dict, keys: list, config: dict):
    """Gap at the widest and the narrowest bit width joined by a line, one line per language
    (its first test set), one panel per model, method "slope_method"."""
    first = {}
    for lang, test in keys:
        first.setdefault(lang, (lang, test))
    method, (wide, narrow) = config["slope_method"], (config["bits"][0], config["bits"][-1])
    figure, axes = plt.subplots(1, len(config["models"]), figsize=(17, 6.5), sharey=True)
    ends = []
    for ax, (model, model_name) in zip(axes, config["models"].items()):
        low, high = gaps[(model, f"{method}-{wide}bit")], gaps[(model, f"{method}-{narrow}bit")]
        found = [key for key in first.values() if key in low and key in high]
        for key in found:
            ax.plot([0, 1], [low[key][GAP[0]], high[key][GAP[0]]], marker="o", markersize=3,
                    color=group_colour(key[0], config), linewidth=1.5)
        ends.append([(high[key][GAP[0]], config["languages"][key[0]]) for key in found])
        ax.set_xlim(-0.1, 1.1)
        ax.set_xticks([0, 1], [f"{wide}-bit", f"{narrow}-bit"])
        style(ax, model_name, "", "Gap (nats per byte)")
    bottom, top = axes[0].get_ylim()  # final, now that every panel is drawn
    for ax, labels in zip(axes, ends):
        end_labels(ax, labels, 1, 0.035 * (top - bottom))
    handles = [plt.Line2D([], [], color=colour, linewidth=1.5) for colour in COLOURS[:3]]
    legend_above(figure, handles, list(config["groups"]))
    return figure
