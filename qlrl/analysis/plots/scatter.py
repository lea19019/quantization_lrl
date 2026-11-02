'''
Scatter figures of the likelihood gap, drawn by qlrl.analysis.plots.figures from damage.csv and
translation.csv (qlrl.analysis): does the gap follow how well the model knows the language,
does it follow the translation drop, and does fine-tuning change it.
'''
import matplotlib.pyplot as plt

from qlrl.analysis.plots.gap import GAP, group_colour
from qlrl.analysis.plots.style import COLOURS, MUTED, legend_above, style


def look(model: str) -> dict:
    """Point style of a model in a scatter: colour and marker by family, hollow with LoRA."""
    qwen = model.startswith("Qwen")
    colour = COLOURS[1] if qwen else COLOURS[0]
    return {"color": colour, "marker": "s" if qwen else "o", "linestyle": "", "markersize": 5,
            "markerfacecolor": "none" if model.endswith("-lora") else colour, "alpha": 0.85}


def plot_gap_vs_fp(gaps: dict, keys: list, config: dict):
    """Gap against full-precision NLL per byte, a point per model and language: rows bit
    widths, columns methods. The first model's points carry their language code."""
    bits, methods = config["bits"], config["methods"]
    figure, axes = plt.subplots(len(bits), len(methods), figsize=(12, 7.5), sharex=True,
                                sharey="row")
    for row, width in enumerate(bits):
        for column, (method, method_name) in enumerate(methods.items()):
            ax = axes[row][column]
            for index, (model, model_name) in enumerate(config["models"].items()):
                found = gaps.get((model, f"{method}-{width}bit"), {})
                xs = [found[key]["FP NLL per byte"] for key in keys if key in found]
                ys = [found[key][GAP[0]] for key in keys if key in found]
                ax.plot(xs, ys, **look(model), label=model_name)
                if index == 0:
                    for x, y, key in zip(xs, ys, [key for key in keys if key in found]):
                        ax.annotate(key[0].upper(), (x, y), xytext=(4, 2), fontsize=6,
                                    textcoords="offset points")
            ax.axhline(0, color=MUTED, linewidth=0.8)
            style(ax, f"{method_name}, {width}-bit", "Full-precision NLL (nats per byte)",
                  "Gap (nats per byte)")
    legend_above(figure, *axes[0][0].get_legend_handles_labels())
    return figure


def plot_gap_vs_translation(gaps: dict, drops: dict, config: dict):
    """chrF++ drop against gap, a point per model, method and language, floors left out:
    rows directions, columns bit widths. drops = {(model, config, metric, direction): rows}."""
    directions = ("eng→X", "X→eng")
    figure, axes = plt.subplots(len(directions), len(config["bits"]), figsize=(10, 8),
                                sharex="col", sharey="col")
    for row, direction in enumerate(directions):
        for column, bits in enumerate(config["bits"]):
            ax = axes[row][column]
            for model, model_name in config["models"].items():
                xs, ys = [], []
                for method in config["methods"]:
                    run = (model, f"{method}-{bits}bit")
                    for key, drop in drops.get(run + ("chrF++", direction), {}).items():
                        if drop["reading"] != "at floor" and key in gaps.get(run, {}):
                            xs.append(gaps[run][key][GAP[0]])
                            ys.append(drop["drop"])
                ax.plot(xs, ys, **look(model), label=model_name)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            style(ax, f"{direction}, {bits}-bit", "Gap (nats per byte)", "chrF++ drop")
    legend_above(figure, *axes[0][0].get_legend_handles_labels())
    return figure


def plot_finetune_gap(gaps: dict, config: dict):
    """Gap of Llama 3.1 8B + LoRA against the base model, a point per language, test set and
    method (marker), coloured by language group; one panel per bit width; y = x dashed."""
    figure, axes = plt.subplots(1, len(config["bits"]), figsize=(10, 4.5))
    markers = ("o", "s", "^")
    for ax, bits in zip(axes, config["bits"]):
        for marker, method in zip(markers, config["methods"]):
            base = gaps[("Llama-3.1-8B", f"{method}-{bits}bit")]
            tuned = gaps[("Llama-3.1-8B-lora", f"{method}-{bits}bit")]
            for key in base.keys() & tuned.keys():
                ax.plot(base[key][GAP[0]], tuned[key][GAP[0]], marker, markersize=5,
                        color=group_colour(key[0], config), alpha=0.85)
        low, high = min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1],
                                                                 ax.get_ylim()[1])
        ax.plot([low, high], [low, high], color=MUTED, linewidth=0.8, linestyle="--")
        style(ax, f"{bits}-bit", "Gap, Llama 3.1 8B (nats per byte)",
              "Gap, Llama 3.1 8B + LoRA")
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="") for c in COLOURS[:3]]
    handles += [plt.Line2D([], [], color=MUTED, marker=m, linestyle="") for m in markers]
    legend_above(figure, handles, list(config["groups"]) + list(config["methods"].values()))
    return figure
