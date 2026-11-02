'''
Fine-tuning loss figures, drawn by qlrl.analysis.plots.figures.

history.jsonl (qlrl.finetune) has one row per logged step: "train_loss" every log_every
steps, and "dev_loss" {lang: loss, "mean": ...} at step 0 and every eval_every steps. Both
are mean cross-entropy per graded token (nats), so they share one axis.

Figure 1: (a) training loss and mean dev loss, then the dev loss per language, one panel per
language group, all on the same y scale. Figure 2: every curve in one panel, colour the
group and marker the language. Every line has its name at its end.
'''
import json
from pathlib import Path

import matplotlib.pyplot as plt

from qlrl.analysis.plots.style import COLOURS, INK, MUTED, end_labels, style

MARKERS = ["o", "s", "^", "D", "v"]


def read_history(path: Path) -> tuple[list, list, list, list]:
    """Return (train steps, train losses, dev steps, dev rows {lang: loss, "mean": loss})."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    train = [row for row in rows if "train_loss" in row]
    dev = [row for row in rows if "dev_loss" in row]
    return ([row["step"] for row in train], [row["train_loss"] for row in train],
            [row["step"] for row in dev], [row["dev_loss"] for row in dev])


def dev_groups(history: tuple, config: dict) -> dict:
    """Return {group: {language: name}} of the languages in the dev loss, in config order."""
    dev = history[3][0]
    return {group: {lang: config["languages"][lang] for lang in codes if lang in dev}
            for group, codes in config["groups"].items()}


def plot_finetune_loss(history: tuple, groups: dict):
    """Figure 1: panel (a) training and mean dev loss, then one dev panel per group."""
    steps, losses, dev_steps, dev = history
    figure, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    axes = axes.flatten()
    for ax in axes[2:]:
        ax.sharey(axes[1])  # before plotting, so the shared scale covers every group
    axes[0].plot(steps, losses, color=COLOURS[0], linewidth=1.5, label="train")
    axes[0].plot(dev_steps, [row["mean"] for row in dev], color=COLOURS[1], linewidth=2,
                 marker="o", markersize=3, label="dev (mean over languages)")
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK)
    style(axes[0], "(a) Training and mean dev loss", "", "Loss (nats per token)")
    for ax, letter, (group, names) in zip(axes[1:], "bcd", groups.items()):
        ends = []
        for colour, (lang, name) in zip(COLOURS, names.items()):
            curve = [row[lang] for row in dev]
            ax.plot(dev_steps, curve, color=colour, linewidth=2, marker="o", markersize=3)
            ends.append((curve[-1], name))
        style(ax, f"({letter}) Dev loss, {group}", "Step" if letter != "b" else "")
        end_labels(ax, ends, dev_steps[-1], gap=0.12)
    axes[2].set_ylabel("Loss (nats per token)", fontsize=8, color=MUTED)
    figure.tight_layout()
    return figure


def plot_overlap(history: tuple, groups: dict):
    """Figure 2: every curve in one panel, colour the group and marker the language."""
    steps, losses, dev_steps, dev = history
    figure, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(steps, losses, color=MUTED, linewidth=1, label="train")
    ax.plot(dev_steps, [row["mean"] for row in dev], color=INK, linewidth=1.5, linestyle="--",
            label="dev (mean over languages)")
    ends = [(dev[-1]["mean"], "dev mean")]
    for colour, (group, names) in zip(COLOURS, groups.items()):
        ax.plot([], [], color=colour, linewidth=2, label=f"dev, {group}")  # legend entry only
        for marker, (lang, name) in zip(MARKERS, names.items()):
            curve = [row[lang] for row in dev]
            ax.plot(dev_steps, curve, color=colour, linewidth=1.5, marker=marker, markersize=4)
            ends.append((curve[-1], name))
    end_labels(ax, ends, dev_steps[-1], gap=0.07)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    style(ax, "Training loss and dev loss per language", "Step", "Loss (nats per token)")
    figure.tight_layout()
    return figure
