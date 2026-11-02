'''
What every figure shares: colours, panel style, the legend above the panels, saving, and
reading the analysis CSVs (qlrl.analysis) with their numbers as floats.
'''
import csv
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Categorical colours, checked for colour-vision deficiency on white. Never cycled.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
# Diverging blue <-> red with a neutral midpoint, for signed values centred on 0.
DIVERGING = LinearSegmentedColormap.from_list("signed", ["#256abf", "#f0efec", "#e34948"])


def read_csv(path: Path) -> list[dict]:
    """Return the rows of a CSV, every number as a float and every empty cell as NaN."""
    with open(path, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value or "nan")
            except ValueError:
                pass  # text: model, config, language, reading
    return rows


def by_run(rows: list[dict], *fields: str) -> dict:
    """Return {(row[field], ...): {(language, test set): row}}, e.g. by model and config."""
    table = {}
    for row in rows:
        run = tuple(row[field] for field in fields)
        table.setdefault(run, {})[(row["language"], row["test set"])] = row
    return table


def languages(rows: list[dict], config: dict) -> list[tuple[str, str]]:
    """Return the (language, test set) pairs in the rows, in the config's order."""
    present = {(row["language"], row["test set"]) for row in rows}
    return [(lang, test) for lang in config["languages"] for test in config["test_sets"]
            if (lang, test) in present]


def name(key: tuple[str, str], config: dict) -> str:
    """Return "Language (test set)"."""
    return f"{config['languages'][key[0]]} ({config['test_sets'][key[1]]})"


def style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Title on the left, light grid, no top or right spines, muted ticks and labels."""
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, labelbottom=True)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)


def language_axis(ax, keys: list, config: dict) -> None:
    """One row per language and test set, the first at the top."""
    ax.set_yticks(range(len(keys)), [name(key, config) for key in keys], fontsize=7)
    ax.set_ylim(len(keys) - 0.5, -0.5)


def dots(ax, keys: list, rows: dict, columns: tuple[str, str, str], offset: float,
         look: dict) -> None:
    """One dot per language row with its interval; columns = (value, low, high) of the CSV."""
    chosen = [(index, rows[key]) for index, key in enumerate(keys) if key in rows]
    if not chosen:
        return
    value, low, high = columns
    xs = [row[value] for _, row in chosen]
    errors = [[row[value] - row[low] for _, row in chosen],
              [row[high] - row[value] for _, row in chosen]]
    ax.errorbar(xs, [index + offset for index, _ in chosen], xerr=errors, fmt="o",
                markersize=3.5, elinewidth=1, capsize=0, **look)


def grid(columns: list[dict], keys: list, field: str) -> np.ndarray:
    """Return row[field] as [languages, columns], NaN where missing or read at the floor;
    one column per {(language, test set): row}."""
    matrix = np.full((len(keys), len(columns)), np.nan)
    for column, rows in enumerate(columns):
        for index, key in enumerate(keys):
            if key in rows and rows[key].get("reading") != "at floor":
                matrix[index, column] = rows[key][field]
    return matrix


def heatmap(ax, matrix, keys: list, limit: float, config: dict):
    """Draw matrix [languages, models x methods] on a scale from -limit to limit, centred
    on 0; NaN cells are gray. Return the image, for the colour bar."""
    image = ax.imshow(matrix, cmap=DIVERGING.with_extremes(bad="#a8a7a1"), vmin=-limit,
                      vmax=limit, aspect="auto")
    columns = [f"{model}, {method}" for model in config["models"].values()
               for method in config["methods"].values()]
    ax.set_xticks(range(len(columns)), columns, rotation=90, fontsize=7, color=INK)
    ax.set_yticks(range(len(keys)), [name(key, config) for key in keys], fontsize=7, color=INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    methods = len(config["methods"])
    for edge in range(methods, len(columns), methods):
        ax.axvline(edge - 0.5, color="white", linewidth=3)  # a gap between models
    return image


def legend_above(figure, handles: list, names: list[str]) -> None:
    """Lay out the panels and put the legend above them, where it covers no data."""
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.legend(handles, names, frameon=False, fontsize=8, labelcolor=INK, ncol=len(names),
                  loc="upper right")


def end_labels(ax, ends: list[tuple[float, str]], x: float, gap: float) -> None:
    """Write each name at the right end of its line (y, name), pushed at least `gap` apart."""
    ends = sorted(ends)
    placed = [y for y, _ in ends]
    for _ in range(200):  # move each too-close pair apart by half the shortfall, until none
        for i in range(len(placed) - 1):
            shortfall = gap - (placed[i + 1] - placed[i])
            if shortfall > 0:
                placed[i] -= shortfall / 2
                placed[i + 1] += shortfall / 2
    left, right = ax.get_xlim()
    leader = {"arrowstyle": "-", "color": MUTED, "linewidth": 0.5, "shrinkA": 0, "shrinkB": 2}
    for y, (line_y, label) in zip(placed, ends):
        ax.annotate(label, (x, line_y), xytext=(x + 0.04 * (right - left), y), va="center",
                    fontsize=7, color=INK, annotation_clip=False, arrowprops=leader)


def save(figure, output: Path) -> None:
    """Write the figure as <output>.png; report.tex and the markdown files include it."""
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {output}.png")
