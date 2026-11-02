'''
Every figure of the analysis, from damage.csv and translation.csv just written, scores.json
of the runs (off-target) and the fine-tuning histories.
'''
from pathlib import Path

import yaml

from qlrl.analysis.plots.finetune import dev_groups, plot_finetune_loss, plot_overlap, read_history
from qlrl.analysis.plots.gap import plot_bits_slope, plot_gap, plot_gap_heatmap
from qlrl.analysis.plots.notes import link_figures
from qlrl.analysis.plots.scatter import plot_finetune_gap, plot_gap_vs_fp, plot_gap_vs_translation
from qlrl.analysis.plots.style import by_run, languages, read_csv, save
from qlrl.analysis.plots.translation import (
    plot_drop,
    plot_drop_heatmap,
    plot_off_target,
    plot_scores,
)


def draw_figures(config: dict, analysis: Path) -> None:
    """Draw every figure of the analysis folder into config["output"], the fine-tuning loss
    figures too, and link them in its summary.md."""
    figures = Path(config["output"])
    damage_rows = read_csv(analysis / "damage.csv")
    drop_rows = read_csv(analysis / "translation.csv")
    gaps = by_run(damage_rows, "model", "config")
    drops = by_run(drop_rows, "model", "config", "metric", "direction")
    keys = languages(damage_rows, config)

    save(plot_gap(gaps, keys, config), figures / "results_gap")
    save(plot_gap_heatmap(gaps, keys, config), figures / "results_gap_heatmap")
    save(plot_bits_slope(gaps, keys, config), figures / "results_bits_slope")
    save(plot_gap_vs_fp(gaps, keys, config), figures / "results_gap_vs_fp")
    save(plot_gap_vs_translation(gaps, drops, config), figures / "results_gap_vs_translation")
    save(plot_finetune_gap(gaps, config), figures / "results_finetune_gap")
    for metric, part in (("chrF++", "chrf"), ("AfriCOMET", "africomet")):
        metric_keys = languages([row for row in drop_rows if row["metric"] == metric], config)
        for direction, way in (("eng→X", "eng-x"), ("X→eng", "x-eng")):
            save(plot_drop(drops, metric_keys, metric, direction, config),
                 figures / f"results_{part}_drop_{way}")
            save(plot_scores(drops, metric_keys, metric, direction, config),
                 figures / f"results_{part}_scores_{way}")
    save(plot_drop_heatmap(drops, keys, config), figures / "results_chrf_drop_heatmap")
    save(plot_off_target(keys, config), figures / "results_off_target")

    for name, history_path in config["finetune"].items():
        history = read_history(Path(history_path))
        groups = dev_groups(history, config)
        save(plot_finetune_loss(history, groups), figures / name)
        save(plot_overlap(history, groups), figures / f"{name}_overlap")

    notes = yaml.safe_load(Path(config["notes"]).read_text())
    link_figures(analysis / "summary.md", figures, notes)
