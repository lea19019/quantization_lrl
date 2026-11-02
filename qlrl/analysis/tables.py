'''
Tables of the report, written at the end of qlrl.analysis.

Reads scores.json of every run (qlrl.evaluation) and writes LaTeX that report.tex reads with
\\input. GPTQ and AWQ numbers are the mean over the calibration seeds; RTN and FP are one run.
A loss is how much worse quantized is than full precision (> 0: worse). Loss cells are shaded
by their size within their column, so the languages that lose most stand out.

Summary tables, per bit width and summary: rows are languages grouped by resource level;
columns, per model, the FP value and the loss of each method.
Full tables (appendix), per bit width: one translation table per language group, rows
English→X then X→English, one row per language and method; then one sentence table.
'''
from pathlib import Path

from qlrl.analysis.latex import float_table, header, longtable, number, shaded
from qlrl.analysis.scores import (
    label,
    languages,
    largest,
    loss,
    mean_scores,
    run_folders,
)

HALVES = [("English$\\rightarrow$X", "eng→X"), ("X$\\rightarrow$English", "X→eng")]


def full_table(config: dict, scores: dict, codes: list[str], halves: list, names: list):
    """Return (head, body) of a full table. halves: [(title or None, [field], [decimals])];
    every half has the same columns, named by names."""
    keys, models = languages(config, scores, codes), list(config["models"])
    head = header(["Language (test set)", "Method"], names, list(config["models"].values()))
    body = []
    for title, fields, decimals in halves:
        most = [largest(config, scores, keys, field, models) for field in fields]
        if title:
            span = 2 + len(fields) * len(models)
            body.append(f"\\multicolumn{{{span}}}{{l}}{{\\textbf{{{title}}}}} \\\\ \\midrule")
        for key in keys:
            for index, (method, method_name) in enumerate(config["methods"].items()):
                cells = [label(config, key) if index == 0 else "", method_name]
                for field, places, top in zip(fields, decimals, most):
                    for model in models:
                        fp = scores[(model, "fp")].get(key, {}).get(field)
                        found = scores[(model, method)].get(key, {}).get(field)
                        cells.append(number(found, places) if method == "fp" else
                                     shaded(found, loss(config, field, fp, found), top, places))
                body.append(" & ".join(cells) + " \\\\")
            body.append("\\midrule")
    return head, body


def full_tables(config: dict, scores: dict, bits: int) -> list[str]:
    """Return the full tables of one bit width: each language group, then sentences."""
    tables = []
    models = len(config["models"])
    for group in config["full_tables"]:
        metrics = group["metrics"]
        halves = [(title, [(where, m) for m in metrics], [d for _, d in metrics.values()])
                  for title, where in HALVES]
        head, body = full_table(config, scores, group["languages"], halves,
                                [name for name, _ in metrics.values()])
        caption = f"{group['title']}, {bits}-bit: translation and reference likelihood."
        tables.append(longtable(caption, 2 + len(metrics) * models, head, body))
    metrics = config["sentence_metrics"]
    fields = [(where, m) for where in ("eng", "X") for m in metrics]
    names = [f"{side} {name}" for side in ("Eng", "X") for name, _ in metrics.values()]
    decimals = [d for _ in ("eng", "X") for _, d in metrics.values()]
    head, body = full_table(config, scores, list(config["languages"]),
                            [(None, fields, decimals)], names)
    caption = f"Sentence likelihood, {bits}-bit: each test sentence on its own, English and X side."
    tables.append(longtable(caption, 2 + len(fields) * models, head, body))
    return tables


def summary_row(config: dict, scores: dict, key: tuple, column: dict) -> str:
    """Return one summary row: per model, the FP value then the loss of each method.
    column: {"field": (where, metric), "decimals": [FP, loss], "top": largest loss}."""
    field, top, (fp_places, loss_places) = column["field"], column["top"], column["decimals"]
    cells = [label(config, key)]
    for model in config["summary_models"]:
        fp = scores[(model, "fp")].get(key, {}).get(field)
        cells.append(number(fp, fp_places))
        for method in config["methods"]:
            if method != "fp":
                lost = loss(config, field, fp, scores[(model, method)].get(key, {}).get(field))
                cells.append(shaded(lost, lost, top, loss_places))
    return " & ".join(cells) + " \\\\"


def summary_table(config: dict, scores: dict, spec: dict, bits: int) -> str:
    """Return one summary table, rows grouped by resource level, for "summary_models"."""
    models, methods = config["summary_models"], [m for m in config["methods"] if m != "fp"]
    span = 1 + (1 + len(methods)) * len(models)
    lines = header(["Language (test set)"], [config["models"][m] for m in models],
                   ["FP"] + [config["methods"][m] for m in methods])
    groups = {group: languages(config, scores, codes)
              for group, codes in config["groups"].items()}
    every = [key for keys in groups.values() for key in keys]
    halves = HALVES if spec["side"] is None else [(None, spec["side"])]
    for half, (title, where) in enumerate(halves):
        field = (where, spec["metric"])
        top = largest(config, scores, every, field, models)
        if title:
            rule = "\\midrule " if half else ""
            bold = f"\\textbf{{{title}}}"
            lines.append(f"{rule}\\multicolumn{{{span}}}{{l}}{{{bold}}} \\\\ \\midrule")
        for index, (group, keys) in enumerate(groups.items()):
            rule = "\\midrule " if index else ""
            lines.append(f"{rule}\\multicolumn{{{span}}}{{l}}{{\\textit{{{group}}}}} \\\\")
            column = {"field": field, "decimals": spec["decimals"], "top": top}
            lines += [summary_row(config, scores, key, column) for key in keys]
    return float_table(f"{spec['title']}, {bits}-bit.", span, lines)


def write_tables(config: dict, output: Path) -> None:
    """Write summary_tables.tex and full_tables.tex to output, every bit width."""
    summaries, fulls = [], []
    for bits in config["bits"]:
        scores = {(model, method): mean_scores(config, run_folders(model, method, bits,
                                                                    config["seeds"]))
                  for model in config["models"] for method in config["methods"]}
        summaries += [summary_table(config, scores, spec, bits) for spec in config["summaries"]]
        fulls += full_tables(config, scores, bits)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary_tables.tex").write_text("\n\n".join(summaries) + "\n")
    (output / "full_tables.tex").write_text("\n\n".join(fulls) + "\n")
    print(f"wrote {output}/summary_tables.tex and full_tables.tex")
