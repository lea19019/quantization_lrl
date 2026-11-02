'''
LaTeX pieces shared by qlrl.results_tables and qlrl.summary_tables: numbers, shaded loss
cells, two-level booktabs headers, and the table environments.

A loss cell is tinted with the colour "loss" (defined in report.tex) in proportion to its
size within its column group: the largest loss gets SHADE percent of the colour, no loss or
a gain gets none. The number is always printed, so the table still reads in black and white.
'''
# Written by Claude Code at Adrian's request, 2026-10-29.

SHADE = 60  # percent of the shading colour used for the largest loss


def number(value: float | None, decimals: int) -> str:
    """Return a value with fixed decimals, or a dash where it was not computed."""
    return "--" if value is None else f"{value:.{decimals}f}"


def shaded(value: float | None, loss: float | None, largest: float, decimals: int) -> str:
    """Return the cell text of a value, tinted by its loss relative to the largest loss."""
    text = number(value, decimals)
    if loss is None or loss <= 0 or largest <= 0:
        return text
    return f"\\cellcolor{{loss!{round(SHADE * loss / largest)}}}{text}"


def header(text_columns: list[str], groups: list[str], subs: list[str]) -> list[str]:
    """Return a two-level booktabs header: each group spans one column per sub-name."""
    first = len(text_columns)
    top = " & " * (first - 1) + "".join(f" & \\multicolumn{{{len(subs)}}}{{c}}{{{g}}}"
                                        for g in groups)
    rules = "".join(f"\\cmidrule(lr){{{first + 1 + i * len(subs)}-{first + (i + 1) * len(subs)}}}"
                    for i in range(len(groups)))
    second = " & ".join(text_columns + subs * len(groups))
    return ["\\toprule", top + " \\\\", rules, second + " \\\\", "\\midrule"]


def longtable(caption: str, columns: int, head: list[str], body: list[str]) -> str:
    """Return a landscape longtable in a small font, its header repeated on every page."""
    align = "ll" + "r" * (columns - 2)
    lines = ["\\begin{landscape}", "\\tiny", "\\setlength{\\tabcolsep}{2pt}",
             f"\\begin{{longtable}}{{{align}}}", f"\\caption{{{caption}}}\\\\"]
    lines += head + ["\\endfirsthead"] + head + ["\\endhead", "\\bottomrule", "\\endfoot"]
    return "\n".join(lines + body + ["\\end{longtable}", "\\end{landscape}"])


def float_table(caption: str, columns: int, lines: list[str]) -> str:
    """Return a page-width table float around the rows."""
    return "\n".join(["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
                      "\\setlength{\\tabcolsep}{3pt}", f"\\caption{{{caption}}}",
                      f"\\begin{{tabular}}{{l{'r' * (columns - 1)}}}"] + lines
                     + ["\\bottomrule", "\\end{tabular}", "\\end{table}"])
