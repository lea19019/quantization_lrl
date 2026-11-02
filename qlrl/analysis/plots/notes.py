'''
The Figures section of the analysis summary.md: each figure with its title and how to read it,
from configs/analysis/figure_notes.yaml.
'''
import os
from pathlib import Path

START, END = "<!-- figures -->", "<!-- /figures -->"


def link_figures(summary: Path, figures: Path, notes: dict) -> None:
    """Put the Figures section in summary.md after its legend, replacing an older one."""
    text = summary.read_text()
    if START in text:
        text = text[:text.index(START)] + text[text.index(END) + len(END):].lstrip("\n")
    folder = os.path.relpath(figures, summary.parent)
    block = [START, "## Figures", ""]
    for number, (name, note) in enumerate(notes.items(), start=1):
        block += [f"### Figure {number}. {note['title']}", "",
                  f"![{note['title']}]({folder}/{name}.png)", "",
                  f"**How to read.** {note['how']}", ""]
    head, _, tail = text.partition("\n## ")
    summary.write_text(head.rstrip("\n") + "\n\n" + "\n".join(block + [END]) + "\n\n## " + tail)
