'''
Clean a set of pairs with the lab's data-cleaning tool (~/data-cleaning-pipeline).

Used on the external train sets only: the church data is already cleaned by the lab, and
dev and test sets stay exactly as published so scores compare with other papers.

The tool has its own environment (it needs editdistance and tqdm), so clean() runs this
same file with the tool's Python, from the tool's folder (it reads config/ from there).
The tool's command-line mode for plain text files is broken (pipeline.py reads
args.keep_meta, which does not exist), so the bottom of this file calls its Pipeline class.

The tool's work folder keeps src.txt and tgt.txt (input), cleaned/ and deleted/ (what
was removed, for inspection).
'''
import os
import subprocess
import sys


def clean(pairs: list, folder: str, cleaner: str, config_name: str) -> list[tuple[str, str]]:
    """Return the pairs the cleaning tool keeps. `config_name` picks its language settings."""
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)
    for index, name in enumerate(("src", "tgt")):
        with open(f"{folder}/{name}.txt", "w", encoding="utf-8") as handle:
            handle.writelines(pair[index] + "\n" for pair in pairs)
    python = f"{cleaner}/.venv/bin/python"
    command = [python, os.path.abspath(__file__), folder, config_name]
    subprocess.run(command, cwd=cleaner, check=True)
    sides = []
    for name in ("src", "tgt"):
        with open(f"{folder}/cleaned/{name}.txt", encoding="utf-8") as handle:
            sides.append([line.rstrip("\n") for line in handle])
    return list(zip(*sides, strict=True))


def run_tool(folder: str, config_name: str) -> None:
    """Run the tool on folder/{src,tgt}.txt. Runs inside the tool's Python and folder."""
    # The tool's own folder first, so its `clean` module is the one imported.
    sys.path.insert(0, os.getcwd())
    from pipeline import Pipeline

    Pipeline(
        folder,
        save_deleted=True,
        is_multi=False,
        parallel_text=True,
        srclang="en-US",
        tgtlang=config_name,
        srcpath=f"{folder}/src.txt",
        tgtpath=f"{folder}/tgt.txt",
    ).run_pipeline_parallel_text()


if __name__ == "__main__":
    run_tool(sys.argv[1], sys.argv[2])
