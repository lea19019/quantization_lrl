'''
What an evaluation run writes: per-sentence rows, scores.json rounded for reading, and the
manifest with every version behind the numbers.
'''
import importlib.metadata
import json
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download


def rounded(scores: dict) -> dict:
    """Return a copy with every float rounded to 2 decimals, nested dicts included."""
    copy = {}
    for key, value in scores.items():
        if isinstance(value, dict):
            copy[key] = rounded(value)
        elif isinstance(value, float):
            copy[key] = round(value, 2)
        else:
            copy[key] = value
    return copy


def model_revision(path: str) -> dict:
    """Return what pins the evaluated weights: the Hub commit, or the folder and its git sha."""
    if Path(path).is_dir():
        manifest = Path(path) / "manifest.json"
        git_sha = json.loads(manifest.read_text()).get("git_sha") if manifest.exists() else None
        return {"folder": path, "git_sha": git_sha}
    # The cached snapshot folder is named after the Hub commit that from_pretrained loaded.
    snapshot = snapshot_download(path, local_files_only=True, allow_patterns="config.json")
    return {"repo": path, "commit": Path(snapshot).name}


def comet_version(comet_score: str) -> str:
    """Return the unbabel-comet version installed in the environment of comet-score."""
    python = Path(comet_score).parent / "python"
    code = "import importlib.metadata as m; print(m.version('unbabel-comet'))"
    found = subprocess.run([str(python), "-c", code], capture_output=True, text=True,
                           check=False)
    return found.stdout.strip() or "not found"


def versions(config: dict, signatures: dict) -> dict:
    """Return the model revision, library versions, GlotLID, AfriCOMET and metric signatures."""
    block = {"model": model_revision(config["model"])}
    for package in ("torch", "transformers", "sacrebleu", "fasttext-numpy2-wheel"):
        block[package] = importlib.metadata.version(package)
    block["unbabel-comet"] = comet_version(config["africomet"]["comet_score"])
    block["GlotLID"] = {"repo": config["glotlid"]["repo"],
                        "revision": config["glotlid"]["revision"]}
    block["AfriCOMET"] = {"repo": config["africomet"]["repo"],
                          "revision": config["africomet"]["revision"]}
    for metric, signature in signatures.items():
        block[f"{metric} signature"] = signature
    return block


def write_manifest(folder: Path, config: dict, details: dict) -> None:
    """Write manifest.json: config, git sha, what each test set held, and versions."""
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=False).stdout.strip()
    manifest = {"config": config, "git_sha": git_sha,
                "test sets": rounded(details["languages"]),
                "versions": versions(config, details["signatures"])}
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    (folder / "manifest.json").write_text(text)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
