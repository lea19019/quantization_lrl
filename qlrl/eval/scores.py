'''
Scores of generated translations: chrF++, BLEU, AfriCOMET and the off-target rate.
Settings and reasons: docs/project/evaluation.md.
'''
import json
import os
import subprocess
import tempfile
from functools import cache
from pathlib import Path

import fasttext
import sacrebleu
import torch
from huggingface_hub import hf_hub_download


def chrfPP_eval(hypotheses: list[str], references: list[str]) -> dict:
    """Return corpus chrF++ (char order 6, word order 2, beta 2, case-sensitive)."""
    chrf = sacrebleu.CHRF(word_order=2)
    score = chrf.corpus_score(hypotheses, [references]).score
    return {"score": score, "signature": str(chrf.get_signature())}


def sacreBLEU_eval(hypotheses: list[str], references: list[str]) -> dict:
    """Return corpus BLEU (mixed case, 13a tokenization, exponential smoothing)."""
    bleu = sacrebleu.BLEU(tokenize="13a", smooth_method="exp")
    score = bleu.corpus_score(hypotheses, [references]).score
    return {"score": score, "signature": str(bleu.get_signature())}


def afriCOMET_eval(sources, hypotheses, references, config: dict) -> dict:
    """Return AfriCOMET x 100 (mean over segments) and the per-segment scores (0-1)."""
    checkpoint = hf_hub_download(config["repo"], "checkpoints/model.ckpt",
                                 revision=config["revision"])
    with tempfile.TemporaryDirectory() as folder:
        paths = [str(Path(folder) / name) for name in ("src", "mt", "ref", "out.json")]
        for path, lines in zip(paths, (sources, hypotheses, references)):
            Path(path).write_text("".join(line.replace("\n", " ") + "\n" for line in lines))
        # unbabel-comet lives in its own environment; its comet-score CLI does the work.
        # Offline: the encoder's tokenizer is already cached, and compute nodes have no web.
        subprocess.run([config["comet_score"], "-s", paths[0], "-t", paths[1], "-r", paths[2],
                        "--model", checkpoint, "--gpus", str(int(torch.cuda.is_available())),
                        "--batch_size", str(config["batch_size"]), "--to_json", paths[3],
                        "--quiet"], check=True, env=os.environ | {"HF_HUB_OFFLINE": "1"})
        segments = [row["COMET"] for row in json.loads(Path(paths[3]).read_text())[paths[1]]]
    return {"score": 100 * sum(segments) / len(segments), "segments": segments}


@cache
def load_glotlid(repo: str, revision: str):
    """Return the GlotLID fastText model, loaded once (1.7 GB)."""
    return fasttext.load_model(hf_hub_download(repo, "model.bin", revision=revision))


def offTargert_eval(hypotheses: list[str], config: dict, tgt: str) -> dict:
    """Return the share of off-target outputs and the GlotLID label of every output."""
    # Only "is it English?" is asked. GlotLID cannot separate close varieties (on the human
    # references it labels 91% of Kichwa 'qvi' and 27% of church Swahili 'swc'), so a list of
    # accepted labels per language would count good outputs as off target.
    # English -> X: off target if the output is English (the model fell back to English).
    # X -> English: off target if the output is anything but English.
    glotlid = load_glotlid(config["repo"], config["revision"])
    found = [glotlid.predict(text.replace("\n", " "))[0][0].removeprefix("__label__")
             for text in hypotheses]
    if tgt == "eng":
        off = sum(label != "eng_Latn" for label in found)
    else:
        off = sum(label == "eng_Latn" for label in found)
    return {"rate": off / len(found), "labels": found}


def score_translations(rows: list[dict], lang: str, tgt: str, config: dict):
    """Return (scores, metric signatures) of the outputs in rows; add per-sentence columns."""
    sources = [row["src"] for row in rows]
    hypotheses = [row["hyp"] for row in rows]
    references = [row["ref"] for row in rows]
    metrics, scores, signatures = config["metrics"], {}, {}
    if "chrf" in metrics:
        chrf = chrfPP_eval(hypotheses, references)
        scores["chrF++"], signatures["chrF++"] = chrf["score"], chrf["signature"]
    if "bleu" in metrics:
        bleu = sacreBLEU_eval(hypotheses, references)
        scores["BLEU"], signatures["BLEU"] = bleu["score"], bleu["signature"]
    if "africomet" in metrics and lang in config["africomet"]["languages"]:
        comet = afriCOMET_eval(sources, hypotheses, references, config["africomet"])
        for row, segment in zip(rows, comet["segments"]):
            row["africomet"] = segment
        scores["AfriCOMET"] = comet["score"]
    if "off_target" in metrics:
        lid = offTargert_eval(hypotheses, config["glotlid"], tgt)
        for row, label in zip(rows, lid["labels"]):
            row["lid"] = label
        scores["off-target %"] = 100 * lid["rate"]
    return scores, signatures
