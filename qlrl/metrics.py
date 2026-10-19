'''
The measurements behind qlrl/eval.py: log-probabilities of given text, and scores of
generated translations. Settings and reasons: docs/project/evaluation.md.
'''
import json
import math
import os
import subprocess
import tempfile
from functools import cache
from pathlib import Path

import fasttext
import sacrebleu
import torch
from huggingface_hub import hf_hub_download

from qlrl.quant.utils.perplexity import gold_logprobs


def text_logprob(model, tokenizer, prefix: str, text: str) -> tuple[float, int]:
    """Return (summed log p in nats, token count) of the tokens of `text` given `prefix`."""
    # Tokenized apart and joined, so `text` keeps its own tokens: in one string Llama merges
    # the prompt's ">" with the first letter of the reference into a single token.
    prefix_ids = tokenizer(prefix).input_ids
    text_ids = tokenizer(text, add_special_tokens=False).input_ids
    ids = torch.tensor([prefix_ids + text_ids], device=model.device)  # [1, seq]
    with torch.no_grad():
        logits = model(ids).logits.float()  # [1, seq, vocab]; bf16 logits lose precision
    gold = gold_logprobs(logits, ids)  # [seq - 1]; gold[t - 1] scores token t
    return gold[len(prefix_ids) - 1:].sum().item(), len(text_ids)


def referenceLogProb_eval(model, tokenizer, prefixes: list[str], texts: list[str]):
    """Return, per text, its log p given its prefix, with token, word and byte counts."""
    rows = []
    for prefix, text in zip(prefixes, texts):
        logprob, tokens = text_logprob(model, tokenizer, prefix, text)
        rows.append({"logprob": logprob, "tokens": tokens,
                     "words": len(text.split()), "bytes": len(text.encode())})
    return rows


def perplexity_eval(model, tokenizer, texts: list[str]) -> list[dict]:
    """Return referenceLogProb_eval rows for bare sentences, after only a BOS token."""
    # Llama's tokenizer adds BOS itself; GPT-2's does not, so give it one to predict from.
    adds_bos = tokenizer("a").input_ids[0] == tokenizer.bos_token_id
    prefix = "" if adds_bos else tokenizer.bos_token
    return referenceLogProb_eval(model, tokenizer, [prefix] * len(texts), texts)


def summarize(rows: list[dict]) -> dict:
    """Return perplexity per token and negative log p per word and per byte, over all rows."""
    logprob = sum(row["logprob"] for row in rows)
    return {"perplexity": math.exp(-logprob / sum(row["tokens"] for row in rows)),
            "nll_per_word": -logprob / sum(row["words"] for row in rows),
            "nll_per_byte": -logprob / sum(row["bytes"] for row in rows)}


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
    """Return the share of outputs whose GlotLID label is not one of tgt's labels, and labels."""
    glotlid = load_glotlid(config["repo"], config["revision"])
    found = [glotlid.predict(text.replace("\n", " "))[0][0].removeprefix("__label__")
             for text in hypotheses]
    off = sum(label not in config["labels"][tgt] for label in found)
    return {"rate": off / len(found), "labels": found}
