'''
Entry point for evaluation: python -m qlrl.eval configs/eval.yaml

Loads one model (full precision or quantized) and, for every test set of every language in
the YAML and both directions (English -> X and X -> English), measures:

• Translation: three dev pairs of the same test set, then the sentence, as
      <eng>Hello.</eng> <yor>Báwo ni.</yor>  ...  <eng>{sentence}</eng> <yor>
  Greedy; stop at </yor> or at 2x the source tokens. Scored with chrF++ and BLEU, AfriCOMET
  on the African core, and the off-target rate (outputs not identified as the target).
• Reference log-probability: the same prompt followed by the human reference; the summed
  log p of the reference tokens (forced decoding). No generation, so no decoding noise.
• Perplexity: each bare sentence, no prompt. Compare a language with itself only.

Per-sentence numbers go to results/eval/<model>/, so damage (full minus quantized log p)
and every loss is a difference between two runs. Settings: docs/project/evaluation.md.
AfriCOMET runs in its own environment, because unbabel-comet pins an older transformers:
    uv venv .venv-comet --python 3.11
    uv pip install --python .venv-comet unbabel-comet==2.2.7 "setuptools<70"
'''
import json
import subprocess
import sys
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from qlrl.data.dataset import ParallelData
from qlrl.metrics import (
    afriCOMET_eval,
    chrfPP_eval,
    offTargert_eval,
    perplexity_eval,
    referenceLogProb_eval,
    sacreBLEU_eval,
    summarize,
)


def load_data(config: dict) -> list[dict]:
    """Return one dict per (language, source) test set: its dev examples and test rows."""
    data = ParallelData(config["data"])
    test_sets = []
    for lang in config["languages"]:
        for source in sorted(set(data.split("test", [lang])["source"])):
            if config["sources"] and source not in config["sources"]:
                continue
            test = list(data.split("test", [lang], [source]))
            dev = list(data.split("dev", [lang], [source]))
            test_sets.append({"lang": lang, "source": source,
                              "examples": dev[:config["examples"]],
                              "rows": test[:config["max_sentences"] or len(test)]})
    return test_sets


def get_prompts(examples: list[tuple[str, str]], sentences: list[str], src: str, tgt: str):
    """Return one prompt per sentence: the example pairs, then the sentence and an open tag."""
    shots = "\n".join(f"<{src}>{s}</{src}> <{tgt}>{t}</{tgt}>" for s, t in examples)
    return [f"{shots}\n<{src}>{sentence}</{src}> <{tgt}>" for sentence in sentences]


def translate(model, tokenizer, prompts: list[str], sources: list[str], tgt: str, batch_size):
    """Return the greedy continuation of each prompt, cut at </tgt> and at 2x source tokens."""
    stop = f"</{tgt}>"
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True)
        batch = batch.to(model.device)
        caps = [2 * len(tokenizer(s, add_special_tokens=False).input_ids)
                for s in sources[i:i + batch_size]]
        with torch.no_grad():
            ids = model.generate(**batch, max_new_tokens=max(caps), do_sample=False,
                                 stop_strings=[stop], tokenizer=tokenizer,
                                 pad_token_id=tokenizer.pad_token_id)
        new = ids[:, batch.input_ids.shape[1]:]  # [batch, new tokens], prompt dropped
        for row, cap in zip(new, caps):
            text = tokenizer.decode(row[:cap], skip_special_tokens=True)
            outputs.append(text.split(stop)[0].strip())
    return outputs


def load_model(path: str, dtype: str):
    """Return (model, tokenizer) on the GPU if there is one, ready for batched generation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype)).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token  # GPT-2 has none
    return model.to(device), tokenizer


def evaluate_direction(model, tokenizer, test_set: dict, src: str, config: dict, folder):
    """Translate one test set in one direction, score it, save per-sentence rows; return scores."""
    lang = test_set["lang"]
    tgt = lang if src == "eng" else "eng"
    side = {"eng": "en", lang: "tgt"}  # the column holding each language
    examples = [(row[side[src]], row[side[tgt]]) for row in test_set["examples"]]
    sources = [row[side[src]] for row in test_set["rows"]]
    references = [row[side[tgt]] for row in test_set["rows"]]
    prompts = get_prompts(examples, sources, src, tgt)
    metrics, scores = config["metrics"], {}
    rows = [{"src": s, "ref": r} for s, r in zip(sources, references)]

    if "reference_logprob" in metrics:
        logprobs = referenceLogProb_eval(model, tokenizer, prompts, references)
        rows = [row | logprob for row, logprob in zip(rows, logprobs)]
        scores["reference_logprob"] = summarize(logprobs)
    if {"chrf", "bleu", "africomet", "off_target"} & set(metrics):
        hypotheses = translate(model, tokenizer, prompts, sources, tgt, config["batch_size"])
        rows = [row | {"hyp": h} for row, h in zip(rows, hypotheses)]
    if "chrf" in metrics:
        scores["chrf++"] = chrfPP_eval(hypotheses, references)
    if "bleu" in metrics:
        scores["bleu"] = sacreBLEU_eval(hypotheses, references)
    if "africomet" in metrics and lang in config["africomet"]["languages"]:
        comet = afriCOMET_eval(sources, hypotheses, references, config["africomet"])
        rows = [row | {"africomet": c} for row, c in zip(rows, comet.pop("segments"))]
        scores["africomet"] = comet
    if "off_target" in metrics:
        lid = offTargert_eval(hypotheses, config["glotlid"], tgt)
        rows = [row | {"lid": label} for row, label in zip(rows, lid.pop("labels"))]
        scores["off_target"] = lid

    name = f"{lang}-{test_set['source']}-{src}2{tgt}"
    write_jsonl(folder / "translation" / f"{name}.jsonl", rows)
    return scores


def evaluate_model(model, tokenizer, config: dict, folder: Path) -> dict:
    """Run every measurement on every test set; return the scores keyed by set and direction."""
    scores = {}
    for test_set in load_data(config):
        lang, source = test_set["lang"], test_set["source"]
        for src in ("eng", lang):
            tgt = lang if src == "eng" else "eng"
            key = f"{lang}-{source}-{src}2{tgt}"
            scores[key] = evaluate_direction(model, tokenizer, test_set, src, config, folder)
            print(key, json.dumps(scores[key])[:200])
        if "perplexity" in config["metrics"]:
            for language, column in (("eng", "en"), (lang, "tgt")):
                texts = [row[column] for row in test_set["rows"]]
                rows = perplexity_eval(model, tokenizer, texts)
                key = f"{lang}-{source}-{language}"
                write_jsonl(folder / "perplexity" / f"{key}.jsonl", rows)
                scores[f"perplexity/{key}"] = summarize(rows)
    return scores


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    folder = Path(config["output"]) / Path(config["model"]).name
    folder.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(config["model"], config["dtype"])
    scores = evaluate_model(model, tokenizer, config, folder)
    (folder / "scores.json").write_text(json.dumps(scores, indent=2))
    manifest = {
        "config": config,
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
