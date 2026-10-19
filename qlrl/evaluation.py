'''
Entry point for evaluation: python -m qlrl.evaluation configs/eval.yaml

Loads one model (full precision or quantized) and, for the one test set of every language in
the YAML (`test_sets`) and both directions (English -> X and X -> English), measures:

• Translation: three dev pairs of the same test set, then the sentence, as
      <eng>Hello.</eng> <yor>Báwo ni.</yor>  ...  <eng>{sentence}</eng> <yor>
  Greedy; stop at </yor> or at the length cap (see generation_caps). Scored with chrF++ and
  BLEU, AfriCOMET on the African core, and the off-target rate (see offTargert_eval).
• Reference log-probability: the same prompt followed by the human reference; the summed
  log p of the reference tokens (forced decoding). No generation, so no decoding noise.
• Perplexity: each bare sentence, no prompt. Compare a language with itself only.

Per-sentence numbers go to <output>/<model>/, so damage (full minus quantized log p)
and every loss is a difference between two runs. Settings: docs/project/evaluation.md.
The pieces live in qlrl/eval/: translation, logprob, scores, report.
AfriCOMET runs in its own environment, because unbabel-comet pins an older transformers:
    uv venv .venv-comet --python 3.11
    uv pip install --python .venv-comet unbabel-comet==2.2.7 "setuptools<70"
'''
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from qlrl.eval import (
    generation_caps,
    get_prompts,
    load_data,
    perplexity_eval,
    referenceLogProb_eval,
    rounded,
    score_translations,
    summarize,
    token_ratio,
    translate,
    write_jsonl,
    write_manifest,
)


def load_model(path: str, dtype: str):
    """Return (model, tokenizer) on the GPU if there is one, ready for batched generation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype)).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token  # GPT-2 has none
    return model.to(device), tokenizer


def evaluate_direction(model, tokenizer, test_set: dict, src: str, config: dict, folder):
    """Translate one test set in one direction, score it, save per-sentence rows.

    Returns (scores, details) where details hold the cap ratio and the metric signatures."""
    lang = test_set["lang"]
    tgt = lang if src == "eng" else "eng"
    side = {"eng": "en", lang: "tgt"}  # the column holding each language
    examples = [(row[side[src]], row[side[tgt]]) for row in test_set["examples"]]
    sources = [row[side[src]] for row in test_set["rows"]]
    references = [row[side[tgt]] for row in test_set["rows"]]
    prompts = get_prompts(examples, sources, src, tgt)
    rows = [{"src": s, "ref": r} for s, r in zip(sources, references)]
    scores, reference, details = {}, {}, {"signatures": {}}

    if "reference_logprob" in config["metrics"]:
        logprobs = referenceLogProb_eval(model, tokenizer, prompts, references)
        rows = [row | logprob for row, logprob in zip(rows, logprobs)]
        summary = summarize(logprobs)
        reference["reference NLL per word"] = summary["NLL per word"]
        reference["reference NLL per byte"] = summary["NLL per byte"]
        reference["reference perplexity"] = summary["perplexity"]
    if {"chrf", "bleu", "africomet", "off_target"} & set(config["metrics"]):
        dev_pairs = [(row[side[src]], row[side[tgt]]) for row in test_set["dev"]]
        details["cap ratio"] = token_ratio(tokenizer, dev_pairs)
        caps = generation_caps(tokenizer, sources, details["cap ratio"])
        hypotheses = translate(model, tokenizer, prompts, caps, tgt, config["batch_size"])
        rows = [row | {"hyp": h} for row, h in zip(rows, hypotheses)]
        scores, details["signatures"] = score_translations(rows, lang, tgt, config)
    scores |= reference  # translation scores first, then the reference log-probability

    name = f"{lang}-{test_set['source']}-{src}2{tgt}"
    write_jsonl(folder / "translation" / f"{name}.jsonl", rows)
    return scores, details


def perplexity_scores(model, tokenizer, test_set: dict, folder: Path) -> dict:
    """Return perplexity of the bare English and target sentences; save per-sentence rows."""
    lang, scores = test_set["lang"], {}
    for language, column in (("eng", "en"), (lang, "tgt")):
        texts = [row[column] for row in test_set["rows"]]
        rows = perplexity_eval(model, tokenizer, texts)
        name = f"{lang}-{test_set['source']}-{language}"
        write_jsonl(folder / "perplexity" / f"{name}.jsonl", rows)
        scores[language] = summarize(rows)
    return scores


def evaluate_model(model, tokenizer, config: dict, folder: Path) -> tuple[dict, dict]:
    """Run every measurement on every language; return (scores, details for the manifest)."""
    scores, details = {}, {"languages": {}, "signatures": {}}
    for test_set in load_data(config):
        lang = test_set["lang"]
        entry = {"test set": test_set["source"], "sentences": len(test_set["rows"])}
        about = entry | {"dev examples in prompt": len(test_set["examples"]),
                         "dev pairs for cap ratio": len(test_set["dev"]), "cap ratio": {}}
        for src in ("eng", lang):
            tgt = lang if src == "eng" else "eng"
            direction = f"{src}→{tgt}"
            entry[direction], found = evaluate_direction(model, tokenizer, test_set, src,
                                                         config, folder)
            about["cap ratio"][direction] = found.get("cap ratio")
            details["signatures"] |= found["signatures"]
            print(lang, direction, json.dumps(rounded(entry[direction]), ensure_ascii=False))
        if "perplexity" in config["metrics"]:
            entry["perplexity"] = perplexity_scores(model, tokenizer, test_set, folder)
        scores[lang.upper()] = entry
        details["languages"][lang.upper()] = about
    return scores, details


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    folder = Path(config["output"]) / Path(config["model"]).name
    folder.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(config["model"], config["dtype"])
    scores, details = evaluate_model(model, tokenizer, config, folder)
    scores_text = json.dumps(rounded(scores), indent=2, ensure_ascii=False)
    (folder / "scores.json").write_text(scores_text)
    write_manifest(folder, config, details)


if __name__ == "__main__":
    main()
