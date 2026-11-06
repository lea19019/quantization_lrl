'''
Entry point for Front 1: python -m qlrl.experiments.run configs/experiments/front1.yaml

Loads the full model and one quantized model (the Slurm array index picks it from the YAML). Per
sentence of every language, English -> X, one full and one quantized pass give the logit lens,
noise vs flatness and the activation change; one more quantized pass per site gives the patching.
Writes <output>/<quantized model>/: one .jsonl per language with one row per sentence,
recovered.json and head_change.json per language, manifest.json.
'''
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from qlrl.eval import get_prompts, load_data, write_jsonl
from qlrl.evaluation import load_model
from qlrl.experiments.capture import activation_change, capture, head_change, reference_ids
from qlrl.experiments.lens import lens_curve
from qlrl.experiments.noise import noise_flatness
from qlrl.experiments.patching import patch_sites, recovered


def gold_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the log p of each target, [tokens], from float logits [tokens, vocab]."""
    return torch.log_softmax(logits, dim=-1)[torch.arange(targets.shape[0]), targets]


def run_sentence(model, q_model, tokenizer, prompt: str, reference: str) -> tuple[dict, dict]:
    """Return (the sentence's row of results, its head_change)."""
    ids, positions, targets = reference_ids(tokenizer, prompt, reference, model.device)
    z_full, full = capture(model, ids, positions)
    z_quant, quant = capture(q_model, ids, positions)
    gold_full, gold_quant = gold_logprobs(z_full, targets), gold_logprobs(z_quant, targets)
    row = {"full": gold_full.sum().item(), "quant": gold_quant.sum().item(),
           "tokens": len(targets),
           "patched": patch_sites(q_model, tokenizer, prompt, reference, full),
           "lens": {"full": lens_curve(model, full, positions, targets),
                    "quant": lens_curve(model, quant, positions, targets)},
           "noise": noise_flatness(z_full, z_quant) | {"gold full": gold_full.tolist(),
                                                       "gold quant": gold_quant.tolist()},
           "change": activation_change(full, quant, positions)}
    return row, head_change(full, quant, positions, model.config.num_attention_heads)


def run_test_set(model, q_model, tokenizer, test_set: dict) -> tuple[list[dict], dict]:
    """Return one row per sentence and, per layer and head, ||quant - full|| / ||full||."""
    examples = [(row["en"], row["tgt"]) for row in test_set["examples"]]
    sources = [row["en"] for row in test_set["rows"]]
    references = [row["tgt"] for row in test_set["rows"]]
    prompts = get_prompts(examples, sources, "eng", test_set["lang"])
    rows, distance, norm = [], {}, {}
    for source, reference, prompt in zip(sources, references, prompts):
        row, heads = run_sentence(model, q_model, tokenizer, prompt, reference)
        rows.append({"src": source, "ref": reference} | row)
        for name, (squared_distance, squared_norm) in heads.items():
            distance[name] = distance.get(name, 0) + squared_distance
            norm[name] = norm.get(name, 0) + squared_norm
    return rows, {name: (distance[name] / norm[name]).sqrt().tolist() for name in distance}


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    config["quantized"] = config["quantized"][int(sys.argv[2]) if len(sys.argv) > 2 else 0]
    folder = Path(config["output"]) / Path(config["quantized"]).name
    folder.mkdir(parents=True, exist_ok=True)
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=False).stdout.strip()
    manifest = {"config": config, "git_sha": git_sha}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))

    model, tokenizer = load_model(config["model"], config["dtype"])
    q_model, _ = load_model(config["quantized"], config["dtype"])
    shares, head_changes = {}, {}
    for test_set in load_data(config):
        lang, start = test_set["lang"], time.time()
        rows, head_changes[lang] = run_test_set(model, q_model, tokenizer, test_set)
        write_jsonl(folder / f"{lang}-{test_set['source']}-eng2{lang}.jsonl", rows)
        shares[lang] = recovered(rows)
        (folder / "recovered.json").write_text(json.dumps(shares, indent=2))
        (folder / "head_change.json").write_text(json.dumps(head_changes))
        print(lang, len(rows), "sentences", round(time.time() - start, 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
