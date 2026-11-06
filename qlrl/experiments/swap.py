'''
Mixed-precision swap: python -m qlrl.experiments.swap configs/experiments/swap.yaml

Question. Are the late layers damaged themselves, or do they amplify errors from earlier layers?
Method. Take the full model and use the quantized weights for one range of layers only, then
measure the damage on the reference tokens and the activation change at every layer.
Limit. The hybrids are artificial: GPTQ quantized each layer on inputs from quantized earlier
layers.

For each sentence (tokenized as in qlrl/eval/logprob.py):
    1. Full run and quantized run, as in qlrl/experiments/run.py.
    2. For each hybrid (first layer, last layer, both included), quantized layers in place:
       log p_hybrid = summed log p of the reference tokens.
       change = ||hybrid - full|| and ||full|| at every site.

Writes <output>/<quantized model>/: one .jsonl per language with one row per sentence,
manifest.json.
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
from qlrl.experiments.capture import activation_change, capture, reference_ids
from qlrl.experiments.run import gold_logprobs


def capture_hybrid(model, q_model, ids: torch.Tensor, positions: torch.Tensor, first: int,
                   last: int):
    """Return capture() of `model` with layers first..last taken from `q_model`, then restore."""
    # Modules are shared, not copied: no extra GPU memory. The norm, the head and the embeddings
    # are not quantized, so swapping the layers is all that separates the two models.
    layers = model.model.layers
    originals = {index: layers[index] for index in range(first, last + 1)}
    for index in originals:
        layers[index] = q_model.model.layers[index]
    result = capture(model, ids, positions)
    for index, layer in originals.items():
        layers[index] = layer
    return result


def run_sentence(model, q_model, tokenizer, prompt: str, reference: str, hybrids: dict) -> dict:
    """Return the sentence's row: log p and activation change of the quantized model and hybrids."""
    ids, positions, targets = reference_ids(tokenizer, prompt, reference, model.device)
    z_full, full = capture(model, ids, positions)
    z_quant, quant = capture(q_model, ids, positions)
    row = {"full": gold_logprobs(z_full, targets).sum().item(),
           "quant": gold_logprobs(z_quant, targets).sum().item(), "tokens": len(targets),
           "change": activation_change(full, quant, positions), "hybrids": {}}
    for name, (first, last) in hybrids.items():
        z_hybrid, hybrid = capture_hybrid(model, q_model, ids, positions, first, last)
        row["hybrids"][name] = {"logprob": gold_logprobs(z_hybrid, targets).sum().item(),
                                "change": activation_change(full, hybrid, positions)}
    return row


def run_test_set(model, q_model, tokenizer, test_set: dict, hybrids: dict) -> list[dict]:
    """Return one row per sentence of a test set, English -> the language."""
    examples = [(row["en"], row["tgt"]) for row in test_set["examples"]]
    sources = [row["en"] for row in test_set["rows"]]
    references = [row["tgt"] for row in test_set["rows"]]
    prompts = get_prompts(examples, sources, "eng", test_set["lang"])
    return [{"src": source, "ref": reference}
            | run_sentence(model, q_model, tokenizer, prompt, reference, hybrids)
            for source, reference, prompt in zip(sources, references, prompts)]


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
    for test_set in load_data(config):
        lang, start = test_set["lang"], time.time()
        rows = run_test_set(model, q_model, tokenizer, test_set, config["hybrids"])
        write_jsonl(folder / f"{lang}-{test_set['source']}-eng2{lang}.jsonl", rows)
        print(lang, len(rows), "sentences", round(time.time() - start, 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
