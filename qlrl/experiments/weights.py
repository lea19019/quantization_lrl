'''
Reading the rounding error in the weights:
    python -m qlrl.experiments.weights configs/experiments/weights.yaml

Question 1. Is the error E = W_quant - W_full structured, or does it look like noise?
Method. Singular values of E in each down_proj (the matrix that writes into the residual), against
two shuffles of E. Shuffling every entry keeps only their overall spread. Shuffling within each row
and quantization group keeps the rounding step of every group, because the quantizer sets one range
per row and group, and removes any pattern shared across rows (qlrl/quant/gptq.py).
Question 2. Which tokens does the error favor?
Method. As in the logit lens (Wendler 2024), read a residual direction u through the final norm and
the head, which are not quantized: logits = lm_head (norm_weight * u). The norm's division by the
vector's RMS only changes the scale, so it is left out. Sign of u is arbitrary: both ends are read.
A random direction and the shuffled error's top direction show what no token preference looks like.

Writes <output>/<quantized model>/: structure.json with one entry per layer, manifest.json.
'''
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from transformers import AutoTokenizer


def load_tensors(folder: str, names: list[str]) -> dict[str, torch.Tensor]:
    """Return { name: float tensor } from a model folder's model.safetensors."""
    with safe_open(Path(folder) / "model.safetensors", "pt") as file:
        return {name: file.get_tensor(name).float() for name in names}


def shuffle_all(error: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Return `error` with all its entries in random positions."""
    order = torch.randperm(error.numel(), generator=generator)
    return error.flatten()[order].reshape(error.shape)


def shuffle_in_groups(error: torch.Tensor, group_size: int,
                      generator: torch.Generator) -> torch.Tensor:
    """Return `error` with each row's entries shuffled inside each group of columns."""
    blocks = []
    for start in range(0, error.shape[1], group_size):
        block = error[:, start:start + group_size]
        order = torch.rand(block.shape, generator=generator).argsort(dim=1)
        blocks.append(block.gather(1, order))
    return torch.cat(blocks, dim=1)


def spectrum(sigma: torch.Tensor) -> dict:
    """Return the top singular values, the stable rank and the top 10's share of the energy."""
    energy = sigma ** 2
    return {"sigma": sigma[:10].tolist(), "stable rank": (energy.sum() / energy[0]).item(),
            "top 10 energy": (energy[:10].sum() / energy.sum()).item()}


def read_direction(u: torch.Tensor, norm: torch.Tensor, head: torch.Tensor, tokenizer,
                   tokens: int) -> dict:
    """Return the largest |z-score| of the logits along `u` and the tokens at each end."""
    logits = head @ (norm * u)  # [vocab]
    z = (logits - logits.mean()) / logits.std()
    ends = {"positive": z.topk(tokens).indices, "negative": (-z).topk(tokens).indices}
    return {"max z": z.abs().max().item(),
            **{end: [tokenizer.decode([i]) for i in ids.tolist()] for end, ids in ends.items()}}


def read_layer(w_full, w_quant, norm, head, tokenizer, config: dict, generator) -> dict:
    """Return the size, the spectra and the token readouts of one down_proj's error."""
    error = w_quant - w_full
    u, sigma, _ = torch.linalg.svd(error, full_matrices=False)
    u_full, sigma_full, _ = torch.linalg.svd(w_full, full_matrices=False)
    u_shuffled, sigma_groups, _ = torch.linalg.svd(
        shuffle_in_groups(error, config["group_size"], generator), full_matrices=False)
    k, tokens = config["directions"], config["tokens"]
    random_u = torch.randn(u.shape[0], generator=generator)
    return {
        "relative size": (error.norm() / w_full.norm()).item(),
        "error": spectrum(sigma), "weights": spectrum(sigma_full),
        "shuffled in groups": spectrum(sigma_groups),
        "shuffled all": spectrum(torch.linalg.svdvals(shuffle_all(error, generator))),
        # Mean squared cosine between the top k directions of the error and of the weights.
        "overlap with weights": ((u[:, :k].T @ u_full[:, :k]) ** 2).sum().item() / k,
        "directions": [read_direction(u[:, i], norm, head, tokenizer, tokens) for i in range(k)],
        "weight directions": [read_direction(u_full[:, i], norm, head, tokenizer, tokens)
                              for i in range(k)],
        "shuffled direction": read_direction(u_shuffled[:, 0], norm, head, tokenizer, tokens),
        "random direction": read_direction(random_u / random_u.norm(), norm, head, tokenizer,
                                           tokens),
    }


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    config["quantized"] = config["quantized"][int(sys.argv[2]) if len(sys.argv) > 2 else 0]
    folder = Path(config["output"]) / Path(config["quantized"]).name
    folder.mkdir(parents=True, exist_ok=True)
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=False).stdout.strip()
    (folder / "manifest.json").write_text(json.dumps({"config": config, "git_sha": git_sha},
                                                     indent=2))
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    shared = load_tensors(config["model"], ["model.norm.weight", "lm_head.weight"])
    generator = torch.Generator().manual_seed(config["seed"])
    first, last = config["layers"]
    layers = {}
    for index in range(first, last + 1):
        name = f"model.layers.{index}.mlp.down_proj.weight"
        w_full = load_tensors(config["model"], [name])[name]
        w_quant = load_tensors(config["quantized"], [name])[name]
        layers[f"layer{index}"] = read_layer(w_full, w_quant, shared["model.norm.weight"],
                                             shared["lm_head.weight"], tokenizer, config, generator)
        (folder / "structure.json").write_text(json.dumps(layers, indent=2, ensure_ascii=False))
        print(f"layer{index}", layers[f"layer{index}"]["relative size"], flush=True)


if __name__ == "__main__":
    main()
