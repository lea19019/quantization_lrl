'''
Fine-tuning notes (no code yet): python -m qlrl.finetune configs/finetune.yaml, mode full | lora.

1. Data: ParallelData train split, all languages. Never FLORES or the eval test sets.
   Validation: church dev.
2. Mix: sample language L with p ∝ n_L^α (α ≈ 0.3–0.7) or cap pairs per language.
3. Format: <eng>{en}</eng> <yor>{x}</yor>, both directions, several pairs per sequence.
   Loss on target tokens only (source labels = -100).
4. Model: Llama-3.1-8B, bf16.
   full: AdamW 8-bit + gradient checkpointing (1 GPU), or FSDP (2-4 GPUs). lr ~1e-5.
   lora: adapters on q, k, v, o, gate, up, down. lr ~1e-4.
5. Loop: forward, loss, backward, clip 1.0, step; log; validation loss; checkpoints.
6. Save: merge LoRA into W; models/<name>/ with tokenizer and manifest.json.
7. Then: evaluate, quantize (BOS calibration, 5 seeds), evaluate; check eng/fra forgetting.
'''


# Fine-tuning: python -m qlrl.finetune configs/finetune/lora.yaml (or full, or test on CPU).
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import transformers
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from qlrl.checkpoint import load as checkpoint_load
from qlrl.checkpoint import save as checkpoint_save
from qlrl.data.dataset import ParallelData
from qlrl.data.training import TrainingData

# ---- Model: load it, choose what trains (full or lora) ----

def load_model(config: dict):
    """Return the model on the GPU if there is one, set up for config["mode"], and its tokenizer.

    full: every weight trains. lora: peft freezes the base and adds the adapters.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, config["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token  # Llama and GPT-2 have none
    if config.get("tiny"):  # test: a small random model, real tokenizer and layer names
        model_config = AutoConfig.from_pretrained(config["model"], **config["tiny"])
        model = AutoModelForCausalLM.from_config(model_config, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(config["model"], dtype=dtype)
    model.to(device)  # type: ignore[arg-type]
    if config["train"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # frozen embeddings (LoRA) must still pass gradients
    if config["mode"] == "lora":
        lora = config["lora"]
        model = get_peft_model(model, LoraConfig(
            r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=lora["targets"], task_type="CAUSAL_LM",
        ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{config['mode']}: {trainable:,} of {total:,} parameters train "
          f"({100 * trainable / total:.2f}%)", flush=True)
    return model, tokenizer


# ---- Training loop ----

def make_optimizer(parameters: list, settings: dict) -> torch.optim.Optimizer:
    """Return AdamW, or 8-bit AdamW (Dettmers et al. 2022: m and v stored in 8 bits)."""
    if settings["optimizer"] == "adamw8bit":
        import bitsandbytes  # GPU venv only
        return bitsandbytes.optim.AdamW8bit(
            parameters, lr=settings["lr"], weight_decay=settings["weight_decay"])
    return torch.optim.AdamW(parameters, lr=settings["lr"], weight_decay=settings["weight_decay"])


@torch.no_grad()
def dev_loss(model, dev: TrainingData) -> dict[str, float]:
    """Return the dev loss of each language and their plain mean ("mean": every language
    counts the same, however many pairs it trains on).

    A language's loss is the mean over its graded tokens, as in training, with dropout off
    and nothing updated: how well the model predicts translations it never trained on.
    """
    model.eval()
    losses = {}
    for lang, row in dev.stats.items():
        if not row["sequences"]:
            continue
        total, tokens = 0.0, 0
        for batch in dev.loader(lang):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            graded = (batch["labels"][:, 1:] != -100).sum().item()  # the model shifts by one
            total += model(**batch).loss.item() * graded
            tokens += graded
        losses[lang] = total / tokens
    losses["mean"] = sum(losses.values()) / len(losses)
    model.train()
    return losses


def record(history: Path, entry: dict) -> None:
    """Print one logged step and append it to history.jsonl, the file the plots read."""
    print(json.dumps(entry), flush=True)
    with history.open("a") as file:
        file.write(json.dumps(entry) + "\n")


def start(model, optimizer, scheduler, dev: TrainingData, checkpoint: Path,
          history: Path) -> dict:
    """Resume from the checkpoint if there is one, dropping the history logged after it;
    otherwise start history.jsonl with the untrained model's dev loss. Return the step and
    micro-batch to continue from."""
    resumed = checkpoint_load(checkpoint, model, optimizer, scheduler)
    if resumed is None:
        history.write_text("")
        record(history, {"step": 0, "dev_loss": dev_loss(model, dev)})
        return {"step": 0, "batch": 0}
    lines = history.read_text().splitlines(keepends=True) if history.exists() else []
    history.write_text("".join(line for line in lines
                               if json.loads(line)["step"] <= resumed["step"]))
    return resumed


def train(model, loader, dev: TrainingData, config: dict, folder: Path, results: Path) -> dict:
    """Run one pass over the batches; return the steps taken and the last logged entry.

    Each optimizer step averages `accumulation` micro-batches of `batch_size` rows; the
    learning rate warms up linearly, then follows a cosine down to 0. The train loss every
    log_every steps and the dev loss at step 0 and every eval_every steps go to
    <results>/history.jsonl; a checkpoint every checkpoint_every steps goes to
    <folder>/checkpoint.pt, and a rerun of the same config resumes from it.
    """
    settings = config["train"]
    accumulation = settings["accumulation"]
    steps = math.ceil(len(loader) / accumulation)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(parameters, settings)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, round(settings["warmup_ratio"] * steps), steps)
    history, checkpoint = results / "history.jsonl", folder / "checkpoint.pt"
    resumed = start(model, optimizer, scheduler, dev, checkpoint, history)

    model.train()
    step, losses, last = resumed["step"], [], {}
    for i, batch in enumerate(loader):  # same seed, same order: skip what was already used
        if i < resumed["batch"]:
            continue
        batch = {key: value.to(model.device) for key, value in batch.items()}
        loss = model(**batch).loss
        (loss / accumulation).backward()
        losses.append(loss.item())
        if (i + 1) % accumulation and i + 1 < len(loader):
            continue
        torch.nn.utils.clip_grad_norm_(parameters, settings["clip"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1
        entry = {"step": step}
        if step % settings["log_every"] == 0 or step == steps:
            entry |= {"lr": scheduler.get_last_lr()[0], "train_loss": sum(losses) / len(losses)}
            losses = []
        if step % settings["eval_every"] == 0 or step == steps:
            entry["dev_loss"] = dev_loss(model, dev)
        if len(entry) > 1:
            record(history, entry)
            last = entry
        if step % settings["checkpoint_every"] == 0 and step < steps:
            checkpoint_save(checkpoint, model, optimizer, scheduler, step, i + 1)
    return {"steps": step, "last": last}


# ---- Save ----

def save(model, tokenizer, config: dict, stats: dict, folder: Path, results: Path) -> None:
    """Merge LoRA if used, then write the model, tokenizer and manifest.json to the folder,
    a plain checkpoint the quantizers and the evaluation load; the manifest also goes to the
    results folder, next to history.jsonl."""
    if config["mode"] == "lora":
        model = model.merge_and_unload()
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    manifest = {
        "config": config,
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=False).stdout.strip(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "stats": stats,
    }
    for path in (folder, results):
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved {folder}", flush=True)


def main() -> None:
    """Read the YAML, load the model, build the train and dev data from ParallelData, train,
    save. The model goes to <output>/<name>/, history.jsonl to <results>/<name>/."""
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    torch.manual_seed(config["seed"])
    folder = Path(config["output"]) / config["name"]
    results = Path(config["results"]) / config["name"]
    for path in (folder, results):
        path.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(config)
    source = ParallelData(config["data"])
    languages, seed = config["languages"], config["seed"]
    data = TrainingData(source.split("train", languages, caps=config["caps"], seed=seed),
                        tokenizer, config)
    dev_caps = {lang: config["dev"]["pairs"] for lang in languages}
    dev = TrainingData(source.split("dev", languages, sources=config["dev"]["sources"],
                                    caps=dev_caps, seed=seed), tokenizer, config)
    for lang in languages:
        print(lang, "train", data.stats[lang], "dev", dev.stats.get(lang), flush=True)
    train_stats = train(model, data.loader(), dev, config, folder, results)
    stats = {"languages": data.stats, "dev": dev.stats, **train_stats}
    save(model, tokenizer, config, stats, folder, results)
    (folder / "checkpoint.pt").unlink(missing_ok=True)  # finished: a rerun starts fresh


if __name__ == "__main__":
    main()
