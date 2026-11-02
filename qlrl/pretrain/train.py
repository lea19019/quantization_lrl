'''
Training the from-scratch model: one pass over the token files.

    run = Pretraining(config)     # config: configs/pretrain/llama_576m.yaml
    if run.run():                 # True once the pass is complete; False when the job's hours end
        run.finish()

A Hugging Face LlamaForCausalLM (so the quantizers and the evaluation load it like Llama 3.1),
AdamW, warmup-stable-decay. Under torchrun each GPU trains a copy on its own windows and
DistributedDataParallel averages the gradients. A job saves and stops after train.hours; the
next job with the same config resumes. Writes, under <checkpoints>/<name>/:
- step-<n>/: bf16 weights every save_every steps, the learning curve;
- resume.pt: fp32 weights, AdamW state and step (qlrl/checkpoint.py);
- state-<n>.pt: copies of resume.pt at the branch_at fractions of the run, for later cooldowns;
and <results>/<name>/history.jsonl (learning rate, train loss, validation loss per language),
and the final model, tokenizer and manifest.json in <output>/<name>/.
'''
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from qlrl.checkpoint import load as checkpoint_load
from qlrl.checkpoint import save as checkpoint_save
from qlrl.pretrain.data import TokenData
from qlrl.pretrain.tokenizer import git_sha


def learning_rate(step: int, steps: int, settings: dict) -> float:
    """Return the fraction of the peak learning rate at this step: linear warmup, constant,
    then 1 - sqrt((t - t_decay) / (T - t_decay)) over the last cooldown_fraction (Hägele 2024)."""
    decay_start = steps - round(settings["cooldown_fraction"] * steps)
    if step < settings["warmup_steps"]:
        return (step + 1) / settings["warmup_steps"]
    if step < decay_start:
        return 1.0
    return 1.0 - math.sqrt((step - decay_start) / max(1, steps - decay_start))


def make_optimizer(model, settings: dict) -> torch.optim.AdamW:
    """Return AdamW with weight decay on the linear layers only: not on the token embeddings
    or the RMSNorm gains (OLMo 2)."""
    decayed, other = [], []
    for name, parameter in model.named_parameters():
        if parameter.dim() == 2 and "embed_tokens" not in name:
            decayed.append(parameter)
        else:
            other.append(parameter)
    groups = [{"params": decayed, "weight_decay": settings["weight_decay"]},
              {"params": other, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=settings["lr"], betas=tuple(settings["betas"]),
                             eps=settings["eps"], fused=torch.cuda.is_available())


def summed_loss(model, windows: torch.Tensor, autocast: bool) -> torch.Tensor:
    """Return the summed next-token cross-entropy of windows [batch, n]: token i predicts
    token i + 1, so a window gives n - 1 predictions. bf16 matmuls, the loss in fp32."""
    with torch.autocast(windows.device.type, dtype=torch.bfloat16, enabled=autocast):
        logits = model(input_ids=windows[:, :-1]).logits  # [batch, n - 1, vocab]
    return F.cross_entropy(logits.float().flatten(0, 1), windows[:, 1:].flatten(),
                           reduction="sum")


class Pretraining:
    def __init__(self, config: dict):
        self.config, self.settings = config, config["train"]
        self.began = time.time()
        if "WORLD_SIZE" in os.environ:  # started by torchrun: one process per GPU
            dist.init_process_group("nccl")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            self.rank, self.world = dist.get_rank(), dist.get_world_size()
            self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        else:
            self.rank, self.world = 0, 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config["seed"])

        self.data = TokenData(config["tokens"], self.settings["seq_len"], config["seed"])
        self.per_step = self.settings["batch_tokens"] // self.settings["seq_len"]  # windows
        self.accumulation = self.per_step // (self.settings["micro_batch"] * self.world)
        assert self.accumulation * self.settings["micro_batch"] * self.world == self.per_step
        self.steps = self.data.windows // self.per_step  # one pass; the leftover is dropped
        self.branch_steps = {round(f * self.steps) for f in self.settings["branch_at"]}

        self.raw = LlamaForCausalLM(LlamaConfig(**config["model"])).to(self.device)
        self.model = torch.compile(self.raw) if self.settings["compile"] else self.raw
        if self.world > 1:
            self.model = DistributedDataParallel(self.model, device_ids=[self.device.index])
        self.optimizer = make_optimizer(self.raw, self.settings)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: learning_rate(step, self.steps, self.settings))

        self.folder = Path(config["checkpoints"]) / config["name"]
        self.results = Path(config["results"]) / config["name"]
        self.history = self.results / "history.jsonl"

    def train_step(self, step: int) -> float:
        """Run one optimizer step over per_step windows; return this GPU's mean token loss."""
        micro, total = self.settings["micro_batch"], 0.0
        for m in range(self.accumulation):
            first = step * self.per_step + (m * self.world + self.rank) * micro
            windows = self.data.batch(range(first, first + micro)).to(self.device)
            last = m == self.accumulation - 1  # GPUs average gradients on the last one only
            with nullcontext() if last or self.world == 1 else self.model.no_sync():
                loss = summed_loss(self.model, windows, self.settings["autocast"])
                loss = loss / (micro * self.settings["seq_len"])
                (loss / self.accumulation).backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(self.raw.parameters(), self.settings["clip"])
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return total / self.accumulation

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Return each language's validation loss (nats per token, over at most valid_windows
        windows) and their plain mean."""
        self.raw.eval()
        size, losses = self.data.size, {}
        for lang, ids in sorted(self.data.valid.items()):
            ids = torch.from_numpy(ids[: self.settings["valid_windows"] * size].astype("int64"))
            total, count = 0.0, 0
            for window in ids.split(size):
                if len(window) > 1:
                    window = window[None].to(self.device)
                    total += summed_loss(self.raw, window, self.settings["autocast"]).item()
                    count += window.shape[1] - 1
            losses[lang] = total / count
        self.raw.train()
        return losses | {"mean": sum(losses.values()) / len(losses)}

    def record(self, entry: dict) -> None:
        """Print one entry and append it to history.jsonl (GPU 0 only)."""
        if self.rank == 0:
            print(json.dumps(entry), flush=True)
            with self.history.open("a") as file:
                file.write(json.dumps(entry) + "\n")

    def start(self) -> int:
        """Return the step to run from: resume.pt's step, with history.jsonl cut back to it;
        or 0, with history.jsonl started by the untrained model's validation loss."""
        self.folder.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        resumed = checkpoint_load(self.folder / "resume.pt", self.raw, self.optimizer,
                                  self.scheduler)
        step = resumed["step"] if resumed else 0
        if self.rank == 0 and step == 0:
            self.history.write_text("")
            self.record({"step": 0, "valid_loss": self.evaluate()})
        elif self.rank == 0:
            lines = self.history.read_text().splitlines(keepends=True)
            kept = [line for line in lines if json.loads(line)["step"] <= step]
            self.history.write_text("".join(kept))
        return step

    def out_of_time(self) -> bool:
        """Return True on every GPU once any of them has used up the job's hours."""
        late = time.time() - self.began > self.settings["hours"] * 3600
        late = torch.tensor(float(late), device=self.device)
        if self.world > 1:
            dist.all_reduce(late, op=dist.ReduceOp.MAX)
        return bool(late.item())

    def save(self, step: int, stopping: bool) -> None:
        """Save what is due at this step: bf16 weights and resume.pt every save_every steps,
        resume.pt before the job stops, and a copy of it at the branch steps."""
        snapshot = step % self.settings["save_every"] == 0
        if not (snapshot or stopping or step in self.branch_steps):
            return
        if self.rank == 0:
            if snapshot:
                self.save_weights(self.folder / f"step-{step}")
            checkpoint_save(self.folder / "resume.pt", self.raw, self.optimizer,
                            self.scheduler, step, step)
            if step in self.branch_steps:
                shutil.copy(self.folder / "resume.pt", self.folder / f"state-{step}.pt")
        if self.world > 1:
            dist.barrier()  # the other GPUs wait for the files

    def save_weights(self, folder: Path) -> None:
        """Write the model in bf16 as a plain Hugging Face checkpoint, with the tokenizer."""
        state = {name: value.to(torch.bfloat16) for name, value in self.raw.state_dict().items()}
        self.raw.save_pretrained(folder, state_dict=state)
        AutoTokenizer.from_pretrained(self.config["tokenizer"]).save_pretrained(folder)

    def run(self) -> bool:
        """Train from the resumed step; return True when the pass is complete."""
        settings, step = self.settings, self.start()
        stop = min(self.steps, settings["max_steps"] or self.steps)
        losses, clock = [], time.time()
        while step < stop:
            losses.append(self.train_step(step))
            step += 1
            if step % settings["log_every"] == 0 or step == self.steps:
                loss = torch.tensor(sum(losses) / len(losses), device=self.device)
                if self.world > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                speed = len(losses) * settings["batch_tokens"] / (time.time() - clock)
                self.record({"step": step, "lr": self.scheduler.get_last_lr()[0],
                             "train_loss": loss.item(), "tokens_per_second": speed})
                losses, clock = [], time.time()
            if self.rank == 0 and (step % settings["eval_every"] == 0 or step == self.steps):
                self.record({"step": step, "valid_loss": self.evaluate()})
            if step == self.steps:
                return True
            stopping = step == stop or self.out_of_time()
            self.save(step, stopping)
            if stopping:
                return False
        return step == self.steps

    def finish(self) -> None:
        """Write the final model, its tokenizer and manifest.json (GPU 0)."""
        if self.rank == 0:
            folder = Path(self.config["output"]) / self.config["name"]
            self.save_weights(folder)
            manifest = {"config": self.config, "steps": self.steps, "world": self.world,
                        "windows": self.data.windows, "train_tokens": self.data.tokens(),
                        "git_sha": git_sha(), "torch": torch.__version__,
                        "transformers": transformers.__version__}
            for path in (folder, self.results):
                (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"saved {folder}", flush=True)
