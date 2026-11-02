'''
Pretraining of the from-scratch model (docs/project/pretraining_model.md):
    torchrun --standalone --nproc_per_node=<gpus> -m qlrl.pretrain configs/pretrain/llama_576m.yaml
    python -m qlrl.pretrain configs/pretrain/llama_test.yaml           tiny model, CPU or 1 GPU

A Llama-architecture decoder (Hugging Face LlamaForCausalLM, so the quantizers and the
evaluation load it like Llama 3.1) trained on one pass over the token files
(qlrl/data/tokens.py), AdamW, warmup-stable-decay. Each GPU runs a copy of the model on its own
windows; DistributedDataParallel averages the gradients. A job stops itself before the time
limit and the next job with the same config resumes. Writes:
- <checkpoints>/<name>/step-<n>/: bf16 weights every save_every steps, the learning curve;
- <checkpoints>/<name>/resume.pt: fp32 weights, AdamW state and step (qlrl/checkpoint.py);
- <checkpoints>/<name>/state-<n>.pt: copies of resume.pt at branch_at, for later cooldowns;
- <results>/<name>/history.jsonl: learning rate, train loss, validation loss per language;
- <output>/<name>/: the final model, its tokenizer and manifest.json.
'''
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
# Written by Claude Code at Adrian's request, 2026-10-30.
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
import yaml
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from qlrl.checkpoint import load as checkpoint_load
from qlrl.checkpoint import save as checkpoint_save
from qlrl.data.tokens import TokenData
from qlrl.recipe import learning_rate, make_optimizer, setup, summed_loss


class Pretraining:
    def __init__(self, config: dict):
        self.config, self.settings = config, config["train"]
        self.rank, self.world, self.device = setup()
        torch.manual_seed(config["seed"])
        self.data = TokenData(config["tokens"], self.settings["seq_len"], config["seed"])
        self.per_step = self.settings["batch_tokens"] // self.settings["seq_len"]  # windows
        self.accumulation = self.per_step // (self.settings["micro_batch"] * self.world)
        assert self.accumulation * self.settings["micro_batch"] * self.world == self.per_step
        self.steps = self.data.windows // self.per_step  # one pass; the leftover is dropped
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
        tokens = micro * self.settings["seq_len"]
        for m in range(self.accumulation):
            first = step * self.per_step + (m * self.world + self.rank) * micro
            windows = self.data.batch(list(range(first, first + micro))).to(self.device)
            if self.world > 1:  # average the gradients across GPUs on the last micro-batch only
                self.model.require_backward_grad_sync = m == self.accumulation - 1
            loss = summed_loss(self.model, windows, self.settings["autocast"]) / tokens
            (loss / self.accumulation).backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(self.raw.parameters(), self.settings["clip"])
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return total / self.accumulation

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Return each language's validation loss (mean nats per token over at most
        valid_windows windows) and their plain mean; each GPU scores every world-th language."""
        self.raw.eval()
        size, losses = self.settings["seq_len"] + 1, {}
        for lang in sorted(self.data.valid)[self.rank::self.world]:
            ids = self.data.valid[lang][: self.settings["valid_windows"] * size]
            total, count = 0.0, 0
            for start in range(0, len(ids) - 1, size):
                window = torch.from_numpy(ids[start:start + size].astype("int64"))[None]
                total += summed_loss(self.raw, window.to(self.device),
                                     self.settings["autocast"]).item()
                count += window.shape[1] - 1
            losses[lang] = total / count
        parts = [losses]
        if self.world > 1:
            parts = [None] * self.world
            dist.all_gather_object(parts, losses)
        merged = {lang: loss for part in parts for lang, loss in part.items()}
        self.raw.train()
        return dict(sorted(merged.items())) | {"mean": sum(merged.values()) / len(merged)}

    def record(self, entry: dict) -> None:
        """Print one logged step and append it to history.jsonl (rank 0 only)."""
        if self.rank == 0:
            print(json.dumps(entry), flush=True)
            with self.history.open("a") as file:
                file.write(json.dumps(entry) + "\n")

    def save_weights(self, folder: Path) -> None:
        """Write the model in bf16 as a plain Hugging Face checkpoint, with the tokenizer."""
        state = {name: value.to(torch.bfloat16) for name, value in self.raw.state_dict().items()}
        self.raw.save_pretrained(folder, state_dict=state)
        AutoTokenizer.from_pretrained(self.config["tokenizer"]).save_pretrained(folder)

    def save(self, step: int, late: bool) -> None:
        """At step (steps done): bf16 weights, resume state and branch states as configured;
        late: the job is about to stop, so the resume state is written now."""
        settings, last = self.settings, step == self.steps
        if self.rank == 0:
            if step % settings["save_every"] == 0 and not last:
                self.save_weights(self.folder / f"step-{step}")
            if step % settings["resume_every"] == 0 or step in self.branches() or late:
                checkpoint_save(self.folder / "resume.pt", self.raw, self.optimizer,
                                self.scheduler, step, step)
            if step in self.branches():
                shutil.copy(self.folder / "resume.pt", self.folder / f"state-{step}.pt")
        if self.world > 1:
            dist.barrier()

    def branches(self) -> set[int]:
        """Return the steps whose full state is kept for cooldown branches (Hägele 2024)."""
        return {round(fraction * self.steps) for fraction in self.settings["branch_at"]}

    def start(self) -> int:
        """Resume from resume.pt if there is one (history after it dropped); else start
        history.jsonl with the untrained model's validation loss. Return the step to run."""
        self.folder.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        resumed = checkpoint_load(self.folder / "resume.pt", self.raw, self.optimizer,
                                  self.scheduler)
        if self.world > 1:
            dist.barrier()  # nobody writes history before everyone has read the checkpoint
        if resumed is None:
            if self.rank == 0:
                self.history.write_text("")
            self.record({"step": 0, "valid_loss": self.evaluate()})
            return 0
        if self.rank == 0:
            lines = self.history.read_text().splitlines(keepends=True)
            self.history.write_text("".join(line for line in lines
                                            if json.loads(line)["step"] <= resumed["step"]))
        return resumed["step"]

    def out_of_time(self, began: float) -> bool:
        """Return True on every GPU once any GPU has used up the job's hours."""
        late = torch.tensor(float(time.time() - began > self.settings["hours"] * 3600))
        if self.world > 1:
            late = late.to(self.device)
            dist.all_reduce(late, op=dist.ReduceOp.MAX)
        return bool(late.item())

    def run(self) -> bool:
        """Train from the resumed step; return True when the pass is complete."""
        began, step = time.time(), self.start()
        stop = min(self.steps, self.settings.get("max_steps") or self.steps)
        losses, clock = [], time.time()
        while step < stop:
            losses.append(self.train_step(step))
            step += 1
            entry = {"step": step}
            if step % self.settings["log_every"] == 0 or step == self.steps:
                loss = torch.tensor(sum(losses) / len(losses), device=self.device)
                if self.world > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                seconds, clock = time.time() - clock, time.time()
                entry |= {"lr": self.scheduler.get_last_lr()[0], "train_loss": loss.item(),
                          "tokens_per_second": len(losses) * self.settings["batch_tokens"]
                          / seconds}
                losses = []
            if step % self.settings["eval_every"] == 0 or step == self.steps:
                entry["valid_loss"] = self.evaluate()
            if len(entry) > 1:
                self.record(entry)
            late = step < self.steps and (step == stop or self.out_of_time(began))
            self.save(step, late)
            if late:
                return False
        return step == self.steps

    def finish(self) -> None:
        """Write the final model, tokenizer and manifest.json (rank 0)."""
        if self.rank != 0:
            return
        folder = Path(self.config["output"]) / self.config["name"]
        self.save_weights(folder)
        manifest = {"config": self.config, "steps": self.steps, "world": self.world,
                    "windows": self.data.windows, "train_tokens": self.data.tokens(),
                    "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], text=True,
                                              capture_output=True, check=False).stdout.strip(),
                    "torch": torch.__version__, "transformers": transformers.__version__}
        for path in (folder, self.results):
            (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"saved {folder}", flush=True)


def main() -> None:
    """Train until the pass is complete or the job's hours are used; a rerun resumes."""
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    if (Path(config["output"]) / config["name"] / "manifest.json").exists():
        print(f"{config['name']} is already trained", flush=True)
        return
    run = Pretraining(config)
    if run.run():
        run.finish()
    if run.world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
