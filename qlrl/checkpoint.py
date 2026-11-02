'''
Training checkpoints: what a fine-tune needs to continue after a crash or a time limit.

    save(path, model, optimizer, scheduler, step=500, batch=4000)
    resumed = load(path, model, optimizer, scheduler)    # {"step", "batch"}, or None

Only the weights that train are saved (LoRA's A and B, or every weight in full mode), with
the optimizer's m and v, the learning-rate schedule, the position in the data and the random
states. The frozen base is loaded from the original model again on resume.
'''
from pathlib import Path

import torch


def save(path: Path, model, optimizer, scheduler, step: int, batch: int) -> None:
    """Write the checkpoint; the previous one is replaced only once the new one is complete,
    so a job killed while writing still leaves a usable checkpoint."""
    state = {
        "trainable": {name: parameter.detach().cpu()
                      for name, parameter in model.named_parameters() if parameter.requires_grad},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "batch": batch,  # micro-batches already used: the loop skips these on resume
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    print(f"checkpoint at step {step}: {path}", flush=True)


def load(path: Path, model, optimizer, scheduler) -> dict | None:
    """Restore the trained weights, optimizer, schedule and random states from the checkpoint;
    return its step and batch, or None when there is no checkpoint (a fresh run)."""
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state["trainable"].items():
            parameters[name].copy_(value)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"]:
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    print(f"resumed from step {state['step']}: {path}", flush=True)
    return {"step": state["step"], "batch": state["batch"]}
