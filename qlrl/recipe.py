'''
Training recipe of the from-scratch model (docs/project/pretraining_model.md §3): the
learning-rate schedule, the optimizer, the loss, and the GPUs. Used by qlrl/pretrain.py.
'''
# Written by Claude Code at Adrian's request, 2026-10-30.
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def learning_rate(step: int, steps: int, settings: dict) -> float:
    """Return the fraction of the peak learning rate at this step (warmup-stable-decay).

    Linear warmup, constant, then the (1 - sqrt) cooldown of Hägele 2024 (§3) over the last
    cooldown_fraction of the steps: lr(t) = peak * (1 - sqrt((t - t_decay) / (T - t_decay))).
    Every checkpoint before t_decay sits at the same point of the schedule: PTQ error moves
    when the learning rate decays, not with the tokens seen (Catalan-Tatjer 2026).
    """
    decay_start = steps - round(settings["cooldown_fraction"] * steps)
    if step < settings["warmup_steps"]:
        return (step + 1) / settings["warmup_steps"]
    if step < decay_start:
        return 1.0
    return 1.0 - math.sqrt((step - decay_start) / max(1, steps - decay_start))


def setup() -> tuple[int, int, torch.device]:
    """Return rank, world size and device; join the process group when started by torchrun."""
    if "WORLD_SIZE" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group("nccl")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local)


def make_optimizer(model, settings: dict) -> torch.optim.AdamW:
    """Return AdamW with weight decay on the linear layers only, not on the token embeddings
    or the RMSNorm gains (OLMo 2 §3.3: decayed embeddings shrink and destabilise)."""
    decayed, other = [], []
    for name, parameter in model.named_parameters():
        linear = parameter.dim() == 2 and "embed_tokens" not in name
        (decayed if linear else other).append(parameter)
    groups = [{"params": decayed, "weight_decay": settings["weight_decay"]},
              {"params": other, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=settings["lr"], betas=tuple(settings["betas"]),
                             eps=settings["eps"], fused=torch.cuda.is_available())


def summed_loss(model, windows: torch.Tensor, autocast: bool) -> torch.Tensor:
    """Return the summed next-token cross-entropy (nats) of windows [batch, n]: token i
    predicts token i + 1, so each window gives n - 1 predictions. Logits in float32."""
    with torch.autocast(windows.device.type, dtype=torch.bfloat16, enabled=autocast):
        logits = model(input_ids=windows[:, :-1]).logits  # [batch, n - 1, vocab]
    return F.cross_entropy(logits.float().flatten(0, 1), windows[:, 1:].flatten(),
                           reduction="sum")
