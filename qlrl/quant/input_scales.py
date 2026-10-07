'''
Option 3: scale a linear's input explicitly instead of folding 1/s into the layer before it.
Written by Claude Code on 2026-10-07 at Adrian's request.

SmoothQuant and AWQ both rewrite y = x W^T as y = (x / s) (W s)^T, one s per input channel.
Version 1 folded 1/s into the norm or linear that produces x. That is free at inference, but
it only works when that producer feeds the matrix directly, so it needed a hand-written map
per architecture (architectures.py) and could not reach a matrix behind a non-linearity
(GPT-2's c_proj after GELU: GELU(x / s) != GELU(x) / s) or behind attention.

Here every scaled linear gets a forward pre-hook that divides its input by s just before the
multiplication. That works for any linear in any model, and every linear gets its own s.
The cost is one extra element-wise division per linear at inference; an int8 or 4-bit
kernel would fuse it into the step that rounds x.

Hooks are not saved by save_pretrained, so the scales are saved next to the model in
input_scales.pt and attached again after loading with load_input_scales.
'''
from pathlib import Path

import torch

SCALES_FILE = "input_scales.pt"


def attach_input_scale(linear: torch.nn.Module, scales: torch.Tensor) -> None:
    """Make the linear divide its input by s (one value per input channel) before multiplying."""

    def divide_input(_module, args):
        x = args[0]
        # Divide in float32, then return to the model's dtype (bfloat16) for the matmul.
        return ((x.float() / scales).to(x.dtype),)

    linear.register_forward_pre_hook(divide_input)


def save_input_scales(folder: str, scales: dict[str, torch.Tensor]) -> None:
    """Save {module name: s} next to a saved model."""
    torch.save({name: value.cpu() for name, value in scales.items()}, Path(folder) / SCALES_FILE)


def load_input_scales(model: torch.nn.Module, folder: str) -> bool:
    """Attach the scales saved in `folder` to the model, if there are any; return whether so."""
    path = Path(folder) / SCALES_FILE
    if not path.exists():
        return False
    for name, scales in torch.load(path).items():
        linear = model.get_submodule(name)
        attach_input_scale(linear, scales.to(linear.weight.device))
    return True
