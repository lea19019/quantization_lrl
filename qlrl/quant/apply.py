"""Walk a model and quantize its linear layers with any quantizer from this package.

The quantizer is a strategy: `make_quantizer()` returns a fresh instance per layer, since
calibrated methods keep per-layer state (GPTQ's Hessian, AWQ's channel means). If the
instance has `add_batch`, calibration inputs are captured at that layer with a forward hook
and fed to it before `quantize`. GPTQ and AWQ take the inputs from a model whose earlier
layers are already quantized (Frantar 2022 §4, "sequential"), so blocks are processed in
forward order. `sites` selects module names (q_proj, down_proj, ...) or layer indices so the
per-module map (design.md, H2a) is the same function with a filter.
"""

from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn


def find_linears(model: nn.Module, sites: list[str] | None) -> dict[str, nn.Linear]:
    """Return {qualified name: nn.Linear} in forward order, filtered by `sites` if given."""
    raise NotImplementedError


def capture_inputs(layer: nn.Linear, quantizer: object) -> Callable[[], None]:
    """Register a forward pre-hook that passes the layer input [tokens, in] to
    `quantizer.add_batch`; return a function that removes the hook."""
    raise NotImplementedError


def install_activation_hook(layer: nn.Linear, quantizer: object) -> None:
    """For quantizers with `quantize_input` (SmoothQuant): apply it to the input of `layer`
    on every forward pass."""
    raise NotImplementedError


@torch.no_grad()
def quantize_model(
    model: nn.Module,
    make_quantizer: Callable[[], object],
    calibration: Iterable[Tensor] | None,
    sites: list[str] | None,
) -> nn.Module:
    """Quantize the selected linear layers in place and return the model.

    `calibration` yields token-id batches [batch, seq]; None for RTN. Weights are replaced
    by their fake-quantized values in the model's dtype (bf16), per design.md §3.
    """
    # For each linear in forward order: quantizer = make_quantizer(); if it has add_batch,
    # run the calibration batches with the capture hook on; weight <- quantizer.quantize
    # (weight); if it has quantize_input, install the activation hook.
    raise NotImplementedError
