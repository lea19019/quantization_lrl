'''
One forward pass that keeps what the experiments read, and the activation change between two
such passes. All tensors here are from one sentence.

Saved per layer: the attention and MLP outputs, the residual stream after the layer, and the input
of o_proj, which is the attention heads' outputs side by side before o_proj mixes them.
Change is measured on the positions that predict reference tokens: ||quant - full|| and ||full||.
'''
import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerBase

from qlrl.experiments.patching import make_save_hook


def reference_ids(tokenizer: PreTrainedTokenizerBase, prompt: str, reference: str, device):
    """Return ids [1, seq], the positions predicting reference tokens, and those tokens."""
    # Tokenized apart and joined, as in qlrl/eval/logprob.py. Position t predicts token t + 1.
    prompt_ids = tokenizer(prompt).input_ids
    reference_ids = tokenizer(reference, add_special_tokens=False).input_ids
    ids = torch.tensor([prompt_ids + reference_ids], device=device)
    positions = torch.arange(len(prompt_ids) - 1, ids.shape[1] - 1, device=device)
    return ids, positions, ids[0, positions + 1]


def make_heads_hook(name: str, saved: dict[str, torch.Tensor]):
    """Return a pre-hook that stores o_proj's input, [1, seq, heads * head size], in `saved`."""
    def save(_module, args):
        saved[name] = args[0]
    return save


def capture(model: LlamaForCausalLM, ids: torch.Tensor, positions: torch.Tensor):
    """Return (float logits at `positions`, [tokens, vocab]; saved activations by name)."""
    saved, handles = {}, []
    for index, layer in enumerate(model.model.layers):
        for part, module in (("attention", layer.self_attn), ("mlp", layer.mlp),
                             ("residual", layer)):
            handles.append(module.register_forward_hook(
                make_save_hook(f"layer{index}.{part}", saved)))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
            make_heads_hook(f"layer{index}.heads", saved)))
    with torch.no_grad():
        logits = model(ids).logits.float()  # bf16 logits lose precision
    for handle in handles:
        handle.remove()
    return logits[0, positions], saved


def activation_change(full: dict, quant: dict, positions: torch.Tensor) -> dict:
    """Return { "distance": { name: ||quant - full|| }, "norm": { name: ||full|| } }."""
    change = {"distance": {}, "norm": {}}
    for name in full:
        if name.endswith(".heads"):
            continue
        full_part = full[name][0, positions].float()
        change["distance"][name] = (quant[name][0, positions].float() - full_part).norm().item()
        change["norm"][name] = full_part.norm().item()
    return change


def head_change(full: dict, quant: dict, positions: torch.Tensor, heads: int) -> dict:
    """Return { layer name: (||quant - full||^2, ||full||^2) per head, each [heads] on the CPU }."""
    change = {}
    for name in full:
        if not name.endswith(".heads"):
            continue
        full_part = full[name][0, positions].float().reshape(len(positions), heads, -1)
        quant_part = quant[name][0, positions].float().reshape(len(positions), heads, -1)
        change[name] = (((quant_part - full_part) ** 2).sum(dim=(0, 2)).cpu(),
                        (full_part ** 2).sum(dim=(0, 2)).cpu())
    return change
