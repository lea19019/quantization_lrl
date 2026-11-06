'''
Activation patching
Question. Where in the model does the extra low-resource damage come from?
Method. Feed the translation prompt with the reference forced in to the full and the quantized model. Copy the full model's activation at one site (a layer, module or attention head) into the quantized model and measure how much of the damage on the reference tokens disappears. A first-order screen (attribution patching, Kramár 2024) scores every site from one forward and one backward pass. Exact patching then confirms the top sites.
Output. A map of sites x languages, each cell the share of the gap recovered at that site.
Controls. Random sites of the same size, the sites most damaged on English, and the same sites on French.

Algorithm. One part patched at a time.
Steps 1, 2 and 4 are shared with the other experiments, in qlrl/experiments/run.py.
Sites: the output of the attention and of the MLP in each layer of the model.

For each sentence (the prompt and its human reference, tokenized as in qlrl/eval/logprob.py):
    1. Full run. Feed the tokens to the full model. Save the output of every site, at all positions.
       log p_full = summed log p of the reference tokens.
    2. Quantized run, no patch. Feed the same tokens to the quantized model.
       log p_quant = summed log p of the reference tokens.
    3. For each site s, starting again from the unpatched quantized model:
       Feed the same tokens. Replace the output of s with its saved full output, at all positions.
       Every other part, the later layers included, is computed by the quantized model from the result.
       log p_patched(s) = summed log p of the reference tokens.
    4. Save the sentence's log p_full, log p_quant and log p_patched(s) for every s.

For each site s, over all the sentences:
    damage = sum(log p_full - log p_quant)
    damage left = sum(log p_full - log p_patched(s))
    recovered(s) = 1 - damage left / damage
'''

import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerBase

from qlrl.eval.logprob import text_logprob


def get_sites(model: LlamaForCausalLM) -> dict[str, torch.nn.Module]:
    """Return { site name: module } for the attention and the MLP of every layer."""
    sites = {}
    for index, layer in enumerate(model.model.layers):
        sites[f"layer{index}.attention"] = layer.self_attn
        sites[f"layer{index}.mlp"] = layer.mlp
    return sites


def make_save_hook(name: str, saved: dict[str, torch.Tensor]):
    """Return a forward hook that stores its module's output, [1, seq, hidden], in `saved`."""
    def save(_module, _inputs, output):
        # Attention returns (activation, attention weights); the MLP returns the activation.
        saved[name] = output[0] if isinstance(output, tuple) else output
    return save


def make_replace_hook(full_output: torch.Tensor):
    """Return a forward hook that replaces its module's output with `full_output`."""
    def replace(_module, _inputs, output):
        # Keep the tuple for attention: the decoder layer unpacks two values from it.
        return (full_output, *output[1:]) if isinstance(output, tuple) else full_output
    return replace


def patch_sites(q_model: LlamaForCausalLM, tokenizer: PreTrainedTokenizerBase, prompt: str,
                reference: str, saved: dict[str, torch.Tensor]) -> dict[str, float]:
    """Return, per site, the reference's log p (nats) with that site's saved full output patched."""
    patched = {}
    for name, module in get_sites(q_model).items():
        handle = module.register_forward_hook(make_replace_hook(saved[name]))
        patched[name], _ = text_logprob(q_model, tokenizer, prompt, reference)
        handle.remove()
    return patched


def recovered(rows: list[dict]) -> dict[str, float]:
    """Return, per site, the share of the damage over all `rows` that patching it removes."""
    damage = sum(row["full"] - row["quant"] for row in rows)
    shares = {}
    for name in rows[0]["patched"]:
        damage_left = sum(row["full"] - row["patched"][name] for row in rows)
        shares[name] = 1 - damage_left / damage
    return shares
