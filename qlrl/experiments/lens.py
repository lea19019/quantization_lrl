'''
Logit lens
Question. At what depth does a language's prediction break?
Method. Decode the residual stream at every layer through the final norm and the output head, in the full and the quantized model (Wendler 2024).
Output. Per language, the probability of the correct next token at each layer for both models. The layer where the quantized curve leaves the full one is where the damage starts. It narrows the layers that activation patching needs to search.
Controls. The same curves on English and French.
'''


import torch
from transformers import LlamaForCausalLM


def lens_curve(model: LlamaForCausalLM, saved: dict[str, torch.Tensor], positions: torch.Tensor,
               targets: torch.Tensor) -> list[float]:
    """Return, per layer, the summed log p (nats) of the reference decoded from that layer."""
    # The norm and the head are not quantized, so both models decode through the same weights.
    curve = []
    with torch.no_grad():
        for index in range(len(model.model.layers)):
            hidden = saved[f"layer{index}.residual"][0, positions]  # [tokens, hidden]
            logits = model.lm_head(model.model.norm(hidden)).float()  # [tokens, vocab]
            logprobs = torch.log_softmax(logits, dim=-1)
            curve.append(logprobs[torch.arange(targets.shape[0]), targets].sum().item())
    return curve
