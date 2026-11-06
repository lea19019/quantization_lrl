'''
Noise vs flatness
Question. Does a low-resource language lose more because more rounding noise reaches its logits, or because its predictions were flatter to begin with?
Method. If quantization changes the logits by noise of size σ, the expected rise in NLL of the correct token is about ½ σ² (1 − Σ_j p_j²), with p the full model's next-token distribution (second-order expansion of the log-softmax). Per token, measure σ² as the variance of z_quant − z_full and the flatness 1 − Σ_j p_j² from the full model, and see which of the two carries the gap between languages.
Output. Per language, the share of the gap explained by noise and by flatness. It says which kind of cause to look for in the other experiments.
Controls. English and French, compared at matched full-precision confidence.
'''


import torch


def noise_flatness(z_full: torch.Tensor, z_quant: torch.Tensor) -> dict[str, list[float]]:
    """Return, per reference token, sigma^2 and flatness; z: [tokens, vocab] float logits."""
    # A constant added to all logits does not change softmax; the variance ignores it too.
    sigma2 = (z_quant - z_full).var(dim=-1, unbiased=False)
    probabilities = torch.softmax(z_full, dim=-1)
    flatness = 1 - (probabilities ** 2).sum(dim=-1)
    return {"sigma2": sigma2.tolist(), "flatness": flatness.tolist()}
