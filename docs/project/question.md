# The question and the hypotheses

What this project is trying to find out, and the explanations it will test. The problem
itself is defined in [problem.md](problem.md); the full list of candidate explanations, with
evidence, is [../concepts/explanations.md](../concepts/explanations.md). Experiments and
design come later and are not in this file.

## The question

**Why does post-training quantization remove more of a multilingual LLM's ability to
predict the next word in low-resource languages than in high-resource languages?**

Two clarifications the question carries with it:

- *Same treatment, different outcome.* One model, one set of weights, one rounding. The
  languages differ only in what text is fed in. So whatever produces the difference is
  already inside the model before rounding, or in how the rounding interacts with it.
- *First establish that the language is the cause.* The published gap could come from the
  model checkpoint, the quantization method, the bit width, the metric, or the calibration
  sample, rather than from the language. Most reported 4-bit gaps are smaller than the
  spread caused by changing the calibration sample alone (Williams 2024, p.6). So the
  question has a prerequisite: show that the extra loss survives when those are held
  fixed and averaged over. If it does not, the answer is "it is not the language".

## The hypotheses

### H1. Low-resource languages are under-fitted, so the same perturbation costs them more (E1)

**Statement.** The model's weights sit at a minimum of the loss for the whole training
corpus, not for each language. A language with a small share of the data is further from
its own minimum: the loss surface there is steeper (larger gradient norm) and more curved
(larger Hessian eigenvalues). Rounding moves the weights by roughly the same amount for
every language, and the same move costs more where the surface is steeper.

**Why we think so.** Tran et al. 2022 prove this for pruning in a classifier: the excess
loss of a group after a weight perturbation is bounded by the group's gradient norm and
Hessian eigenvalue, and groups with fewer samples have larger values of both at
convergence (NeurIPS 2022, Proposition 1, p.4). The bound holds for any perturbation,
so it applies to rounding. Zeng et al. 2024 make the same argument for languages inside a
multilingual LLM: the model is at "a global local minimum, rather than individual local
minima for each language" (p.3), and find that "the larger the proportion of a language in
the model training dataset, the more resistant it is to compression" (p.1). Catalan-Tatjer
et al. 2026 show that at model level PTQ damage tracks loss-surface sharpness (ICLR 2026,
p.9). Across languages, every gap paper finds damage grows as full-precision quality falls
(Marie & Fujita 2025, p.5), which is what this hypothesis predicts if quality tracks data
share.

**What is against it.** Diddee et al. 2022 find no data-size trend under int8 across eight
low-resource languages (p.9). Chang 2025 and Wu 2026 find that within English the
poorly-modelled tail is hit *less* by quantization, not more. Nobody has measured
per-language gradient norm or curvature in an LLM.

**If true.** Per-language gradient norm and curvature at the released checkpoint predict
damage, beyond what full-precision loss already predicts; and in a model whose data
shares we set, damage follows the share.

### H2. The damage lives in specific parts of the model that low-resource languages depend on more (E5)

**Statement.** Rounding error is not spread evenly. It concentrates in particular weights
(MLP down-projections, the rows that produce outlier and massive activations), particular
residual-stream directions (the few dimensions that carry language identity or a
frequency prior), and, in mixture-of-experts models, particular experts. A language's loss
depends on how much its computation runs through those parts.

**Why we think so.** Chang 2025 finds that within English the inputs that break under
3–4-bit quantization depend on precise late-layer activations, and that restoring the MLP
down-projection outputs recovers most of the loss while restoring attention outputs
barely helps (EMNLP 2025, p.8); the full-precision residual-stream norm predicts which
inputs break (p.1). An et al. 2025 show weight outliers concentrate in the down-projection
matrices of a few layers and align with the activation-outlier dimensions (ICLR 2025,
p.4–5); Sun et al. 2024 show those massive activations act as fixed biases the model needs
(COLM 2024). Zhong et al. 2025 find language identity in a sparse set of residual
dimensions (p.1), and Puccetti et al. 2022 and Hämmerl et al. 2023 find outlier dimensions
tied to token frequency and anisotropy that grows as a language's data shrinks (p.1; p.6).
Koishekenov et al. 2023 find language-specific experts in NLLB-200 (p.2). None of these
papers compares languages under quantization; together they say the damage has an
address, and that low-resource languages may live at a different address.

**What is against it.** Chimoto 2026 finds the outlier channels AWQ selects are the same
for every calibration language; only magnitudes differ (p.8). Sun 2024 finds massive
activations at fixed dimensions and positions regardless of input. So the address may be
shared, and the difference would have to be in how much each language relies on it.

**If true.** Quantizing one module at a time locates each language's loss in different
places, or in the same place to different degrees; patching the full model's activations
into the quantized model at those places recovers the low-resource loss; the massive
activations, residual norms, or language dimensions change differently by language after
rounding.

### The null. The gap is model, method, metric and noise, not language (E7)

**Statement.** Much of the reported gap is variance in checkpoint, quantization format,
bit width, metric normalisation and calibration draw.

**Why it must be tested first.** Seventeen papers use nine definitions of damage; recomputed
under different normalisations, the published orderings change (Marchisio's Latin vs
non-Latin gap is 0.50, 0.18 or 2.06 points depending on the denominator). Hossain 2026:
"quantization impact tracks architecture and format more than it tracks language" (p.7).
No multilingual paper reports run-to-run spread.

**If true.** Under the damage definition in problem.md §3, with repeated calibration draws
and two model families, the language effect at 4 bits is inside the noise. Then H1 and H2
are asked only at 3 bits and below, where the gap is unambiguous.

## Relation between the two

H1 says *why* a language is fragile: it is further from its optimum. H2 says *where* that
fragility shows up in the network. They can both be true, and if under-fitting appears as
smaller residual norms or different use of outlier dimensions in the low-resource
languages, H2 is how H1 manifests. They can also come apart: H1 without H2 means the loss
is diffuse; H2 without H1 means the address matters even for well-fitted languages.
