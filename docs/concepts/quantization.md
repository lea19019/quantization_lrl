# Background: the concepts, in plain language

What you need to understand the design in [design.md](../project/design.md). Method formulas and
the published numbers to validate against are in [papers/methods.md](../papers/methods.md).

## 1. Post-training quantization

A trained model stores each weight as a 16-bit float (bf16). Quantization replaces each
weight by the nearest point on a coarse grid with 2^b levels, so it can be stored in b
bits. *Post-training* means no retraining: the weights are rounded once and used as is.

**Two families.**

- *Weight-only* (RTN, GPTQ, AWQ, GGUF k-quants, NF4): only weights are rounded;
  activations stay in 16 bits. The low-resource gap in the literature appears already
  here, at 4 bits.
- *Weight-and-activation* (W8A8 SmoothQuant, W4A4): both are rounded. Activation
  outliers (a few channels with huge values) are the central problem for this family.

**Why it makes inference faster.** Generating one token means reading every weight
once. An 8B model is 16 GB in bf16 and 4 GB at 4 bits. The GPU is waiting on memory,
not arithmetic, so reading a quarter of the bytes is close to a 4× speedup per token.
Fused kernels dequantize on the fly inside the matrix multiply.

**Fake quantization.** We round the weights and store them back as bf16. The model runs
at full speed and full memory; only the *values* sit on the 4-bit grid. That is enough
to measure damage. Real speedups need fused kernels and are not needed here.

## 2. The grid

For a group of g weights (we use g = 128) with minimum `wmin` and maximum `wmax`:

```
step  = (wmax - wmin) / (2^b - 1)          # asymmetric, per-group scale
q     = round((w - wmin) / step)           # integer in [0, 2^b - 1]
w_hat = q * step + wmin                    # dequantized weight
```

Every weight is within half a step of its original. That bound is the first test to
write. Smaller groups give finer scales at the cost of more scale storage.

## 3. The methods

- **RTN (round-to-nearest).** The grid above, nothing else. No data. Seconds. The
  baseline in every paper.
- **GPTQ** (Frantar 2023). Rounds one weight column at a time and updates the columns
  not yet rounded to compensate for the error, using the Hessian of the layer's
  reconstruction loss, `H = 2 X X^T`, estimated from a small calibration set (128
  windows of 2048 English tokens in most papers). *Act-order* rounds the columns with
  the largest activations first, while there are still many columns left to absorb
  their error. Same grid as RTN; better placement on it.
- **AWQ** (Lin 2024). Before rounding, scales each input channel of a weight matrix up
  by a factor chosen from that channel's activation magnitude (and scales the
  activation down to match), so that the channels that matter most get finer relative
  resolution. Grid search over one exponent per layer. Also needs calibration data.
- **SmoothQuant** (Xiao 2023). For W8A8: migrates difficulty from activations to
  weights by dividing each activation channel by a per-channel factor and multiplying
  the matching weight row by it, so both become quantizable at 8 bits. The migration
  strength alpha is a hyperparameter. Calibration data sets the factors.

GPTQ, AWQ and SmoothQuant all set something from a calibration sample. That is what E3
is about: does the sample cover the activations Swahili produces?

## 4. Why perplexity hides the gap

Perplexity is the exponential of the mean negative log-probability over a corpus. A
corpus is mostly English, or mostly easy tokens, so a small average change can hide a
large change on a subset. Marchisio 2024 made the same point about automatic metrics vs
human raters. Everything here is measured **per token, per language**.

## 5. The token-level measurements

For every token position, the full model and the quantized model see the same input.

- **`d_gold`** = `logp_full(gold) - logp_quant(gold)`. How much log-probability the
  correct next token lost. Positive is damage. The main number.
- **Top-1 flip**: the argmax changed. **KL(full ‖ quant)**: how much the whole
  distribution moved.
- **Confidence margin** = top-1 minus top-2 log-prob of the *full* model. Small margin
  means a close call. E2 predicts damage concentrates where margins are small (Tran
  2022; Proskurina 2024).
- **Entropy and gold log-prob** of the full model: two more "how sure was it" controls.
- **Fertility** = tokens per word for a language under the model's tokenizer (Rust
  2021). **Continuation flag**: whether a token starts a word or continues one. Both
  for E4.
- **Residual-stream norm**: the length of the hidden vector at a late layer for each
  token. Chang 2025 found it predicts which English inputs break. Timkey & van
  Schijndel 2021 warn that a few rogue dimensions can dominate any norm; standardize
  them before trusting the measure.

**Numerics.** Compute the final projection in float32. bf16 logits come in steps of
0.125, so margins tie in blocks and any tie-break becomes an artefact.

## 6. Statistics

- **Sentence-level bootstrap** for CIs: resample sentences, not tokens, because tokens
  within a sentence are not independent.
- **Confidence matching (E2)**: bin tokens by full-model margin decile, subtract the
  decile mean of damage, and ask how much of the between-language variance
  (eta-squared) survives. Compare against a null where margins are shuffled.
- **Seeds.** GPTQ on GPU is not bit-reproducible and its calibration windows are a
  random draw. Any GPTQ or AWQ claim needs at least three seeds; run-to-run spread is
  reported next to the sentence CI. RTN is deterministic.

## 7. The explanations

The candidate explanations E1–E7, with their evidence and predictions, are in
[explanations.md](explanations.md). The measures above map onto them: margin and entropy
(E2), calibration variants (E3), fertility and token premium (E4), residual norms and
per-module sensitivity (E5), off-target rate (E6), seeds and normalisation (E7).

## 8. An angle deliberately not taken

Massive activations and outlier features (Sun 2024; An 2025; the outlier papers in
[papers/outliers.md](../papers/outliers.md)) are a problem for *activation* quantization.
The gap we study appears under weight-only quantization, which never rounds an
activation. They stay in the reading list as background for E3, E5 and SmoothQuant.
