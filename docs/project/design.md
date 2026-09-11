# Experimental design

What gets run, on what, and in what order. The question and hypotheses are in
[question.md](question.md); how results are measured and judged in
[evaluation.md](evaluation.md); data in [data.md](data.md). Numbers marked *proposed* are
settled with the advisor.

## 1. Models

| Model | Role | Why |
|---|---|---|
| Llama 3.1 8B | **core** | Published numbers to check against (Marie & Fujita 2025, Chimoto 2026, Chang 2025); none of sw/yo/zu in its official list, so the fine-tuning intervention is meaningful |
| GPT-2 small (124M), trained by us | **core** | The only setting where data per language is set by us; gives the learning curve of damage |
| Qwen 2.5 7B | add-on | Replicates the language ordering on a second family |
| Gemma 4 26B A4B and 31B | add-on | MoE vs dense on the same data and tokenizer; official QAT checkpoints |
| Llama 3.1 8B fine-tuned on lab text | core, phase 3 | The H1 intervention |

All H2 experiments run on the core model only. Add-on models replicate the per-language
ordering of damage and COMET loss, nothing more.

## 2. Languages

English and French as high-resource anchors present in every model; Swahili, Yoruba and
Zulu as the lab languages, absent from Llama's official list, present in Gemma's. Xhosa
and Shona substitute for Zulu or Yoruba if the lab text for either is unusable.
Coverage table (fill from model cards before phase 0; this is the H1 prediction written
down: largest damage in the ✗ cells):

| | Llama 3.1 8B | Qwen 2.5 7B | Gemma 4 | GPT-2 (ours) |
|---|---|---|---|---|
| English, French | ✓ | ✓ | ✓ | set by us |
| Swahili | ✗ | ? | ✓ | set by us |
| Yoruba, Zulu | ✗ | ? | ? | set by us |

## 3. Quantization methods

| Method | Family | Calibration | Bits | Implemented by |
|---|---|---|---|---|
| RTN | weight-only | none | 4, 3, 2 | us |
| GPTQ (act-order) | weight-only, error compensation | 128 × 2048 tokens | 4, 3, 2 | us |
| AWQ | weight-only, activation-aware scales | yes | 4, 3 | us |
| SmoothQuant | weight + activation, run as W4A8 | yes | 4 (W), 8 (A) | us |
| bitsandbytes NF4 | weight-only, calibration-free | none | 4 | library |

Group size 128 throughout. Fake quantization: weights rounded and stored back in bf16.
Calibration text is English (FLORES dev, or C4 to match the papers) by default; the
target-language and mixed variants are one add-on cell each, kept because they are cheap
once GPTQ exists. Every calibration-based method: five draws.

## 4. The experiments

### Phase 0. Baseline and the null

Full precision and every method × bits on the core model, all five languages. Output:
damage, COMET loss and the paired gaps ([evaluation.md](evaluation.md) §3–4). Decision:
if the 4-bit gap does not exceed the between-draw spread and hold on the second family,
H1 and H2 are judged at 3 bits. Also the check that Llama's full-precision quality on
sw/yo/zu is above the floor.

### Phase 1. Cheap evidence on both hypotheses (released model, no training)

- **H1a, loss surface.** For each language, one backward pass per FLORES sentence at the
  full-precision checkpoint: gradient norm and Hessian trace (Hutchinson, *proposed* 10
  random vectors). Correlate with damage across languages and configurations, partial on
  full-precision loss.
- **H2a, per-module map.** RTN 4-bit and 3-bit applied to one module type at a time (q, k,
  v, o, gate, up, down, embeddings, output head) and to one layer at a time; score all
  languages. Output: a table of language × site.
- **H2b, internal statistics.** Hooks on the residual stream in the full and the quantized
  model: per-layer norm, magnitude of the massive activations at their dimensions,
  outlier-channel activations, Zhong's language dimensions; rogue dimensions
  standardised first (Timkey 2021). Report the per-language delta.

### Phase 2. GPT-2 from scratch (started early, trained once)

- 124M parameters, *proposed* 5B tokens, byte-level BPE (32k) trained on the same mix,
  learning-rate decay at the end, checkpoints every *proposed* 250M tokens. Shares per
  language set after the corpora are measured ([data.md](data.md)); the same shares are
  never changed after the run starts.
- **H1b, learning curve.** Quantize each checkpoint (RTN and GPTQ, 4 and 3 bits); damage
  per language against tokens seen for that language, and the final checkpoint's damage
  against share. Compare checkpoints at matched points of the schedule; final after decay.
- **H2 on GPT-2.** The per-module map and internal statistics, to see whether the address
  of the damage is the same in a model where the shares are known.

### Phase 3. Interventions

- **H1c, fine-tune and re-quantize.** Continue training Llama 3.1 8B on lab Swahili text
  (then Yoruba, Zulu if time), re-quantize, and measure that language's damage, gradient
  norm and COMET loss. Controls: the full-precision baseline rises too, so the drop in
  damage must exceed what the baseline change predicts; English and French are measured
  before and after for forgetting.
- **H2c, activation patching.** Run the quantized model and overwrite the activations at
  one site (each module type, each layer) with the full model's; damage recovered per
  language. The causal version of the per-module map.
- **H2d, logit lens.** Decode from every layer in both models; the depth at which the
  quantized prediction diverges, per language.

### Add-ons, in order

1. Qwen 2.5 7B and the Gemma 4 pair through phase 0 and the per-module map; expert
   routing per language before and after quantization on the MoE.
2. Target-language and mixed calibration cells for GPTQ and AWQ.
3. Cross-lingual translation (sw↔zu, yo↔sw) and off-target rate.
4. An intervention aimed at whatever phase 1–3 points to: mixed precision on the
   sensitive modules, or target-language calibration, or fine-tune-then-requantize.

## 5. Order and dependencies

Phase 0 needs the quantizers validated and the scorer passing its sanity checks
([evaluation.md](evaluation.md) §2) and nothing else. Phase 1 needs phase 0. Phase 2
needs only the data and can start in week 1. Phase 3 needs phases 1 and 2 to say where to
look, and lab text prepared. Add-ons wait for the week-9 draft.

## 6. Open, to settle with the advisor

- GPT-2 training corpora and the share vector (see [data.md](data.md)).
- The "learning curve" reading of H1: data-scaling curve per language.
- Whether SmoothQuant at W4A8 or at W8A8, or both.
- Hutchinson vector count, checkpoint spacing, and the 5B-token budget.
