# The problem

**In one sentence.** We do not know why rounding the same weights to 4 or 3 bits removes
more of a model's ability to predict low-resource-language text than high-resource-language
text, or whether that difference is even a property of the language rather than of the
model, the method and the measurement.

One page that says exactly what is being explained. Everything else in `docs/` serves
this. Terms are defined operationally so that two people would measure the same thing.

## 1. The observation

Round a trained multilingual LLM's weights to 4 or 3 bits without retraining. Measure its
quality per language before and after. Some languages lose much more than others.
Reported by Marchisio 2024 (accuracy, Aya and Command), Marie & Fujita 2025 (COMET, 55
languages), Chimoto 2026 (perplexity, Llama-3.1-8B and Qwen2.5-7B), Soualhi 2026
(accuracy, small Gemma and Qwen models).

What the published numbers support, once recomputed under different normalisations (audit
of 2026-09-09): the gap is large and robust at 2 bits and under extreme sparsity; at 4 bits
most reported per-language gaps are within the run-to-run spread from calibration draw
alone (0.9 to 1.6 accuracy points, Williams 2024 p.6) and no paper reports that spread. The
predictor the papers actually show is full-precision quality, not resource level (Marie &
Fujita p.5: "the worse the baseline score of a language, the more it suffers"). Languages
near a metric floor look resilient or even improve, which is an artefact. So the fact this
project starts from is: **absolute degradation grows as full-precision quality falls, until
the language hits a floor**, and whether that is a resource-level effect at 4 bits is itself
to be established, with seeds.

## 1a. Why we care

Quantization is how LLMs reach cheap hardware. An 8B model is 16 GB in bf16 and 4 GB at
4 bits, and since decoding is memory-bound that is close to a 4× speedup per token
([../concepts/quantization.md](../concepts/quantization.md) §1). Cheap hardware is where
speakers of low-resource languages are: Ahia et al. 2021 call this the "low-resource
double bind", the languages with the least data are also the ones that must be served
compressed (Findings of EMNLP 2021, p.1).

The cost is not small and not visible in the usual numbers:

- Marie & Fujita 2025, Llama-3.1-8B NF4, translation into English: French loses 0.3 COMET,
  Japanese 1.5, Bengali 7.7, Malayalam 9.7 (arXiv:2508.20893, p.5). Llama-3.3-70B: Polish
  loses 0.9, Zulu 6.2 (p.5). At 2 bits Qwen3-8B loses about 2 COMET on ja/fr and 17 on bn/ml.
- Marchisio et al. 2024: at W4 on the 103B model, Latin-script languages lose 0.7% and
  non-Latin 1.9% (Findings of EMNLP 2024, p.2); human raters saw Japanese drop 16.0% where
  automatic metrics showed 1.7% (p.2).
- Chimoto et al. 2026, Llama-3.1-8B GPTQ with English calibration: per-token loss rises
  0.125 nats on English and 0.545 on Hausa (EACL 2026, Table 2, recomputed as ln of the
  perplexity ratio).

## 1b. Potential impact

- **If H1 holds**, the damage is a matter of how far a language is along its learning
  curve, and the fix is data: continued training on that language before quantizing, with
  the amount estimated from the curve. It also gives a way to predict which languages a
  quantized model will fail on from the loss surface alone, before quantizing.
- **If H2 holds**, the damage has an address, and the fix is precision allocation: keep
  the modules, dimensions or experts the low-resource languages depend on in higher
  precision, at a cost of a few percent of the memory saved. It also tells interpretability
  work where to look.
- **Either way**, a definition of damage that survives normalisation, floors and
  calibration-draw noise, and a check of whether it predicts translation quality, is
  reusable by anyone who quantizes a multilingual model. Today no multilingual study
  reports run-to-run spread.
- **Who benefits.** Anyone deploying a quantized model to speakers of Swahili, Yoruba, Zulu
  or the other languages served compressed or not at all, and the lab's own work on them.

## 2. Terms

- **Post-training quantization (PTQ).** Weights replaced by the nearest point on a
  grid with 2^b levels, per group of 128 weights, no retraining. Weight-only unless
  stated. Methods: RTN, GPTQ, AWQ; SmoothQuant for W8A8. See
  [../concepts/quantization.md](../concepts/quantization.md).
- **Low-resource language (LRL).** Operationally, for a given model: a language absent
  from the model's official supported-language list, or with a small share of its
  pretraining data where that share is known. For the from-scratch model the share is
  set by us. "Low-resource" is a property of a language *and a model*, not of a language.
- **Full precision.** The released bf16 checkpoint of the same model. Every damage number
  is relative to it.
- **Damage of a language L under configuration c.** `D(L, c)`, defined in §3.
- **The gap.** `G(L, c) = D(L, c) − D(English, c)`, and the ratio `D(L, c) / D(English, c)`.
  Both are reported; they can disagree.

## 3. What "damage" means here

The goal is translation quality: how much COMET a language loses under quantization
relative to English. Damage, defined below, is the diagnostic used to explain that loss
at the level of single predictions. Whether low damage means good translation is checked,
not assumed ([evaluation.md](evaluation.md) §1).

Seventeen papers use nine definitions of damage and four define none. Relative drops,
floors and tokenization each create or hide a gap by themselves. This project fixes the
definitions below and requires an ordering to hold under all of them before saying "more".

**Per token.** Feed the same sentence to the full model and to the quantized model. At each
position both output a probability for the next token, and we know which token is correct
(the gold token, since the sentence is given). If the full model gave the gold token
probability 0.40 and the quantized model gave it 0.30, the damage at that position is
`log 0.40 − log 0.30 = 0.28`. In general `d_gold(t) = log p_full(gold_t) − log p_quant(gold_t)`.
Positive means the quantized model got worse there; zero, no change; negative, better.

**Per word and per byte.** A tokenizer may cut one Swahili word into three pieces and one
English word into one, so per-token numbers are not comparable across languages: the
Swahili word contributes three times and each later piece is easy once the first is known.
Summing `d_gold` over the pieces of a word gives the drop in log-probability of the whole
word, independent of how it was cut. Dividing a sentence's total `d_gold` by its byte count
gives per-byte damage, which is comparable even where words are not well defined. Both are
used for comparing languages. No paper in the corpus reports either.

**Per language, model-internal.** `D(L, c)` is the mean per-word damage over the FLORES-200
devtest sentences of language L under configuration c, with a confidence interval from
resampling sentences (tokens in one sentence are not independent) and at least five
calibration draws for GPTQ and AWQ, with the spread reported.

**Per language, standard metrics.** For a task score `s` (accuracy, COMET, chrF) with full-
precision value `s_fp` and quantized value `s_q`, report all four:

```
abs   = s_fp − s_q                       the only one interpretable as harm; lead with it
rel   = (s_fp − s_q) / s_fp              report, never lead with; inflates low baselines
skill = (s_fp − s_q) / (s_fp − chance)   fraction of above-chance skill destroyed
head  = (s_fp − s_q) / (1 − s_fp)        fraction of headroom lost; inflates high baselines
```

**Gates.** Floor: `skill` is computed only where `s_fp − chance` exceeds three standard
errors; a language at or below chance, or at a metric floor (COMET near 40), is excluded and
said to be excluded, never reported as resilient. Noise: a per-language gap smaller than
the calibration-draw spread (about 1.6 accuracy points or 1.6 COMET in the literature) is
reported as unresolved. Unit: per-token perplexity is never compared across languages.

**Checks.** Top-1 flip rate; KL(full ‖ quant); off-target language rate on generated text.

## 4. The question

Given a model, a quantization configuration and the five languages: **why is
`G(L, c) > 0` for the low-resource languages, and what mechanism produces it?** The
candidate mechanisms are E1–E7 in [../concepts/explanations.md](../concepts/explanations.md).
An answer is a mechanism whose prediction differs from the other candidates' and from the
null, tested with criteria written before the run.

## 5. Scope

Models Llama 3.1 8B, Qwen 2.5 7B, a from-scratch GPT-2-scale decoder (core); Aya and the
Gemma 4 pair (extension). Languages English, French, Swahili, Yoruba, Zulu. Methods RTN
and GPTQ at 4 and 3 bits (core); AWQ and SmoothQuant (extension). Text: FLORES-200
devtest. Details in [design.md](design.md).

## 6. Not the problem

- Making quantized inference faster; kernels; real int4 storage. Fake quantization is enough.
- Quantization-aware training, except as a comparison point if Gemma 4 QAT checkpoints are used.
- Activation-quantization outlier handling for its own sake.
- Social bias or fairness in the sense of Ramesh 2023 and Gonçalves 2023.
- Explaining why a language is weak at full precision. Only the *extra* loss from rounding.

## 7. What counts as done

- The gap reproduced on the core models with CIs and three seeds, under the §3 definition,
  and shown to exceed the run-to-run and calibration-draw spread (the E7 null).
- For each selected explanation: its prediction stated in advance, the measurement made,
  the pre-set criterion applied, a one-paragraph verdict.
- A coverage table and one results table per explanation in the report.
