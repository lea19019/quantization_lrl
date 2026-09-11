# Evaluation

Quantities are defined in [problem.md](problem.md) §3; hypotheses in
[question.md](question.md). Criteria for judging hypotheses come with the experiment design.

## 1. Metrics

| Metric | Role | What it measures | How |
|---|---|---|---|
| **Human score** | goal | translation quality as judged by speakers | direct assessment 0–100, §3 |
| **COMET** | goal | translation quality, automatic | `Unbabel/wmt22-comet-da`, reference-based, × 100 |
| **`D(L, c)`** | diagnostic | per-word next-word-prediction loss under quantization | problem.md §3 |
| Per-byte damage | diagnostic | same, tokenizer-independent | sentence damage / UTF-8 bytes |
| Flip rate, KL | diagnostic | argmax changes; distribution shift | per token |
| Margin, entropy | control | full-model confidence per token | for H-level analysis |
| Token premium, fertility | control | fragmentation per tokenizer | Petrov 2023; Rust 2021 |
| chrF, spBLEU | extension | continuity with older tables | sacrebleu; `tokenize=flores200` |
| Off-target rate | extension | wrong output language | NLLB `lid218e` |

Every loss is `full − quantized`, reported with its full-precision baseline, and for the
goal metrics also as relative, skill- and headroom-normalised drops with the floor gate
(problem.md §3).

**Goal vs diagnostic.** Translation quality is what matters to a speaker. Damage is what
we can measure at every token and explain mechanically. Low damage does not guarantee
good translation (a model can lose little per word and still switch language or repeat),
so the link is measured (§4), never assumed. Hypotheses are judged on damage; where damage
and quality disagree, that is reported as a finding. Where COMET and human loss disagree,
the human number carries the conclusion: Marchisio 2024 saw a 16% human drop where
metrics saw 1.7% (p.2).

## 2. Gates before any result is read

- Each quantizer reproduces the published WikiText-2 perplexity on Llama-2-7B within 0.1
  ([../papers/methods.md](../papers/methods.md)). WikiText-2 is a small English Wikipedia
  corpus every quantization paper reports on; it is the only public correctness check.
- The full model scored against itself gives zero damage, KL and flips. Logits in float32.
- RTN is bit-identical across runs. GPTQ and AWQ use five calibration draws; every number
  from them is a mean with its standard deviation.

## 3. Protocol

**Data.** FLORES-200 devtest, 1012 parallel sentences, eng_Latn, fra_Latn, swh_Latn,
yor_Latn, zul_Latn. Dev supplies calibration text and prompt examples, never scores.

**Translation.** X→en and en→X for the four non-English languages. Fixed 5-shot prompt
from dev, same examples for every configuration of a pair; greedy; output capped at 256
tokens; prompt stored with results.

**Human evaluation.** Raters who speak Swahili, Yoruba or Zulu (French if available, as the
high-resource control) rate translations into their language. 100 fixed devtest
sentences per language; configurations: full precision and each core quantized one; two
raters per language, each rating all sentences of all configurations, blind and shuffled.
Reported: mean per configuration with interval, loss vs full precision, Krippendorff's
alpha, per-sentence correlation with COMET loss. Minimum budget 100 × 3 × 2 ratings per
language, planned before the runs.

**Damage.** Each sentence alone after BOS, no prompt. A word is a maximal run of
non-whitespace; tokens map to words by offset mapping, a token spanning two words goes to
the first; word damage is the sum of `d_gold`; the first token of a word is word-initial,
the rest continuation. `D(L, c)` is the mean word damage over the 1012 sentences.

**Fragmentation.** Once per tokenizer: token premium against English on the same
sentences, tokens per word, share of words split into more than one token.

## 4. Uncertainty and the damage-to-quality link

- **Sentence bootstrap.** 1000 resamples, 95% percentile interval, for every mean.
- **Paired gap.** FLORES is parallel, so the gap between L and English is bootstrapped over
  shared sentence indices. This interval backs any claim that a language loses "more".
- **Noise gate.** A gap whose paired interval includes zero, or is smaller than the
  between-draw SD, is unresolved.
- **Does damage predict quality?** Per language and configuration: Spearman between
  per-sentence damage and COMET loss, and with human loss on the rated subset; across
  languages, whether the orderings by damage, COMET loss and human loss agree. In every
  results table; disagreements flagged.

## 5. Comparing

- **Across languages, one model.** Losses with intervals; paired gaps; the orderings.
- **Across configurations, one model.** Same sentences; RTN is the floor, so a gap present
  under RTN is not attributable to calibration.
- **Across models.** Never raw `D`, since tokenizers differ: per-byte damage, ratio to
  English, COMET and human loss, rank correlation of orderings. The core is one model;
  others replicate the ordering or fail to.
- **Against the literature.** Same model, method and language: our loss next to theirs
  under their normalisation.

## 6. Every results table

Per language: full-precision COMET and loss; human score and loss where rated; `D(L, c)`;
per-byte damage; flip rate; KL; paired gap to English; between-draw SD; damage-to-COMET
and damage-to-human correlation. All with intervals. No drop without its baseline.
