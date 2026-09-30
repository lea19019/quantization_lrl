# Evaluation

What is reported as project evidence, what only supports interpretation, and what is used
internally to explain quantization. Data and splits are in [data.md](data.md).

## 1. Reported translation metrics

| Metric | Status | Languages | What the report uses it for |
|---|---|---|---|
| Human direct assessment | primary outcome | all | Translation adequacy judged by qualified speakers, 0–100 |
| chrF++ | primary automatic outcome | all | Reference overlap robust to inflection and spelling variation |
| AfriCOMET 1.1 | primary automatic outcome | English, French, Yoruba, Zulu | Learned reference-based translation quality |
| BLEU | secondary automatic outcome | all | Comparison with earlier machine-translation work |

Human evaluation carries the conclusion when it disagrees with an automatic metric.
Raters also mark off-target, unintelligible and empty output; no automatic language-ID
score substitutes for this because available classifiers do not cover every language.

Compute chrF++ and BLEU at corpus level with `sacrebleu==2.6.0` and save their complete
signatures. chrF++ is case-sensitive with character order 6, word order 2, beta 2 and no
whitespace characters. BLEU uses mixed case, `13a` tokenization and exponential smoothing.

Use author-released [`masakhane/africomet-stl-1.1`](https://huggingface.co/masakhane/africomet-stl-1.1/tree/b4065f4424a962cf6e52373a9ec8a22a624fb278),
pinned to revision `b4065f4424a962cf6e52373a9ec8a22a624fb278`. It scores `(source,
translation, reference)` from 0 to 1; tables multiply it by 100. No different COMET model
is substituted for unsupported languages.

For every metric, show the full-precision score, quantized score and absolute loss:
`full score - quantized score`. A positive loss means quantization made translation worse.

## 2. Supporting and internal measurements

These are not headline translation metrics. They appear in a supporting table, a
mechanism analysis or an appendix only when they answer a stated question.

| Measurement | Status | Origin and purpose |
|---|---|---|
| Per-language perplexity | supporting result | Standard language-model measure used by Chimoto; compare full vs quantized within one language, never raw values across languages |
| WikiText-2 perplexity | validation only | Pass/fail check that a quantizer reproduces published values; not a multilingual result |
| Gold-token damage, `D(L,c)` | internal primary diagnostic | Project adaptation of token negative-log-likelihood change; localizes the loss caused by quantization |
| Per-word damage | internal normalization | Project aggregation of token damage to reduce tokenizer-fragmentation bias |
| Per-byte damage | exploratory normalization | Project proposal with no direct paper precedent; a robustness check, not a reported outcome metric |
| Top-1 flip rate | internal diagnostic | Project disagreement check: whether the preferred next token changed |
| KL divergence | internal diagnostic | Full-vs-quantized distribution shift following Lotfi 2026 |
| Margin and entropy | internal controls | Full-model confidence controls motivated by Proskurina 2024 and Lotfi 2026 |
| Token fertility | internal control | Tokens per word, following Rust 2021; tests tokenizer fragmentation |

Perplexity and `D(L,c)` use the same gold-token probabilities. Perplexity summarizes how
well one model predicts the corpus; `D(L,c)` subtracts full-model loss from quantized-model
loss to isolate the change caused by quantization. Sources and their exact uses are in
[mechanism.md](../papers/mechanism.md),
[tokenizer_data_nmt.md](../papers/tokenizer_data_nmt.md) and
[multilingual_gap.md](../papers/multilingual_gap.md).

## 3. Metrics by language

`Yes` marks report metrics. Per-language perplexity is supporting for every row; internal
diagnostics are collected for every row when required by a mechanism analysis.

| Language | Code used here | Final evaluation text | Human | chrF++ | BLEU | AfriCOMET |
|---|---|---|:---:|:---:|:---:|:---:|
| English | ISO `eng`; FLORES `eng_Latn` | FLORES-200 devtest | Yes | Yes | Yes | Yes |
| French | ISO `fra`; FLORES `fra_Latn` | FLORES-200 devtest | Yes | Yes | Yes | Yes |
| Yoruba | ISO `yor`; FLORES `yor_Latn` | FLORES-200 devtest | Yes | Yes | Yes | Yes |
| Zulu | ISO `zul`; FLORES `zul_Latn` | FLORES-200 devtest | Yes | Yes | Yes | Yes |
| Spanish | ISO `spa`; FLORES `spa_Latn` | Church and pair-specific test sets | Yes | Yes | Yes | No |
| Portuguese | ISO `por`; FLORES `por_Latn` | Fixed church test | Yes | Yes | Yes | No |
| Q'eqchi' | ISO/MayanV `kek` | MayanV test | Yes | Yes | Yes | No |
| Guarani | ISO `grn`; AmericasNLP `gn` | AmericasNLP test | Yes | Yes | Yes | No |
| Nahuatl | collective/AmericasNLP `nah` | AmericasNLP test | Yes | Yes | Yes | No |
| Ecuadorian Quichua/Kichwa | Napo `qvo`; church variety unresolved | Fixed church test after variety audit | Yes | Yes | Yes | No |
| Bribri | ISO/AmericasNLP `bzd` | AmericasNLP test | Yes | Yes | Yes | No |

Do not label Ecuadorian Kichwa as FLORES `quy_Latn`: that code is Ayacucho Quechua.
AfriCOMET is restricted to the four-language core so every COMET comparison uses one
checkpoint and the same parallel FLORES design. Portuguese AfriCOMET may be a separately
labelled add-on, not part of the shared Latin-American evaluation.

## 4. Protocol and uncertainty

- Score each translation direction separately. Never pool directions or test corpora.
- African-core comparisons use the same 1,012 FLORES sentence indices. Latin-American
  corpora differ in content and domain, so their raw scores do not establish a clean
  cross-language ordering.
- Human evaluation uses 100 fixed test sentences per direction, two qualified raters per
  target language, blind and shuffled. Report the mean, 95% interval, loss from full
  precision, off-target rate and Krippendorff's alpha.
- Fix prompts, decoding and references before comparing precision configurations. No
  evaluation sentence may appear in training or prompts.
- Bootstrap 1,000 times over sentences for 95% percentile intervals. Full and quantized
  models are paired on identical sentences. A loss whose interval includes zero, or is
  smaller than calibration-draw variation, is unresolved.
- Raw perplexity is not compared across languages because tokenizer fragmentation changes
  its unit. Report each language's full and quantized perplexity and their change.

## 5. Presentation in the report

The main results table contains human, chrF++, BLEU and AfriCOMET where supported, always
with the full baseline, quantized score, absolute loss and confidence interval. A separate
supporting table contains per-language perplexity. WikiText-2 appears only in the method
validation section. Internal diagnostics appear only in the experiments that use them to
explain a result; they are not presented as translation-quality metrics.
