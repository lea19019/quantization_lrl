# Explanations for the gap

Candidate explanations for the problem in [../project/problem.md](../project/problem.md):
why rounding the weights removes more of a model's ability to predict the next word in
some languages than in others. "Damage" below means `D(L, c)`, per-word log-probability
loss, and "the gap" means `G(L, c)` as defined there. Each entry states the claim, what the
papers show, and what it predicts. Selection of which to test is pending; scope and
criteria are in [../project/project.md](../project/project.md). Page numbers are PDF pages.

## E1. Under-fitted languages

**Claim.** The weights sit at a minimum for the whole corpus, not for each language. A
language with a small share of the data is further from its own optimum: larger gradient
norm, sharper curvature. A fixed weight perturbation costs it more.

**Evidence.** What the papers show is that damage grows as full-precision quality falls
(Marie & Fujita 2025 p.5), and full-precision quality tracks data share. The direct
data-size correlation is weaker than it looks: Marchisio 2024 p.7 (R² 0.63 at W4) uses an
mC4 proxy over nine languages and its ordering changes under normalisation; Zeng 2024
p.1 shows it on BLOOM. Theory for groups under pruning (Tran 2022 p.4). Curvature drives
PTQ damage at model level (Catalan-Tatjer 2026 p.9). Against: Diddee 2022 p.9 finds no
data-size trend under int8; Soualhi 2026 p.3 finds English hurt as much as Hindi.

**Predicts.** Per-language gradient norm at the released checkpoint predicts damage,
and does so after conditioning on full-precision loss. Calibration-free RTN shows the
same language ordering as GPTQ. In the from-scratch model, damage tracks the data share we
set. If false: gradient norm adds nothing once full-precision loss is known.

## E2. Uncertain positions

**Claim.** Rounding noise flips predictions where the next-token distribution is flat
(small top-1 minus top-2 margin, high entropy). Low-resource text is assumed to have more
such positions; that is itself unmeasured.

**Evidence.** Within English: Proskurina 2024 p.4 (low-confidence samples move most), Lotfi
2026 p.2 (KL after quantization tracks entropy, Spearman 0.92). Against: Chang 2025 p.9
finds the inputs that break are the well-modelled, low-loss ones.

**Predicts.** One flip-rate-vs-margin curve shared across languages; the gap disappears
after conditioning on margin. If false: the gap persists at equal margin.

## E3. Calibration coverage

**Claim.** Calibration data (usually English web text) under-covers the activation range a
target language produces at inference. GPTQ then compensates error for the wrong input
covariance; AWQ picks scales from the wrong magnitudes; SmoothQuant clips. The grid itself
is set from the weights and does not depend on calibration (Frantar p.3, p.6).

**Evidence.** Multilingual calibration lowers perplexity, GPTQ far more than AWQ (Chimoto
2026 p.8; Zeng 2024 p.1). Effect collapses at 4 bits and above (Marie & Fujita 2025 p.5;
Wang 2025 Table 5; Borgersen 2025 null at 70B). Chimoto's best set is English code and
math mixed with ten languages, not the target language.

**Predicts.** Within one method, bit width and group size, target-language calibration
reduces that language's damage relative to English calibration. Clipping rate of
test-time activations against the calibration range is measurable and predicts damage.
RTN is the floor: what remains under RTN is not this. Effect grows at 3 bits.

## E4. Tokenization and vocabulary

**Claim.** A language's cost per unit of meaning (token premium against English on parallel
text) and the share of embedding rows it owns set how much of the model it touches per
word and how exposed it is when embeddings and the output head are quantized.

**Evidence.** Premiums are real and partly intrinsic to script and whitespace (Petrov 2023
p.7; Ahia 2023 p.2; Arnett 2025 p.8). Embeddings are most of a multilingual encoder's
parameters and pruning them "can completely negate" the other findings (Ogueji 2022 p.1;
Limisiewicz 2023 p.1). Never measured against quantization. Continuation pieces are the
highest-probability predictions, so on a margin argument they should be the most robust.

**Predicts.** Damage changes when embeddings and lm_head enter or leave the quantized
set. Within a language, word-initial and continuation pieces take different damage under
a word-level metric. Same language under two tokenizers: damage follows the tokenizer.
Measure token premium on FLORES-200, not tokens per word alone.

## E5. Where the damage lives

**Claim.** Damage is concentrated in specific parameters and directions: MLP
down-projections and their outlier weights, a few residual dimensions that carry
language identity or a frequency prior, particular experts in a mixture-of-experts. Which
of these a language depends on decides its loss, and there is a cliff near 2 bits below
which every language collapses.

**Evidence.** Patching MLP down-projection outputs recovers most of the loss (Chang 2025
p.8); outlier weights sit in down-projections of a few layers (An 2025 p.4); language
lives in sparse dimensions (Zhong 2025 p.1); outlier dimensions favour frequent tokens
and anisotropy grows as data shrinks (Puccetti 2022 p.1; Hämmerl 2023 p.6); language-
specific experts in NLLB (Koishekenov 2023 p.2); two failure regimes (SignalDegradation
2026 p.2). Residual-stream norm predicts which English inputs break (Chang 2025 p.1).

**Predicts.** Per-module quantization locates each language's loss; keeping
down-projections in higher precision closes the gap more than keeping attention. Per-
language residual norm, anisotropy, or expert overlap predicts damage. The gap shrinks
above the cliff and vanishes below it.

## E6. Generation-side failure

**Claim.** The model keeps the knowledge but fails in decoding: it switches language,
repeats, or emits invalid answer tokens, and errors compound over the extra steps a
fragmented language needs.

**Evidence.** Quantization raises off-target rate 5% to 17.5% in M2M-100 (Mohammadshahi
2022 p.6); sequential tasks degrade most (Wang 2025 p.7); sub-random scores mean invalid
tokens (Soualhi 2026 p.2); pruned models loop (Lee 2026). Against: Marchisio 2024 p.6
finds no language-confusion increase.

**Predicts.** Gap on generative tasks, none on constrained-choice tasks. Off-target rate
by language identification on outputs. Constrained decoding recovers most of the loss.

## E7. The null: model, format, metric and seed

**Claim.** Much of the reported gap is variance in checkpoint, quantization format, bit
width, benchmark composition, metric normalisation and calibration draw, not language.

**Evidence.** "Quantization impact tracks architecture and format more than it tracks
language" (Hossain 2026 p.7). Ten English calibration draws from one source give
66.7% ± 4.7 on BoolQ and a 0.9 to 1.6 point range on GPTQ 4-bit (Williams 2024 p.6, p.7);
most published 4-bit per-language gaps are smaller than that and none reports it. The
seventeen gap papers use nine definitions of damage, and the published orderings change
under normalisation: Marchisio's Latin vs non-Latin gap is 0.50, 0.18 or 2.06 points by
denominator; Soualhi's Yoruba is least or most damaged by denominator; Marie & Fujita's
Zulu "improves" at 2 bits from a COMET floor. Parametric recall degrades before in-context
processing (Jin 2024 p.1), so the gap depends on what the benchmark taxes.

**Predicts.** In a design crossed over two model families, two formats, five calibration
draws, with damage under the definition in problem.md §3, the language main effect
shrinks or vanishes at 4 bits. This must be beaten before any other explanation is
claimed. If false: the language effect exceeds the draw spread under every normalisation.
