# Papers: index of everything in `papers/`

77 PDFs, all read (first 12 pages, more where a table was needed) on 2026-09-09. One
line per paper here; full entries (venue as printed, what it does, what it finds, how
this project uses it, caveats) are in the theme files in this folder. Codes E1–E7 refer to the explanations in
[../concepts/explanations.md](../concepts/explanations.md). Venues
are as printed on the PDF; where the PDF is an arXiv version of a published paper the
entry says so and names the source used to confirm the venue.

File naming: `FirstAuthor-Year-Short-Title.pdf`. Four files break the rule and should be
renamed (Adrian's call): `Zhang-2024-Multilingual-Brain-Surgeon` (first author is Zeng),
`Wang-2025-Task-Circuit-Quantization` (Xiao), `Amazon-2025-T5-Emergent-Outlier-Properties`
(Zhao, NAACL 2025), `NVFP4-2026-Outlier-Dynamics-Pretraining` (Dong and Fan, 2026).

## Read first

The nine the proposal rests on, in order:

1. Marchisio 2024, the gap with human evaluation.
2. Marie & Fujita 2025, the gap on 55 languages including Zulu.
3. Chimoto 2026, calibration language on exactly our models (E3).
4. Proskurina 2024, GPTQ hurts low-confidence inputs (E1/E2 mechanism).
5. Tran 2022, why under-represented groups sit near the decision boundary (E1/E2 theory).
6. Chang 2025, per-input predictors of breakage and the interpretability toolkit.
7. Rust 2021, defines fertility (E4).
8. Frantar 2023 (GPTQ) and Lin 2024 (AWQ), the methods to implement.
9. Catalan-Tatjer 2026, the learning-rate confound for the from-scratch arm.

## The multilingual gap ([papers/multilingual_gap.md](multilingual_gap.md))

| Paper | Venue | One line |
|---|---|---|
| [Marchisio 2024](../../papers/Marchisio-2024-Quantization-Multilingual-LLMs.pdf) | Findings EMNLP 2024 | Aya/Command quantized: damage grows as data share shrinks, worse for non-Latin scripts, humans see 16% where metrics see 1.7% |
| [Marie & Fujita 2025](../../papers/Marie-Fujita-2025-Uneven-Impact-PTQ-Machine-Translation.pdf) | preprint | 5 LLMs, 55 languages: 4-bit costs <2 COMET on fr/ja, 8–10 on bn/ml; Zulu included; calibration language helps only at 2-bit |
| [Chimoto 2026](../../papers/Chimoto-2026-Calibrating-Beyond-English-Quantized-Multilingual-LLM.pdf) | EACL 2026 | Multilingual calibration beats English on Llama-3.1-8B / Qwen2.5-7B; GPTQ far more calibration-sensitive than AWQ |
| [Borgersen 2025](../../papers/Borgersen-2025-English-K-Quantization-Multilingual-Performance.pdf) | preprint | Null result: imatrix language does not matter for GGUF k-quants at 70B (Norwegian) |
| [Soualhi 2026](../../papers/Soualhi-2026-Multilingual-Quantization-Tax-Edge-SLMs.pdf) | preprint | NF4 on Gemma 4 / Qwen 3.5 small models: weak-baseline languages (hi, yo) collapse first; n = 1 |
| [Hossain 2026](../../papers/Hossain-2026-Quantization-Effects-Bangla-NLU.pdf) | preprint | 8-bit is lossless on Bangla for Llama/Qwen; damage must be sought at 4-bit and below |
| [Zeng 2024 (file: Zhang)](../../papers/Zhang-2024-Multilingual-Brain-Surgeon.pdf) | preprint | Multilingual Brain Surgeon: per-language Hessians; damage predicted by data share and activation-norm similarity to the calibration language |
| [Gurgurov 2025](../../papers/Gurgurov-2025-Multilingual-Encoder-Compression-Low-Resource.pdf) | preprint | Encoder compression for mt/sk/sw: loss tracks language-specific data size |
| [Ogueji 2022](../../papers/Ogueji-2022-Intriguing-Properties-Compression-Multilingual.pdf) | EMNLP 2022 | mBERT pruning in 40 languages: moderate sparsity helps, extreme sparsity hurts the lowest-resource most |
| [Mohammadshahi 2022](../../papers/Mohammadshahi-2022-Compressed-Multilingual-MT-Forget.pdf) | preprint | Compressed M2M-100: flat average hides losses on low-resource pairs; off-target rate 5% to 17% |

## The long tail ([papers/long_tail.md](long_tail.md))

| Paper | Venue | One line |
|---|---|---|
| [Hooker 2019](../../papers/Hooker-2019-What-Do-Compressed-DNNs-Forget.pdf) | preprint | Pruning Identified Exemplars: compression keeps top-1 and fails on atypical, long-tail images |
| [Hooker 2020](../../papers/Hooker-2020-Characterising-Bias-Compressed-Models.pdf) | preprint | Same on CelebA: errors concentrate on under-represented attributes |
| [Ahia 2021](../../papers/Ahia-2021-Low-Resource-Double-Bind-Pruning-MT.pdf) | Findings EMNLP 2021 | Pruned NMT for yo/ha/ig: frequent sentences survive, rare ones do not; the "double bind" |
| [Tran 2022](../../papers/Tran-2022-Pruning-Disparate-Impact-Accuracy.pdf) | NeurIPS 2022 | Theorem: smaller groups have larger gradient norms and sit closer to the boundary, so pruning hurts them more |
| [Tropeano 2025](../../papers/Tropeano-2025-As-Easy-As-PIE-Pruning-Disagree.pdf) | preprint | First PIE study on text: disagreement is not tied to rare classes; longer, harder inputs are hit hardest |
| [Liebenwein 2021](../../papers/Liebenwein-2021-Lost-In-Pruning-Beyond-Test-Accuracy.pdf) | MLSys 2021 | Pruned nets match accuracy but lose robustness to shift first; "prune potential" |
| [Jaiswal 2024](../../papers/Jaiswal-2024-Compressing-LLMs-Truth-Rarely-Pure-LLM-KICK.pdf) | ICLR 2024 | LLM-KICK: knowledge tasks fail before perplexity moves; quantization milder than pruning |
| [Jin 2024](../../papers/Jin-2024-Cost-Of-Down-Scaling-Fact-Recall-Deteriorates.pdf) | ICLR 2024 | Fact recall dies at 30–40% sparsity while in-context learning survives to 60–70% |
| [Wang 2026](../../papers/Wang-2026-Through-Compressed-Lens-Quantization-Factual-Recall.pdf) | TrustNLP @ ACL 2026 | GPTQ/AWQ 4-bit cut factual recall on Llama-3-8B and Qwen2.5-7B; unsaturated relations degrade most |
| [Wu 2026](../../papers/Wu-2026-Asymmetric-Harms-LLM-Compression.pdf) | preprint | Head/middle/tail facts under 11 compressions: tail is not proportionally hit harder; models stay confident when wrong |
| [Gonçalves 2023](../../papers/Goncalves-2023-Model-Compression-Social-Bias-LLMs.pdf) | EMNLP 2023 | int8 PTQ lowers social-bias scores; compression is not uniformly harmful |
| [Ramesh 2023](../../papers/Ramesh-2023-Model-Compression-Fairness-Language-Models.pdf) | ACL 2023 | Compression effects on fairness vary by language and metric; European languages only |

## Mechanism and interpretability ([papers/mechanism.md](mechanism.md))

| Paper | Venue | One line |
|---|---|---|
| [Proskurina 2024](../../papers/Proskurina-2024-When-Quantization-Affects-Confidence-LLMs.pdf) | Findings NAACL 2024 | GPTQ 4-bit lowers confidence most where the full model was already unsure; English |
| [Chang 2025](../../papers/Chang-2025-Why-Some-Inputs-Break-Low-Bit-Quantization.pdf) | EMNLP 2025 | 50 method pairs agree on which inputs break; small late-layer residual norm predicts error; logit lens and patching |
| [Lotfi 2026](../../papers/Lotfi-2026-Quantized-Reasoning-Models-Think-Longer.pdf) | preprint | Per-position KL after quantization tracks full-model entropy at Spearman 0.92 |
| [Kumar 2025](../../papers/Kumar-2025-Scaling-Laws-For-Precision.pdf) | ICLR 2025 | Scaling law: PTQ damage grows with training tokens and shrinks with size |
| [Ouyang 2025](../../papers/Ouyang-2025-Low-Bit-Quantization-Favors-Undertrained-LLMs.pdf) | ACL 2025 | 1500+ Pythia checkpoints: same trend; "undertrained = robust" |
| [Catalan-Tatjer 2026](../../papers/Catalan-Tatjer-2026-Training-Dynamics-PTQ-Robustness.pdf) | ICLR 2026 | Damage is flat at constant LR and spikes during decay; the token-count trend is mostly a schedule confound |
| [Williams 2024](../../papers/Williams-2024-Calibration-Data-Pruning-Quantization.pdf) | NAACL 2024 | Calibration-set choice moves GPTQ zero-shot by 1–1.6 points, pruning by 2–5; English |
| [Timkey & van Schijndel 2021](../../papers/Timkey-vanSchijndel-2021-All-Bark-No-Bite-Rogue-Dimensions.pdf) | EMNLP 2021 | One to five rogue dimensions dominate cosine and norms; z-score per dimension first |
| [Zhong 2025](../../papers/Zhong-2025-Language-Lives-in-Sparse-Dimensions.pdf) | preprint, venue unverified | Language identity lives in ~400 sparse residual dimensions; overwriting them switches output language |
| [Dumas 2025](../../papers/Dumas-2025-Separating-Tongue-From-Thought.pdf) | ACL 2025 | Activation patching: language fixed by layer 12, concept by 16 in Llama-2-7B |

## Tokenizers, data, NMT ([papers/tokenizer_data_nmt.md](tokenizer_data_nmt.md))

| Paper | Venue | One line |
|---|---|---|
| [Rust 2021](../../papers/Rust-2021-How-Good-Is-Your-Tokenizer.pdf) | ACL 2021 | Defines fertility and proportion of continued words; tokenizer matters as much as data size |
| [Ahia 2023](../../papers/Ahia-2023-Do-All-Languages-Cost-The-Same-Tokenization.pdf) | EMNLP 2023 | Tokens per sentence on FLORES-200: up to 5× English; Latin-script LRL fragment less than expected |
| [Petrov 2023](../../papers/Petrov-2023-Tokenizers-Introduce-Unfairness-Between-Languages.pdf) | NeurIPS 2023 | Token premium vs English on FLORES-200: fr 1.6, ar 3.0, Shan 15; byte-level still 4× |
| [Limisiewicz 2023](../../papers/Limisiewicz-2023-Tokenization-Impacts-Multilingual-LM-Vocabulary-Allocation.pdf) | Findings ACL 2023 | Characters per token and vocabulary allocation; fertility is not the only tokenizer variable |
| [Arnett 2025](../../papers/Arnett-2025-Explaining-Mitigating-Crosslingual-Tokenizer-Inequities.pdf) | NeurIPS 2025 | 7,000 tokenizers, 97 languages: premiums are largely intrinsic to language and pre-tokenization |
| [NLLB Team 2022](../../papers/NLLB-2022-No-Language-Left-Behind.pdf) | preprint | FLORES-200: 3,001 sentences, dev 997 / devtest 1,012; codes eng_Latn, fra_Latn, swh_Latn, yor_Latn, zul_Latn |
| [Martins 2025](../../papers/Martins-2025-EuroLLM-9B-Technical-Report.pdf) | preprint | EuroLLM-9B report; Llama-3 tokenizer fragments non-English more than English; fertility recipe |
| [Diddee 2022](../../papers/Diddee-2022-Too-Brittle-To-Touch-Quantization-Distillation-LowResource-MT.pdf) | WMT 2022 | int8 PTQ is stable on pretrained mT5 but collapses from-scratch low-resource NMT |
| [Gumma 2023](../../papers/Gumma-2023-Knowledge-Distillation-Compressing-Multilingual-NMT.pdf) | preprint | Distilling IndicTrans; low-resource languages respond most to data quality |
| [Koishekenov 2023](../../papers/Koishekenov-2023-NLLB200-Language-Specific-Expert-Pruning.pdf) | preprint | Per-language expert pruning of NLLB-200; spBLEU and chrF++ diverge after compression |
| [Aji 2020](../../papers/Aji-2020-Compressing-NMT-4-Bit-Precision.pdf) | WNGT 2020 | 4-bit log quantization of NMT costs about 1 BLEU on En-De without retraining |
| [Prato 2020](../../papers/Prato-2020-Fully-Quantized-Transformer-MT.pdf) | Findings EMNLP 2020 | Fully quantized Transformer: 8-bit free, 6-bit PTQ hurts, 4-bit collapses |

## Methods ([papers/methods.md](methods.md))

| Paper | Venue | One line |
|---|---|---|
| [Frantar 2023 (file: 2022)](../../papers/Frantar-2022-GPTQ.pdf) | ICLR 2023 | GPTQ: column-by-column rounding with inverse-Hessian error compensation; act-order is in the code, not the paper |
| [Lin 2024 (file: 2023)](../../papers/Lin-2023-AWQ.pdf) | MLSys 2024 | AWQ: scale salient input channels by activation magnitude before rounding; Table 4 is the Llama-2-7B validation target |
| [Shao 2024](../../papers/Shao-2024-OmniQuant.pdf) | ICLR 2024 | OmniQuant; Table 1 is the best RTN/GPTQ/AWQ cross-check grid for Llama-1/2 |
| [Xiao 2023](../../papers/Xiao-2023-SmoothQuant.pdf) | ICML 2023 | SmoothQuant: migrate activation outliers into weights; Llama-2-7B W8A8 5.515 at alpha 0.85 |
| [Wei 2023](../../papers/Wei-2023-Outlier-Suppression-Plus.pdf) | EMNLP 2023 | Outlier Suppression+: shift then scale |
| [Bondarenko 2023](../../papers/Bondarenko-2023-Quantizable-Transformers.pdf) | NeurIPS 2023 | Outliers come from attention heads learning a no-op on delimiter tokens |
| [Chen 2024](../../papers/Chen-2024-PrefixQuant.pdf) | preprint | PrefixQuant: two tokens carry 95% of per-token 4-bit error; prefix them |
| [Lin 2024](../../papers/Lin-2024-DuQuant.pdf) | NeurIPS 2024 | DuQuant: normal vs massive outliers; rotations and permutations |
| [Liu 2024 (file)](../../papers/Liu-2024-SpinQuant.pdf) | ICLR 2025 | SpinQuant: learned rotations, then GPTQ |
| [Xiao 2025 (file: Wang)](../../papers/Wang-2025-Task-Circuit-Quantization.pdf) | preprint | Task-Circuit Quantization: keep 0.35% salient weights in 16-bit; calibration set dominates at 2–3 bits |
| [Zhou 2026 (file: SignalDegradation)](../../papers/SignalDegradation-2026-Two-Failure-Modes-Quantization.pdf) | preprint | Two failure modes on Llama-3.1-8B GPTQ: signal degradation at 4-bit, computation collapse at 2-bit |

## Outliers ([papers/outliers.md](outliers.md))

| Paper | Venue | One line |
|---|---|---|
| [Sun 2024](../../papers/Sun-2024-COLM-Massive-Activations-in-LLMs.pdf) | COLM 2024 | Massive activations: fixed dims, first token, act as biases and attention sinks |
| [An 2025](../../papers/An-2025-Systematic-Outliers-in-LLMs.pdf) | ICLR 2025 | Weight, activation and attention outliers align; all from softmax |
| [Hämmerl 2023](../../papers/Hammerl-2023-Anisotropy-Outliers-Multilingual-LMs.pdf) | Findings ACL 2023 | Per-language anisotropy in mBERT rises as data shrinks (en 0.49, sw 0.69); outlier dims differ by language |
| [Zhao 2025 (file: Amazon)](../../papers/Amazon-2025-T5-Emergent-Outlier-Properties.pdf) | NAACL 2025 | Outlier dimensions in T5 reach 10^5; zeroing four dims costs 15 points |
| [He 2024](../../papers/He-2024-Outlier-Features-Kurtosis.pdf) | NeurIPS 2024 | Kurtosis and max/median as outlier metrics; they track W8A8 error |
| [Liao 2024](../../papers/Liao-2024-Free-Lunch-Removing-Outliers-Pretraining.pdf) | preprint | Removing outliers in pretraining is not free for quality |
| [Macocco 2025](../../papers/Macocco-2025-Outlier-Dims-Across-Checkpoints.pdf) | preprint | Last-layer outlier dims encode a frequent-token prior |
| [Dong 2026 (file: NVFP4)](../../papers/NVFP4-2026-Outlier-Dynamics-Pretraining.pdf) | preprint | Outlier dynamics during 4-bit pretraining; hot channels settle early |
| [Pierro 2024](../../papers/Pierro-2024-Mamba-PTQ-Outlier-Channels.pdf) | ICML 2024 workshop | Mamba has outlier channels too |
| [Puccetti 2022](../../papers/Puccetti-2022-Outlier-Dimensions-Driven-by-Frequency.pdf) | Findings EMNLP 2022 | Outlier-dimension magnitude correlates with pretraining token frequency |
| [Yang 2024](../../papers/Yang-2024-Activation-Spikes-GLU-Variants.pdf) | preprint | Activation spikes at down-proj input on first-occurrence tokens hijack per-tensor scales |

## Pruning ([papers/pruning.md](pruning.md))

| Paper | Venue | One line |
|---|---|---|
| [Frantar 2023](../../papers/Frantar-2023-SparseGPT.pdf) | ICML 2023 | SparseGPT: GPTQ's Hessian machinery for pruning |
| [Sun 2023 (file)](../../papers/Sun-2023-Wanda-Simple-Effective-Pruning.pdf) | ICLR 2024 | Wanda: weight × activation norm, no update |
| [Behnke 2020](../../papers/Behnke-2020-Losing-Heads-Lottery-Pruning-Attention-NMT.pdf) | EMNLP 2020 | Pruning attention heads in NMT, tr-en |
| [Dong 2024](../../papers/Dong-2024-Pruner-Zero.pdf) | ICML 2024 | Pruner-Zero: searched pruning metric |
| [Kim 2024](../../papers/Kim-2024-Shortened-LLaMA-Depth-Pruning.pdf) | preprint | Shortened LLaMA: depth pruning needs retraining |
| [Lee 2026](../../papers/Lee-2026-FOCUS-RePAIR-Pruned-LLM-Degeneration.pdf) | ICML 2026 | Pruned models loop; perplexity hides generation failures |
| [Lu 2024](../../papers/Lu-2024-AlphaPruning-HeavyTailed-Layerwise.pdf) | preprint | AlphaPruning: per-layer sparsity from heavy-tailed spectra |
| [Siddiqui 2024](../../papers/Siddiqui-2024-Deeper-Look-Depth-Pruning.pdf) | ICML 2024 workshop | Aggregate metrics flat while GSM-8k collapses |
| [Yang 2024](../../papers/Yang-2024-LaCo-Layer-Collapse-Pruning.pdf) | preprint | LaCo: layer collapse; bilingual Baichuan not broken down by language |
| [Yin 2024](../../papers/Yin-2024-OWL-Outlier-Weighed-Layerwise-Sparsity.pdf) | ICML 2024 | OWL: layer-wise outlier density sets sparsity |
| [Zhang 2024](../../papers/Zhang-2024-RIA-Plug-and-Play-Pruning.pdf) | ICLR 2024 | RIA: activation rankings stable across English calibration sets |

## Not in the folder

Llama 3 report (arXiv:2407.21783), WikiText (Merity 2017), the Gemma 4 model card, Aya
model cards, Qwen 2.5 report. Cite from the web; add the PDFs if they get read closely.
