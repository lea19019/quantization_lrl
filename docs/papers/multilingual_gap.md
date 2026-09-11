# The multilingual quantization gap

Evidence that PTQ and other compression hurt some languages more than others. Bears on E1/E2 and E3. Back to [../papers.md](index.md).

### Marchisio 2024. How Does Quantization Affect Multilingual LLMs?
- File: `papers/Marchisio-2024-Quantization-Multilingual-LLMs.pdf`
- Venue: Findings of EMNLP 2024, pp. 15928–15947 (ACL Anthology 2024.findings-emnlp.935; the PDF is the arXiv version, 2407.03211)
- Does: Quantizes Command R/R+ (35B/103B) and Aya 23 (8B/35B) with W8, W8A8, W8A8+SmoothQuant, W4 column-wise and W4 group-wise GPTQ (128 English calibration samples; bitsandbytes int8/NF4 for Aya) and evaluates 10 primary languages (up to 23 for Aya) on mMMLU, MGSM, FLORES, Language Confusion, plus LLM/RM-as-judge and human evaluation in fr/es/ja/ko/en.
- Finds: Non-Latin-script languages degrade more (103B W4: -0.7% Latin vs -1.9% non-Latin; 8B: -3.0% vs -3.7%); per-language degradation correlates with mC4 data size and the correlation strengthens with lower precision (R2 0.24 at W8 to 0.63 at W4); automatic metrics understate damage (fr -0.3%/ja -1.7% automatic vs -16.6%/-16.0% by humans at W4-g); math (MGSM) drops fastest (-13.1% for 35B W4-g); group-wise scaling and SmoothQuant reduce but do not remove the non-Latin gap.
- Use here: Main prior for E1/E2 (less training data -> more damage, sharper at 4-bit); defines the relative %-change-vs-FP16 damage metric and the W8A8/SmoothQuant comparison the project reuses.
- Note: Only model-supported, mostly high-resource languages (no sw/yo/zu); automatic scores averaged over 5 sampling runs but one quantization run per setting; no confidence intervals.

### Marie 2025. The Uneven Impact of Post-Training Quantization in Machine Translation
- File: `papers/Marie-Fujita-2025-Uneven-Impact-PTQ-Machine-Translation.pdf`
- Venue: preprint, arXiv:2508.20893 (v1, Aug 2025)
- Does: Quantizes Qwen3-1.7B/8B/32B, Llama-3.1-8B-Instruct and Llama-3.3-70B with AWQ, bitsandbytes NF4, GGUF k-quants (imatrix) and AutoRound at 4-bit and 2-bit, and scores WMT24++ translation (55 languages, COMET) with per-language reporting for ja, fr, pl, bn, ml, zu; also sweeps temperature/top-p and English-vs-Bengali imatrix calibration.
- Finds: Languages with the lowest FP16 score lose the most ("inverse-performance law"): Llama-3.1-8B NF4 into English loses 0.3 (fr) and 1.5 (ja) COMET but 7.7 (bn) and 9.7 (ml); Llama-3.3-70B loses 0.9 (pl) vs 6.2 (zu); at 2-bit Qwen3-8B GGUF drops about 2 COMET on ja/fr but 17 on bn/ml. Bengali-language calibration helps only at 2-bit (+3.1 COMET en->bn) and not at 4-bit (differences <=0.2).
- Use here: Direct evidence for E1/E2 on Llama 3.1 8B including Zulu; a weak-to-null result for E3 at 4-bit (calibration language barely matters for GGUF imatrix at 4-bit) that the project's GPTQ/AWQ calibration experiments should be compared against.
- Note: Single-run COMET scores, no CI (authors state this); calibration-language test is GGUF-only on one model; the Xx->En direction uses translationese sources.

### Chimoto 2026. Calibrating Beyond English: Language Diversity for Better Quantized Multilingual LLMs
- File: `papers/Chimoto-2026-Calibrating-Beyond-English-Quantized-Multilingual-LLM.pdf`
- Venue: EACL 2026 (Proceedings of the 19th Conference of the European Chapter of the ACL, Vol. 1 Long Papers, pp. 4822-4838)
- Does: Quantizes Llama3.1 8B-Instruct and Qwen2.5 7B-Instruct (plus BLOOMZ-7B1-MT in appendix) to 4-bit/g128 with GPTQ and AWQ using eight calibration sets (en, fr, sw, zh, xh, three multilingual mixes; also code/math additions) at a fixed token budget, and measures Wikipedia/C4 perplexity on en, fr, sw, zh, xh, st, zu, yo, ig, ha plus XNLI/XStoryCloze/Global-MMLU; inspects per-layer quantization error, activation ranges and inverse-Hessian shifts.
- Finds: Non-English and multilingual calibration beat English-only almost everywhere; largest average gain 3.52 ppl (Llama GPTQ, multi10); GPTQ is far more calibration-sensitive (swings up to 3.52 ppl) than AWQ (<=0.35 ppl), because AWQ keeps the same salient channels and only rescales them; language-matched calibration gives the best per-language AWQ result and xh calibration also helps zu and st. Two failure cases (fr under AWQ, sw under GPTQ on Llama) are traced to calibration activation ranges narrower than test-time ranges, which clips outliers.
- Use here: The closest prior for E3 on exactly the project's models (Llama 3.1 8B, Qwen 2.5 7B), quantizers (GPTQ, AWQ) and languages (sw, yo, zu); its activation-range and Hessian-distance analyses are templates for the project's activation-stats interpretability.
- Note: Perplexity is the main metric and downstream margins are small; one quantization run per setting (error bars are per-language SEs, not seed variance); calibration token budget differs between GPTQ and AWQ.

### Borgersen 2025. English K_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance
- File: `papers/Borgersen-2025-English-K-Quantization-Multilingual-Performance.pdf`
- Venue: preprint, arXiv:2503.03592 (v4 dated 22 Jan 2026; Springer LNCS-style template but no venue printed)
- Does: Quantizes Llama 3.3 70B with llama.cpp k-quants (Q4_K_S, Q3_K_S, Q2_K_S) using importance matrices built from English, machine-translated Norwegian, or Malayalam text, and evaluates MixEval (2000 multiple-choice + free-form items) in English and Norwegian with single-token constrained answers, temperature 0.
- Finds: No statistically significant difference between calibration languages on either test language (lowest MCQ p = 0.2373; free-form p-values that dip below 0.05 do not survive Bonferroni at 0.00417); Norwegian imatrix does not help Norwegian and may slightly hurt; all quants show only a slight monotone decline with bit width.
- Use here: A published null result for E3 at 70B with a Latin-script, English-adjacent language; useful as the counter-case the project must explain if it finds calibration-language effects at 7-8B for sw/yo/zu.
- Note: Single model at 70B, single run, only one non-English language (Norwegian, machine-translated eval and imatrix); p-values from a Monte Carlo binomial model rather than repeated quantization.

### Soualhi 2026. The Multilingual Quantization Tax: Structural Collapse and Typological Fragility in Edge SLMs
- File: `papers/Soualhi-2026-Multilingual-Quantization-Tax-Edge-SLMs.pdf`
- Venue: preprint, arXiv:2608.09941 (v1; independent researcher)
- Does: Compares bf16 vs calibration-free bitsandbytes NF4 for Gemma 4 E2B/E4B-it and Qwen 3.5 2B/4B (thinking disabled) on MMLU-ProX-Lite and Global PIQA (native and translated subsets) in en, ar, ru, zh, ja, hi, sw, yo, zero-shot via lm-eval-harness.
- Finds: Average absolute accuracy tax 4.56% (Gemma) vs 5.30% (Qwen); 2B models lose more than 4B (5.76% vs 4.10%, p = 0.036 paired over 224 subtasks); low-resource/non-Latin cases collapse below the 10% chance floor (Qwen 3.5 4B on Hindi and Yoruba at 3.6% already at bf16); Qwen 3.5 4B Swahili loses 1.0 point on native PIQA but 11.65 on translated PIQA; hard-science subjects lose about twice as much as soft-science ones; apparent post-quantization gains are within SE bounds.
- Use here: Supporting evidence for E1/E2 (weak-baseline languages including sw/yo collapse first) and a stated but untested link to E4 (author attributes Cyrillic/Devanagari fragility to subword over-fragmentation); its SE-bounding of small deltas is a cheap sanity check the project can copy.
- Note: Single run, tiny Global PIQA subsets (integer-percent accuracies suggest about 100 items), yo/hi baselines already at chance so their "tax" is noise; mechanistic claims about outlier weights and cross-lingual routing are hypotheses, not measured.

### Hossain 2026. Quantization Effects on Bangla Language Understanding in Large Language Models: A Systematic Evaluation
- File: `papers/Hossain-2026-Quantization-Effects-Bangla-NLU.pdf`
- Venue: preprint, arXiv:2608.24615 (v1)
- Does: Compares full precision vs off-the-shelf 8-bit checkpoints (Qwen2.5-7B-Instruct GPTQ-Int8, Llama-3.1-8B-Instruct GPTQ-Q8, GPT-OSS-20B GGUF-W8A16) zero-shot on five translated Bangla benchmarks (Bangla MMLU, CommonsenseQA-BN, OpenBookQA-BN, PIQA-BN, BoolQ-BN) via lm-eval-harness log-likelihood scoring.
- Finds: 8-bit GPTQ costs Llama and Qwen at most 1.5 points (some benchmarks tick up slightly); the GPT-OSS GGUF-W8A16 checkpoint loses 36.7-57.4% on reasoning tasks but only 5.6% on BoolQ-BN; reasoning/commonsense tasks degrade more than reading comprehension in all families.
- Use here: Background only; confirms that 8-bit is essentially lossless for Llama 3.1 8B and Qwen 2.5 7B even on a low-resource Indic language, so the project's damage signal must be sought at 4-bit and below.
- Note: Family, format and size are not crossed (GPT-OSS is the only GGUF, only 20B, only MoE, and its calibration data is undocumented), so the big drop cannot be attributed to language; single run (authors report <0.3% re-run variance), no CI.

### Zeng 2024. Multilingual Brain Surgeon: Large Language Models Can be Compressed Leaving No Language Behind
- File: `papers/Zhang-2024-Multilingual-Brain-Surgeon.pdf`
- Venue: preprint, arXiv:2404.04748 (v2, May 2025)
- Does: Proposes MBS: sample calibration data from many languages in proportion to their share of the pre-training mix (or equally), and applies it to GPTQ (3-bit, group 1024), SparseGPT and Wanda (50% sparsity) on BLOOM-560m/7b1 with CC-100 calibration (256 x 2048 tokens) and XL-Sum perplexity plus zero-shot tasks over 20 languages (incl. sw, yo, ig, bn, hi); also gives an OBS-style theory where each language contributes its own Hessian term H_n = X_n^T X_n.
- Finds: English-only calibration inflates perplexity most for the least-represented languages (mr, gu) while calibrating on Igbo barely hurts English; MBS cuts the average perplexity increase (e.g. BLOOM-7b1 GPTQ average XL-Sum ppl about 31.2 English-only vs about 24.3 MBS) and lifts average zero-shot accuracy (55.6% to 57.1%); damage is predicted by (1) a language's share of training data and (2) cosine similarity of its per-input-channel activation norms ||X||^2 to the calibration language's.
- Use here: Core prior for E3 (English calibration mis-fits other languages' Hessians/activation norms) and a ready-made similarity measure (cosine of per-channel activation-norm vectors) that the project can compute for en/fr/sw/yo/zu on Llama/Qwen.
- Note: Filename mismatch: first author is Hongchuan Zeng (SJTU), not Zhang; BLOOM only, 3-bit GPTQ only, one run per setting, no CI.

### Gurgurov 2025. On Multilingual Encoder Language Model Compression for Low-Resource Languages
- File: `papers/Gurgurov-2025-Multilingual-Encoder-Compression-Low-Resource.pdf`
- Venue: preprint, arXiv:2505.16956 (v2, Nov 2025)
- Does: Compresses mBERT and XLM-R-base into monolingual Maltese, Slovak and Swahili models via two-step knowledge distillation (halving layers), structured FFN pruning (3072 to 2048), hidden-size truncation (768 to 564/456/312) and vocabulary trimming to 40k, evaluated on topic classification, sentiment, NER and POS with adapters; no quantization.
- Finds: Up to 92% parameter reduction with average F1 drops of 2-10% (moderate) and 8-13% (maximum); at 92% compression Slovak (1032 MB adaptation data) drops 2.9%, Swahili (332 MB) 5.2%, Maltese (188 MB) 19.2%, i.e. degradation tracks how much language-specific data the teacher saw; NER on tiny training sets is most fragile, POS most robust; hidden-size truncation is the step that costs most.
- Use here: Background only (encoders, distillation/pruning, not PTQ); its data-size-vs-damage ordering is weak circumstantial support for E1/E2.
- Note: Only 3 languages; F1 averaged over 3 seeds; language-adapted teachers required, so "data size" here is fine-tuning data, not pre-training share.

### Ogueji 2022. Intriguing Properties of Compression on Multilingual Models
- File: `papers/Ogueji-2022-Intriguing-Properties-Compression-Multilingual.pdf`
- Venue: EMNLP 2022 (ACL Anthology; the PDF is the arXiv version, 2211.02738)
- Does: Applies iterative magnitude pruning (50-98% sparsity, with and without the embedding matrix) during mBERT fine-tuning for WikiAnn NER in 40 languages, in monolingual and multilingual fine-tuning, three seeds, and tests robustness with entity-replacement perturbations within language, script and family.
- Finds: Moderate sparsity (50-70%, embeddings kept dense) slightly improves F1 for 26/40 languages and for all three 100-example languages (yo, my, jv), but at 70-98% the lowest-resource, lowest-baseline languages/scripts/families lose the most, so extreme compression amplifies existing disparities; dense models collapse to near-0 F1 on perturbed entities while partially pruned models recover, unless embeddings are pruned too.
- Use here: Background for E1/E2 (compression harm scales inversely with resourcedness) and a reminder that mild compression can act as regularization; the "which module is pruned matters" finding motivates the project's per-module sensitivity analysis.
- Note: Pruning during fine-tuning of an encoder, not PTQ of a decoder; WikiAnn is silver-standard and noisy.

### Mohammadshahi 2022. What Do Compressed Multilingual Machine Translation Models Forget?
- File: `papers/Mohammadshahi-2022-Compressed-Multilingual-MT-Forget.pdf`
- Venue: preprint, arXiv:2205.10828 (v4, Jun 2023)
- Does: Applies post-training magnitude pruning (30%, 45%, per Transformer layer incl. embeddings) and 8-bit weight+activation PTQ (Wu et al. 2020 scheme, MSE calibration) to M2M-100 12B without fine-tuning, and evaluates 3,763 FLORES-101 directions (spBLEU, sentence ChrF), MT-Gender and DiBiMT.
- Finds: Average spBLEU barely moves (22.44 to 20.95 at 30% pruning, 22.31 quantized) but very-low/low-resource pairs, especially low-resource targets, drop sharply; losing sentences are driven by off-target output (quantization raises off-target rate from 5.2% to 17.5%) and hallucination (cross-attention alignment metric 1.96x baseline); some medium-resource pairs improve because compression removes memorized noisy samples; pruning amplifies gender bias (+67% at 45%) and both methods raise semantic-bias metrics even for high-resource languages.
- Use here: Background for E1/E2 (damage concentrates in under-represented pairs behind a flat average) and a source of per-sentence damage diagnostics (winning/losing sentences, off-target rate, attention-variance hallucination metric) that parallel the project's per-token damage measures.
- Note: Single model (M2M-100 12B), 8-bit only for quantization, one run, no CI; MT-specific.
