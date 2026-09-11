# Tokenizers, evaluation data, and NMT compression

E4 sources (fertility, token premium), FLORES-200, and the older machine-translation compression line. Back to [../papers.md](index.md).

### Rust 2021. How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models
- File: `papers/Rust-2021-How-Good-Is-Your-Tokenizer.pdf`
- Venue: ACL-IJCNLP 2021, pages 3118–3135
- Does: Compares mBERT with monolingual BERTs for 9 languages (ar, en, fi, id, ja, ko, ru, tr, zh) on NER, sentiment, QA, dependency parsing, POS; then pretrains new models for ar, fi, id, ko, tr on identical Wikipedia data with either the monolingual tokenizer or the mBERT tokenizer to separate tokenizer effect from data size. Defines, following Ács (2019) and computed on UD v2.6 treebanks: **fertility** = "the average number of subwords produced per tokenized word" (minimum 1 means every word is in the vocabulary); **proportion of continued words** = "the proportion of words where the tokenized word is continued across at least two sub-tokens (denoted by continuation symbols ##)", i.e. fertility measures how aggressively a tokenizer splits, continued-words measures how often it splits.
- Finds: mBERT fertility is lowest for English and much higher for ar, fi, ko, ru, tr. With data held fixed, the monolingual tokenizer beats the mBERT tokenizer in 38/48 task–model–language combinations; swapping only mBERT's tokenizer/embeddings improves it in 20/24 settings. Decreases in fertility and continued-word proportion correlate with downstream gains about as strongly as pretraining-corpus size (Spearman rho 0.28–0.69 by task).
- Use here: Defines the fertility measurement to use for E4; also the E1/E2-vs-E4 disentangling template (same data, different tokenizer).
- Note: Encoder models, no quantization; correlation analysis on few data points (authors' own caveat); test scores are from the best-of-3 dev initialization.

### Ahia 2023. Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models
- File: `papers/Ahia-2023-Do-All-Languages-Cost-The-Same-Tokenization.pdf`
- Venue: EMNLP 2023, pages 9904–9923
- Does: Measures fragmentation of the ChatGPT (gpt-3.5-turbo) and BLOOMZ tokenizers on the FLORES-200 dev set for 22 languages, defining fragmentation as average tokens per sentence on parallel text (explicitly instead of fertility, because many languages lack word tokenizers); links this to API cost, in-context-learning utility (XNLI, XFACT, XQUAD, XLSUM, CrossSum) and HDI. Also trains byte-level BPE tokenizers on parallel data, one language per script, vocab 5k–50k, to separate data-proportion from script effects.
- Finds: Non-Latin-script languages need up to 5x more tokens than English (Telugu, Amharic ~4–5x cost); Latin-script low-resource languages (Swahili is in the low-cost group in Fig. 5) fragment less than mid-resource non-Latin ones. Disparity persists even with content and vocabulary size controlled, so script/language properties matter, not only data share. High fragmentation leaves no room for demonstrations, lowering utility; cost–HDI Spearman about −0.41.
- Use here: E4 — supports reporting tokens-per-sentence on parallel FLORES-200 alongside fertility, and warns that Latin-script sw/yo/zu may fragment less than expected from data share alone.
- Note: Tokenizer statistics only; no model training or quantization; ChatGPT training data assumed to mirror CC100.

### Petrov 2023. Language Model Tokenizers Introduce Unfairness Between Languages
- File: `papers/Petrov-2023-Tokenizers-Introduce-Unfairness-Between-Languages.pdf`
- Venue: NeurIPS 2023 (37th Conference on Neural Information Processing Systems)
- Does: Defines **tokenizer parity/premium**: for a sentence and its translation, premium of A relative to B is |t(s_A)|/|t(s_B)|, measured on FLORES-200 (parallel, 200 languages) for GPT-2/RoBERTa, ChatGPT/GPT-4 (cl100k_base), FlanT5, non-English BERTs, multilingual XLM-R/NLLB/mT5/M2M100/BLOOM, and byte/char-level CANINE/ByT5.
- Finds: ChatGPT/GPT-4 premiums vs English: French 1.60, Bulgarian 2.64, Arabic 3.04, Shan 15.05; even multilingual tokenizers have premiums >2.5 for some languages; byte-level models still show >4x gaps because of UTF-8 widths. RoBERTa processing time is roughly linear in token count (Shan ~2x English). Argues fertility (tokens per word) is not comparable across languages and proposes parallel-corpus premiums instead.
- Use here: E4 — the token-premium metric (ratio to English on FLORES-200) is the cleanest cross-language fragmentation number; report it next to fertility for en, fr, sw, yo, zu.
- Note: Yoruba and Zulu appear only in Fig. 1/2 plots; their numeric premiums are not in the first 12 pages (full table is in Appendix C, not extracted). No quantization.

### Limisiewicz 2023. Tokenization Impacts Multilingual Language Modeling: Assessing Vocabulary Allocation and Overlap Across Languages
- File: `papers/Limisiewicz-2023-Tokenization-Impacts-Multilingual-LM-Vocabulary-Allocation.pdf`
- Venue: Findings of ACL 2023, pages 5661–5681
- Does: Proposes tokenizer metrics — vocabulary allocation via **average rank** (AR) and **characters per token** (CPT = corpus characters / token count), and vocabulary overlap via Jensen–Shannon divergence between per-language token distributions — for Unigram, BPE, and two merged-monolingual tokenizers (NoOverlap, TokMix); trains small XLM-R-style models (vocab 120k, CC-100 with alpha=0.25) on 6 then 20 languages (fr and sw included) and probes MLM, POS, NER, dependency labeling, NLI, retrieval.
- Finds: CPT correlates with word-level task scores (Spearman r > 0.65 for POS, NER, dep labeling) and negatively with MLM MRR (r < −0.9); BPE and TokMix allocate more vocabulary than Unigram and score higher on word-level tasks; overlap helps NER and sentence tasks within the same script but hurts POS/dep transfer. Results are means over 5 seeds (6-language set) or 3 seeds (20-language set) with std reported.
- Use here: E4 — CPT is a word-boundary-free fragmentation measure usable for sw/yo/zu; vocabulary allocation (how many embedding rows a language actually uses) is a candidate mediator for why quantization noise could hit low-resource languages harder.
- Note: Small encoder models with linear probes, not decoder LLMs; no quantization.

### Arnett 2025. Explaining and Mitigating Crosslingual Tokenizer Inequities
- File: `papers/Arnett-2025-Explaining-Mitigating-Crosslingual-Tokenizer-Inequities.pdf`
- Venue: NeurIPS 2025 (39th Conference on Neural Information Processing Systems); also arXiv:2510.21909 (24 Oct 2025)
- Does: Trains ~7,000 monolingual BPE and Unigram tokenizers for 97 languages with matched 300 MB training data and vocab sizes 8,192–262,144, and measures **corpus token count** (CTC) on FLORES-200 (token premium = CTC / English CTC). Tests data similarity, mean token length, whitespace proportion, byte-premium scaling, parallel training data, per-language "optimal" vocab size, and SuperBPE (merges across whitespace).
- Finds: Identically trained monolingual tokenizers still show wide token premiums, so premiums are not only a data-share artifact. Byte-premium scaling has no effect; BPE compresses best. Train–eval similarity R²=0.24, FLORES mean token length 0.17, whitespace proportion 0.16 (combined 0.30). Training on parallel data cuts CTC by ~1% only; per-language optimal vocab size sharply reduces CTC variance (F-test p<0.001); SuperBPE lowers CTC and variance at every vocab size. Explicitly rejects fertility because "wordhood" is ill-defined across languages.
- Use here: E4 — fragmentation is partly intrinsic to the language/script and whitespace pre-tokenization, so a fertility gap for sw/yo/zu cannot by itself be read as "under-representation"; report CTC/premium and fertility together and treat them as a language property, not a model defect.
- Note: No language models trained, so no link from compression to performance; FLORES is translated from English (translationese caveat, authors' own).

### NLLB Team 2022. No Language Left Behind: Scaling Human-Centered Machine Translation
- File: `papers/NLLB-2022-No-Language-Left-Behind.pdf`
- Venue: preprint (no venue printed; no arXiv id printed in the PDF — other papers cite it as arXiv:2207.04672)
- Does: Builds FLORES-200, NLLB-Seed, Toxicity-200, LASER3-mined bitext, and NLLB-200 (54.5B Mixture-of-Experts) plus 3.3B/1.3B dense and 1.3B/600M distilled models covering 202 languages; evaluates 40,602 directions with spBLEU/chrF++ and human evaluation. **FLORES-200**: 3,001 sentences from 842 Wikimedia articles (about one third each from Wikinews, Wikijunior, Wikivoyage), ~21 words per sentence, professionally translated from English into 204 languages with a 90% QA threshold; splits **dev (997)**, **devtest (1,012)**, **test (992, hidden)**. Language codes listed in its Table 1 for this project's languages: **eng_Latn, fra_Latn, swh_Latn, yor_Latn, zul_Latn**.
- Finds: Claims +44% BLEU over the previous state of the art (40% on Flores-101); low-resource defined as <1M publicly available parallel sentences.
- Use here: Defines the evaluation benchmark and the spBLEU/chrF++ convention; use devtest for reported scores and keep dev separate (it is a natural non-English calibration source for E3).
- Note: 192-page report; the first 12 pages stop in Section 3, so FLORES details above come from pp. 13–22 via pdftotext; 44% BLEU claim is the abstract's summary, not a per-language number.

### Martins 2025. EuroLLM-9B: Technical Report
- File: `papers/Martins-2025-EuroLLM-9B-Technical-Report.pdf`
- Venue: preprint, arXiv:2506.04079 (v2, 16 Jun 2025)
- Does: Trains a 9B decoder LLM from scratch on ~4T tokens for the 24 EU languages plus 11 others, with a SentencePiece BPE (byte-fallback) tokenizer of 128k pieces; English share 50% in phase 1 and 32.5% later. Compares **fertility (tokens per word)** on FLORES-200 + Universal Dependencies against Mistral-7B (32k), LLaMa-3 (128,256), Gemma-2/Salamandra (256k), Teuken (250,680). Evaluates on EU20 benchmarks and WMT24++ with COMET-22 against Llama-3.1-8B, Qwen-2.5-7B, Gemma-2-9B and others.
- Finds: LLaMa-3 has lower fertility than EuroLLM for English but higher fertility for most other languages; EuroLLM's 128k vocab matches the 256k-vocab tokenizers' fertility with half the embedding parameters. EuroLLM-9B-IT beats all compared models on WMT24++ both directions (>3 COMET-22 points over Gemma-2-9B-IT); pre-trained EuroLLM-9B and Qwen-2.5-7B share the top Borda ranks on EU benchmarks.
- Use here: E4 — the Llama-3 tokenizer (same as Llama 3.1 8B) is documented to fragment non-English text more than English; also gives the fertility recipe (FLORES-200 + UD, tokens per word) to reuse. Background for the English-share (E1/E2) framing of pretraining mixes.
- Note: Per-language fertility values exist only in Figure 1 (not in the text extraction); sw, yo, zu are not EuroLLM languages, so no numbers for them here. No quantization.

### Diddee 2022. Too Brittle To Touch: Comparing the Stability of Quantization and Distillation Towards Developing Lightweight Low-Resource MT Models
- File: `papers/Diddee-2022-Too-Brittle-To-Touch-Quantization-Distillation-LowResource-MT.pdf`
- Venue: WMT 2022 (Seventh Conference on Machine Translation), pages 870–885
- Does: For 8 low-resource targets (Bribri 7K, Wixarica 8K, Mundari 10K, Gondi 26K, Assamese 140K, Odia 1M, Punjabi 2.4M, Gujarati 3M pairs; HRL→LRL), trains a from-scratch 6+6 Transformer and fine-tunes mT5-small, then compares hard sequence-level distillation against post-training quantization to int8 (weights + activations, ~3x memory reduction), reporting spBLEU and chrF2.
- Finds: PTQ of the pretrained mT5 stays within about 1 spBLEU of the uncompressed model for the lowest-resource languages (Bribri 6.4→7.4, Gondi 14.3→13.8, Odia 27.4→21.0), while int8 PTQ of the from-scratch Transformers collapsed: Odia 23.7→8.4 BLEU, Gujarati 35.9→16.0, Punjabi 38.4→19.1. Distillation is sensitive to student architecture, hyperparameters and pseudo-label quantity, with diminishing returns as teacher data shrinks.
- Use here: E1/E2 — closest precedent: models trained on thin data (from-scratch) lose far more under PTQ than pretrained multilingual models; directly motivates the from-scratch GPT-2 vs Llama/Qwen comparison.
- Note: Single runs, no CIs; int8 only (no 4-bit); encoder-decoder MT models; authors attribute the from-scratch collapse partly to overfit fine-tuning, so the E1/E2 reading is suggestive, not clean.

### Gumma 2023. An Empirical Study of Leveraging Knowledge Distillation for Compressing Multilingual Neural Machine Translation Models
- File: `papers/Gumma-2023-Knowledge-Distillation-Compressing-Multilingual-NMT.pdf`
- Venue: preprint (no venue printed; only a "© 2023 The authors, CC-BY-ND" line)
- Does: Sequence-level distillation of IndicTrans (474M, 11 Indic→English) on Samanantar into students from base (95M) to base24L, comparing word+sequence KD, batch/global selective distillation and a language-wise queue variant, plus depth vs width, recurrent stacking, high-quality-subset fine-tuning (LaBSE-filtered ~20%), and adapters; evaluated on Flores-101 dev (997) / test (1,012) with BLEU and chrF++.
- Finds: Every distilled base student stays ~3.1 BLEU below the teacher (28.3 vs 31.4 avg); KD variants are within 0.2 BLEU of each other and "statistically insignificant"; deeper thin models match wider ones with fewer parameters; HQ fine-tuning adds ~0.3 BLEU on average, most for Assamese and Odia; adapters give no gain.
- Use here: Background only (distillation, not quantization); useful as a precedent that low-resource languages respond most to data-quality interventions, and for the Flores dev/test size convention.
- Note: Single runs; COMET not used because unavailable for Indic languages; lists post-training quantization as future work.

### Koishekenov 2023. Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model
- File: `papers/Koishekenov-2023-NLLB200-Language-Specific-Expert-Pruning.pdf`
- Venue: preprint, arXiv:2212.09811 (v3, 7 Jul 2023)
- Does: Prunes experts of the 54.5B NLLB-200 MoE (12 MoE layers x 128 experts) at inference without fine-tuning, ranking experts by gate statistics (top-1 activity; importance = activity x exp(confidence)) collected on FLORES-200 dev, at global, language-pair, or per-language granularity; reports chrF++ (main) and spBLEU (sacrebleu tok:flores200) on FLORES-200 devtest for 30, 53 and all 202 languages, grouped high/low/very-low resource.
- Finds: 80% expert pruning (216 encoder / 72 decoder experts, fits one 32GB GPU) with per-language statistics loses ~0.2 chrF++ vs the full model and beats the 3.3B dense model by 0.8 chrF++ on 53 languages; global pruning is worse than the dense model. Decoder experts prune more aggressively than encoder; decoder experts are shared 68–87% within a target language vs 13–39% across. Pruning induces occasional over-generation, which spBLEU penalizes more than chrF++.
- Use here: Background on compressing the NLLB family; adopt its resource-tier reporting and its warning that spBLEU and chrF++ can diverge after compression because of over-generation, which is worth checking in quantized LLM outputs too.
- Note: Single model, no fine-tuning; per-tier averages without CIs in the main tables (std only in appendix Table 13).

### Aji 2020. Compressing Neural Machine Translation Models with 4-bit Precision
- File: `papers/Aji-2020-Compressing-NMT-4-Bit-Precision.pdf`
- Venue: WNGT 2020 (4th Workshop on Neural Generation and Translation), pages 35–42
- Does: 4-bit logarithmic (power-of-two) weight quantization of Transformer and deep-RNN NMT on WMT17 En→De, with a per-tensor scale fit by least squares, biases kept in fp32 (~0.2% of parameters), and optional re-training with an error-feedback residual; also 3/2/1-bit sweeps and 4-bit dot products.
- Finds: Baseline 35.66 BLEU. Pure post-training 4-bit log quantization with optimized scale and fp32 biases gives 34.31 (−1.35); re-training with error feedback recovers to 35.47 (−0.19) at 7.88x compression, while re-training without error feedback stays at 34.45; 4-bit fixed-point (with re-training) is 34.61 (−1.05). Quality falls to 34.95 / 33.40 / 29.43 BLEU at 3 / 2 / 1 bits; the RNN degrades less than the Transformer.
- Use here: Background/method — sets the expectation that plain 4-bit PTQ costs ~1 BLEU even for high-resource En-De and that non-uniform (log) grids fit the near-zero weight distribution better; useful reference point when reading RTN vs GPTQ/AWQ gaps.
- Note: Single high-resource pair, single runs; weights are decompressed to fp32 for inference; activations only quantized in a separate experiment.

### Prato 2020. Fully Quantized Transformer for Machine Translation
- File: `papers/Prato-2020-Fully-Quantized-Transformer-MT.pdf`
- Venue: Findings of EMNLP 2020, pages 1–14
- Does: FullyQT — uniform quantization-aware training of every matmul, attention/softmax, feed-forward and LayerNorm activation of the Transformer with per-output-row bucketing and straight-through estimation, at 8/6/4 bits, on WMT14 En-De, En-Fr, En-Cs, Ru-En, Es-En; compares against naive "quantize everything" and against post-training quantization (weights frozen, ranges estimated over a few hundred steps).
- Finds: 8-bit QAT matches or beats fp32 (base En-De 26.38 vs 26.46; En-Fr 38.41 vs 38.34; better in 21 of 35 experiments; 3.91x compression). Post-training 8-bit is essentially free (26.44 / 38.30) but post-training 6-bit drops to 24.84 vs 26.98 for 6-bit QAT, and 4-bit QAT collapses (18.32 En-De, 1.59 En-Fr). Naive full quantization diverges (nan) because of LayerNorm denominators. Base results averaged over 5 trials (std 0.09–0.51).
- Use here: Background — calibrates expectations: 8-bit is harmless, 6-bit PTQ already hurts, 4-bit needs more than uniform grids; the LayerNorm/softmax fragility is the same activation-outlier problem SmoothQuant targets.
- Note: High-resource pairs only; big-model rows are single runs; the 4-bit collapse is under QAT, not PTQ, so it is a lower bound on PTQ difficulty.
