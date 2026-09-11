# Pruning

Background only. The long-tail line began with pruning, and SparseGPT shares GPTQ's Hessian machinery. Back to [../papers.md](index.md).

### Frantar 2023. SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot
- File: `papers/Frantar-2023-SparseGPT.pdf`
- Venue: ICML 2023 (PMLR; the PDF is arXiv:2301.00774 v3, no venue printed)
- Does: One-shot post-training pruning of GPT-scale models by solving a layer-wise sparse regression with the OBS update and a shared sequence of inverse Hessians (same column-by-column machinery as GPTQ), using 128 C4 calibration segments and no retraining.
- Finds: OPT-175B and BLOOM-176B reach 50-60% unstructured sparsity with negligible WikiText2 perplexity increase, while magnitude pruning collapses past 10-30%; 4:8 and 2:4 cost only +0.11 and +0.39 perplexity at 175B; joint 50% sparsity + 4-bit weights is nearly lossless on OPT-175B; larger models are easier to prune.
- Use here: Background only; the Hessian-based error-compensation and calibration setup is shared with GPTQ, so any calibration-language effect found for GPTQ plausibly transfers.
- Note: Evaluation is English-only (WikiText2, PTB, C4, English zero-shot tasks); nothing on languages.

### Sun 2023. A Simple and Effective Pruning Approach for Large Language Models (Wanda)
- File: `papers/Sun-2023-Wanda-Simple-Effective-Pruning.pdf`
- Venue: ICLR 2024 (arXiv:2306.11695)
- Does: Prunes LLM weights by the score |W_ij| * ||X_j||_2 (weight magnitude times input-activation norm from calibration data), compared per output row, with no weight update; shown to be a diagonal-Hessian reduction of the SparseGPT metric.
- Finds: At 50% unstructured sparsity LLaMA-7B reaches 7.26 WikiText perplexity vs 7.22 for SparseGPT and 17.29 for magnitude; 50%-sparse LLaMA-65B/LLaMA-2-70B match dense zero-shot accuracy; metric computation is about 300x faster than SparseGPT and robust even with a single calibration sample.
- Use here: Background only; the key idea that outlier input features drive which weights matter links pruning saliency to the activation-outlier story used in quantization.
- Note: Filename says 2023 but the PDF header is "Published as a conference paper at ICLR 2024".

### Behnke 2020. Losing Heads in the Lottery: Pruning Transformer Attention in Neural Machine Translation
- File: `papers/Behnke-2020-Losing-Heads-Lottery-Pruning-Attention-NMT.pdf`
- Venue: EMNLP 2020
- Does: Applies lottery-ticket-style iterative pruning early in training to remove whole attention heads from transformer NMT models, using Voita et al.'s attention-confidence heuristic; tests Turkish-English (low-resource, transformer-big) and English-German (high-resource student model).
- Finds: Up to 72% of all heads can be removed from the Turkish-English model with about -0.1 BLEU on average and 1.5x faster inference; the English-German student loses 0.2 BLEU at 75% encoder-head sparsity; reinitialised models with the same structure are consistently worse.
- Use here: Background only; an early NMT example where a low-resource pair (tr-en) was pruned without a big loss, but it is a training-time method, not post-training, and does not compare languages.
- Note: Speed gains come from removing heads outright; the paper does not report per-language degradation comparisons.

### Dong 2024. Pruner-Zero: Evolving Symbolic Pruning Metric from scratch for Large Language Models
- File: `papers/Dong-2024-Pruner-Zero.pdf`
- Venue: ICML 2024 (PMLR 235; arXiv:2406.02924)
- Does: Uses genetic programming over weights, gradients and activations to search for a symbolic post-training pruning metric (found: ||W|*|W|| * min-max-scaled |G|), evaluated by WikiText2 perplexity on LLaMA-2-7B in under 5 minutes per candidate.
- Finds: At 50% unstructured sparsity, LLaMA-7B perplexity 6.95 vs 7.22 (SparseGPT) and 7.26 (Wanda); gains are consistent on LLaMA/LLaMA-2 and on 4:8 and 2:4 patterns, with no weight update.
- Use here: Background only; shows the pruning-metric design space, not relevant to language disparity.
- Note: Requires gradients from 128 calibration samples, so it is more expensive than Wanda.

### Kim 2024. Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods
- File: `papers/Kim-2024-Shortened-LLaMA-Depth-Pruning.pdf`
- Venue: preprint, arXiv:2402.02834 (v2, Jun 2024; no venue printed)
- Does: Removes whole Transformer blocks from LLaMA-7B and Vicuna-7B/13B chosen by a calibration-set perplexity or Taylor criterion (keeping first four and last two blocks), then compares LoRA vs continued pretraining (CPT) for recovery and measures small-batch latency.
- Finds: Depth pruning gives real latency gains where width pruning (LLM-Pruner, FLAP, Wanda-sp) does not at small batch sizes; with LoRA it matches width pruning at moderate ratios (about 20%), and under aggressive pruning (>40% removed, under 3.7B params) only CPT restores fluent generation.
- Use here: Background only; relevant mainly as a reminder that structured compression needs retraining to recover, which PTQ in this project does not do.
- Note: Results are on English benchmarks; text excerpt tables were partly garbled so exact averages are not quoted here.

### Lee 2026. FOCUS & RePAIR: Mitigating Text Degeneration via Token-Level Guidance for Pruned Large Language Models
- File: `papers/Lee-2026-FOCUS-RePAIR-Pruned-LLM-Degeneration.pdf`
- Venue: ICML 2026 (PMLR 306; arXiv:2608.26676)
- Does: Analyses repetition-loop degeneration in pruned Llama models (25% width pruned with LLM-Pruner, LoRA-finetuned on Alpaca) as loop-entry plus loop-persistence, and proposes two post-pruning distillation objectives: FOCUS (reweight KD toward high-confidence teacher tokens) and RePAIR (margin loss on onset-centred positive/negative continuations).
- Finds: Pruning raises the repetition rate from 5.9% to 12.4% (width) and 15.4% (depth) under sampling and from 26.6% to about 63% under greedy decoding even when perplexity and accuracy look intact; on WikiText-103 with Llama-3.1-8B, RePAIR lowers CREP from 7.3 to 2.23 and raises MAUVE from 0.61 to 0.68 at near-unchanged perplexity.
- Use here: Reusable idea: perplexity and accuracy can hide generation failures after compression, so the project's evaluation should include a generation-quality or repetition check per language, not only perplexity.
- Note: Only English generation is evaluated; the fix is training-based, so it does not apply directly to a PTQ-only pipeline.

### Lu 2024. AlphaPruning: Using Heavy-Tailed Self Regularization Theory for Improved Layer-wise Pruning of Large Language Models
- File: `papers/Lu-2024-AlphaPruning-HeavyTailed-Layerwise.pdf`
- Venue: preprint, arXiv:2410.10912 (v1, Oct 2024; no venue printed)
- Does: Allocates non-uniform per-layer sparsity from a data-free weight statistic: the power-law exponent (PL_Alpha_Hill) of each layer's weight eigenvalue spectrum, giving less sparsity to more heavy-tailed (better-trained) layers; plugs into magnitude, Wanda and SparseGPT.
- Finds: On LLaMA-7B at 70% sparsity it cuts perplexity by 61.91 (Wanda) and 7.76 (SparseGPT) versus uniform; at 80% it beats OWL by 304.31 perplexity on LLaMA-7B with +4.6% mean zero-shot accuracy and 3.06x CPU speedup; shape metrics beat scale (norm) metrics for layer allocation.
- Use here: Background only; the per-layer "how well-trained is this layer" idea could inspire mixed-precision layer allocation, but nothing here is about languages.

### Siddiqui 2024. A deeper look at depth pruning of LLMs
- File: `papers/Siddiqui-2024-Deeper-Look-Depth-Pruning.pdf`
- Venue: TF2M workshop at ICML 2024 (arXiv:2407.16286)
- Does: Compares block-importance metrics (cosine, relative L1/L2, Shapley on loss/logits/MMLU) for dropping blocks from LLaMA-2-7B and Mistral-7B, splits blocks into attention vs feed-forward layers, and tests two cheap recovery methods (mean "emulated update" bias vs a low-rank adapter).
- Finds: Up to 33% of self-attention layers in Mistral-7B can be removed without MMLU loss; adaptive (Shapley) metrics trade one task off against another; GSM-8k degrades sharply even after removing a single block; the simple emulated update recovers up to 5% absolute MMLU and matches or beats the trained adapter.
- Use here: Reusable idea: aggregate metrics (MMLU) can stay flat while specific capabilities (GSM-8k) collapse, an analogue of the per-language uneven-damage effect this project studies.
- Note: 7B models only; the authors say task-specific metrics buy accuracy on the targeted task at the cost of others.

### Yang 2024. LaCo: Large Language Model Pruning via Layer Collapse
- File: `papers/Yang-2024-LaCo-Layer-Collapse-Pruning.pdf`
- Venue: preprint, arXiv:2402.11187 (v2, Oct 2024; no venue printed)
- Does: Training-free structured pruning that merges rear layers into an earlier layer by adding parameter differences (RDSC layer merge), accepting each merge only if last-layer representations on a few calibration sentences stay above a cosine-similarity threshold; tested on Llama2-7B/13B and bilingual Baichuan2-7B/13B.
- Finds: At 25-30% pruning LaCo keeps 80.28% (Llama2-7B), 85.21% (Llama2-13B), 73.26% (Baichuan2-7B) and 87.94% (Baichuan2-13B) of dense average score across 13 benchmarks, versus under 70% for LLM-Pruner and SliceGPT; pruned Llama2-7B has WikiText perplexity 13.93 vs 4.46 dense; pruning takes about 15 s.
- Use here: Background only; the Baichuan2 experiments use Chinese and English benchmarks (CMNLI, CHID, CMMLU, C3) but the paper does not break results down by language.
- Note: The Chinese/English benchmark mix could be re-analysed by language from Table 1 if a bilingual pruning data point is ever needed.

### Yin 2024. Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity
- File: `papers/Yin-2024-OWL-Outlier-Weighed-Layerwise-Sparsity.pdf`
- Venue: ICML 2024 (PMLR 235; arXiv:2310.05175, v4 dated Jun 2025)
- Does: Measures the Layerwise Outlier Distribution (fraction of activations more than M times the layer mean), shows SparseGPT and Wanda succeed partly because they preserve outliers, and sets per-layer sparsity inversely to each layer's outlier ratio (bounded by a hyperparameter lambda) on top of Wanda or SparseGPT.
- Finds: At 70% sparsity on LLaMA-7B, OWL lowers WikiText perplexity by 61.22 vs Wanda and 6.80 vs SparseGPT, with 2.6x CPU speedup in DeepSparse; gains grow as models shrink from 65B to 7B; brief LoRA on 30k C4 tokens cuts OWL+SparseGPT perplexity from 19.49 to 11.15 on LLaMA-7B.
- Use here: Reusable idea: layer-wise outlier density as a direct bridge between pruning damage and the activation-outlier story in PTQ; the LOD metric could be computed per language to test whether low-resource inputs activate outliers differently.
- Note: Outlier definition follows Dettmers et al. 2022 and is computed on English calibration data.

### Zhang 2024. Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models (RIA)
- File: `papers/Zhang-2024-RIA-Plug-and-Play-Pruning.pdf`
- Venue: ICLR 2024
- Does: Proposes Relative Importance and Activations (RIA): weight magnitude normalised by both its input-channel and output-channel sums, times the input-activation norm to the power 0.5, to avoid "channel corruption" (whole channels pruned away); plus a channel-permutation step so N:M patterns keep more important weights.
- Finds: At 50% unstructured sparsity LLaMA-7B perplexity is 7.12 vs 7.24 (SparseGPT) and 7.26 (Wanda), LLaMA-65B 4.38 vs 4.57; RI alone (no calibration data) can match SparseGPT; activation rankings are positively Spearman-correlated across WikiText2, C4 and PTB; 2:4 with permutation on LLaMA2-70B beats dense on 3 of 5 zero-shot tasks.
- Use here: Background only; the observation that activation outliers are stable across (English) calibration sets is a claim this project can test across languages.
- Note: Cross-dataset activation stability was checked only on English corpora.
