# Outliers and massive activations

Background for E3 and for the per-language activation statistics in [../design.md](../project/design.md) §5. Back to [../papers.md](index.md).

### Sun 2024. Massive Activations in Large Language Models
- File: `papers/Sun-2024-COLM-Massive-Activations-in-LLMs.pdf`
- Venue: COLM 2024 (footnote: "Published as a conference paper in First Conference on Language Modeling (COLM), 2024"); arXiv:2402.17762
- Does: Characterises "massive activations": a handful of scalar residual-stream values (>100 and ~1,000x the median) that sit in fixed feature dims and at the start token / first delimiter token, across LLaMA2, Mixtral, Phi-2, GPT-2, and ViTs.
- Finds: LLaMA2-7B has dims 1415/2533 with values ~2,500 vs median 0.2, appearing abruptly at layer 2 and persisting to layer 30; zeroing four of them sends WikiText PPL to inf, setting them to their mean changes nothing (they act as fixed biases). They attract attention (attention-sink mechanism); explicit attention bias k',v' removes them in GPT-2 training. Massive activations are distinct from Dettmers-style outlier features (10 and 25 such dims in 7B/13B, no overlap).
- Use here: Defines the token- and dim-level measurement (top-k magnitude vs median per layer) and the massive-activation criterion the project should reproduce per language; check whether the same dims/tokens carry the massive values for non-English input.
- Note: Only English inputs (RedPajama, 100 x 4,096 tokens); nothing cross-lingual.

### An 2025. Systematic Outliers in Large Language Models
- File: `papers/An-2025-Systematic-Outliers-in-LLMs.pdf`
- Venue: ICLR 2025 ("Published as a conference paper at ICLR 2025"); arXiv:2502.06415
- Does: Defines activation, weight and attention outliers (threshold tau=1000 x mean abs), maps where they sit in LLaMA2-7B, and argues they all arise from the softmax in self-attention and act as context-aware scaling factors.
- Finds: Weight outliers (down_proj, layer 2 and last two layers) align 100% in feature dim with activation outliers (dims 1415/2533); activation outliers align 95% in sequence position with attention outliers (start token, ".", "_"). In GPT-2 training, only attention variants with a context-aware scaling factor prevent outliers; that variant cuts AbsMax W8 PPL from 93.44 to 29.22 and 50%-sparse PPL from 7235 to 39.
- Use here: Supports the mechanism story that outlier channels are structural (fixed dims tied to down_proj weights), so their location should be language-independent while their magnitudes at low-semantic tokens may not be; use the tau definition as one of the outlier-channel metrics.
- Note: English only; mechanism experiments are on GPT-2 small models.

### Hammerl 2023. Exploring Anisotropy and Outliers in Multilingual Language Models for Cross-Lingual Semantic Sentence Similarity
- File: `papers/Hammerl-2023-Anisotropy-Outliers-Multilingual-LMs.pdf`
- Venue: Findings of ACL 2023 (ACL Anthology; the PDF is arXiv:2306.00458, no venue printed)
- Does: Measures outlier dimensions (3-sigma / 5-sigma on mean-pooled sentence embeddings) and anisotropy (mean cosine of random sentence pairs) in XLM-R, mBERT and a multilingual S-BERT, per language, on Tatoeba (36 languages) and Wikipedia (ar, en, es, su, sw, tr); then zeroes outliers or applies whitening/CBIE.
- Finds: Yes, it compares languages. XLM-R layer 8 has one dominant outlier (dim 588, mean -15.2, 0.77 of the cosine contribution); the largest outliers occur in all or most languages but the set of top-10 dims differs by language (18 distinct dims across 36 languages). Final-layer anisotropy per language (mBERT / S-BERT): en 0.49/0.17, es 0.56/0.19, tr 0.60/0.17, su 0.64/0.28, ar 0.65/0.21, sw 0.69/0.59, i.e. "anisotropy increasing roughly as training data size decreases" (XLM-R is ~0.996 for all). Removing 18 large dims gives +9.7 Tatoeba acc; whitening/CBIE +~20.
- Use here: Direct precedent for per-language activation statistics differing with pretraining data size; adapt its per-language outlier and anisotropy measurements to decoder LLM residual streams and relate them to PTQ damage.
- Note: Encoder models (XLM-R, mBERT) and sentence embeddings only; no quantization experiments; the data-size link is an observation on 6 languages, not a controlled test.

### Zhao 2025. On the Analysis and Distillation of Emergent Outlier Properties in Pre-trained Language Models
- File: `papers/Amazon-2025-T5-Emergent-Outlier-Properties.pdf`
- Venue: NAACL 2025 (Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the ACL: HLT, Vol. 1 Long Papers, pp. 8475-8507)
- Does: Extends the BERT/GPT outlier-dimension findings to T5 and Flan-T5 (Small to 11B), identifying outlier dims via LayerNorm gamma > 3 std, then proposes EOFD, a distillation loss that up-weights high-variance (outlier) dims.
- Finds: T5-11B activation outliers reach >10^5 (up to 150,000); magnitude grows with layer depth, not model size (T5-Large -> 3B -> 11B decreases at fixed depth). Encoder and decoder have different outlier dims. Zeroing 4 of 1024 dims in T5-11B drops zero-shot accuracy by 14.7 points absolute; muting 512 non-outlier dims in Flan-T5-XXL costs 0.64. EOFD-BERT6 scores 84.4 GLUE vs teacher 85.0.
- Use here: background only.
- Note: File is named by affiliation; real first author is Tianyang Zhao (AWS AI Labs / Amazon Alexa AI), 2025. Suggested rename: Zhao-2025-NAACL-Emergent-Outliers-T5.

### He 2024. Understanding and Minimising Outlier Features in Transformer Training
- File: `papers/He-2024-Outlier-Features-Kurtosis.pdf`
- Venue: NeurIPS 2024 ("38th Conference on Neural Information Processing Systems"); arXiv:2405.19279
- Does: Defines two scale-free outlier-feature metrics on the residual stream (kurtosis of per-neuron RMS activations, Eq. 1, and max/median ratio, Eq. 2) and studies which architecture and optimiser choices cause outlier features during training, up to 7B params.
- Finds: Outlier features emerge with any Norm (LN, RMSNorm, weight-less SRMSNorm), Pre- or Post-Norm; the unnormalised "Outlier Protected" block cuts peak kurtosis by up to 4 orders of magnitude at equal loss (1.2B, 7B: kurtosis <10 vs >600 for Pre-Norm). Larger LR and smaller Adam eps increase outliers; non-diagonal preconditioning (SOAP/Shampoo) reduces them. On OPT-125m, kurtosis tracks W8A8 error: Pre-LN+AdamW 25.6 kurtosis, PPL 16.0 -> 63.4 int8; OP+SOAP 1.2 kurtosis, 14.71 -> 14.87.
- Use here: Adopt its kurtosis and max/median metrics as the per-language activation statistics, since the paper shows they correlate with W8A8 PTQ error.
- Note: English/code training data only; kurtosis-PTQ link is shown on 125m models.

### Liao 2024. Is It a Free Lunch for Removing Outliers during Pretraining?
- File: `papers/Liao-2024-Free-Lunch-Removing-Outliers-Pretraining.pdf`
- Venue: preprint, arXiv:2402.12102 (no venue printed)
- Does: Tests whether Bondarenko et al.'s clipped softmax (outlier-free pretraining) hurts full-precision quality, and proposes a normalised clipped softmax (NCS) that is invariant to sequence length so it also works for causal models.
- Finds: Clipped-softmax BERT-base scores 68.1 vs 81.7 on GLUE (vanilla) despite lower MLM PPL; NCS lifts it to 73.8. Vanilla BERT-base W8A8 PPL is 4612.6 vs 4.93/4.95 with CS/NCS (max |X| 3857 vs 79, kurtosis 3932 vs 160). For OPT-125M NCS gives W8A8 PPL 18.33 vs 21.18 vanilla and 37.20 CS; kurtosis and max-norm do not correlate well with W8A8 for OPT-350M.
- Use here: background only.
- Note: Models <350M; quantization setting is stricter than standard PTQ (embeddings and norms quantized, no grouping).

### Macocco 2025. Not a nuisance but a useful heuristic: Outlier dimensions favor frequent tokens in language models
- File: `papers/Macocco-2025-Outlier-Dims-Across-Checkpoints.pdf`
- Venue: preprint, arXiv:2503.21718 (no venue printed)
- Does: Studies last-layer outlier dimensions (median activation in the top 1% of all |activations|) in eight 7B-14B decoder LLMs on WikiText-103, ablates them, decomposes logits into OD vs non-OD contributions, and tracks their emergence over Pythia-12b checkpoints.
- Finds: 4-38 ODs per model (z-scores 8.8-29); ablating them drops next-token accuracy e.g. pythia-12b 43.0 -> 34.3, qwen-14b 49.9 -> 32.2, stable-12b 49.4 -> 30.2, while random dims change <0.1. Keeping only ODs makes models predict a few frequent tokens (_the, _a, _in); ablation lowers the slope of prediction-frequency vs corpus-frequency (1.35 -> 0.74 for pythia-12b). ODs are boosted by the last MLP down-projection and final LayerNorm and appear around training steps 3000-4000. opt-13b and gemma-9b are exceptions (ODs contribute ~0 logit).
- Use here: Suggests the project's "rogue dimension" statistics in the last layer encode a frequent-token prior tuned to the (English-dominated) training distribution; compare OD activation levels and their logit contribution across languages.
- Note: Filename says "Across Checkpoints", but checkpoint analysis is a small final experiment on pythia-12b only; English only.

### Dong 2026. Dissecting Outlier Dynamics in LLM NVFP4 Pretraining
- File: `papers/NVFP4-2026-Outlier-Dynamics-Pretraining.pdf`
- Venue: preprint, arXiv:2602.02047 ("Preprint. February 3, 2026")
- Does: Tracks kurtosis, top-k magnitudes and flush-to-zero rates of activations and weights during NVFP4 (4-bit) pretraining of Qwen3 (softmax attention) and GLA/GSA/Gated-DeltaNet (linear attention) models, 340M-7B, then proposes Hot-Channel Patch and the CHON recipe.
- Finds: Outliers come from softmax (SA), the exp gating in gk_proj (LA, range about [-120, 80]) and SwiGLU with growing W_up/W_gate alignment; "post-QK" projections (W_v for SA, W_o for LA) are the most quantization-sensitive. Outliers drift across channels early (steps 400-5,400) then settle into a small fixed set of hot channels; kurtosis and magnitudes accumulate in the last ~4 layers. CHON cuts the NVFP4-to-BF16 loss gap on GLA-1.3B (60B tokens) from 0.94% to 0.58%.
- Use here: background only.
- Note: Filename is by topic; real first authors are Peijie Dong and Ruibo Fan (equal contribution), HKUST(GZ)/Alibaba, 2026. Suggested rename: Dong-2026-NVFP4-Outlier-Dynamics-Pretraining. About QAT/low-bit training, not PTQ.

### Pierro 2024. Mamba-PTQ: Outlier Channels in Recurrent Large Language Models
- File: `papers/Pierro-2024-Mamba-PTQ-Outlier-Channels.pdf`
- Venue: ICML 2024 Efficient Systems for Foundation Models Workshop (Vienna); arXiv:2407.12397
- Does: Checks whether Mamba (130m-2.8B) has activation outlier channels (abs-max > 6 std from layer mean, on WikiText-2) and reports naive per-tensor absmax PTQ results on six zero-shot tasks.
- Finds: The in, x and out projections have outliers in <1% of channels (in-proj outliers consistent across layers); zeroing them drops average accuracy by 12.6 (130m) and 17.5 (2.8B) points. W8 on all linears is nearly lossless for 1.4B, W4 collapses (LAMBADA 0%), W8A8 (all) loses ~10 LAMBADA points on 1.4B and ~18 on 2.8B.
- Use here: background only.
- Note: Preliminary 4-page workshop paper; no outlier-aware method actually evaluated beyond describing SmoothQuant.

### Puccetti 2022. Outlier Dimensions that Disrupt Transformers are Driven by Frequency
- File: `papers/Puccetti-2022-Outlier-Dimensions-Driven-by-Frequency.pdf`
- Venue: Findings of EMNLP 2022 (ACL Anthology; the PDF is arXiv:2205.11380, no venue printed)
- Does: Replicates the Kovaleva LayerNorm-outlier effect in BERT-base (dims 308, 381) and RoBERTa-base (77, 588), then correlates the hidden-state magnitude at those dims with pretraining-corpus token frequency, attention patterns, and a controlled pretraining experiment that changes the token frequency distribution.
- Finds: Exact claim: "the magnitude of hidden state coefficients corresponding to outlier dimensions correlates with the frequency of encoded tokens in pre-training data" (Pearson, Wikitext-2 vs BookCorpus+Wikipedia counts), much higher than random dims; for O381 the correlation and the removal damage both peak in layers 4-6, for O308 the correlation is high early but damage grows towards layers 10-11. Removing 308+381 drops MNLI 84.5 -> 58.4 and CoLA 56.9 -> 15.9. A BERT-medium pretrained with 50% of frequent tokens replaced by rare ones (SENTENCE_FREQ) develops no outlier as damaging as O378 in the normal models (removal cost 1-2.4 vs 12.7-31.8 MNLI points). Disabling outliers makes the MLM predict more high-frequency tokens.
- Use here: Core link for E1/E2/E3: outlier-dimension magnitude is a function of how often a token was seen in pretraining, so low-resource-language tokens (rare, fragmented) should sit at different positions along the outlier axes than English calibration text; measure this correlation per language.
- Note: BERT/RoBERTa encoders only; the authors say the correlation shows "mutual influence", not proven causation, and the controlled experiment also degrades the model.

### Yang 2024. Mitigating Quantization Errors Due to Activation Spikes in GLU-Based LLMs
- File: `papers/Yang-2024-Activation-Spikes-GLU-Variants.pdf`
- Venue: preprint, arXiv:2405.14428 ("Preprint. Under review.")
- Does: Calibrates on 512 C4 samples and measures per-module input-activation abs-max in LLaMA-2/3, Mistral, Mixtral, SOLAR, Gemma; finds "activation spikes" at the FFN down-projection input in early and late layers, on the first occurrence of a few tokens (BOS, "\n", "'"), and proposes QFeM (leave high max/median-ratio modules unquantized) and QFeP (precompute a 3-token prefix KV cache).
- Finds: Spikes reach ~1.5k (LLaMA-2-7B layer 2) and ~6k (70B layer 9) for one token, dominating per-tensor scales. Quantizing only the top-4 max/median-ratio modules raises LLaMA-2-13B PPL 6.84 -> 15.09 and Mistral 8.35 -> 69.45; middle/bottom-4 do nothing. Per-tensor W8A8 on LLaMA-2-13B gives PPL 34.09 / 55.3% avg acc; +QFeM+QFeP gives 5.13 / 71.6%. SmoothQuant alone fails on 13B (PPL 34.87) but works with QFeM (5.12).
- Use here: Directly relevant to the W8A8/SmoothQuant arm: calibration scales are hostage to first-occurrence spikes on a few tokens, so record per-language max/median token-scale ratios per module and check whether spike-trigger tokens differ by language or script.
- Note: Per-tensor dynamic activation quantization is the main setting; per-token quantization (as in SmoothQuant's own paper) largely avoids the spike problem (Fig. 7).
