# Architecture and training recipe of the from-scratch model

Sources for the phase-2 decoder's components, shape, schedule and token budget
([../project/pretraining_model.md](../project/pretraining_model.md)). Back to [../papers.md](index.md).

### Grattafiori 2024. The Llama 3 Herd of Models
- File: `papers/Grattafiori-2024-Llama-3-Herd-Of-Models.pdf`
- Venue: technical report, arXiv:2407.21783 (v3, 23 Nov 2024; no venue printed; author line "Llama Team, AI @ Meta")
- Does: Data, architecture, scaling laws, pre-training, post-training and FP8 inference of Llama 3 (8B, 70B, 405B); results are for Llama 3.1. A dense Transformer "not deviat[ing] significantly from Llama and Llama 2", changed only by GQA with 8 KV heads, a mask blocking attention across documents in one sequence, a 128K vocabulary and RoPE base 500,000.
- Finds: Table 3, 8B: 32 layers, d_model 4,096, FFN 14,336, 32 heads / 8 KV heads, peak LR 3e-4, SwiGLU, vocab 128,000, RoPE theta 500,000. Tokenizer: 100K tiktoken tokens plus 28K for non-English; English 3.17 to 3.94 chars/token vs Llama 2. Mix about 50% general, 25% math/reasoning, 17% code, 8% multilingual. 405B schedule: AdamW, 8,000 warmup steps, cosine 8e-5 to 8e-7, final linear anneal to 0 over 40M tokens with checkpoint averaging. FP8 (§6.2): attention, first and last layers kept in high precision; high-perplexity tokens give large activations, so scales are capped at 1200; benchmarks "do not adequately reflect the effects of FP8 quantization".
- Use here: Phase-1 Llama 3.1 8B is the template (GQA, SwiGLU, RoPE, AdamW, untied, no biases); its ratios (FFN/d_model 3.5, 4 query heads per KV head) are the reference. The FP8 notes are first-party evidence for E5/E7.
- Note: The paper does not state norm type, Pre-norm, biases or tying (these come from Llama 1/2 and the released configs). Cosine plus a short anneal, not WSD. No per-language token counts.

### Qwen Team 2025. Qwen2.5 Technical Report
- File: `papers/Qwen-2025-Qwen2.5-Technical-Report.pdf`
- Venue: technical report, arXiv:2412.15115 (v2, 3 Jan 2025; no venue printed)
- Does: The Qwen2.5 dense models (0.5B–72B) and MoE API models: architecture, 18T-token pre-training, hyperparameter scaling laws, long context, SFT and RL.
- Finds: GQA, SwiGLU, RoPE, QKV bias, Pre-RMSNorm. Table 1: 0.5B 24 layers, 14/2 heads, tied; 1.5B 28, 12/2, tied; 3B 36, 16/2, tied; 7B 28, 28/4, untied; 14B–72B untied. Byte-level BPE, 151,643 regular plus 22 control tokens. Pre-training at 4,096, then 32,768 with RoPE base 10,000 to 1,000,000. Decontamination: LCS of 13+ tokens and 0.6 of the shorter sequence.
- Use here: Second phase-1 reference, same family as Llama and the from-scratch model; differs in QKV biases, 7:1 head ratio and 151.6K vocabulary. Small Qwen models tie; a tied model cannot quantize embeddings and lm_head separately in the per-module map.
- Note: No hidden or FFN sizes, schedule, optimizer or language count given.

### Zhang 2019. Root Mean Square Layer Normalization
- File: `papers/Zhang-2019-RMSNorm.pdf`
- Venue: NeurIPS 2019; arXiv:1910.07467
- Does: Drops LayerNorm's mean subtraction and normalises by the RMS alone, a_i / RMS(a) * g_i (Eq. 4); invariant to rescaling, not to shifts. Tested on NMT, reading comprehension, retrieval, CIFAR-10.
- Finds: Transformer-base WMT14 En-De: LayerNorm 26.6 / 27.7 BLEU, RMSNorm 26.8 / 27.7, 6.9% faster; speed-ups 7–64% across models. With init shifted to 0.2, LayerNorm becomes unstable and RMSNorm does not.
- Use here: RMSNorm in the from-scratch model, as in both phase-1 models. The gain g is a per-channel scale right before q/k/v and gate/up, so large gains mark outlier channels; such scales fold into the next linear layer (what AWQ-style smoothing relies on).
- Note: Pre-LLM experiments; no quantization. He 2024 finds outlier features appear under RMSNorm as under LN.

### Shazeer 2020. GLU Variants Improve Transformer
- File: `papers/Shazeer-2020-GLU-Variants-Improve-Transformer.pdf`
- Venue: preprint, arXiv:2002.05202
- Does: Replaces the FFN with gated variants without biases, FFN_SwiGLU(x) = (Swish_1(xW) ⊗ xV) W2 (Eq. 6), shrinking d_ff by 2/3 to keep parameters fixed; T5-base on C4, then GLUE, SuperGLUE, SQuAD.
- Finds: Log-perplexity at 524,288 steps: ReLU 1.677, GELU 1.679, GEGLU 1.633, SwiGLU 1.636. GLUE: ReLU 83.80, SwiGLU 84.36. SuperGLUE: ReLU 72.76, SwiGLU 74.56. No explanation offered ("divine benevolence").
- Use here: The gate/up/down layout of the from-scratch and phase-1 models; the 2/3 rule sets d_ff. The down-projection input gate ⊗ up is where activation spikes concentrate (Yang 2024, An 2025; E5).
- Note: English T5 encoder-decoder, one run per variant; no quantization.

### Su 2021. RoFormer: Enhanced Transformer with Rotary Position Embedding
- File: `papers/Su-2021-RoFormer-RoPE.pdf`
- Venue: arXiv:2104.09864 (v5, 8 Nov 2023; no venue printed); later Neurocomputing 2024 per a SciSpace listing, volume and DOI not confirmed
- Does: Rotates each 2-D pair of query/key dimensions by m·θ_i, θ_i = 10000^(-2(i-1)/d) (Eq. 15), so q_m·k_n depends only on n − m (Eq. 16); no parameters; proves long-term decay with distance.
- Finds: WMT14 En-De: Transformer-base 27.3, RoFormer 27.5 BLEU. BERT with RoPE converges faster; GLUE better on 3 of 6 tasks.
- Use here: Positions in the from-scratch and phase-1 models; the base (10,000 here, 500,000 Llama 3) is fixed and reported. No weights, so nothing to add to the per-module map.
- Note: Small, mostly encoder experiments; no quantization.

### Ainslie 2023. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- File: `papers/Ainslie-2023-GQA-Grouped-Query-Attention.pdf`
- Venue: EMNLP 2023 (ACL Anthology 2023.emnlp-main.298, confirmed by web search); arXiv:2305.13245v3
- Does: H query heads in G groups, each sharing one key and one value head (GQA-1 = MQA, GQA-H = MHA); uptrains T5 checkpoints by mean-pooling K/V heads.
- Finds: Inference time and average score: MHA-XXL 1.51 s, 47.2; MQA-XXL 0.24 s, 46.6; GQA-8-XXL 0.28 s, 47.1.
- Use here: Llama 3.1 8B (32/8) and Qwen2.5-7B (28/4) use GQA, so the from-scratch model does; each K/V weight serves several query heads, so rounding error there reaches more heads.
- Note: T5 uptraining, not GQA from scratch; English; no quantization.

### Xiong 2020. On Layer Normalization in the Transformer Architecture
- File: `papers/Xiong-2020-Layer-Normalization-Pre-LN.pdf`
- Venue: ICML 2020 (header prints "PMLR 108"; ICML 2020 is PMLR 119); arXiv:2002.04745
- Does: Mean-field comparison of Post-LN and Pre-LN at initialization; tests training without warmup on IWSLT14, WMT14 and BERT-base.
- Finds: Theorem 1: last-FFN gradient O(d·sqrt(ln d)) for Post-LN, O(d·sqrt(ln d / L)) for Pre-LN. Lemma 2: the Pre-LN hidden-state norm grows linearly with depth, (1 + l/2)d ≤ E||x_l||² ≤ (1 + 3l/2)d. Pre-LN BERT reaches Post-LN's 500k-step loss at 300k steps.
- Use here: Pre-norm in the from-scratch model, as in both phase-1 models. Lemma 2 is the quantization link: the residual stream is never normalised and grows with depth, which is where massive activations live (Sun 2024) and what predicts breaking inputs (Chang 2025, E5).
- Note: Analysis at initialization with simplifications; LayerNorm, not RMSNorm.

### Press 2017. Using the Output Embedding to Improve Language Models
- File: `papers/Press-Wolf-2017-Tying-Word-Embeddings.pdf`
- Venue: EACL 2017, Vol. 2, pp. 157–163 (ACL Anthology E17-2025, confirmed by web search); arXiv:1608.05859v3
- Does: Compares input and output embeddings of LSTM LMs and NMT; proposes tying them.
- Finds: Untied: "the only row that is updated in the input embedding is the row representing the current input word … vectors representing rare words are updated only a small number of times". PTB large LSTM 78.4 to 74.3 perplexity with tying (66M to 51M params); EN-FR NMT 33.13 to 33.46 BLEU with three-way tying.
- Use here: The case for tying, which the from-scratch model declines: untied matches the phase-1 models and lets the per-module map quantize embeddings and lm_head separately. Predicts (E1/E4) that input rows of tokens used only by small languages are rarely updated.
- Note: LSTMs and NMT up to about 168M params; no Transformers, no quantization.

### Liu 2024. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases
- File: `papers/Liu-2024-MobileLLM-Sub-Billion.pdf`
- Venue: ICML 2024 (PMLR 235); arXiv:2402.14905
- Does: Ablates design choices for 125M and 350M decoders (SwiGLU, depth vs width over 19 models, embedding sharing, GQA, layer sharing), Adam, cosine LR, 0.25T tokens for ablations and 1T for final models; Appendix A extends to 600M–1.5B; W8A8 PTQ on the 0.25T models.
- Finds: Deep-and-thin wins at fixed size; the optimal depth is "near 30 layers" below 1B. SwiGLU 42.6 to 43.9 average accuracy (125M). Embedding sharing saves 16M params (11.8%) for −0.2 accuracy. 16 query / 4 KV heads comparable to 16/16. MobileLLM-600M: 40 layers, 18/6 heads, dim 1152, FFN 3072, 603.1M params, average 54.3. W8A8 costs <0.5 points.
- Use here: The closest published shape to the ~580M from-scratch decoder; sets its depth (30 layers) and query:KV ratio.
- Note: English-only; one seed per configuration, many differences <1 point; no 4/3-bit or outlier statistics.

### Allal 2025. SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model
- File: `papers/Allal-2025-SmolLM2.pdf`
- Venue: preprint, arXiv:2502.02737 ("Preprint. Under review."; later venue not confirmed)
- Does: SmolLM2-1.7B (24 layers, dim 2048, FFN 8192, 32 heads, tied, RoPE θ=10,000, sequence 2048, 2M tokens per batch, 49,152 vocabulary) on about 11T tokens in four hand-rebalanced stages; AdamW (0.9, 0.95), WSD with 2,000 warmup steps, peak 5.0e-4, linear decay to 0 over the last 10%. 360M (4T tokens) and 135M (2T) with GQA, WSD 20% decay, LR 3.0e-3.
- Finds: Average score 55.50 → 56.76 → 57.47 over the stable stages and 60.24 after the decay; math and code gains come mostly in the decay with upsampled data.
- Use here: The WSD template for a small Llama-style model; stable-phase checkpoints are usable as learning-curve points. The decay data mix must stay the same as the stable mix, or per-language damage after decay mixes schedule and data (Catalan-Tatjer 2026).
- Note: English plus code and math; no quantization. Park 2025 Table 3 shows SmolLM2-1.7B near random (avg 26.2) at 4 bits.

### Team OLMo 2025. 2 OLMo 2 Furious
- File: `papers/OLMo-2025-2-OLMo-2-Furious.pdf`
- Venue: arXiv:2501.00656v3; the arXiv comment says a shorter version was accepted to COLM 2025 (proceedings page not found)
- Does: Fully open 7B/13B/32B (and 1B) models; §3 ablates stability changes: repeated n-gram filter, N(0, 0.02) init, RMSNorm, reordered norm (h = x + RMSNorm(Attn(x))), QK-norm, z-loss, no weight decay on embeddings, AdamW ε 1e-8, RoPE θ 5e5; cosine then a linear anneal to 0 on a mid-training mix, final models averaged ("soups").
- Finds: Reordered norm plus QK-norm cut the gradient spike score 0.108 to 0.069, neither alone suffices; the new init 0.40 to 0.03. After decay to 0, peak LRs 3e-4 to 12e-4 reach the same loss. z-loss weight given as 1e-4 (§3.3.3) and 1e-5 (Table 1).
- Use here: The stability options, each of which also changes activation statistics (norm gains on residual writes, bounded logits; He 2024 finds smaller ε increases outliers). The from-scratch model uses plain pre-norm without them, fixed before any kurtosis or massive-activation measurement.
- Note: Single-run ablations; no quantization or outlier measurements; English-centric data.

### Hägele 2024. Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations
- File: `papers/Hagele-2024-Scaling-Laws-Beyond-Fixed-Durations-WSD.pdf`
- Venue: NeurIPS 2024; arXiv:2405.18392
- Does: Compares cosine (to 10%) with constant LR plus cooldown (WSD), cooldown shapes and lengths, weight averaging and schedule-free AdamW on Llama-style 33M–360M models (SlimPajama, AdamW (0.9, 0.95), weight decay 0.1, clip 1.0), plus 1B on 100B tokens.
- Finds: Constant LR plus a 20% linear cooldown matches the best cosine, with an optimal LR about half the cosine one. A (1-sqrt) cooldown beats linear, more so for longer runs; gains plateau at about 20% cooldown. 1B / 100B tokens: cosine 46.26, (1-sqrt) 20% 46.23. Cooldowns branched from one run reproduce cosine scaling fits at about half the FLOPs.
- Use here: The learning-curve schedule: one constant-LR run with saved checkpoints, short cooldowns branched from chosen ones, so every point is quantized at the same schedule phase (the Catalan-Tatjer 2026 confound). Defaults: 20% cooldown, (1-sqrt) shape.
- Note: Loss and perplexity, English data; no quantization.

### Hoffmann 2022. Training Compute-Optimal Large Language Models
- File: `papers/Hoffmann-2022-Chinchilla-Compute-Optimal.pdf`
- Venue: NeurIPS 2022 (proceedings title "An empirical analysis of compute-optimal large language model training", confirmed by web search); arXiv:2203.15556v1
- Does: Over 400 models (70M to 16B+, 5B–500B tokens) on MassiveText; compute-optimal N and D by training-curve minima, IsoFLOP profiles and a parametric fit L(N,D) = E + A/N^α + B/D^β.
- Finds: N_opt ∝ C^a, D_opt ∝ C^b with a, b = 0.50/0.50, 0.49/0.51, 0.46/0.54: parameters and tokens scale equally, about 20 tokens per parameter. Approach 1: 400M params → 8.0B tokens, 1B → 20.2B. Intermediate points of a longer cosine overestimate loss.
- Use here: The budget reference: about 11–12B tokens for 580M parameters (derived), close to one pass over all the pretraining data. Also why points along one cosine run are not comparable; WSD cooldowns fix it.
- Note: English-dominated data; no quantization.
