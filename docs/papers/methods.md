# Quantization methods

The four methods this project implements (RTN, GPTQ, AWQ, SmoothQuant), their three parents (OBS, OBC, the Qualcomm white paper), the outlier-handling methods that followed, the published perplexities to validate against, and plain-language algorithm sketches. Back to [../papers.md](index.md).

### Hassibi 1993. Second order derivatives for network pruning: Optimal Brain Surgeon
- File: `papers/Hassibi-1993-Optimal-Brain-Surgeon.pdf`
- Venue: NeurIPS 1992 (Advances in Neural Information Processing Systems 5, 1993; proceedings scan from proceedings.neurips.cc, no venue line printed)
- Does: Expands the loss at a trained minimum to second order, dE = 1/2 dw^T H dw (Eq. 1), and removes the weight whose removal costs least under the exact constrained optimum (Eq. 3-4): saliency L_q = w_q^2 / (2 [H^-1]_qq) and update dw = -(w_q / [H^-1]_qq) H^-1 e_q (Eq. 5), so every other weight is adjusted without gradient descent. Gives a recursion for H^-1 from training data (Eq. 15-17).
- Finds: On XOR and the MONK problems OBS prunes 90%, 76% and 62% of the weights where magnitude pruning and Optimal Brain Damage remove the wrong weights (abstract, Sect. 6).
- Use here: The origin of the update GPTQ applies at every column: Frantar 2022 GPTQ Eq. 3 is Eq. 5 here with w_q replaced by w_q - quant(w_q). Read Sect. 2 only.
- Note: Scanned PDF; extracted text garbles the equations, read them on the page.

### Frantar 2022. Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning
- File: `papers/Frantar-2022-Optimal-Brain-Compression.pdf`
- Venue: NeurIPS 2022 (printed "36th Conference on Neural Information Processing Systems (NeurIPS 2022)"; arXiv:2208.11580v2)
- Does: Makes OBS exact and cheap per layer: the layer reconstruction loss ||W X - W_hat X||^2 has Hessian H = 2 X X^T shared by every row, so weights are removed one at a time with exact OBS updates and a rank-one update of H^-1 (ExactOBS, Algorithm 1, Sect. 4). Sect. 5 turns it into a quantizer, OBQ: the constraint target becomes quant(w_p) - w_p, giving the selection rule (quant(w_p) - w_p)^2 / [H^-1]_pp and the update delta_p = -(w_p - quant(w_p)) / [H^-1]_pp * H^-1_{:,p} (Eq. 7). Weights with error above half a step are quantized first (Sect. 5, "Quantization Outliers").
- Finds: ResNet18/50, YOLOv5, BERT. Asymmetric per-channel weight quantization, each layer optimised independently: ResNet50 4-bit 75.72 vs BRECQ 75.88 and AdaRound 75.84 (FP 76.13); 3-bit 75.24 vs 75.32 (Table 4). Pruning: ExactOBS best at 2x-4x FLOP reduction on all three models (Table 1).
- Use here: The direct parent of GPTQ. GPTQ is OBQ with a fixed column order for all rows at once, lazy block updates and a Cholesky factor in place of the H^-1 recursion. Read Sect. 3 and 5 before Frantar 2022 GPTQ Sect. 3-4.
- Note: Same first author and year as the GPTQ file; the two are distinguished by title in the file name.

### Nagel 2021. A White Paper on Neural Network Quantization
- File: `papers/Nagel-2021-White-Paper-Neural-Network-Quantization.pdf`
- Venue: preprint, arXiv:2106.08295v1 (Qualcomm AI Research, June 2021; no venue printed)
- Does: A tutorial. Uniform affine (asymmetric) quantization x_int = clamp(round(x / s) + z, 0, 2^b - 1), x_hat = s (x_int - z) (Eq. 4-7); symmetric as the z = 0 case (Sect. 2.2.1); per-tensor vs per-channel granularity (Sect. 2.2.3, 2.4.2); simulated (fake) quantization (Sect. 2.3); PTQ range setting by min-max vs MSE (Sect. 3.1), cross-layer equalization (3.2), bias correction (3.3), AdaRound (3.4), a standard PTQ pipeline (3.5) and a debugging guide (3.7); QAT (Sect. 4).
- Finds: No new results; recommends symmetric per-channel weights with MSE range setting, asymmetric activations, and AdaRound as the step that makes 4-bit weight PTQ work on ImageNet CNNs (Sect. 3.5-3.6).
- Use here: The reference for the RTN grid: Eq. 7 is the rounding, Sect. 2.4.1 the symmetric vs asymmetric trade-off, Sect. 2.3 fake quantization. Cross-layer equalization (Sect. 3.2) is the ancestor of the per-channel scaling in AWQ and SmoothQuant.
- Note: Computer-vision era, no LLMs and no grouping; GPTQ/AWQ "group size" is per-channel scaling applied along the input dimension in blocks.

### Frantar 2022. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- File: `papers/Frantar-2022-GPTQ.pdf`
- Venue: ICLR 2023 (printed "Published as a conference paper at ICLR 2023"; arXiv:2210.17323v2)
- Does: One-shot weight-only PTQ that quantizes each linear layer column by column and uses second-order (inverse-Hessian) information to update the not-yet-quantized weights, so OPT-175B / BLOOM-176B can be quantized to 3-4 bits in ~4 GPU hours. Calibration: 128 random 2048-token C4 segments; per-row asymmetric min-max grid; block size 128; Cholesky of the damped inverse Hessian.
- Finds: 4-bit GPTQ on OPT-175B loses 0.03 ppl (8.34 -> 8.37) where RTN loses 2.2 (10.54); at 3 bits RTN collapses (OPT-175B 7.3e3) while GPTQ gives 8.68 (Table 3). OPT-6.7B: FP16 10.86, RTN4 12.10, GPTQ4 11.39, RTN3 5.8e3, GPTQ3 14.86. Grouping g128 adds ~0.15 bits and closes most of the remaining gap (Table 5).
- Use here: Defines the GPTQ and RTN baselines the project implements; validate the OPT-6.7B numbers above (per-row, no grouping) before moving to Llama.
- Note: Filename says 2022 but the PDF prints ICLR 2023. The paper does NOT describe the "act-order" variant; that comes from the released code (AWQ calls it GPTQ-R / "reorder trick"). No LLaMA numbers in this paper.

### Lin 2023. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- File: `papers/Lin-2023-AWQ.pdf`
- Venue: MLSys 2024 (printed "Proceedings of the 7th MLSys Conference, 2024. Best Paper Award"; arXiv:2306.00978v6)
- Does: Weight-only INT3/INT4 g128 PTQ that scales up "salient" input channels (chosen by mean activation magnitude, not weight magnitude) before rounding, with the inverse scale folded into the previous op; the per-channel scale is s = s_X^alpha with alpha found by a 20-point grid search on [0,1] plus MSE weight clipping. No backprop or reconstruction; calibration from the Pile.
- Finds: Keeping 1% of channels in FP16 chosen by activation magnitude drops OPT-6.7B INT3-g128 ppl from 23.54 (RTN) to 11.39, weight-magnitude selection does not help (Table 1); scaling by s=2 reaches 11.92 without mixed precision (Table 2). Llama-2-7B INT4-g128: RTN 5.73, GPTQ 5.69, GPTQ-R 5.63, AWQ 5.60; INT3-g128: RTN 6.66, GPTQ 6.43, GPTQ-R 6.42, AWQ 6.24 (Table 4). AWQ needs ~10x fewer calibration sequences than GPTQ and is less sensitive to calibration domain (Fig. 8).
- Use here: Defines the AWQ method to implement; Table 4 is the primary Llama-1/Llama-2 validation target for RTN, GPTQ, GPTQ-R and AWQ at 3/4-bit g128.
- Note: Filename says 2023 (first arXiv version) but the PDF is the 2024 MLSys version (v6). The paper writes the quantizer in symmetric form (Delta = max|w| / 2^(N-1)).

### Shao 2024. OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models
- File: `papers/Shao-2024-OmniQuant.pdf`
- Venue: ICLR 2024 (printed "Published as a conference paper at ICLR 2024"; arXiv:2308.13137v3)
- Does: Learns quantization parameters instead of hand-setting them: Learnable Weight Clipping (sigmoid-parameterised clip strengths gamma, beta on the min-max range) and a Learnable Equivalent Transformation (channel-wise scale + shift, initialised from SmoothQuant / OS+), optimised block-by-block with AdamW on 128 WikiText2 samples; 1-16 h on one A100 for LLaMA-2 7B-70B.
- Finds: Table 1 gives the full RTN / GPTQ / AWQ / OmniQuant WikiText2 grid for LLaMA-1 7B-65B and LLaMA-2 7B-70B at W2/W3/W4, per-channel and g128 (numbers copied into the table below). OmniQuant beats GPTQ/AWQ everywhere, most at low bits (LLaMA-13B W2A16: 13.21 vs GPTQ 3832). W8A8 is explicitly skipped because SmoothQuant is already near-lossless there.
- Use here: Background method (not implemented), but Table 1 is the single best cross-check table for the project's RTN/GPTQ/AWQ implementations on Llama-2-7B.
- Note: OmniQuant reproduces SmoothQuant/OS+ with per-channel weight + per-token activation quantization; the GPTQ entries in Table 1 do not say whether act-order was used.

### Xiao 2023. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
- File: `papers/Xiao-2023-SmoothQuant.pdf`
- Venue: ICML 2023 (printed "Proceedings of the 40th International Conference on Machine Learning, PMLR 202, 2023"; arXiv:2211.10438v7)
- Does: Training-free W8A8 PTQ. Observes that activation outliers sit in fixed channels, so it divides each input channel by s_j = max|X_j|^alpha / max|W_j|^(1-alpha) and multiplies the matching weight rows by s_j (mathematically equivalent), migrating quantization difficulty from activations to weights; s is fused into the preceding LayerNorm/linear. Calibrated on 512 Pile sentences; three efficiency levels O1-O3 (per-token dynamic -> per-tensor static).
- Finds: alpha = 0.5 works for OPT/BLOOM, 0.75 for GLM-130B; the usable band is 0.4-0.6 on OPT-175B (Fig. 10). OPT-175B WikiText: FP16 10.99 -> 11.11/11.14/11.17 for O1/O2/O3, while naive W8A8, ZeroQuant and Outlier Suppression give ~1e5 (Table 3). Llama-2-7B W8A8 (per-token act, per-channel weight, seq 2048): FP16 5.474 -> 5.515 at alpha = 0.85; 13B 4.950 -> 4.929 (alpha 0.85); 70B 3.320 -> 3.359 (alpha 0.9) (Table 7). LLaMA-1 7B at seq 512: 11.51 -> 11.56 with alpha = 0.8 (Table 6).
- Use here: Defines the SmoothQuant method to implement (W8A8); validate against Table 7 (Llama-2-7B 5.515 at alpha 0.85).
- Note: Llama numbers use higher alpha (0.8-0.9) than the OPT default 0.5; Table 6 uses sequence length 512 so its FP16 value (11.51) is not comparable to the usual 5.68.

### Wei 2023. Outlier Suppression+: Accurate quantization of large language models by equivalent and effective shifting and scaling
- File: `papers/Wei-2023-Outlier-Suppression-Plus.pdf`
- Venue: EMNLP 2023 (ACL Anthology; the PDF is arXiv:2304.09145 v3, no venue printed)
- Does: Extends SmoothQuant-style migration with a channel-wise *shift* z_j = (max X_j + min X_j)/2 to remove outlier asymmetry, then a channel-wise scale chosen via a single "outlier threshold" t found by grid search on the quantized-vs-FP output MSE (attention output for Q/K/V); both folded into LayerNorm and the following weights. Evaluated on BERT, OPT 13B-175B, BLOOM(Z)-176B, LLaMA 7B-65B.
- Finds: On OPT-66B one outlier channel spans (-97,-58) and another (5.7,43), so the tensor range is 140 although no channel exceeds 40 (Fig. 1). Near-FP results at INT8 and INT6 with per-tensor activations; on LLaMA-1-7B with per-token quantization, WikiText2 FP16 5.68 -> INT6 5.76 (SmoothQuant 5.85, MinMax 6.00) and INT4 14.17 (SmoothQuant 16.87, MinMax 473.97) (Table 3). SmoothQuant gives the smallest activation error but not the smallest output error (Table 6).
- Use here: Background only; useful as the reference for the shift-then-scale idea and as a second W6/W4 activation-quantization baseline on LLaMA-1-7B if the project ever goes below 8-bit activations.
- Note: OS+ was later published at EMNLP 2023 but the PDF in hand carries no venue line, so it is indexed as a preprint.

### Bondarenko 2023. Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing
- File: `papers/Bondarenko-2023-Quantizable-Transformers.pdf`
- Venue: NeurIPS 2023 (printed "37th Conference on Neural Information Processing Systems (NeurIPS 2023)"; arXiv:2306.12929v2)
- Does: Explains the origin of activation outliers: some attention heads learn a "no-op" by dumping probability mass on delimiter tokens ([SEP], ".", ",") whose values are small; because softmax cannot output exact zeros, the pre-softmax range and therefore the FFN output feeding LayerNorm keep growing. Proposes two pre-training changes, clipped softmax and gated attention, tested on BERT-base, OPT-125M/350M/1.3B and ViT-S.
- Finds: >97% of BERT outliers coincide with delimiter tokens (Fig. 1). Gated attention on OPT-125M cuts max activation inf-norm from 340 to 8.7 and kurtosis from 1778 to 18.9, and W8A8 ppl from 21.18 to 16.02 (FP 15.84) (Table 2); OPT-1.3B W8A8 ppl 989.6 -> 29.95 (Table 3). Clipped softmax fails on OPT (no explanation given).
- Use here: Background only; the mechanistic story (softmax no-op -> outliers -> hard activations) the project can cite when relating outlier channels to token frequency / language; also relevant to the from-scratch GPT-2 (gated attention is a possible ablation).
- Note: Models are <= 1.3B; the authors note this does not cover long-trained large LLMs.

### Chen 2024. PrefixQuant: Eliminating Outliers by Prefixed Tokens for Large Language Models Quantization
- File: `papers/Chen-2024-PrefixQuant.pdf`
- Venue: preprint, arXiv:2410.05265 (v2, Jan 2025; no venue printed in the PDF)
- Does: Targets *token-wise* outliers (a few tokens with values > 1000, mostly the first token and delimiters like "." / "\n"). Finds the high-frequency outlier tokens in ~12 s, prefixes them (plus [BOS]) and stores their K/V in full precision, so no outlier token appears in the quantized forward pass; then optional block-wise fine-tuning. Built on top of Hadamard rotation (QuaRot/SpinQuant); W4A4KV4 and W4A8KV4 on Llama-2/3, Mistral, Qwen-2-7B.
- Finds: In Llama-2-7B two tokens out of 2048 carry 94.7% of the 4-bit per-token quantization error (Fig. 1); Hadamard rotation only cuts the max-to-median ratio from 4161 to 461, prefixing brings it to 2.4. Llama-3-8B W4A4KV4 WikiText2: QuaRot 8.41, SpinQuant 7.36, PrefixQuant-O1 7.26 (FP16 6.14) (Table 2); ablation on Llama-3-8B W4A4KV4: RTN 1282 -> +rotation 24.98 -> +prefix 7.53 -> +fine-tune 7.23 (Table 5). Llama-3-8B and Qwen-2-7B show outlier tokens only at the initial position (Table 5 / Fig. 5).
- Use here: Background only; gives the token-level (as opposed to channel-level) outlier picture and per-model outlier-token lists that the project can compare against its own Llama-3.1 / Qwen-2.5 activation statistics.
- Note: Its GPTQ/RTN baselines are weight+activation (W4A4/W4A8), not weight-only, so its numbers are not validation targets for this project.

### Lin 2024. DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs
- File: `papers/Lin-2024-DuQuant.pdf`
- Venue: NeurIPS 2024 (printed "38th Conference on Neural Information Processing Systems (NeurIPS 2024)"; arXiv:2406.01721v3)
- Does: Separates "normal" outliers (large in fixed channels across all tokens) from "massive" outliers (~1400 at a few tokens at the input of FFN down_proj). After SmoothQuant-style smoothing it applies a block-diagonal rotation built greedily around the largest-outlier channel, a zigzag channel permutation to equalise outlier mass across blocks, then a second rotation; all folded into the weights. W4A4 / W6A6 on LLaMA-1/2/3, Vicuna, Mistral, Phi-2.
- Finds: SmoothQuant alone moves massive outliers into the down_proj weights and yields new weight outliers (Fig. 1). W4A4 WikiText2 on Llama-2-7B: SmoothQuant 83.12, OmniQuant 14.26, Atom 8.40, DuQuant 6.28 (FP16 5.47) (Table 1); ablation: smoothing only NaN, +rotation 8.48, +permutation+rotation 6.28 (Table 6); DuQuant+RTN matches QuaRot+GPTQ (Table 8). Quantizes Llama-2-7B in 50 s.
- Use here: Background only; use its normal-vs-massive outlier taxonomy and the down_proj observation when analysing which layers/tokens break under quantization.
- Note: Also reports the SmoothQuant smoothing formula Lambda_j = max|X_j|^alpha / max|W_j|^(1-alpha) as its base step, matching Xiao 2023.

### Liu 2024. SpinQuant: LLM Quantization with Learned Rotations
- File: `papers/Liu-2024-SpinQuant.pdf`
- Venue: ICLR 2025 (printed "Published as a conference paper at ICLR 2025"; arXiv:2405.16406v4)
- Does: Rotates the residual stream (R1) and the V/out-proj pair (R2) with orthogonal matrices that fold into the weights (numerically identical FP network), plus optional online Hadamard rotations R3/R4 for KV and down_proj inputs. Instead of random/Hadamard rotations it learns R1, R2 on the Stiefel manifold with Cayley SGD (800 WikiText2 samples, 100 iters, ~25 min for LLaMA-2 7B), then applies GPTQ to the rotated weights.
- Finds: Rotation drops activation kurtosis from >200 to ~3 (Fig. 3). Random rotations vary by up to 13 points in zero-shot accuracy, random Hadamard by 6 (Fig. 4). W4A4KV4 LLaMA-2 7B: 2.9-point zero-shot gap to FP (LLM-QAT 22.0, SmoothQuant 25.0) with WikiText2 5.9 vs FP 5.5 (Table 1). With W4A16 RTN on LLaMA-2 7B, Cayley-optimised rotation gives 5.5 vs 6.7-6.9 for unoptimised rotation (Table 4).
- Use here: Background only; documents the rotation family and that SpinQuant's weight step is GPTQ (128 WikiText2 samples, seq 2048), which is the same recipe the project validates.
- Note: Filename says 2024 (arXiv) but the PDF prints ICLR 2025. Its Table 1 RTN/GPTQ rows are W4 with A8 or A4, not pure weight-only, so they are not directly comparable to the table below.

### Wang 2025. Task-Circuit Quantization: Leveraging Knowledge Localization and Interpretability for Compression
- File: `papers/Wang-2025-Task-Circuit-Quantization.pdf`
- Venue: preprint, arXiv:2504.07389 (v2, Jul 2025; no venue printed in the PDF)
- Does: Mixed-precision PTQ on top of GPTQ: keeps the top p% (~0.35%) of weights in 16-bit, chosen by the saliency |W_ij| * |dL/dW_ij| * |Q(W_ij) - W_ij| (a first-order Taylor estimate of the loss change caused by quantizing each weight, computed from one backward pass), and quantizes the rest with GPTQ. Calibration may be task-specific (MMLU splits, GSM8k, Spider) or general (C4/WikiText2). Llama-3-8B-Instruct, Qwen2.5-7B/32B-Instruct.
- Finds: 2-bit GPTQ collapses to near-random on MMLU (26%) and 0% on Spider; TaCQ recovers 49.2% MMLU and 21.9% Spider at 2.1 bits, and 96% of FP MMLU at 3.1 bits (Tables 2, 3). Calibration set matters a lot at 2-3 bits (10-17 points) but little at 4 bits and above (Tables 4, 5). Appendix Table 14, Llama-3-8B base, WikiText2: FP16 6.14, GPTQ 3-bit 12.28, GPTQ 2-bit 410.63, TaCQ 3.1-bit 7.53 (C4 calibration, 128 x 2048).
- Use here: Background only; its evidence that calibration-set choice dominates at 2-3 bits is directly relevant to whether English-only calibration hurts low-resource languages, and Table 14 is a rough GPTQ sanity value for Llama-3-8B.
- Note: Filename says "Wang" but the first author printed is Hanqi Xiao (Mohit Bansal's group, UNC). Writes the GPTQ Hessian as H = X X^T (no factor 2, no damping shown).

### SignalDegradation 2026. From Signal Degradation to Computation Collapse: Uncovering the Two Failure Modes of LLM Quantization
- File: `papers/SignalDegradation-2026-Two-Failure-Modes-Quantization.pdf`
- Venue: preprint, arXiv:2604.19884 (v1, 21 Apr 2026; no venue printed in the PDF)
- Does: Mechanistic study of GPTQ (GPTQModel, g128, 128 x 2048 C4) on Llama-3.1-8B, Qwen3-8B, Mistral-7B-Instruct, Gemma-2-9B at 8/4/3/2 bits, using a Pararel factual-recall probe with logit lens, activation patching, attention entropy/JSD, FFN gate sign-flip rate, CKA and SVD of hidden states. AWQ checked in an appendix.
- Finds: 4-bit = "signal degradation": the correct answer's rank drops but stays near the top, representations keep CKA structure, error subspace is only ~0.3 aligned with the signal, and a training-free fix (keep first 2 layers at 8-bit or protect high-kurtosis weights, then amplify the peak-confidence layer's logits) recovers 64-81% of the 4-bit failure cases (Table 1). 2-bit = "computation collapse": FFN gate sign flips > 30%, top-1% neuron Jaccard ~0.1, error aligned ~0.8 with the signal; quantizing only layers 0-1 of Llama-3.1-8B to 2-bit already drops robust-subset accuracy from 100% to 41.65% (Table 2). Llama/Mistral are most sensitive in early layers and in MLP down_proj / v_proj; Qwen/Gemma degrade uniformly. No perplexity table in the extracted pages.
- Use here: Background only, but the closest methodological template for this project: the same Llama-3.1-8B + GPTQ setting, and its layer/component sensitivity tools could be rerun per language.
- Note: First author is Chenxi Zhou (CAS); the filename carries no author name.

## Numbers to validate against

WikiText-2 test perplexity, weight-only unless stated. "Omni T1" = Shao 2024 Table 1; "AWQ T4" = Lin 2023 Table 4; "SQ T7" = Xiao 2023 Table 7; "GPTQ T3" = Frantar 2022 Table 3. All Llama numbers follow the GPTQ evaluation protocol (2048-token windows) as stated in OmniQuant/AWQ. Where two papers disagree, both are given.

| Setting | Llama-2-7B | LLaMA-1-7B | OPT-6.7B (per-row, no groups) |
|---|---|---|---|
| FP16 | 5.47 (Omni T1, AWQ T4); 5.474 (SQ T7) | 5.68 (Omni T1, AWQ T4) | 10.86 (GPTQ T3) |
| RTN 4-bit g128 | 5.72 (Omni T1); 5.73 (AWQ T4) | 5.96 (Omni T1, AWQ T4) | 12.10 (GPTQ T3, no grouping) |
| GPTQ 4-bit g128 | 5.61 (Omni T1); 5.69 no reorder / 5.63 GPTQ-R (AWQ T4) | 5.85 (Omni T1); 6.22 / 5.83 GPTQ-R (AWQ T4) | 11.39 (GPTQ T3, no grouping) |
| AWQ 4-bit g128 | 5.62 (Omni T1); 5.60 (AWQ T4) | 5.81 (Omni T1); 5.78 (AWQ T4) | — |
| RTN 3-bit g128 | 6.66 (Omni T1, AWQ T4) | 7.01 (Omni T1, AWQ T4) | 5.8e3 (GPTQ T3, no grouping); 23.54 g128 (AWQ T3) |
| GPTQ 3-bit g128 | 6.29 (Omni T1); 6.43 no reorder / 6.42 GPTQ-R (AWQ T4) | 6.55 (Omni T1); 8.81 / 6.53 GPTQ-R (AWQ T4) | 14.86 (GPTQ T3, no grouping) |
| AWQ 3-bit g128 | 6.24 (Omni T1, AWQ T4) | 6.46 (Omni T1); 6.35 (AWQ T4) | 11.39 g128 (AWQ T3) |
| RTN 4-bit per-channel | 6.11 (Omni T1) | 6.43 (Omni T1) | — |
| GPTQ 4-bit per-channel | 5.83 (Omni T1) | 6.13 (Omni T1) | — |
| RTN 3-bit per-channel | 539.48 (Omni T1) | 25.73 (Omni T1) | — |
| GPTQ 3-bit per-channel | 8.37 (Omni T1) | 8.06 (Omni T1) | — |
| W8A8 SmoothQuant | 5.515, alpha=0.85, per-token act / per-channel weight, seq 2048 (SQ T7) | 11.56 vs FP16 11.51, alpha=0.8, seq 512 (SQ T6; not comparable to 5.68) | OPT-175B only: 10.99 -> 11.11/11.14/11.17 O1/O2/O3 (SQ T3) |

Extra sanity values for the 8B class (different model from Llama-3.1-8B, use only as a rough check): Llama-3-8B base FP16 6.14 (TaCQ Table 14, PrefixQuant Table 2, DuQuant Table 5); GPTQ 3-bit 12.28 and 2-bit 410.63 with 128 x 2048 C4 calibration (TaCQ Table 14). None of these papers report Llama-3.1-8B, Qwen-2.5-7B or GPT-2 perplexities.

Caveats: the AWQ and OmniQuant tables differ by up to 0.08 ppl for the same nominal setting (e.g. GPTQ-4bit-g128 Llama-2-7B 5.61 vs 5.69), presumably calibration data (WikiText2 vs C4 vs Pile) and act-order; GPTQ-R in AWQ T4 is the act-order variant. Treat agreement within ~0.1 ppl as a pass.

## Algorithm sketches

### RTN (baseline used by all four papers)
1. For each weight row (or each group of g=128 consecutive input columns within a row), take min and max.
2. Asymmetric N-bit grid: scale = (max - min) / (2^N - 1), zero = round(-min / scale).
3. q = clamp(round(w / scale) + zero, 0, 2^N - 1); dequantize w_hat = scale * (q - zero). Fake quantization keeps w_hat in FP16 and never stores integers.
4. That is the whole method: no calibration data, no error feedback. OmniQuant notes its LWC reduces to this when gamma = beta = 1.

### GPTQ (Frantar 2022, Algorithm 1)
1. Run the calibration set (128 x 2048 tokens) through the model block by block; for each linear layer collect the input matrix X (d_col x n_tokens) and form H = 2 X X^T (d_col x d_col). Use inputs from the already-quantized preceding blocks, not the FP model.
2. Damping: H <- H + lambda * I with lambda = 0.01 * mean(diag(H)). Invert and take the upper Cholesky factor: U = chol(H^-1)^T; row j of U plays the role of [H^-1]_{j, j:}.
3. Fix the quantization grid (per-row min-max, optionally recomputed per g=128 group when the group starts). Walk columns j = 0..d_col-1 in blocks of B = 128:
   q_j = quant(W[:, j]);  err_j = (W[:, j] - q_j) / U[j, j];  W[:, j+1 : blockend] -= err_j * U[j, j+1 : blockend].
4. After each block, push the accumulated block errors E onto all later columns: W[:, blockend:] -= E * U[block, blockend:].
5. Output Q (quantized columns) and the reconstruction error; all rows share the same H so the update runs vectorized over rows. Cost O(max(d_row d_col^2, d_col^3)).
6. Act-order variant (from the released code, not the paper text; "GPTQ-R" in AWQ): permute columns so that diag(H) is decreasing (largest activation energy first), run steps 3-4 in that order, then un-permute Q. With groups this makes group membership follow the permuted order unless "static groups" are used.
7. Why it helps: rounding a column with large [H^-1]_jj is cheap, and every rounding error is compensated by the OBS update delta_F = -(w_q - quant(w_q)) / [H^-1]_qq * [H^-1]_{:,q} on the remaining weights.

### AWQ (Lin 2023, Eq. 4-5)
1. Collect the calibration inputs X of each linear layer (small Pile subset) and compute the per-input-channel mean absolute activation s_X[j] = mean_t |X[t, j]|.
2. Search space: s = s_X^alpha for one scalar alpha in [0, 1] (alpha = 0 means no scaling). Grid of 20 alphas.
3. For each alpha, form W' = W * diag(s) (scale input channel j of the weight by s_j), quantize W' with the plain RTN grouped quantizer Q (INT3/INT4, g128), and measure the layer-output error L(s) = || Q(W diag(s)) (diag(s)^-1 X) - W X ||.
4. Keep alpha* = argmin L. Store Q(W diag(s*)) as the quantized weight; fold diag(s*)^-1 into the preceding op (LayerNorm gain or previous linear) so the runtime activation is already X / s. In fake quantization it is enough to keep W_hat = Q(W diag(s*)) diag(s*)^-1.
5. Optionally grid-search a clipping ratio for each weight group to minimise the MSE of quantization (the paper reports using this).
6. Intuition (Eq. 2-3): scaling a salient channel by s > 1 leaves the group max (and thus Delta) almost unchanged, so the rounding error of that channel shrinks by ~1/s; too large s starts to inflate Delta and hurts the other channels (best s ~ 2 on OPT-6.7B, Table 2).
7. No gradients, no reconstruction; the only calibration statistic is s_X, which is why it needs few samples and transfers across domains.

### SmoothQuant (Xiao 2023, Eq. 3-4)
1. Run calibration text (512 Pile sentences in the paper) and record, for every linear layer that reads an activation with outliers (q/k/v projections and the first FFN linear; the project's W8A8 scope), the per-input-channel activation max a_j = max_t |X[t, j]| and the per-input-channel weight max w_j = max_k |W[j, k]|.
2. Smoothing factor with migration strength alpha: s_j = a_j^alpha / w_j^(1 - alpha). alpha = 0.5 splits the difficulty evenly (OPT/BLOOM default); larger alpha pushes more onto the weights (0.75 for GLM-130B, 0.8-0.9 for LLaMA / Llama-2 in Tables 6-7). alpha = 1 would equalise all activation channels; alpha = 0 would do nothing to the activations.
3. Equivalent transform: X_hat = X diag(s)^-1, W_hat = diag(s) W, so X_hat W_hat = X W exactly. Fold diag(s)^-1 into the preceding LayerNorm/RMSNorm gain (or previous linear); in fake quantization just scale X by 1/s in the forward pass.
4. Quantize W_hat to INT8 (per-channel or per-tensor) and X_hat to INT8 (per-token dynamic for O1, per-tensor dynamic O2, per-tensor static O3). Symmetric grid: Delta = max|.| / 127. The Llama-2 numbers in Table 7 use per-token activations and per-channel weights.
5. Softmax, LayerNorm and other element-wise ops stay in FP16; the paper also quantizes the attention BMMs, which the project can skip.
6. Choose alpha by a quick grid search on a held-out calibration subset (the paper's Fig. 10 shows a 0.4-0.6 sweet spot on OPT-175B: below it activations break, above it weights break).
