# Phase 1: data, setup and methodology

Facts taken from the code, configs and result manifests as of 2026-10-23. The numbers are in
`phase1_results.xlsx` in this folder (sheets listed at the end of this file).

## 1. Datasets

| Dataset | Version | What it is | Raw size | Used for |
|---|---|---|---|---|
| Church corpus, 2026 release (BYU MT lab) | `church-data-2026/<Language>/filtered/` | English–X human translations: scripture, manuals, talks, software strings. Already cleaned by the lab | fra 2,245,512; spa 2,610,396; por 2,457,565; swh 437,118; kek 227,804; amh 187,539; grn 119,981; zul 109,726; yor 99,769; qug 18,225 lines | Fine-tune train and dev loss; Kichwa test (main eval); every test set of the church eval |
| FLORES+ | `data/external/flores_plus`, commit `e707e62e` | Multi-way parallel professional translations of the same English sentences | dev 997, devtest 1,012 per language | Test set (devtest) and prompt examples + length cap (dev) for fra, spa, por, yor, zul, swh, amh, grn |
| AmericasNLP 2021 | commit `d3f519c6` | Spanish–Guarani / Nahuatl / Bribri | grn 26,032 / 995 / 1,003; nah 16,145 / 672 / 1,003; bzd 7,508 / 996 / 1,003 (train / dev / test) | Fine-tune train; dev loss (nah, bzd, part of grn); test for nah, bzd |
| MayanV (Q'eqchi') | commit `07d1426c` | Spanish–Q'eqchi' | 2,114 / 1,000 / 1,000 | Fine-tune train; test for kek (main eval) |
| Napo Kichwa–Spanish lexicon | commit `b05c9a5f` | Headwords (4,135) and short example sentences (2,178) | 6,313 entries | Fine-tune train only (Kichwa) |
| C4 (English) | `c4-train.00000-of-01024.json.gz` | English web text | one shard | GPTQ / AWQ calibration only |
| WikiText-2 raw v1, test | parquet | English Wikipedia | test split | Quantizer validation only |

**English side of the Spanish–X corpora.** AmericasNLP, MayanV and the Kichwa lexicon pair the
language with Spanish. Their Spanish side was translated to English with Qwen3-32B (revision
`9216db57`, Slurm job 14131729): prompt "Translate Spanish to English." plus the first three
FLORES+ Spanish–English dev pairs as `<spa>…</spa> <eng>…</eng>`, greedy, stop at `</eng>`,
thinking mode off. 47,913 distinct sentences; 115 lines over 1,000 characters were left
untranslated and their pairs dropped. The X side stays human-written. This applies to train,
dev **and test**: kek, nah and bzd X→English references and English→X sources are machine
translations.

## 2. Splits

Code: `qlrl/data/prepare.py`, `qlrl/data/split.py`; config `configs/data/data.yaml`; output
`data/processed/<lang>/{train,dev,test}.jsonl`.

- **Church:** no published split. Per language, all `filtered/` lines are shuffled with
  `random.Random(0)`; the first 1,000 are dev, the next 1,000 test, the rest train. By line,
  not by document, because the metadata has no usable document unit.
- **External corpora:** published train / dev / test used as is. FLORES+ dev → dev,
  devtest → test (FLORES+ has no train).
- **Pairs with an empty side are dropped** (this is why Nahuatl test is 995 of 1,003: eight
  empty lines in `test.nah`; Nahuatl dev 671 of 672).
- **Cleaning:** the lab cleaning tool (`~/data-cleaning-pipeline`) on external train sets only.
  Church is already clean; dev and test are left as published.
- **Decontamination:** a train pair is dropped if its English or target side, lowercased with
  only letters and single spaces kept, equals either side of any dev or test pair of **any**
  language, or any Spanish line of an external dev/test set. Across languages because the same
  English verse appears in many languages.

| Corpus (train) | Raw | Into cleaner | Kept by cleaner | Final after decontamination |
|---|---:|---:|---:|---:|
| grn AmericasNLP | 26,032 | 26,032 | 11,788 (−54.7%) | 11,787 |
| nah AmericasNLP | 16,145 | 15,981 | 12,641 | 12,637 |
| bzd AmericasNLP | 7,508 | 7,508 | 6,132 | 6,125 |
| kek MayanV | 2,114 | 2,114 | 1,981 | 1,965 |
| qug Kichwa lexicon | 6,313 | 6,279 | 2,008 (−68.0%) | 2,007 |

Rows in `data/processed` (git `f2c22d93`):

| Lang | Train (church) | Train (external) | Dev | Test | Removed from train |
|---|---:|---|---|---|---:|
| fra | 2,211,867 | — | church 1,000, flores 997 | church 1,000, flores 1,012 | 31,645 |
| spa | 2,574,467 | — | church 1,000, flores 997 | church 1,000, flores 1,012 | 33,929 |
| por | 2,423,120 | — | church 1,000, flores 997 | church 1,000, flores 1,012 | 32,445 |
| yor | 89,832 | — | church 1,000, flores 997 | church 1,000, flores 1,012 | 7,937 |
| zul | 100,099 | — | church 1,000, flores 997 | church 1,000, flores 1,012 | 7,627 |
| swh | 418,161 | — | church 1,000, flores 997 | church 1,000, flores 1,012 | 16,957 |
| amh | 173,590 | — | church 1,000, flores 997 | church 1,000, flores 1,012 | 11,949 |
| kek | 213,543 | mayanv 1,965 | church 1,000, mayanv 1,000 | church 1,000, mayanv 1,000 | 12,277 |
| grn | 109,702 | americasnlp 11,787 | church 1,000, flores 997, americasnlp 994 | church 1,000, flores 1,012, americasnlp 1,003 | 8,280 |
| nah | 0 | americasnlp 12,637 | americasnlp 671 | americasnlp 995 | 4 |
| bzd | 0 | americasnlp 6,125 | americasnlp 994 | americasnlp 1,003 | 7 |
| qug | 14,735 | kichwa 2,007 | church 1,000 | church 1,000 | 1,491 |

## 3. Where each piece of data is used

| Use | Data |
|---|---|
| Fine-tune train | train split of spa, por, swh, yor, zul, amh, kek, grn, nah, bzd, qug (not fra); spa, por, swh capped to a seeded random 100,000 pairs |
| Fine-tune dev loss | 400 seeded random dev pairs per language from church + AmericasNLP dev (kek: church only; grn: church + AmericasNLP; nah, bzd: AmericasNLP) |
| Prompt examples | first 3 rows of the dev split of the same corpus as the test set; identical for every model |
| Generation cap | all dev pairs of that corpus (mean target/source token ratio) |
| Test, main eval (`results/eval`) | FLORES+ devtest: fra, spa, por, yor, zul, swh, amh, grn. MayanV test: kek. AmericasNLP test: nah, bzd. Church test: qug |
| Test, church eval (`results/eval_church`) | church test of fra, spa, por, yor, zul, swh, amh, kek, grn |
| Calibration | C4, 128 windows × 2,048 tokens, seeds 0–4 |
| Validation | WikiText-2 test, non-overlapping 2,048-token windows (Llama 141, Qwen 146) |

Never in training, calibration or prompts: any test set. FLORES+ dev appears only as prompt
examples and in the cap ratio.

## 4. Models

| Model | Revision | Role |
|---|---|---|
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B` @ `d04e592b` | core |
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B` | second family (no BOS token) |
| Llama 3.1 8B + LoRA | `models/Llama-3.1-8B-lora`, git `c0d96f03` | fine-tuned on the project languages, merged |

Each was quantized to 22 checkpoints (at 4 and 3 bits: RTN once, GPTQ and AWQ with seeds 0–4).
With its full-precision run that is 23 runs per model, 69 in total, each evaluated on both test
designs.

## 5. Quantization

Own implementation (`qlrl/quant/`), entry `python -m qlrl.quantization <yaml>`.

- **Weight-only, fake quantization:** every linear in the decoder blocks (Llama: q, k, v, o,
  gate, up, down; 224 matrices) is rounded and stored back in bf16. Embeddings, `lm_head` and
  norms stay bf16; activations are not quantized.
- **Grid:** asymmetric min–max per group of 128 input columns per row, 2^bits levels, round to
  nearest. 4 and 3 bits.
- **RTN:** the grid alone; no calibration; one run.
- **GPTQ** (Frantar et al. 2023): columns rounded left to right; each column's error spread over
  the remaining columns using the upper Cholesky factor of H⁻¹, H = XXᵀ; damping 1% of mean
  diag(H); lazy updates per 128-column group; fp32. **No act-order** (natural column order).
- **AWQ** (Lin et al. 2024): per linear, s = mean|x|^α (normalised), Ŵ = Q(W·s)/s, α in
  {0, 0.05, …, 0.95}, keep the α with the smallest output error. Simplified: s divided back
  into W, error per linear, no clipping search.
- **Sequential, block by block** (GPTQ, AWQ): each decoder block's statistics come from the
  outputs of the already-quantized blocks before it.
- **Calibration:** C4, 128 × 2,048 tokens, five seeds. Llama windows start with BOS
  (BOS + 2,047 tokens); Qwen has no BOS.
- **Not run in phase 1:** NF4, 2 bits.

WikiText-2 check (perplexity; all seeds in the Quantized models sheet):

| | FP | RTN 4 | GPTQ 4 | AWQ 4 | RTN 3 | GPTQ 3 | AWQ 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama, calibration without BOS (job 14284378) | 6.240 | 6.933 | 6.643 | 6.652 | | | |
| Llama, BOS calibration (final, seed 0) | 6.240 | 6.933 | 7.150 | 6.645 | 14.979 | 24.667 | 8.625 |
| Qwen (seed 0) | 6.851 | 7.212 | 7.087 | 7.108 | 8.978 | 8.149 | 8.317 |
| Llama + LoRA (seed 0) | 8.843 | 9.943 | 10.229 | 9.386 | 22.044 | 32.276 | 12.069 |

## 6. Fine-tuning

`configs/finetune/lora.yaml`, `qlrl/finetune.py`, `qlrl/data/training.py`.

- LoRA (peft) r 64, alpha 128, dropout 0.05 on q, k, v, o, gate, up, down: 167,772,160 of
  8,198,033,408 parameters train (2.05%). Merged into the weights after training.
- Row = 4 pairs (3 examples + the graded pair), written exactly as the evaluation prompt, in one
  random direction; loss only on the graded target and its closing tag. Rows over 1,024 tokens
  dropped (4,987 of 259,003, mostly Amharic).
- AdamW, lr 2e-4, weight decay 0, 3% linear warmup then cosine to 0, clip 1.0, bf16, gradient
  checkpointing, seed 0. 4 rows × 8 accumulation = 32 rows (128 pairs) per step; one epoch,
  7,938 steps.
- Dev loss every 500 steps; final mean 1.051 (step 0: 2.100). Final train loss 0.616.
- Slurm job 14363065: 19 h 22 min on one A100-SXM4-80GB.
- Then quantized, evaluated and analysed exactly as the base model.

## 7. Evaluation

`python -m qlrl.evaluation <yaml>`; configs in `configs/eval/`.

- Both directions, English→X and X→English, scored separately.
- Base models, no chat template: three dev pairs then the sentence,
  `<eng>…</eng> <yor>…</yor>` ×3, then `<eng>{sentence}</eng> <yor>`.
- Greedy, batch 32; stop at the closing tag or at ceil(2 × source tokens × dev target/source
  token ratio).
- Whole test sets; bf16 for every model; one A100 80 GB or B200 per run.

| Metric | Settings | Languages |
|---|---|---|
| chrF++ | sacrebleu 2.6.0, `nrefs:1\|case:mixed\|eff:yes\|nc:6\|nw:2\|space:no` | all |
| BLEU | `nrefs:1\|case:mixed\|eff:no\|tok:13a\|smooth:exp` | all |
| AfriCOMET | `masakhane/africomet-stl-1.1` @ `b4065f44`, unbabel-comet 2.2.7; mean × 100 | fra, yor, zul, swh, amh |
| Off-target % | GlotLID @ `85cd6716`; eng→X: share labelled English; X→eng: share not English | all |
| Reference NLL / perplexity | −log p of the human reference after the prompt (forced decoding), per word, per byte, per token | all |
| Perplexity, NLL per word / byte | bare test sentence after BOS, both sides | all |

## 8. Analysis

`python -m qlrl.analysis configs/analysis/all.yaml`, math in `qlrl/analysis/stats.py`.

- **Damage:** d(i) = log p_full(i) − log p_quant(i) of each bare sentence, averaged over seeds;
  D = Σd / Σbytes (also per word). **Gap** = D(language side) − D(English side).
- **Drop** = full − quantized score (mean over seeds), for chrF++ and AfriCOMET.
- **Intervals:** paired percentile bootstrap over sentences, 1,000 resamples, seed 0, 95%.
- **Floor:** score of the references shuffled (seed 0). AfriCOMET floors (FLORES / church),
  eng→X | X→eng: fra 26.8 / 25.7 | 26.1 / 24.8; yor 45.7 / 50.1 | 30.8 / 30.4;
  zul 33.8 / 33.4 | 26.9 / 25.8; swh 29.6 / 30.6 | 27.4 / 27.0; amh 30.4 / 30.6 | 30.0 / 29.2.
- **Reading of a drop:** "at floor" if full − floor ≤ 3 bootstrap SE; "noise" if the interval
  contains 0 or |drop| ≤ seed SD; otherwise "loss" or "gain".

## 9. Sheets of `phase1_results.xlsx`

| Sheet | One row per |
|---|---|
| Eval scores | evaluation × model × method × bits × seed × language × direction |
| Perplexity | evaluation × run × language × side (English or X) |
| Analysis damage | evaluation × model × configuration × language |
| Analysis translation | evaluation × model × configuration × language × direction × metric |
| Quantized models | quantized checkpoint (calibration, WikiText-2) |
| Fine-tune | language (pairs, rows, dev loss); below it, every logged training step |
