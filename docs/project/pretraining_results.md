# From-scratch model: settings and results so far

What has been run for the phase-2 model and what it produced. The reasons behind each setting
are in [pretraining_model.md](pretraining_model.md).

## 1. Tokenizer (job 14923412, 2026-10-30, 19 min 41 s, 16 CPUs)

| Setting | Value |
|---|---|
| Config | `configs/pretrain/tokenizer.yaml`, run at commit `cdcf5fe` |
| Algorithm | byte-level BPE, HF `tokenizers` 0.23.2 |
| Vocabulary | 65,536; `<s>` 0, `</s>` 1, `<pad>` 2 |
| Split before BPE | GPT-4o (o200k) pattern |
| Normalization | NFC |
| Sample | 2.0e9 characters requested, language l drawn ∝ p_l^0.3 (p_l its share of characters) |
| Output | `models/tokenizer-64k/` (`manifest.json`: characters, rate, sampled characters per language) |

Result: 137 languages, 48.97G characters in all, 1.94G sampled.

## 2. Token files (array job 14923413, 2026-10-30, 246 tasks, 8 CPUs each)

Every corpus of `nobackup/archive/freetext` encoded once as `<s> … </s>` per document, uint16,
into `nobackup/archive/pretrain_tokens/<source>/<folder>/`. Validation text: `valid.parquet`, or
the target side of `data/processed_pretrain/<lang>/dev.jsonl` for parallel-only corpora.

| | Value |
|---|---|
| Corpora | 246 (task 246 failed: the array was 0-246, the index has 0-245) |
| Languages | 137 |
| Documents | 82,839,846 |
| Train tokens | 17,068,973,635 |
| Validation tokens | 25,889,961 |
| Size on disk | 32 GB |
| Windows of 2,049 tokens | 8,330,392 |
| Steps at 256 windows | 32,540 |

Most train tokens: jpn 977.5M, rus 930.7M, deu 917.7M, spa 881.1M, fra 839.8M.
Fewest: bzd 118K, cuk 124K, guz 127K, niu 145K, bam 150K.
Fewest validation tokens: kam 9, fon 876, bam 1,792, orm 4,568.

## 3. Per language

Characters: the language's share of all characters; sample: its share of the tokenizer sample;
rate: the fraction of its characters sampled. Characters per token (higher = fewer pieces): the
first 500 target sentences of `data/processed_pretrain/<lang>/test.jsonl`, without special tokens,
ours vs `meta-llama/Llama-3.1-8B`.

| Lang | Characters % | Sample % | Rate | Train tokens | Valid tokens | Chars/token ours | Llama 3.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| eng | 5.802 | 1.980 | 0.0135 | 823.4M | 359,818 | – | – |
| spa | 6.133 | 2.013 | 0.0130 | 881.1M | 490,624 | 3.37 | 3.47 |
| fra | 5.511 | 1.950 | 0.0140 | 839.8M | 472,064 | 3.19 | 3.42 |
| por | 4.445 | 1.828 | 0.0163 | 652.3M | 384,768 | 3.41 | 3.34 |
| tha | 0.776 | 1.083 | 0.0554 | 158.0M | 440,332 | 2.63 | 2.01 |
| cmn_hans | 0.468 | 0.931 | 0.0789 | 186.1M | 276,332 | 1.20 | 1.14 |
| swh | 0.179 | 0.698 | 0.1543 | 29.0M | 198,208 | 3.19 | 2.55 |
| mya | 0.176 | 0.694 | 0.1565 | 36.0M | 281,529 | 2.22 | 0.51 |
| grn | 0.080 | 0.548 | 0.2717 | 14.1M | 204,803 | 2.90 | 2.21 |
| kek | 0.054 | 0.487 | 0.3580 | 9.5M | 57,719 | 2.94 | 2.25 |
| amh | 0.037 | 0.436 | 0.4619 | 11.4M | 131,173 | 1.73 | 0.41 |
| yor | 0.036 | 0.432 | 0.4718 | 7.2M | 117,762 | 2.60 | 1.56 |
| zul | 0.031 | 0.412 | 0.5272 | 5.3M | 103,978 | 3.13 | 2.43 |
| nah | 0.013 | 0.318 | 0.9639 | 2.4M | 36,342 | 2.77 | 2.57 |
| qug | 0.005 | 0.116 | 1.0000 | 0.73M | 45,226 | 3.27 | 2.51 |
| bzd | 0.0004 | 0.009 | 1.0000 | 0.12M | 36,062 | 1.77 | 1.70 |

eng: there is no `data/processed_pretrain/eng/`. All 137 languages:
`models/tokenizer-64k/manifest.json` and the token-file manifests.

## 4. Cluster test of the training loop (job 14923410, 2026-10-30, 1 min 25 s, 2 B200)

| Setting | Value |
|---|---|
| Config | `configs/pretrain/llama_test.yaml` |
| Model | Llama, 2 layers, width 64, 2 query / 1 key-value heads, vocabulary 2,048 |
| Data | `--test` tokenizer and token files (first 200 documents of each corpus) |
| Batch | 2,048 tokens (16 windows of 128), warmup 5, cooldown 20%, peak lr 2e-3 |
| Precision | fp32, no compile |

| Step | Train loss | Mean valid loss |
|---:|---:|---:|
| 0 | – | 7.650 (ln 2048 = 7.625) |
| 1387 (end) | 3.19513 | 4.285 |

Same config in one CPU process: final train loss 3.19514. Throughput about 180K tokens/s.
Results: `results/test/pretrain/llama-test/`.

## 5. Not run yet

Training `configs/pretrain/llama_576m.yaml` (575.7M parameters, 32,540 steps).
