# The from-scratch model: tokenizer, architecture, training

How the phase-2 decoder is built and trained, and why. Its data is in
[pretraining_data.md](pretraining_data.md); the papers behind each choice are in
[../papers/architecture.md](../papers/architecture.md) and
[../papers/tokenizer_data_nmt.md](../papers/tokenizer_data_nmt.md).

The rule behind every choice: **build it like the phase-1 models.** Phase 1 found the gap in
Llama 3.1 8B and Qwen 2.5 7B, so the model the gap is explained in is built the same way. The
newest training changes are exactly the ones that change outliers, the thing being measured:
Muon with single-scale RMSNorm takes activation kurtosis from 1818.56 to 0.04 (Park 2025),
gating the attention output takes maximum activations from 1053 to 94 (Qiu 2025), and
non-diagonal optimizers reduce outlier features (He 2024). None of them is used.

## 1. Tokenizer

`qlrl/data/tokenizer.py`, `configs/data/tokenizer.yaml`; output `models/tokenizer-64k/`.

| Choice | Value | Why |
|---|---|---|
| Algorithm | byte-level BPE (HF `tokenizers` 0.23.2, pinned) | as Llama 3 and Qwen 2.5; every script covered, nothing unknown (Sennrich 2016). HF and SentencePiece BPE share only 0.61–0.77 of their vocabularies on the same data (Ali 2024), so the library is part of the setup |
| Vocabulary | 65,536 | Tao 2024's English optimum at our size (~0.4B parameters outside embeddings) is 10K–24K; multilingual needs up to 3× an English vocabulary (Ali 2024); ids fit in uint16 |
| Split before BPE | GPT-4o (o200k) pattern | the GPT-4 pattern cuts words at combining marks (Devanagari, Tamil, Thai, Khmer, Burmese); the split matters more than vocabulary size (Dagan 2024) |
| Normalization | NFC, the same for tokenizer, training and evaluation | Land 2024 |
| Special tokens | `<s>` 0, `</s>` 1, `<pad>` 2 | every document is `<s> … </s>`; the tokenizer adds `<s>` in front of every text, as the evaluation prompts expect |
| Training text | 2.0G characters sampled with q_l ∝ p_l^0.3 | see below |

**Sampling (decided 2026-10-30: alpha = 0.3).** The merges are learned from a sample in
which language l is drawn with probability ∝ p_l^alpha, p_l its share of all characters
(Conneau 2020). Natural shares (alpha = 1) spend the vocabulary on the large languages and
split the small ones into many pieces; equal shares (alpha = 0) give the small languages
vocabulary rows the model rarely trains on, left near their random start (Land 2024). 0.3 is
the standard middle. Sizes are counted in characters, not bytes (Arnett 2024: a Burmese
character takes three UTF-8 bytes). A language with less text than its quota is taken whole.
`manifest.json` holds each language's characters, sampling rate and sampled characters.

Both effects are measured after training, per language: fragmentation (tokens per sentence
on the shared core and FLORES sentences) and under-trained tokens (input-embedding norm per
row, Land 2024).

## 2. Architecture

Hugging Face `LlamaForCausalLM`, set entirely in `configs/pretrain/llama_576m.yaml`, so the
quantizers and the evaluation load it as they load Llama 3.1. 575.7M parameters, 424.7M
outside the two embedding matrices.

| Choice | Value | Why |
|---|---|---|
| Blocks | 30 layers, width 1152 | deep and thin below 1B: MobileLLM's optimum is near 30 layers; its 600M model is 40 × 1152 |
| Normalization | pre-norm RMSNorm | as Llama and Qwen (Zhang 2019, Xiong 2020) |
| MLP | SwiGLU, 3072 (8/3 × width) | Shazeer 2020 |
| Attention | 9 query heads, 3 key/value heads (GQA), head size 128, no biases | GQA as Llama (32/8) and Qwen (28/4) (Ainslie 2023); no biases as Llama |
| Positions | RoPE, base 10,000, context 2048 | Su 2021; 10,000 is enough at 2048 |
| Embeddings | untied | as Llama 3.1 8B and Qwen 2.5 7B; the per-module map quantizes embeddings and output head separately, which a tied matrix cannot (Press & Wolf 2017 is the case for tying) |
| Not used | QK-norm, reordered norm, z-loss (OLMo 2); attention gates; single-scale RMSNorm | they change activation statistics; neither phase-1 model uses them. Added only if training diverges |

## 3. Training recipe

`qlrl/pretrain.py` (loop, checkpoints), `qlrl/recipe.py` (schedule, optimizer, loss, GPUs),
`qlrl/data/tokens.py` (token files and windows).

- **Data: one pass, natural mix.** Every train token of every corpus once, in its natural
  proportion; no upsampling. The stream is cut into windows of 2,049 tokens (input plus
  next-token target) and shuffled once by the seed. A window may cross from one document into
  the next; `<s>` and `</s>` mark the boundary. Languages whose validation loss stays at the
  floor are reported as below it, not analysed.
- **Budget:** whatever the data adds up to: 17.07B train tokens in 137 languages (measured
  2026-10-30), 8,330,392 windows, 32,540 steps.
  Chinchilla's 20 tokens per parameter gives about 11.5B for 576M (Hoffmann 2022).
- **Optimizer:** AdamW (0.9, 0.95), eps 1e-8, weight decay 0.1 on linear layers only (not
  embeddings or norm gains, OLMo 2), clip 1.0; fp32 weights and optimizer, bf16 matmuls.
  Peak learning rate 2e-3 (proposed: MobileLLM 600M 2e-3, SmolLM2 360M 3e-3).
- **Schedule: warmup-stable-decay.** 2,000 warmup steps, constant, then a (1 − sqrt) decay
  to 0 over the last 20% of steps (Hägele 2024). PTQ error changes when the learning rate
  decays, not with tokens seen (Catalan-Tatjer 2026), so every checkpoint before the decay
  sits at the same point of the schedule.
- **Batch:** 524,288 tokens per step (256 windows), 8 windows per GPU per forward.
- **Validation:** every 1,000 steps, loss per language on up to 16 windows of its validation
  text: free-text `valid.parquet`, or the target side of the church or external dev split for
  parallel-only languages. Kikamba has a single dev sentence; its loss is noise.

## 4. Outputs

| Where | What |
|---|---|
| `nobackup/archive/pretrain_tokens/<source>/<folder>/` | `train.bin`, `valid.bin` (uint16), `manifest.json` (language, documents, tokens) |
| `nobackup/archive/pretrain/<name>/step-<n>/` | bf16 weights every 500 steps (262M tokens): the learning curve, about 1.2 GB each |
| `nobackup/archive/pretrain/<name>/resume.pt` | fp32 weights, AdamW state, step; the next job resumes from it |
| `nobackup/archive/pretrain/<name>/state-<n>.pt` | copies of `resume.pt` at 25%, 50%, 75% of the run, about 7 GB each: cooldowns branch from them (Hägele 2024) |
| `results/pretrain/<name>/history.jsonl` | learning rate, train loss, tokens per second, validation loss per language |
| `models/<name>/` | the final model, its tokenizer, `manifest.json` (config, tokens per language, git sha) |

The tokens of language l seen by checkpoint n are fixed by the seed: the windows of steps
1..n in the shuffled order.

## 5. Running it

Tests first: `pytest tests/test_pretrain.py` (CPU, under a minute: tokenizer round trip,
windows, schedule, training lowers the loss, a resumed run equals an uninterrupted one).

1. Tokenizer, one CPU job:
   `sbatch --gpus=0 --cpus-per-task=16 --mem=200G --time=08:00:00 slurm/run.sh qlrl.data.tokenizer configs/data/tokenizer.yaml`
2. Token files, one task per corpus (246):
   `sbatch --array=0-245 --gpus=0 --cpus-per-task=8 --mem=32G --time=02:00:00 slurm/run.sh qlrl.data.tokens configs/data/tokenizer.yaml`
3. Cluster test of the training loop on two GPUs (tiny model on the `--test` token files):
   `sbatch -J llama-test --gpus=2 --time=00:15:00 slurm/pretrain.sh configs/pretrain/llama_test.yaml`
4. Training, eight GPUs of one node. The qos stops a job at 24 h; the run saves after 23 h
   and the next job with the same name resumes:
   `for i in 1 2 3 4; do sbatch -J llama-576m --dependency=singleton slurm/pretrain.sh configs/pretrain/llama_576m.yaml; done`.
   The first job's tokens per second and memory set the real count.

Steps 1–2 and the CPU half of 3 also run with `--test` (first 200 documents of each corpus,
2,048-token vocabulary) into `models/tokenizer-64k_test` and `pretrain_tokens_test`.
