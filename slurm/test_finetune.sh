#!/usr/bin/env bash
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
# Test job: runs configs/finetune/llama_test.yaml (Llama 3.1 8B, LoRA, 200 pairs per language)
# on one GPU to catch cluster-only failures and measure memory and speed before the full job.
# Submit from the repo root: mkdir -p results/slurm && sbatch slurm/test_finetune.sh
#SBATCH --job-name=test_finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --qos=cs
#SBATCH --gpus=a100:1
#SBATCH --output=results/slurm/%x-%j.out

# Pinned to A100: measured ~7.6 s/step there, ~33 s/step on a B200 (test jobs 14361518, 14362900).
set -euo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Compute nodes may have no internet: Llama 3.1 8B and the prepared data are already local.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module load python/3.11
cd "$SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# GPU memory in use every 30 s, to size the full job.
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 30 &

# GPU torch lives in this venv; the repo's .venv is CPU-only.
# Each output line is timestamped, to measure seconds per step for the full job's --time.
~/.venvs/quantization_lrl/bin/python -u -m qlrl.finetune configs/finetune/llama_test.yaml 2>&1 \
    | while IFS= read -r line; do printf '%s %s\n' "$(date +%T)" "$line"; done
kill %1
