#!/usr/bin/env bash
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
# Runs one entry point with one config; in an array, the task index picks the seed or model.
# From the repo root (set --time to the job: ~30 min quantize, ~3-6 h evaluate):
#   mkdir -p results/slurm
#   sbatch --array=0-4 --time=01:00:00 slurm/run.sh qlrl.quantization configs/quantize/qwen_4bit.yaml
#   sbatch --array=0-22 slurm/run.sh qlrl.evaluation configs/eval/qwen.yaml
#   sbatch --time=01:00:00 slurm/run.sh qlrl.analysis configs/analysis/eval_v2.yaml
#SBATCH --job-name=run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --output=results/slurm/%x-%A_%a.out

set -euo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Compute nodes may have no internet: models, C4, WikiText-2, AfriCOMET, GlotLID are cached.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module load python/3.11
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# GPU torch lives in this venv; the repo's .venv is CPU-only. AfriCOMET uses .venv-comet.
~/.venvs/quantization_lrl/bin/python -m "$1" "$2" ${SLURM_ARRAY_TASK_ID:-}
