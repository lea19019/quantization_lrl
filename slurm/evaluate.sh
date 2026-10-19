#!/usr/bin/env bash
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
# Full evaluation: one array task per config in configs/eval_llama/ (awq, full, gptq, rtn,
# in sorted order). Submit only after test_evaluate.sh passes. From the repo root:
#   mkdir -p results/slurm && sbatch slurm/evaluate.sh
#SBATCH --job-name=evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --array=0-3
#SBATCH --output=results/slurm/%x-%A_%a.out

set -euo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Compute nodes may have no internet: Llama, AfriCOMET and GlotLID are already cached.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module load python/3.11
cd "$SLURM_SUBMIT_DIR"

configs=(configs/eval_llama/*.yaml)  # sorted by the glob under LC_ALL=C
config=${configs[$SLURM_ARRAY_TASK_ID]}
echo "$config"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# GPU torch lives in this venv; the repo's .venv is CPU-only. AfriCOMET uses .venv-comet.
~/.venvs/quantization_lrl/bin/python -m qlrl.eval "$config"
