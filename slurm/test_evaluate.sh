#!/usr/bin/env bash
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
# Test job: runs configs/eval_llama_test.yaml (Llama 3.1 8B, Yoruba and Nahuatl, 32
# sentences) on one GPU to catch cluster-only failures before the full evaluation.
# Submit from the repo root: mkdir -p results/slurm && sbatch slurm/test_evaluate.sh
#SBATCH --job-name=test_evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Compute nodes may have no internet: Llama, AfriCOMET and GlotLID are already cached.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module load python/3.11
cd "$SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# GPU torch lives in this venv; the repo's .venv is CPU-only. AfriCOMET uses .venv-comet.
time ~/.venvs/quantization_lrl/bin/python -m qlrl.eval configs/eval_llama_test.yaml
