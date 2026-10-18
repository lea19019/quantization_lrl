#!/usr/bin/env bash
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
# Full job: quantize with every method in the given config and check WikiText-2 perplexity.
# Submit only after test_quantize.sh passes. From the repo root:
#   mkdir -p results/slurm && sbatch slurm/quantize.sh configs/quantize_llama.yaml
#SBATCH --job-name=quantize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Compute nodes may have no internet: the models and the C4/WikiText-2 files are already local.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module load python/3.11
cd "$SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# GPU torch lives in this venv; the repo's .venv is CPU-only.
~/.venvs/quantization_lrl/bin/python -m qlrl.quantization "$1"
