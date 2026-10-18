#!/usr/bin/env bash
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
# Data preparation: runs configs/data.yaml (python -m qlrl.data.prepare). CPU only, unless
# the config has a `translate:` section, which needs one 80 GB GPU.
# Submit from the repo root: mkdir -p results/slurm && sbatch slurm/prepare_data.sh [--test]
#SBATCH --job-name=prepare_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module load python/3.11
cd "$SLURM_SUBMIT_DIR"

~/.venvs/quantization_lrl/bin/python -m qlrl.data.prepare configs/data.yaml "$@"
