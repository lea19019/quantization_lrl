#!/usr/bin/env bash
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md
# Usage: sbatch slurm/eval_wikitext.sh <model-path>... [--test]   (submit from the repo root)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --job-name=wikitext-ppl
#SBATCH --output=results/wikitext2/slurm-%x-%j.out
set -euo pipefail

export LC_ALL=C
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HOME=${HF_HOME:-/home/vacl2/.cache/huggingface}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
module load python/3.11
cd "$SLURM_SUBMIT_DIR"

/home/vacl2/.venvs/quantization_lrl/bin/python -m qlrl.eval.perplexity "$@"
