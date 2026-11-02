#!/usr/bin/env bash
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
# Pretraining on every GPU of one node (qlrl/pretrain.py, one process per GPU). The qos stops a
# job at 24 h: the run saves before (train.hours) and the next job with the same name resumes,
# so submit the chain at once; a job that finds the model trained exits. From the repo root:
#   mkdir -p slurm/output
#   sbatch -J llama-test --gpus=2 --time=00:15:00 slurm/pretrain.sh configs/pretrain/llama_test.yaml
#   for i in 1 2 3; do sbatch -J llama-576m --dependency=singleton slurm/pretrain.sh \
#       configs/pretrain/llama_576m.yaml; done
#SBATCH --job-name=pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --qos=cs
#SBATCH --gpus=8
#SBATCH --output=slurm/output/%x-%j.out

set -euo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=4
# Compute nodes may have no internet: the tokenizer and token files are local.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module load python/3.11
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# GPU torch lives in this venv; the repo's .venv is CPU-only.
~/.venvs/quantization_lrl/bin/torchrun --standalone --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    -m qlrl.pretrain "$1"
