#!/usr/bin/env bash
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
# Pretraining of the from-scratch model (qlrl/pretraining.py, one process per GPU).
#
# The pretraining runs in the background for days or weeks, so it uses few of the lab's GPUs:
# one GPU by default, at most two, one job at a time. Each job goes to the fastest GPU type
# with enough free GPUs at submit time (B200, then H100, then A100); when none is free it waits
# for whichever of the three frees first. The qos stops a job at 24 h; the run saves and stops
# after train.hours (23 h), and the job then submits the next one, which resumes from
# resume.pt. The chain ends by itself once models/<name>/manifest.json exists. If a job crashes
# (non-zero exit) or ran under an hour, nothing more is submitted: look at slurm/output/ and
# start it again to resume. The GPU type and count may change between jobs: every step trains on
# the same windows either way.
#
# Meanwhile other experiments use the remaining slots. Every save_every steps a checkpoint
# appears in <checkpoints>/<name>/step-<n>/ as a plain Hugging Face model, so the quantization
# and evaluation configs can point at it as it arrives and the results build up during the run.
#
# Start it with bash from the repo root, on the login node; extra sbatch options are kept for
# every job of the chain:
#   mkdir -p slurm/output
#   bash slurm/pretrain.sh configs/pretrain/llama_test.yaml --gpus=2 --time=00:15:00
#   bash slurm/pretrain.sh configs/pretrain/llama_576m.yaml               1 GPU
#   bash slurm/pretrain.sh configs/pretrain/llama_576m.yaml --gpus=2      2 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --output=slurm/output/%x-%j.out

set -euo pipefail
config=$1
shift
options=("$@")
name=$(awk '/^name:/ {print $2}' "$config")
model=$(awk '/^output:/ {print $2}' "$config")/$name
gpus=1
for option in "${options[@]}"; do
    case $option in --gpus=*) gpus=${option#--gpus=} ;; esac
done

partition() {  # the fastest partition with a node that has $gpus free GPUs now, else all three
    for p in cs3 cs2 cs; do  # B200, H100, A100
        free=$(sinfo -h -N -p "$p" -t idle,mix -O "Gres:40,GresUsed:60" | awk '
            { split($1, a, ":"); split(a[3], t, "("); split($2, b, ":"); split(b[3], u, "(")
              if (t[1] - u[1] > best) best = t[1] - u[1] }
            END { print best + 0 }')
        if [ "$free" -ge "$gpus" ]; then echo "$p"; return; fi
    done
    echo cs3,cs2,cs
}

submit() {
    sbatch -J "$name" --partition="$(partition)" "${options[@]}" slurm/pretrain.sh \
        "$config" "${options[@]}"
}

if [ -z "${SLURM_JOB_ID:-}" ]; then  # started by hand: queue the first job
    submit
    exit
fi

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
    -m qlrl.pretraining "$config"

# Not finished: the job stopped at train.hours. Queue the next one.
if [ ! -f "$model/manifest.json" ] && [ "$SECONDS" -gt 3600 ]; then
    submit
fi
