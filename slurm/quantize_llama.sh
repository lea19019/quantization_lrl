#!/usr/bin/env bash
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md
set -euo pipefail

repo_root=${QLRL_REPO_ROOT:-$(realpath "$(dirname "$0")/..")}
model_path=${2:-meta-llama/Llama-3.1-8B}
output_root=${3:-"$repo_root/models"}
python_bin=${QLRL_PYTHON:-/home/vacl2/.venvs/quantization_lrl/bin/python}

run_quantizer() {
    export LC_ALL=C
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export HF_HOME=${HF_HOME:-/home/vacl2/.cache/huggingface}
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    module load python/3.11
    cd "$repo_root"

    test_args=()
    if [[ ${QLRL_TEST:-0} == 1 ]]; then
        test_args+=(--test)
    fi
    read -ra extra_args <<< "${QLRL_EXTRA_ARGS:-}"
    "$python_bin" -m "qlrl.quant.$QLRL_QUANTIZER" \
        "$QLRL_MODEL_PATH" "$QLRL_OUTPUT_FOLDER" "${test_args[@]}" "${extra_args[@]}"
}

submit_job() {
    local quantizer=$1
    local test_flag=$2
    local output_folder=$3
    local time_limit=$4
    local memory=$5
    sbatch --parsable \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem="$memory" \
        --time="$time_limit" \
        --qos=cs \
        --gpus=1 \
        --job-name="llama-$quantizer" \
        --output="$output_root/slurm-%x-%j.out" \
        --export="ALL,QLRL_REPO_ROOT=$repo_root,QLRL_QUANTIZER=$quantizer,QLRL_TEST=$test_flag,QLRL_MODEL_PATH=$model_path,QLRL_OUTPUT_FOLDER=$output_folder" \
        "$0" run
}

if [[ -n ${SLURM_JOB_ID:-} ]]; then
    run_quantizer
    exit 0
fi

mode=${1:-}
if [[ $mode != test && $mode != full ]]; then
    echo "Usage: $0 {test|full} [model-path] [output-root]" >&2
    exit 2
fi
if [[ ! -x $python_bin ]]; then
    echo "CUDA Python environment not found: $python_bin" >&2
    exit 1
fi

mkdir -p "$output_root"
if [[ $mode == test ]]; then
    submit_job rtn 1 "$output_root/test-llama-rtn-4bit" 00:10:00 16G
    submit_job gptq 1 "$output_root/test-llama-gptq-4bit" 00:10:00 16G
else
    submit_job rtn 0 "$output_root/llama-3.1-8b-rtn-4bit" 02:00:00 64G
    submit_job gptq 0 "$output_root/llama-3.1-8b-gptq-4bit" 08:00:00 64G
fi
