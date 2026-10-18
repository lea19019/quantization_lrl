#!/usr/bin/env bash
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
# Test job. Description only. Claude Code, 2026-10-17, at Adrian's request.
#
# • Roadmap:
#   1. #SBATCH header: 1 node, 1 task, 2 CPUs, 16G, 10 minutes, --qos=cs, --gpus=1.
#   2. #SBATCH --output=results/slurm/%x-%j.out (create the folder first).
#   3. set -euo pipefail, LC_ALL=C, OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK.
#   4. HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1 (compute nodes may have no internet).
#   5. module load python/3.11; cd "$SLURM_SUBMIT_DIR".
#   6. Use ~/.venvs/quantization_lrl/bin/python (GPU torch; the repo's .venv is CPU-only).
#   7. Run rtn, gptq and perplexity with --test.
#   8. Submit from the repo root: sbatch slurm/test_quantize.sh.
#
# • Still to validate:
#   - sacct shows State COMPLETED.
#   - The log has no traceback.
#   - Every output folder has its manifest.
#   - Note Elapsed and MaxRSS to size the full job.
