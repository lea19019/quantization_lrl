#!/usr/bin/env bash
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
# Full job. Description only. Claude Code, 2026-10-17, at Adrian's request.
# Submit only after test_quantize.sh passes.
#
# • Roadmap:
#   1. Same header and environment as test_quantize.sh.
#   2. Set --mem and --time from the test job's MaxRSS and Elapsed, with margin.
#   3. --array=0-4 for the five calibration seeds; pass --seed $SLURM_ARRAY_TASK_ID.
#   4. Run rtn, gptq and perplexity at full size.
#   5. One results folder per run, each with its manifest.
#   6. Submit: sbatch slurm/quantize.sh.
#   7. Watch: squeue -u $USER at most once a minute; read the log in results/slurm/.
#   8. After: sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ReqMem.
#
# • Still to validate:
#   - gptq.py creates torch.eye and torch.zeros on the weight's device (GPU).
#   - All five seeds finish and write their own folders.
#   - MaxRSS and Elapsed are well under what was requested.
