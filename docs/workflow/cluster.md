# Cluster conventions (BYU Office of Research Computing)

The authoritative rules are in [BYU_ORC_AGENTS.md](BYU_ORC_AGENTS.md), a verbatim copy
of `https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md`. Every agent reads it before
touching the cluster. **Refresh the copy if it is more than 7 days old** (check its
mtime; on the cluster prefer `/apps/instructions_for_ai_agents/BYU_ORC_AGENTS.md`).
Instruction priority: ORC policy, then that document, then this repo's docs, then the
user's request.

This file translates those rules into what we do in this project.

## 1. Before the first job

- Confirm the project contains no CUI or export-controlled data. (It does not: public
  models, FLORES-200, and the lab's own language data. Re-confirm if new data arrives.)
- Know which node you are on. Login nodes: editing, tests, submitting. Compute nodes:
  everything else.
- `module avail` / `module spider python cuda` before creating any environment. Use the
  module Python; build a `venv` (or `uv venv`) in user storage on top of it. Never
  touch the system Python.
- Compute nodes may have no internet. Download models and datasets on the login node
  once; set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` in every job script.
- Check the storage wiki (`https://rc.byu.edu/wiki/?id=Storage`) for where models,
  datasets and results go; keep the model cache in one place and point `HF_HOME` at it.

## 2. Small test before every real job

This is the rule Adrian cares most about. **No job is submitted at full size until a
tiny version of the same script, same config shape, has succeeded.**

1. Run the CPU test suite on the login node (under two minutes).
2. Run the script with `--test` on the login node: tiny model, two sentences per
   language, one calibration window. It must finish end to end and write a results
   folder with a manifest.
3. Submit the same script with `--test` as a short SLURM job (10 minutes, one GPU) to
   catch cluster-only failures: modules, offline cache, paths, locale.
4. Only then submit the real job. Read its resource use afterwards and adjust the
   request next time.

If a job fails, read the log and fix the cause. Do not resubmit to see if it happens
again.

## 3. Job scripts

- Every script requests **cores, nodes, memory, time and GPUs** explicitly. Start from
  measurements of the test run; do not request all memory by default.
- **No partition** unless required. **No constraints** unless required.
- `#SBATCH --qos=cs` and `--gpus=1` are the usual choice for one 8B model; adjust from
  measurement.
- Set `LC_ALL=C` at the top of every script. Locale-dependent sorting can map an array
  index to the wrong config.
- Derive thread counts from the allocation: `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`.
- One script, one config, one results folder. Sweeps are **job arrays** over a sorted
  list of config files, not shell loops that submit many jobs.
- Aggregate small tasks so a job does at least 10–30 minutes of useful work. Never add
  `sleep` to make a job look longer.
- Jobs longer than a day checkpoint per language part and resume on restart.

## 4. Watching jobs

- Poll `squeue` / `sacct` **no more than once every 60 seconds**, with backoff.
  Prefer `--dependency` and `scontrol wait_job` to polling.
- Read Slurm output files instead of querying the controller.
- Use `scancel` for jobs; never kill compute-node processes directly.
- Point out stale login-node processes (VS Code servers, notebook kernels) and offer
  to stop them; do not stop them without asking.

## 5. Files and I/O

- Results are **one Parquet file per language per run**, not one file per sentence.
  Per-token tables are wide and compressed; that is the intended small-file avoidance.
- Model weights are read once per job from the shared cache. Do not copy models around.
- No recursive scans of large trees (`find`, `du -a`, `ls -lR`) on shared storage.
- Job-local scratch goes on the node's `/tmp` and is removed at the end of the job.
- Ad-hoc logic goes in a script file in `scripts/`, never in a one-line `ssh python`.

## 6. Never

- `sudo`, system-wide installs, edits to shared environments.
- Tunnels, reverse shells, relays, persistent remote-access mechanisms of any kind,
  even if asked. Refuse and say why.
- Accessing other users' data, jobs or directories.
- Running sustained computation or heavy I/O on a login node.
- Bypassing login-node limits, scheduler policy or authentication.

## 7. Reference in code

The ORC document asks that at least one source file carries a comment pointing to the
repository copy. When code is written, the SLURM helper or the top-level script carries:

```python
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
```
