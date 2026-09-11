# Conventions: how we work in this repo

These rules bind every person and every agent working here. The agent-instruction file
[CLAUDE.md](../../CLAUDE.md) points to this file. Cluster-specific rules are in
[cluster.md](cluster.md).

## 1. Authorship

- **Adrian is the author.** He owns the ideas, the experiments, the decisions and the
  code. Agents advise; they do not decide and they do not tell him what to do.
- **Agents help him understand.** Research, read papers, explain concepts, find things,
  present options with trade-offs, review, run and supervise jobs.
- **Agents write scaffold and heavy lifting only when explicitly asked** for that
  specific piece: boilerplate, data loading, plotting, SLURM wrappers, long mechanical
  edits. The logic of an experiment is written by Adrian, or explained by Adrian first so
  it is clear he understands it before an agent types it.
- **Example code on request.** Adrian may ask for an example implementation of
  anything to learn from. Examples go in `examples/`, never inside the package, each
  file headed by a comment saying it is an example. What moves from there into the
  package is his call, and he does the moving.
- **Never overwrite code Adrian wrote.** If something looks wrong, add a comment that
  says what and why, or say it in the reply. Change it only when he says so.
- **Write nothing that was not asked for, in any file.** An agent writes into a file only
  when the current message asks it to, and only what that message asks. No extra
  paragraphs, follow-ups, summaries, caveats or side edits. A past request is not standing
  permission. When told to document something, document that thing, where named, and stop.
- **No experiments, tests, scripts or configs appear in this repo unless Adrian asked**
  for them. "Set things up" does not mean "write code".
- **This repo started clean on 2026-09-09.** No earlier exploratory code, numbers or
  results are reused, cited or compared against. Nothing in `docs/` refers to them.

## 2. Code style

Modelled on Karpathy's minGPT / nanoGPT / nanochat: few files, each one readable top to
bottom, no framework underneath.

- **Small files.** A source file is under 200 lines. At 150, split it. The same limit
  holds for docs, configs and tests. A file that must be longer is a sign the design is
  wrong, not a reason for an exception.
- **Small functions.** One function does one thing; under 40 lines; at most five
  parameters. A function that needs a comment to explain its sections is two functions.
- **Plain Python.** Standard library, `torch`, `numpy`, `transformers`, `datasets`,
  `pyyaml`. No custom base classes, registries, decorators or plugin systems. No
  metaprogramming. Prefer a function over a class; a class only when it holds state.
- **Explicit over clever.** Loops that show the math beat one-line tensor tricks that
  hide it. If a one-liner is needed for speed, keep the readable version in a comment.
- **Comments explain concepts and math, not syntax.** A quantizer says what the rounding
  grid is and cites the paper and equation. A scorer says what `d_gold` is. Anything
  that a reader with an ML background might find hard gets a short comment. Nothing else
  does. Do not over-comment.
- **Docstrings** on every public function: one line saying what it returns, plus units
  or shapes for tensors. Type hints on signatures.
- **Names** are full words: `gold_logprob`, not `glp`. Tensor shapes in variable names
  or comments when they are not obvious (`hidden: [batch, seq, dim]`).
- **No hidden state.** Randomness is seeded from the config. Every number that could
  change a result lives in the config, not in a default argument.
- **Formatting and lint** follow `ruff` with the default rules plus `E501` at 100
  columns. No exceptions committed.

## 3. Layout (to be created only when Adrian asks)

```
quantization_lrl/
  CLAUDE.md          agent instructions (points here)
  docs/              project, design, background, literature, papers index, conventions
  papers/            PDFs, named FirstAuthor-Year-Short-Title.pdf
  qlrl/              the package, one module per concept (see §4); Adrian's code
  examples/          example implementations written on request, for learning only
  configs/           one YAML per run; nothing else
  scripts/           entry points; each is a thin wrapper around package functions
  slurm/             job scripts; each calls one script with one config
  tests/             CPU-only tests; run before every submission
  results/           one folder per run, named by config; see §5
```

## 4. Modules (planned shape; each file under 200 lines)

| Module | Holds |
|---|---|
| `qlrl/quant/rtn.py` | round-to-nearest, grid, group size |
| `qlrl/quant/gptq.py` | Hessian, column-by-column update, act-order |
| `qlrl/quant/awq.py` | per-channel scale search |
| `qlrl/quant/smoothquant.py` | migration scales, W8A8 fake quant |
| `qlrl/quant/apply.py` | walk a model, replace weights, one module at a time if asked |
| `qlrl/data.py` | FLORES loading, calibration windows, language codes |
| `qlrl/score.py` | per-token measures: `d_gold`, KL, flips, margin, entropy |
| `qlrl/probe.py` | hooks for residual norms and activation statistics |
| `qlrl/stats.py` | bootstrap CIs, eta-squared, Spearman |
| `qlrl/manifest.py` | git sha, config, versions, seed |
| `qlrl/config.py` | load and validate YAML |

One concept per file. A file that imports from more than three sibling modules is
doing too much.

## 5. Experiments and results

- **One YAML per run.** The config names the model, method, bits, group size, seed,
  languages and calibration set. Nothing is passed on the command line except the
  config path and optional `--test` for a tiny run.
- **Results folder per run**, named after the config: `results/<config-name>/`.
  It holds `manifest.json`, the per-token table (Parquet, one file per language) and a
  short `summary.md` written by the analysis script. Never overwrite a manifest; a
  resumed run writes a timestamped one next to it.
- **Per-language parts with resume.** A crash at language four of five does not cost the
  first three.
- **Every result table names its seed count and its CI method.**
- **Analysis scripts read from `results/` and write figures to `results/figures/`.**
  Nothing is computed in a notebook that is not also in a script.
- **Three seeds minimum** for anything calibration-based. RTN is exactly reproducible.

## 6. Tests and fitness functions

Tests are CPU-only, run in under two minutes, and run before every SLURM submission.

**Fitness functions** (tests that enforce these conventions on the code itself):

- every `.py` file under `qlrl/`, `scripts/`, `tests/` is under 200 lines;
- every function is under 40 lines and has at most five parameters;
- every public function has a docstring and type hints;
- no file in `qlrl/` imports from more than three sibling modules;
- `ruff check` and `ruff format --check` pass;
- every config in `configs/` loads and validates.

**Correctness tests** (written before GPU time):

- RTN: every quantized weight is within half a step of the original.
- GPTQ equals RTN under an identity Hessian; GPTQ beats RTN on correlated inputs.
- Scorer: a model scored against itself gives zero damage on every measure.
- Final logits are float32.
- Every script runs end to end on a tiny model (`--test`) on CPU.

**Before any GPU result is read:** the quantizer matches published WikiText-2
perplexities within a stated tolerance.

## 7. Documentation

- **Write only what changes what a reader does.** No restating, no filler, no
  "for context" paragraphs, no summaries of things a link already says. If a sentence
  can be deleted without losing a decision, a definition or an instruction, delete it.
- **Only what sticks.** A doc holds information that will be read again: definitions,
  the current design, methods, results. Not what changed, not what was reversed, not
  "we used to". The one place for history is [progress.md](progress.md). Do not create a
  doc, section or file the author did not ask for; say it is missing and wait.
- **`project/` is the source of truth** and follows a paper's shape: problem, question,
  evaluation, design, data, and later `background.md`, `results.md`, `conclusion.md` as
  they are needed. Results there are the condensed numbers only. Full results live in a
  separate results folder, still readable by a person, never a dump.
- **One concepts file per method.** A method named in `project/` (a quantizer, an
  interpretability technique) gets one file in `concepts/` that says what it is, and no more.
- **Keep files in step.** When something is settled (the data, a model, a number), the
  files it touches are updated in the same turn, in this style, and nothing else is.
- **One folder per line of thought** under `docs/` (`project`, `concepts`, `papers`,
  `workflow`, `proposal`). A new doc goes into its folder and gets a line in
  [../README.md](../README.md). A new line of thought gets a new folder, never a loose
  file at the top. No numbers or prefixes in names; order lives in the map.
- **First line of every doc says what it is for.** If that cannot be said in one line,
  the doc is two docs.
- Every doc under 200 lines. The one exception is `workflow/BYU_ORC_AGENTS.md`, a
  verbatim external copy that must not be edited.
- Plain language. Explain a concept the first time it appears, or link to
  [../concepts/quantization.md](../concepts/quantization.md).
- Dates are absolute (2026-09-09), never "yesterday".
- A decision changes the file it touches. The dated record of decisions and progress is
  [progress.md](progress.md), history only, one line per entry.
- Hours are logged by category from day one in [hours.csv](hours.csv).
