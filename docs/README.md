# Docs

> **Docs ahead of code (2026-10-15).** The code was reset to Adrian's RTN and GPTQ of
> 2026-09-29 (commit `2de1c37`) to be rebuilt step by step. These docs still describe the
> earlier, fuller codebase: scripts they name (e.g. `split_church.py`, `decontaminate.py`,
> `qlrl/eval/`, `slurm/`) no longer exist on `main`. That code is on branch
> `archive/before-reset`; what each piece did is summarised in `qlrl/*.py` descriptions.

One folder per line of thought. Read in this order.

| Folder | Holds | Files, in reading order |
|---|---|---|
| `project/` | What we are doing | `problem.md` (the problem, defined), `question.md` (the question and the hypotheses), `evaluation.md` (how results are measured, compared and judged), `design.md` (models, methods, experiments, phases), `data.md` (what text is used for what), `church_data.md` (church corpus layout and use), `project.md` (scope, timeline, open questions) |
| `concepts/` | What we must understand | `quantization.md` (PTQ in plain language), `explanations.md` (the candidate explanations E1–E7), `literature.md` (what is known, what nobody measured), `research.md` (research vs development, kinds of project, human evaluation) |
| `papers/` | The PDFs in `../papers/` | `index.md` first; the rest are one entry per paper, by theme, lookup only |
| `workflow/` | How we work | `conventions.md` (code, tests, results, docs), `decisions.md` (dated decision history), `cluster.md` (BYU cluster), `BYU_ORC_AGENTS.md` (verbatim policy) |
| `course/` | Course records and deliverables | `roadmap.md` (remaining work), `progress.md` (time remaining and completed work), `hours.csv` (work sessions), `proposal_draft.md` and `698R_Proposal_Form.pdf` (proposal), `report/report.tex` (report) |

Start with `project/problem.md`, then `project/project.md`, then `concepts/quantization.md`,
then `concepts/explanations.md`, then `project/design.md`.
