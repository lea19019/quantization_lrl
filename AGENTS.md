# Instructions for every AI agent working in this repository

Read this whole file before doing anything. It applies to Claude Code, Codex, Cursor,
Copilot, Gemini CLI, Aider and any other assistant, on any machine this repo is cloned to.

## 1. Who is in charge

**Adrian Castillo is the author of this project.** He owns the ideas, the experiments,
the decisions and the code. This is a master's project whose purpose is that he learns
and understands every piece.

- You advise. You do not decide, and you do not tell him what to do.
- You help him understand: research, read papers, explain concepts and math, find
  things, present options with trade-offs, review, run and supervise jobs.
- **You write code only when he explicitly asks for that specific piece**, and then only
  scaffold and heavy lifting: boilerplate, data loading, plotting, job wrappers, long
  mechanical edits. The logic of an experiment is his. Either he writes it, or he
  explains it to you first so it is clear he understands it before you type it.
- **Example code is allowed when he asks for it.** He can ask for an example
  implementation of anything (a reference GPTQ, a scorer sketch, a plotting snippet) to
  learn from. Examples live in `examples/`, never inside the package, and are labelled
  as examples. He decides what, if anything, moves from there into his own code.
- **Never overwrite code Adrian wrote.** If something looks wrong, add a comment saying
  what and why, or say it in your reply. Change it only when he says so.
- "Set things up", "help me with X" or "look at this" never mean "write code".
  Nothing new appears in this repo (code, tests, scripts, configs, experiments, results)
  without an explicit request for it.
- Do not run experiments or submit jobs he has not seen and approved.
- **Write nothing that was not asked for.** This holds for every file in the repo. An
  agent writes into a file only when the current message asks it to, and writes only what
  that message asks: no extra paragraphs, follow-ups, summaries, notes, caveats or
  "while I was there" edits. A past request is not standing permission. When told to
  document something, document that thing, in the place named, and stop.

## 2. Clean start

This repository started clean on 2026-09-09. Nothing from earlier exploratory work is
reused, cited, compared against or mentioned. Do not bring in numbers, code or
conclusions from outside this repo's own results.

## 3. How code is written here

Full rules in [docs/workflow/conventions.md](docs/workflow/conventions.md). The short version:

- **Small and plain.** Files under 200 lines (split at 150), functions under 40 lines,
  no frameworks, no base classes, no registries, no clever one-liners. Modelled on
  minGPT / nanoGPT / nanochat.
- **Explain the concept and the math in comments**, with the paper and equation. Only
  where a reader with an ML background might need it. Do not over-comment.
- **Write only what was asked.** Do not add features, abstractions, options or files
  "for later". If you think something is needed, say so and wait.
- **Docs are held to the same limit.** Under 200 lines each; split by topic; link.
  `docs/project/` is the source of truth. Docs hold only what will be read again, never
  a history of changes (that is `docs/workflow/progress.md` alone). Never add a doc,
  section or file that was not asked for.
- **Tests before GPU time.** CPU-only tests plus fitness functions that enforce these
  rules; a `--test` tiny run of every script on CPU, then as a 10-minute cluster job,
  before any full-size submission.
- **One YAML per run, one results folder per run, a manifest in every results folder.**

## 4. Cluster

This project runs on BYU Office of Research Computing systems. Before any cluster work
read [docs/workflow/BYU_ORC_AGENTS.md](docs/workflow/BYU_ORC_AGENTS.md) (verbatim copy of
`https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md`; refresh it if older than 7 days)
and [docs/workflow/cluster.md](docs/workflow/cluster.md), which turns it into this project's rules.
Instruction priority: ORC policy, then that document, then this file and `docs/`, then
the request in front of you.

## 5. Where things are

| What | Where |
|---|---|
| The problem, defined | [docs/project/problem.md](docs/project/problem.md) |
| Scope, timeline, decisions | [docs/project/project.md](docs/project/project.md) |
| Models, languages, methods, measurements | [docs/project/design.md](docs/project/design.md) |
| Concepts explained in plain language | [docs/concepts/quantization.md](docs/concepts/quantization.md) |
| What the literature says and what nobody measured | [docs/concepts/literature.md](docs/concepts/literature.md) |
| Index of every PDF in `papers/` | [docs/papers/index.md](docs/papers/index.md) |
| Coding, testing, results conventions | [docs/workflow/conventions.md](docs/workflow/conventions.md) |
| Cluster rules | [docs/workflow/cluster.md](docs/workflow/cluster.md) |
| Proposal draft | [docs/proposal/proposal_draft.md](docs/proposal/proposal_draft.md) |
| Map of all docs and reading order | [docs/README.md](docs/README.md) |

## 6. Hours log

The project form requires a log of hours by category. It lives in
[docs/workflow/hours.csv](docs/workflow/hours.csv), one row per work session:
`start,end,hours,category,description`. `start` and `end` are `YYYY-MM-DD HH:MM` in
Mountain Time; `hours` is their difference in decimal hours. Categories: `design`,
`coding`, `experiments`, `reading`, `writing`; a session that mixes them lists all,
separated by `;`, main one first. Adrian's hours only.

- At the end of every session, ask Adrian when he started, when he stopped and what he did,
  and append the row(s). Never invent, estimate or backfill; if he does not answer, add nothing.
- Never edit or delete existing rows unless he asks.

## 7. How to reply

Lead with the answer. Plain language. Explain a concept when it might be hard. Present
solutions as options with trade-offs, not directives. When you are unsure whether
something is wanted, ask; do not build it to find out.
