# Project: scope, priority and decisions

Adrian Castillo, CS 698R Master's Project, Fall 2026. Advisor meetings Thursdays 2 pm.
Week-9 draft report mid-semester. Final report and committee presentation finals week.
Goal: 130 logged work hours by 2026-12-10; compute is not the constraint, hours are.

This file is the single source of truth for scope, priority and decisions. The problem is
in [problem.md](problem.md), the question and hypotheses in [question.md](question.md),
how results are judged in [evaluation.md](evaluation.md), the experiments in
[design.md](design.md), the data in [data.md](data.md). When in doubt about what to do
next, come back here.

## 1. In one sentence

We do not know why post-training quantization removes more of a model's ability to
predict the next word in low-resource languages than in high-resource languages, or
whether that difference is a property of the language rather than of the model, the
method and the measurement ([problem.md](problem.md)).

## 2. Hypotheses

Two hypotheses and a null, stated with their reasoning in [question.md](question.md):

- **H1, under-fitted languages.** A language with a small share of the training data sits
  further from its own loss minimum, so the same rounding costs it more.
- **H2, where the damage lives.** Rounding error concentrates in specific modules,
  dimensions and experts, and low-resource languages depend on those more.
- **The null.** The gap is model, method, metric and noise, not language. Tested first;
  every claim must beat it.

The other candidate explanations remain listed in
[explanations.md](../concepts/explanations.md) as controls or extensions. Pre-set criteria
for each hypothesis are written with the experiment design and confirmed with the advisor
before any run.

## 3. Core vs add-ons

**Core (complete and written up by the week-9 draft):**

- Model: Llama 3.1 8B. All H2 experiments run on it alone.
- GPT-2 small trained by us, once, early, with shares per language set by us.
- Languages: English, French, Swahili, Yoruba, Zulu.
- Methods: RTN, GPTQ (act-order), AWQ, SmoothQuant (W4A8), bitsandbytes NF4; 4, 3 and
  2 bits where supported; five calibration draws for calibration-based methods.
- Evaluation: COMET and human evaluation as the goal metrics, per-word damage as the
  diagnostic ([evaluation.md](evaluation.md)).
- Experiments: phase 0 baseline and null; phase 1 gradient norm and curvature, per-module
  map, internal statistics; phase 2 the GPT-2 learning curve; phase 3 the fine-tune-and-
  requantize intervention, activation patching, logit lens ([design.md](design.md) §4).
- Deliverable: coverage table filled; one results table per hypothesis; a written verdict
  per hypothesis.

**Add-ons, in priority order (only after core is done):**

1. Qwen 2.5 7B and the Gemma 4 pair (26B A4B and 31B): replicate the ordering; expert
   routing per language on the MoE.
2. Target-language and mixed calibration cells for GPTQ and AWQ.
3. Cross-lingual translation (sw↔zu, yo↔sw) and off-target rate; extended human evaluation.
4. An intervention aimed at whatever the core points to: mixed precision on sensitive
   modules, target-language calibration, or fine-tune-then-requantize.

**Rule:** if core is not done by week 9, add-ons wait. A bug found on model four is a bug
fixed four times. Get the pipeline right on one model first.

## 4. Progress and remaining-hours plan

As of 2026-09-28, [hours.csv](../course/hours.csv) records **21.3 of 130 hours (16.4%)**.
The work so far established the repository and research design, reviewed the literature,
worked through RTN and GPTQ concepts and math, produced an initial RTN implementation
and an early GPTQ implementation, and investigated the evaluation metrics.

From 2026-09-29 through the 2026-12-10 deadline there are **10 weeks and 2 days**.
That leaves **108.7 hours**, an average of **10.6 hours per week**. The allocation is:

| Dates | Hours | Work |
|---|---:|---|
| 2026-09-29–2026-10-04 | 8.7 | Finish the method foundations; settle scorer and metric decisions; prepare validation. |
| 2026-10-05–2026-10-18 | 20 | Validate RTN and GPTQ on WikiText-2; build scorer sanity checks, data and COMET pipeline; then AWQ, SmoothQuant and NF4. |
| 2026-10-19–2026-11-01 | 20 | Phase 0 on Llama 3.1 8B; test the null; fix the human-evaluation subset and arrange raters. |
| 2026-11-02–2026-11-15 | 20 | Phase 1 measurements; quantize and score GPT-2 checkpoints; collect human ratings. |
| 2026-11-16–2026-11-22 | 10 | Write the draft report; make the phase-3 and add-on go/no-go decision with the advisor. |
| 2026-11-23–2026-12-06 | 20 | Run approved phase-3 work or add-ons; freeze results; complete analysis and figures. |
| 2026-12-07–2026-12-10 | 10 | Finish the report and committee presentation. |

These are planning targets. Actual sessions remain recorded by category in
[hours.csv](../course/hours.csv), and the plan is recalculated from those entries.

## 5. Deliverables

- Written and verbal reports; draft report at week 9 covering the core.
- Repository with code, configs, tests and experimental results.
- Trained GPT-2 checkpoints and the fine-tuned model.
- Human evaluation ratings and protocol.

## 6. Open questions (resolve with advisor or by checking)

- GPT-2 training corpora, share vector, and epoch repetition for the small languages.
- The advisor's "learning curve" reading of H1: confirm it means the data-scaling curve
  per language.
- Lab text per language: token counts, parallel or not, cleaning, FLORES overlap.
- Qwen 2.5 7B and Gemma 4 official language lists: sw, yo, zu.
- Human evaluation: raters available per language, and French as control.
- SmoothQuant at W4A8 only, or W8A8 too.
- Pre-set criteria and the *proposed* numbers in design.md and evaluation.md.
- Verify: Zhong 2025 venue; Marie & Fujita 2025 title/version to cite.
