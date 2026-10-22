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
- Languages: English as the anchor; French, Spanish and Portuguese as high-resource
  controls; Yoruba, Zulu, Swahili and Amharic (African); Q'eqchi', Guarani, Nahuatl,
  Kichwa and Bribri (Latin American).
- Methods: RTN, GPTQ (act-order), AWQ, bitsandbytes NF4; 4, 3 and
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

As of 2026-10-22, [hours.csv](../course/hours.csv) records **66.0 of 130 hours (50.8%)**,
through 2026-10-20. Phase 1 is done: low-resource languages lose more on Llama 3.1 8B,
Qwen 2.5 7B and the fine-tuned Llama, at 4 and 3 bits ([progress.md](../course/progress.md)).

From 2026-10-22 through the 2026-12-10 deadline there are **7 weeks**. That leaves
**64.0 hours**, an average of **9.1 hours per week**. The allocation is:

| Dates | Hours | Work |
|---|---:|---|
| 2026-10-22–2026-10-28 | 9 | Write the phase-1 results and set-up; list the open questions. |
| 2026-10-29–2026-11-18 | 28 | Experiments on where in the model the extra damage happens. |
| 2026-11-19–2026-12-02 | 18 | Finish the report. |
| 2026-12-03–2026-12-10 | 9 | Committee presentation. |

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
- Pre-set criteria and the *proposed* numbers in design.md and evaluation.md.
- Verify: Zhong 2025 venue; Marie & Fujita 2025 title/version to cite.
