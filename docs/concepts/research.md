# Research, projects and products: what is what

The vocabulary of research and the kinds of project work, so that at any point one can say
what the body of the current work is and what the next step is. Written 2026-09-11 for
Adrian's notes; it applies beyond this project.

## 1. Research versus development

**Research produces a claim nobody could make before, with evidence someone else can
check.** Development produces an artifact that meets a specification. The test for research
is "what do we know now that we did not know before, and how do we know it?" The test for
development is "does it work?" Most projects contain both. This project is research with a
development component inside it: the pipeline exists to produce the evidence.

## 2. The ladder of words

Each level narrows the one above it.

- **Problem.** A gap between what is known and what is needed: "we do not know why X",
  "we cannot do Y". Broad, motivating, not yet answerable. Here: `project/problem.md`.
- **Research question.** The problem cut to one thing an investigation can answer. Three
  tests: the answer is not already known; one can say in advance what evidence would answer
  it; it fits the time available. Here: the first line of `project/question.md`.
- **Hypothesis.** A candidate answer, stated so that it can be wrong. It makes a
  prediction: "if this is true we will observe X, otherwise Y." Not every question has
  hypotheses; "why" questions do, because "why" is answered by a mechanism and a mechanism
  is testable. Here: H1, H2 and the null.
- **Objective.** A thing that will be done to answer the question. Objectives are verbs:
  measure, build, compare, train. On a project form they are the deliverables a committee
  checks. Here: the phases in `project/design.md`.
- **Metric, criterion, verdict.** A metric is a number a procedure yields. A criterion is
  the rule that turns the number into a decision, written before the run. A verdict is the
  decision. Here: metrics in `project/evaluation.md`; criteria still to be written.
- **Project.** The container: scope, time, budget, deliverables, decisions. Here:
  `project/project.md`.

## 3. Kinds of research question

The kind says whether hypotheses are needed.

| Kind | Form | Needs hypotheses? | Example here |
|---|---|---|---|
| Descriptive | What is X like? | No; needs a measurement plan | How much does each language lose? (phase 0) |
| Explanatory | Why does X happen? | Yes, one per mechanism | Why do low-resource languages lose more? (core) |
| Predictive | Can Y predict X? | Yes: "Y predicts X" | Does gradient norm predict damage? (H1a) |
| Comparative | Is A better than B? | Yes, or a null | Does GPTQ beat RTN on Swahili? |
| Constructive | Can we build X that does Y? | A claim about the artifact | Can mixed precision close the gap? (add-on 4) |

This project's core question is explanatory, with a descriptive base and one predictive
piece.

## 4. Kinds of project and their spine

Every kind of work has one artifact that is its body. When going in circles, read that
artifact and ask which item in it is furthest from done. If it does not exist, writing it is
the next step.

| Kind | Starts with | The spine | Done when | When lost |
|---|---|---|---|---|
| Explanatory research | A problem | Question, hypotheses, predictions, criteria | Every hypothesis has a verdict with evidence | Write the criterion you cannot yet state |
| Descriptive research | A population and a measurement | The list of what to measure, on what, how reported | The list is measured and reported | Cut the list; it only grows |
| Constructive research (most ML method papers) | A claim that X is possible | The claim, the baselines, the evaluation fixed before building | The artifact exists and beats or fails the baselines | Reread the claim; the artifact drifts from it |
| Replication | A paper | Their setup, their numbers, a tolerance | Numbers match within tolerance; then deviate | Match one number before changing anything |
| Software product | A need | The specification: what, for whom, acceptance criteria | Acceptance tests pass | Pick the riskiest unmet requirement |
| Infrastructure or tooling | A slow or broken workflow | Baseline measurement, the change, the after measurement | The number improved | Measure before touching anything |
| Data analysis | A question and a dataset | The question, the data dictionary, the analysis plan | The plan is executed and reported | Write the plan before opening the data |
| Learning project | A skill one lacks | A curriculum and a test of understanding | One can rebuild it from memory or explain it | Rebuild the smallest piece without looking |

Two rules hold for all of them. The spine is written before the work and changed only
deliberately: in software that is the spec, in research the pre-registered design. The next
step is always the item in the spine most likely to make everything else wasted; in
research that is usually the evaluation, because a result that cannot be judged is worth
nothing.

## 5. When must the metrics be fixed?

The goal metric and the decision criteria are fixed before any result is read. They need
not be fixed before code is written. Choosing the metric after seeing results is the
classic way to fool oneself: there are always many metrics and one will agree with you.
The quantizers, the loaders and the per-token scorer do not depend on the criterion; they
depend on what is already fixed (languages, FLORES, log-probability per token, the
WikiText check). So:

- Code now: what has a settled spec.
- Settle before phase 0 is read: the criterion per hypothesis, the human evaluation
  protocol and which configurations are rated, the reliability of the automatic metric
  for each language.

Coding and settling run in parallel. What is never done: run, look at the numbers, then
decide what counts as a gap.

## 6. Human evaluation

People who speak the language judge the outputs. It is the reference standard for
translation quality; automatic metrics are validated only by their agreement with human
judgments, and COMET is a network trained to predict human scores. Comparing the two is
therefore a check on whether the automatic number can be trusted for a language.

**Protocols**, cheapest to most reliable:

- **Pairwise preference.** Source plus two translations, hidden and shuffled; pick the
  better or "same". Cheap, high agreement, relative answer only.
- **Direct Assessment (DA).** Source (or reference) plus one translation, scored 0–100.
  WMT 2017–2021. Scores are z-normalised per rater. The modern variant adds anchors
  (0 nonsense, 33 some meaning, 66 most meaning kept, 100 perfect).
- **Error Span Annotation (ESA).** DA plus marking the wrong spans. WMT 2024.
- **MQM.** Every error marked with a category (mistranslation, omission, grammar,
  terminology) and severity (minor, major, critical); score is a weighted error count.
  Most reliable; raters need training; several times slower. Freitag et al. 2021.

**Choices written down before collecting anything.** Source-based (bilingual rater) or
reference-based (monolingual rater); blind and shuffled; number of raters and overlap
(two raters on everything gives agreement; Krippendorff's alpha above about 0.67 is
usable, above 0.8 good); attention checks (deliberately broken outputs; a rater who scores
them high is dropped); fatigue (about 30 s per DA item, sessions under 90 minutes).

**Analysis**, in order: agreement between raters (if low, stop); mean per configuration
after per-rater normalisation, bootstrap interval over sentences; human loss as
full-precision mean minus quantized mean on the same sentences, paired test (Wilcoxon
signed-rank or paired bootstrap); comparison to the automatic metric at system level (does
the ordering of configurations agree?) and segment level (rank correlation of per-sentence
losses; low segment correlation is normal, low system correlation means the metric is
wrong for that language).
