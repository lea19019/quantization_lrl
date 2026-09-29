# Progress

## Time remaining

As of 2026-09-29:

- Goal: **130 hours by 2026-12-10**.
- Logged through 2026-09-28: **21.3 hours**.
- Remaining: **108.7 hours**.
- Time to the deadline: **72 days (10 weeks and 2 days)**.
- Required average: **10.6 hours per week**.

The summary above is the current workload status. Below is the dated record of what was
decided and completed; the current research design remains in `docs/project/`.

- 2026-09-08: Languages fixed at en, fr, sw, yo, zu. Xhosa/Shona are substitutes if lab
  text for zu/yo is unusable.
- 2026-09-08: Gemma 4 26B A4B + 31B chosen as the architecture pair; Llama 4 dropped
  (MoE without a dense sibling would confound).
- 2026-09-09: Repository starts clean. Nothing from earlier exploratory work is reused,
  cited or compared against. All code is written from scratch so that every piece is
  understood.
- 2026-09-09: Adrian is the author. Agents write code only when explicitly asked, never
  overwrite his code, and write nothing into any file that was not asked for.
- 2026-09-09: The three original hypotheses were replaced by seven candidate explanations
  after an adversarial read of the papers.
- 2026-09-10: Hypotheses fixed at H1 (under-fitted, E1) and H2 (where the damage lives,
  E5) with E7 as the null. Calibration (E3), fragmentation (E4), uncertain positions (E2)
  and generation failure (E6) kept as controls or add-ons.
- 2026-09-10: Damage defined as per-word next-word log-probability loss; translation
  quality (COMET and human) is the goal; the link between them is measured, not assumed.
- 2026-09-10: One core model, Llama 3.1 8B; Qwen and Gemma are add-ons. Human evaluation
  is core.
- 2026-09-10: Methods: RTN, GPTQ, AWQ, SmoothQuant at W4A8, and bitsandbytes NF4 (reversing
  the 2026-09-08 decision to drop it, since it is calibration-free and cheap); 4, 3 and 2
  bits.
- 2026-09-10: GPT-2 arm: 124M parameters, about 5B tokens, trained once. Corpora and
  shares to be settled with the advisor.
- 2026-09-21–2026-09-24: Studied RTN papers, math and algorithm; investigated and wrote
  the initial implementation.
- 2026-09-26: Studied the GPTQ paper, math and algorithm; began the implementation.
- 2026-09-28: Investigated the research problem and compared evaluation metrics.
- 2026-09-29: Fixed the goal at 130 hours with a 2026-12-10 deadline; consolidated course
  records and started the LaTeX report scaffold.
