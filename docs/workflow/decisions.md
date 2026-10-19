# Decisions

A dated record of project decisions, including superseded choices. The current design
lives in `docs/project/`.

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
- 2026-09-29: African core fixed at English, French, Yoruba and Zulu. Zulu supplies
  Bantu–Yoruboid contrast while retaining FLORES, human-evaluation and AfriCOMET coverage;
  Swahili and Xhosa are excluded.
- 2026-10-13: Church data comes from the 2026 release, now readable for every project
  language (counts in [church_data.md](../project/church_data.md)).
- 2026-10-13: Llama 3.1 8B is fine-tuned three times, all full fine-tuning: one
  multilingual model on every language, and one model each for Yoruba and Zulu, to see
  whether the per-language and multilingual results differ.
- 2026-10-13: English is the only anchor: training and main evaluation are English–X.
  Spanish–X zero-shot evaluation is set aside for now.
- 2026-10-13: External Spanish–X data (AmericasNLP, MayanV, Kichwa lexicon) gets a
  synthetic English source by machine-translating its Spanish side; the target side stays
  human-written. The list of what to translate is in [data.md](../project/data.md).
- 2026-10-13: Spanish sides are translated with Qwen3-32B (Apache-2.0), greedy, prompt of
  three FLORES+ dev pairs in the format `<spa>…</spa> <eng>…</eng>`.
- 2026-10-13: The church data (already cleaned by the lab) is split by line, not by
  document: 1,000 dev and 1,000 test per language. Near-duplicate: equal after lowercasing
  and removing digits and punctuation; such pairs leave train. External train sets are
  cleaned with the lab pipeline; external dev and test sets are left as published.
- 2026-10-13: Translation evaluation: base model, three dev examples per test set (the
  same for every model), both directions, greedy, answer capped at 2x the source tokens,
  full test sets. Damage is measured on the bare FLORES+ devtest sentence.
- 2026-10-18: SmoothQuant dropped from the methods and the code.
