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
- 2026-10-18: AfriCOMET 1.1 scores all six African-core languages: English, French,
  Yoruba, Zulu, Swahili and Amharic (Swahili and Amharic are project languages; the
  2026-09-29 exclusion no longer holds).
- 2026-10-18: Translation prompt: three dev pairs of the same test set, as
  `<eng>…</eng> <yor>…</yor>`; greedy; stop at the closing tag or 2x the source tokens.
- 2026-10-18: Two supporting measurements added: the log-probability of the human
  reference given the translation prompt, and the off-target rate with GlotLID.
- 2026-10-18: Fine-tuning: one multilingual model on every language, no per-language
  models; replaces the three models of 2026-10-13.
- 2026-10-18: Nahuatl and Bribri X→English are scored against machine-translated English
  references; kept, and stated as a limitation.
- 2026-10-18: SmoothQuant dropped from the methods and the code.
- 2026-10-19: Generation cap: 2x the source tokens times the test set's dev-set ratio of
  target to source tokens. The flat 2x cut every English→Amharic output (Amharic needs
  about 7.8x the tokens of English).
- 2026-10-19: Off-target rate: English→X counts outputs GlotLID identifies as English;
  X→English counts outputs it does not. Per-language label lists dropped: GlotLID cannot
  separate close varieties (it labels 91% of the Kichwa references `qvi`).
- 2026-10-19: One test set per language: FLORES-200 devtest for French, Spanish,
  Portuguese, Yoruba, Zulu, Swahili, Amharic and Guarani; MayanV for Q'eqchi';
  AmericasNLP for Nahuatl and Bribri; church for Kichwa. Other splits are free for
  fine-tuning and its validation.
- 2026-10-19: GPTQ and AWQ calibration windows start with BOS (`bos: true`). Calibrated
  without BOS, GPTQ was worse than RTN on every language when sentences start with BOS
  (Amharic damage 7.8 vs 3.5 nats per word); with BOS windows it is 2.3. The WikiText-2
  validation check stays as in the papers.
- 2026-10-19: Five calibration seeds (0-4) for GPTQ and AWQ on Llama 3.1 8B; RTN once.
- 2026-10-19: Greedy decoding kept. Repetition loops of the base model on low-resource
  targets are expected and not corrected.
- 2026-10-19: Results of the first full evaluation (no BOS calibration, flat cap, all test
  sets) are kept in `results/eval/`; the new evaluation writes to `results/eval_v2/`.
