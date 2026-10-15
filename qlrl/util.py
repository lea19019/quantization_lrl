'''
What the archived data-preparation scripts did, and whether we need them again.
Description only, no code. Written by Claude Code on 2026-10-15 at Adrian's request.
The code itself is on branch archive/before-reset. Their outputs are kept in data/.

• split_church.py: church corpus -> train / dev / test
  Read the church data (English in src.txt, the language in tgt.txt, filtered/ folder).
  Per language, draw 1,000 random pairs for dev and 1,000 for test (seed 0); the rest is
  train. Then drop every train pair that is a near-duplicate of any dev or test sentence
  (same after lowercasing and keeping only letters). The English side is checked across
  all languages, since the same English sentence appears in many of them.
  Output: data/church/<code>/{train,dev,test}.{en,<code>} and split.json with the counts.
  Done for spa, por, fra, yor, zul, grn, kek, qug.
  Again? Yes, for Swahili and Amharic. And if a new language is split alone, the
  English-side check no longer covers every language at once, so all languages should be
  split again together.

• translate_spanish.py: give the external data an English side
  AmericasNLP (Guarani, Nahuatl, Bribri), MayanV (Q'eqchi') and the Kichwa lexicon pair the
  language with Spanish, not English. Qwen3-32B translated the Spanish side into English
  (three FLORES+ dev examples in the prompt, greedy). The Indigenous side stays human.
  Output: data/translated/<name>.en, about 48,000 lines (Slurm job 14131729).
  Again? No. Expensive (a GPU job with a 32B model), and the result is kept.

• clean_external.py: clean the external train sets
  Ran the lab's data-cleaning pipeline (~/data-cleaning-pipeline) on the external train
  sets only; dev and test stay exactly as published.
  Output: data/external_clean/<name>/cleaned/{src,tgt}.txt, deleted lines in deleted/.
  Again? No, unless the cleaning rules change.

• decontaminate.py: keep evaluation sentences out of training
  Drop every training pair whose English or target side, normalized, equals any sentence
  of any evaluation set (FLORES+ dev and devtest, AmericasNLP, MayanV, church dev and
  test). An evaluation sentence seen in training would make the model look better than it is.
  Output: data/train/<code>/<source>.{en,<code>}, the final training files, and
  data/train/removed.json with what was dropped (very little: 0 to 43 pairs per set).
  Again? Yes, after Swahili and Amharic are split, since their test sets are new.

• Still missing for Swahili and Amharic
  FLORES+ devtest and dev files (swh_Latn, amh_Ethi) are not downloaded yet.
'''
