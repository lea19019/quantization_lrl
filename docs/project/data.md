# Data

What text is used for what. One rule above all others: **FLORES-200 devtest is evaluation
only.** Nothing trained, fine-tuned or calibrated may contain it.

| Use | Source | Languages | Notes |
|---|---|---|---|
| Evaluation of damage and translation | FLORES-200 devtest, 1012 parallel sentences | en, fr, sw, yo, zu | Codes eng_Latn, fra_Latn, swh_Latn, yor_Latn, zul_Latn |
| Human evaluation | 100 fixed sentences from the same devtest | sw, yo, zu (fr if a rater exists) | Same sentences for every configuration |
| Calibration text, default | English: FLORES dev, or C4 to match the papers | en | 128 windows × 2048 tokens; five draws |
| Calibration text, add-on | FLORES dev in the target language; a mix | sw, yo, zu; mix | Dev, never devtest |
| Translation prompt examples | FLORES dev | all five | The same five examples per language pair |
| Quantizer validation | WikiText-2 test (and train for GPTQ calibration where the papers did) | en | Only to reproduce published perplexities |
| GPT-2 training | **to be decided with the advisor** | all five | See below |
| Fine-tuning intervention | Lab text | sw, yo, zu | Text confirmed to exist; amount and cleaning to be recorded |

## GPT-2 training data (open)

FLORES cannot train a model: about 3000 sentences per language. A 124M model needs on the
order of 5B tokens total, so each language's share must come from a large monolingual
corpus. Candidates with all five languages: MADLAD-400, CC-100, OSCAR, FineWeb-2. The
share vector is set after measuring how many tokens each corpus holds for Swahili, Yoruba
and Zulu; small languages may need two to three epochs to reach their share, which is
recorded because repetition changes where a language sits on its learning curve. The
tokenizer is trained on the same mix. **Flagged for discussion with the advisor.**

## Lab text

Used for the fine-tuning intervention (phase 3) and possibly as part of the GPT-2 mix. Before
use, record per language: token count, source, whether it is parallel with English,
whether it overlaps FLORES (it must not), and how it was cleaned.

## Contamination check

Before any training or fine-tuning run, every FLORES devtest sentence is searched for in
the training text (exact and near-duplicate). A hit removes the sentence from the training
text, and the manifest records the count.
