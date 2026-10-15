# Data

What text is used for training and evaluation. Evaluation splits never enter training,
fine-tuning, calibration or prompt examples.

## Latin-American languages

| Language | Family | Geographic origin | Role and reason for inclusion |
|---|---|---|---|
| Spanish | Indo-European, Romance | Iberian Peninsula; now dominant across most of Latin America | High-resource pivot for every Indigenous pair, the main human-evaluation language, and the largest church corpus. |
| Portuguese | Indo-European, Romance | Portugal; the Latin-American variety is Brazilian Portuguese | Second high-resource control: it tests whether a result found with Spanish repeats in a closely related language. Confirm the church corpus variety before calling it Brazilian Portuguese. |
| Q'eqchi' (Kekchi, `kek`) | Mayan, K'ichean | Guatemala and Belize | Large church corpus, an independent informal-domain corpus, and a family unlike the other selected languages. |
| Guarani (`gn`) | Tupian, Tupi-Guarani | Paraguay and neighboring Bolivia, Brazil and Argentina | Different family and morphology, church plus benchmark data, and strong regional importance. |
| Nahuatl (`nah`) | Uto-Aztecan, Nahuan | Central Mexico | No church data, but an established Spanish–Nahuatl benchmark; adds a third unrelated Indigenous family. The corpus mixes historical and modern varieties, so variety is tracked per row where possible. |
| Ecuadorian Quichua/Kichwa | Quechuan, Northern Quechua | Ecuadorian Andes and Amazon | Gives South American Indigenous coverage and church data. The external source is specifically Napo Lowland Kichwa (`qvo`) and is merged only if the church variety is compatible. |
| Bribri (`bzd`) | Chibchan, Talamancan | Talamanca region of Costa Rica and Panama | Extremely low-resource language from another family, with a published benchmark and a possible path to human evaluation. |

## Additional Latin-American church languages (exploratory)

Church data only, English–X, no external benchmark. Line counts are `filtered/` lines; see
[church_data.md](church_data.md).

| Language | Family | Geographic origin | Church lines |
|---|---|---|---:|
| Aymara (`aym`) | Aymaran | Andes of Bolivia, Peru and Chile | 8,400 |
| Guna (`cuk`) | Chibchan | Guna Yala, Panama | 2,770 |
| Kaqchikel (`cak`) | Mayan, K'ichean | Central highlands of Guatemala | 2,811 |
| K'iche' (`quc`) | Mayan, K'ichean | Western highlands of Guatemala | 2,770 |
| Mam (`mam`) | Mayan, Mamean | Western Guatemala and Chiapas, Mexico | 2,769 |
| Nivaclé (`cag`) | Matacoan | Gran Chaco of Paraguay and Argentina | 123 |
| Cusco Quechua (`quz`) | Quechuan, Southern Quechua | Cusco region, Peru | 2,768 |
| South Bolivian Quechua (`quh`) | Quechuan, Southern Quechua | Southern Bolivia | 2,765 |
| Tzotzil (`tzo`) | Mayan, Tzeltalan | Chiapas, Mexico | 2,787 |
| Yucatec Maya (`yua`) | Mayan, Yucatecan | Yucatán Peninsula, Mexico and Belize | 2,784 |
| Haitian Creole (`hat`) | French-based creole | Haiti | 324,028 |
| Papiamento (`pap`) | Iberian-based creole | Aruba, Curaçao and Bonaire | 87,896 |

## African core languages

| Language | Family | Geographic origin | Church segments | Role and reason for inclusion |
|---|---|---|---:|---|
| English (`eng_Latn`) | Indo-European, Germanic | England | Not reported separately | The anchor: one side of every training and evaluation pair. |
| French (`fra_Latn`) | Indo-European, Romance | Northern France | 2,615,266 | High-resource African lingua-franca control with substantial church data. |
| Yoruba (`yor_Latn`) | Niger-Congo, Volta-Niger, Yoruboid | Southwestern Nigeria, Benin and Togo | 84,373 | Low-resource tonal language with FLORES, human-evaluation and AfriCOMET coverage. |
| Zulu (`zul_Latn`) | Niger-Congo, Bantu, Nguni | South Africa | 95,741 | Low-resource language with the same evaluation coverage; its Bantu structure gives stronger typological contrast with Yoruba. |
| Swahili (`swh_Latn`) | Niger-Congo, Bantu | East African coast; lingua franca of East Africa | Not in the 2024 inventory; 437,118 lines in 2026 | Bantu like Zulu with far more data: the resource contrast for H1. |
| Amharic (`amh_Ethi`) | Afro-Asiatic, Semitic | Ethiopia | Not in the 2024 inventory; 187,539 lines in 2026 | Different family and Ge'ez script: tests whether results hold beyond Niger-Congo and Latin script, and separates tokenization effects. |

## Parallel-corpus inventory

Church counts are from the 2024 inventory reported by the data provider; the 2026 files
are now readable and their measured line counts are in [church_data.md](church_data.md).
External counts below were measured from the downloaded files.

Church data location on the cluster:
`/home/vacl2/groups/grp_mtlab/nobackup/archive/all-data/church-data-2026`.
Layout, access and how to read it: [church_data.md](church_data.md).

| Language | Church pairs | External train/pool | External held out | Total identified |
|---|---:|---:|---:|---:|
| English | Not reported separately | 0 | 1,012 FLORES | At least 1,012 |
| French | 2,615,266 | 0 | 1,012 FLORES | 2,616,278 |
| Yoruba | 84,373 | 0 | 1,012 FLORES | 85,385 |
| Zulu | 95,741 | 0 | 1,012 FLORES | 96,753 |
| Swahili | Not inventoried | 0 | 1,012 FLORES | Not inventoried |
| Amharic | Not inventoried | 0 | 1,012 FLORES | Not inventoried |
| Spanish | 3,137,864 | 0 | 0 | 3,137,864 |
| Portuguese | 2,876,104 | 0 | 0 | 2,876,104 |
| Q'eqchi' | 202,255 | 2,114 | 2,000 | 206,369 |
| Guarani | 86,331 | 26,032 | 1,998 | 114,361 |
| Nahuatl | 0 | 16,145 | 1,675 | 17,820 |
| Ecuadorian Quichua/Kichwa | 18,074 | 6,313 unsplit lexicon entries | 0 | 24,387 |
| Bribri | 0 | 7,508 | 1,999 | 9,507 |

The totals include evaluation data, which cannot be trained on. They are counts of
parallel pairs or lexicon entries, not tokens. Spanish, Portuguese and church-backed
languages still need a held-out church split, so their eventual training counts will be
lower. The Kichwa pool contains 6,287 accepted and 26 review entries.

| Language | External source and use |
|---|---|
| English | [FLORES-200](https://huggingface.co/datasets/facebook/flores) devtest: 1,012 evaluation-only sentences; the church source-side count was not reported separately. |
| French | FLORES-200 devtest for evaluation; reserve church data for training after deduplication and a church holdout. |
| Yoruba | FLORES-200 devtest for evaluation; use church data for training after deduplication and a church holdout. |
| Zulu | FLORES-200 devtest for evaluation; use church data for training after deduplication and a church holdout. |
| Swahili | FLORES-200 devtest for evaluation; use church data for training after deduplication and a church holdout. |
| Amharic | FLORES-200 devtest for evaluation; use church data for training after deduplication and a church holdout. |
| Spanish | No external data needed now; reserve a fixed church test split. |
| Portuguese | No external data needed now; identify its regional variety, then reserve a fixed church test split. |
| Q'eqchi' | [MayanV](https://github.com/transducens/mayanv/tree/07d1426cbf783c81e4b8fe623051836b6b3fd0f4/MayanV/kek): add only train to church training and preserve dev/test. Its README reports 4,133 pairs, while the pinned files contain 4,114. |
| Guarani | [AmericasNLP 2021](https://github.com/AmericasNLP/americasnlp2021/tree/d3f519c6b38299d8149311e65a369047350e6849/data/guarani-spanish): add only train to church training and preserve dev/test. |
| Nahuatl | [AmericasNLP 2021](https://github.com/AmericasNLP/americasnlp2021/tree/d3f519c6b38299d8149311e65a369047350e6849/data/nahuatl-spanish): preserve dev/test. This already derives from Axolotl, so Axolotl is not added again. |
| Ecuadorian Quichua/Kichwa | [Napo Kichwa–Spanish](https://huggingface.co/datasets/aimeri/kichwa-spanish/tree/b05c9a5f1f9f1606ac286d1465125b4225501783): use the permissive `lexicon` only after variety and quality audits. Exclude the duplicate instructions and GPL tier. |
| Bribri | [AmericasNLP 2021](https://github.com/AmericasNLP/americasnlp2021/tree/d3f519c6b38299d8149311e65a369047350e6849/data/bribri-spanish): preserve dev/test and convert its intermediate orthography before human evaluation. |

Downloaded snapshots:

- AmericasNLP commit `d3f519c6b38299d8149311e65a369047350e6849`:
  [`data/external/americasnlp2021`](../../data/external/americasnlp2021)
- MayanV commit `07d1426cbf783c81e4b8fe623051836b6b3fd0f4`:
  [`data/external/mayanv`](../../data/external/mayanv)
- Napo Kichwa–Spanish commit `b05c9a5f1f9f1606ac286d1465125b4225501783`:
  [`data/external/kichwa-spanish`](../../data/external/kichwa-spanish)

## Spanish sides to translate into English

Every Spanish side below gets a machine-translated English version (decision of
2026-10-13); the Indigenous side stays human-written. Lines counted on 2026-10-13.

| Language | Source | Split | File (Spanish side) | Lines |
|---|---|---|---|---:|
| Nahuatl | AmericasNLP | train | `americasnlp2021/data/nahuatl-spanish/train.es` | 16,145 |
| Nahuatl | AmericasNLP | dev | `americasnlp2021/data/nahuatl-spanish/dev.es` | 672 |
| Nahuatl | AmericasNLP | test | `americasnlp2021/data/nahuatl-spanish/test.es` | 1,003 |
| Bribri | AmericasNLP | train | `americasnlp2021/data/bribri-spanish/train.es` | 7,508 |
| Bribri | AmericasNLP | dev | `americasnlp2021/data/bribri-spanish/dev.es` | 996 |
| Bribri | AmericasNLP | test | `americasnlp2021/data/bribri-spanish/test.es` | 1,003 |
| Q'eqchi' | MayanV | train | `mayanv/MayanV/kek/train/data.es` | 2,114 |
| Q'eqchi' | MayanV | dev | `mayanv/MayanV/kek/dev/data.es` | 1,000 |
| Q'eqchi' | MayanV | test | `mayanv/MayanV/kek/test/data.es` | 1,000 |
| Guarani | AmericasNLP | train | `americasnlp2021/data/guarani-spanish/train.es` | 26,032 |
| Guarani | AmericasNLP | dev | `americasnlp2021/data/guarani-spanish/dev.es` | 995 |
| Guarani | AmericasNLP | test | `americasnlp2021/data/guarani-spanish/test.es` | 1,003 |
| Kichwa | Napo Kichwa–Spanish | lexicon | `kichwa-spanish/lexicon/train-*.parquet`, column `spanish` | 6,313 |

Paths are under `data/external/`. About 65,800 lines in total. Nahuatl, Bribri and
Guarani test targets are in `americasnlp2021/test_data/` (`test.nah`, `test.bzd`,
`test.gn`). The Kichwa lexicon is 4,135
headwords (single words or short phrases) and 2,178 example sentences; 26 rows are
flagged `review`.

## Translation metrics

All seven languages use the same reference-based metrics so their results are comparable:

1. **Primary: corpus chrF++**, using `sacrebleu==2.6.0`, case-sensitive, character
   order 6, word order 2, beta 2 and no whitespace characters. This is robust to
   inflection and spelling variation and is the official AmericasNLP metric family.
2. **Secondary: corpus BLEU**, using the same SacreBLEU version with mixed case,
   `13a` tokenization and exponential smoothing. It is reported for continuity, not
   used to rank languages, because word segmentation and morphology affect it strongly.
3. **Human evaluation**, on a fixed shared subset wherever a qualified rater is
   available. Lack of a rater does not remove a language from automatic evaluation.

No COMET score is reported for this seven-language extension: there is no single
author-released COMET checkpoint with defensible coverage of all seven languages.

The African core uses the same chrF++ and BLEU settings plus the author-released
[`masakhane/africomet-stl-1.1`](https://huggingface.co/masakhane/africomet-stl-1.1/tree/b4065f4424a962cf6e52373a9ec8a22a624fb278)
checkpoint at revision `b4065f4424a962cf6e52373a9ec8a22a624fb278`. Its core is English,
French, Yoruba and Zulu. FLORES-200 devtest is retained for that core because it supplies
the same parallel test sentences for every core language; it is not required for the
Latin-American extension.

## Split and contamination rules

- External `dev` and `test` splits remain evaluation-only; only `train` is combined
  with church training data.
- Deduplicate church and external training data before use and record how many pairs
  were removed.
- Search every evaluation sentence for exact and near duplicates in training data;
  remove any hit from training and record the count.
- FLORES-200 devtest is evaluation-only. Nothing trained, fine-tuned, calibrated or
  used as a prompt example may contain it.

## GPT-2 training data (open)

The parallel corpora above are too small to supply the full GPT-2 token budget. The
monolingual sources and per-language shares remain to be settled before training.
