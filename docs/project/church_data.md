# Church data

How the church parallel corpus is laid out and how to read it for fine-tuning and for
training a model from scratch. What each language is used for is in [data.md](data.md).

## Where it is and who can read it

| Copy | Path (under `/home/vacl2/groups/grp_mtlab/nobackup/archive/all-data/`) | Group needed |
|---|---|---|
| 2026 release | `church-data-2026/<Language>/` | access granted to `vacl2` |
| 2024 release | `all_filtered_2024.zip` (3.3 GB, 297 files, `filtered/` only) | `grp_mtlab` |

The 2026 folder has 141 language directories. Access was granted; on 2026-10-13 every
project language was readable (Spanish, Portuguese, French, Yoruba, Zulu, Kekchi, Guarani,
Quichua). Thirteen other folders are still closed (mode `700`, owned by another user); do
not try to work around the permissions. The 2026 release is the one to use.

Directory names are human-readable language names, not ISO codes, and some contain spaces
or apostrophes (`K'iche'`, `Quichua (Ecuador)`). In the 2024 zip, spaces become
underscores (`Quichua_(Ecuador)`). Always quote paths.

## Layout

```
church-data-2026/<Language>/
    filtered/   src.txt  tgt.txt  meta.txt     <- use this one
    pathsay/    src.txt  tgt.txt  meta.txt
```

- Line `i` of `src.txt`, `tgt.txt` and `meta.txt` belong to the same segment. The three
  files always have the same number of lines.
- **`src.txt` is English and `tgt.txt` is the named language.** The corpus is
  English-centric: there is no Spanish source side, even for Latin-American languages.
- Files are UTF-8, one segment per line, `\n` line endings, no tabs, no empty lines.
- The provider says to use `filtered/`. In every language checked, `pathsay/` was a
  strict subset of `filtered/` (about 2% fewer lines). What extra filter `pathsay`
  applies is not documented; ask the provider before using it.

## `meta.txt`

One entry per line, written as a Python literal (single quotes), so parse it with
`ast.literal_eval`, not `json.loads`.

- 2026 release: the project languages, Haitian and Papiamento have metadata as below;
  the small Latin-American languages (Aymara, K'iche', Mam, ...) have `[]` on every line.
- 2024 release: each line is a list of translation-memory records, each a list of
  `[key, value]` pairs, for example `creationdate`, `Client:SingleString`,
  `ID:SingleString`, `Sensitivity Status:SinglePicklist` and `source_file_name`
  (e.g. `14 043 - Kekchi - SoftwareStrings.tmx`). Meta files are 2–4 times larger than
  the text, so stream them line by line.

## What the text looks like

- Scripture verses (often with the verse number kept at the start, `7 And he…` /
  `7. Ri areʼ…`), church manuals and talks, and software UI strings
  (`An internet connection is required.`).
- Verse numbers are not always formatted the same on both sides (`29` vs `29.`).
- A few English lines repeat with different translations (Aymara: 8,400 pairs,
  8,382 unique English lines).
- The small Latin-American languages share almost the same English side: K'iche' and
  Mam share 2,763 of about 2,770 English lines.

## Sizes (lines in `filtered/src.txt`)

Project languages, 2026 release (counted 2026-10-13), with the 2024 zip for comparison:

| Language | 2026 lines | 2024 lines |
|---|---:|---:|
| Spanish | 2,610,396 | 2,088,171 |
| Portuguese | 2,457,565 | 1,965,589 |
| French | 2,245,512 | 1,770,344 |
| Swahili | 437,118 | 349,138 |
| Kekchi | 227,804 | 185,867 |
| Amharic | 187,539 | 129,023 |
| Guarani | 119,981 | 81,267 |
| Zulu | 109,726 | 89,784 |
| Yoruba | 99,769 | 79,098 |
| Quichua (Ecuador) | 18,225 | 18,031 |

Efik swati sesotho

`src.txt`, `tgt.txt` and `meta.txt` have equal line counts in every language above.
Both releases are below the 2024 inventory counts in [data.md](data.md) (for example,
Spanish 3,137,864); the inventory may count before filtering; this is unconfirmed.

## Using it to fine-tune a pretrained model

`filtered/` is already cleaned by the lab pipeline; it is split, not cleaned again.

1. `qlrl/split_church.py` reads `filtered/src.txt` and `tgt.txt` as (English, target)
   pairs and draws 1,000 dev and 1,000 test pairs at random (seed 0) per language. The
   metadata has no usable document unit (`source_file_name` is a coarse bucket such as
   "Scripture"; Quichua has one), so the split is by line.
2. Train drops every pair whose English or target side is a near-duplicate of a dev or
   test sentence: equal after lowercasing and keeping only letters. The English side is
   checked against the dev and test of all languages.
3. `qlrl/decontaminate.py` then drops training pairs that match any evaluation set
   ([data.md](data.md)). Final training files: `data/train/<code>/church.{en,<code>}`.

## Using it to train a model from scratch

- Both sides can serve as monolingual text: `tgt.txt` for the target language and
  `src.txt` for English. The English side repeats heavily across languages, so
  deduplicate it across languages before counting tokens.
- Train the tokenizer on the training split only, never on held-out church data or on
  external dev/test sets.
- Even Spanish is about two million segments, far below a GPT-2 token budget; see the
  open item at the end of [data.md](data.md).

## Open question: the pivot language

The external benchmarks (AmericasNLP, MayanV) are Spanish–X, while church data is
English–X. Options:

| Option | What it means | Trade-off |
|---|---|---|
| English–X | Train and evaluate church pairs with English as source | Simple; does not match the Spanish–X benchmarks |
| Pivot through English | Join Spanish and X lines that share the same English line to build Spanish–X | Matches the benchmarks; loses lines with no exact English match and inherits two alignment errors |
| Both | English–X from church, Spanish–X from external data | Two directions to report and compare |

## Latin-American languages in the 2026 release

| Language | Readable now | `filtered` lines | Notes |
|---|---|---:|---|
| Aymara | Yes | 8,400 | Bolivia, Peru |
| Guarani | Yes | 119,981 | Project language |
| Guna Yala | Yes | 2,770 | Kuna; Panama |
| Haitian | Yes | 324,028 | Creole; Haiti |
| Kaqchikel | Yes | 2,811 | Mayan; Guatemala |
| Kekchi | Yes | 227,804 | Q'eqchi'; project language |
| K'iche' | Yes | 2,770 | Mayan; Guatemala |
| Mam | Yes | 2,769 | Mayan; Guatemala, Mexico |
| Nivacle | Yes | 123 | Mataguayan; Paraguay, Argentina |
| Papiamento | Yes | 87,896 | Creole; Aruba, Curaçao, Bonaire |
| Quechua (Cusco) | Yes | 2,768 | Peru |
| Quechua (South Bolivia) | Yes | 2,765 | Bolivia |
| Quichua (Ecuador) | Yes | 18,225 | Project language |
| Tzotzil | Yes | 2,787 | Mayan; Chiapas, Mexico |
| Yucatec Maya | Yes | 2,784 | Mayan; Yucatán, Mexico |
