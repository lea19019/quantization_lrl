# Pretraining data for the from-scratch model

What text the from-scratch decoder (phase 2, about 550M parameters) is trained on, where it
comes from, how it is cleaned and how it is split. Parallel-data rules shared with phase 1
are in [data.md](data.md); the church corpus layout is in [church_data.md](church_data.md).

## 1. Plan in one paragraph

Two stages. **Pretraining** on free text: Wikipedia, the cleaned FineWeb-2 web text of the
chosen Latin-American languages, and the train splits read as monolingual text (every target
side; the church English side once). **Translation stage** on the parallel data in the evaluation prompt
format, `<eng>…</eng> <tgt>…</tgt>`, both directions, loss on the target side only. Dev and
test sentences of every language stay out of both stages.

## 2. Languages and sources

| Source | Languages | Size | Kind |
|---|---|---|---|
| Church corpus 2026 (`filtered/`) | 124 | 58.8M pairs; target sides 7.3 GB; 3.24M unique English lines (306 MB) | parallel, English–X |
| Wikipedia, `wikimedia/wikipedia` 20231101 | 107 wikis (Wayuu and `qu` dropped; `zh` split in two) | 62 GB parquet (English alone about 29G characters); cleaned: 22.2M articles, 43.2G characters | free text |
| FineWeb-2, `HuggingFaceFW/fineweb-2` | 12 Latin-American | 1–40 MB each before cleaning; cleaned: 20.5k documents, 57M characters | free text (web) |
| AmericasNLP 2021 | Nahuatl, Bribri, Guarani, Ayacucho Quechua | 7.5k–125k train pairs | parallel, Spanish–X (English side machine-translated) |
| MayanV; Napo Kichwa lexicon | Q'eqchi'; Kichwa | 2.1k; 6.3k | parallel, Spanish–X |
| FLORES+ (`e707e62e`) | the phase-1 nine plus `quy_Latn`, `ayr_Latn` | 997 dev / 1,012 devtest | evaluation only |

### Church corpus: 124 languages (141 folders minus the 17 with at most 123 pairs)

- **1M pairs or more (24):** Spanish, Portuguese, French, Italian, Japanese, German, Korean,
  Russian, Chinese T, Samoan, Tagalog, Cebuano, Dutch, Tongan, Danish, Swedish, Finnish,
  Norwegian, Ukrainian, Mongolian, Thai, Indonesian, Cambodian, Romanian.
- **100k–1M (54):** Albanian, Polish, Czech, Chinese S, Latvian, Malagasy, Bulgarian,
  Vietnamese, Croatian, Lithuanian, Estonian, Armenian, Hungarian, Fijian, Tahitian, Greek,
  Swahili, Marshallese, Kiribati, Malay, Icelandic, Bislama, Haitian, Slovenian, Slovak,
  Serbian, Portuguese (Euro), Ilokano, Kekchi, Arabic, Turkish, Amharic, Urdu, Lao, Hmong,
  Cantonese, Hindi, Maltese, Hiligaynon, Lingala, Tamil, Georgian, Telugu, Persian, Guarani,
  Fante, Igbo, Kosraean, Chuukese, Zulu, Sinhalese, Xhosa, Pohnpeian, Shona.
- **10k–100k (24):** Yoruba, Twi, Papiamento, Navajo, Hindi (Fiji), Yapese, Palauan,
  Afrikaans, ASL, Tok Pisin, Efik, Tswana, Nepali, Macedonian, Kinyarwanda, Tshiluba, Burmese,
  South Sotho, Maori, Quichua (Ecuador), Catalan, Rarotongan, Hawaiian, Swazi.
- **Under 10k (22):** Chewa, Kannada, Chinese (Hanyu Pinyin), Pangasinan, Aymara, Kazakh,
  Kikamba, Bengali, Bikol, Kaqchikel, Kapampangan, Chamorro, Waray, Tzotzil, Yucatec Maya,
  Gusii, Guna Yala, K'iche', Niuean, Mam, Quechua (Cusco), Quechua (South Bolivia). The
  Latin-American ones here have about 2,770 pairs each with nearly the same English side.
- **Dropped (17):** Armenian (Western), Bambara, Basque, Belarusian, Fon, Kurdish (Central),
  Luxembourgish, Malayalam, Motu, Nivacle, Northern Sotho, Oromo, S'gaw Karen, Siksika,
  South Ndebele, Stoney, Tuvaluan.

Decided: ASL (written gloss of a sign language) is dropped, leaving 123 church languages.
Chinese S, Chinese T and Pinyin are three languages (`cmn_hans`, `cmn_hant`, `cmn_latn`).
Portuguese is Brazilian and Portuguese (Euro) is European (`por`, `por_pt`): one language in
two written standards that differ in spelling, grammar and vocabulary ("estou fazendo" /
"estou a fazer", "fato" / "facto"); only 7% of the European lines appear in the Brazilian
file. They stay separate: the same language with 2.5M and 249k pairs.

### Latin-American languages (option B)

Nahuatl, Mayan and Quechua languages plus the Indigenous languages with the most data. eBible
(255 more languages, mostly one New Testament each, CC BY-NC-ND) is not used: too little
text per language, near-identical varieties, and Bible-only domain.

| Language | Parallel | Free text (FineWeb-2 code, before → after cleaning; Wikipedia, cleaned) |
|---|---|---|
| Nahuatl | AmericasNLP 12.6k (cleaned) | `nah_Latn` 5.2 MB → 3.3M chars, mostly UNAM Temoa and Wikipedia; `nah` wiki 1.2M chars |
| Q'eqchi' | church 228k, MayanV | `kek_Latn` 5.5 MB → 1.1M chars |
| K'iche' | church 2.8k | `quc_Latn` 12 MB → 8.5M chars, half Guatemalan government sites |
| Yucatec Maya | church 2.8k | `yua_Latn` 18 MB → 6.8M chars |
| Kaqchikel | church 2.8k | none |
| Mam, Tzotzil | church 2.8k each | `mam_Latn` 12 MB → 0.6M chars, `tzo_Latn` 15 MB → 1.2M chars (mostly religious sites, blocked) |
| Ayacucho Quechua | AmericasNLP 125k (all JW300; 108.6k after cleaning) | `quy_Latn` 8.6 MB → 3.9M chars |
| Cusco / Bolivian Quechua | church 2.8k each | `quz_Latn` 6.4 MB → 2.3M chars / `quh_Latn` 1.2 MB → 0.7M chars |
| Ecuadorian Quichua | church 18k, Kichwa lexicon | `qug_Latn` 16 MB → 19k chars (90% a Bible site, blocked) |
| Guarani | church 120k, AmericasNLP 11.8k | `gug_Latn` 39 MB → 21.2M chars; `gn` wiki 5.1M chars |
| Aymara | church 8.4k | `ayr_Latn` 11 MB → 6.9M chars, 40% Global Voices; `ay` wiki 1.9M chars |
| Bribri | AmericasNLP 7.5k | none |

Dropped: Tzeltal (web text only, no parallel data to evaluate on) and Wayuu.

JW300 was withdrawn from OPUS, reportedly over copyright: usable for internal research,
a problem only if the model is published. The religious share was measured by matching word
6-grams against eBible and church verses and by site.

## 3. Splits of the parallel data

Done by `python -m qlrl.data.prepare configs/data/pretrain.yaml` (`qlrl/data/prepare.py`,
`split.py`), the same code as phase 1's `data.yaml` with all 126 languages (123 church plus
Nahuatl, Bribri, Ayacucho Quechua) and its own folders: output `data/processed_pretrain`,
cleaning `data/cleaning_pretrain`, translations `data/translated_pretrain` (links to
`data/translated` and `data/translated_quechua`). The original files are only read, and
phase 1's `data/processed` is untouched.

- **Church: shared held-out verses** (`split: shared`, `split.hold_out_shared`), as FLORES
  does. Per language, dev and test get up to 1,000 pairs each, and at most 10% of its pairs
  each, in two parts:
  - **Core** (`held_out_core: 250`, written to `data/processed_pretrain/core.jsonl`): the
    250 dev and 250 test English lines that the most church languages have (each is in 102 or
    more of the 123), ties broken by a seeded SHA-256 hash. Every language holds out the core
    lines it has: the same sentences everywhere, so **compare languages on the core**.
  - **Rest:** up to the cap by the same hash, a sample of the language's own text, never a
    line of a language with fewer than 5,000 pairs (`protect_below`). A line held out
    anywhere leaves every train set, so mid-size languages' held-out sets would take 13% of
    the smallest languages' train. The smallest languages hold out core lines only.
  - The hash's last bit picks dev or test, so a verse is in the same split in every language.
  - Only pairs with a usable reference are held out: not a target without letters (a bare
    verse number, 12% of Mam's lines) and not a length ratio more than 2x off the language's
    median (misaligned verses, up to 20% in the small Latin-American corpora). They stay in
    train.
  - A language with 2,770 pairs holds out about 210–250 dev and as many test (all core) and
    loses 25–70 more lines to other languages' held-out sets: about 2,270 train.
- Same files and config give the same split every time. Phase 1's `data.yaml` keeps its
  per-language random split. Per-language random held-out sets with 126 languages together
  held out most verses: decontamination removed 18% of all pairs and every train pair of 14
  small languages.
- The church dev and test sets of the phase-1 languages differ from phase 1's; FLORES+ is the
  same in both phases. Full church test sets differ in content by language (Mam's are Book of
  Mormon verses, 30 English words on average; Spanish's 18); the core does not.
- **External corpora:** published train / dev / test as they are; FLORES+ dev and devtest
  are dev and test only (also for Aymara and Ayacucho Quechua). External train sets go through the lab cleaning tool.
- **Decontamination:** a train pair is dropped if either side, or a sentence inside either
  side (`decontaminate_sentences`), normalized (lowercase, letters and single spaces), equals
  any dev or test sentence of **any** language: the same verse appears in many languages,
  and short verses recur inside longer ones. Near-paraphrases are not caught.
- Adding a language can add held-out verses (its own rest), so other languages' train sets
  can shrink slightly; the core changes only if coverage ranks shift.

The Spanish side of AmericasNLP Ayacucho Quechua was translated to English with the phase-1
setup (Qwen3-32B, `translate.py`; Slurm job 14704514, config
`~/nobackup/archive/corpus_survey/quechua/quechua.yaml`) into `data/translated_quechua`.

## 4. Cleaning the free text

`python -m qlrl.data.freetext configs/data/freetext.yaml <index> [--test]`
(`qlrl/data/freetext.py`, `text_filters.py`, tests in `tests/test_text_filters.py`). One
corpus per Slurm array task (119 corpora: 107 Wikipedia tasks, 12 FineWeb-2). Per corpus:

1. **Sample:** at most `max_docs` = 1M documents, chosen by a seeded hash. Only the largest
   Wikipedias exceed it (English has 6.4M articles); 1M English articles are about 4.5G characters.
2. **Templates:** bot-written articles (6.1M Cebuano stubs such as "Kaliwatan sa langaw ang
   … Una ning gihulagway ni … ni adtong …") share one text with names and numbers swapped.
   Every word outside the 100 words found in the most documents becomes "_" (runs of them
   one "_"); a skeleton seen 100 times or more is a template and only its first 10 documents
   stay. By document count, not word count: a genus name repeated in its 300 species stubs
   would otherwise be "common" and split the template by genus. On a 50k-document sample this
   flags 95% of Cebuano, 52% of Swahili, 30% of Quechua and 0.5–3% of English, Spanish and
   German (disambiguation pages and one-line stubs); real articles are too long to collide.
   In scripts written without spaces (Chinese, Japanese, Thai, Lao, Khmer, Burmese) each
   character counts as a word.
3. **Blocked sites (FineWeb-2):** religious sites (bible.is, jw.org, ebible.org, …, or
   "bible", "biblia", "scripture", "gospel", "evangel", "iglesia" in the url),
   machine-translated sites (hielscher.com, mokosmart.com, …) and Wikipedia pages (already in
   the Wikipedia corpora; 26% of Nahuatl and 28% of Quechua FineWeb-2 text). Religious text is
   already supplied by the church corpus, and web Bible verses could be church test
   sentences. This removes nearly all FineWeb-2 text of Ecuadorian Quichua, Mam, Tzotzil and
   most of Q'eqchi': those languages are almost only church text.
   **Varieties are not mixed.** The `qu` wiki (mixed Southern Quechua) is not used. The
   `zh` wiki mixes Simplified and Traditional articles, separate languages here: two tasks
   (`zh_hans`, `zh_hant`) keep the articles whose script is clear, by the characters found
   only in the church Chinese S or only in Chinese T (about 1,000 each), outnumbering the
   other's 3 to 1. About 40% of articles mix both scripts and are dropped (35% go to
   Simplified, 25% to Traditional). The `pt` wiki (mostly Brazilian, some European) counts as
   Brazilian Portuguese (`por`).
4. **Non-prose lines (Wikipedia):** headings, list items, tables and link lists are dropped:
   a line stays if it ends a sentence (after closing quotes or brackets) or has 80+
   characters. This also drops short content list items.
5. **Language ID:** GlotLID (`cis-lmu/glotlid`, pinned) labels each line of 30+ characters
   from its first 200. A line is dropped only if GlotLID is at least 0.9 sure it is a language
   outside the corpus's own labels (in the config: its top GlotLID label plus dialects,
   macro-language members and confusions seen on a sample; never Spanish, English or a variety
   that is a separate language here: other Quechua varieties, Kaqchikel in K'iche', Tzeltal
   in Tzotzil, Nynorsk in Norwegian, Mandarin in Cantonese, Cantonese in Chinese). Unsure and short lines stay, since GlotLID is weak on small languages.
6. **Decontamination:** a line is dropped if it, or a sentence in it, equals a dev or test
   sentence of any language: every dev and test file of `data/processed_pretrain`, and
   FLORES+ dev and devtest of the 11 languages downloaded. Phase 1's church held-out sets are not used: phase 2 is not scored on them.
7. **No deduplication** beyond templates (FineWeb-2 is already deduplicated).
8. **Validation:** a seeded 1% of the documents left, lowered to about 500 for big corpora
   (spread over the whole corpus), goes to
   `valid.parquet`; the rest to `train.parquet`. Translation-stage validation uses the dev
   splits of section 3.

Output: `/home/vacl2/nobackup/archive/freetext/<wikipedia|fineweb>/<code>/{train,valid}.parquet`
(columns `id`, `text`) and `manifest.json` with every count removed per step. Raw downloads
are in `/home/vacl2/nobackup/archive/{wikipedia,fineweb2}`.

**Free text from the parallel data:** `python -m qlrl.data.monolingual
configs/data/freetext.yaml` (`qlrl/data/monolingual.py`) writes the target side of every
train pair to `<output>/parallel/<lang>/train.parquet`, and the church English side, each
line once, to `<output>/parallel/eng/train.parquet`. The machine-translated English of the
Spanish-X corpora is left out. Train splits only, so already decontaminated; one sentence
per row, with no document context. Its validation text is the dev splits.

## 5. Caveats for the share analysis

- The `pt` wiki is counted as Brazilian but has some European Portuguese articles.
- Nahuatl and Bribri English sides are Qwen3-32B translations, so their scores are bounded by
  that translation's quality.
- JW300 (all Ayacucho Quechua train pairs) is Jehovah's Witnesses text, though jw.org is
  blocked on the web.

## 6. Still to decide

- Shares per language (how much Wikipedia per language, upsampling of small languages) and
  the total token budget.
- Test set of each new language; the Bible overlap between church and other sources.
