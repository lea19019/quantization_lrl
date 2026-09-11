# Literature: what is known and what nobody has measured

Search date 2026-09-07. Audited against the PDFs on 2026-09-09. The explanations
the papers support or contradict are in [explanations.md](explanations.md). Per-paper entries are in
[../papers/index.md](../papers/index.md). Page numbers are PDF pages.

## 1. What is known

1. **The gap is replicated.** PTQ hurts some languages much more than others, under four
   methods and four papers (Marchisio 2024; Marie & Fujita 2025; Chimoto 2026; Soualhi 2026).
2. **Damage correlates with a data-size proxy, weakly.** Marchisio p.7: R² 0.24 at W8 to 0.63
   at W4, against mC4 gigabytes over nine languages, and "a definitive relationship ... cannot
   be determined" because mixtures are unreleased. Zeng 2024 p.1 is the better evidence:
   "the larger the proportion of a language in the model training dataset, the more
   resistant it is to compression" (BLOOM, 20 languages). Counterweight: Ahia 2021 p.6, once
   data size is held fixed, per-language damage rates converge.
3. **Within one language, where damage lands is disputed.** Proskurina 2024 p.4 and Lotfi
   2026 p.1: low-confidence and high-entropy inputs move most. Chang 2025 p.9: "long-tail
   examples are relatively less affected"; the inputs that break are low-NLL, head inputs.
   None of these vary language.
4. **The long-tail story has a theory.** Tran 2022: groups with fewer samples have larger
   gradient norms and sit closer to the decision boundary, so pruning hurts them more.
   Vision only; never tested with languages as groups.
5. **At model level, more training means more PTQ damage** (Kumar 2025; Ouyang 2025), and
   Catalan-Tatjer 2026 p.4 says that is "mostly confounded by the learning rate". Nobody
   has asked the per-language version.
6. **Fragmentation is well measured without quantization** (Rust 2021; Ahia 2023; Petrov
   2023; Arnett 2025). Soualhi 2026 p.2 discusses fertility under quantization without
   plotting it. No paper measures fragmentation against quantization damage.
7. **Compression does not always hurt the tail more.** Ogueji 2022: moderate sparsity helps
   low-data languages. Wu 2026 p.8: head knowledge is "proportionally the least retained".
   Diddee 2022 p.1: quantization is "more consistent ... especially the lowest-resource
   languages". Only aggressive compression is uneven.
8. **The clean "which cause" experiment does not exist under quantization.** Rust 2021 holds
   data fixed and varies the tokenizer, but at full precision. No study reports
   per-language confidence margins before and after quantization.
9. **Statistics are thin but not absent.** Soualhi reports SEs and a paired t-test; Borgersen
   reports Bonferroni-corrected p-values; Marchisio averages five sampling runs. No
   multilingual paper reports seed or calibration-draw spread, and Williams 2024 p.7 shows
   that spread alone is ±4.7 points on BoolQ.
10. **Most of the long-tail evidence base is pruning or distillation, not PTQ** (Hooker, Ahia 2021,
    Ogueji, Tran, Tropeano, Jin, Gurgurov), and Hooker 2019 p.6 and Jaiswal 2024 p.1 both
    say quantization is the milder method.

## 2. What nobody has measured

1. **Per-token margin by language, before and after quantization.**
2. **Fragmentation vs quantization damage on the same model**, with a word-level metric.
3. **Separating data size, script and fertility.** Same language under two tokenizers;
   language pairs matched on data size; a from-scratch model with controlled proportions.
4. **Per-language version of Kumar/Ouyang.**
5. **Tran 2022's theory with languages as groups**: per-language gradient norm at the
   released checkpoint.
6. **Chang 2025's residual-norm predictor across languages.**
7. **Long tail within a low-resource language** under PTQ.
8. **Seed and calibration-draw spread** on any multilingual result.
9. **Calibration language within one method, one bit width, one group size**, with RTN as
   the floor, across 8/4/3/2 bits. Chimoto is closest and omits SmoothQuant and the sweep.
10. **Embeddings and lm_head in or out of scope**, per language (Ogueji's decisive ablation,
    never run for quantization).
