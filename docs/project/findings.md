# Front 1 findings

What the experiments in [experiments.md](experiments.md) have shown so far. Read this before
continuing the work.

**Scope.** Llama-LoRA, 1,000 sentences per language (Nahuatl 995, Bribri 1,003). All 12 languages
on GPTQ seed 0, 4 and 3 bits; Nahuatl and Bribri use the AmericasNLP test set, the others church.
GPTQ seeds 1 and 2, RTN and AWQ seed 0 on Spanish, Swahili and Amharic; swap on GPTQ seed 1, RTN
and AWQ at 3 bits on Spanish, Q'eqchi', Swahili, Yoruba and Amharic. Everything else is GPTQ seed 0.

**Numbers.** Everything except the weight reading comes from `results/experiments/analysis/*.csv`:
`python -m qlrl.experiments.analyze configs/experiments/analyze.yaml`. It reads whatever has
finished. Weight reading: `wp/quantization/findings/queries.md`. Codes: Spa Spanish, Por Portuguese,
Fra French, Kek Q'eqchi', Grn Guarani, Swh Swahili, Yor Yoruba, Zul Zulu, Qug Quechua, Nah Nahuatl,
Bzd Bribri, Amh Amharic.

## Damage

Damage is log p_full − log p_quant of the reference, in nats. GPTQ seed 0:

| | Spa | Por | Fra | Kek | Grn | Swh | Yor | Zul | Qug | Nah | Bzd | Amh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 bits, per byte | .0011 | .0002 | .0024 | .0048 | .0083 | .0059 | .0098 | .0136 | .0127 | .0179 | .0188 | .0158 |
| 3 bits, per byte | .025 | .033 | .037 | .046 | .067 | .072 | .085 | .095 | .086 | .105 | .112 | .182 |
| 3 bits, per word | .150 | .200 | .225 | .286 | .555 | .475 | .605 | .831 | .786 | .849 | .631 | 2.478 |

**Across seeds and methods**, per byte (GPTQ seeds 0, 1, 2; RTN; AWQ seed 0):

| | Spanish | Swahili | Amharic |
|---|---:|---:|---:|
| 3 bits, GPTQ | .025, .022, .024 | .072, .076, .086 | .182, .229, .232 |
| 3 bits, RTN, AWQ | .056, .021 | .230, .060 | .555, .113 |
| 4 bits, GPTQ | .0011, .0049, .0027 | .0059, .0114, .0099 | .0158, .0179, .0154 |
| 4 bits, RTN, AWQ | .0060, .0030 | .0132, .0115 | .0226, .0109 |

At 3 bits the order holds in every model; seed 0 is low for Amharic (intervals do not overlap
seeds 1 and 2). At 4 bits seed 0 is low for Spanish and Swahili, so its ratios overstate the gap
(Amharic/Spanish 14, 3.7, 5.7); Portuguese at 4 bits is not above zero, so its 4-bit shares (−30 to
3.9) mean nothing. Use 3 bits for the map and do not compare 4-bit seed 0 languages by ratio.

## Measured, 3 bits unless stated

**Shorter residual, disturbance about equal.** Residual stream at layer 28, per token:

| | Spa | Por | Fra | Kek | Grn | Swh | Yor | Zul | Qug | Nah | Bzd | Amh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ‖full‖ | 52.1 | 51.9 | 49.6 | 44.7 | 45.3 | 44.1 | 40.4 | 41.9 | 43.1 | 42.2 | 39.4 | 35.4 |
| ‖quant − full‖ | 18.7 | 19.4 | 18.9 | 20.3 | 19.8 | 18.7 | 20.6 | 20.0 | 20.8 | 19.4 | 17.8 | 21.6 |

‖full‖ is about equal at layer 20 (19–20); the low-resource languages fall behind after it. At 4
bits ‖quant − full‖ is 7.9–9.3 with the same pattern. Across GPTQ seeds, RTN and AWQ, Spanish and
Swahili stay within about 1 of each other and Amharic is highest (by only .5 in AWQ). At layer 31
the ratio ‖quant − full‖/‖full‖ is .39–.43 in all languages and no longer follows damage.

**Same late disturbance, different damage.** With only layers 20–31 quantized, ‖hybrid − full‖ at
layer 28 is 13.0–16.4 in every language and model, with no order, but late-only damage per byte
differs about threefold within church. Q'eqchi' and Swahili match on ‖full‖ (44.7, 44.1) and late
disturbance (15.6, 15.3) and lose .022 and .039; Swahili loses 1.8–2.4 times more in all four
models. Growth after layer 28 does not explain it: Nahuatl and Bribri reach the largest ‖full‖ at
layer 31 (112, 115) and have the largest late-only damage.

**Where the damage comes from.** Swap, GPTQ seed 0. Damage per byte with only that range
quantized, and the share that appears only when all are quantized together:

| | Spa | Por | Fra | Kek | Grn | Swh | Yor | Zul | Qug | Nah | Bzd | Amh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Late (20–31) | .015 | .018 | .019 | .022 | .034 | .039 | .036 | .048 | .045 | .065 | .063 | .053 |
| Middle (6–19) | .004 | .005 | .005 | .007 | .010 | .007 | .009 | .016 | .014 | .012 | .014 | .020 |
| Early (0–5) | .004 | .006 | .009 | .009 | .014 | .013 | .019 | .018 | .016 | .023 | .028 | .046 |
| Share only when combined | .08 | .11 | .10 | .17 | .13 | .18 | .24 | .15 | .13 | .05 | .06 | .34 |

**Swap across models.** Spanish, Q'eqchi', Swahili, Yoruba, Amharic:

| | GPTQ seed 0 | GPTQ seed 1 | RTN | AWQ |
|---|---|---|---|---|
| Late-only | .015 .022 .039 .036 .053 | .013 .019 .045 .033 .061 | .008 .015 .033 .027 .047 | .010 .017 .031 .022 .043 |
| Early-only | .004 .009 .013 .019 .046 | .004 .007 .010 .013 .055 | .021 .066 .084 .111 .257 | .003 .006 .007 .009 .016 |
| Share only when combined | .08 .17 .18 .24 .34 | .07 .19 .18 .26 .40 | .38 .40 .45 .45 .41 | .10 .15 .20 .23 .29 |

The late-only order is the same in every model, although RTN has no calibration and, by the weight
reading, no structure in its error. RTN's excess is early and combined; its late-only damage is
below GPTQ's, which may reflect GPTQ's artificial hybrids (each layer was quantized on inputs from
quantized earlier layers). Combined damage exists in every language without calibration; only the
calibrated methods show a gradient across languages in its share.

**Clean late layers double any disturbance, in every language.** With only layers 0–5 quantized,
‖hybrid − full‖ grows 1.95–2.16 times from layer 20 to 28 in all 12. Amharic carries the most early
disturbance at layer 20 (6.0), then Nahuatl 5.1, Yoruba 4.8, Bribri 4.5, the rest 3.5–4.0.

**Patching.** Share recovered:

| | Spa | Por | Fra | Kek | Grn | Swh | Yor | Zul | Qug | Nah | Bzd | Amh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Best MLP, layers 0–5 | .12 | .13 | .16 | .16 | .18 | .18 | .25 | .16 | .16 | .18 | .15 | .34 |
| Best MLP, layers 8–14 | .05 | .04 | .07 | .03 | .04 | .03 | .05 | .02 | .03 | .04 | .04 | .06 |
| `layer29.mlp` | .26 | .32 | .24 | .32 | .29 | .34 | .41 | .31 | .30 | .19 | .22 | .48 |
| `layer31.mlp` | .03 | −.01 | .16 | −.03 | .22 | .22 | .20 | .39 | .22 | .43 | .46 | .60 |
| `layer0.attention` | .09 | .11 | .14 | .12 | .12 | .14 | .19 | .13 | .10 | .09 | .12 | .32 |

Two regions, early and late, with almost nothing between. Shares overlap and do not add to 1; late
sites may score high partly for being near the output. Across GPTQ seeds, RTN and AWQ (Spanish,
Swahili, Amharic): `layer29.mlp` .23–.28, .33–.38, .42–.49; `layer31.mlp` .00–.20, .22–.34, .57–.68.
At 4 bits `layer31.mlp` makes things worse in some languages, but it depends on the seed (Spanish
−3.23, −.02, −1.75; Swahili −.68, .03, −.19); Amharic is stable (.41, .45, .42).

**AWQ removes Amharic's early signal.** `layer0.attention` recovers .32–.40 in Amharic under GPTQ and
RTN, .06 under AWQ; `layer0.mlp` .31–.39 against .11. Early-only damage is .046 (GPTQ) and .016 (AWQ).

**Early signal and non-ASCII bytes.** Amharic (92% non-ASCII bytes) and Yoruba (53%) have the highest
early recovery. Nahuatl (.3%) has high early disturbance, though on a different test set.

**Logit lens.** The prediction becomes readable late: mean log p of the reference token goes from
about −9.5 at layer 24 to −.65 at layer 31 (Spanish). The late full − quant gap peaks at layers
28–30 (Spanish .059, Swahili .276, Amharic .640 nats per byte) and narrows by layer 31. Before
about layer 24 no prediction has formed and the lens is hard to read.

**Noise vs flatness.**

| | Spa | Por | Fra | Kek | Grn | Swh | Yor | Zul | Qug | Nah | Bzd | Amh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Share ½σ²(1 − Σp²) predicts | .72 | .67 | .70 | .60 | .58 | .47 | .44 | .46 | .58 | .90 | .82 | .30 |
| Share on confident tokens (flatness < .1) | .18 | .20 | .17 | .17 | .10 | .23 | .24 | .18 | .11 | .00 | .01 | .45 |
| Damage per token at Spanish's flatness mix | .092 | .116 | .114 | .105 | .119 | .176 | .185 | .192 | .157 | .096 | .060 | .249 |
| σ² at Spanish's flatness mix | .76 | .88 | .85 | .74 | .70 | .87 | .92 | .89 | .94 | .57 | .62 | 1.07 |

At matched flatness and the same noise, damage still differs: Portuguese and Swahili both have σ²
.87–.88 but lose .116 and .176. Across models the formula share stays ordered (Spanish .56–.79,
Swahili .28–.51, Amharic .13–.34). Under AWQ, Amharic at matched flatness loses .163 (σ² .77), near
Swahili's .146 (σ² .78). The formula works only as an average: per-token correlation is .22–.30.
Nahuatl and Bribri have almost no confident tokens, so their per-token numbers are not comparable.

**French resembles Spanish and Portuguese** despite not being fine-tuned, so fine-tuning does not
look like the main cause of the pattern.

**Heads.** Per-head change is in `head_change.json` in each run folder. Layer 0 has the largest
ratios (some above 1), possibly heads with small norms; not checked.

## Weight reading

Done in its own conversation; `wp/quantization/findings/queries.md` is the record and supersedes
anything here. One result used above: RTN's error has no directions above the shuffle.

## Hypotheses (not established)

- Two effects. Late: the same disturbance costs low-resource languages more, in every method. Early
  and combined: depends on the method; AWQ largely removes Amharic's. Calibration data is English,
  which could explain the second; not tested.
- The shorter late residual is why low-resource languages lose more. Against: Q'eqchi' and Swahili.
  Smaller vectors may simply go with lower confidence.
- Adrian's working idea of P: the same machinery does translation for all languages, and some
  property arises more with low-resource languages. Damage showing late does not mean it started
  late. Nothing contradicts it; nothing identifies the property yet.

## Open

- Why equal late disturbance costs Swahili more than Q'eqchi'. Candidates: how the disturbance lines
  up with the correct token, or properties of the prediction that flatness does not capture.
- Whether sentences with a shorter late residual or other properties lose more within one language.
- Whether early damage comes from non-ASCII bytes split into separate tokens.
- Seeds and methods cover three languages only (swap: five).
- Random-site control for patching: not run.

## Pitfalls already hit

- A ratio can differ because of its top or its bottom. Read ‖quant − full‖ and ‖full‖ apart.
- Do not compare per-token numbers across languages without matching: Amharic has about one token
  per byte. Matching on flatness is fairer, not a full fix.
- Matching on the reference token's log p builds in regression to the mean. Match on flatness.
- A top singular direction can be only a step-size effect. Compare with the shuffle within groups.
- 4-bit seed 0 is not representative; one seed at 4 bits can mislead.
- English is only the source side of the prompt here. It is not scored.
