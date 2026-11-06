# Experiments

Two fronts. Front 1 looks for clues about why quantization costs low-resource translations
more confidence than high-resource ones. Front 2 applies the lab peer's detection method to
quantized models.

## Front 1. Clues to the property P

Phase 1 found that quantization raises the NLL of reference translations more for
low-resource languages than for English. The model loses more confidence on them. The
working idea is that some property P of how the model handles these languages exists at full
precision, and rounding turns it into extra damage. Front 1 does not identify P. It maps the
damage so that hypotheses about P can be formed and tested.

- **Model.** Llama-LoRA, full vs quantized. French is the control: LoRA did not train on it.
  Base Llama is used later, only to confirm P once it is found.
- **Setup.** The phase-1 translation setup (`qlrl/eval/logprob.py`): the few-shot
  translation prompt the LoRA was trained on, with the human reference forced in after it.
  No text is generated. Damage is the rise in NLL of the reference tokens only, the prompt
  excluded.
- **Damage.** log p_full − log p_quant of the reference, in nats, summed over a language's
  sentences. Compared across languages per byte or per word, never per token. English is only
  the source side of the prompt and is not scored.
- **Languages.** French (high-resource, not fine-tuned), Spanish and Portuguese (high-resource),
  and Yoruba, Zulu, Swahili, Amharic, Q'eqchi', Guarani, Kichwa, Nahuatl and Bribri
  (low-resource). Church is the test set; Nahuatl and Bribri have none, so they use AmericasNLP.
- **Configurations.** GPTQ seed 0 at 4 and 3 bits on every language. RTN, AWQ seed 0 and GPTQ
  seeds 1 and 2, at 4 and 3 bits, on Spanish, Amharic and Swahili. The YAMLs are listed at the
  end of this front.

### Activation patching

- **Question.** Where in the model does the extra low-resource damage come from?
- **Method.** Feed the translation prompt with the reference forced in to the full and the
  quantized model. Copy the full model's output at one site, the attention or the MLP of one
  layer, into the quantized model and measure how much of the damage on the reference tokens
  disappears. One site at a time, 64 sites, exact patching on every sentence.
- **Output.** A map of sites × languages, each cell the share of the damage recovered at that
  site. Shares overlap, so they do not add up to 1.
- **Controls.** Random sites of the same size, and the same sites on French.

### Logit lens

- **Question.** At what depth does a language's prediction break?
- **Method.** Decode the residual stream at every layer through the final norm and the
  output head, in the full and the quantized model (Wendler 2024).
- **Output.** Per language, the probability of the correct next token at each layer for both
  models. The layer where the quantized curve leaves the full one is where the damage starts.
  It narrows the layers that activation patching needs to search.
- **Controls.** The same curves on French.

### Noise vs flatness

- **Question.** Does a low-resource language lose more because more rounding noise reaches its
  logits, or because its predictions were flatter to begin with?
- **Method.** If quantization changes the logits by noise of size σ, the expected rise in NLL
  of the correct token is about ½ σ² (1 − Σ_j p_j²), with p the full model's next-token
  distribution (second-order expansion of the log-softmax). Per token, measure σ² as the
  variance of z_quant − z_full and the flatness 1 − Σ_j p_j² from the full model, and see which
  of the two carries the gap between languages.
- **Output.** Per language, the share of the damage explained by noise and by flatness. It says
  which kind of cause to look for in the other experiments.
- **Controls.** French, compared at matched full-precision confidence.

### Activation change

- **Question.** How far does rounding move the model's activations at each layer, and how large
  are those activations?
- **Method.** On the positions that predict reference tokens, measure ‖quant − full‖ and ‖full‖
  for the residual stream, the attention and the MLP of every layer, and per attention head on
  the input of `o_proj`. It uses the same forward passes as the logit lens.
- **Output.** Per language and layer, both sizes and their ratio. The two are read apart: a ratio
  can differ because the disturbance differs or because the activation does.

### Mixed-precision swap

- **Question.** Are the late layers damaged themselves, or do they amplify errors from earlier
  layers?
- **Method.** Take the full model and use the quantized weights for one range of layers only:
  early (0–5), middle (6–19) or late (20–31). Measure the damage and the activation change.
- **Output.** Per language, the share of the damage each range causes alone, what appears only
  when they are combined, and how a disturbance from one range grows through the clean layers.
- **Limit.** The hybrids are artificial: GPTQ quantized each layer on inputs from quantized
  earlier layers.

### Weight-space reading

- **Question.** Can reading the weights show what is wrong with the quantized model, and why
  its damage is larger for low-resource languages?
- **Method.** Weight-space programming (Takehi, Teoh, Zhang and Khattab 2026, blog post), in
  its large-network form. The weights do not fit in the model's context, so the AI agent reads
  them through a Python kernel without a GPU. It chooses each query from what the last one
  showed, and every query is logged. Forward passes are a separate resource with a budget fixed
  before starting.
- **Starting points.** The rounding error W_quant − W_full and the weights W_full of the late
  layers, read through the final norm and the output head, which are not quantized.
- **Rules.** Hypotheses are formed on dev. The test split stays untouched until the end and
  judges each claim once.
- **Proof.** A claim counts only if a weight edit changes the damage as it predicts, compared
  with a random edit of the same size.
- **Output.** A log of queries and hypotheses, and the claims that survived the proof.
- **Open.** The size of the forward-pass budget.

### Configs

In `configs/experiments/`. Each one writes one folder per quantized model under its output.

| YAML | Script | Languages | Quantized models | Output |
|---|---|---|---|---|
| `front1.yaml` | `run` | Spanish, Amharic, Swahili | GPTQ seed 0, 4 and 3 bits | `results/front1` |
| `front1_fra.yaml` | `run` | French | GPTQ seed 0, 4 and 3 bits | `results/front1_fra` |
| `front1_b.yaml` | `run` | Portuguese, Yoruba, Zulu, Q'eqchi' | GPTQ seed 0, 4 and 3 bits | `results/front1_b` |
| `front1_c.yaml` | `run` | Guarani, Kichwa, Nahuatl, Bribri | GPTQ seed 0, 4 and 3 bits | `results/front1_c` |
| `front1_variants.yaml` | `run` | Spanish, Amharic, Swahili | RTN; AWQ seed 0; GPTQ seeds 1, 2; 4 and 3 bits | `results/front1_variants` |
| `front1_test.yaml` | `run` | Spanish, Amharic, Swahili, 5 sentences | GPTQ seed 0, 4 and 3 bits | `results/test/front1` |
| `swap.yaml` | `swap` | Spanish, French, Amharic, Swahili | GPTQ seed 0, 4 and 3 bits | `results/swap` |
| `swap_b.yaml` | `swap` | The other eight | GPTQ seed 0, 4 and 3 bits | `results/swap_b` |
| `swap_test.yaml` | `swap` | Spanish, French, Amharic, Swahili, 5 sentences | GPTQ seed 0, 4 and 3 bits | `results/test/swap` |
| `weights.yaml` | `weights` | None: weights only, `down_proj` of layers 20–31 | GPTQ seed 0, 3 and 4 bits | `results/weights` |

`run` is `qlrl.experiments.run` (patching, logit lens, noise vs flatness, activation change),
`swap` is `qlrl.experiments.swap`, `weights` is `qlrl.experiments.weights`, a fixed first pass over
the rounding error. Every config uses Llama-LoRA.

## Front 2. Detection (lab peer's method)

Front 2 asks a different question from Front 1: whether the model's hidden states register
that its translation got worse. It uses the model's own translations, not the human
reference, and needs quality labels for them.

### The peer's method

From `external/probing_teacher_forcing` (draft: `paper/draft.md`, "Wrong Token, Right
Direction"), on NLLB-200 600M and 3.3B, en→es and de→ru.

1. **Perturb.** Feed the gold translation with teacher forcing. At one position t*, replace
   the gold token with the token at the same position of another sentence's translation
   (`src/mtprobe/extract.py`).
2. **Record the shift.** Δh = h_perturbed − h_clean at every layer and at offsets 0 to 31
   after t*. Positions before t* do not change, because attention is causal.
3. **Split the shift.** The part along embed(wrong) − embed(correct) is the injected token
   itself. The rest is how the model reacts to it, the "correction" part
   (`scripts/decompose_pilot.py`).
4. **Build the direction.** Average the correction parts over all perturbations, per layer
   and offset, and normalise. No quality labels are used.
5. **Probe.** Generate a translation greedily, re-run it with teacher forcing, and take the
   mean cosine between the hidden states and the direction (`src/mtprobe/quality_probe.py`).
   About 400 layer × offset cells are tried. The best is chosen on a selection split and
   reported on a separate confirmation split.

Their results (confirmation split): Spearman ρ against XCOMET of −0.38 (600M) and −0.42
(3.3B) per sentence, and a token-level AUROC of about 0.70 against XCOMET error spans,
above 500 random directions. Their own review names the main weakness: there is no
baseline on the model's token log-probability or entropy, and the best cells sit at the
last layer, so the probe may only track confidence.

### Whiteboard detection

- **Question.** Does the quantized model's hidden state show that its translation got worse,
  and does that signal appear sooner for low-resource languages?
- **Model.** Llama-LoRA. Llama is decoder-only, so the prompt and the translation are one
  sequence, and the perturbation and the cosines are taken on the translation tokens only.
- **Languages.** The Front 1 set.
- **Configurations.** RTN and GPTQ (one calibration seed), at every bit width from 8 to 2.
- **Step 1, language directions.** Per language and layer, the mean difference between its
  hidden states and English's, in the full and each quantized model. Check that they survive
  quantization, with the split-half noise floor (Varshney 2026). This tests that directions
  can be measured in these languages at all.
- **Step 2, rebuild the peer's direction.** Steps 1 to 4 above, on the full-precision model,
  with dev sentences of each language as the fit set.
- **Step 3, validate it.** Step 5 above on the full-precision model: does the cosine track
  per-sentence chrF++ and AfriCOMET of the model's own translations? Selection and
  confirmation splits as in the peer's work.
- **Step 4, alert along the bit widths.** Apply the validated direction to the model at bf16
  and at every configuration. An alert is a cosine past a threshold set at full precision.
- **Output.** Alert rate against bit width, one line per language, and whether a sentence's
  alert matches its drop in chrF++ and AfriCOMET.
- **Controls.** A detector built from the model's own token log-probability and entropy (the
  probe must beat it), 500 random directions matched by layer, the split-half noise floor for
  every cosine, and French and Spanish as high-resource lines.
- **Open.** Whether the direction is fixed from the full model or rebuilt in each quantized
  model. Token-level quality labels for languages XCOMET does not cover. The threshold rule.
