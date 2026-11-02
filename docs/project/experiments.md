# Experiments

Four experiments in two fronts. Front 1 looks for clues about why quantization costs
low-resource translations more confidence than English. Front 2 applies the lab peer's
detection method to quantized models.

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
- **Gap.** How much more a low-resource language loses than English on the same parallel
  sentences.
- **Languages.** English (anchor), French (high-resource, not fine-tuned), Spanish
  (high-resource, fine-tuned), and Amharic, Guarani, Yoruba and Swahili (low-resource,
  fine-tuned, above the translation floor with a clear gap). Church is the main test set,
  FLORES the check.
- **Configurations.** RTN and GPTQ (one calibration seed), at 4 and 3 bits.

### Activation patching

- **Question.** Where in the model does the extra low-resource damage come from?
- **Method.** Feed the translation prompt with the reference forced in to the full and the
  quantized model. Copy the full model's activation at one site (a layer, module or attention
  head) into the quantized model and measure how much of the damage on the reference tokens
  disappears. A first-order screen (attribution patching, Kramár 2024) scores every site from
  one forward and one backward pass. Exact patching then confirms the top sites.
- **Output.** A map of sites × languages, each cell the share of the gap recovered at that
  site.
- **Controls.** Random sites of the same size, the sites most damaged on English, and the
  same sites on French.

### Logit lens

- **Question.** At what depth does a language's prediction break?
- **Method.** Decode the residual stream at every layer through the final norm and the
  output head, in the full and the quantized model (Wendler 2024).
- **Output.** Per language, the probability of the correct next token at each layer for both
  models. The layer where the quantized curve leaves the full one is where the damage starts.
  It narrows the layers that activation patching needs to search.
- **Controls.** The same curves on English and French.

### Noise vs flatness

- **Question.** Does a low-resource language lose more because more rounding noise reaches its
  logits, or because its predictions were flatter to begin with?
- **Method.** If quantization changes the logits by noise of size σ, the expected rise in NLL
  of the correct token is about ½ σ² (1 − Σ_j p_j²), with p the full model's next-token
  distribution (second-order expansion of the log-softmax). Per token, measure σ² as the
  variance of z_quant − z_full and the flatness 1 − Σ_j p_j² from the full model, and see which
  of the two carries the gap between languages.
- **Output.** Per language, the share of the gap explained by noise and by flatness. It says
  which kind of cause to look for in the other experiments.
- **Controls.** English and French, compared at matched full-precision confidence.

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
