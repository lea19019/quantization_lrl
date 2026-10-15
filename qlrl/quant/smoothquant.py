'''
SmoothQuant W8A8 (Xiao et al. 2023, ICML) for Llama 3.1 8B. Description only, no code yet.
Written by Claude Code on 2026-10-15 at Adrian's request.

• The problem
  RTN and GPTQ round only the weights. To run the multiplication Y = X W^T fully in 8-bit,
  the activations X (the inputs flowing through the model) must be rounded too. Weights are
  easy to round: their values are spread evenly. Activations are not: a few input channels
  are ~100x larger than the rest (outlier features). The grid's step is set by the largest
  value, so the ordinary channels fall into one or two steps and lose their information.

• The idea (Sec. 4, Eq. 3-4)
  For any positive vector s, one value per input channel j,
      Y = X W^T = (X / s) (W * s)^T
  Exactly the same result, but dividing channel j of X by s_j shrinks the outlier, and
  multiplying column j of W by s_j moves that difficulty into the weights, which tolerate it.
      s_j = max|X_j|^alpha / max|W_j|^(1 - alpha)
  alpha = 0.5 splits the difficulty evenly; the paper uses 0.85 for Llama.

• Why it costs nothing at inference
  The inputs of q/k/v and of gate/up come out of an RMSNorm: x / rms(x) * g. Dividing g by s
  divides the norm's output by s, so "X / s" is baked into the norm (architectures.py).

• Roadmap
  1. Run calibration text (data.py) through the model; record max|X_j| per input channel at
     the inputs of q_proj and gate_proj.
  2. For each layer compute s for (q, k, v) and for (gate, up).
  3. Divide each RMSNorm's g by s; multiply the matching weight columns by s.
  4. Round every linear's weights to 8 bits: one scale per output row, symmetric.
  5. At inference, round every linear's input to 8 bits: one scale per token, symmetric.
     This happens while the model runs, so evaluation has to switch it on.

• What "done" looks like
  WikiText-2 perplexity (eval.py) close to full precision. The archived code measured
  6.240 full precision, 6.365 W8A8 without smoothing, 6.343 with smoothing (alpha 0.85).
  The smoothing should help, but only a little at 8 bits.
'''
