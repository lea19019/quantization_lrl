'''
AWQ, Activation-aware Weight Quantization (Lin et al. 2024, MLSys), for Llama 3.1 8B.
Description only, no code yet. Written by Claude Code on 2026-10-15 at Adrian's request.
Weight-only, 4 bits, groups of 128: the same grid as rtn.py.

• The problem
  RTN treats every weight the same. But about 1% of input channels carry large activations,
  and the weight columns that multiply them matter most: keeping just those in 16 bits
  removes most of the 4-bit damage (Table 1). Mixed precision is awkward on hardware, so
  AWQ protects those columns another way.

• The idea (Sec. 3.2)
  Multiply weight column j by s_j > 1 before rounding and divide input channel j by s_j:
      y = Q(w * s) * (x / s)
  Rounding moves a weight by at most half a grid step. If the step stays about the same,
  that error is now relative to a weight s times bigger, so dividing by s afterwards
  shrinks it by about 1/s. Scale too much and the group's max grows, which hurts the other
  weights, so AWQ searches for the best s:
      s = mean|x_j| ^ alpha,   alpha tried in {0, 0.05, ..., 0.95}
  and keeps the alpha whose rounded weights give outputs closest to the original ones.
  Like SmoothQuant, x / s is folded into whatever produces x (architectures.py), so it is
  free at inference. Unlike SmoothQuant, the activations stay in 16 bits.

• Roadmap, the simple version
  1. Run calibration text (data.py) through the model; save the input of each group of
     linears that share an input (q/k/v; gate/up; down).
  2. For each group, for each alpha: scale, round with rtn.py's grid, measure the output
     error on the saved inputs. Keep the best alpha.
  3. Fold the chosen s into the producer; multiply the weight columns by s.
  4. Round every linear with rtn.py's grid.

• Left out on purpose (add later if needed)
  - Measuring the error after the whole attention or MLP block instead of the linear alone.
  - Clipping: shrinking each group's range so one extreme weight does not stretch the grid.
  - Processing one decoder layer at a time to save memory.
  The archived code had all three and reached 6.620 on WikiText-2 (full precision 6.240,
  RTN 6.910). The simple version should land between RTN and that.
'''
