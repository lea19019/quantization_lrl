'''
Calibration data for the quantizers. Description only, no code yet.
Written by Claude Code on 2026-10-15 at Adrian's request.

• What calibration data is
  RTN looks only at the weights. GPTQ, AWQ and SmoothQuant also look at the activations: the
  inputs each linear actually receives when the model reads text. To get those, we run some
  text through the model and record them. GPTQ builds its Hessian H = X X^T from them; AWQ
  and SmoothQuant measure how large each input channel gets. That text is the calibration
  data. It is never the text we evaluate on.

• Which text
  The papers use C4 (Raffel et al. 2020), a large cleaned English web crawl, so our numbers
  can be compared with theirs. One C4 file is already downloaded:
  data/c4/c4-train.00000-of-01024.json.gz. Calibrating on Yoruba or Zulu text instead is a
  planned add-on (docs/project/design.md, section 3).

• What this file will do
  Return N windows of L tokens each, as a tensor of token ids [N, L]:
  1. Read the C4 documents.
  2. Pick a random document (fixed seed, so a run can be repeated exactly).
  3. Tokenize it; if it is shorter than L tokens, pick another.
  4. Take L consecutive tokens starting at a random position.
  5. Repeat until there are N windows.
  GPTQ's paper uses N = 128, L = 2048. The seed matters: design.md asks for five
  calibration draws per method, which means five seeds.

• What it replaces
  Your gptq.py calibrates on the single sentence "The capital of France is". That is fine
  for checking the code runs; a real run needs these windows.
'''
