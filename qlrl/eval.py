'''
Evaluation: how we measure what quantization did to a model. Description only, no code yet.
Written by Claude Code on 2026-10-15 at Adrian's request. Replaces the seven archived files
in qlrl/eval/. The reasoning behind each measure is in docs/project/evaluation.md and
docs/project/problem.md, section 3.

Five pieces, in the order to build them. Each takes a model (full precision or quantized)
and writes its numbers to results/<measure>/<model>/.

• 1. WikiText-2 perplexity: is the quantizer correct?
  Not a project result, only a check that a quantizer works, because every paper reports it.
  1. Join the WikiText-2 test split into one text; tokenize it once.
  2. Cut the tokens into windows of 2048; drop the leftover.
  3. In each window, every position predicts the next token; keep log p(true next token).
     That is your gold_logprobs() in rtn.py.
  4. Perplexity = exp(-mean of all those log-probs). That is your perplexity().
  Pass: Llama 3.1 8B full precision about 6.24; 4-bit RTN a bit higher (about 6.9); GPTQ
  and AWQ in between. Data already downloaded: data/wikitext-2-raw-v1/.

• 2. Damage: how much worse does the model predict each language?
  The project's diagnostic. On the same FLORES+ devtest sentences for every language:
      d(token) = log p_full(true token) - log p_quant(true token)
  Positive means the quantized model got worse. Add d over a sentence and divide by its
  number of words (or bytes), never by tokens: a tokenizer may cut a Yoruba word into six
  pieces and an English word into one. D(language) is the mean over sentences. Its 95%
  interval: resample the sentences 1,000 times (bootstrap) and take the 2.5th and 97.5th
  percentiles. Each sentence is fed alone, with no prompt.

• 3. Translation: generate the translations
  For every test set and both directions (English -> X and X -> English):
  prompt = three fixed example pairs from the dev set, then the sentence, as
      <eng>Hello.</eng> <yor>Báwo ni.</yor>  ... (3 examples)  <eng>{sentence}</eng> <yor>
  The model continues; stop at </yor> or at 2x the source length. Greedy decoding. Same
  prompts for every model, so only the model changes. Test sets: FLORES+ devtest, the
  church test split, AmericasNLP and MayanV (docs/project/data.md).

• 4. chrF++ and BLEU: score the translations
  Compare each translation with the human reference (sacrebleu 2.6.0, settings fixed in
  data.md). chrF++ counts matching character and word pieces, so a word with the right
  stem and wrong ending still gets partial credit; it is the primary automatic score.
  BLEU is reported for comparison with other papers.

• 5. AfriCOMET: a learned score
  A model trained on human ratings reads (source, translation, reference) and predicts a
  0-1 quality score; unlike chrF++ it can credit a correct paraphrase. Only for the African
  languages (which ones is still open: docs/project/evaluation.md). It needs its own Python
  environment, because unbabel-comet pins its own torch.

• Then: the loss
  For every score: loss = full precision score - quantized score. Positive means
  quantization made things worse.
'''
