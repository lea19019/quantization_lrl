# Project roadmap

This checklist orders the remaining project work by dependency, not by date.

## Core project

- [ ] Obtain the church corpus and the remaining language datasets.
- [ ] Clean, deduplicate, split and inventory the multilingual data.
- [ ] Finish the RTN implementation.
- [ ] Run RTN correctness tests on small controlled inputs.
- [ ] Validate RTN on WikiText-2.
- [ ] Finish the GPTQ implementation.
- [ ] Run GPTQ correctness tests on small controlled inputs.
- [ ] Validate GPTQ on WikiText-2.
- [ ] Build the translation and perplexity evaluation pipeline.
- [ ] Evaluate the full-precision Llama baseline.
- [ ] Quantize the base Llama model with RTN and GPTQ.
- [ ] Evaluate the quantized base models.
- [ ] Prepare the multilingual fine-tuning mixture.
- [ ] Run a small Llama fine-tuning test.
- [ ] Fine-tune Llama on the multilingual mixture.
- [ ] Evaluate the full-precision fine-tuned model.
- [ ] Quantize the fine-tuned model with RTN and GPTQ.
- [ ] Evaluate the quantized fine-tuned models.
- [ ] Run human evaluation.
- [ ] Run interpretability experiments on the model differences.
- [ ] Analyze the results and test the resulting hypotheses.
- [ ] Create the final tables and figures.
- [ ] Write the report.
- [ ] Prepare the presentation.

## AWQ stretch project

- [ ] Decide whether the core project is far enough along to add AWQ.
- [ ] Study the AWQ algorithm and choose the implementation scope.
- [ ] Implement AWQ.
- [ ] Run AWQ correctness tests on small controlled inputs.
- [ ] Validate AWQ on WikiText-2.
- [ ] Quantize the base and fine-tuned Llama models with AWQ.
- [ ] Evaluate AWQ and compare it with RTN and GPTQ.

## GPT-2 stretch project

- [ ] Decide whether the core project is far enough along to start GPT-2.
- [ ] Fix the model size, languages, data mixture and training budget.
- [ ] Obtain and clean the monolingual training data.
- [ ] Train the tokenizer.
- [ ] Implement the GPT-2 architecture and training loop.
- [ ] Add CPU correctness tests.
- [ ] Overfit a tiny batch.
- [ ] Complete a short cluster test.
- [ ] Train the model and save scheduled checkpoints.
- [ ] Quantize the checkpoints with RTN and GPTQ.
- [ ] Evaluate how quantization effects change during training.
- [ ] Add the GPT-2 results to the report.
