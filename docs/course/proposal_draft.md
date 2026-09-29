CS Master’s Project Proposal Draft
Adrian Castillo

Why does post-training quantization hurt low-resource languages more?


Problem

Post-training quantization (PTQ) rounds a trained LLM's weights to 4 or 3 bits without retraining. An 8B model that needs 16 GB in bf16 then fits in 4 GB and runs close to four times faster, which is how LLMs reach cheap hardware. The cost is uneven across languages. Marie & Fujita (2025) measured translation quality on 55 languages with COMET, reported here times 100. Quantizing Llama 3.1 8B to 4 bits (NF4) moves translation into English from 81.6 to 81.3 in French and 79.3 to 77.8 in Japanese, but from 72.2 to 64.5 in Bengali and 71.8 to 62.1 in Malayalam; on Llama 3.3 70B, Polish goes from 81.1 to 80.2 and Zulu from 66.9 to 60.7. 



Marchisio et al. (2024) found the same ordering on the Aya and Command models with accuracy, BLEU and language-confusion benchmarks, then had native speakers compare quantized and full-precision answers to the same 150 prompts: at 4 bits the quantized Command R+ won 42% of the Japanese comparisons, against the 50% expected if nothing had changed, while the automatic benchmarks had moved 1.7% on average. Chimoto et al. (2026) found the ordering on Llama 3.1 8B and Qwen 2.5 7B in per-language perplexity, with no human check. 


The languages that lose most are the ones with the least training data, and their speakers are the ones on cheap hardware, so quantization hurts most exactly where it is needed most. The gap has been described several times but never explained: we do not know why rounding the same weights removes more of a model's ability to predict the next word in a low-resource language than in a high-resource one.

Knowing why decides the fix. If the cause is how little the model has learned of the language, the fix is more training on that language before quantizing. If the cause is that the damage concentrates in particular parts of the network that these languages depend on, the fix is to keep those parts in higher precision at a small memory cost. Today practitioners guess, and the best published fix, calibrating on multilingual text, recovers part of the loss and leaves roughly a twofold gap (Chimoto et al. 2026). There is also a measurement problem to solve on the way: the published studies each define damage differently, none repeats its runs, and so part of what is reported as a language effect may be noise; establishing the effect cleanly is part of the work.

Research Question and Hypotheses

Why does PTQ remove more of a multilingual LLM's ability to predict the next word in low-resource languages than in high-resource ones? 

Two hypotheses will be tested, with a null that both must beat.

H1, under-fitted languages: the model has learned the low-resource languages less thoroughly, so its weights are only roughly right for them, and the same small change from rounding disturbs a language more when its weights were only roughly right to begin with. Training settles the weights where they fit the data the model saw most, mostly English; for a language it saw little of, the loss is still steep at the final weights, and rounding, a nudge of the same size for every language, costs more where the loss is steep. Tran et al. (2022) prove this for pruning, and Zeng et al. (2024) find that the larger a language's share of the training data, the more resistant it is to compression. If true, how steep each language's loss is at the released weights predicts its damage, and in a model whose data shares we set, damage follows the share.

H2, where the damage lives: rounding error concentrates in specific modules and dimensions of the network, and low-resource languages depend on those parts more than high-resource languages do. Chang et al. (2025) found, within English, that restoring the outputs of the MLP down-projections in a quantized model recovers most of its loss while restoring attention barely helps, and An et al. (2025) show that the largest weights sit in those same down-projections and feed the few residual dimensions the model depends on most. No paper compares languages. If true, quantizing one module at a time locates each language's loss in different places or to different degrees, and patching the full model's activations into the quantized model at those places recovers the low-resource loss.

The null: the difference between languages is explained by the model, the quantization method, the bit width, the metric and run-to-run noise, not by the language. It is tested first, with five calibration draws per method and a second model family; if the 4-bit gap does not exceed that noise, H1 and H2 are judged at 3 bits, where the gap is unambiguous.

Methodology

The core model is Llama 3.1 8B, which the closest papers use and which lists none of the three lab languages as supported. A GPT-2 small (124M parameters) is trained from scratch on a five-language mix whose shares we set, the only setting where the amount of training data per language is controlled. Qwen 2.5 7B and the Gemma 4 pair (mixture-of-experts and dense) are add-ons that replicate the ordering. The languages are English and French as high-resource anchors and Swahili, Yoruba and Zulu from the lab. The methods are round-to-nearest, GPTQ, AWQ, SmoothQuant and bitsandbytes NF4, at 4, 3 and 2 bits where supported, implemented from the papers and validated against published WikiText-2 perplexities first. Everything is measured on FLORES-200 devtest, 1012 sentences translated across the five languages and used for evaluation only. The goal metric is translation quality: COMET, and human ratings by speakers of the three lab languages on 100 fixed sentences each. The diagnostic is damage: the drop in log-probability of the correct next word between the full and the quantized model, averaged per language with a bootstrap interval and compared to English on the same sentences. Whether low damage means good translation is measured, not assumed.

The experiments run in four phases. Phase 0: the baseline, full precision against every method and bit width, and the null decision. Phase 1, no training: per-language gradient norm and curvature at the full-precision weights (H1); quantizing one module or layer at a time and recording residual norms, massive activations and language dimensions in both models, per language (H2). Phase 2: quantize the GPT-2's saved checkpoints to plot damage per language against the tokens seen for that language (H1). Phase 3: continue training Llama on the lab's Swahili text, re-quantize and check whether Swahili's damage fell while English held (H1); activation patching and a logit lens to confirm where the damage lives (H2). The GPT-2 corpora and shares will be settled with the advisor in the first two weeks. Each hypothesis gets a prediction, a measurement and a criterion fixed before the run.

Objectives

Reproduce the low-resource quantization gap on Llama 3.1 8B under a definition of damage that survives normalization, floors and calibration-draw noise, at the prediction level and in translation quality including human judgment.
Test H1 by measuring per-language gradient norm and curvature, by training a model whose data shares are controlled, and by moving one language along its learning curve through fine-tuning.
Test H2 by locating the damage per module, layer and dimension for each language, and confirming causally with activation patching.
State a verdict per hypothesis against criteria fixed in advance, and identify the intervention the mechanism implies.

What will be learned

How the main PTQ methods work at the weight level and why they fail, by implementing them and validating against published numbers; how to define a measurement so that a claimed difference is not an artifact of metric, floor or noise; how to design experiments with predictions and controls written before the run; small-scale pretraining and large-model fine-tuning; interpretability methods on transformers (per-module ablation, activation statistics, patching, logit lens); multilingual translation evaluation including human evaluation; and reproducible experiment engineering on the cluster.

Deliverables for Grade:
Written and verbal reports
Repository with codebase, configs, tests, and experimental results
Midterm and final report
Trained GPT-2 checkpoints and fine-tuned Llama model
Weekly meetings: Thursdays at 2 pm.
