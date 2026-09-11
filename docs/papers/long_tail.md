# Compression forgets the long tail

The origin of E1/E2: compressed models keep average accuracy and fail on rare, under-represented or knowledge-heavy inputs. Back to [../papers.md](index.md).

### Hooker 2019. What Do Compressed Deep Neural Networks Forget?
- File: `papers/Hooker-2019-What-Do-Compressed-DNNs-Forget.pdf`
- Venue: preprint, arXiv:1911.05248
- Does: Trains populations of 30 ResNets on CIFAR-10, CelebA and ImageNet, applies magnitude pruning (30–90%) and three PTQ schemes (float16, dynamic int8, fixed-point int8), and tests per-class recall shifts with Welch's t-test; defines Pruning Identified Exemplars (PIEs) as images whose modal prediction differs between compressed and uncompressed model populations.
- Finds: Top-1 barely moves but the number of significantly affected ImageNet classes grows from 170 (50% sparsity) to 637 (90%); fixed-point int8 affects only 119 classes. At 90% sparsity 10.27% of ImageNet test images are PIEs and uncompressed top-1 on PIEs is 39.81% vs 76.75% overall; PIEs over-index on mislabeled, multi-object and fine-grained images. Pruned models are also more sensitive to ImageNet-C/A shifts.
- Use here: Origin of E1/E2 ("compression cannibalises the long tail"); supplies the per-group significance test and the PIE disagreement measure that this project can port to per-token, per-language damage; also shows PTQ is milder than pruning, so E1/E2 effects may be small.
- Note: v3 of the arXiv paper is dated 6 Sep 2021, filename year 2019 refers to v1; vision only; no venue printed.

### Hooker 2020. Characterising Bias in Compressed Models
- File: `papers/Hooker-2020-Characterising-Bias-Compressed-Models.pdf`
- Venue: preprint, arXiv:2010.03058
- Does: ResNet-18 on the CelebA blond/non-blond task with 30 models per setting, magnitude pruning 30–99% and two int8 PTQ schemes; measures error, FPR and FNR by Male/Young subgroups and introduces Compression Identified Exemplars (Modal CIE and a Taxicab-distance ranking) as a label-free auditing tool.
- Finds: At 95% pruning the normalised FPR increase is 49.54% for Male vs 6.32% for not-Male, tracking training-set share (Blond Male 0.85% vs Blond not-Male 14%). Baseline accuracy on Modal CIEs is 49.82% vs 94.76% overall; int8 PTQ yields ~404–414 modal CIEs vs 555 at 30% pruning. Underrepresented attributes over-index in CIEs.
- Use here: E1/E2 — evidence that compression amplifies error on underrepresented data; Taxicab CIE is a usable continuous disagreement score for ranking tokens or sentences by quantization damage.
- Note: single binary task; subgroup tables average 10 models; no CIs reported.

### Ahia 2021. The Low-Resource Double Bind: An Empirical Study of Pruning for Low-Resource Machine Translation
- File: `papers/Ahia-2021-Low-Resource-Double-Bind-Pruning-MT.pdf`
- Venue: Findings of EMNLP 2021
- Does: Trains 60M-parameter transformer NMT models en→{de, yo, ig, ha} on JW300 with magnitude pruning to 0–98% sparsity, in Full and Limited (200k sentence) data regimes; evaluates on a frequent-sentence Global test set, a Random test set, six out-of-distribution corpora, and a 500-pair human rating study.
- Finds: 50–80% sparsity keeps ≥95% of dense BLEU for all languages; degradation is sharper on the Random (long-tail) set above 90% sparsity. On German training data binned by typicality, low-typicality BLEU is 62.04 dense vs 24.93 at 90% sparse, i.e. capacity drives memorisation of rare sentences. Under Limited data, 90% sparse models beat dense models on OOD corpora; humans show no clear preference between dense and 90%-sparse outputs.
- Use here: Closest precedent for E1/E2 in an LRL setting (includes Yoruba); its frequent-vs-random test split and typicality binning are directly reusable for per-token frequency analysis in this project.
- Note: Pruning applied during training, not PTQ; BLEU only, no per-token measures; models trained from scratch per language.

### Tran 2022. Pruning has a disparate impact on model accuracy
- File: `papers/Tran-2022-Pruning-Disparate-Impact-Accuracy.pdf`
- Venue: NeurIPS 2022
- Does: Derives an upper bound on a group's excess loss after pruning in terms of the group's gradient norm, the maximum eigenvalue of the group Hessian, and the parameter perturbation norm; validates on UTKFace (ResNet-18/50), CIFAR-10 and SVHN with magnitude pruning, and proposes a loss-equalisation mitigation.
- Finds: Smaller groups have larger gradient norms and larger Hessian eigenvalues (i.e. sit closer to the decision boundary), so they lose more under a fixed pruning rate; the 7%-share "Others" group collapses while the 42%-share "White" group holds or improves ("rich get richer"). Metrics are averages over 10 repetitions.
- Use here: Mechanism for E1/E2 — the bound holds for any weight perturbation, including rounding noise, so per-language gradient norm and margin-to-boundary are candidate predictors of quantization damage; suggests measuring these on the from-scratch GPT-2 where groups are languages.
- Note: Vision only; theory assumes twice-differentiable loss and local optimum.

### Tropeano 2025. As easy as PIE: understanding when pruning causes language models to disagree
- File: `papers/Tropeano-2025-As-Easy-As-PIE-Pruning-Disagree.pdf`
- Venue: preprint, arXiv:2503.21714
- Does: First PIE study in NLP: BERT-base and BiLSTM on IMDB, SNLI, Reuters and AAPD with 8 unstructured pruning methods at 20–99% sparsity, 30 initialisations each (9,840 runs); characterises PIEs by class frequency, EL2N influence score and eight readability/length measures.
- Finds: PIEs occur across all classes roughly in proportion to class frequency, so disagreement is not tied to rare classes; BERT is more PIE-prone than BiLSTM. Up to 80–100% of the most influential (EL2N) training examples are PIEs for BERT. PIE text is up to 1.03x harder by readability indices, has 1.06x more difficult words and is 1.02x longer than average.
- Use here: Extends PIE to text and gives a cautionary result for E1/E2 — disagreement need not follow frequency; its finding that longer, more complex inputs are hit hardest is a weak link to E4 (fragmented LRL text is longer in tokens).
- Note: Classification only, pruning with retraining not PTQ; the class-frequency effect is only shown for multi-label datasets.

### Liebenwein 2021. Lost in Pruning: The Effects of Pruning Neural Networks beyond Test Accuracy
- File: `papers/Liebenwein-2021-Lost-In-Pruning-Beyond-Test-Accuracy.pdf`
- Venue: MLSys 2021
- Does: Iterative prune-retrain (weight thresholding, SiPP, filter thresholding, PFP) on CIFAR-10, ImageNet and VOC; defines functional-distance measures (shared informative pixels, label agreement under input noise), "prune potential" (max sparsity within a 0.5% loss margin) and "excess error" under CIFAR10-C / ImageNet-C corruptions, 3 repetitions each.
- Finds: Pruned nets stay functionally close to their parent, but prune potential collapses under distribution shift: near 0% for Gaussian/shot/impulse noise, and VGG16's weight prune potential drops from 98% on clean data to 80% averaged over corruptions. Excess-error gap reaches 2–3% and grows with prune ratio; robust retraining recovers most of it.
- Use here: Background for E1/E2 — "prune potential" translates naturally into a per-language "bit-width potential", and the excess-error framing separates baseline difficulty from added compression damage.
- Note: Vision and pruning only; the noise-agreement metric is a precursor of PIE-style disagreement.

### Jaiswal 2024. Compressing LLMs: The Truth is Rarely Pure and Never Simple
- File: `papers/Jaiswal-2024-Compressing-LLMs-Truth-Rarely-Pure-LLM-KICK.pdf`
- Venue: ICLR 2024
- Does: Introduces LLM-KICK, a knowledge-intensive evaluation suite (FreebaseQA, MMLU, in-context retrieval QA, summarisation, MT-Bench) and evaluates Vicuna-7B/13B under magnitude, SparseGPT and Wanda pruning (unstructured and N:M) and GPTQ at 16/8/4 bits, averaging 3 runs.
- Finds: Perplexity is flat up to 45–60% sparsity, yet factoid-QA drops beyond tolerance at 30–35% sparsity and 8-bit GPTQ already costs ~8–10% exact match. Quantization is more benign than pruning (4-bit GPTQ matches on MMLU for 13B, 8-bit for 7B). Humanities/social-science subjects degrade more than STEM; open-book retrieval survives to ~40–50% sparsity. SparseGPT is sensitive to calibration sample count, Wanda is not.
- Use here: E1/E2 — knowledge stored in weights fails before perplexity moves, which argues for task-level per-language metrics rather than perplexity alone; also shows calibration data matters for calibration-dependent methods (relevant to E3).
- Note: English only; only GPTQ among this project's four methods.

### Jin 2024. The Cost of Down-Scaling Language Models: Fact Recall Deteriorates before In-Context Learning
- File: `papers/Jin-2024-Cost-Of-Down-Scaling-Fact-Recall-Deteriorates.pdf`
- Venue: ICLR 2024 (per iclr.cc listing; the PDF is arXiv:2310.04680 v1 dated Oct 2023, no venue printed)
- Does: Prunes OPT-13B/30B and LLaMA-13B/33B (Pythia in appendix) with SparseGPT (Wanda in appendix) and compares closed-book QA (fact recall) against open-book, overriding-context QA and in-context learning of linear/NN/decision-tree functions; repeats the comparison for dense models of different sizes and for FFN-only vs attention-only pruning.
- Finds: With a 5% relative tolerance, fact recall holds only to 30% (TriviaQA) or 40% (WebQA) sparsity, while open-book QA holds to 50–60%, overriding QA to 70% and ICL to 60–70%; dense down-scaling shows the same split. Pruning 60% of FFN weights hurts TriviaQA 14 points more than pruning 60% of attention weights.
- Use here: E1/E2 — in-weight knowledge is the fragile capability; motivates contrasting closed-book vs in-context prompts per language and quantizing FFN vs attention separately to localise where LRL knowledge is lost.
- Note: Filename says 2024 but the arXiv v1 is dated 7 Oct 2023 and no venue is printed; pruning only, no quantization; English only.

### Wang 2026. Through a Compressed Lens: Investigating The Impact of Quantization on Factual Knowledge Recall
- File: `papers/Wang-2026-Through-Compressed-Lens-Quantization-Factual-Recall.pdf`
- Venue: TrustNLP workshop at ACL 2026 (per the acknowledgements; the PDF is arXiv:2505.13963, no venue printed)
- Does: Compares full-precision Llama3-8B, Qwen2.5-7B and Qwen2.5-14B with GPTQ (4/8-bit), AWQ and bitsandbytes (4/8-bit) on LRE one-hop recall (9,696 queries) and TwoHop-Fact, using neuron-level knowledge attribution and layer-wise entity-recall / consistency scores.
- Finds: LRE accuracy for Qwen2.5-7B falls from 63.25% to 60.10% (GPTQ4) and 60.60% (AWQ) but is unchanged at 8 bits; Llama3-8B falls from 77.62% to ~71–72% under GPTQ/AWQ; Qwen2.5-14B GPTQ4 collapses to 25.20%. Relations whose full-precision accuracy is unsaturated degrade most; contribution-score loss concentrates in the last layers (Qwen) or mid-to-late layers (Llama); the first hop of two-hop queries suffers up to 30% while the second hop loses ~4%.
- Use here: Direct support for E1/E2's "thin knowledge is fragile" claim with two of this project's models and two of its methods; the neuron/layer attribution recipe is reusable for the interpretability arm.
- Note: English only; Table 1 lists Llama3 GPTQ as 4-bit while Table 2 labels it gptq8, so the Llama GPTQ bit-width is ambiguous; single run, no CIs.

### Wu 2026. The Asymmetric Harms of LLM Compression
- File: `papers/Wu-2026-Asymmetric-Harms-LLM-Compression.pdf`
- Venue: preprint, arXiv:2608.19670
- Does: Evaluates Llama-3.1-8B-Instruct, Qwen-3-8B and Gemma-2-9B-it under 11 compression settings (GPTQ, AWQ, OmniQuant, AQLM at 2–4 bits; magnitude, Wanda, SparseGPT at 30–70% and N:M; ShortGPT and other layer dropping; SliceGPT) on PopQA and Head-to-Tail split into head/middle/tail popularity, measuring group accuracy, relative retention shift, knowledge-loss rate, confidence on newly wrong answers, and WinoBias/BBQ subgroup bias.
- Finds: Head facts stay the most accurate but are proportionally the least retained: 4-bit GPTQ/AWQ/OmniQuant shift tail retention by up to +5.3 pp relative to overall, N:M pruning by +22.7 to +39.5 pp. Even mild settings (4-bit GPTQ, 30% Wanda) flip 22–35% of previously correct PopQA answers, and median confidence on those wrong answers stays at 0.4–0.6. Small aggregate bias changes hide opposing subgroup shifts of ±6–9 pp.
- Use here: Challenges the naive form of E1/E2 — rare facts are not proportionally hit harder in these models; the project should report relative retention shift per language, knowledge-loss rate, and confidence on flipped tokens rather than only accuracy.
- Note: English only; 95% CIs given only for bias scores; instruct-tuned 8–9B models.

### Gonçalves 2023. Understanding the Effect of Model Compression on Social Bias in Large Language Models
- File: `papers/Goncalves-2023-Model-Compression-Social-Bias-LLMs.pdf`
- Venue: EMNLP 2023
- Does: Applies PyTorch dynamic int8 PTQ and distillation to BERT/RoBERTa base and large and to Pythia 70M–6.9B across 21 pretraining checkpoints, measuring intrinsic bias on Bias Bench (CrowS-Pairs, StereoSet, SEAT).
- Finds: int8 PTQ and distillation lower stereotype scores about as much as dedicated debiasing methods (e.g. RoBERTa-base CrowS gender 60.15 → 53.64) with little LM-score loss; larger models and longer pretraining are more biased; under int8 the best LM score occurs at ~20% of pretraining vs ~80% for full precision.
- Use here: Background only — shows int8 PTQ changes model behaviour in ways not visible in LM score, but has no multilingual or knowledge component.
- Note: Filename drops the cedilla (Gonçalves); intrinsic metrics only, no downstream tasks; quantization runs on CPU.

### Ramesh 2023. A Comparative Study on the Impact of Model Compression Techniques on Fairness in Language Models
- File: `papers/Ramesh-2023-Model-Compression-Fairness-Language-Models.pdf`
- Venue: ACL 2023
- Does: Benchmarks Prune-OFA pruning (85–90%), dynamic int8 PTQ and distillation (DistilBERT, MiniLM) for BERT, RoBERTa, mBERT and XLM-R on intrinsic (StereoSet, CrowS-Pairs) and extrinsic text-classification fairness (Jigsaw toxicity AUCs, AAVE–SAE sentiment flips, multilingual hate-speech and Trustpilot review equality differences for en/it/pl/pt/es and en/fr/de/da).
- Finds: Distilled models are the most biased extrinsically; quantized models score best on StereoSet ICAT but raise AAVE–SAE non-concurrent predictions (BERT-base 327 → 374); in the multilingual tasks equality-difference changes are small and inconsistent across languages and demographic dimensions.
- Use here: Background — shows compression effects on fairness vary by language and by metric, supporting extrinsic per-language evaluation; no LRLs or per-token measures.
- Note: Only int8 dynamic quantization; multilingual languages are all European.
