# Project progress

| Completed | Remaining | Deadline | Time left | Weekly pace |
|---:|---:|---:|---:|---:|
| **66.0 / 130 hours** | **64.0 hours** | **2026-12-10** | **49 days**<br>7 weeks | **9.1 hours** |

*Status on 2026-10-22; logged hours run through 2026-10-20.*

## Milestones

| Date | Completed milestone |
|---|---|
| 2026-09-09 | Established the repository, research question, initial project design and working conventions. |
| 2026-09-13 | Collected and indexed the foundational quantization papers and documented the implementation reading order. |
| 2026-09-24 | Worked through RTN papers, math and algorithm; wrote the initial implementation. Validation remains. |
| 2026-09-26 | Worked through the GPTQ paper, math and algorithm; began the implementation. |
| 2026-09-28 | Completed the initial investigation of the research problem and evaluation metrics. |
| 2026-09-29 | Consolidated course records and created a LaTeX report scaffold that builds successfully. |
| 2026-10-07 | Validated RTN, GPTQ and AWQ on WikiText-2. |
| 2026-10-13 | Built the data pipeline and defined the data and evaluation plan. |
| 2026-10-17 | Rewrote GPTQ with Cholesky of the inverse Hessian, lazy batch updates and C4 calibration. |
| 2026-10-19 | Built the evaluation (chrF++, BLEU, AfriCOMET, off-target rate, reference log-probability, perplexity) on one test set per language. |
| 2026-10-20 | Evaluated Llama 3.1 8B and Qwen 2.5 7B at full precision and at 4 and 3 bits (RTN; GPTQ and AWQ with 5 calibration seeds); analysis with bootstrap intervals and the AfriCOMET floor. |
| 2026-10-21 | LoRA fine-tune of Llama 3.1 8B on every project language except French; quantized, evaluated and analysed as the base model. |
| 2026-10-21 | Phase-1 result: in every model, method and bit width, Yoruba, Zulu, Swahili, Amharic and Guarani lose more than French, Spanish and Portuguese; after fine-tuning the extra loss roughly halves but remains. |
