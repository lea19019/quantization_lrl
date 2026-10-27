# Analysis of results/eval_v2

Intervals: 95% paired percentile bootstrap, 1000 resamples of sentences.
Gap: damage per byte (nats) of the language's text minus that of the English side. Drops:
full precision minus quantized (mean over seeds); (number) is inside the interval or the
seed spread; "floor": the full-precision score is not clearly above shuffled references.
Every number, with intervals, is in damage.csv and translation.csv.

## Llama-3.1-8B, rtn-4bit (1 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.63 | 0.012 [0.010, 0.013] | 1.5 | 1.0 | 1.0 | 0.5 |
| SPA | flores | 0.69 | 0.008 [0.006, 0.010] | 0.8 | 0.6 | – | – |
| POR | flores | 0.73 | 0.012 [0.010, 0.014] | 1.9 | 1.2 | – | – |
| YOR | flores | 1.40 | 0.115 [0.111, 0.119] | 1.6 | 3.7 | floor | 5.9 |
| ZUL | flores | 1.45 | 0.118 [0.114, 0.122] | floor | 4.9 | floor | 5.9 |
| SWH | flores | 1.16 | 0.101 [0.097, 0.105] | 5.8 | 8.8 | 9.5 | 6.6 |
| AMH | flores | 0.85 | 0.250 [0.245, 0.254] | floor | 6.5 | floor | 11.6 |
| KEK | mayanv | 3.21 | 0.065 [0.057, 0.072] | floor | floor | – | – |
| GRN | flores | 1.74 | 0.133 [0.128, 0.138] | floor | 3.1 | – | – |
| NAH | americasnlp | 2.50 | 0.030 [0.024, 0.035] | floor | 1.0 | – | – |
| BZD | americasnlp | 3.02 | 0.059 [0.056, 0.063] | floor | (-0.2) | – | – |
| QUG | church | 2.05 | -0.081 [-0.088, -0.073] | floor | (-0.2) | – | – |

## Llama-3.1-8B, gptq-4bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.63 | 0.006 [0.005, 0.007] | 1.0 | 0.3 | 0.6 | 0.2 |
| SPA | flores | 0.69 | 0.006 [0.005, 0.007] | 0.6 | (0.2) | – | – |
| POR | flores | 0.73 | 0.010 [0.008, 0.011] | 0.9 | 0.6 | – | – |
| YOR | flores | 1.40 | 0.083 [0.081, 0.086] | 1.7 | 1.4 | floor | 2.1 |
| ZUL | flores | 1.45 | 0.087 [0.084, 0.089] | floor | 1.6 | floor | 1.8 |
| SWH | flores | 1.16 | 0.062 [0.060, 0.065] | 4.8 | 3.1 | 8.0 | 2.3 |
| AMH | flores | 0.85 | 0.172 [0.169, 0.174] | floor | 4.3 | floor | 7.1 |
| KEK | mayanv | 3.21 | 0.031 [0.026, 0.036] | floor | floor | – | – |
| GRN | flores | 1.74 | 0.094 [0.091, 0.097] | floor | 0.8 | – | – |
| NAH | americasnlp | 2.50 | 0.033 [0.029, 0.036] | floor | (0.1) | – | – |
| BZD | americasnlp | 3.02 | 0.014 [0.012, 0.016] | floor | (0.0) | – | – |
| QUG | church | 2.05 | -0.062 [-0.067, -0.057] | floor | (-0.6) | – | – |

## Llama-3.1-8B, awq-4bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.63 | 0.003 [0.002, 0.005] | 1.0 | (0.2) | 0.4 | (0.1) |
| SPA | flores | 0.69 | 0.003 [0.002, 0.005] | 0.4 | (0.1) | – | – |
| POR | flores | 0.73 | 0.006 [0.005, 0.008] | 0.6 | 0.4 | – | – |
| YOR | flores | 1.40 | 0.053 [0.050, 0.055] | 2.0 | 1.2 | floor | 1.7 |
| ZUL | flores | 1.45 | 0.072 [0.069, 0.074] | floor | 1.3 | floor | 1.1 |
| SWH | flores | 1.16 | 0.052 [0.050, 0.055] | 5.3 | 2.4 | 9.0 | 1.7 |
| AMH | flores | 0.85 | 0.108 [0.106, 0.111] | floor | 2.8 | floor | 5.2 |
| KEK | mayanv | 3.21 | 0.056 [0.051, 0.061] | floor | floor | – | – |
| GRN | flores | 1.74 | 0.088 [0.084, 0.091] | floor | 0.9 | – | – |
| NAH | americasnlp | 2.50 | 0.053 [0.049, 0.057] | floor | 0.6 | – | – |
| BZD | americasnlp | 3.02 | 0.036 [0.033, 0.038] | floor | (0.0) | – | – |
| QUG | church | 2.05 | -0.057 [-0.063, -0.052] | floor | (-0.3) | – | – |

## Llama-3.1-8B, rtn-3bit (1 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.63 | 0.100 [0.094, 0.106] | 9.5 | 8.1 | 10.5 | 7.2 |
| SPA | flores | 0.69 | 0.090 [0.085, 0.095] | 5.1 | 6.7 | – | – |
| POR | flores | 0.73 | 0.132 [0.126, 0.138] | 9.6 | 13.8 | – | – |
| YOR | flores | 1.40 | 0.510 [0.501, 0.520] | 2.4 | 9.8 | floor | 20.7 |
| ZUL | flores | 1.45 | 0.570 [0.560, 0.579] | floor | 12.8 | floor | 21.7 |
| SWH | flores | 1.16 | 0.630 [0.619, 0.639] | 27.6 | 34.4 | 36.2 | 43.0 |
| AMH | flores | 0.85 | 0.852 [0.842, 0.861] | floor | 14.0 | floor | 28.6 |
| KEK | mayanv | 3.21 | 0.170 [0.156, 0.184] | floor | floor | – | – |
| GRN | flores | 1.74 | 0.830 [0.812, 0.848] | floor | 11.4 | – | – |
| NAH | americasnlp | 2.50 | 0.348 [0.333, 0.363] | floor | 4.7 | – | – |
| BZD | americasnlp | 3.02 | 0.259 [0.249, 0.268] | floor | 4.4 | – | – |
| QUG | church | 2.05 | 0.090 [0.076, 0.103] | floor | 1.8 | – | – |

## Llama-3.1-8B, gptq-3bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.63 | 0.050 [0.047, 0.053] | 5.8 | 1.9 | 3.9 | 1.1 |
| SPA | flores | 0.69 | 0.052 [0.048, 0.054] | 2.7 | 1.7 | – | – |
| POR | flores | 0.73 | 0.081 [0.078, 0.085] | 5.1 | 2.6 | – | – |
| YOR | flores | 1.40 | 0.425 [0.417, 0.433] | 7.9 | 7.4 | floor | 11.0 |
| ZUL | flores | 1.45 | 0.455 [0.449, 0.462] | floor | 9.1 | floor | 12.0 |
| SWH | flores | 1.16 | 0.427 [0.420, 0.433] | 19.0 | 18.6 | 29.5 | 17.6 |
| AMH | flores | 0.85 | 0.800 [0.792, 0.807] | floor | 13.2 | floor | 20.6 |
| KEK | mayanv | 3.21 | 0.167 [0.154, 0.178] | floor | floor | – | – |
| GRN | flores | 1.74 | 0.514 [0.504, 0.524] | floor | 5.2 | – | – |
| NAH | americasnlp | 2.50 | 0.229 [0.221, 0.238] | floor | 2.9 | – | – |
| BZD | americasnlp | 3.02 | 0.126 [0.120, 0.131] | floor | 1.1 | – | – |
| QUG | church | 2.05 | -0.074 [-0.086, -0.063] | floor | 1.0 | – | – |

## Llama-3.1-8B, awq-3bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.63 | 0.028 [0.025, 0.031] | 5.9 | 1.6 | 3.8 | 0.8 |
| SPA | flores | 0.69 | 0.024 [0.020, 0.027] | 3.2 | 1.9 | – | – |
| POR | flores | 0.73 | 0.034 [0.031, 0.037] | 5.2 | 2.2 | – | – |
| YOR | flores | 1.40 | 0.270 [0.263, 0.276] | 5.6 | 5.1 | floor | 9.9 |
| ZUL | flores | 1.45 | 0.312 [0.306, 0.319] | floor | 6.2 | floor | 10.0 |
| SWH | flores | 1.16 | 0.296 [0.290, 0.302] | 18.9 | 16.0 | 28.0 | 15.7 |
| AMH | flores | 0.85 | 0.556 [0.549, 0.563] | floor | 11.4 | floor | 18.8 |
| KEK | mayanv | 3.21 | 0.120 [0.109, 0.131] | floor | floor | – | – |
| GRN | flores | 1.74 | 0.373 [0.365, 0.382] | floor | 4.1 | – | – |
| NAH | americasnlp | 2.50 | 0.172 [0.165, 0.181] | floor | 1.1 | – | – |
| BZD | americasnlp | 3.02 | 0.066 [0.060, 0.071] | floor | 0.7 | – | – |
| QUG | church | 2.05 | -0.100 [-0.112, -0.088] | floor | 0.7 | – | – |

## Qwen2.5-7B, rtn-4bit (1 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.62 | 0.003 [0.001, 0.004] | 1.2 | 0.6 | 0.7 | (0.1) |
| SPA | flores | 0.69 | 0.003 [0.001, 0.004] | 0.5 | 0.8 | – | – |
| POR | flores | 0.71 | -0.003 [-0.004, -0.001] | 1.0 | 0.8 | – | – |
| YOR | flores | 1.67 | 0.028 [0.026, 0.030] | floor | 0.8 | floor | 0.9 |
| ZUL | flores | 1.80 | 0.029 [0.026, 0.031] | floor | 0.5 | floor | 0.8 |
| SWH | flores | 1.51 | 0.034 [0.032, 0.037] | floor | 0.6 | floor | 0.7 |
| AMH | flores | 1.19 | 0.029 [0.027, 0.030] | floor | (0.2) | floor | (0.3) |
| KEK | mayanv | 3.27 | 0.021 [0.016, 0.027] | floor | floor | – | – |
| GRN | flores | 2.37 | 0.012 [0.010, 0.014] | floor | (0.0) | – | – |
| NAH | americasnlp | 2.66 | 0.020 [0.016, 0.024] | floor | (0.2) | – | – |
| BZD | americasnlp | 3.05 | 0.023 [0.019, 0.026] | floor | (-0.3) | – | – |
| QUG | church | 2.17 | 0.005 [0.003, 0.008] | floor | 1.0 | – | – |

## Qwen2.5-7B, gptq-4bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.62 | 0.003 [0.002, 0.004] | 1.0 | 0.3 | 0.4 | (0.1) |
| SPA | flores | 0.69 | 0.002 [0.002, 0.003] | 0.4 | (0.2) | – | – |
| POR | flores | 0.71 | 0.005 [0.004, 0.006] | 0.6 | 0.3 | – | – |
| YOR | flores | 1.67 | 0.035 [0.034, 0.036] | floor | 0.3 | floor | (0.2) |
| ZUL | flores | 1.80 | 0.037 [0.036, 0.039] | floor | 0.4 | floor | 0.6 |
| SWH | flores | 1.51 | 0.038 [0.036, 0.039] | floor | 0.8 | floor | 1.0 |
| AMH | flores | 1.19 | 0.061 [0.060, 0.062] | floor | 0.5 | floor | 0.7 |
| KEK | mayanv | 3.27 | -0.005 [-0.009, -0.002] | floor | floor | – | – |
| GRN | flores | 2.37 | 0.011 [0.010, 0.013] | floor | (-0.1) | – | – |
| NAH | americasnlp | 2.66 | 0.019 [0.017, 0.022] | floor | (0.2) | – | – |
| BZD | americasnlp | 3.05 | 0.005 [0.004, 0.007] | floor | (-0.1) | – | – |
| QUG | church | 2.17 | -0.005 [-0.007, -0.003] | floor | (0.2) | – | – |

## Qwen2.5-7B, awq-4bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.62 | 0.002 [0.001, 0.003] | 1.0 | (0.3) | 0.5 | (0.1) |
| SPA | flores | 0.69 | 0.002 [0.001, 0.003] | 0.6 | (0.2) | – | – |
| POR | flores | 0.71 | 0.001 [-0.000, 0.002] | 0.8 | 0.4 | – | – |
| YOR | flores | 1.67 | 0.025 [0.023, 0.026] | floor | (0.1) | floor | (-0.0) |
| ZUL | flores | 1.80 | 0.029 [0.028, 0.031] | floor | (0.3) | floor | 0.6 |
| SWH | flores | 1.51 | 0.030 [0.028, 0.032] | floor | 0.7 | floor | 0.9 |
| AMH | flores | 1.19 | 0.040 [0.038, 0.041] | floor | (0.2) | floor | (0.4) |
| KEK | mayanv | 3.27 | 0.039 [0.035, 0.043] | floor | floor | – | – |
| GRN | flores | 2.37 | 0.016 [0.014, 0.017] | floor | (0.1) | – | – |
| NAH | americasnlp | 2.66 | 0.010 [0.007, 0.013] | floor | 0.4 | – | – |
| BZD | americasnlp | 3.05 | 0.004 [0.002, 0.006] | floor | (-0.2) | – | – |
| QUG | church | 2.17 | -0.005 [-0.007, -0.003] | floor | 0.5 | – | – |

## Qwen2.5-7B, rtn-3bit (1 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.62 | 0.007 [0.004, 0.011] | 6.4 | 3.6 | 4.6 | 2.1 |
| SPA | flores | 0.69 | 0.007 [0.004, 0.010] | 4.3 | 3.1 | – | – |
| POR | flores | 0.71 | 0.016 [0.013, 0.019] | 7.3 | 3.7 | – | – |
| YOR | flores | 1.67 | 0.144 [0.139, 0.150] | floor | 2.1 | floor | 6.4 |
| ZUL | flores | 1.80 | 0.192 [0.186, 0.197] | floor | 2.5 | floor | 5.4 |
| SWH | flores | 1.51 | 0.204 [0.199, 0.210] | floor | 6.6 | floor | 10.0 |
| AMH | flores | 1.19 | 0.195 [0.190, 0.200] | floor | 1.6 | floor | 3.7 |
| KEK | mayanv | 3.27 | 0.166 [0.153, 0.178] | floor | floor | – | – |
| GRN | flores | 2.37 | 0.099 [0.094, 0.104] | floor | 2.1 | – | – |
| NAH | americasnlp | 2.66 | 0.056 [0.046, 0.065] | floor | 1.6 | – | – |
| BZD | americasnlp | 3.05 | 0.079 [0.072, 0.086] | floor | 0.7 | – | – |
| QUG | church | 2.17 | 0.043 [0.036, 0.049] | floor | 2.8 | – | – |

## Qwen2.5-7B, gptq-3bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.62 | 0.019 [0.017, 0.021] | 6.6 | 3.6 | 4.0 | (2.3) |
| SPA | flores | 0.69 | 0.020 [0.018, 0.022] | 3.5 | (2.9) | – | – |
| POR | flores | 0.71 | 0.034 [0.031, 0.036] | 6.3 | (3.4) | – | – |
| YOR | flores | 1.67 | 0.146 [0.142, 0.149] | floor | 1.8 | floor | 4.4 |
| ZUL | flores | 1.80 | 0.166 [0.162, 0.169] | floor | 2.0 | floor | 4.1 |
| SWH | flores | 1.51 | 0.182 [0.178, 0.185] | floor | 5.4 | floor | 7.5 |
| AMH | flores | 1.19 | 0.342 [0.338, 0.345] | floor | 1.9 | floor | 4.1 |
| KEK | mayanv | 3.27 | 0.006 [-0.002, 0.013] | floor | floor | – | – |
| GRN | flores | 2.37 | 0.074 [0.071, 0.077] | floor | 1.7 | – | – |
| NAH | americasnlp | 2.66 | 0.056 [0.051, 0.061] | floor | 1.0 | – | – |
| BZD | americasnlp | 3.05 | 0.052 [0.048, 0.055] | floor | 0.5 | – | – |
| QUG | church | 2.17 | -0.016 [-0.021, -0.012] | floor | 2.4 | – | – |

## Qwen2.5-7B, awq-3bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.62 | 0.007 [0.005, 0.010] | 5.4 | 4.2 | 4.7 | (3.8) |
| SPA | flores | 0.69 | 0.010 [0.007, 0.013] | 2.9 | 3.5 | – | – |
| POR | flores | 0.71 | 0.020 [0.017, 0.023] | 4.7 | 4.2 | – | – |
| YOR | flores | 1.67 | 0.127 [0.123, 0.131] | floor | 2.8 | floor | 7.1 |
| ZUL | flores | 1.80 | 0.143 [0.139, 0.147] | floor | 2.6 | floor | 5.7 |
| SWH | flores | 1.51 | 0.179 [0.174, 0.183] | floor | 5.9 | floor | 9.4 |
| AMH | flores | 1.19 | 0.220 [0.217, 0.224] | floor | 2.6 | floor | 5.8 |
| KEK | mayanv | 3.27 | 0.124 [0.114, 0.136] | floor | floor | – | – |
| GRN | flores | 2.37 | 0.065 [0.061, 0.069] | floor | 2.8 | – | – |
| NAH | americasnlp | 2.66 | 0.069 [0.061, 0.077] | floor | 1.3 | – | – |
| BZD | americasnlp | 3.05 | 0.071 [0.066, 0.076] | floor | 0.7 | – | – |
| QUG | church | 2.17 | -0.013 [-0.018, -0.008] | floor | 2.2 | – | – |

## Llama-3.1-8B-lora, rtn-4bit (1 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.70 | 0.024 [0.022, 0.026] | 0.9 | 0.5 | 0.7 | 0.3 |
| SPA | flores | 0.76 | 0.010 [0.008, 0.012] | (0.2) | 0.6 | – | – |
| POR | flores | 0.80 | 0.011 [0.009, 0.013] | 0.7 | (0.3) | – | – |
| YOR | flores | 1.61 | 0.050 [0.046, 0.053] | 1.0 | 1.7 | 1.5 | 2.8 |
| ZUL | flores | 1.77 | 0.063 [0.057, 0.068] | 1.9 | 1.8 | 3.8 | 2.6 |
| SWH | flores | 1.43 | 0.084 [0.080, 0.088] | 1.3 | 2.8 | 3.1 | 3.0 |
| AMH | flores | 1.12 | 0.044 [0.040, 0.047] | 2.1 | 2.0 | 4.6 | 2.8 |
| KEK | mayanv | 2.44 | 0.041 [0.033, 0.050] | 0.6 | 1.3 | – | – |
| GRN | flores | 1.56 | 0.033 [0.029, 0.037] | 0.8 | 1.4 | – | – |
| NAH | americasnlp | 2.42 | 0.100 [0.093, 0.107] | 1.4 | 1.1 | – | – |
| BZD | americasnlp | 2.32 | 0.083 [0.079, 0.088] | 0.5 | 0.8 | – | – |
| QUG | church | 1.96 | -0.048 [-0.059, -0.035] | 1.8 | 4.3 | – | – |

## Llama-3.1-8B-lora, gptq-4bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.70 | 0.013 [0.012, 0.014] | 0.7 | 0.3 | 0.5 | 0.2 |
| SPA | flores | 0.76 | 0.011 [0.010, 0.012] | (0.1) | 0.3 | – | – |
| POR | flores | 0.80 | 0.017 [0.016, 0.019] | 0.8 | (0.2) | – | – |
| YOR | flores | 1.61 | 0.048 [0.046, 0.050] | 0.4 | 1.0 | 0.5 | 0.8 |
| ZUL | flores | 1.77 | 0.069 [0.066, 0.073] | 1.5 | 0.9 | 2.6 | 0.9 |
| SWH | flores | 1.43 | 0.081 [0.079, 0.084] | 1.0 | 1.6 | 2.5 | 1.0 |
| AMH | flores | 1.12 | 0.060 [0.058, 0.062] | 1.9 | 0.9 | 4.5 | 1.4 |
| KEK | mayanv | 2.44 | 0.023 [0.018, 0.027] | (0.2) | 0.5 | – | – |
| GRN | flores | 1.56 | 0.034 [0.032, 0.036] | 0.8 | 0.5 | – | – |
| NAH | americasnlp | 2.42 | 0.045 [0.042, 0.049] | (0.5) | (0.3) | – | – |
| BZD | americasnlp | 2.32 | 0.029 [0.026, 0.031] | 0.3 | (0.2) | – | – |
| QUG | church | 1.96 | -0.008 [-0.015, -0.001] | 0.8 | 2.1 | – | – |

## Llama-3.1-8B-lora, awq-4bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.70 | 0.009 [0.008, 0.010] | 0.6 | 0.4 | (0.3) | 0.2 |
| SPA | flores | 0.76 | 0.008 [0.007, 0.009] | 0.3 | 0.4 | – | – |
| POR | flores | 0.80 | 0.014 [0.012, 0.015] | 0.8 | 0.6 | – | – |
| YOR | flores | 1.61 | 0.036 [0.034, 0.038] | 0.2 | 1.0 | (0.3) | 1.2 |
| ZUL | flores | 1.77 | 0.045 [0.042, 0.049] | 1.3 | 1.3 | 1.9 | 1.4 |
| SWH | flores | 1.43 | 0.047 [0.044, 0.049] | 0.6 | 1.7 | 1.6 | 1.2 |
| AMH | flores | 1.12 | 0.046 [0.044, 0.049] | 1.8 | 1.0 | 4.4 | 1.7 |
| KEK | mayanv | 2.44 | 0.038 [0.033, 0.042] | (0.2) | (0.4) | – | – |
| GRN | flores | 1.56 | 0.041 [0.038, 0.044] | 0.7 | 0.8 | – | – |
| NAH | americasnlp | 2.42 | 0.039 [0.035, 0.042] | 0.5 | 0.4 | – | – |
| BZD | americasnlp | 2.32 | 0.037 [0.034, 0.039] | 0.7 | 0.3 | – | – |
| QUG | church | 1.96 | -0.021 [-0.028, -0.015] | 0.8 | 2.3 | – | – |

## Llama-3.1-8B-lora, rtn-3bit (1 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.70 | 0.227 [0.220, 0.233] | 8.9 | 8.0 | 11.8 | 6.5 |
| SPA | flores | 0.76 | 0.174 [0.168, 0.181] | 4.7 | 6.9 | – | – |
| POR | flores | 0.80 | 0.229 [0.222, 0.236] | 7.1 | 9.0 | – | – |
| YOR | flores | 1.61 | 0.405 [0.394, 0.417] | 14.6 | 14.5 | 30.5 | 20.8 |
| ZUL | flores | 1.77 | 0.317 [0.304, 0.330] | 17.7 | 19.8 | 26.9 | 26.3 |
| SWH | flores | 1.43 | 0.661 [0.648, 0.674] | 24.7 | 30.6 | 35.4 | 37.3 |
| AMH | flores | 1.12 | 0.664 [0.652, 0.676] | 17.4 | 20.7 | 33.7 | 30.4 |
| KEK | mayanv | 2.44 | 0.737 [0.709, 0.764] | 4.3 | 13.2 | – | – |
| GRN | flores | 1.56 | 0.539 [0.527, 0.552] | 7.9 | 13.5 | – | – |
| NAH | americasnlp | 2.42 | 0.511 [0.494, 0.526] | 3.0 | 6.9 | – | – |
| BZD | americasnlp | 2.32 | 0.467 [0.455, 0.480] | 2.5 | 5.3 | – | – |
| QUG | church | 1.96 | 0.231 [0.202, 0.263] | 28.3 | 34.1 | – | – |

## Llama-3.1-8B-lora, gptq-3bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.70 | 0.092 [0.089, 0.095] | 3.9 | 1.8 | 3.8 | 1.1 |
| SPA | flores | 0.76 | 0.074 [0.071, 0.077] | 1.8 | 1.8 | – | – |
| POR | flores | 0.80 | 0.109 [0.105, 0.112] | 3.7 | 2.0 | – | – |
| YOR | flores | 1.61 | 0.202 [0.196, 0.208] | 3.4 | 4.6 | 6.6 | 5.6 |
| ZUL | flores | 1.77 | 0.225 [0.218, 0.233] | 6.5 | 5.6 | 12.6 | 6.3 |
| SWH | flores | 1.43 | 0.242 [0.236, 0.248] | 7.3 | 8.3 | 13.6 | 8.0 |
| AMH | flores | 1.12 | 0.348 [0.341, 0.356] | 12.0 | 10.7 | 26.9 | 14.2 |
| KEK | mayanv | 2.44 | 0.242 [0.228, 0.256] | 1.5 | 6.1 | – | – |
| GRN | flores | 1.56 | 0.197 [0.191, 0.204] | 3.0 | 3.6 | – | – |
| NAH | americasnlp | 2.42 | 0.187 [0.179, 0.194] | 2.4 | 2.0 | – | – |
| BZD | americasnlp | 2.32 | 0.125 [0.120, 0.131] | 1.8 | 1.8 | – | – |
| QUG | church | 1.96 | 0.064 [0.048, 0.083] | 7.9 | 10.9 | – | – |

## Llama-3.1-8B-lora, awq-3bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | flores | 0.70 | 0.063 [0.061, 0.066] | 3.9 | 2.4 | 4.0 | 1.5 |
| SPA | flores | 0.76 | 0.046 [0.043, 0.049] | 1.9 | 2.2 | – | – |
| POR | flores | 0.80 | 0.069 [0.066, 0.073] | 3.3 | 2.3 | – | – |
| YOR | flores | 1.61 | 0.118 [0.112, 0.123] | 2.8 | 4.5 | 5.0 | 6.0 |
| ZUL | flores | 1.77 | 0.147 [0.140, 0.154] | 5.9 | 6.1 | 10.3 | 7.2 |
| SWH | flores | 1.43 | 0.190 [0.184, 0.195] | 6.3 | 8.9 | 11.7 | 9.1 |
| AMH | flores | 1.12 | 0.358 [0.351, 0.364] | 9.4 | 6.8 | 21.0 | 8.8 |
| KEK | mayanv | 2.44 | 0.076 [0.066, 0.086] | 1.3 | 2.7 | – | – |
| GRN | flores | 1.56 | 0.142 [0.136, 0.149] | 3.4 | 3.8 | – | – |
| NAH | americasnlp | 2.42 | 0.178 [0.170, 0.186] | 2.6 | 2.6 | – | – |
| BZD | americasnlp | 2.32 | 0.164 [0.158, 0.170] | 2.5 | 1.6 | – | – |
| QUG | church | 1.96 | 0.041 [0.027, 0.055] | 8.9 | 10.6 | – | – |
