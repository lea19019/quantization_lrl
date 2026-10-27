# Analysis of results/eval_church

Intervals: 95% paired percentile bootstrap, 1000 resamples of sentences.
Gap: damage per byte (nats) of the language's text minus that of the English side. Drops:
full precision minus quantized (mean over seeds); (number) is inside the interval or the
seed spread; "floor": the full-precision score is not clearly above shuffled references.
Every number, with intervals, is in damage.csv and translation.csv.

## Llama-3.1-8B, rtn-4bit (1 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.81 | 0.008 [0.004, 0.012] | (0.4) | 1.1 | (0.3) | 0.9 |
| SPA | church | 0.84 | 0.014 [0.009, 0.018] | 1.3 | 1.0 | – | – |
| POR | church | 0.90 | 0.012 [0.008, 0.016] | 1.2 | 1.7 | – | – |
| YOR | church | 1.36 | 0.095 [0.088, 0.103] | 3.5 | 6.8 | floor | 9.4 |
| ZUL | church | 1.60 | 0.066 [0.059, 0.074] | (0.0) | 5.3 | floor | 6.2 |
| SWH | church | 1.32 | 0.079 [0.073, 0.085] | 6.3 | 7.7 | 9.7 | 5.0 |
| AMH | church | 0.89 | 0.226 [0.218, 0.234] | 2.8 | 8.4 | floor | 11.9 |
| KEK | church | 2.47 | 0.023 [0.018, 0.028] | floor | 1.3 | – | – |
| GRN | church | 1.91 | 0.086 [0.078, 0.093] | floor | 4.4 | – | – |

## Llama-3.1-8B, gptq-4bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.81 | 0.003 [0.000, 0.005] | 0.7 | 0.9 | 0.6 | (0.3) |
| SPA | church | 0.84 | 0.005 [0.003, 0.007] | 0.8 | 0.8 | – | – |
| POR | church | 0.90 | 0.011 [0.009, 0.013] | 0.7 | 1.3 | – | – |
| YOR | church | 1.36 | 0.066 [0.061, 0.071] | 2.6 | 3.7 | floor | 4.7 |
| ZUL | church | 1.60 | 0.049 [0.044, 0.054] | 0.5 | 2.2 | floor | 2.2 |
| SWH | church | 1.32 | 0.052 [0.049, 0.056] | 3.1 | 3.3 | 6.2 | 2.0 |
| AMH | church | 0.89 | 0.146 [0.140, 0.151] | 2.1 | 5.5 | floor | 6.8 |
| KEK | church | 2.47 | 0.019 [0.016, 0.023] | floor | 0.7 | – | – |
| GRN | church | 1.91 | 0.070 [0.065, 0.075] | floor | 0.9 | – | – |

## Llama-3.1-8B, awq-4bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.81 | -0.000 [-0.003, 0.002] | 1.0 | (0.5) | 0.8 | (0.3) |
| SPA | church | 0.84 | 0.003 [0.000, 0.005] | 1.0 | 1.2 | – | – |
| POR | church | 0.90 | 0.007 [0.004, 0.009] | 0.5 | 1.2 | – | – |
| YOR | church | 1.36 | 0.036 [0.031, 0.041] | 2.6 | 3.3 | floor | 3.8 |
| ZUL | church | 1.60 | 0.036 [0.031, 0.041] | -0.5 | 2.0 | floor | 2.4 |
| SWH | church | 1.32 | 0.044 [0.040, 0.048] | 4.6 | 3.4 | 7.2 | 2.4 |
| AMH | church | 0.89 | 0.088 [0.083, 0.092] | 1.4 | 3.6 | floor | 5.0 |
| KEK | church | 2.47 | 0.014 [0.011, 0.018] | floor | 0.6 | – | – |
| GRN | church | 1.91 | 0.066 [0.061, 0.071] | floor | 3.1 | – | – |

## Llama-3.1-8B, rtn-3bit (1 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.81 | 0.091 [0.080, 0.102] | 6.3 | 6.6 | 8.2 | 5.9 |
| SPA | church | 0.84 | 0.096 [0.086, 0.105] | 8.7 | 8.4 | – | – |
| POR | church | 0.90 | 0.128 [0.118, 0.138] | 7.8 | 9.9 | – | – |
| YOR | church | 1.36 | 0.490 [0.472, 0.508] | 8.3 | 15.7 | floor | 18.7 |
| ZUL | church | 1.60 | 0.433 [0.417, 0.450] | 7.4 | 19.6 | floor | 25.1 |
| SWH | church | 1.32 | 0.551 [0.535, 0.567] | 25.1 | 41.8 | 33.6 | 41.4 |
| AMH | church | 0.89 | 0.768 [0.751, 0.786] | 5.7 | 26.0 | floor | 35.0 |
| KEK | church | 2.47 | 0.129 [0.117, 0.141] | floor | 9.5 | – | – |
| GRN | church | 1.91 | 0.646 [0.625, 0.668] | floor | 13.7 | – | – |

## Llama-3.1-8B, gptq-3bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.81 | 0.031 [0.022, 0.039] | 4.1 | 3.8 | 3.5 | 2.0 |
| SPA | church | 0.84 | 0.049 [0.042, 0.056] | 4.8 | 4.1 | – | – |
| POR | church | 0.90 | 0.081 [0.074, 0.088] | 5.1 | 4.3 | – | – |
| YOR | church | 1.36 | 0.391 [0.376, 0.407] | 8.9 | 11.1 | floor | 14.3 |
| ZUL | church | 1.60 | 0.329 [0.315, 0.344] | 4.6 | 13.5 | floor | 13.6 |
| SWH | church | 1.32 | 0.375 [0.363, 0.387] | 17.4 | 22.5 | 25.8 | 19.4 |
| AMH | church | 0.89 | 0.715 [0.699, 0.731] | 4.9 | 17.5 | floor | 21.9 |
| KEK | church | 2.47 | 0.120 [0.110, 0.130] | floor | 5.2 | – | – |
| GRN | church | 1.91 | 0.393 [0.377, 0.407] | floor | 8.4 | – | – |

## Llama-3.1-8B, awq-3bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.81 | 0.013 [0.006, 0.021] | 4.0 | 2.4 | 3.6 | 2.1 |
| SPA | church | 0.84 | 0.013 [0.008, 0.019] | 4.5 | 3.4 | – | – |
| POR | church | 0.90 | 0.027 [0.022, 0.033] | 4.9 | 3.7 | – | – |
| YOR | church | 1.36 | 0.225 [0.213, 0.238] | 6.0 | 9.0 | floor | 11.8 |
| ZUL | church | 1.60 | 0.206 [0.193, 0.220] | 3.3 | 9.4 | floor | 10.3 |
| SWH | church | 1.32 | 0.274 [0.263, 0.285] | 16.6 | 18.3 | 23.4 | 15.4 |
| AMH | church | 0.89 | 0.484 [0.470, 0.497] | 4.5 | 13.4 | floor | 18.2 |
| KEK | church | 2.47 | 0.048 [0.040, 0.057] | floor | 2.7 | – | – |
| GRN | church | 1.91 | 0.277 [0.264, 0.290] | floor | 4.9 | – | – |

## Qwen2.5-7B, rtn-4bit (1 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.79 | 0.002 [-0.000, 0.004] | 1.1 | (0.2) | 0.9 | (0.2) |
| SPA | church | 0.83 | 0.001 [-0.001, 0.003] | 0.7 | 0.5 | – | – |
| POR | church | 0.87 | -0.003 [-0.005, -0.001] | 0.8 | (0.4) | – | – |
| YOR | church | 1.69 | 0.027 [0.024, 0.030] | floor | (0.2) | floor | (0.3) |
| ZUL | church | 1.91 | 0.026 [0.023, 0.029] | floor | 1.0 | floor | (0.0) |
| SWH | church | 1.63 | 0.031 [0.028, 0.034] | -0.8 | 1.4 | floor | 1.5 |
| AMH | church | 1.25 | 0.031 [0.029, 0.033] | floor | 1.1 | floor | 1.1 |
| KEK | church | 2.54 | 0.017 [0.015, 0.020] | floor | 0.7 | – | – |
| GRN | church | 2.44 | 0.009 [0.007, 0.011] | 1.5 | (0.4) | – | – |

## Qwen2.5-7B, gptq-4bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.79 | 0.001 [-0.001, 0.002] | 5.2 | 4.8 | 2.8 | 3.5 |
| SPA | church | 0.83 | 0.003 [0.001, 0.004] | 3.1 | 4.4 | – | – |
| POR | church | 0.87 | 0.004 [0.003, 0.005] | 2.9 | 3.8 | – | – |
| YOR | church | 1.69 | 0.036 [0.034, 0.037] | floor | (0.3) | floor | 2.6 |
| ZUL | church | 1.91 | 0.032 [0.030, 0.034] | floor | 0.7 | floor | 2.2 |
| SWH | church | 1.63 | 0.034 [0.032, 0.036] | -0.9 | 2.4 | floor | 2.9 |
| AMH | church | 1.25 | 0.060 [0.058, 0.061] | floor | 0.9 | floor | 3.3 |
| KEK | church | 2.54 | 0.011 [0.010, 0.013] | floor | (0.3) | – | – |
| GRN | church | 2.44 | 0.009 [0.008, 0.011] | 3.8 | 1.5 | – | – |

## Qwen2.5-7B, awq-4bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.79 | -0.000 [-0.002, 0.001] | 5.1 | 4.6 | 2.7 | 3.7 |
| SPA | church | 0.83 | 0.002 [0.001, 0.004] | 3.3 | 4.5 | – | – |
| POR | church | 0.87 | -0.001 [-0.002, 0.001] | 2.8 | 3.5 | – | – |
| YOR | church | 1.69 | 0.022 [0.019, 0.024] | floor | 0.5 | floor | 3.1 |
| ZUL | church | 1.91 | 0.021 [0.018, 0.023] | floor | 0.9 | floor | 2.5 |
| SWH | church | 1.63 | 0.025 [0.023, 0.027] | (0.0) | 1.9 | floor | 2.8 |
| AMH | church | 1.25 | 0.033 [0.031, 0.035] | floor | 0.7 | floor | 3.5 |
| KEK | church | 2.54 | 0.012 [0.010, 0.014] | floor | (0.2) | – | – |
| GRN | church | 2.44 | 0.014 [0.013, 0.016] | 4.2 | 2.2 | – | – |

## Qwen2.5-7B, rtn-3bit (1 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.79 | -0.003 [-0.009, 0.002] | 5.1 | 3.5 | 3.7 | 2.1 |
| SPA | church | 0.83 | 0.004 [-0.002, 0.009] | 4.9 | 3.8 | – | – |
| POR | church | 0.87 | 0.009 [0.004, 0.014] | 4.3 | 4.4 | – | – |
| YOR | church | 1.69 | 0.141 [0.134, 0.148] | floor | 1.7 | floor | 6.2 |
| ZUL | church | 1.91 | 0.152 [0.143, 0.160] | floor | 2.7 | floor | 4.3 |
| SWH | church | 1.63 | 0.175 [0.167, 0.183] | 5.5 | 9.3 | floor | 9.1 |
| AMH | church | 1.25 | 0.162 [0.155, 0.168] | floor | 3.5 | floor | 6.4 |
| KEK | church | 2.54 | 0.085 [0.079, 0.091] | floor | 2.4 | – | – |
| GRN | church | 2.44 | 0.080 [0.072, 0.087] | 6.5 | 4.3 | – | – |

## Qwen2.5-7B, gptq-3bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.79 | 0.011 [0.006, 0.015] | 4.2 | 3.2 | 3.0 | 1.7 |
| SPA | church | 0.83 | 0.017 [0.013, 0.021] | 3.7 | 2.6 | – | – |
| POR | church | 0.87 | 0.033 [0.030, 0.036] | 4.2 | 3.4 | – | – |
| YOR | church | 1.69 | 0.135 [0.129, 0.140] | floor | 0.7 | floor | 2.1 |
| ZUL | church | 1.91 | 0.134 [0.127, 0.139] | floor | 4.6 | floor | 4.0 |
| SWH | church | 1.63 | 0.162 [0.156, 0.168] | 3.9 | 6.8 | floor | 6.0 |
| AMH | church | 1.25 | 0.319 [0.312, 0.325] | floor | 3.5 | floor | 4.4 |
| KEK | church | 2.54 | 0.043 [0.038, 0.047] | floor | 1.8 | – | – |
| GRN | church | 2.44 | 0.055 [0.050, 0.060] | 1.9 | 2.7 | – | – |

## Qwen2.5-7B, awq-3bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.79 | -0.002 [-0.006, 0.003] | 5.9 | 7.0 | 5.2 | 5.2 |
| SPA | church | 0.83 | 0.009 [0.005, 0.014] | 5.1 | 7.3 | – | – |
| POR | church | 0.87 | 0.014 [0.010, 0.018] | 5.7 | 6.7 | – | – |
| YOR | church | 1.69 | 0.107 [0.100, 0.114] | floor | 2.5 | floor | 6.3 |
| ZUL | church | 1.91 | 0.109 [0.102, 0.116] | floor | 4.6 | floor | 7.7 |
| SWH | church | 1.63 | 0.159 [0.153, 0.165] | 2.9 | 8.6 | floor | 9.4 |
| AMH | church | 1.25 | 0.194 [0.189, 0.200] | floor | 4.2 | floor | 8.3 |
| KEK | church | 2.54 | 0.064 [0.058, 0.069] | floor | 3.1 | – | – |
| GRN | church | 2.44 | 0.050 [0.044, 0.056] | 6.2 | 3.9 | – | – |

## Llama-3.1-8B-lora, rtn-4bit (1 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.84 | 0.020 [0.017, 0.023] | 0.7 | 1.5 | 0.8 | 0.7 |
| SPA | church | 0.82 | 0.010 [0.007, 0.012] | (0.2) | 0.6 | – | – |
| POR | church | 0.90 | 0.004 [0.001, 0.007] | 0.4 | (0.5) | – | – |
| YOR | church | 1.25 | 0.070 [0.065, 0.075] | 1.4 | 2.6 | 0.5 | 1.9 |
| ZUL | church | 1.65 | 0.043 [0.036, 0.050] | 1.3 | 2.1 | 1.7 | 1.7 |
| SWH | church | 1.41 | 0.079 [0.073, 0.085] | 1.3 | 1.7 | 1.6 | 1.3 |
| AMH | church | 0.91 | 0.048 [0.044, 0.052] | 1.5 | 2.8 | 3.1 | 1.8 |
| KEK | church | 1.14 | 0.029 [0.024, 0.033] | 0.8 | 1.4 | – | – |
| GRN | church | 1.47 | 0.036 [0.030, 0.042] | 0.8 | 2.4 | – | – |

## Llama-3.1-8B-lora, gptq-4bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.84 | 0.012 [0.010, 0.013] | 0.6 | 0.6 | 0.6 | 0.2 |
| SPA | church | 0.82 | 0.009 [0.008, 0.010] | (0.2) | 0.4 | – | – |
| POR | church | 0.90 | 0.017 [0.016, 0.019] | 0.5 | 0.4 | – | – |
| YOR | church | 1.25 | 0.066 [0.063, 0.069] | 0.9 | 1.0 | 0.5 | 0.7 |
| ZUL | church | 1.65 | 0.073 [0.068, 0.077] | 0.9 | 1.6 | 1.0 | 0.9 |
| SWH | church | 1.41 | 0.089 [0.086, 0.093] | 1.1 | 0.6 | 1.3 | 0.5 |
| AMH | church | 0.91 | 0.061 [0.059, 0.064] | 1.6 | 1.4 | 2.5 | 0.7 |
| KEK | church | 1.14 | 0.042 [0.039, 0.044] | 0.5 | 0.5 | – | – |
| GRN | church | 1.47 | 0.046 [0.042, 0.049] | 0.5 | 1.0 | – | – |

## Llama-3.1-8B-lora, awq-4bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.84 | 0.009 [0.007, 0.010] | 0.5 | 0.7 | 0.5 | 0.2 |
| SPA | church | 0.82 | 0.007 [0.006, 0.008] | (0.3) | 0.3 | – | – |
| POR | church | 0.90 | 0.012 [0.010, 0.014] | 0.6 | (0.3) | – | – |
| YOR | church | 1.25 | 0.049 [0.046, 0.052] | 0.9 | 1.0 | 0.3 | 0.8 |
| ZUL | church | 1.65 | 0.047 [0.044, 0.051] | 0.6 | 1.3 | 0.9 | 0.9 |
| SWH | church | 1.41 | 0.046 [0.043, 0.050] | 1.1 | 0.5 | 0.6 | 0.4 |
| AMH | church | 0.91 | 0.029 [0.026, 0.031] | 1.2 | 1.6 | 2.3 | 0.6 |
| KEK | church | 1.14 | 0.022 [0.019, 0.025] | 0.4 | 0.8 | – | – |
| GRN | church | 1.47 | 0.042 [0.038, 0.046] | 0.4 | 1.2 | – | – |

## Llama-3.1-8B-lora, rtn-3bit (1 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.84 | 0.230 [0.220, 0.241] | 9.0 | 10.0 | 10.8 | 6.9 |
| SPA | church | 0.82 | 0.157 [0.149, 0.165] | 5.7 | 6.8 | – | – |
| POR | church | 0.90 | 0.205 [0.195, 0.214] | 6.2 | 8.2 | – | – |
| YOR | church | 1.25 | 0.506 [0.487, 0.526] | 30.7 | 43.3 | 24.2 | 36.4 |
| ZUL | church | 1.65 | 0.354 [0.334, 0.374] | 19.8 | 36.6 | 23.7 | 31.6 |
| SWH | church | 1.41 | 0.693 [0.676, 0.711] | 24.8 | 41.1 | 28.4 | 35.2 |
| AMH | church | 0.91 | 0.750 [0.735, 0.766] | 32.8 | 42.2 | 51.3 | 42.5 |
| KEK | church | 1.14 | 0.686 [0.668, 0.704] | 12.5 | 22.1 | – | – |
| GRN | church | 1.47 | 0.584 [0.565, 0.603] | 13.4 | 26.4 | – | – |

## Llama-3.1-8B-lora, gptq-3bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.84 | 0.087 [0.083, 0.092] | 3.0 | 2.8 | 3.7 | 1.6 |
| SPA | church | 0.82 | 0.063 [0.060, 0.067] | 2.1 | 2.1 | – | – |
| POR | church | 0.90 | 0.100 [0.095, 0.105] | 3.6 | 2.4 | – | – |
| YOR | church | 1.25 | 0.306 [0.294, 0.317] | 8.3 | 10.0 | 4.0 | 7.0 |
| ZUL | church | 1.65 | 0.253 [0.242, 0.264] | 6.5 | 9.4 | 8.6 | 7.0 |
| SWH | church | 1.41 | 0.250 [0.242, 0.259] | 6.4 | 6.5 | 8.2 | 5.0 |
| AMH | church | 0.91 | 0.462 [0.453, 0.472] | 19.9 | 21.4 | 32.0 | 18.0 |
| KEK | church | 1.14 | 0.226 [0.217, 0.233] | 3.0 | 4.0 | – | – |
| GRN | church | 1.47 | 0.221 [0.212, 0.231] | 4.0 | 6.2 | – | – |

## Llama-3.1-8B-lora, awq-3bit (5 seeds)

| Language | Test set | FP NLL/byte | Gap/byte [95% CI] | chrF++ eng→X | chrF++ X→eng | AfriCOMET eng→X | AfriCOMET X→eng |
|---|---|---|---|---|---|---|---|
| FRA | church | 0.84 | 0.056 [0.052, 0.060] | 3.1 | 3.1 | 3.6 | 1.6 |
| SPA | church | 0.82 | 0.035 [0.031, 0.039] | 2.0 | 2.5 | – | – |
| POR | church | 0.90 | 0.058 [0.054, 0.063] | 2.9 | 2.3 | – | – |
| YOR | church | 1.25 | 0.175 [0.168, 0.181] | 5.6 | 7.6 | 2.5 | 5.0 |
| ZUL | church | 1.65 | 0.128 [0.118, 0.138] | 5.2 | 8.1 | 6.4 | 5.4 |
| SWH | church | 1.41 | 0.175 [0.167, 0.183] | 5.5 | 5.6 | 6.8 | 4.2 |
| AMH | church | 0.91 | 0.305 [0.297, 0.314] | 11.0 | 9.6 | 16.6 | 6.5 |
| KEK | church | 1.14 | 0.142 [0.136, 0.149] | 3.4 | 3.9 | – | – |
| GRN | church | 1.47 | 0.159 [0.151, 0.169] | 4.0 | 6.1 | – | – |
