# Extra tables

## Dense-tool agent: closed pool

| system | mode | B=10 | B=20 | B=40 |
|---|---|---|---|---|
| bm25 | keyword | 0.241 (F1 0.10) | 0.359 (F1 0.10) | 0.524 (F1 0.09) |
| haiku-5.5 | hybrid | 0.273 (F1 0.28) | 0.432 (F1 0.32) | 0.626 (F1 0.39) |
| haiku-5.5 | keyword | 0.331 (F1 0.33) | 0.442 (F1 0.32) | 0.581 (F1 0.37) |
| haiku-5.5-dense | hybrid | 0.285 (F1 0.30) | 0.365 (F1 0.32) | 0.523 (F1 0.33) |
| sonnet-5.5 | hybrid | 0.417 (F1 0.43) | 0.488 (F1 0.42) | 0.666 (F1 0.49) |
| sonnet-5.5 | keyword | 0.400 (F1 0.40) | 0.518 (F1 0.45) | 0.585 (F1 0.46) |
| sonnet-5.5-dense | hybrid | – | – | 0.615 (F1 0.49) |
| specter2 | keyword | 0.289 (F1 0.13) | 0.469 (F1 0.13) | 0.635 (F1 0.11) |

- haiku-5.5-dense hybrid − haiku-5.5 hybrid (B=10, recall): +0.012 [-0.089, +0.098] (n=8)
- haiku-5.5-dense hybrid − specter2 keyword (B=10, recall): -0.004 [-0.159, +0.139] (n=8)
- haiku-5.5-dense hybrid − haiku-5.5 hybrid (B=20, recall): -0.067 [-0.228, +0.085] (n=8)
- haiku-5.5-dense hybrid − specter2 keyword (B=20, recall): -0.105 [-0.274, +0.045] (n=8)
- haiku-5.5-dense hybrid − haiku-5.5 hybrid (B=40, recall): -0.103 [-0.181, -0.037] (n=8)
- haiku-5.5-dense hybrid − specter2 keyword (B=40, recall): -0.112 [-0.265, +0.026] (n=8)
- sonnet-5.5-dense hybrid − sonnet-5.5 hybrid (B=40, recall): -0.051 [-0.093, -0.004] (n=8)
- sonnet-5.5-dense hybrid − sonnet-5.5 hybrid (B=40, f1): +0.003 [-0.033, +0.040] (n=8)
- sonnet-5.5-dense hybrid − specter2 keyword (B=40, recall): -0.020 [-0.105, +0.084] (n=8)
- sonnet-5.5-dense hybrid − specter2 keyword (B=40, f1): +0.383 [+0.267, +0.494] (n=8)
- sonnet-5.5-dense hybrid − bm25 keyword (B=40, recall): +0.091 [-0.025, +0.208] (n=8)
- sonnet-5.5-dense hybrid − bm25 keyword (B=40, f1): +0.401 [+0.278, +0.511] (n=8)

## Dense-tool agent: open graph

| system | mode | B=10 | B=20 | B=40 |
|---|---|---|---|---|
| bm25 | keyword | 0.161 (F1 0.07) | 0.276 (F1 0.08) | 0.396 (F1 0.07) |
| haiku-5.5 | hybrid | 0.279 (F1 0.31) | 0.450 (F1 0.36) | 0.499 (F1 0.33) |
| haiku-5.5 | keyword | 0.344 (F1 0.33) | 0.450 (F1 0.34) | 0.454 (F1 0.31) |
| haiku-5.5-dense | hybrid | 0.260 (F1 0.27) | 0.397 (F1 0.31) | 0.488 (F1 0.30) |
| sonnet-5.5 | hybrid | 0.312 (F1 0.36) | 0.493 (F1 0.44) | 0.619 (F1 0.43) |
| sonnet-5.5-dense | hybrid | – | – | 0.628 (F1 0.48) |
| specter2 | keyword | 0.244 (F1 0.10) | 0.353 (F1 0.10) | 0.528 (F1 0.08) |

- haiku-5.5-dense hybrid − haiku-5.5 hybrid (B=10, recall): -0.019 [-0.108, +0.064] (n=8)
- haiku-5.5-dense hybrid − specter2 keyword (B=10, recall): +0.016 [-0.101, +0.129] (n=8)
- haiku-5.5-dense hybrid − haiku-5.5 hybrid (B=20, recall): -0.052 [-0.140, +0.043] (n=8)
- haiku-5.5-dense hybrid − specter2 keyword (B=20, recall): +0.045 [-0.101, +0.194] (n=8)
- haiku-5.5-dense hybrid − haiku-5.5 hybrid (B=40, recall): -0.012 [-0.090, +0.084] (n=8)
- haiku-5.5-dense hybrid − specter2 keyword (B=40, recall): -0.040 [-0.186, +0.082] (n=8)
- sonnet-5.5-dense hybrid − sonnet-5.5 hybrid (B=40, recall): +0.009 [-0.057, +0.060] (n=8)
- sonnet-5.5-dense hybrid − sonnet-5.5 hybrid (B=40, f1): +0.051 [-0.007, +0.106] (n=8)
- sonnet-5.5-dense hybrid − specter2 keyword (B=40, recall): +0.100 [+0.020, +0.183] (n=8)
- sonnet-5.5-dense hybrid − specter2 keyword (B=40, f1): +0.400 [+0.274, +0.516] (n=8)
- sonnet-5.5-dense hybrid − bm25 keyword (B=40, recall): +0.232 [+0.086, +0.382] (n=8)
- sonnet-5.5-dense hybrid − bm25 keyword (B=40, f1): +0.416 [+0.280, +0.537] (n=8)

## 40 extra life-science reviews (B=40; training reviews of the 1.7B model only)

| system | mode | B=40 |
|---|---|---|
| bm25 | keyword | 0.405 (F1 0.10) |
| haiku-5.5 | hybrid | 0.493 (F1 0.35) |
| sonnet-5.5 | hybrid | 0.558 (F1 0.43) |

- sonnet-5.5 hybrid − bm25 keyword (B=40, recall): +0.154 [+0.059, +0.248] (n=40)
- haiku-5.5 hybrid − bm25 keyword (B=40, recall): +0.088 [-0.010, +0.183] (n=40)
- sonnet-5.5 hybrid − haiku-5.5 hybrid (B=40, recall): +0.066 [+0.026, +0.106] (n=40)
- sonnet-5.5 hybrid − bm25 keyword (B=40, f1): +0.330 [+0.290, +0.371] (n=40)

Pooled 48 life-science reviews (8 eval + 40):

- sonnet-5.5 hybrid − bm25 keyword (B=40, recall): +0.152 [+0.071, +0.233] (n=48)
- haiku-5.5 hybrid − bm25 keyword (B=40, recall): +0.090 [+0.006, +0.173] (n=48)
- sonnet-5.5 hybrid − haiku-5.5 hybrid (B=40, recall): +0.061 [+0.026, +0.099] (n=48)

## Training seed 1 vs seed 2: closed pool

| system | mode | B=10 | B=20 | B=40 |
|---|---|---|---|---|
| qwen3-1.7b-grpo | hybrid | 0.297 (F1 0.24) | 0.420 (F1 0.22) | 0.467 (F1 0.18) |
| qwen3-1.7b-grpo-s2 | hybrid | 0.281 (F1 0.22) | 0.415 (F1 0.22) | 0.539 (F1 0.20) |
| qwen3-1.7b-sft | hybrid | 0.285 (F1 0.26) | 0.368 (F1 0.23) | 0.433 (F1 0.21) |
| qwen3-1.7b-sft-s2 | hybrid | 0.234 (F1 0.23) | 0.387 (F1 0.24) | 0.492 (F1 0.21) |

- qwen3-1.7b-grpo hybrid − qwen3-1.7b-sft hybrid (B=20, recall): +0.052 [-0.022, +0.120] (n=8)
- qwen3-1.7b-grpo-s2 hybrid − qwen3-1.7b-sft-s2 hybrid (B=20, recall): +0.028 [-0.020, +0.077] (n=8)
- qwen3-1.7b-grpo-s2 hybrid − qwen3-1.7b-grpo hybrid (B=20, recall): -0.005 [-0.059, +0.061] (n=8)
- qwen3-1.7b-grpo hybrid − qwen3-1.7b-sft hybrid (B=40, recall): +0.034 [-0.027, +0.096] (n=8)
- qwen3-1.7b-grpo-s2 hybrid − qwen3-1.7b-sft-s2 hybrid (B=40, recall): +0.047 [+0.022, +0.074] (n=8)
- qwen3-1.7b-grpo-s2 hybrid − qwen3-1.7b-grpo hybrid (B=40, recall): +0.072 [+0.044, +0.101] (n=8)

## Training seed 1 vs seed 2: open graph

| system | mode | B=10 | B=20 | B=40 |
|---|---|---|---|---|
| qwen3-1.7b | hybrid | 0.089 (F1 0.11) | 0.081 (F1 0.08) | 0.072 (F1 0.06) |
| qwen3-1.7b-grpo | hybrid | 0.243 (F1 0.21) | 0.369 (F1 0.20) | 0.458 (F1 0.16) |
| qwen3-1.7b-grpo-s2 | hybrid | 0.197 (F1 0.15) | 0.254 (F1 0.13) | 0.438 (F1 0.15) |
| qwen3-1.7b-sft | hybrid | 0.196 (F1 0.19) | 0.322 (F1 0.22) | 0.449 (F1 0.20) |
| qwen3-1.7b-sft-s2 | hybrid | 0.171 (F1 0.14) | 0.288 (F1 0.16) | 0.364 (F1 0.15) |

- qwen3-1.7b-grpo hybrid − qwen3-1.7b-sft hybrid (B=20, recall): +0.047 [+0.010, +0.084] (n=8)
- qwen3-1.7b-grpo-s2 hybrid − qwen3-1.7b-sft-s2 hybrid (B=20, recall): -0.034 [-0.093, +0.019] (n=8)
- qwen3-1.7b-grpo-s2 hybrid − qwen3-1.7b-grpo hybrid (B=20, recall): -0.115 [-0.210, -0.029] (n=8)
- qwen3-1.7b-grpo hybrid − qwen3-1.7b-sft hybrid (B=40, recall): +0.009 [-0.048, +0.077] (n=8)
- qwen3-1.7b-grpo-s2 hybrid − qwen3-1.7b-sft-s2 hybrid (B=40, recall): +0.074 [+0.012, +0.139] (n=8)
- qwen3-1.7b-grpo-s2 hybrid − qwen3-1.7b-grpo hybrid (B=40, recall): -0.020 [-0.100, +0.057] (n=8)

## Training seed 1 vs seed 2: transfer

| system | mode | B=10 | B=20 | B=40 |
|---|---|---|---|---|
| qwen3-1.7b | hybrid | – | 0.090 (F1 0.09) | 0.112 (F1 0.10) |
| qwen3-1.7b-grpo | hybrid | – | 0.378 (F1 0.24) | 0.483 (F1 0.24) |
| qwen3-1.7b-grpo-s2 | hybrid | – | 0.391 (F1 0.26) | 0.524 (F1 0.25) |
| qwen3-1.7b-sft | hybrid | – | 0.351 (F1 0.26) | 0.459 (F1 0.27) |
| qwen3-1.7b-sft-s2 | hybrid | – | 0.376 (F1 0.27) | 0.480 (F1 0.27) |

- qwen3-1.7b-grpo hybrid − qwen3-1.7b-sft hybrid (B=20, recall): +0.027 [-0.002, +0.054] (n=24)
- qwen3-1.7b-grpo-s2 hybrid − qwen3-1.7b-sft-s2 hybrid (B=20, recall): +0.015 [-0.007, +0.038] (n=24)
- qwen3-1.7b-grpo-s2 hybrid − qwen3-1.7b-grpo hybrid (B=20, recall): +0.013 [-0.013, +0.041] (n=24)
- qwen3-1.7b-grpo hybrid − qwen3-1.7b-sft hybrid (B=40, recall): +0.025 [-0.017, +0.065] (n=24)
- qwen3-1.7b-grpo-s2 hybrid − qwen3-1.7b-sft-s2 hybrid (B=40, recall): +0.044 [+0.015, +0.076] (n=24)
- qwen3-1.7b-grpo-s2 hybrid − qwen3-1.7b-grpo hybrid (B=40, recall): +0.041 [+0.018, +0.071] (n=24)

