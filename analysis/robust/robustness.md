# Robustness checks

## Temperature 0.7 repeats (closed pool, hybrid, 8 reviews, 3 repeats)

| agent | B | T=0 recall | T=0.7 repeat means | spread | mean per-review SD |
|---|---|---|---|---|---|
| haiku-5.5 | 10 | 0.273 | 0.367, 0.338, 0.342 | 0.029 | 0.065 |
| haiku-5.5 | 20 | 0.432 | 0.496, 0.436, 0.443 | 0.060 | 0.066 |
| haiku-5.5 | 40 | 0.626 | 0.551, 0.525, 0.585 | 0.060 | 0.045 |
| sonnet-5.5 | 40 | 0.666 | 0.637, 0.655, 0.654 | 0.018 | 0.016 |

Sonnet − Haiku hybrid recall at B=40, per repeat (paired over reviews):

- repeat 0: +0.086 [-0.038, +0.251]
- repeat 1: +0.129 [+0.024, +0.269]
- repeat 2: +0.069 [-0.020, +0.211]

## Memorization-adjusted recall: closed pool (8 reviews)

25 of 341 gold papers (7.3%) were named from memory by at least one model and are removed; recall is recomputed over the remaining gold.

| system | mode | B | recall | adjusted | change |
|---|---|---|---|---|---|
| bm25 | keyword | 40 | 0.524 | 0.526 | +0.002 |
| haiku-5.5 | citation | 40 | 0.139 | 0.143 | +0.004 |
| haiku-5.5 | hybrid | 40 | 0.626 | 0.624 | -0.002 |
| haiku-5.5 | keyword | 40 | 0.581 | 0.575 | -0.005 |
| qwen3-1.7b-grpo | citation | 40 | 0.133 | 0.140 | +0.007 |
| qwen3-1.7b-grpo | hybrid | 40 | 0.467 | 0.456 | -0.011 |
| qwen3-1.7b-grpo | keyword | 40 | 0.517 | 0.510 | -0.007 |
| qwen3-1.7b-sft | citation | 40 | 0.062 | 0.068 | +0.006 |
| qwen3-1.7b-sft | hybrid | 40 | 0.433 | 0.412 | -0.021 |
| qwen3-1.7b-sft | keyword | 40 | 0.478 | 0.466 | -0.011 |
| qwen3-32b | citation | 40 | 0.047 | 0.049 | +0.002 |
| qwen3-32b | hybrid | 40 | 0.310 | 0.303 | -0.007 |
| qwen3-32b | keyword | 40 | 0.203 | 0.207 | +0.004 |
| random | hybrid | 40 | 0.068 | 0.058 | -0.010 |
| snowball | citation | 40 | 0.076 | 0.076 | -0.001 |
| snowball | hybrid | 40 | 0.139 | 0.138 | -0.001 |
| snowball | keyword | 40 | 0.105 | 0.121 | +0.015 |
| snowball-specter2 | hybrid | 40 | 0.130 | 0.120 | -0.011 |
| sonnet-5.5 | citation | 40 | 0.340 | 0.320 | -0.019 |
| sonnet-5.5 | hybrid | 40 | 0.666 | 0.656 | -0.010 |
| sonnet-5.5 | keyword | 40 | 0.585 | 0.579 | -0.007 |
| specter2 | keyword | 40 | 0.635 | 0.644 | +0.009 |

Paired differences at B=40 (bootstrap over reviews), original vs memorization-adjusted:

- sonnet-5.5 hybrid − bm25 keyword: +0.142 [+0.051, +0.242] → +0.129 [+0.049, +0.215] (n=8)
- sonnet-5.5 hybrid − specter2 keyword: +0.031 [-0.051, +0.113] → +0.012 [-0.082, +0.109] (n=8)
- sonnet-5.5 hybrid − haiku-5.5 hybrid: +0.040 [-0.040, +0.128] → +0.031 [-0.045, +0.115] (n=8)
- haiku-5.5 hybrid − bm25 keyword: +0.102 [-0.057, +0.252] → +0.098 [-0.046, +0.230] (n=8)
- qwen3-1.7b-grpo hybrid − qwen3-32b hybrid: +0.156 [-0.027, +0.366] → +0.152 [-0.033, +0.363] (n=8)

## Memorization-adjusted recall: transfer (24 reviews)

87 of 2233 gold papers (3.9%) were named from memory by at least one model and are removed; recall is recomputed over the remaining gold.

| system | mode | B | recall | adjusted | change |
|---|---|---|---|---|---|
| bm25 | keyword | 40 | 0.474 | 0.466 | -0.008 |
| haiku-5.5 | hybrid | 40 | 0.477 | 0.459 | -0.017 |
| qwen3-1.7b | hybrid | 40 | 0.112 | 0.109 | -0.002 |
| qwen3-1.7b-grpo | hybrid | 40 | 0.483 | 0.482 | -0.001 |
| qwen3-1.7b-sft | hybrid | 40 | 0.459 | 0.451 | -0.008 |
| qwen3-32b | hybrid | 40 | 0.351 | 0.349 | -0.002 |
| snowball-tuned | hybrid | 40 | 0.095 | 0.089 | -0.006 |
| sonnet-5.5 | hybrid | 40 | 0.561 | 0.547 | -0.013 |

Paired differences at B=40 (bootstrap over reviews), original vs memorization-adjusted:

- sonnet-5.5 hybrid − bm25 keyword: +0.087 [+0.013, +0.158] → +0.081 [+0.006, +0.154] (n=24)
- sonnet-5.5 hybrid − haiku-5.5 hybrid: +0.084 [+0.045, +0.128] → +0.088 [+0.048, +0.133] (n=24)
- haiku-5.5 hybrid − bm25 keyword: +0.003 [-0.081, +0.078] → -0.006 [-0.091, +0.071] (n=24)
- qwen3-1.7b-grpo hybrid − qwen3-32b hybrid: +0.133 [+0.078, +0.191] → +0.134 [+0.079, +0.193] (n=24)

