# Calibrated recall estimation

Screener rates fit on the 80 calibration episodes (life-science 40): TPR 0.92; FPR included 0.870, excluded 0.433, unexposed 0.185.

| set | agent | n | true recall | estimator | mean estimate | MAE | corr | lower bound holds | mean lower bound |
|---|---|---|---|---|---|---|---|---|---|
| life-science 8 | haiku-5.5 | 8 | 0.626 | sample, gold screener | 0.613 | 0.201 | 0.66 | 100% | 0.119 |
| life-science 8 | haiku-5.5 | 8 | 0.626 | sample, Haiku raw | 0.208 | 0.418 | 0.49 | 100% | 0.120 |
| life-science 8 | haiku-5.5 | 6 | 0.543 | sample, Haiku calibrated | 0.607 | 0.233 | 0.84 | 83% | 0.396 |
| life-science 8 | haiku-5.5 | 8 | 0.626 | yield curve | 0.398 | 0.282 | 0.48 | – | – |
| life-science 8 | qwen3-1.7b-grpo | 7 | 0.533 | sample, gold screener | 0.731 | 0.247 | 0.81 | 100% | 0.141 |
| life-science 8 | qwen3-1.7b-grpo | 8 | 0.467 | sample, Haiku raw | 0.248 | 0.250 | 0.51 | 88% | 0.156 |
| life-science 8 | qwen3-1.7b-grpo | 1 | 0.143 | sample, Haiku calibrated | 0.234 | 0.091 | nan | 0% | 0.155 |
| life-science 8 | qwen3-1.7b-grpo | 8 | 0.467 | yield curve | 0.363 | 0.200 | 0.84 | – | – |
| life-science 8 | sonnet-5.5 | 8 | 0.666 | sample, gold screener | 0.807 | 0.208 | 0.74 | 100% | 0.144 |
| life-science 8 | sonnet-5.5 | 8 | 0.666 | sample, Haiku raw | 0.173 | 0.493 | 0.78 | 100% | 0.098 |
| life-science 8 | sonnet-5.5 | 7 | 0.618 | sample, Haiku calibrated | 0.725 | 0.275 | 0.66 | 43% | 0.636 |
| life-science 8 | sonnet-5.5 | 8 | 0.666 | yield curve | 0.751 | 0.247 | 0.47 | – | – |
| transfer 24 | haiku-5.5 | 24 | 0.477 | sample, gold screener | 0.615 | 0.227 | 0.42 | 100% | 0.164 |
| transfer 24 | haiku-5.5 | 24 | 0.477 | sample, Haiku raw | 0.235 | 0.250 | 0.72 | 100% | 0.149 |
| transfer 24 | haiku-5.5 | 20 | 0.453 | sample, Haiku calibrated | 0.490 | 0.347 | 0.23 | 90% | 0.254 |
| transfer 24 | haiku-5.5 | 24 | 0.477 | yield curve | 0.452 | 0.229 | 0.56 | – | – |
| transfer 24 | qwen3-1.7b-grpo | 24 | 0.483 | sample, gold screener | 0.529 | 0.112 | 0.86 | 100% | 0.172 |
| transfer 24 | qwen3-1.7b-grpo | 24 | 0.483 | sample, Haiku raw | 0.313 | 0.171 | 0.87 | 100% | 0.209 |
| transfer 24 | qwen3-1.7b-grpo | 9 | 0.378 | sample, Haiku calibrated | 0.217 | 0.328 | 0.25 | 100% | 0.090 |
| transfer 24 | qwen3-1.7b-grpo | 24 | 0.483 | yield curve | 0.398 | 0.186 | 0.64 | – | – |
| transfer 24 | sonnet-5.5 | 24 | 0.561 | sample, gold screener | 0.663 | 0.192 | 0.61 | 100% | 0.183 |
| transfer 24 | sonnet-5.5 | 24 | 0.561 | sample, Haiku raw | 0.257 | 0.304 | 0.80 | 100% | 0.158 |
| transfer 24 | sonnet-5.5 | 19 | 0.513 | sample, Haiku calibrated | 0.744 | 0.279 | 0.65 | 68% | 0.460 |
| transfer 24 | sonnet-5.5 | 24 | 0.561 | yield curve | 0.747 | 0.269 | 0.35 | – | – |

Pooled over all evaluation episodes (8 + 24 reviews, all agents):

- sample, gold screener: MAE 0.187, corr 0.66, mean estimate 0.630 vs true 0.532 (n=95)
- sample, Haiku raw: MAE 0.278, corr 0.66, mean estimate 0.254 vs true 0.527 (n=96)
- sample, Haiku calibrated: MAE 0.300, corr 0.51, mean estimate 0.562 vs true 0.483 (n=62)
- yield curve: MAE 0.232, corr 0.51, mean estimate 0.525 vs true 0.527 (n=96)
