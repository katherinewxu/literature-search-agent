# End-of-episode recall estimation

| set | agent | n | true recall | estimator | mean estimate | MAE | corr with true | lower bound holds |
|---|---|---|---|---|---|---|---|---|
| life-science 8 | haiku-5.5 | 8 | 0.626 | yield curve | 0.398 | 0.282 | 0.48 | – |
| life-science 8 | haiku-5.5 | 8 | 0.626 | sample, gold screener | 0.613 | 0.201 | 0.66 | 100% |
| life-science 8 | haiku-5.5 | 8 | 0.626 | sample, Haiku screener | 0.208 | 0.418 | 0.49 | 100% |
| life-science 8 | sonnet-5.5 | 8 | 0.666 | yield curve | 0.751 | 0.247 | 0.47 | – |
| life-science 8 | sonnet-5.5 | 8 | 0.666 | sample, gold screener | 0.807 | 0.208 | 0.74 | 100% |
| life-science 8 | sonnet-5.5 | 8 | 0.666 | sample, Haiku screener | 0.173 | 0.493 | 0.78 | 100% |
| life-science 40 | haiku-5.5 | 40 | 0.493 | yield curve | 0.375 | 0.281 | 0.21 | – |
| life-science 40 | haiku-5.5 | 40 | 0.493 | sample, gold screener | 0.620 | 0.238 | 0.58 | 100% |
| life-science 40 | haiku-5.5 | 40 | 0.493 | sample, Haiku screener | 0.210 | 0.287 | 0.64 | 100% |
| life-science 40 | sonnet-5.5 | 40 | 0.558 | yield curve | 0.642 | 0.246 | 0.28 | – |
| life-science 40 | sonnet-5.5 | 40 | 0.558 | sample, gold screener | 0.666 | 0.203 | 0.42 | 100% |
| life-science 40 | sonnet-5.5 | 40 | 0.558 | sample, Haiku screener | 0.202 | 0.360 | 0.49 | 100% |
| transfer 24 | haiku-5.5 | 24 | 0.477 | yield curve | 0.452 | 0.229 | 0.56 | – |
| transfer 24 | haiku-5.5 | 24 | 0.477 | sample, gold screener | 0.615 | 0.227 | 0.42 | 100% |
| transfer 24 | haiku-5.5 | 24 | 0.477 | sample, Haiku screener | 0.235 | 0.250 | 0.72 | 100% |
| transfer 24 | sonnet-5.5 | 24 | 0.561 | yield curve | 0.747 | 0.269 | 0.35 | – |
| transfer 24 | sonnet-5.5 | 24 | 0.561 | sample, gold screener | 0.663 | 0.192 | 0.61 | 100% |
| transfer 24 | sonnet-5.5 | 24 | 0.561 | sample, Haiku screener | 0.257 | 0.304 | 0.80 | 100% |
| life-science 8 | qwen3-1.7b-grpo | 8 | 0.467 | yield curve | 0.363 | 0.200 | 0.84 | – |
| life-science 8 | qwen3-1.7b-grpo | 7 | 0.533 | sample, gold screener | 0.731 | 0.247 | 0.81 | 100% |
| life-science 8 | qwen3-1.7b-grpo | 8 | 0.467 | sample, Haiku screener | 0.248 | 0.250 | 0.51 | 88% |
| transfer 24 | qwen3-1.7b-grpo | 24 | 0.483 | yield curve | 0.398 | 0.186 | 0.64 | – |
| transfer 24 | qwen3-1.7b-grpo | 24 | 0.483 | sample, gold screener | 0.529 | 0.112 | 0.86 | 100% |
| transfer 24 | qwen3-1.7b-grpo | 24 | 0.483 | sample, Haiku screener | 0.313 | 0.171 | 0.87 | 100% |

Screener usage: {"calls": 0, "cache_hits": 3452, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
