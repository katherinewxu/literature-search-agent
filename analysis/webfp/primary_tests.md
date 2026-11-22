# Primary comparisons (pre-specified; Holm over P1-P4)

Recall at B=40, held-out transfer split, paired over reviews; diff = B - A.

| id | comparison (B - A) | n | A | B | diff | 95% CI | up/down | sign p | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| P1 | live web: Haiku agent loop - no-loop Haiku query list | 24 | 0.181 | 0.098 | -0.083 | [-0.115, -0.054] | 0/21 | 0.0000 | 0.0000 |
| P2 | closed graph: trained Qwen3-1.7B (SFT+GRPO) - prompted Qwen3-32B | 24 | 0.351 | 0.483 | +0.133 | [+0.078, +0.193] | 18/3 | 0.0015 | 0.0045 |
| P3 | closed graph: Haiku agent - BM25 baseline | 24 | 0.474 | 0.477 | +0.003 | [-0.082, +0.080] | 14/10 | 0.5413 | 0.5413 |
| P4 | live web: v2 environment - original, Haiku agent | 24 | 0.098 | 0.134 | +0.037 | [+0.007, +0.070] | 13/7 | 0.2632 | 0.5264 |
