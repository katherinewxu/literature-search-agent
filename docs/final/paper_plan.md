# Paper plan: thesis and pre-specified primary comparisons (written 2026-11-18)

Written **before** the no-loop baseline (`configs/web/*_static.json`) and the live-web runs of the
trained model (`configs/web/*_trained.json`) were run, so the primary tests are fixed in advance.
Edit the thesis wording freely; do not change the primary list after seeing those results.

## Candidate thesis (one sentence; pick or rewrite)
In budgeted literature search, LLM agents fail by not reaching eligible studies, not by
rejecting them: screening loses at most 5% of gold, agents act mostly as query reformulators,
and a 1.7B model trained on a frontier model's trajectories recovers most of that model's recall.

## Primary comparisons (Holm-corrected as one family of 4; everything else exploratory)
Paired over reviews, recall at B=40, hybrid unless stated; bootstrap 95% CI + exact sign test.

| # | Comparison | Split | Why it is primary |
|---|---|---|---|
| P1 | LLM agent (Haiku) vs no-loop LLM baseline (Haiku writes 8 queries once, title screening) | live web, transfer (24) | Is the agent loop needed at all? |
| P2 | Trained Qwen3-1.7B (SFT+GRPO v1) vs prompted Qwen3-32B | closed graph, transfer (24) | Headline training claim (already run) |
| P3 | Haiku hybrid vs best non-LLM baseline (BM25 include-all) | closed graph, transfer (24) | Is an LLM agent better than IR? (already run) |
| P4 | v2 environment vs original, Haiku hybrid | live web, transfer (24, held out) | The diagnosed environment flaw |

Secondary (reported with CIs, not corrected, labelled exploratory): every eval-set (n=8)
comparison, Sonnet results, citation-only mode, budget curves, judge-based "beyond search"
estimates, trained model on the live web (vs Haiku and vs the no-loop baseline).

## How each result will be read
- P1: if the loop's CI excludes zero in its favour, "the loop matters"; if not, report that a
  static LLM query list matches the agent at this budget and move the agent's value to
  adaptivity in citation mode / cost. Also report the include-all variant as the retrieval
  ceiling of the static queries.
- Trained model on web: compared with Haiku hybrid on the same reviews. Report as transfer of the
  closed-graph training to the live web, whatever the sign.

## Results so far (2026-11-19; transfer 21-22/24 reviews, Sonnet no-loop and trained-on-web pending)
**P1 came out against the agent loop.** Haiku, B=40, paired over reviews:

| split | agent recall | no-loop recall | agent - no-loop | up/down | F1 agent vs no-loop |
|---|---|---|---|---|---|
| transfer (21) | .098 | .176 | -.086 [-.121, -.055] | 0/18, p<.001 | .087 vs .088 (equal) |
| eval (8) | .171 | .368 | -.198 [-.310, -.103] | 0/8 | .137 vs .160 (CI spans 0) |

Why: the no-loop baseline spends the whole budget on search (40 of 40) and includes ~4x more
papers (eval 106 vs 28); the agent leaves about half its budget unspent and screens more strictly
(precision .148 vs .108 on eval). Same F1, different operating point: the loop buys precision,
not recall. Include-all ceiling of the static queries: .364 eval / .196 transfer, so title
screening loses little. No-loop Haiku also costs less ($0.004 vs $0.007 per episode).
Sonnet hybrid agent (.324 eval) is still below no-loop Haiku (.368) on recall.

**Update 2026-11-21: P1 on all 24 transfer reviews:** agent - no-loop = -.083 [-.116, -.054], 0/21,
p<.001. The first Sonnet no-loop runs were invalid: they hit three `Stale file handle` errors on
node rye-01 and stopped after about 3 actions. They were deleted and are being rerun.

**Closed graph (exploratory, `scripts/closed_static_compare.py`; diff = no-loop - agent, Haiku B=40):**

| split | recall | precision | F1 | included | budget spent (agent / no-loop) |
|---|---|---|---|---|---|
| transfer (24) | +.112 [+.076, +.149], 22/2 | -.144, 0/24 | **-.081 [-.130, -.039], 3/21** | 88 vs 166 | 30.0 / 40 |
| eval (8) | +.025 [-.023, +.075] | -.101, 0/8 | -.100, 0/8 | 75 vs 131 | 32.2 / 40 |

The same pattern holds on both graphs: the loop trades recall for precision and stops early. On
the closed graph that trade *wins* on F1 (transfer .390 vs .309); on the live web F1 is equal.
Revised reading: the loop's value is a precision/stopping policy, not coverage. The bigger the
candidate space (live web), the more the agent's early stopping costs it.

## Final primary results (2026-11-22; `analysis/webfp/primary_tests.md`, `scripts/primary_tests.py`)
Recall at B=40, 24 held-out transfer reviews, paired; exact sign test, Holm over P1-P4.

| id | comparison (B - A) | A | B | diff [95% CI] | up/down | Holm p |
|---|---|---|---|---|---|---|
| P1 | live web: Haiku agent loop - no-loop Haiku query list | .181 | .098 | -.083 [-.115, -.054] | 0/21 | <.001 |
| P2 | closed: trained Qwen3-1.7B - prompted Qwen3-32B | .351 | .483 | +.133 [+.078, +.193] | 18/3 | .005 |
| P3 | closed: Haiku agent - BM25 | .474 | .477 | +.003 [-.082, +.080] | 14/10 | .54 |
| P4 | live web: v2 env - original, Haiku agent | .098 | .134 | +.037 [+.007, +.070] | 13/7 | .53 |

P1 and P2 survive correction. P3 is a null: the Haiku agent does not beat BM25 on recall in the
closed graph. P4's bootstrap CI excludes 0, but the sign test does not survive correction, so
report it as suggestive only.

Exploratory, completed 2026-11-22:
- Sonnet, eval B=40: agent .324 vs no-loop .351, diff -.027 [-.090, +.049], 2/6. Same direction
  as Haiku but not significant. Sonnet no-loop costs $0.145 per episode.
- Trained Qwen3-1.7B (GRPO) on the live web, transfer B=40: .097, which ties the Haiku agent (.098;
  -.001 [-.029, +.028], 9/10) without any web training. GRPO vs SFT: +.009 [-.005, +.022].
  On eval it is below Haiku (-.088 [-.164, -.015], 3/5). It is below the no-loop baseline
  (-.084, 2/21).
- OpenRouter spend is $60.14 (the cap was raised to $61 for the last two Sonnet runs).
