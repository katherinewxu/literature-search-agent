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
