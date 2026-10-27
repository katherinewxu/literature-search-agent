# Results (living document)

All numbers are means over the 8 frozen SYNERGY+ reviews (`data/synergy-v3`; train
Meijboom/Leenaars/Brouwer/Bos, dev Hall/Donners, test Muthu/Appenzeller-Herzog), one
episode per review × mode × budget, environment r3 semantics (free screening after the
budget is spent; per-review BM25 index). LLM runs are r4 (`runs/synergy-v3-api-r4`,
`runs/synergy-v3-open-api-r4`). Paired differences use a bootstrap over the 8 reviews.
Full tables: `analysis/closed-r4/analysis.md`, `analysis/open-r4/analysis.md`.

Because 8 reviews is small, every claim below is stated with its review-level paired CI
and win/loss count; per-split (test-only) numbers are reported separately in the final
write-up. Prompts, thresholds and budgets were fixed before looking at dev/test results;
the only tuned baseline parameter (snowball include threshold, 0.16 -> 0.04) was chosen
on the 4 train reviews only (`runs/snowball-tune-train`).

## Systems

| label | what it is |
|---|---|
| bm25 | ranked BM25 over the review title + criteria, includes everything it sees (recall ceiling for lexical search, precision floor) |
| snowball / snowball-tuned | scripted: search (hybrid/keyword), read frontier papers, lexical screening, chase refs+citers of included papers |
| random | unscreened random traversal |
| sonnet-5.5, haiku-5.5, qwen3-32b | prompted LLM agent (`litsearch.llm`), temperature 0, one call per turn (screen + one charged action), via OpenRouter |

Thinking policy: haiku reasoning disabled; qwen3-32b `/no_think`; sonnet-5.5 has
mandatory short reasoning on OpenRouter (~130 tokens/turn).

## Q1. Prompted LLM vs. scripted snowballing (closed graph)

Mean recall (F1) by budget:

| system | B=10 | B=20 | B=40 |
|---|---|---|---|
| sonnet-5.5 / hybrid | .417 (.43) | .488 (.42) | .666 (.49) |
| sonnet-5.5 / keyword | .400 (.40) | .518 (.45) | .585 (.46) |
| sonnet-5.5 / citation | .266 (.34) | .264 (.32) | .340 (.40) |
| haiku-5.5 / hybrid | .273 (.28) | .432 (.32) | .626 (.39) |
| qwen3-32b / hybrid | .102 (.14) | .219 (.24) | .310 (.21) |
| bm25 / keyword | .241 (.10) | .359 (.10) | .524 (.09) |
| snowball-tuned / hybrid | .023 (.03) | .089 (.10) | .173 (.16) |
| random / hybrid | .012 (.02) | .019 (.03) | .041 (.05) |

Paired (sonnet-5.5/hybrid minus reference):

| reference | B | Δrecall [95% CI] | ΔF1 [95% CI] | wins/losses |
|---|---|---|---|---|
| snowball/hybrid | 10 | +.394 [+.260, +.538] | +.398 | 8/0 |
| snowball/hybrid | 40 | +.527 [+.375, +.684] | +.360 | 8/0 |
| bm25/keyword | 10 | +.177 [+.087, +.268] | +.328 | 7/1 |
| bm25/keyword | 40 | +.142 [+.053, +.243] | +.398 [+.281, +.505] | 5/2 |

**Answer.** A prompted frontier model is far better than scripted snowballing at every
budget (8/8 reviews) and beats exhaustive BM25 inclusion in recall while being ~8x more
precise. This is not an exposure artifact: at B=40 sonnet/hybrid is *shown* 336 papers vs.
BM25's 377, yet 77% of gold is among them (BM25: 52%), and it includes 64 papers vs. 377.
The cheap haiku-5.5 recovers most of the gain (.626 at B=40); the open qwen3-32b does not.

## Grader question: is the citation graph limiting?

Two pieces of evidence (`analysis/*/analysis.md`, reachability + open-graph runs):

1. **Reachability is not the bottleneck.** From the 5 BM25 seeds, 77–100% of gold is within
   3 undirected citation hops inside the closed pool (Hall 82%, Muthu 77% are the hardest);
   in the open graph 89–100%. Within 2 hops the open graph helps substantially (Muthu
   26% -> 60%, Hall 28% -> 38%, Bos 70% -> 100%).
2. **Opening the graph lowers recall anyway.** Adding up to 5,000 out-of-pool cited works per
   review (universe roughly doubles) reduces every system's recall: bm25 .524 -> .396,
   sonnet/citation .340 -> .262, sonnet/hybrid .666 -> .619, haiku/hybrid .626 -> .499,
   snowball-tuned/hybrid .173 -> .067. Citation-only chasing suffers most; search +
   chasing (hybrid) is the most robust.

**Answer.** Within a 40-call budget the agents are limited by *budget and distractors*, not by
graph connectivity: more graph means more paths to gold but more off-topic neighbors to
pay for. A more general tool (search over a larger universe) helps only when paired with
search; citation access alone is the weakest tool in every condition. The open graph is a
bounded one-hop backward expansion (not forward citers, not full-web search), so the claim
is about this regime; true web search is left as future work (see limitations).

## Q3. Emergent strategies (prompted agents; RL agent pending)

Share of charged budget at B=40, closed graph:

| system | search | expand refs | expand citers | read | unique queries/episode |
|---|---|---|---|---|---|
| sonnet-5.5 / hybrid | .89 | .00 | .01 | .08 | 32.4 |
| haiku-5.5 / hybrid | .80 | .01 | .11 | .05 | 25.5 |
| qwen3-32b / hybrid | .81 | .00 | .00 | .15 | 3.6 |
| sonnet-5.5 / citation | – | .20 | .67 | .08 | – |

Given both tools, frontier models behave as **query reformulators**, not snowballers: they
spend ~90% of the budget on new searches (one new query per call) and almost never chase
citations. In citation mode they prefer forward citations (citers) to references 3:1.
Qwen3-32B issues few distinct queries, reads more, and stops with budget left (premature
stopping; see failures).

## Failure analysis (B=40, closed graph; counts of missed gold over episodes)

| system | exposed but not included | reachable but not reached | unreachable in 4 hops & BM25 rank>100 | missing abstract |
|---|---|---|---|---|
| sonnet-5.5 / hybrid | 35 | 90 | 9 | 34 |
| sonnet-5.5 / keyword | 45 | 100 | 10 | 35 |
| sonnet-5.5 / citation | 9 | 222 | 24 | 43 |
| bm25 / keyword | 0 | 167 | 15 | 43 |

- Most misses are gold that was reachable but not reached within the budget: a search
  coverage failure, not a graph failure.
- Screening errors are the second source: Sonnet saw 77% of gold but kept 67% (title-only
  screening; ~25% of gold lacks an OpenAlex abstract).
- Weak models fail differently: qwen3-32b stops early with unused budget (B=40 citation:
  36 of 40 units unused on average) and haiku-5.5 in citation mode also under-spends.
- Hall_2011 (software engineering) and Muthu_2020 (spine surgery) have sparse gold citation
  neighborhoods (mean gold degree ~9 vs. 13–43 elsewhere) and are the hardest reviews for
  citation-only agents.

## Pending (filled in as runs complete)

- Q2: small open model (Qwen3-1.7B) prompted, SFT on rejection-sampled haiku teacher
  episodes from 40 disjoint life-science reviews, then GRPO; vs. the large prompted models.
- Q4: cross-field transfer on 24 held-out non-life-science reviews (`ext-transfer-api-r4`).
- Dense retrieval (SPECTER2) baselines.

## Limitations

- 8 evaluation reviews (2 test); CIs are wide and per-review heterogeneity is large.
- Search cutoff = review publication date (SYNERGY has no search dates); this can only add
  non-gold distractors, never hide gold.
- Candidate pools are the reviews' own search results, so the closed graph is easier than the
  open literature; the open-graph condition partially relaxes this.
- ~27% of papers have no abstract in OpenAlex; title-only screening is a real constraint.
- Frontier models may have seen these published reviews in pretraining; queries and inclusions
  could partly reflect memorization. The leakage filter removes the review and its citers from
  the environment but cannot remove model knowledge.
- Sonnet's mandatory reasoning differs from the other models' non-thinking mode.
