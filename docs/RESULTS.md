# Results (living document)

All numbers are means over the 8 frozen SYNERGY+ reviews (`data/synergy-v3`; train
Meijboom/Leenaars/Brouwer/Bos, dev Hall/Donners, test Muthu/Appenzeller-Herzog), one
episode per review × mode × budget, environment r3 semantics (free screening after the
budget is spent; per-review BM25 index). LLM runs are r4 (`runs/synergy-v3-api-r4`,
`runs/synergy-v3-open-api-r4`). Paired differences use a bootstrap over the 8 reviews.
Full tables: `analysis/closed-r5/analysis.md`, `analysis/open-r5/analysis.md`,
`analysis/transfer-r5/analysis.md` (r5 = r4 plus dense, small-model and transfer runs).

Because 8 reviews is small, every claim below is stated with its review-level paired CI
and win/loss count; per-split (test-only) numbers are reported separately in the final
write-up. Prompts, thresholds and budgets were fixed before looking at dev/test results;
the only tuned baseline parameter (snowball include threshold, 0.16 -> 0.04) was chosen
on the 4 train reviews only (`runs/snowball-tune-train`).

## Systems

| label | what it is |
|---|---|
| bm25 | ranked BM25 over the review title + criteria, includes everything it sees (recall ceiling for lexical search, precision floor) |
| specter2 | same, but ranked by SPECTER2 dense retrieval (ad-hoc query adapter vs. proximity document embeddings) |
| snowball-specter2 | snowball with SPECTER2 search |
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
| specter2 / keyword | .289 (.11) | .469 (.13) | .635 (.12) |
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

**Caveat: a strong dense retriever closes the recall gap.** Exhaustive SPECTER2 inclusion
reaches .635 at B=40. Sonnet/hybrid minus specter2: +.129 [-.012, +.287] (B=10, 6/2),
+.019 [-.072, +.123] (B=20, 3/5), +.031 [-.046, +.116] (B=40, 4/2); haiku/hybrid is within
±.04 at every budget. So with the same call budget the prompted agent's *recall* advantage
is over lexical search, not over good dense search; its robust advantage is precision
(.415 vs. .064 at B=40, i.e. a reviewer screens 64 papers instead of 378). On the open
graph, where distractors double, the agent's recall lead over SPECTER2 grows
(+.141 [+.007, +.290] at B=20; +.091 [-.017, +.218] at B=40) because dense retrieval
degrades (.635 -> .528) more than sonnet/hybrid (.666 -> .619). Dense search inside the
snowball baseline barely helps (.130 vs. .173 tuned lexical at B=40): the scripted policy,
not the retriever, is its bottleneck.

### Test split only (Muthu_2020, Appenzeller-Herzog_2019; recall, B=10/20/40)

| system | B=10 | B=20 | B=40 |
|---|---|---|---|
| sonnet-5.5 / hybrid | .177 | .079 | .331 |
| sonnet-5.5 / citation | .135 | .038 | .154 |
| haiku-5.5 / hybrid | .053 | .110 | .331 |
| qwen3-32b / hybrid | .000 | .058 | .115 |
| specter2 / keyword | .067 | .139 | .336 |
| bm25 / keyword | .029 | .069 | .261 |
| snowball-tuned / hybrid | .000 | .014 | .071 |

The two test reviews are the hardest in the set (sparse gold neighborhoods, many gold
without abstracts); absolute recall is low for every system and n=2 supports no ranking
claim beyond "snowball < everything else". B=20 below B=10 for sonnet is one episode per
cell at temperature 0, i.e. trajectory variance, not a budget effect.

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
- **Citation mode induces premature stopping in every model.** Mean unused budget at B=40,
  citation / keyword / hybrid: sonnet 20.4 / 2.0 / 2.5; haiku 32.4 / 14.4 / 7.8; qwen3-32b
  36.2 / 27.1 / 12.6 (r4, after charging disallowed tools instead of ending the episode).
  With only expansion available, agents run out of papers they consider worth chasing and
  stop; with search available, strong models keep reformulating. This is a second way the
  citation-only tool set is limiting, separate from reachability.
- Weaker models stop early in every mode (qwen3-32b leaves a third of the hybrid budget).
- Hall_2011 (software engineering) and Muthu_2020 (spine surgery) have sparse gold citation
  neighborhoods (mean gold degree ~9 vs. 13–43 elsewhere) and are the hardest reviews for
  citation-only agents.

## Q4. Cross-field transfer (24 non-life-science SYNERGY+ reviews, closed graph)

`data/synergy-ext-transfer-v2`: 24 held-out reviews (software engineering, education,
psychology, social science, environment...) disjoint from both the 8 evaluation reviews
and the 40 SFT-teacher reviews. Prompts unchanged.

| system | B=20 recall | B=40 recall | B=40 precision | Δrecall vs bm25 at B=40 [95% CI] (wins/losses) |
|---|---|---|---|---|
| sonnet-5.5 / hybrid | – | .561 | .441 | +.087 [+.016, +.158] (15/9) |
| haiku-5.5 / hybrid | .370 | .477 | .390 | +.003 [-.078, +.077] (14/10) |
| qwen3-32b / hybrid | .271 | .351 | .302 | -.123 [-.212, -.040] (9/15) |
| bm25 / keyword | .321 | .474 | .071 | – |
| snowball-tuned / hybrid | .057 | .095 | .163 | -.379 (0/24) |

**Answer.** The qualitative picture transfers: frontier agents keep a large precision /
F1 advantage (ΔF1 vs. BM25 +.28 to +.34, CIs exclude 0), sonnet keeps a smaller but
significant recall lead, and snowballing loses on all 24 reviews. The recall margin shrinks
relative to life science (sonnet +.142 -> +.087; haiku +.10 -> +.00), partly because many
transfer reviews have very sparse citation neighborhoods (Kapuka 0% of gold within 4 hops,
Attai 34%, Harms 25%) and about half of missed gold lacks an abstract (sonnet 450/963, bm25 548/1117). Agents also stop earlier
here (haiku leaves 25% of B=40 unused; 42% of episodes stop with >=25% left).

## Dense retrieval (SPECTER2)

See the Q1 caveat. SPECTER2 is the strongest non-LLM baseline in every condition
(closed .635, open .528 at B=40), beats BM25 by +.110 [+.026, +.217] at B=20 (5/1), and
misses fewer abstract-less gold (24 vs. 43 missed at B=40), consistent with title-only
embeddings carrying more signal than title-only BM25. It does not help the scripted snowball.
An LLM agent with a dense search tool was not run (budget); it is the obvious next system.

## Pending

- Q2: small open model (Qwen3-1.7B) prompted (done, below), SFT on rejection-sampled haiku
  teacher episodes from 40 disjoint life-science reviews, then GRPO; vs. the large prompted
  models. Training is running (SFT job 1917322 -> GRPO 1917323 -> eval 1917365).

Prompted Qwen3-1.7B (closed, B=40): hybrid .125, keyword .111, citation .012. It *exposes*
a lot of gold (search recall .369 hybrid, .476 keyword) but screens badly (screen recall
~.21) and in citation mode wastes the budget on disallowed tools (stops after ~5 papers).
The small model's failure is screening and protocol-following, not search.

## Reproducibility

- Fresh-environment test (2026-10-31): clean Python 3.12 venv from a source copy,
  `pip install -e .`, README quickstart end to end (fixture build -> audit -> run ->
  replay), `python -m unittest discover -s tests`: 42 tests OK.
- Replays: `demo/muthu_sonnet_hybrid_b40.html` and `demo/muthu_sonnet_citation_b40.html`
  (same test review, same model, the two tool sets) are self-contained HTML.

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
