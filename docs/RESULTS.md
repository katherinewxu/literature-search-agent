# Results (living document)

All numbers are means over the 8 frozen SYNERGY+ reviews (`data/synergy-v3`; train
Meijboom/Leenaars/Brouwer/Bos, dev Hall/Donners, test Muthu/Appenzeller-Herzog), one
episode per review × mode × budget, environment r3 semantics (free screening after the
budget is spent; per-review BM25 index). LLM runs are r4 (`runs/synergy-v3-api-r4`,
`runs/synergy-v3-open-api-r4`). Paired differences use a bootstrap over the 8 reviews.
Full tables: `analysis/closed-r5/analysis.md`, `analysis/open-r5/analysis.md`,
`analysis/transfer-r5/analysis.md`, `analysis/small-{closed,open,transfer}-r5/analysis.md` (r5 = r4 plus dense, small-model and transfer runs).

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
bounded one-hop backward expansion; the live-web condition below tests full-web search directly.

### Live web search (all of OpenAlex, ~250M works)

`runs/synergy-v3-web-r5` (104 episodes, 8 reviews; `litsearch.web`): the same agents, but
`search` is OpenAlex's live relevance search and `expand_refs` / `expand_citers` return a
work's full reference list / most-cited citers, all restricted to the review's cutoff date,
with the review and its citers hidden. Found records are mapped back to the frozen store by
DOI or title, so a gold paper found under another OpenAlex record still counts. Recall (paired
difference web − closed pool, 95% CI):

| system | B=10 | B=20 | B=40 | closed B=40 | open B=40 | web − closed, B=40 |
|---|---|---|---|---|---|---|
| sonnet-5.5 / hybrid | .215 | .295 | .324 | .666 | .619 | −.342 [−.476, −.217] |
| haiku-5.5 / hybrid | .141 | .165 | .171 | .626 | .499 | −.455 [−.616, −.298] |
| sonnet-5.5 / citation | – | – | .148 | .340 | .262 | −.192 [−.278, −.100] |
| haiku-5.5 / citation | .050 | .087 | .108 | .139 | .037 | −.031 [−.186, +.157] |
| OpenAlex search / BM25, include all | .010 | .010 | .010 | .524 | .396 | −.514 [−.718, −.312] |

**Answer (literal).** Internet-scale search does not help a 40-call agent; it halves recall.
Two things change with the universe size:
1. *The value of the LLM grows.* Sonnet's lead over include-everything retrieval is +.14 on the
   closed pool, +.22 on the open graph and +.31 on the web, where raw OpenAlex relevance search
   finds almost nothing (1% recall). Sonnet's lead over Haiku also widens (+.04 closed → +.15 web): writing
   good queries matters more when the index is large.
2. *Citation chasing degrades least.* Citation-only agents lose little or nothing on the web
   (Haiku .139 → .108, n.s.), because a paper's references and citers are the same set whatever
   the universe; search-based agents lose the most.
Closed-pool benchmarks (including ours) therefore overstate real-world recall by about 2×.

The transfer set replicates this (`runs/ext-transfer-web-r5`, 24 reviews, 96 episodes): Haiku
hybrid falls from .477 to .098 at B=40 (−.379 [−.465, −.301]) and from .370 to .084 at B=20;
raw OpenAlex search gets .018 at both budgets. Haiku − OpenAlex search on the web is
+.080 [+.056, +.105].

### Forcing citation chasing (does the agent ignore a useful tool?)

PaSa (He et al., 2025) loses 23–32 recall points without citation expansion, but our agents
almost never expand. To test whether they are wrong, `litsearch.forced.ForcedExpansionAgent`
replaces every third charged action with expansion (alternating citers/refs) of the most recent
paper the agent itself included (`runs/*-forced-r5`). Paired difference forced − free, recall:

| condition | haiku-5.5 | sonnet-5.5 |
|---|---|---|
| closed B=20 | −.044 [−.168, +.072] | – |
| closed B=40 | −.097 [−.186, −.020] | −.018 [−.048, +.021] |
| open B=40 | −.052 [−.163, +.043] | – |
| transfer B=40 (24) | −.071 [−.116, −.025] | – |

The trained 1.7B agent agrees: −.064 [−.116, −.007] at closed B=20 and −.003 [−.040, +.039] at
closed B=40 (`runs/synergy-v3-small-forced-r5`).

Forced chasing never helps and significantly hurts Haiku (F1 also drops on transfer, −.046).
The agents are right: here a citation hop from an included study yields fewer new eligible
studies than another query does. This matches the evidence from manual reviews that citation
searching is a low-yield supplement (Rajit et al., 2025: median 36% recall from automated
citation searching alone) and contrasts with PaSa's CS queries, whose targets cluster in
reference lists.


## Q3. Emergent strategies (prompted agents; trained agent in Q3 addendum)

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

## Larger evaluation: 40 more life-science reviews

All 72 SYNERGY+ reviews are already used (8 evaluation, 40 training, 24 transfer), so the
evaluation set cannot grow with untouched reviews. The 40 training reviews were used only to
train the 1.7B model (and as the Haiku teacher's T=0.7 trajectories), so the *prompted*
systems and BM25 can be scored on them without contamination (`runs/ext-lifesci40-r5`,
B=40, hybrid, T=0; the trained models are not evaluated here).

| system | recall | F1 |
|---|---|---|
| bm25 (include all) | .405 | .10 |
| haiku-5.5 | .493 | .35 |
| sonnet-5.5 | .558 | .43 |

| paired difference (B=40) | 40 new reviews | pooled 48 life-science reviews |
|---|---|---|
| sonnet − BM25 | +.154 [+.059, +.248] | +.152 [+.071, +.233] |
| haiku − BM25 | +.088 [−.010, +.183] | +.090 [+.006, +.173] |
| sonnet − haiku | +.066 [+.026, +.106] | +.061 [+.026, +.099] |

On 48 reviews every gap that was positive-but-uncertain on the original 8 becomes
significant, with similar magnitudes (sonnet − BM25 was +.142 on 8 reviews). Together with the
24 transfer reviews, the prompted-agent conclusions now rest on 72 reviews.

## Dense retrieval (SPECTER2)

See the Q1 caveat. SPECTER2 is the strongest non-LLM baseline in every condition
(closed .635, open .528 at B=40), beats BM25 by +.110 [+.026, +.217] at B=20 (5/1), and
misses fewer abstract-less gold (24 vs. 43 missed at B=40), consistent with title-only
embeddings carrying more signal than title-only BM25. It does not help the scripted snowball.

**LLM agent with SPECTER2 as its search tool** (`runs/synergy-v3{,-open}-api-dense-r5`;
same prompt and protocol, only `search` is backed by SPECTER2 instead of BM25). Recall (F1):

| system | closed B=20 | closed B=40 | open B=20 | open B=40 |
|---|---|---|---|---|
| haiku-5.5, BM25 tool | .432 (.32) | .626 (.39) | .450 (.36) | .499 (.33) |
| haiku-5.5, SPECTER2 tool | .365 (.32) | .523 (.33) | .397 (.31) | .488 (.30) |
| sonnet-5.5, BM25 tool | .488 (.42) | .666 (.49) | .493 (.44) | .619 (.43) |
| sonnet-5.5, SPECTER2 tool | – | .615 (.49) | – | .628 (.48) |
| specter2 exhaustive (no LLM) | .469 (.13) | .635 (.11) | .353 (.10) | .528 (.08) |

A better retriever does not make a better agent. On the closed pool the dense tool *hurts*
both models at B=40 (haiku −.103 [−.181, −.037], sonnet −.051 [−.093, −.004]); on the open
graph it is a tie (haiku −.012, sonnet +.009; sonnet F1 +.051 [−.007, +.106]). The agents
write short keyword queries and re-page them, which suits a lexical engine; SPECTER2 is
tuned for paper-to-paper similarity, so short keyword queries map to diffuse neighbourhoods
(exhaustive SPECTER2 search with the full review text as query is the strong baseline, not
SPECTER2 with agent queries). Versus exhaustive SPECTER2, the dense-tool Sonnet ties on recall
in the closed pool (−.020 [−.105, +.084]) and wins on the open graph (+.100 [+.020, +.183]),
with F1 +.38 to +.40 in both. So the LLM's value over dense retrieval is mostly screening
(precision) plus, in the larger open graph, recall.

## Q2. Small trained agent vs. large prompted agents

**Training (pass).** The gate passed (360 valid teacher trajectories >= 200; leakage audit
ok; one-GPU smoke test ok), so training was run, once, with pre-registered configs
(`configs/train/*.json`); no hyperparameter search.
- SFT: Qwen3-1.7B + LoRA (r=32) on 2,702 turns from 180 rejection-sampled haiku-5.5
  episodes (top half by reward) on 40 life-science reviews disjoint from all evaluation
  reviews; 2 epochs, 337 steps, 26 min on one L40S; loss .18 -> .12.
- GRPO from the SFT adapter: reward recall - 0.1*FP/|S| - c*cost/B, group 8, 4 tasks/step,
  40 steps (B=20, all three modes), KL to SFT beta=.02; 2h14m. Training reward by
  10-step window .125 / .162 / .179 / .160 (tasks change every step, so this is noisy);
  KL stayed < .002/token; 5 parse failures in 1,280 rollouts.

Recall (precision) on the 8 evaluation reviews, hybrid mode:

| system | params | closed B=20 | closed B=40 | open B=40 | transfer B=40 (24 reviews) |
|---|---|---|---|---|---|
| qwen3-1.7b prompted | 1.7B | .125 (.10) | .125 (.10) | .072 (.09) | .112 (.22) |
| qwen3-1.7b SFT | 1.7B | .368 (.19) | .433 (.15) | .449 (.14) | .459 (.23) |
| qwen3-1.7b SFT+GRPO | 1.7B | .420 (.17) | .467 (.12) | .458 (.11) | .483 (.20) |
| qwen3-32b prompted | 32B | .219 (.28) | .310 (.17) | .255 (.22) | .351 (.30) |
| haiku-5.5 prompted (teacher) | – | .432 (.32) | .626 (.29) | .499 (.26) | .477 (.39) |
| sonnet-5.5 prompted | – | .488 (.43) | .666 (.42) | .619 (.35) | .561 (.44) |
| bm25 / specter2 (include all) | – | .359 / .469 | .524 / .635 | .396 / .528 | .474 / – |

Paired differences (recall; 95% CI over reviews; wins/losses):

| comparison | closed B=40 | open B=40 | transfer B=40 |
|---|---|---|---|
| SFT - prompted 1.7B | +.308 [+.182, +.437] 7/0 | +.377 [+.193, +.558] 7/0 | +.347 [+.271, +.437] 24/0 |
| GRPO - SFT | +.034 [-.024, +.096] 4/2 | +.009 [-.051, +.079] 3/4 | +.025 [-.017, +.064] 14/7 |
| GRPO - qwen3-32b | +.156 [-.020, +.378] 4/2 | +.203 [+.064, +.379] 6/1 | +.133 [+.078, +.191] 18/3 |
| GRPO - haiku-5.5 | -.159 [-.342, -.028] 1/7 | -.042 [-.189, +.113] 3/5 | +.007 [-.037, +.050] 10/11 |
| GRPO - bm25 (ΔF1) | +.089 [+.040, +.136] 7/1 | +.094 [+.052, +.142] 8/0 | +.134 [+.101, +.168] 23/1 |

**Answer.** A 1.7B model trained on ~2.7k teacher turns goes from useless (.125) to beating
a prompted 32B model of the same family on recall (significant on open and transfer, and
at closed B=20), and matches its teacher's recall on the 24 out-of-field transfer reviews
(+.007, 10/11) while staying below the teacher on the in-domain life-science set at B=40.
Almost all of the gain comes from SFT; GRPO adds a small, consistent but not significant
recall gain (+.01 to +.05; significant only at open B=20, +.047 [+.012, +.083]) and
costs F1 (-.03 [-.055, -.003] closed B=40): with lambda=0.1 the cheapest way to raise reward
was to include more (169 vs. 131 papers at B=40). The small model's precision (.11-.20)
stays well below the prompted frontier models (.26-.44); it beats include-everything BM25
on F1 everywhere but does not match haiku/sonnet as a screener. In citation-only mode the
trained model is still poor (.133 GRPO vs .062 SFT at B=40; it stops with 72% of the budget
left). Caveats: 40 GRPO steps, n=8 evaluation reviews.

**Second training seed.** SFT and GRPO were rerun end to end with a different seed (4242:
LoRA init, data order, rollout sampling; `configs/train/*_s2.json`, `runs/*-s2-r4`), with
everything else identical. Hybrid recall at B=40:

| | closed | open | transfer |
|---|---|---|---|
| SFT seed 1 / seed 2 | .433 / .492 | .449 / .364 | .459 / .480 |
| GRPO seed 1 / seed 2 | .467 / .539 | .458 / .438 | .483 / .524 |
| GRPO − SFT, seed 1 | +.034 [−.027, +.096] | +.009 [−.048, +.077] | +.025 [−.017, +.065] |
| GRPO − SFT, seed 2 | +.047 [+.022, +.074] | +.074 [+.012, +.139] | +.044 [+.015, +.076] |

GRPO − SFT at B=40 is positive in all six seed × condition cells and significant in the three
seed-2 cells, so the GRPO gain is real but small (+.01 to +.07). The seed-to-seed difference
between two GRPO models (closed +.072 [+.044, +.101], open −.020, transfer +.041) is as large
as the GRPO effect itself: a single training run cannot rank two training recipes that differ
by less than about .07 recall. At B=20 the sign is not stable (open seed 2: −.034). The main
Q2 conclusions (SFT does almost all the work; trained 1.7B beats prompted 32B; matches the
teacher out of field) hold for both seeds: seed-2 GRPO beats Haiku on transfer (.524 vs .477).

## Q3 addendum: what the trained agent learned

At B=40 hybrid, the trained 1.7B agent spends 97% of the budget on search, never expands
citations (its haiku teacher spent 11% on citers), never spends budget on `read`
(screening is free), and issues ~13.5 distinct queries over ~39 searches: it re-issues a
query to page deeper (search returns only unexposed papers), i.e. it learned a
"few queries, deep pages, screen everything" policy - effectively multi-query BM25 with a
learned screener. Screen recall rose from .21 (prompted) to .67 (SFT) to .74 (GRPO).

## Can the agent tell how much it missed? (end-of-episode recall estimation)

`litsearch.recall_estimation`, `scripts/recall_est.py`, `scripts/recall_calibrate.py`; tables in
`analysis/robust/recall_{estimates,calibrated}.md`. After each B=40 hybrid episode (Sonnet,
Haiku, trained 1.7B; 8 + 40 + 24 reviews) we estimate recall without gold:
- *yield curve*: fit included(t) = A(1 − e^{−kt}) over charged cost, recall = included / A;
- *sample*: screen the included set, 30 exposed-but-excluded and 80 never-seen papers from the
  agent's universe; recall = found / (found + estimated missed), with a 95% Clopper–Pearson
  lower bound. Screener = gold labels (isolates sampling error) or Haiku (deployable; $1.17).

Pooled over the 96 evaluation episodes (8 + 24 reviews; true mean recall .53):

| estimator | mean estimate | mean abs. error | corr. with true recall | lower bound holds | mean lower bound |
|---|---|---|---|---|---|
| yield curve | .525 | .232 | .51 | – | – |
| sample, gold screener | .630 | .187 | .66 | 100% | ≈ .15 |
| sample, Haiku screener | .254 | .278 | .66 | 99% | ≈ .15 |
| sample, Haiku, calibrated on the 40 training reviews | .562 | .300 | .51 | 43–100% | – |

**Answer: not yet, and the reason is informative.**
1. Every estimator *ranks* episodes reasonably (r ≈ .5–.7), so an agent can tell a bad search
   from a good one, but no estimator is accurate per episode (error ≈ .2).
2. Valid bounds are too loose to use: with 80 samples from pools of thousands where 1–3% are
   eligible, the 95% lower bound is ≈ .15 when true recall is ≈ .55.
3. The LLM screener shares the agent's errors. Calibrated on the training reviews, Haiku says
   "include" to 92% of gold but also to 87% of the agent's false inclusions (and 43% of the
   papers the agent excluded). The agent's mistakes are exactly the papers an LLM finds
   plausible, so an LLM cannot audit them, and a Rogan–Gladen correction becomes unstable
   (bounds hold in only 43–90% of episodes).
Certified stopping for search agents therefore needs a screener whose errors are independent
of the agent's (human spot checks, or a differently trained classifier) and a much larger
random sample than a 40-call budget implies.

## Robustness: sampling variance and memorization

Full tables: `analysis/robust/robustness.md` (script `scripts/robustness.py`).

**Sampling variance.** Every number above is a single temperature-0 episode per cell. To size
the run-to-run noise, Haiku (hybrid, all budgets) and Sonnet (hybrid, B=40) were rerun three
times at temperature 0.7 on the 8 closed-pool reviews (96 episodes).

| agent | B | T=0 | T=0.7 repeats (mean over 8 reviews) | per-review SD |
|---|---|---|---|---|
| haiku-5.5 | 10 | .273 | .367 / .338 / .342 | .065 |
| haiku-5.5 | 20 | .432 | .496 / .436 / .443 | .066 |
| haiku-5.5 | 40 | .626 | .551 / .525 / .585 | .045 |
| sonnet-5.5 | 40 | .666 | .637 / .655 / .654 | .016 |

Sonnet is stable: three samples land within .02 of each other and of the T=0 run. Haiku moves
by up to .06 between repeats, and its T=0 run at B=40 sits at the top of its sampled range (its
B=10 run sits below it). So Haiku's single-run numbers carry about ±.05 of sampling noise,
which is as large as several of the Sonnet−Haiku gaps reported above. Sonnet − Haiku at B=40
is positive in all three repeats (+.086, +.129, +.069), with a bootstrap CI excluding zero in
one of them; this matches the closed-pool T=0 conclusion (positive but not significant on 8
reviews) and the transfer result (+.084 [+.045, +.128] on 24 reviews).

**Memorization.** A closed-book probe (`litsearch probe-memory`) gives each model exactly the
agent's starting text (review question and eligibility criteria), no tools, and asks it to list
up to 50 included studies. Titles are matched to the review's candidate pool (normalized-title
similarity ≥ .8; checked by hand: matches above .84 were all the same study).

| model | closed pool (8 reviews): gold named / 341 | mean closed-book recall | transfer (24): gold named / 2233 | mean recall |
|---|---|---|---|---|
| sonnet-5.5 | 22 | .063 | 76 | .062 |
| haiku-5.5 | 9 | .030 | 23 | .017 |
| qwen3-32b | 0 | .000 | 8 | .006 |

Memorization exists but is concentrated: Hall_2011 (software defect prediction, a heavily
cited review) accounts for 15 of Sonnet's 22 closed-pool hits; six of eight reviews have 0–2.
Sonnet's tool-based recall (.666) is ten times its closed-book recall. To test whether
memorized papers drive the comparisons, every gold paper named from memory by *any* model was
removed (25/341 closed, 87/2233 transfer) and all systems were rescored on the rest:

| paired difference at B=40 | original | memorization-adjusted |
|---|---|---|
| closed: sonnet hybrid − BM25 | +.142 [+.051, +.242] | +.129 [+.049, +.215] |
| closed: sonnet hybrid − SPECTER2 | +.031 [−.051, +.113] | +.012 [−.082, +.109] |
| transfer: sonnet hybrid − BM25 | +.087 [+.013, +.158] | +.081 [+.006, +.154] |
| transfer: sonnet − haiku (hybrid) | +.084 [+.045, +.128] | +.088 [+.048, +.133] |
| transfer: GRPO 1.7B − qwen3-32b | +.133 [+.078, +.191] | +.134 [+.079, +.193] |

Every system loses at most .02 recall, and no conclusion changes sign or significance. Sonnet's
small edge over exhaustive SPECTER2 on the closed pool shrinks toward zero (+.012), so that
comparison should be read as a tie.

## Reproducibility

- Fresh-environment test (2026-10-31): clean Python 3.12 venv from a source copy,
  `pip install -e .`, README quickstart end to end (fixture build -> audit -> run ->
  replay), `python -m unittest discover -s tests`: 42 tests OK.
- Replays: `demo/muthu_sonnet_hybrid_b40.html` and `demo/muthu_sonnet_citation_b40.html`
  (same test review, same model, the two tool sets) are self-contained HTML.

## Limitations

- The trained models are evaluated on 8 life-science reviews plus 24 transfer reviews
  (their 40 training reviews cannot be reused); prompted systems are evaluated on 72.
- Two training seeds; seed-to-seed variation (up to .07 recall) is as large as the GRPO effect.
- Search cutoff = review publication date (SYNERGY has no search dates); this can only add
  non-gold distractors, never hide gold.
- Candidate pools are the reviews' own search results, so the closed graph is easier than the
  open literature; the open-graph condition partially relaxes this.
- ~27% of papers have no abstract in OpenAlex; title-only screening is a real constraint.
- Frontier models have seen some of these reviews in pretraining. The closed-book probe
  bounds the effect (≤ 7% of gold nameable from memory; removing it changes no conclusion), but
  memory can also shape queries in ways a title probe does not measure.
- Main-table cells are single T=0 episodes; Haiku's sampled repeats vary by up to .06 recall.
- Sonnet's mandatory reasoning differs from the other models' non-thinking mode.

## GRPO v2 evaluation (final checkpoints; full tables in analysis/robust/grpo2.md)

**A second training recipe: GRPO v2.** It uses an F2 reward, turn-level credit from each call's
marginal gold gain, a redundancy penalty, mixed budgets and 120 steps, with two seeds plus an
ablation without turn credit. We fixed the success criteria in advance: precision of at least .25
with recall within .02 of GRPO v1, and a gain over SFT larger than seed noise. **It missed them.**
At budgets 20 and 40 no run passed the precision and recall test (only five budget-10 cells did),
and the recall gain over SFT was not consistently beyond seed noise.

| transfer, hybrid, B=40 | recall | precision | F1 |
|---|---|---|---|
| SFT (seed 1 / seed 2) | .459 / .480 | .228 / .223 | .266 / .268 |
| GRPO v1 | .483 / .524 | .196 / .196 | .244 / .251 |
| GRPO v2, turn credit | .442 / .475 | .255 / .236 | .286 / .276 |
| GRPO v2, no turn credit (seed 1) | .494 | .193 | .244 |

Two things did come out of it:

- *Turn credit moves the policy along the recall-precision trade-off.* Compared with the same
  recipe without it, on transfer at B=40: recall −.052 [−.100, −.008], precision +.062
  [+.045, +.080], F1 +.042 [+.022, +.062]. Compared with v1, F1 rises in both seeds (+.042, +.025)
  and recall falls in both (−.041, −.050). This is a choice of operating point, not a better search.
  With hybrid tools every policy still spends 100% of its budget on search.
- *With citation tools only, v2 stops early less often (exploratory).* It spends 20–28 of 40 calls
  where SFT spends about 8, and recall rises over SFT in all six seed × universe cells (+.09 to
  +.24; five CIs exclude zero and the sixth touches it). This wasn't one of the pre-stated tests, it
  rests on 8 reviews, and absolute recall stays low (.13–.30).
