# One-week execution sprint

The non-negotiable deliverable is a leakage-audited benchmark, matched-budget
citation/keyword/hybrid comparison, reproducible traces, and presentation-ready replay.
Training is allowed to consume time only after that package exists.

## Day 1 - freeze data and contracts

- Materialize eight SYNERGY reviews and extended OpenAlex metadata on FarmShare scratch.
- Resolve review questions, eligibility criteria, review-paper IDs, and search cutoffs.
- Freeze review IDs and run the leakage/data-quality audit.
- Run BM25 and snowballing smoke episodes on every review.

Exit gate: every selected review loads, has at least 15 gold studies where possible,
and completes B=10/20/40 episodes without manual repair.

## Day 2 - baseline matrix

- Build BM25 and SPECTER2 indexes.
- Launch citation-only, keyword-only, and hybrid baselines as a Slurm array.
- Diagnose missing edges, post-cutoff leakage, empty abstracts, and identifier loss.
- Freeze the benchmark artifact and do not change test labels afterward.

Exit gate: baseline summary and per-review recall/cost curves exist.

## Day 3 - prompted agents

- Run the frontier prompted model in all three modes.
- Run the small local model in hybrid mode.
- Repeat two development reviews three times to quantify nondeterminism.
- Cache all prompts/responses and track token use, errors, and cost.

Exit gate: the main comparison table is complete or every missing cell has a recorded
failure reason.

## Day 4 - robustness and analysis

- Run matched paper-exposure sensitivity analysis.
- Bootstrap review-level confidence intervals.
- Label up to 20 plausible non-gold inclusions.
- Categorize stopping, loop, direction, query-drift, and over-inclusion failures.

Exit gate: all headline claims have a table/figure and a trace-level audit trail.

## Day 5 - submission-quality artifact

- Generate the static HTML traversal replay and final figures.
- Run the full test/leakage/reproducibility suite from a fresh environment.
- Draft results, limitations, grader-feedback response, and README quick start.
- Copy compact outputs back locally.

Exit gate: benchmark, results, demo, and documentation stand alone without RL.

## Days 6-7 - gated training only

Start QLoRA SFT only if there are at least 200 valid trajectories and a one-GPU smoke
test fits. Attempt one GRPO configuration only after SFT evaluation works. Stop training
immediately if it threatens analysis or reproducibility time; report the gate outcome as
a result rather than hiding it.

## Parallel execution

Use `litsearch matrix --count` to size the experiment and submit up to 16 cells at once:

```bash
MAX_PARALLEL=16 bash farmshare/submit_one_week.sh \
  /scratch/users/$USER/literature-search-agent/mvp_matrix.json
```

Aggregate only after the array finishes:

```bash
litsearch evaluate \
  --run-dir /scratch/users/$USER/literature-search-agent/runs/mvp \
  --output /scratch/users/$USER/literature-search-agent/runs/mvp/summary.json
```
