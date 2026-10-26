# Codex continuation handoff

## Mission

Finish the one-week MVP for the budgeted literature-search final project. Work
autonomously: inspect, implement, test, submit FarmShare jobs, diagnose failures,
and iterate. Do not stop merely to present a plan. Ask the user only for credentials
or an irreversible/product-defining decision.

The required deliverable is a leakage-audited real-data benchmark, matched-budget
citation/keyword/hybrid baselines, reproducible traces and summaries, and a replayable
demo. Training is optional and must remain gated until the benchmark package is done.
See `ONE_WEEK_SPRINT.md` for the milestone order.

## Locations and access

- Local repository: `/Users/katherine/literature-search-agent`
- FarmShare login: `kwx04@login.farmshare.stanford.edu`
- Existing SSH control socket: `/tmp/farmshare-litsearch.sock`
- Remote repository: `/home/users/kwx04/literature-search-agent`
- Remote scratch root: `/scratch/users/kwx04/literature-search-agent`
- Remote virtualenv: `/home/users/kwx04/literature-search-agent/.venv`
- FarmShare is Slurm-managed. Do substantial computation in jobs, not on a rice login
  node.

Use the existing connection as follows:

```bash
ssh -S /tmp/farmshare-litsearch.sock \
  kwx04@login.farmshare.stanford.edu '<command>'
```

If that socket no longer exists, the user must establish a new authenticated SSH
connection. Do not request or handle their password.

Sync code without raw data, virtualenvs, or run artifacts:

```bash
cd /Users/katherine/literature-search-agent
rsync -az \
  --exclude .git --exclude .venv --exclude artifacts --exclude runs \
  --exclude __pycache__ --exclude '*.pyc' \
  -e 'ssh -S /tmp/farmshare-litsearch.sock' ./ \
  kwx04@login.farmshare.stanford.edu:~/literature-search-agent/
```

## Current state (2026-10-25, commit c29eb5c+)

Tests: `PYTHONPATH=src python3.12 -m unittest discover -s tests` -> 30/30 (local
`python3` is conda 3.10 and cannot import `StrEnum`; use 3.12). Sync with
`bash farmshare/sync.sh` (stamps `GIT_SHA` so traces record the code version).

### Frozen benchmark: `/scratch/users/kwx04/literature-search-agent/data/synergy-v3`

Built by `farmshare/build_synergy_v3.sh` (Slurm job 1894930). 66,361 papers, 8 reviews,
341 gold, audit `data/synergy-v3-audit.json`: `ok=true`, 0 leakage issues, every gold
study present and visible; one warning (Bos_2018 has 10 gold < 15). Split 4/2/2:
train Meijboom, Leenaars, Brouwer, Bos; dev Hall_2011, Donners; test Muthu, Appenzeller-Herzog.

Decisions baked into v3 (report them in the write-up):

- Reviews: SYNERGY+ renamed `Hall_2012` -> `Hall_2011` and dropped `Jeyaraman_2020`;
  `Muthu_2020` (spine surgery RCTs, 70 gold) replaces Jeyaraman.
- Ingestion reads SYNERGY+ source folders (`~/.synergy_dataset_source/synergy-dataset-plus/<key>/
  labels.csv + works_*.zip`) via `--source synergy-source`. The `synergy_dataset` CSV
  exporter keeps only open-access works with abstracts and dropped 50-96% of records
  *including gold* (Hall: 104 -> 4 gold). The CSVs in `synergy/raw/` are therefore NOT used.
- Abstracts: the source keeps abstracts only for OA works; `data backfill-abstracts`
  filled 30,774 from the OpenAlex API (cache `cache/openalex`, sha256 URL keys). 18,047
  still empty (absent in OpenAlex too); per-review empty rates 0.6%-49% in the audit.
- Episodes are scoped to each review's candidate pool (`Review.candidate_ids`).
  Citation expansion therefore only reaches in-pool papers (in-pool edge coverage
  47-87% of candidates, 93-100% of gold).
- Cutoff = review publication date (no search dates in SYNERGY); this only exposes
  extra non-gold papers between search and publication.
- Question = review title (SYNERGY `research_question` is inconsistent).
- Internal research artifact: do not redistribute works.jsonl (non-OA abstracts).

### Baselines: `runs/synergy-v3-baselines-r2` (array 1895103, 168/168 completed)

Config `configs/synergy_v3_baselines.json`. `report.md` + `summary.json` there. Mean
recall at B=10/20/40: bm25/keyword .21/.32/.44; snowball/hybrid .05/.08/.13;
snowball/keyword .05/.08/.12; snowball/citation .04/.04/.08; random/hybrid .02/.02/.04.
`runs/synergy-v3-baselines` (r1) is superseded (old snowball hybrid never searched).
Replay demo: `demo/meijboom_bm25_b40.html` (from r1 trace).

Caveat for analysis: inclusion requires exposure, not reading. BM25 includes 10
papers per 1-cost search (~377 exposed at B=40) while snowball pays 1 per read (~34),
so the matched paper-exposure analysis is required before headline claims.

## Next actions

1. Prompted agents (needs the user's model/API choice and spend cap): frontier model in
   citation/keyword/hybrid on all 8 reviews, small open model hybrid; cache responses.
2. Grader feedback ("is the citation graph limiting?"): add an open-graph condition with
   bounded OpenAlex one-hop enrichment beyond the candidate pool (`data enrich`; must
   union new neighbors into `candidate_ids` and re-run the audit) and compare.
3. Matched paper-exposure analysis + recall-vs-cost curves (figures) from traces.
4. SPECTER2 index (GPU job `farmshare/specter_index.sbatch`) and dense keyword baseline.
5. Snowball include threshold is untuned (0.16); tune only on train reviews if at all.
6. Failure analysis, fresh-env test run, README quick start, training gate (>=200 valid
   trajectories) decision.

## Known design constraints and pitfalls

- The correct scratch path is `/scratch/users/$USER/...`, not `/scratch/$USER/...`.
- The official login is `login.farmshare.stanford.edu`; the working account name is
  `kwx04`, not `katherine`.
- Raw exports may be large. Avoid copying them back to the laptop or committing them.
- Current citation edges cover references among works present in the selected candidate
  pools. Measure and report graph coverage; bounded OpenAlex enrichment can follow.
- BM25 currently scores the allowed corpus in pure Python. Profile one real episode
  before launching all cells. If too slow, add a persisted or per-review cache rather
  than blindly increasing Slurm time.
- The review-year suffix is only a fallback cutoff. Prefer actual search dates from
  metadata when available, and document any conservative inference.
- Gold labels are private to the environment/evaluator. Never put them in prompts,
  seeds, observations, or retriever queries.
- Do not start QLoRA/GRPO merely to claim training. The explicit gate is at least 200
  valid trajectories plus a passing leakage audit and a one-GPU smoke test.
- Do not expose API keys in commands, logs, traces, commits, or handoff notes.

## Completion standard

Do not declare the one-week milestone complete until all of the following exist and are
verified:

- frozen real-data artifact plus manifest, splits, and passing audit;
- per-review/mode/budget baseline traces with hashes;
- aggregate summary and matched-budget comparison table/curves;
- at least one self-contained traversal replay;
- fresh-environment test results and reproducible run instructions;
- a candid limitations/failure analysis and an explicit pass/skip result for training.

