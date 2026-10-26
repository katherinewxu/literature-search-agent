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

## What is implemented

The repository contains a dependency-free core package with:

- typed paper/review/action models and a Gym-like budgeted environment;
- leakage filters, explicit action costs, and exposure/read accounting;
- BM25, optional SPECTER2, scripted snowballing, random, and prompted agents;
- recall, precision, F1, normalized AURC, censored WSS@95, bootstrap summaries;
- immutable data manifests, checksummed traces, replay, matrix execution, and CLI;
- SYNERGY CSV ingestion, OpenAlex enrichment/cache support, and training gates;
- Slurm scripts, including one-cell-per-array-task execution.

The latest local test result was **23/23 passing**:

```bash
cd /Users/katherine/literature-search-agent
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m litsearch matrix \
  --config configs/fixture_matrix.json --count  # expected: 15
```

The newest ingestion patch parses `referenced_works` from JSON/Python-list/pipe
representations, reconstructs reverse citer edges, infers review years, and derives a
domain. It has been synced to FarmShare. The fixture workflow previously completed on
FarmShare, and Slurm job `1894720` ran all then-current tests successfully.

No git commit exists yet. All project files appear untracked because the repository was
initialized during this session. Do not discard them. Make the first commit only after
the real-data build and baseline smoke test are verified.

## Real SYNERGY data state

The remote package was installed with `.[data]`; `pandas>=2.2` was added to
`pyproject.toml` because `synergy-dataset` did not declare the pandas dependency needed
by its CSV export.

Eight reviews are targeted:

```text
Appenzeller-Herzog_2019
Bos_2018
Brouwer_2019
Donners_2021
Hall_2012
Jeyaraman_2020
Leenaars_2019
Meijboom_2021
```

The first four extended exports exist under:

```text
/scratch/users/kwx04/literature-search-agent/synergy/raw/
```

Known sizes at last inspection:

```text
Appenzeller-Herzog_2019.csv   5,403,716 bytes
Bos_2018.csv                 29,825,655 bytes
Brouwer_2019.csv            132,133,413 bytes
Donners_2021.csv              1,087,218 bytes
```

The first exporter stopped before writing review metadata. A second export for the
remaining four reviews was submitted as Slurm job **1894878**. It writes to the separate,
initially empty directory:

```text
/scratch/users/kwx04/literature-search-agent/synergy/raw_rest/
```

and logs to:

```text
/scratch/users/kwx04/literature-search-agent/logs/synergy-export-1894878.log
```

The monitoring command was interrupted, so the next agent must check the actual state:

```bash
ssh -S /tmp/farmshare-litsearch.sock kwx04@login.farmshare.stanford.edu \
  'sacct -j 1894878 --format=JobID,State,ExitCode,Elapsed,MaxRSS; \
   task_root=/scratch/users/$USER/literature-search-agent; \
   tail -n 80 "$task_root/logs/synergy-export-1894878.log" 2>/dev/null || true; \
   find "$task_root/synergy/raw_rest" -maxdepth 2 -type f -printf "%P %s bytes\\n" | sort'
```

The exporter refuses a non-empty output directory. That is why `raw_rest` is separate.
Do not delete or overwrite the four completed CSVs. After job success, copy the four
new review CSVs into `synergy/raw/`. Preserve the generated
`raw_rest/metadata/review_metadata.csv` as a separate input/reference rather than
silently overwriting anything.

## Immediate next actions

1. Check Slurm job `1894878`. If it failed, diagnose its log and resubmit to a new empty
   output directory. Never erase the successful exports.
2. Integrate SYNERGY review metadata into `build_from_synergy_csv`. The package exports
   `metadata/review_metadata.csv`; load rows keyed by `key`, using at least `title`,
   `eligibility_criteria`, publication year/date, and the review OpenAlex ID where
   available. Explicit `reviews.json` overrides should win. Add focused tests.
3. Combine the eight CSV inputs non-destructively and run:

   ```bash
   task_root=/scratch/users/$USER/literature-search-agent
   ~/literature-search-agent/.venv/bin/litsearch data build \
     --source synergy \
     --input "$task_root/synergy/raw" \
     --output "$task_root/data/synergy"
   ~/literature-search-agent/.venv/bin/litsearch data audit \
     --data "$task_root/data/synergy" > "$task_root/data/audit.json"
   ```

   If the output already exists, inspect it first and build to a versioned sibling
   directory instead of overwriting it.
4. Inspect the audit and data statistics: paper/review/gold counts, missing identifiers,
   empty abstracts, post-cutoff gold studies, duplicate works, citation edge coverage,
   and per-review candidate sizes. Fix root causes before baselines.
5. Add a checked-in real baseline matrix config using the eight reviews, BM25 retrieval,
   snowball citation/keyword/hybrid, random hybrid, budgets 10/20/40, seed 329, and the
   exact scratch paths. Exclude prompted models and SPECTER2 from this first CPU matrix.
6. Run one cell as a smoke test, then launch the matrix array:

   ```bash
   MAX_PARALLEL=16 bash farmshare/submit_one_week.sh \
     /scratch/users/$USER/literature-search-agent/<baseline-config>.json
   ```

7. Monitor with `squeue`/`sacct`, inspect every failed array log, resubmit only failed
   cells, then aggregate with `litsearch evaluate`.
8. Only after reproducible CPU baselines exist, add SPECTER2, prompted-agent runs,
   analysis figures/replay, and finally the gated training decision.

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

