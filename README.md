# Budgeted Literature Search

An offline, leakage-audited benchmark for testing whether an agent can recover the
studies included in a systematic review under a fixed search budget. The same
environment supports citation-only, keyword-only, and hybrid search so their
recall/cost curves are directly comparable.

The dependency-free core includes:

- a typed, Gym-like search environment;
- BM25 seeding and scripted snowballing;
- provider-neutral OpenAI-compatible prompted agents;
- charged tool costs and separate paper-exposure accounting;
- recall, precision, F1, normalized AURC, and censored WSS@95;
- immutable data manifests, checksummed run traces, and leakage audits;
- a self-contained HTML graph/timeline replay;
- Slurm scripts for FarmShare; and
- optional SPECTER2, QLoRA SFT, and training dependencies.

## Quick start

Python 3.11 or newer is required. The fixture workflow needs no third-party packages.

```bash
cd /path/to/literature-search-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

litsearch data build --source fixture --output artifacts/fixture
litsearch data audit --data artifacts/fixture
litsearch run \
  --data artifacts/fixture \
  --review fixture_depression \
  --agent snowball \
  --mode hybrid \
  --budget 20 \
  --output runs/fixture.json
litsearch demo build --trace runs/fixture.json --output runs/fixture.html
```

Open `runs/fixture.html` in any browser. It does not load a server, CDN, model, or
external JavaScript.

### Run the LLM agent on one example

Any OpenAI-compatible endpoint works. With OpenRouter (one fixture episode costs well under $0.01):

```bash
export MODEL_API_BASE=https://openrouter.ai/api/v1
export MODEL_API_KEY=...            # your key; it is sent only to MODEL_API_BASE and never logged
litsearch run \
  --data artifacts/fixture \
  --review fixture_depression \
  --agent prompted \
  --model anthropic/claude-haiku-5.5 \
  --mode hybrid \
  --budget 20 \
  --output runs/fixture_llm.json
litsearch report --run-dir runs                       # recall / precision / F1 table
litsearch demo build --trace runs/fixture_llm.json --output runs/fixture_llm.html
```

The full evaluation (72 SYNERGY+ reviews) is built and run with the data, matrix and
FarmShare instructions below. Every reported number, with the run directory it came
from, is in `docs/RESULTS.md`.

Run the test suite without installing pytest:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Data build

### SYNERGY

Install the data extra and download/unpack SYNERGY according to its official package.
The builder accepts a directory containing one CSV per review. Common field aliases
for OpenAlex ID, title, abstract, year, and inclusion label are recognized.

```bash
python -m pip install -e '.[data]'
litsearch data build \
  --source synergy \
  --input /scratch/users/$USER/synergy/raw \
  --output /scratch/users/$USER/litsearch/data/synergy
```

An optional `reviews.json` in the input directory supplies review-level fields:

```json
{
  "Brouwer_2019": {
    "question": "Which factors predict recurrence after remission from depression?",
    "eligibility_criteria": "Prospective longitudinal human studies ...",
    "publication_year": 2019,
    "search_cutoff_year": 2017,
    "review_paper_id": "W123456",
    "domain": "medicine"
  }
}
```

Enrich a bounded deterministic subset with cached OpenAlex metadata and one-hop
references/citers. The cache makes subsequent builds offline and inspectable:

```bash
litsearch data enrich \
  --data /scratch/users/$USER/litsearch/data/synergy \
  --output /scratch/users/$USER/litsearch/data/synergy-openalex \
  --cache /scratch/users/$USER/litsearch/openalex-cache \
  --email "$OPENALEX_EMAIL" \
  --max-seed-papers 5000
litsearch data audit --data /scratch/users/$USER/litsearch/data/synergy-openalex
```

The portable artifact contract is `works.jsonl`, `reviews.jsonl`, `splits.json`, and
`manifest.json`. Raw data and caches are not committed.

## Environment contract

Episodes receive the review question, eligibility criteria, five common BM25 seeds,
the current frontier, prior decisions, and remaining budget. Gold IDs remain private.
The common seed retrieval costs one budget unit and appears as step zero in every trace.

| Action | Availability | Maximum result count | Charged cost |
|---|---|---:|---:|
| `search(query)` | keyword, hybrid | 10 | `ceil(results/10)` |
| `expand_refs(p)` | citation, hybrid | 20 | `ceil(results/10)` |
| `expand_citers(p)` | citation, hybrid | 20 | `ceil(results/10)` |
| `read(p)` | all | one abstract | 1 |
| `include/exclude/stop` | all | - | 0 |

All metadata exposures and full-abstract reads are logged separately. Papers after the
review cutoff, the review paper, and works citing the review are filtered before an
episode starts. Invalid tools, unknown papers, and insufficient budgets produce typed
errors rather than exceptions. Once the budget is spent the agent may still screen
(include/exclude, free) what it has already seen; the episode ends on `stop` or on the
next attempt to spend.

Conditions:

- **closed graph** (`data/synergy-v3`): each review's universe is its SYNERGY screening pool.
- **open graph** (`data/synergy-v3-open-v2`, `litsearch data open-graph`): the universe also
  contains up to 5,000 out-of-pool works the visible pool cites (gold-blind ranking),
  with their own references, so citation chasing can leave the pool and search faces
  extra distractors. Gold sets are unchanged and the leakage audit is re-run.
- `index_scope: pool` (all r3 runs) builds a BM25 index per review over its universe.

## Prompted agents and experiment matrix

`litsearch.llm.LLMSearchAgent` makes one model call per *turn*: the model screens the
unscreened papers it has seen (free) and picks exactly one charged action, replying with
`{"include": [...], "exclude": [...], "action": {...}}`. Any OpenAI-compatible endpoint
works (OpenRouter, vLLM). Responses are cached by request hash; an optional shared
spend ledger enforces a USD cap across concurrent jobs.

```bash
# API models (CPU job): key read from ~/.config/litsearch/openrouter.key, never logged
sbatch --export=ALL,MATRIX_CONFIG=configs/llm/v3_api.json farmshare/api_matrix.sbatch
# Local open model (GPU job): vLLM server + matrix in one job
sbatch --export=ALL,MODEL=Qwen/Qwen3-1.7B,SERVED=qwen3-1.7b,MATRIX_CONFIG=configs/llm/v3_qwen3-1.7b.json \
  farmshare/llm_matrix.sbatch
litsearch report --run-dir runs/<dir>                      # recall/precision/AURC tables
litsearch analyze --data data/synergy-v3 --run-dir runs/<a> --run-dir runs/<b> \
  --open-data data/synergy-v3-open-v2 --output analysis/<name>
```

`analyze` writes `analysis.md`, `analysis.json`, `misses.jsonl` and figures: recall vs.
cost and vs. papers exposed, search recall (gold ever exposed) vs. selection recall,
gold reachability from the seeds (closed vs. open graph), per-system action mix
(strategy), and a classification of every missed gold study.

Each run is separate at B=10, 20, and 40; lower-budget metrics are not truncations of
a policy that knew it had a larger budget.

## Dense retrieval

`litsearch index --data <store> --output index/specter2-<store>` encodes every paper with
SPECTER2 (proximity adapter); queries use the ad-hoc query adapter. Retrieval is exact
inner product over the review's universe. See `configs/synergy_v3_dense_r3.json`.

## Training (SFT -> GRPO)

1. Teacher episodes on the 40 disjoint life-science training reviews
   (`configs/llm/teacher_api.json`, temperature 0.7).
2. `litsearch collect-sft --run-dir runs/<teacher> --output sft/<name>.jsonl` keeps the
   best-rewarded half per (mode, budget); reward `r = recall - λ·FP/|S| - c·cost/B`.
3. `litsearch train-sft --config configs/train/sft_qwen3_1.7b.json` (LoRA; refuses to start
   unless the audit passed and >= 200 valid teacher episodes exist).
4. `litsearch train-grpo --config configs/train/grpo_qwen3_1.7b.json` via
   `farmshare/train_grpo.sbatch`: vLLM serves rollouts with the current LoRA hot-loaded,
   group-normalized advantages, token-level loss, KL to the SFT policy.

Evaluation uses the 8 v3 reviews (train/dev/test split in `splits.json`) and, for
cross-field transfer, 24 held-out non-life-science reviews (`data/synergy-ext-transfer-v2`).

## FarmShare

Use `$HOME` for code and `/scratch/users/$USER/literature-search-agent` for datasets, caches,
models, and runs. Do not run builds or models on a login node. `bash farmshare/sync.sh`
rsyncs code (stamping `GIT_SHA` so traces record the version);
`sbatch farmshare/setup_ml_env.sbatch` creates `envs/ml` (vLLM/TRL/PEFT) and `envs/emb`
(SPECTER2). OpenAlex now enforces a per-IP daily budget without an API key: set
`OPENALEX_API_KEY`, or list uncached URLs with `--list-missing` and fill the cache with
`litsearch data fetch-urls` from another network.

## Repository layout

```text
src/litsearch/       environment, retrieval, agents, metrics, data, CLI
tests/               unit, leakage, contract, and end-to-end fixture tests
configs/             matrix and optional training examples
farmshare/           Slurm batch entrypoints
scripts/             local matrix helper
```

## Reproducibility checklist

1. Freeze raw dataset versions and OpenAlex cache.
2. Run `litsearch data audit`; retain its JSON output.
3. Freeze `splits.json` before tuning prompts or thresholds.
4. Pin model names, prompts, budgets, modes, and seeds in the matrix config.
5. Preserve each run JSON and adjacent SHA-256 file.
6. Report per-review results as well as macro averages.
7. Report WSS@95 as `null` when 95% recall is not reached.

