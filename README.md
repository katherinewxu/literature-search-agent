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
errors rather than exceptions.

## Prompted agents and experiment matrix

The client uses the OpenAI chat-completions wire format but is provider-neutral:

```bash
export MODEL_API_BASE=https://provider.example/v1
export MODEL_API_KEY=...
export MODEL_NAME=frontier-model-version
litsearch run ... --agent prompted --model "$MODEL_NAME"
```

Credentials are never written to traces. To execute a matrix, copy
`configs/mvp_matrix.example.json`, set its data/output paths, and run:

```bash
litsearch matrix --config configs/mvp_matrix.json
litsearch evaluate --run-dir runs/mvp --output runs/mvp/summary.json
```

Each run is separate at B=10, 20, and 40; lower-budget metrics are not truncations of
a policy that knew it had a larger budget. Summary confidence intervals use a paired
review-level bootstrap when at least two reviews are present.

## Dense retrieval and training gate

Install model support only on a GPU/ML machine:

```bash
python -m pip install -e '.[ml,train]'
```

`Specter2Retriever` loads the official SPECTER2 base plus proximity adapter. QLoRA SFT
is launched with `litsearch train-sft --config configs/train_sft.example.json`. Training
refuses to start unless the configured audit passed and the minimum trajectory count is
present. The GRPO command currently validates the same gate in dry-run mode; interactive
rollout training remains deliberately disabled until the environment adapter is audited.

## FarmShare

Use `$HOME` for code and `/scratch/users/$USER/literature-search-agent` for datasets, caches,
models, and runs. Do not run builds or models on a rice login node.

```bash
rsync -av --exclude .git ./literature-search-agent/ \
  "$USER"@dtn.farmshare.stanford.edu:~/literature-search-agent/
ssh farmshare
cd ~/literature-search-agent
module load python/3.11.7
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ml]'
sbatch farmshare/build_data.sbatch
sbatch farmshare/eval_matrix.sbatch
```

The scripts write logs and artifacts beneath `/scratch/users/$USER/literature-search-agent`.
Inspect live partitions with `sinfo` before submitting GPU work because availability and
limits can change.

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

