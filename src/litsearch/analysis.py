"""Evaluator-side analyses over run traces (gold labels are read here, never by agents).

- curves: recall vs. charged cost and vs. papers exposed, averaged over reviews
- exposure decomposition: search recall (gold ever exposed) vs. selection recall (gold included)
- matched exposure: recall once a system has exposed N papers
- reachability: gold reachable from the seeds within h citation hops (closed vs. open graph)
- strategy: action mix, query reformulation, expansion sources (Q3)
- misses: why each gold study was not found (failure analysis)
- paired: per-review paired differences between systems with bootstrap CIs
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from .environment import LiteratureSearchEnv, allowed_ids_for
from .models import EpisodeSpec
from .retrieval import BM25Retriever
from .store import CorpusStore


def system_key(run: dict[str, Any]) -> str:
    return f"{run['agent']}/{run['episode']['mode']}"


def load_runs(run_dirs: Iterable[Path]) -> list[dict[str, Any]]:
    runs = []
    for run_dir in run_dirs:
        for path in sorted(Path(run_dir).glob("*.json")):
            run = json.loads(path.read_text(encoding="utf-8"))
            if "episode" in run and "trace" in run:
                run["_path"] = str(path)
                runs.append(run)
    return runs


def _mean_ci(values: list[float], seed: int = 329, samples: int = 2000) -> tuple[float, float, float]:
    mean = statistics.fmean(values) if values else float("nan")
    if len(values) < 2:
        return mean, mean, mean
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples))
    return mean, draws[int(0.025 * samples)], draws[int(0.975 * samples)]


# -- per-run derived series ------------------------------------------------------
def run_series(run: dict[str, Any], gold: frozenset[str]) -> dict[str, list[float]]:
    """Cumulative cost, exposed count, exposure recall and selection recall after each event."""
    exposed: set[str] = set()
    cost = 0
    series: dict[str, list[float]] = {"cost": [0], "exposed": [0], "exposure_recall": [0.0], "recall": [0.0]}
    for event in run["trace"]:
        cost += event["charged_cost"]
        exposed.update(event.get("result_ids") or [])
        included = set(event.get("included_ids") or [])
        series["cost"].append(cost)
        series["exposed"].append(len(exposed))
        series["exposure_recall"].append(len(exposed & gold) / len(gold))
        series["recall"].append(len(included & gold) / len(gold))
    return series


def _value_at(xs: list[float], ys: list[float], x: float) -> float:
    """Step-function value of the last point with xs <= x (series are non-decreasing in x)."""
    value = ys[0]
    for xi, yi in zip(xs, ys):
        if xi > x:
            break
        value = yi
    return value


def curves(runs: list[dict[str, Any]], gold: dict[str, frozenset[str]], *, budget: int) -> dict[str, Any]:
    """Mean recall vs. cost (0..budget) and vs. exposed papers, per system, at one budget."""
    by_system: dict[str, list[dict[str, list[float]]]] = defaultdict(list)
    for run in runs:
        if run["episode"]["budget"] != budget:
            continue
        by_system[system_key(run)].append(run_series(run, gold[run["episode"]["review_id"]]))
    cost_grid = list(range(0, budget + 1))
    exposure_grid = [0, 10, 25, 50, 100, 200, 300, 400]
    result: dict[str, Any] = {"budget": budget, "cost_grid": cost_grid, "exposure_grid": exposure_grid, "systems": {}}
    for system, series_list in sorted(by_system.items()):
        entry: dict[str, Any] = {"n": len(series_list)}
        for metric in ("recall", "exposure_recall"):
            entry[f"{metric}_by_cost"] = [
                statistics.fmean(_value_at(s["cost"], s[metric], c) for s in series_list) for c in cost_grid
            ]
            # Only runs that reached N exposed papers contribute at N (no extrapolation).
            by_exposure = []
            for n in exposure_grid:
                values = [_value_at(s["exposed"], s[metric], n) for s in series_list if s["exposed"][-1] >= n]
                by_exposure.append(statistics.fmean(values) if values else None)
            entry[f"{metric}_by_exposed"] = by_exposure
        entry["final_exposed"] = statistics.fmean(s["exposed"][-1] for s in series_list)
        result["systems"][system] = entry
    return result


def exposure_table(runs: list[dict[str, Any]], gold: dict[str, frozenset[str]]) -> dict[str, Any]:
    """Search recall (ever exposed) vs. selection recall (included), and screening precision."""
    groups: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for run in runs:
        g = gold[run["episode"]["review_id"]]
        series = run_series(run, g)
        included = set(run["predictions"])
        exposed_gold = series["exposure_recall"][-1]
        groups[(system_key(run), run["episode"]["budget"])].append(
            {
                "exposure_recall": exposed_gold,
                "recall": run["metrics"]["recall"],
                "screen_recall": (run["metrics"]["recall"] / exposed_gold) if exposed_gold else 0.0,
                "precision": run["metrics"]["precision"],
                "exposed": series["exposed"][-1],
                "included": len(included),
            }
        )
    table: dict[str, Any] = {}
    for (system, budget), rows in sorted(groups.items()):
        table[f"{system}|B{budget}"] = {
            key: _mean_ci([row[key] for row in rows]) for key in rows[0]
        } | {"n": len(rows)}
    return table


# -- reachability -----------------------------------------------------------------
def seeds_for(store: CorpusStore, review_id: str, seed_count: int = 5) -> tuple[str, ...]:
    review = store.review(review_id)
    allowed = allowed_ids_for(store, review)
    retriever = BM25Retriever(store.papers[i] for i in sorted(allowed))
    env = LiteratureSearchEnv(
        store, retriever, EpisodeSpec(review.id, review.question, review.eligibility_criteria, 40, "hybrid", seed_count)
    )
    return env.seed_ids


def reachability(store: CorpusStore, review_id: str, seeds: Iterable[str], max_hops: int = 4) -> dict[str, Any]:
    """Fraction of gold within h undirected citation hops of the seeds, inside the visible universe."""
    review = store.review(review_id)
    allowed = allowed_ids_for(store, review)
    distance: dict[str, int] = {}
    queue: deque[str] = deque()
    for seed in seeds:
        if seed in allowed:
            distance[seed] = 0
            queue.append(seed)
    while queue:
        node = queue.popleft()
        if distance[node] >= max_hops:
            continue
        paper = store.papers[node]
        for neighbor in (*paper.references, *paper.citers):
            if neighbor in allowed and neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    gold = [g for g in review.gold_ids if g in allowed]
    within = {h: sum(1 for g in gold if distance.get(g, max_hops + 1) <= h) / len(gold) for h in range(max_hops + 1)}
    degree = [len([n for n in (*store.papers[g].references, *store.papers[g].citers) if n in allowed]) for g in gold]
    return {
        "universe": len(allowed),
        "gold": len(gold),
        "reachable_within": within,
        "reachable_nodes_within_max": len(distance),
        "gold_isolated": sum(1 for d in degree if d == 0) / len(gold),
        "gold_mean_degree": statistics.fmean(degree) if degree else 0.0,
        "gold_distance": {g: distance.get(g) for g in sorted(gold)},
    }


# -- strategy (Q3) ----------------------------------------------------------------
def strategy_profile(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[f"{system_key(run)}|B{run['episode']['budget']}"].append(run)
    for key, group in sorted(groups.items()):
        spend: Counter[str] = Counter()
        calls: Counter[str] = Counter()
        queries: list[int] = []
        first_switch: list[int] = []
        stops_early = 0
        for run in group:
            seen_queries = set()
            kinds = []
            for event in run["trace"]:
                kind = event["action"].get("kind")
                calls[kind] += 1
                spend[kind] += event["charged_cost"]
                if kind == "search":
                    seen_queries.add((event["action"].get("query") or "").strip().lower())
                if event["charged_cost"] > 0 and kind != "seed":
                    kinds.append(kind)
            queries.append(len(seen_queries))
            switch = next((i for i, k in enumerate(kinds) if k.startswith("expand")), None)
            if switch is not None:
                first_switch.append(switch)
            final_budget = run["trace"][-1]["budget_after"] if run["trace"] else 0
            stops_early += int(final_budget > 0)
        total = sum(spend.values()) or 1
        out[key] = {
            "n": len(group),
            "budget_share": {k: v / total for k, v in sorted(spend.items()) if v},
            "calls_per_episode": {k: v / len(group) for k, v in sorted(calls.items())},
            "unique_queries": statistics.fmean(queries) if queries else 0.0,
            "first_expand_step": statistics.fmean(first_switch) if first_switch else None,
            "stop_early_rate": stops_early / len(group),
        }
    return out


# -- failure analysis -------------------------------------------------------------
def miss_analysis(
    runs: list[dict[str, Any]], store: CorpusStore, *, reach: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Classify every missed gold study: never exposed (and why) vs. exposed but not included."""
    rows = []
    rank_cache: dict[str, dict[str, int]] = {}
    for run in runs:
        review_id = run["episode"]["review_id"]
        review = store.review(review_id)
        if review_id not in rank_cache:
            allowed = allowed_ids_for(store, review)
            ranked = BM25Retriever(store.papers[i] for i in sorted(allowed)).search(review.question, len(allowed), allowed)
            rank_cache[review_id] = {paper_id: index + 1 for index, paper_id in enumerate(ranked)}
        exposed = {i for event in run["trace"] for i in (event.get("result_ids") or [])}
        included = set(run["predictions"])
        for gold_id in sorted(review.gold_ids - included):
            paper = store.papers[gold_id]
            distance = None if reach is None else reach.get(review_id, {}).get("gold_distance", {}).get(gold_id)
            rows.append(
                {
                    "review": review_id,
                    "system": system_key(run),
                    "budget": run["episode"]["budget"],
                    "gold": gold_id,
                    "exposed": gold_id in exposed,
                    "has_abstract": bool(paper.abstract),
                    "question_bm25_rank": rank_cache[review_id].get(gold_id),
                    "seed_distance": distance,
                }
            )
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = f"{row['system']}|B{row['budget']}"
        if row["exposed"]:
            reason = "exposed_not_included"
        elif row["question_bm25_rank"] is None:
            reason = "no_lexical_overlap"
        elif row["seed_distance"] is None:
            reason = "unreachable_graph_and_rank>100" if row["question_bm25_rank"] > 100 else "unreached_rank<=100"
        else:
            reason = "not_reached_in_budget"
        summary[key][reason] += 1
        summary[key]["no_abstract" if not row["has_abstract"] else "has_abstract"] += 1
    return {"summary": {k: dict(v) for k, v in sorted(summary.items())}, "rows": rows}


# -- paired comparisons -----------------------------------------------------------
def paired(runs: list[dict[str, Any]], a: str, b: str, *, budget: int, metric: str = "recall") -> dict[str, Any]:
    """Mean per-review difference a - b (averaging repeats), with bootstrap CI over reviews."""
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        if run["episode"]["budget"] == budget and system_key(run) in {a, b}:
            values[run["episode"]["review_id"]][system_key(run)].append(run["metrics"][metric])
    diffs = [
        statistics.fmean(v[a]) - statistics.fmean(v[b]) for v in values.values() if v.get(a) and v.get(b)
    ]
    mean, lo, hi = _mean_ci(diffs)
    return {"a": a, "b": b, "budget": budget, "metric": metric, "n_reviews": len(diffs),
            "mean_diff": mean, "ci95": [lo, hi], "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs)}
