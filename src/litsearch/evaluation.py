from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = ("recall", "precision", "f1", "aurc", "unique_exposed", "unique_read")


def _bootstrap_ci(values: list[float], seed: int = 329, samples: int = 2000) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [rng.choice(values) for _ in values]
        estimates.append(statistics.fmean(draw))
    estimates.sort()
    return [estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]]


def evaluate_runs(paths: list[Path]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        if "episode" not in run or "metrics" not in run:
            continue
        episode = run["episode"]
        key = f"{run['agent']}|{episode['mode']}|B{episode['budget']}"
        groups[key].append(run)
    summary: dict[str, Any] = {"run_count": sum(len(v) for v in groups.values()), "groups": {}}
    for key, runs in sorted(groups.items()):
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            values = [float(run["metrics"][metric]) for run in runs]
            metrics[metric] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "ci95": _bootstrap_ci(values),
            }
        summary["groups"][key] = {
            "n": len(runs),
            "reviews": sorted({run["episode"]["review_id"] for run in runs}),
            "metrics": metrics,
        }
    return summary

