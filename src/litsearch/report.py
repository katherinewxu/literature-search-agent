from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .evaluation import evaluate_runs


def _load_runs(run_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(run_dir.rglob("*.json")):
        if path.name.startswith(("summary", "manifest", "splits")):
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        if "episode" in run and "metrics" in run:
            runs.append(run)
    return runs


def _system(run: dict[str, Any]) -> str:
    return f"{run['agent']}/{run['episode']['mode']}"


def build_report(run_dir: Path) -> str:
    """Matched-budget comparison tables (Markdown) from a directory of episode traces."""
    runs = _load_runs(run_dir)
    summary = evaluate_runs(sorted(p for p in run_dir.rglob("*.json") if not p.name.startswith("summary")))
    budgets = sorted({run["episode"]["budget"] for run in runs})
    systems = sorted({_system(run) for run in runs})
    lines = [f"# Baseline report: `{run_dir}`", "", f"{len(runs)} episodes.", ""]

    lines += ["## Mean recall by budget (95% bootstrap CI over episodes)", ""]
    lines += ["| system | " + " | ".join(f"B={b}" for b in budgets) + " |"]
    lines += ["|---|" + "---|" * len(budgets)]
    for system in systems:
        cells = []
        for budget in budgets:
            group = summary["groups"].get(f"{system.replace('/', '|')}|B{budget}")
            if not group:
                cells.append("—")
                continue
            recall = group["metrics"]["recall"]
            ci = recall["ci95"]
            cells.append(
                f"{recall['mean']:.3f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
            )
        lines.append(f"| {system} | " + " | ".join(cells) + " |")

    for metric in ("precision", "aurc", "unique_exposed"):
        lines += ["", f"## Mean {metric} by budget", ""]
        lines += ["| system | " + " | ".join(f"B={b}" for b in budgets) + " |"]
        lines += ["|---|" + "---|" * len(budgets)]
        for system in systems:
            cells = []
            for budget in budgets:
                group = summary["groups"].get(f"{system.replace('/', '|')}|B{budget}")
                cells.append(f"{group['metrics'][metric]['mean']:.3f}" if group else "—")
            lines.append(f"| {system} | " + " | ".join(cells) + " |")

    top = max(budgets)
    per_review: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        if run["episode"]["budget"] == top:
            per_review[run["episode"]["review_id"]][_system(run)].append(run["metrics"]["recall"])
    lines += ["", f"## Per-review recall at B={top} (mean over repeats)", ""]
    lines += ["| review | gold | " + " | ".join(systems) + " |"]
    lines += ["|---|---|" + "---|" * len(systems)]
    gold = {run["episode"]["review_id"]: run["metrics"]["gold_count"] for run in runs}
    for review_id in sorted(per_review):
        cells = []
        for system in systems:
            values = per_review[review_id].get(system)
            cells.append(f"{sum(values) / len(values):.2f}" if values else "—")
        lines.append(f"| {review_id} | {gold[review_id]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
