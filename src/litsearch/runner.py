from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .agents import Agent
from .environment import LiteratureSearchEnv
from .metrics import compute_metrics
from .metrics import precision_recall
from .models import Action, ActionKind


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        # Cluster copies are rsynced without .git; the sync step writes GIT_SHA instead.
        stamp = Path(__file__).resolve().parents[2] / "GIT_SHA"
        return stamp.read_text(encoding="utf-8").strip() if stamp.exists() else None


def run_episode(
    env: LiteratureSearchEnv,
    agent: Agent,
    *,
    max_free_actions: int = 100,
) -> dict[str, Any]:
    started = time.time()
    free_actions = 0
    agent_errors: list[str] = []
    while not env.done:
        observation = env.observe()
        try:
            action = agent.act(observation)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            agent_errors.append(message)
            action = Action(ActionKind.NOOP)
            if len(agent_errors) >= 3:
                action = Action(ActionKind.STOP)
        result = env.step(action)
        if result.charged_cost == 0:
            free_actions += 1
            if free_actions >= max_free_actions and not env.done:
                env.step(Action(ActionKind.STOP))
        else:
            free_actions = 0

    metrics = compute_metrics(
        included=env.included,
        gold=env.review.gold_ids,
        trace=env.trace,
        budget=env.spec.budget,
        exposed=env.exposed,
        read_ids=env.read_ids,
        candidate_count=len(env.allowed_ids),
    )
    trace_rows = []
    for event in env.trace:
        row = event.to_dict()
        precision, recall = precision_recall(set(event.included_ids), env.review.gold_ids)
        row["precision_after"] = precision
        row["recall_after"] = recall
        trace_rows.append(row)
    return {
        "schema_version": 1,
        "episode": asdict(env.spec),
        "agent": agent.name,
        "seed_ids": list(env.seed_ids),
        "seed_cost": env.seed_cost,
        "trace": trace_rows,
        "predictions": sorted(env.included),
        "metrics": metrics.to_dict(),
        "runtime_seconds": time.time() - started,
        "agent_errors": agent_errors,
        "provenance": {
            "git_sha": _git_sha(),
            "python": platform.python_version(),
        },
    }


def write_run(run: dict[str, Any], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    return output

