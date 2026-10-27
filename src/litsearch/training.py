from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ValueError("training config must be JSON so cluster runs need no YAML dependency")
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(kind: str, config: dict[str, Any]) -> list[str]:
    """Training gate: passing leakage audit, and >= N valid teacher episodes for SFT."""
    failures = []
    audit_path = Path(config["audit_report"])
    if not audit_path.exists() or not json.loads(audit_path.read_text())["ok"]:
        failures.append("data audit is missing or failed")
    if kind == "sft":
        trajectory_path = Path(config["trajectories"])
        stats_path = trajectory_path.with_suffix(".stats.json")
        if not trajectory_path.exists() or not stats_path.exists():
            failures.append("trajectory file or its stats are missing")
        else:
            episodes = json.loads(stats_path.read_text())["episodes_available"]
            if episodes < int(config.get("minimum_trajectories", 200)):
                failures.append(f"only {episodes} valid teacher episodes are available")
    if kind == "grpo" and config.get("init_adapter") and not Path(config["init_adapter"]).exists():
        failures.append("init_adapter (SFT checkpoint) is missing")
    return failures


def run_training(kind: str, config_path: Path) -> int:
    config = _load_config(config_path)
    failures = _gate(kind, config)
    report = {"kind": kind, "gate_passed": not failures, "failures": failures}
    gate_output = Path(config.get("gate_output", f"{kind}_gate.json"))
    gate_output.parent.mkdir(parents=True, exist_ok=True)
    gate_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        return 2
    if config.get("dry_run", False):
        return 0
    from . import rl

    if kind == "sft":
        rl.train_sft(config)
    else:
        rl.train_grpo(config, os.environ["MODEL_API_BASE"])
    return 0
