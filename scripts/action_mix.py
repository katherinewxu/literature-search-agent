"""Action mix of LLM web agents: share of steps and budget per action kind. Usage: python scripts/action_mix.py OUT.md"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path("/scratch/users/kwx04/literature-search-agent")
L = ["# Action mix of web agents (repeat 0)", ""]
for title, run_dir in (("eval", "synergy-v3-web-r5"), ("transfer", "ext-transfer-web-r5")):
    steps, cost, runs = defaultdict(Counter), defaultdict(Counter), Counter()
    for p in sorted((ROOT / "runs" / run_dir).glob("*.json")):
        r = json.loads(p.read_text())
        if r["matrix"].get("repeat", 0):
            continue
        key = (r["agent"], r["episode"]["mode"], r["episode"]["budget"])
        runs[key] += 1
        for ev in r["trace"]:
            k = str((ev.get("action") or {}).get("kind", "?")).lower()
            steps[key][k] += 1
            cost[key][k] += ev.get("charged_cost", 0)
    kinds = ["search", "expand_refs", "expand_citers", "read", "include", "exclude", "stop", "noop"]
    L += [f"## {title}", "", "Mean per run: steps (budget spent) for each action kind.", "",
          "| agent | mode | B | " + " | ".join(kinds) + " |", "|---" * (3 + len(kinds)) + "|"]
    for key in sorted(runs):
        n = runs[key]
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(f"{steps[key][k] / n:.1f} ({cost[key][k] / n:.1f})" for k in kinds) + " |")
    L.append("")
Path(sys.argv[1]).write_text("\n".join(L) + "\n"); print("\n".join(L))
