"""Why do web agents stop with budget left? Usage: python scripts/stop_analysis.py OUT.md

For every LLM web run (repeat 0): budget left at the end, number of seeds, whether the final
prompt showed an empty frontier ("UNSCREENED PAPERS (0)"), and how many expands returned nothing.
"""
import json, re, statistics, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path("/scratch/users/kwx04/literature-search-agent")
L = ["# Why web agents stop early (repeat 0)", "",
     "left = mean share of budget unspent; empty = final turn saw no unscreened papers;",
     "stop_empty_left = stopped on an empty frontier with more than half the budget left; zero_seed = episode had no seeds;",
     "dead_expand = share of expands that returned no new paper.", ""]
for title, run_dir in (("eval", "synergy-v3-web-r5"), ("transfer", "ext-transfer-web-r5")):
    g = defaultdict(lambda: defaultdict(list))
    for p in sorted((ROOT / "runs" / run_dir).glob("*.json")):
        r = json.loads(p.read_text()); ep = r["episode"]
        if r["matrix"].get("repeat", 0) or r["agent"] not in ("haiku-5.5", "sonnet-5.5"):
            continue
        key = (r["agent"], ep["mode"], ep["budget"])
        left = r["trace"][-1]["budget_after"] / ep["budget"] if r["trace"] else 1.0
        turns = r.get("agent_turns") or []
        last_prompt = turns[-1]["messages"][-1]["content"] if turns else ""
        empty = bool(re.search(r"UNSCREENED PAPERS \(0\)", last_prompt))
        exps = [ev for ev in r["trace"] if str((ev.get("action") or {}).get("kind", "")).startswith("expand")]
        g[key]["left"].append(left)
        g[key]["empty"].append(empty)
        g[key]["stop_empty_left"].append(empty and left > 0.5)
        g[key]["zero_seed"].append(not r.get("seed_ids"))
        if exps:
            g[key]["dead_expand"].append(sum(not ev.get("result_ids") for ev in exps) / len(exps))
    L += [f"## {title}", "", "| agent | mode | B | n | left | empty | stop_empty_left | zero_seed | dead_expand |", "|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(g):
        d = g[key]; m = lambda k: statistics.fmean(d[k]) if d[k] else float("nan")
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(d['left'])} | {m('left'):.2f} | {m('empty'):.2f} | {m('stop_empty_left'):.2f} | {m('zero_seed'):.2f} | {m('dead_expand'):.2f} |")
    L.append("")
Path(sys.argv[1]).write_text("\n".join(L) + "\n"); print("\n".join(L))
