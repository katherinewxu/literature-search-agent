"""Original vs v2 web environment, same agents/reviews/budgets. Usage: python scripts/env_v2_compare.py OUT.md

v2 = screened-out papers stay expandable in the prompt + seed fallback when the question finds no seeds.
Per (agent, mode, budget): recall, precision, budget spent, expands per run, runs that stopped on an
empty frontier with more than half the budget left. Means over the 8 eval reviews (repeat 0).
"""
import json, re, statistics, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path("/scratch/users/kwx04/literature-search-agent")
gold = {json.loads(l)["id"]: set(json.loads(l)["gold_ids"]) for l in open(ROOT / "data/synergy-v3/reviews.jsonl")}


def stats(run_dir):
    out = defaultdict(lambda: defaultdict(list))
    for p in sorted((ROOT / "runs" / run_dir).glob("*.json")):
        r = json.loads(p.read_text()); ep = r["episode"]
        if r["matrix"].get("repeat", 0) or r["agent"] not in ("haiku-5.5", "sonnet-5.5"):
            continue
        g, pred = gold[ep["review_id"]], set(r["predictions"])
        left = r["trace"][-1]["budget_after"] / ep["budget"] if r["trace"] else 1.0
        turns = r.get("agent_turns") or []
        empty = bool(turns) and "UNSCREENED PAPERS (0)" in turns[-1]["messages"][-1]["content"]
        d = out[(r["agent"], ep["mode"], ep["budget"])]
        d["review"].append(ep["review_id"])
        d["recall"].append(len(pred & g) / len(g))
        d["precision"].append(len(pred & g) / len(pred) if pred else 0.0)
        d["spent"].append(1 - left)
        d["expands"].append(sum(str((ev.get("action") or {}).get("kind", "")).startswith("expand") for ev in r["trace"]))
        d["dead_stop"].append(empty and left > 0.5)
        d["seeds"].append(len(r.get("seed_ids") or []))
    return out


old, new = stats("synergy-v3-web-r5"), stats("synergy-v3-web-v2")
L = ["# Original vs v2 web environment (eval, 8 reviews)", "",
     "| agent | mode | B | env | n | recall | precision | budget spent | expands/run | dead-end stops | zero-seed runs |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for key in sorted(new):
    for name, s in (("original", old), ("v2", new)):
        d = s.get(key)
        if not d:
            continue
        m = lambda k: statistics.fmean(d[k])
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | {name} | {len(d['recall'])} | {m('recall'):.3f} | {m('precision'):.3f} | "
                 f"{m('spent'):.2f} | {m('expands'):.1f} | {sum(d['dead_stop'])} | {sum(x == 0 for x in d['seeds'])} |")
    if key in old:
        a = dict(zip(old[key]["review"], old[key]["recall"])); b = dict(zip(new[key]["review"], new[key]["recall"]))
        diffs = [b[r] - a[r] for r in b if r in a]
        L.append(f"| | | | per-review recall change | {len(diffs)} | {statistics.fmean(diffs):+.3f} | up in {sum(x > 0 for x in diffs)}, down in {sum(x < 0 for x in diffs)} | | | | |")
Path(sys.argv[1]).write_text("\n".join(L) + "\n"); print("\n".join(L))
