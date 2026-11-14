"""Reproduce the live-web recall tables offline from the shipped records (stdlib only).

    python scripts/reproduce_web.py [results/web]

Prints mean recall per system and the paired v2-vs-original comparisons with 95% bootstrap
CIs over reviews (seeded; recall is exact, CI endpoints can differ from analysis/webfp/web_robust.md in the third decimal because the resampling order differs).
"""
import json, random, statistics, sys
from collections import defaultdict
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/web")
gold = {k: set(v) for k, v in json.loads((d / "gold.json").read_text()).items()}
rows = [json.loads(l) for f in sorted(d.glob("*.jsonl")) for l in open(f)]
rows = [r for r in rows if r["repeat"] == 0]
recall = lambda r: len(set(r["predictions"]) & gold[r["review_id"]]) / len(gold[r["review_id"]])

groups = defaultdict(dict)
for r in rows:
    groups[(r["split"], r["agent"], r["mode"], r["budget"], r["env"])][r["review_id"]] = recall(r)
print("| split | system | mode | B | env | reviews | recall |\n|---|---|---|---|---|---|---|")
for k in sorted(groups):
    print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {len(groups[k])} | {statistics.fmean(groups[k].values()):.3f} |")

rng = random.Random(20261113)
print("\n| v2 - original | reviews | mean diff | 95% bootstrap CI | up/down |\n|---|---|---|---|---|")
for k in sorted(groups):
    if k[4] != "v2" or (k[:4] + ("original",)) not in groups:
        continue
    a, b = groups[k[:4] + ("original",)], groups[k]
    diffs = [b[r] - a[r] for r in sorted(set(a) & set(b))]
    boots = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
    print(f"| {k[0]} {k[1]} {k[2]} B={k[3]} | {len(diffs)} | {statistics.fmean(diffs):+.3f} | [{boots[250]:+.3f}, {boots[9749]:+.3f}] | "
          f"{sum(x > 0 for x in diffs)}/{sum(x < 0 for x in diffs)} |")
