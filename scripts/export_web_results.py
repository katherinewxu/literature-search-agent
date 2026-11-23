"""Export compact live-web run records for offline reproduction. Usage: python scripts/export_web_results.py OUTDIR

Writes OUTDIR/<run_dir>.jsonl (one line per episode: review, agent, mode, budget, env,
predictions, seed ids, per-step action kinds/costs) and OUTDIR/gold.json (public SYNERGY+
inclusion labels for the 32 reviews). Agent prompts/turns are dropped to keep the files small.
"""
import json, sys
from pathlib import Path
ROOT = Path("/scratch/users/kwx04/literature-search-agent")
DIRS = {"synergy-v3-web-r5": "eval", "synergy-v3-web-v2": "eval", "synergy-v3-web-snowball": "eval",
        "ext-transfer-web-r5": "transfer", "ext-transfer-web-v2": "transfer", "ext-transfer-web-citation-orig": "transfer",
        "ext-transfer-web-snowball": "transfer", "synergy-v3-web-static": "eval", "ext-transfer-web-static": "transfer",
        "synergy-v3-web-trained": "eval", "ext-transfer-web-trained": "transfer"}
DATA = {"eval": "synergy-v3", "transfer": "synergy-ext-transfer-v2"}
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
gold, reviews = {}, {s: set() for s in DATA}
for d, split in DIRS.items():
    if not any((ROOT / "runs" / d).glob("*.json")):
        continue
    with open(out / f"{d}.jsonl", "w") as fh:
        for p in sorted((ROOT / "runs" / d).glob("*.json")):
            r = json.loads(p.read_text()); ep = r["episode"]
            reviews[split].add(ep["review_id"])
            fh.write(json.dumps({"split": split, "review_id": ep["review_id"], "agent": r["agent"], "mode": ep["mode"],
                                 "budget": ep["budget"], "repeat": r["matrix"].get("repeat", 0), "env": "v2" if d.endswith("v2") else "original",
                                 "predictions": r["predictions"], "include_order": [str((e.get("action") or {}).get("paper_id")) for e in r["trace"] if str((e.get("action") or {}).get("kind", "")).endswith(("include", "INCLUDE")) and not str((e.get("action") or {}).get("kind", "")).endswith(("exclude", "EXCLUDE"))], "seed_ids": r.get("seed_ids") or [],
                                 "steps": [[str((e.get("action") or {}).get("kind", "")), e.get("charged_cost", 0)] for e in r["trace"]],
                                 "runtime_seconds": r.get("runtime_seconds")}, sort_keys=True) + "\n")
for split, data in DATA.items():
    for line in open(ROOT / "data" / data / "reviews.jsonl"):
        r = json.loads(line)
        if r["id"] in reviews[split]:
            gold[r["id"]] = sorted(r["gold_ids"])
(out / "gold.json").write_text(json.dumps(gold, indent=0, sort_keys=True) + "\n")
print({p.name: p.stat().st_size for p in out.iterdir()})
