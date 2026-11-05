"""Evaluate end-of-episode recall estimators. Usage: python scripts/recall_est.py CONFIG OUT_DIR

CONFIG: {"screener": {<llm agent config as in configs/llm/*.json>},
         "sets": [{"name": ..., "runs": "<run dir>", "agents": [..], "mode": "hybrid", "budget": 40}],
         "n_unexposed": 80, "n_excluded": 30}
"""

from __future__ import annotations

import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from litsearch.matrix import _backend
from litsearch.recall_estimation import llm_screener, oracle_screener, sample_estimate, saturation_estimate
from litsearch.store import CorpusStore

ROOT = Path("/scratch/users/kwx04/literature-search-agent")


def corr(xs, ys):
    if len(xs) < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    backend = _backend(config["screener"])
    rows = []
    for spec in config["sets"]:
        runs = []
        for path in sorted((ROOT / "runs" / spec["runs"]).glob("*.json")):
            run = json.loads(path.read_text())
            ep = run["episode"]
            if run["agent"] in spec["agents"] and ep["mode"] == spec["mode"] and ep["budget"] == spec["budget"] \
                    and run["matrix"].get("repeat", 0) == 0:
                runs.append(run)
        store = CorpusStore.load(runs[0]["matrix"]["data_dir"])
        oracle, llm = oracle_screener(store), llm_screener(store, backend)
        kw = {"n_unexposed": int(config.get("n_unexposed", 80)), "n_excluded": int(config.get("n_excluded", 30))}

        def one(run):
            o = sample_estimate(run, store, oracle, **kw)
            m = sample_estimate(run, store, llm, **kw)
            return {
                "set": spec["name"], "agent": run["agent"], "review": run["episode"]["review_id"],
                "true_recall": o["true_recall"], "saturation": saturation_estimate(run),
                "oracle_hat": o["recall_hat"], "oracle_lower": o["recall_lower"],
                "llm_hat": m["recall_hat"], "llm_lower": m["recall_lower"],
                "sizes": m["sizes"], "llm_yes": m["screened_yes"], "true_missed": o["true_missed"],
                "strata": m["strata"],
            }

        with ThreadPoolExecutor(max_workers=8) as pool:
            rows += list(pool.map(one, runs))
        print(spec["name"], len(runs), flush=True)
    (out / "recall_estimates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    lines = ["# End-of-episode recall estimation", "",
             "| set | agent | n | true recall | estimator | mean estimate | MAE | corr with true | lower bound holds |",
             "|---|---|---|---|---|---|---|---|---|"]
    groups = {}
    for r in rows:
        groups.setdefault((r["set"], r["agent"]), []).append(r)
    for (name, agent), rs in groups.items():
        truth = [r["true_recall"] for r in rs]
        for est, lower in (("saturation", None), ("oracle_hat", "oracle_lower"), ("llm_hat", "llm_lower")):
            pairs = [(r[est], r["true_recall"], r[lower] if lower else None) for r in rs if r[est] is not None]
            if not pairs:
                continue
            e = [p[0] for p in pairs]
            t = [p[1] for p in pairs]
            mae = st.mean(abs(a - b) for a, b in zip(e, t))
            holds = "–" if lower is None else f"{sum(1 for p in pairs if p[2] is not None and p[1] >= p[2]) / len(pairs):.0%}"
            label = {"saturation": "yield curve", "oracle_hat": "sample, gold screener", "llm_hat": "sample, Haiku screener"}[est]
            lines.append(f"| {name} | {agent} | {len(pairs)} | {st.mean(t):.3f} | {label} | {st.mean(e):.3f} | {mae:.3f} | "
                         f"{corr(e, t):.2f} | {holds} |")
    lines.append("")
    lines.append(f"Screener usage: {json.dumps(backend.usage)}")
    (out / "recall_estimates.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
