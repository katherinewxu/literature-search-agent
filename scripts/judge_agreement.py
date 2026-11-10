"""Two-judge agreement and circularity checks for the pooled web evaluation.

Usage: python scripts/judge_agreement.py JUDGE_A.jsonl JUDGE_B.jsonl OUT.md

* Calibration of each judge and of their AND (both say eligible) on gold / screened-excluded.
* Cohen's kappa between judges on every record, and on unscreened records alone.
* Pooled recall for each LLM agent with three pools of unscreened judged-eligible works:
    all      - every system's inclusions (as in web_judge.py)
    loao     - leave-one-agent-out: only works other systems included, so an agent gets no
               credit in the denominator for its own unique finds
    both     - works both judges call eligible
  Each pool member is weighted by that judge's Rogan-Gladen P(eligible | judged eligible).
Records without an abstract are dropped (the judges accept nearly all of them).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/scratch/users/kwx04/literature-search-agent")
SETS = [("eval (8 reviews)", "synergy-v3-web-r5", "synergy-v3"), ("transfer (24 reviews)", "ext-transfer-web-r5", "synergy-ext-transfer-v2")]
LLM_AGENTS = ("haiku-5.5", "sonnet-5.5")


def load(path):
    out = {}
    for j in map(json.loads, open(path)):
        if j["eligible"] is not None and j["has_abstract"]:
            out[(j["review_id"], j["work_id"])] = j
    return out


def calib(dec, kinds):
    """dec: key -> bool. Returns sens, spec, raw unscreened rate, RG prevalence, PPV."""
    r = lambda k: [dec[x] for x in dec if kinds[x] == k]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    sens, spec, q = mean(r("gold")), 1 - mean(r("excluded")), mean(r("unscreened"))
    p = min(1.0, max(0.0, (q + spec - 1) / (sens + spec - 1)))
    ppv = sens * p / (sens * p + (1 - spec) * (1 - p)) if p > 0 else 0.0
    return sens, spec, q, p, ppv


def kappa(a, b):
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    A, B = load(sys.argv[1]), load(sys.argv[2])
    out_md = Path(sys.argv[3])
    keys = sorted(set(A) & set(B))
    kinds = {k: A[k]["kind"] for k in keys}
    judges = {"A": {k: A[k]["eligible"] for k in keys}, "B": {k: B[k]["eligible"] for k in keys}}
    judges["A and B"] = {k: judges["A"][k] and judges["B"][k] for k in keys}
    L = ["# Two-judge agreement and circularity checks", "", f"A = {sys.argv[1].rsplit('/', 1)[-1]}, B = {sys.argv[2].rsplit('/', 1)[-1]}; "
         f"{len(keys)} records with an abstract judged by both.", "",
         "| judge | sensitivity | specificity | unscreened yes | eligible share (RG) | P(elig \\| yes) |", "|---|---|---|---|---|---|"]
    cal = {}
    for name, dec in judges.items():
        cal[name] = calib(dec, kinds)
        s, sp, q, p, ppv = cal[name]
        L.append(f"| {name} | {s:.2f} | {sp:.2f} | {q:.2f} | {p:.2f} | {ppv:.2f} |")
    ua = [judges["A"][k] for k in keys if kinds[k] == "unscreened"]
    ub = [judges["B"][k] for k in keys if kinds[k] == "unscreened"]
    L += ["", f"Cohen's kappa A vs B: all records {kappa([judges['A'][k] for k in keys], [judges['B'][k] for k in keys]):.2f}; "
          f"unscreened only {kappa(ua, ub):.2f}.", ""]

    for title, run_dir, data in SETS:
        gold = {}
        for line in open(ROOT / "data" / data / "reviews.jsonl"):
            r = json.loads(line)
            gold[r["id"]] = set(r["gold_ids"])
        runs = [json.loads(p.read_text()) for p in sorted((ROOT / "runs" / run_dir).glob("*.json"))]
        runs = [r for r in runs if r["matrix"].get("repeat", 0) == 0]
        L += [f"## {title}", "", "Recall of LLM agents: gold only, then gold + pool for each pool definition.", "",
              "| agent | mode | B | gold | all (A) | loao (A) | all (B) | both |", "|---|---|---|---|---|---|---|---|"]
        cells = defaultdict(list)
        for name, dec in (("A", judges["A"]), ("B", judges["B"]), ("both", judges["A and B"])):
            ppv = cal["A and B" if name == "both" else name][4]
            by_agent = defaultdict(lambda: defaultdict(set))  # review -> agent -> judged-eligible unscreened works
            for run in runs:
                rid = run["episode"]["review_id"]
                for w in run["predictions"]:
                    k = (rid, w)
                    if k in dec and kinds[k] == "unscreened" and dec[k]:
                        by_agent[rid][run["agent"]].add(w)
            for run in runs:
                if run["agent"] not in LLM_AGENTS:
                    continue
                ep = run["episode"]
                rid, g, pred = ep["review_id"], gold[ep["review_id"]], set(run["predictions"])
                tp = len(pred & g)
                pools = {"all": set().union(*by_agent[rid].values()) if by_agent[rid] else set()}
                pools["loao"] = set().union(*(v for a, v in by_agent[rid].items() if a != run["agent"])) if by_agent[rid] else set()
                for pname in (("all", "loao") if name == "A" else ("all",)):
                    pool = pools[pname]
                    rec = (tp + ppv * len(pred & pool)) / (len(g) + ppv * len(pool))
                    cells[(run["agent"], ep["mode"], ep["budget"])].append((f"{pname} ({name})" if name != "both" else "both", rec))
                cells[(run["agent"], ep["mode"], ep["budget"])].append(("gold", tp / len(g)))
        for key in sorted(cells):
            vals = defaultdict(list)
            for col, v in cells[key]:
                vals[col].append(v)
            m = {c: sum(v) / len(v) for c, v in vals.items()}
            L.append(f"| {key[0]} | {key[1]} | {key[2]} | {m['gold']:.3f} | {m['all (A)']:.3f} | {m['loao (A)']:.3f} | {m['all (B)']:.3f} | {m['both']:.3f} |")
        L.append("")
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
