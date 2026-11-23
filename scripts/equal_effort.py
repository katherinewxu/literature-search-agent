"""Equal-effort comparison: is the no-loop baseline better, or does it just include more papers?

Usage: python scripts/equal_effort.py OUT.md [OUT_CURVES.json]
For each pair (agent A, no-loop S) and review, truncates S's inclusions (in the order it made
them: query round-robin, search rank within each batch) to A's number of included papers k_A,
then compares recall at equal inclusion count, paired over reviews. Also writes mean recall vs
number of included papers (k = 5, 10, ..., 200) for every system, for the curve figure.
"""
import json
import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from web_robust import GOLD, boot_ci, load, sign_test  # noqa: E402

PAIRS = [  # (label, split, agent cell, no-loop cell)
    ("live web, transfer, Haiku B=40", "transfer", ("ext-transfer-web-r5", "haiku-5.5", "hybrid", 40), ("ext-transfer-web-static", "haiku-5.5-static", "keyword", 40)),
    ("live web, eval, Haiku B=40", "eval", ("synergy-v3-web-r5", "haiku-5.5", "hybrid", 40), ("synergy-v3-web-static", "haiku-5.5-static", "keyword", 40)),
    ("live web, eval, Sonnet B=40", "eval", ("synergy-v3-web-r5", "sonnet-5.5", "hybrid", 40), ("synergy-v3-web-static", "sonnet-5.5-static", "keyword", 40)),
    ("closed, transfer, Haiku B=40", "transfer", ("ext-transfer-api-r4", "haiku-5.5", "hybrid", 40), ("ext-transfer-static", "haiku-5.5-static", "keyword", 40)),
    ("closed, eval, Haiku B=40", "eval", ("synergy-v3-api-r4", "haiku-5.5", "hybrid", 40), ("synergy-v3-static", "haiku-5.5-static", "keyword", 40)),
]
KS = list(range(5, 205, 5))


def include_order(run):
    order = []
    for ev in run["trace"]:
        a = ev.get("action") or {}
        if str(a.get("kind")) in ("include", "ActionKind.INCLUDE") and a.get("paper_id") and a["paper_id"] not in order:
            order.append(a["paper_id"])
    final = set(run["predictions"])
    return [p for p in order if p in final] + sorted(final - set(order))


def by_review(cell):
    return {r["episode"]["review_id"]: include_order(r) for r in load(*cell)}


def recall(ids, gold):
    return len(set(ids) & gold) / len(gold)


def main(out, curves_out=None):
    L = ["# Equal-effort comparison: recall at the agent's own number of included papers", "",
         "S@k_A = no-loop recall using only its first k_A inclusions (k_A = the agent's included count, per review).",
         "diff = S@k_A - agent; paired over reviews; 95% bootstrap CI; exact sign test.", "",
         "| pair | n | mean k_A | agent recall | no-loop recall (all) | no-loop recall @k_A | diff @k_A | 95% CI | up/down | sign p |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    curves = {}
    for label, split, a_cell, s_cell in PAIRS:
        A, S = by_review(a_cell), by_review(s_cell)
        keys = sorted(set(A) & set(S))
        if not keys:
            L.append(f"| {label} | 0 | | | | | | | | |")
            continue
        g = GOLD[split]
        ra = [recall(A[k], g[k]) for k in keys]
        rs_all = [recall(S[k], g[k]) for k in keys]
        rs_k = [recall(S[k][: len(A[k])], g[k]) for k in keys]
        d = [s - a for s, a in zip(rs_k, ra)]
        lo, hi = boot_ci(d)
        L.append(f"| {label} | {len(keys)} | {statistics.fmean(len(A[k]) for k in keys):.0f} | {statistics.fmean(ra):.3f} | "
                 f"{statistics.fmean(rs_all):.3f} | {statistics.fmean(rs_k):.3f} | {statistics.fmean(d):+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                 f"{sum(x > 0 for x in d)}/{sum(x < 0 for x in d)} | {sign_test(d):.3f} |")
        for name, runs in (("agent", A), ("no-loop", S)):
            # Recall at each inclusion count k; a system that stopped before k keeps its final recall.
            curves[f"{label} | {name}"] = {"k": KS, "recall": [statistics.fmean(recall(runs[r][:k], g[r]) for r in keys) for k in KS],
                                           "mean_included": statistics.fmean(len(runs[r]) for r in keys)}
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    if curves_out:
        open(curves_out, "w").write(json.dumps(curves, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
