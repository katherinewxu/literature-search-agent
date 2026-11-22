"""Pre-specified primary comparisons P1-P4 (docs/final/paper_plan.md), Holm-corrected as one family.

Usage: python scripts/primary_tests.py OUT.md
Recall at B=40, paired over the 24 held-out transfer reviews, repeat 0. Diff = B - A.
Test: exact two-sided sign test (Holm over the 4 p-values) + 95% bootstrap CI of the mean diff.
"""
import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from web_robust import boot_ci, load, recall_by_review, sign_test  # noqa: E402

PRIMARY = [  # (id, label, A=(dir, agent, mode, budget), B=...)
    ("P1", "live web: Haiku agent loop - no-loop Haiku query list",
     ("ext-transfer-web-static", "haiku-5.5-static", "keyword", 40), ("ext-transfer-web-r5", "haiku-5.5", "hybrid", 40)),
    ("P2", "closed graph: trained Qwen3-1.7B (SFT+GRPO) - prompted Qwen3-32B",
     ("ext-transfer-api-r4", "qwen3-32b", "hybrid", 40), ("ext-transfer-small-r4", "qwen3-1.7b-grpo", "hybrid", 40)),
    ("P3", "closed graph: Haiku agent - BM25 baseline",
     ("ext-transfer-baselines-r3", "bm25", "keyword", 40), ("ext-transfer-api-r4", "haiku-5.5", "hybrid", 40)),
    ("P4", "live web: v2 environment - original, Haiku agent",
     ("ext-transfer-web-r5", "haiku-5.5", "hybrid", 40), ("ext-transfer-web-v2", "haiku-5.5", "hybrid", 40)),
]


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main(out):
    rows = []
    for pid, label, a, b in PRIMARY:
        ra, rb = recall_by_review(load(*a), "transfer"), recall_by_review(load(*b), "transfer")
        keys = sorted(set(ra) & set(rb))
        d = [rb[k] - ra[k] for k in keys]
        lo, hi = boot_ci(d)
        rows.append([pid, label, len(d), statistics.fmean(ra[k] for k in keys), statistics.fmean(rb[k] for k in keys),
                     statistics.fmean(d), lo, hi, sum(x > 0 for x in d), sum(x < 0 for x in d), sign_test(d)])
    for row, p_adj in zip(rows, holm([r[-1] for r in rows])):
        row.append(p_adj)
    L = ["# Primary comparisons (pre-specified; Holm over P1-P4)", "",
         "Recall at B=40, held-out transfer split, paired over reviews; diff = B - A.", "",
         "| id | comparison (B - A) | n | A | B | diff | 95% CI | up/down | sign p | Holm p |", "|---|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:+.3f} | [{r[6]:+.3f}, {r[7]:+.3f}] | {r[8]}/{r[9]} | {r[10]:.4f} | {r[11]:.4f} |" for r in rows]
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
