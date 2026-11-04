"""Tables for the dense-tool agent, the 40-review life-science eval, and seed 1 vs seed 2."""
from __future__ import annotations
import statistics as st, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from robustness import load_runs, boot, fmt

def cells(runs):
    g = defaultdict(dict)
    for r in runs:
        if r["repeat"] == 0:
            g[(r["agent"], r["mode"], r["budget"])][r["review"]] = r
    return g

def table(lines, g, budgets=(10, 20, 40)):
    lines += ["| system | mode | " + " | ".join(f"B={b}" for b in budgets) + " |", "|---|---|" + "---|" * len(budgets)]
    for agent, mode in sorted({(a, m) for a, m, _ in g}):
        vals = []
        for b in budgets:
            c = g.get((agent, mode, b))
            vals.append(f"{st.mean(r['recall'] for r in c.values()):.3f} (F1 {st.mean(r['f1'] for r in c.values()):.2f})" if c else "–")
        lines.append(f"| {agent} | {mode} | " + " | ".join(vals) + " |")
    lines.append("")

def pair(lines, g, a, b, budget=40, metric="recall"):
    A, B = g.get((*a, budget), {}), g.get((*b, budget), {})
    ks = sorted(set(A) & set(B))
    if not ks: return
    m, lo, hi = boot([A[k][metric] - B[k][metric] for k in ks])
    lines.append(f"- {a[0]} {a[1]} − {b[0]} {b[1]} (B={budget}, {metric}): {fmt(m)} [{fmt(lo)}, {fmt(hi)}] (n={len(ks)})")

def main():
    L = ["# Extra tables", ""]
    for title, dirs in [("closed pool", ["synergy-v3-api-r4", "synergy-v3-baselines-r3", "synergy-v3-dense-r3", "synergy-v3-api-dense-r5"]),
                        ("open graph", ["synergy-v3-open-api-r4", "synergy-v3-open-baselines-r3", "synergy-v3-open-dense-r3", "synergy-v3-open-api-dense-r5"])]:
        g = cells(load_runs(*dirs))
        L += [f"## Dense-tool agent: {title}", ""]
        table(L, {k: v for k, v in g.items() if k[0] in {"sonnet-5.5","haiku-5.5","sonnet-5.5-dense","haiku-5.5-dense","bm25","specter2"} and k[1] in {"hybrid","keyword"}})
        for b in (10, 20, 40):
            pair(L, g, ("haiku-5.5-dense","hybrid"), ("haiku-5.5","hybrid"), b)
            pair(L, g, ("haiku-5.5-dense","hybrid"), ("specter2","keyword"), b)
        for a, bb in [(("sonnet-5.5-dense","hybrid"),("sonnet-5.5","hybrid")), (("sonnet-5.5-dense","hybrid"),("specter2","keyword")),
                      (("sonnet-5.5-dense","hybrid"),("bm25","keyword"))]:
            pair(L, g, a, bb); pair(L, g, a, bb, metric="f1")
        L.append("")
    g = cells(load_runs("ext-lifesci40-r5"))
    L += ["## 40 extra life-science reviews (B=40; training reviews of the 1.7B model only)", ""]
    table(L, g, (40,))
    pair(L, g, ("sonnet-5.5","hybrid"), ("bm25","keyword")); pair(L, g, ("haiku-5.5","hybrid"), ("bm25","keyword"))
    pair(L, g, ("sonnet-5.5","hybrid"), ("haiku-5.5","hybrid")); pair(L, g, ("sonnet-5.5","hybrid"), ("bm25","keyword"), metric="f1")
    # pooled 48 life-science reviews
    g8 = cells(load_runs("synergy-v3-api-r4", "synergy-v3-baselines-r3"))
    pooled = defaultdict(dict)
    for gg in (g, g8):
        for k, v in gg.items():
            if k[2] == 40: pooled[k].update(v)
    L += ["", "Pooled 48 life-science reviews (8 eval + 40):", ""]
    pair(L, pooled, ("sonnet-5.5","hybrid"), ("bm25","keyword")); pair(L, pooled, ("haiku-5.5","hybrid"), ("bm25","keyword"))
    pair(L, pooled, ("sonnet-5.5","hybrid"), ("haiku-5.5","hybrid"))
    L.append("")
    for title, d1, d2 in [("closed pool", "synergy-v3-small-trained-r4", "synergy-v3-small-trained-s2-r4"),
                          ("open graph", "synergy-v3-open-small-r4", "synergy-v3-open-small-s2-r4"),
                          ("transfer", "ext-transfer-small-r4", "ext-transfer-small-s2-r4")]:
        g = cells(load_runs(d1, d2))
        L += [f"## Training seed 1 vs seed 2: {title}", ""]
        table(L, {k: v for k, v in g.items() if k[1] == "hybrid"})
        for b in (20, 40):
            pair(L, g, ("qwen3-1.7b-grpo","hybrid"), ("qwen3-1.7b-sft","hybrid"), b)
            pair(L, g, ("qwen3-1.7b-grpo-s2","hybrid"), ("qwen3-1.7b-sft-s2","hybrid"), b)
            pair(L, g, ("qwen3-1.7b-grpo-s2","hybrid"), ("qwen3-1.7b-grpo","hybrid"), b)
        L.append("")
    Path(sys.argv[1]).write_text("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
