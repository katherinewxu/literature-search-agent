"""Figures for docs/REPORT.md. Usage: python scripts/report_figures.py OUT_DIR

Reads runs under /scratch/.../runs (FarmShare layout). Every series is direct-labeled
(yellow and aqua are below 3:1 contrast on the surface, so color never carries identity alone).
"""

from __future__ import annotations

import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from robustness import ROOT, boot, load_runs  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SONNET, HAIKU, QWEN, TRAINED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BASE1, BASE2 = "#52514e", "#a3a29c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


_RUNS: dict[str, list] = {}


def cells(*dirs):
    g = defaultdict(dict)
    for d in dirs:
        if d not in _RUNS:
            _RUNS[d] = load_runs(d)
    for r in (r for d in dirs for r in _RUNS[d]):
        if r["repeat"] == 0:
            g[(r["agent"], r["mode"], r["budget"])][r["review"]] = r
    return g


def mean(g, key, metric="recall"):
    c = g.get(key)
    return st.mean(r[metric] for r in c.values()) if c else None


def save(fig, out, name):
    fig.savefig(out / name, dpi=220, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_budget(out):
    """Recall vs budget, hybrid agents vs include-all baselines, closed and open."""
    conds = [
        ("Closed pool (reviews' own candidates)", ["synergy-v3-api-r4", "synergy-v3-baselines-r3", "synergy-v3-dense-r3", "synergy-v3-small-trained-r4", "synergy-v3-snowball-tuned-r3"]),
        ("Open graph (+ 1-hop citation neighbours)", ["synergy-v3-open-api-r4", "synergy-v3-open-baselines-r3", "synergy-v3-open-dense-r3", "synergy-v3-open-small-r4", "synergy-v3-open-snowball-tuned-r3"]),
    ]
    series = [
        ("sonnet-5.5", "hybrid", "Sonnet 5.5", SONNET, "-", "o"),
        ("haiku-5.5", "hybrid", "Haiku 5.5", HAIKU, "-", "o"),
        ("qwen3-1.7b-grpo", "hybrid", "Qwen3-1.7B trained", TRAINED, "-", "D"),
        ("qwen3-32b", "hybrid", "Qwen3-32B", QWEN, "-", "o"),
        ("specter2", "keyword", "SPECTER2 (include all)", BASE1, "--", "s"),
        ("bm25", "keyword", "BM25 (include all)", BASE2, "--", "s"),
        ("snowball-tuned", "hybrid", "Snowball (scripted)", BASE2, ":", "v"),
    ]
    budgets = [10, 20, 40]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True, facecolor=SURFACE)
    for ax, (title, dirs) in zip(axes, conds):
        g = cells(*dirs)
        style(ax)
        ends = []
        for agent, mode, label, color, ls, marker in series:
            ys = [mean(g, (agent, mode, b)) for b in budgets]
            if any(y is None for y in ys):
                continue
            ax.plot(budgets, ys, color=color, linestyle=ls, linewidth=2, marker=marker, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            ends.append([ys[-1], label])
        # direct labels at the right end, nudged apart
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.038)
        for y, label in ends:
            ax.text(41.5, y, label, fontsize=7.5, color=INK, va="center")
        ax.set_xticks(budgets, [f"B={b}" for b in budgets])
        ax.set_xlim(8, 40)
        ax.set_ylim(0, 0.8)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("recall (mean over 8 reviews)", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(wspace=0.55)
    save(fig, out, "fig1_recall_vs_budget.png")


def fig_precision(out):
    """Recall vs precision at B=40, closed pool: screening is where LLMs differ from retrieval."""
    g = cells("synergy-v3-api-r4", "synergy-v3-baselines-r3", "synergy-v3-dense-r3", "synergy-v3-small-trained-r4",
              "synergy-v3-api-dense-r5")
    pts = [
        ("sonnet-5.5", "hybrid", "Sonnet 5.5", SONNET),
        ("sonnet-5.5-dense", "hybrid", "Sonnet 5.5, SPECTER2 tool", SONNET),
        ("haiku-5.5", "hybrid", "Haiku 5.5", HAIKU),
        ("qwen3-32b", "hybrid", "Qwen3-32B", QWEN),
        ("qwen3-1.7b-grpo", "hybrid", "Qwen3-1.7B trained", TRAINED),
        ("specter2", "keyword", "SPECTER2", BASE1),
        ("bm25", "keyword", "BM25", BASE2),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.8), facecolor=SURFACE)
    style(ax)
    ax.grid(True, color=GRID, linewidth=0.8)
    for agent, mode, label, color in pts:
        c = g.get((agent, mode, 40))
        if not c:
            continue
        rec = st.mean(r["recall"] for r in c.values())
        prec = st.mean(r["precision"] for r in c.values())
        hollow = "dense" in agent
        ax.scatter([prec], [rec], s=70, color=SURFACE if hollow else color, edgecolor=color if hollow else SURFACE,
                   linewidth=1.8 if hollow else 1.5, zorder=3)
        ax.annotate(label, (prec, rec), xytext=(7, -3 if not hollow else -11), textcoords="offset points",
                    fontsize=7.5, color=INK)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0.2, 0.75)
    ax.set_xlabel("precision of the included set", fontsize=8.5, color=MUTED)
    ax.set_ylabel("recall", fontsize=8.5, color=MUTED)
    ax.set_title("Closed pool, B=40: same recall band, very different precision", fontsize=9, color=INK, loc="left")
    save(fig, out, "fig2_recall_precision.png")


def fig_forest(out):
    """Key paired differences with 95% bootstrap CIs across evaluation sets."""
    closed = cells("synergy-v3-api-r4", "synergy-v3-baselines-r3", "synergy-v3-dense-r3", "synergy-v3-small-trained-r4")
    life40 = cells("ext-lifesci40-r5")
    transfer = cells("ext-transfer-api-r4", "ext-transfer-baselines-r3", "ext-transfer-small-r4")
    pooled = defaultdict(dict)
    for gg in (closed, life40):
        for k, v in gg.items():
            if k[2] == 40:
                pooled[k].update(v)

    def diff(g, a, b, budget=40):
        A, B = g.get((*a, budget), {}), g.get((*b, budget), {})
        ks = sorted(set(A) & set(B))
        return boot([A[k]["recall"] - B[k]["recall"] for k in ks]), len(ks)

    S, H, BM, Q32, GR = ("sonnet-5.5", "hybrid"), ("haiku-5.5", "hybrid"), ("bm25", "keyword"), ("qwen3-32b", "hybrid"), ("qwen3-1.7b-grpo", "hybrid")
    rows = [
        ("Sonnet − BM25", [("life-science, 8", closed, S, BM), ("life-science, 48", pooled, S, BM), ("transfer, 24", transfer, S, BM)], SONNET),
        ("Haiku − BM25", [("life-science, 8", closed, H, BM), ("life-science, 48", pooled, H, BM), ("transfer, 24", transfer, H, BM)], HAIKU),
        ("Sonnet − Haiku", [("life-science, 8", closed, S, H), ("life-science, 48", pooled, S, H), ("transfer, 24", transfer, S, H)], SONNET),
        ("Trained 1.7B − Qwen3-32B", [("life-science, 8", closed, GR, Q32), ("transfer, 24", transfer, GR, Q32)], TRAINED),
        ("Trained 1.7B − Haiku", [("life-science, 8", closed, GR, H), ("transfer, 24", transfer, GR, H)], TRAINED),
    ]
    flat = []
    for name, sets, color in rows:
        for set_name, g, a, b in sets:
            (m, lo, hi), n = diff(g, a, b)
            flat.append((name, set_name, m, lo, hi, color))
    fig, ax = plt.subplots(figsize=(6.6, 0.3 * len(flat) + 1.0), facecolor=SURFACE)
    style(ax)
    ax.grid(True, color=GRID, linewidth=0.8, axis="x")
    ax.grid(False, axis="y")
    y = 0
    ticks, labels, prev = [], [], None
    for name, set_name, m, lo, hi, color in flat:
        if prev is not None and name != prev:
            y += 0.6
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.scatter([m], [y], s=36, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.text(0.44, y, f"{m:+.3f}", fontsize=7, color=MUTED, va="center", ha="right")
        ticks.append(y)
        labels.append(f"{name}  ·  {set_name}" if name != prev else set_name)
        prev = name
        y += 1
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(ticks, labels, fontsize=7.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.4, 0.45)
    ax.set_xlabel("difference in recall at B=40 (paired, 95% bootstrap CI over reviews)", fontsize=8, color=MUTED)
    ax.set_title("Which gaps hold up as the evaluation set grows", fontsize=9, color=INK, loc="left")
    save(fig, out, "fig3_paired_differences.png")


def fig_training(out):
    """Prompted -> SFT -> GRPO for both seeds, three conditions, B=40 hybrid."""
    conds = [
        ("Closed pool (8)", ["synergy-v3-small-trained-r4", "synergy-v3-small-trained-s2-r4", "synergy-v3-qwen3-1.7b-r4", "synergy-v3-api-r4"]),
        ("Open graph (8)", ["synergy-v3-open-small-r4", "synergy-v3-open-small-s2-r4", "synergy-v3-open-api-r4"]),
        ("Transfer (24)", ["ext-transfer-small-r4", "ext-transfer-small-s2-r4", "ext-transfer-api-r4"]),
    ]
    stages = [("qwen3-1.7b", "prompted"), ("qwen3-1.7b-sft", "SFT"), ("qwen3-1.7b-grpo", "SFT+GRPO")]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.3), sharey=True, facecolor=SURFACE)
    for ax, (title, dirs) in zip(axes, conds):
        g = cells(*dirs)
        style(ax)
        ends = []
        for seed, suffix, ls, mk in (("seed 1", "", "-", "D"), ("seed 2", "-s2", "--", "s")):
            xs, ys = [], []
            for i, (agent, _) in enumerate(stages):
                key = (agent + (suffix if i else ""), "hybrid", 40)
                v = mean(g, key)
                if v is not None:
                    xs.append(i)
                    ys.append(v)
            ax.plot(xs, ys, color=TRAINED, linestyle=ls, linewidth=2, marker=mk, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            ends.append([ys[-1], seed])
        ends.sort()
        if ends[1][0] - ends[0][0] < 0.035:
            mid = (ends[0][0] + ends[1][0]) / 2
            ends[0][0], ends[1][0] = mid - 0.0175, mid + 0.0175
        for y, seed in ends:
            ax.text(2.12, y, seed, fontsize=7.5, color=INK, va="center")
        for agent, label, color in (("haiku-5.5", "Haiku (teacher)", HAIKU), ("qwen3-32b", "Qwen3-32B", QWEN)):
            v = mean(g, (agent, "hybrid", 40))
            if v is not None:
                ax.axhline(v, color=color, linewidth=1.5, linestyle=":")
                ax.text(-0.05, v + 0.012, label, fontsize=7, color=INK)
        ax.set_xticks(range(3), [s for _, s in stages])
        ax.set_xlim(-0.1, 2.6)
        ax.set_ylim(0, 0.7)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("recall at B=40 (hybrid)", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(wspace=0.12)
    save(fig, out, "fig4_training.png")


def fig_web(out):
    """Pool vs open graph vs live OpenAlex, B=40 hybrid (reviews present in all conditions)."""
    web_dir = ROOT / "runs" / "synergy-v3-web-r5"
    if not any(web_dir.glob("*.json")):
        print("no web runs yet")
        return
    conds = [
        ("closed pool", ["synergy-v3-api-r4", "synergy-v3-baselines-r3"]),
        ("open graph", ["synergy-v3-open-api-r4", "synergy-v3-open-baselines-r3"]),
        ("live OpenAlex", ["synergy-v3-web-r5"]),
    ]
    gs = [cells(*d) for _, d in conds]
    series = [("sonnet-5.5", "hybrid", "Sonnet 5.5", SONNET), ("haiku-5.5", "hybrid", "Haiku 5.5", HAIKU),
              ("bm25", "keyword", "BM25 / OpenAlex search (include all)", BASE2)]
    web_alias = {"bm25": "openalex-search"}
    reviews = set(gs[2].get(("sonnet-5.5", "hybrid", 40), {}))
    for agent, *_ in series:
        reviews &= set(gs[2].get((web_alias.get(agent, agent), "keyword" if agent == "bm25" else "hybrid", 40), {}))
    fig, ax = plt.subplots(figsize=(6.0, 3.4), facecolor=SURFACE)
    style(ax)
    width = 0.26
    for j, (agent, mode, label, color) in enumerate(series):
        ys = []
        for i, g in enumerate(gs):
            a = web_alias.get(agent, agent) if i == 2 else agent
            c = g.get((a, mode, 40), {})
            ys.append(st.mean(c[k]["recall"] for k in reviews if k in c) if reviews else 0)
        xs = [i + (j - 1) * width for i in range(3)]
        ax.bar(xs, ys, width=width - 0.03, color=color, edgecolor=SURFACE, linewidth=1)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.012, f"{y:.2f}", ha="center", fontsize=7, color=INK)
        ax.bar([0], [0], color=color, label=label)
    ax.set_xticks(range(3), [c for c, _ in conds])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel(f"recall at B=40 (same {len(reviews)} reviews)", fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("The search universe sets the ceiling", fontsize=9, color=INK, loc="left")
    save(fig, out, "fig5_web.png")


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    fig_budget(out)
    fig_precision(out)
    fig_forest(out)
    fig_training(out)
    fig_web(out)


if __name__ == "__main__":
    main()
