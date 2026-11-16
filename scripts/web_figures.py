"""Figures and the main table for the live-web results, rebuilt offline from shipped records.

    python scripts/web_figures.py [results/web] [docs/figures]

Reads results/web/*.jsonl + gold.json (recall, precision, paired v2 comparisons) and two committed
analysis tables (analysis/webfp/web_miss_analysis.md for miss causes, analysis/webfp/web_robust.md
for cost). Writes:
  figW1_web_recall_budget.png  recall vs budget per system, eval and transfer
  figW2_miss_causes.png        why gold studies were missed (B=40), pooled over gold studies
  figW3_env_v2.png             v2 minus original environment, per review, eval vs held-out
  figW4_cost_recall.png        $ per episode vs recall (B=40)
  tables/web_main.tex          main table: recall [95% CI], precision, $/episode at B=40
Recall and precision are macro averages over reviews; CIs bootstrap over reviews (seeded).
"""
from __future__ import annotations

import json
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SONNET, HAIKU = "#2a78d6", "#eb6834"
BASE1, BASE2 = "#52514e", "#a3a29c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
rng = random.Random(20261115)

SYSTEMS = [  # (agent, mode, label, color, linestyle, marker)
    ("sonnet-5.5", "hybrid", "Sonnet 5.5, search + citations", SONNET, "-", "o"),
    ("sonnet-5.5", "citation", "Sonnet 5.5, citations only", SONNET, "--", "o"),
    ("haiku-5.5", "hybrid", "Haiku 5.5, search + citations", HAIKU, "-", "o"),
    ("haiku-5.5", "citation", "Haiku 5.5, citations only", HAIKU, "--", "o"),
    ("snowball-tuned", "hybrid", "Snowball (scripted), search + citations", BASE1, ":", "v"),
    ("snowball-tuned", "citation", "Snowball (scripted), citations only", BASE2, ":", "v"),
    ("openalex-search", "keyword", "OpenAlex search (include all)", BASE1, "-.", "s"),
]


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, axis=grid_axis)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def save(fig, out, name):
    fig.savefig(out / name, dpi=220, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / name)


def boot_ci(xs, n=10000):
    means = sorted(st.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def load(d):
    gold = {k: set(v) for k, v in json.loads((d / "gold.json").read_text()).items()}
    cells = defaultdict(dict)  # (split, agent, mode, budget, env) -> review -> (recall, precision)
    for f in sorted(d.glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["repeat"]:
                continue
            g, p = gold[r["review_id"]], set(r["predictions"])
            cells[(r["split"], r["agent"], r["mode"], r["budget"], r["env"])][r["review_id"]] = (
                len(p & g) / len(g), len(p & g) / len(p) if p else 0.0)
    return cells


def md_table(path, header_start):
    """Rows (as dicts) of every markdown table in `path` whose header starts with `header_start`."""
    rows, cols, section = [], None, None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            section = line[3:].split(" (")[0].strip()
        if line.startswith(header_start):
            cols = [c.strip() for c in line.strip("|").split("|")]
            continue
        if cols and line.startswith("|---"):
            continue
        if cols and line.startswith("|"):
            vals = [c.strip() for c in line.strip("|").split("|")]
            rows.append({"section": section, **dict(zip(cols, vals))})
        else:
            cols = None
    return rows


def fig_budget(cells, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True, facecolor=SURFACE)
    for ax, split, title in zip(axes, ("eval", "transfer"), ("Eval reviews (8)", "Transfer reviews (24)")):
        style(ax)
        for agent, mode, label, color, ls, mk in SYSTEMS:
            pts = [(b, st.fmean(v[0] for v in cells[(split, agent, mode, b, "original")].values()))
                   for b in (10, 20, 40) if cells.get((split, agent, mode, b, "original"))]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linestyle=ls, marker=mk, markersize=4, linewidth=1.6, label=label)
        ax.set_xticks([10, 20, 40])
        ax.set_xlabel("action budget B", color=MUTED)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("recall of the review's included studies", color=MUTED)
    handles = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    order = [s[2] for s in SYSTEMS if s[2] in handles]
    axes[1].legend([handles[l] for l in order], order, frameon=False, fontsize=7, handlelength=3.2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Live OpenAlex: LLM agents vs non-LLM baselines (original environment)", x=0.07, ha="left", fontsize=9.5, color=INK)
    save(fig, out, "figW1_web_recall_budget.png")


def fig_miss(out):
    rows = [r for r in md_table(REPO / "analysis/webfp/web_miss_analysis.md", "| agent | mode | B | n gold")
            if r["B"] == "40"]
    cats = [("found", ["found"], "#1baf7a"), ("seen, not included", ["rejected_read", "rejected_unread", "seen_undecided"], "#eda100"),
            ("one citation hop away", ["one_hop"], "#2a78d6"), ("found only by another system", ["other_system"], "#a3a29c"),
            ("never surfaced", ["never_surfaced"], "#52514e")]
    names = {"hybrid": "search + citations", "citation": "citations only"}
    labels = [f"{r['section']}: {r['agent'].replace('-5.5', ' 5.5').capitalize()}, {names[r['mode']]}" for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(rows) + 1.2), facecolor=SURFACE)
    style(ax, "x")
    for i, r in enumerate(rows):
        left = 0.0
        for cat, keys, color in cats:
            w = sum(float(r[k]) for k in keys)
            ax.barh(i, w, left=left, color=color, edgecolor=SURFACE, linewidth=1, label=cat if i == 0 else None)
            if w >= 0.06:
                ax.text(left + w / 2, i, f"{w:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if color in ("#2a78d6", "#52514e") else INK)
            left += w
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of gold studies (pooled over gold studies), B=40", color=MUTED)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    save(fig, out, "figW2_miss_causes.png")


def fig_env(cells, out):
    comps = [("eval", "haiku-5.5", "citation", 10), ("eval", "haiku-5.5", "citation", 20), ("eval", "haiku-5.5", "citation", 40),
             ("eval", "sonnet-5.5", "citation", 40), ("eval", "haiku-5.5", "hybrid", 40),
             ("transfer", "haiku-5.5", "citation", 40), ("transfer", "haiku-5.5", "hybrid", 20), ("transfer", "haiku-5.5", "hybrid", 40)]
    names = {"hybrid": "search + citations", "citation": "citations only"}
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=SURFACE)
    style(ax, "x")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    labels = []
    for i, (split, agent, mode, b) in enumerate(comps):
        a, v = cells.get((split, agent, mode, b, "original"), {}), cells.get((split, agent, mode, b, "v2"), {})
        keys = sorted(set(a) & set(v))
        d = [v[k][0] - a[k][0] for k in keys]
        color = SONNET if agent.startswith("sonnet") else HAIKU
        tag = "HELD-OUT" if split == "transfer" else "eval"
        labels.append(f"{tag}: {agent.replace('-5.5', ' 5.5').capitalize()}, {names[mode]}, B={b} (n={len(d)})")
        if not d:
            continue
        ax.scatter(d, [i + rng.uniform(-0.18, 0.18) for _ in d], s=9, color=color, alpha=0.45, linewidths=0)
        lo, hi = boot_ci(d)
        m = st.fmean(d)
        ax.plot([lo, hi], [i, i], color=INK, linewidth=1.6)
        ax.plot([m], [i], marker="D", color=INK, markersize=4.5)
        ax.text(max(hi, max(d)) + 0.01, i, f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=7, color=INK)
    ax.axhline(4.5, color=GRID, linewidth=1.2)
    ax.set_yticks(range(len(comps)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("recall, v2 minus original environment (dots: reviews; bar: mean and 95% bootstrap CI)", color=MUTED)
    ax.set_title("The environment fix helps on the reviews it was designed on, less on held-out reviews",
                 fontsize=9, color=INK, loc="left")
    save(fig, out, "figW3_env_v2.png")


def cost_rows():
    rows = md_table(REPO / "analysis/webfp/web_robust.md", "| split | system | mode | n | LLM calls/ep")
    return {(r["split"], r["system"], r["mode"]): float(r["$/episode"]) for r in rows}


def fig_cost(cells, out):
    cost = cost_rows()
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    style(ax, "both")
    floor = 3e-4
    for agent, mode, label, color, _, mk in SYSTEMS:
        for split, fill in (("eval", True), ("transfer", False)):
            c = cells.get((split, agent, mode, 40, "original"))
            if not c or (split, agent, mode) not in cost:
                continue
            x = max(cost[(split, agent, mode)], floor)
            y = st.fmean(v[0] for v in c.values())
            ax.scatter([x], [y], s=34, marker=mk, color=color if fill else SURFACE, edgecolors=color, linewidths=1.4, zorder=3)
            if cost[(split, agent, mode)] == 0:
                continue  # labelled once below
            short = label.replace(", search + citations", " hybrid").replace(", citations only", " citation").replace(" (include all)", "").replace(" (scripted)", "")
            ax.annotate(f"{short} ({split})", (x, y), xytext=(5, 3), textcoords="offset points", fontsize=6.5, color=INK)
    ax.annotate("non-LLM baselines: OpenAlex search, snowball\n(recall .007-.018, both splits)", (floor, 0.012),
                xytext=(14, 18), textcoords="offset points", fontsize=6.5, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    ax.set_xscale("log")
    ax.axvline(floor * 1.6, color=GRID, linewidth=1)
    ax.text(floor, ax.get_ylim()[1] * 0.98, "no LLM\n($0)", fontsize=6.5, color=MUTED, ha="center", va="top")
    ax.set_xlabel("LLM cost per episode, USD (log scale)", color=MUTED)
    ax.set_ylabel("recall at B=40", color=MUTED)
    ax.set_title("Cost vs recall (filled: eval; open: transfer)", fontsize=9, color=INK, loc="left")
    save(fig, out, "figW4_cost_recall.png")


def table(cells, out):
    cost = cost_rows()
    L = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{Eval (8 reviews)} & \multicolumn{3}{c}{Transfer (24 reviews)} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"System & Mode & Recall [95\% CI] & Prec. & \$/ep. & Recall [95\% CI] & Prec. & \$/ep. \\", r"\midrule"]
    for agent, mode, label, *_ in SYSTEMS:
        name = label.split(",")[0]
        cols = []
        for split in ("eval", "transfer"):
            c = cells.get((split, agent, mode, 40, "original"))
            if not c:
                cols += ["--", "--", "--"]
                continue
            rec = [v[0] for v in c.values()]
            lo, hi = boot_ci(rec)
            usd = cost.get((split, agent, mode))
            cols += [f"{st.fmean(rec):.3f} [{lo:.3f}, {hi:.3f}]", f"{st.fmean(v[1] for v in c.values()):.3f}",
                     "--" if usd is None else f"{usd:.3f}"]
        L.append(f"{name} & {mode} & " + " & ".join(cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables").mkdir(exist_ok=True)
    (out / "tables" / "web_main.tex").write_text("\n".join(L) + "\n")
    print("wrote", out / "tables" / "web_main.tex")
    print("\n".join(L))


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results/web"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "docs/figures"
    out.mkdir(parents=True, exist_ok=True)
    cells = load(d)
    fig_budget(cells, out)
    fig_miss(out)
    fig_env(cells, out)
    fig_cost(cells, out)
    table(cells, out)


if __name__ == "__main__":
    main()
