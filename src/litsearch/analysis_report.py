"""Render ``analysis.py`` results as JSON + Markdown (+ PNG figures if matplotlib is present)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import analysis
from .store import CorpusStore


def _fmt(triple) -> str:
    mean, lo, hi = triple
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


# Validated categorical slots (light surface, all-pairs CVD-safe for 3 series); baselines are neutral.
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BASELINE_STYLES = {"bm25": (MUTED, "--"), "snowball-tuned": (MUTED, ":"), "snowball": ("#a3a29c", ":"),
                   "random": ("#a3a29c", "-."), "specter2": (MUTED, (0, (6, 2, 1, 2)))}
MODES = ("citation", "keyword", "hybrid")


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _figures(output: Path, curve_sets: dict[int, dict[str, Any]], reach: dict[str, Any], exposure: dict[str, Any]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written = []
    agents = sorted({s.split("/")[0] for c in curve_sets.values() for s in c["systems"]})
    learned = [a for a in agents if a.split("-")[0] not in BASELINE_STYLES and a not in BASELINE_STYLES]
    preferred = [a for a in ("sonnet-5.5", "haiku-5.5", "qwen3-32b") if a in learned]
    learned = preferred + [a for a in learned if a not in preferred]
    color = {a: SERIES[i] for i, a in enumerate(learned[: len(SERIES)])}

    def style(agent: str):
        if agent in color:
            return color[agent], "-", 2.0
        if agent in BASELINE_STYLES:
            c, ls = BASELINE_STYLES[agent]
            return c, ls, 1.6
        return None

    for budget, curves in curve_sets.items():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, facecolor=SURFACE)
        for ax, mode in zip(axes, MODES):
            _style_axes(ax)
            ends = []
            for system, entry in sorted(curves["systems"].items()):
                agent, system_mode = system.split("/")
                look = style(agent)
                if look is None or system_mode != mode and not (agent == "bm25" and mode != "citation"):
                    continue
                c, ls, width = look
                ax.plot(curves["cost_grid"], entry["recall_by_cost"], color=c, linestyle=ls, linewidth=width, label=agent)
                ends.append((entry["recall_by_cost"][-1], agent, c))
            # Direct labels at line ends, nudged apart so they never collide.
            ends.sort()
            last = -1.0
            for value, agent, c in ends:
                y = max(value, last + 0.04)  # data units; keeps labels from colliding
                ax.text(curves["cost_grid"][-1] + budget * 0.02, y, agent, fontsize=7, color=INK, va="center")
                last = y
            ax.set_title(f"{mode} mode", fontsize=10, color=INK, loc="left")
            ax.set_xlabel("charged cost", fontsize=9, color=MUTED)
            ax.set_xlim(0, budget * 1.28)
        axes[0].set_ylabel("mean recall (8 reviews)", fontsize=9, color=MUTED)
        handles, labels = [], []
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h); labels.append(l)
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
        fig.suptitle(f"Recall vs. charged cost, B={budget} (BM25 reference shown in keyword and hybrid panels)",
                     fontsize=10, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        path = output / f"curves_b{budget}.png"
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        written.append(path.name)

    top = max(curve_sets)
    rows = [(k.split("|")[0], v) for k, v in exposure.items() if k.endswith(f"|B{top}") and style(k.split("/")[0])]
    if rows:
        rows.sort(key=lambda item: item[1]["recall"][0])
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1.2), facecolor=SURFACE)
        _style_axes(ax)
        for y, (system, row) in enumerate(rows):
            c = style(system.split("/")[0])[0]
            search, selected = row["exposure_recall"][0], row["recall"][0]
            ax.plot([selected, search], [y, y], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
            ax.scatter([search], [y], s=40, facecolor=SURFACE, edgecolor=c, linewidth=1.6, zorder=2)
            ax.scatter([selected], [y], s=40, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_yticks(range(len(rows)), [system for system, _ in rows], fontsize=8, color=INK)
        ax.set_xlabel("fraction of gold studies", fontsize=9, color=MUTED)
        ax.set_title(f"Gold exposed (hollow) vs. gold included (filled), B={top}", fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        path = output / "search_vs_screening.png"
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        written.append(path.name)

    if reach:
        fig, ax = plt.subplots(figsize=(5.5, 3.6), facecolor=SURFACE)
        _style_axes(ax)
        for index, (label, per_review) in enumerate(reach.items()):
            hops = sorted({int(h) for r in per_review.values() for h in r["reachable_within"]})
            within = lambda r, h: r["reachable_within"].get(h, r["reachable_within"].get(str(h)))  # noqa: E731
            means = [sum(within(r, h) for r in per_review.values()) / len(per_review) for h in hops]
            ax.plot(hops, means, marker="o", markersize=6, color=SERIES[index], linewidth=2, label=f"{label} graph")
            ax.annotate(f"{label} graph", (hops[-1], means[-1]), xytext=(6, -10 * index), textcoords="offset points",
                        fontsize=8, color=INK)
        ax.set_xlabel("citation hops from the 5 seeds", fontsize=9, color=MUTED)
        ax.set_ylabel("fraction of gold reachable", fontsize=9, color=MUTED)
        ax.set_title("Gold reachability (mean over reviews)", fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        path = output / "reachability.png"
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        written.append(path.name)
    return written


def write_analysis(data: Path, run_dirs: list[Path], output: Path, *, open_data: Path | None = None) -> str:
    output.mkdir(parents=True, exist_ok=True)
    store = CorpusStore.load(data)
    gold = {review_id: review.gold_ids for review_id, review in store.reviews.items()}
    runs = [run for run in analysis.load_runs(run_dirs) if run["episode"]["review_id"] in gold]
    budgets = sorted({run["episode"]["budget"] for run in runs})
    curve_sets = {budget: analysis.curves(runs, gold, budget=budget) for budget in budgets}
    exposure = analysis.exposure_table(runs, gold)
    strategy = analysis.strategy_profile(runs)

    reach: dict[str, dict[str, Any]] = {}
    stores = {"closed": store}
    if open_data is not None:
        stores["open"] = CorpusStore.load(open_data)
    for label, graph_store in stores.items():
        reach[label] = {}
        for review_id in sorted(store.reviews):
            seeds = analysis.seeds_for(graph_store, review_id)
            reach[label][review_id] = analysis.reachability(graph_store, review_id, seeds)
    misses = analysis.miss_analysis(
        [r for r in runs if r["episode"]["budget"] == max(budgets)], store, reach=reach["closed"]
    )

    systems = sorted({analysis.system_key(run) for run in runs})
    references = [ref for ref in ("bm25/keyword", "snowball/hybrid", "snowball/citation") if ref in systems]
    comparisons = []
    for reference in references:
        for system in systems:
            if system == reference:
                continue
            for budget in budgets:
                for metric in ("recall", "aurc", "f1"):
                    row = analysis.paired(runs, system, reference, budget=budget, metric=metric)
                    if row["n_reviews"]:
                        comparisons.append(row)
    results = {"curves": curve_sets, "exposure": exposure, "strategy": strategy,
               "reachability": reach, "misses": misses["summary"], "paired": comparisons}
    (output / "analysis.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "misses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in misses["rows"]), encoding="utf-8")
    figures = _figures(output, curve_sets, reach, exposure)

    lines = ["# Trace analysis", "", f"Runs: {len(runs)} from {', '.join(str(d) for d in run_dirs)}", ""]
    lines += ["## Search vs. screening (mean [95% bootstrap CI] over episodes)", "",
              "| system | B | search recall (gold exposed) | recall (included) | screen recall | precision | exposed | included |",
              "|---|---|---|---|---|---|---|---|"]
    for key, row in exposure.items():
        system, budget = key.split("|")
        lines.append(
            f"| {system} | {budget[1:]} | {_fmt(row['exposure_recall'])} | {_fmt(row['recall'])} | "
            f"{row['screen_recall'][0]:.3f} | {row['precision'][0]:.3f} | {row['exposed'][0]:.0f} | {row['included'][0]:.0f} |"
        )
    lines += ["", "## Paired per-review differences (system - reference; 95% bootstrap CI over reviews)", "",
              "| system | reference | B | Δrecall | Δaurc | Δf1 | wins/losses (recall) |", "|---|---|---|---|---|---|---|"]
    indexed = {(c["a"], c["b"], c["budget"], c["metric"]): c for c in comparisons}
    for (a, b, budget, metric), row in sorted(indexed.items()):
        if metric != "recall":
            continue
        cells = []
        for m in ("recall", "aurc", "f1"):
            c = indexed.get((a, b, budget, m))
            cells.append("-" if c is None else f"{c['mean_diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
        lines.append(f"| {a} | {b} | {budget} | " + " | ".join(cells) + f" | {row['wins']}/{row['losses']} of {row['n_reviews']} |")
    lines += ["", "## Matched exposure: mean recall once N papers have been exposed", ""]
    for budget, curves in curve_sets.items():
        grid = curves["exposure_grid"]
        lines += [f"### B={budget}", "", "| system | " + " | ".join(f"N={n}" for n in grid) + " |",
                  "|---|" + "---|" * len(grid)]
        for system, entry in curves["systems"].items():
            cells = ["-" if v is None else f"{v:.3f}" for v in entry["recall_by_exposed"]]
            lines.append(f"| {system} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Gold reachability from the 5 BM25 seeds (fraction of gold within h hops)", "",
              "| graph | review | universe | gold | h=1 | h=2 | h=3 | h=4 | isolated gold | mean gold degree |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for label, per_review in reach.items():
        for review_id, row in per_review.items():
            within = row["reachable_within"]
            lines.append(
                f"| {label} | {review_id} | {row['universe']} | {row['gold']} | "
                + " | ".join(f"{within[h]:.2f}" for h in (1, 2, 3, 4))
                + f" | {row['gold_isolated']:.2f} | {row['gold_mean_degree']:.1f} |"
            )
    lines += ["", "## Strategy profile (share of charged budget by action)", "",
              "| system | B | search | expand_refs | expand_citers | read | noop | unique queries | first expand step | leftover budget | stops with >=25% left |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, row in strategy.items():
        system, budget = key.split("|")
        share = row["budget_share"]
        first = row["first_expand_step"]
        lines.append(
            f"| {system} | {budget[1:]} | " + " | ".join(f"{share.get(k, 0):.2f}" for k in ("search", "expand_refs", "expand_citers", "read", "noop"))
            + f" | {row['unique_queries']:.1f} | {'-' if first is None else f'{first:.1f}'} | {row['leftover_budget']:.2f} | {row['stop_early_rate']:.2f} |"
        )
    lines += ["", f"## Missed gold at B={max(budgets)} (counts over episodes)", "",
              "| system | exposed, not included | not reached in budget | unreached, BM25 rank<=100 | unreachable & rank>100 | no lexical overlap | missing abstract |",
              "|---|---|---|---|---|---|---|"]
    for key, row in misses["summary"].items():
        lines.append(
            f"| {key} | {row.get('exposed_not_included', 0)} | {row.get('not_reached_in_budget', 0)} | "
            f"{row.get('unreached_rank<=100', 0)} | {row.get('unreachable_graph_and_rank>100', 0)} | "
            f"{row.get('no_lexical_overlap', 0)} | {row.get('no_abstract', 0)} |"
        )
    if figures:
        lines += ["", "## Figures", ""] + [f"![{name}]({name})" for name in figures]
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output / "analysis.md")
