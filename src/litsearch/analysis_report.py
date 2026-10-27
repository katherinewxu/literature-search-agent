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


def _figures(output: Path, curve_sets: dict[int, dict[str, Any]], reach: dict[str, Any], strategy: dict[str, Any]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written = []
    for budget, curves in curve_sets.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for system, entry in curves["systems"].items():
            axes[0].plot(curves["cost_grid"], entry["recall_by_cost"], label=system)
            xs = [x for x, y in zip(curves["exposure_grid"], entry["recall_by_exposed"]) if y is not None]
            ys = [y for y in entry["recall_by_exposed"] if y is not None]
            axes[1].plot(xs, ys, marker="o", label=system)
        axes[0].set(xlabel="charged cost", ylabel="mean recall", title=f"Recall vs. cost (B={budget})")
        axes[1].set(xlabel="papers exposed", ylabel="mean recall", title="Recall vs. papers exposed")
        axes[1].set_xscale("symlog", linthresh=10)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        path = output / f"curves_b{budget}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)
    if reach:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, per_review in reach.items():
            hops = sorted({int(h) for r in per_review.values() for h in r["reachable_within"]})
            means = [sum(r["reachable_within"][h] for r in per_review.values()) / len(per_review) for h in hops]
            ax.plot(hops, means, marker="o", label=label)
        ax.set(xlabel="citation hops from seeds", ylabel="fraction of gold reachable", title="Gold reachability")
        ax.legend()
        fig.tight_layout()
        path = output / "reachability.png"
        fig.savefig(path, dpi=150)
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
    figures = _figures(output, curve_sets, reach, strategy)

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
