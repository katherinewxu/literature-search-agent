"""Robustness tables: temperature-0.7 repeat spread and memorization-adjusted recall.

Usage: python scripts/robustness.py OUT.md
Reads run directories under runs/ and memo probes under memo/ (FarmShare layout).
Memorization adjustment: every gold paper that ANY model named from memory in the
closed-book probe is removed from that review's gold set, and recall is recomputed for
every system over the remaining gold. Gold labels are read only by this scorer.
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from litsearch.store import CorpusStore

ROOT = Path("/scratch/users/kwx04/literature-search-agent")


def load_runs(*dirs: str) -> list[dict]:
    out = []
    for d in dirs:
        for path in sorted((ROOT / "runs" / d).glob("*.json")):
            run = json.loads(path.read_text())
            ep = run["episode"]
            out.append(
                {
                    "review": ep["review_id"],
                    "agent": run["agent"],
                    "mode": ep["mode"],
                    "budget": ep["budget"],
                    "repeat": run["matrix"].get("repeat", 0),
                    "pred": set(run["predictions"]),
                    "recall": run["metrics"]["recall"],
                    "f1": run["metrics"]["f1"],
                    "precision": run["metrics"]["precision"],
                    "data_dir": run["matrix"]["data_dir"],
                }
            )
    return out


def boot(diffs: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return st.mean(diffs), means[int(0.025 * n)], means[int(0.975 * n)]


def fmt(x: float) -> str:
    return f"{x:+.3f}" if x < 0 or x > 0 else "+.000"


def t07_section(lines: list[str]) -> None:
    hot = load_runs("synergy-v3-api-t07-r5")
    cold = load_runs("synergy-v3-api-r4")
    lines += ["## Temperature 0.7 repeats (closed pool, hybrid, 8 reviews, 3 repeats)", ""]
    lines += ["| agent | B | T=0 recall | T=0.7 repeat means | spread | mean per-review SD |", "|---|---|---|---|---|---|"]
    cells = sorted({(r["agent"], r["budget"]) for r in hot})
    for agent, budget in cells:
        base = agent.replace("-t07", "")
        t0 = [r["recall"] for r in cold if r["agent"] == base and r["mode"] == "hybrid" and r["budget"] == budget]
        reps = defaultdict(list)
        per_review = defaultdict(list)
        for r in hot:
            if r["agent"] == agent and r["budget"] == budget:
                reps[r["repeat"]].append(r["recall"])
                per_review[r["review"]].append(r["recall"])
        means = [st.mean(v) for _, v in sorted(reps.items())]
        sd = st.mean(st.pstdev(v) for v in per_review.values())
        lines.append(
            f"| {base} | {budget} | {st.mean(t0):.3f} | {', '.join(f'{m:.3f}' for m in means)} | "
            f"{max(means) - min(means):.3f} | {sd:.3f} |"
        )
    # Sonnet - Haiku at B40 per repeat
    lines += ["", "Sonnet − Haiku hybrid recall at B=40, per repeat (paired over reviews):", ""]
    for rep in range(3):
        s = {r["review"]: r["recall"] for r in hot if r["agent"] == "sonnet-5.5-t07" and r["budget"] == 40 and r["repeat"] == rep}
        h = {r["review"]: r["recall"] for r in hot if r["agent"] == "haiku-5.5-t07" and r["budget"] == 40 and r["repeat"] == rep}
        if s and h:
            m, lo, hi = boot([s[k] - h[k] for k in s if k in h])
            lines.append(f"- repeat {rep}: {fmt(m)} [{fmt(lo)}, {fmt(hi)}]")
    lines.append("")


def memo_section(lines: list[str], title: str, memo: str, dirs: list[str], pairs: list[tuple]) -> None:
    probe = [json.loads(l) for l in (ROOT / "memo" / memo).read_text().splitlines() if l.strip()]
    memorized: dict[str, set[str]] = defaultdict(set)
    for row in probe:
        memorized[row["review_id"]].update(row["gold_hit_ids"])
    runs = load_runs(*dirs)
    store = CorpusStore.load(runs[0]["data_dir"])
    gold = {rid: set(store.review(rid).gold_ids) for rid in {r["review"] for r in runs}}
    removed = sum(len(memorized[k] & gold[k]) for k in gold)
    total = sum(len(v) for v in gold.values())
    lines += [f"## Memorization-adjusted recall: {title}", ""]
    lines.append(
        f"{removed} of {total} gold papers ({removed / total:.1%}) were named from memory by at least one "
        f"model and are removed; recall is recomputed over the remaining gold.\n"
    )

    def adj(r: dict) -> float:
        g = gold[r["review"]] - memorized[r["review"]]
        return len(r["pred"] & g) / max(1, len(g))

    lines += ["| system | mode | B | recall | adjusted | change |", "|---|---|---|---|---|---|"]
    key = lambda r: (r["agent"], r["mode"], r["budget"])
    groups = defaultdict(list)
    for r in runs:
        if r["repeat"] == 0:
            groups[key(r)].append(r)
    for (agent, mode, budget), rs in sorted(groups.items()):
        if budget != 40 or mode not in {"hybrid", "keyword", "exhaustive", "citation"} and "specter" not in agent:
            continue
        a, b = st.mean(r["recall"] for r in rs), st.mean(adj(r) for r in rs)
        lines.append(f"| {agent} | {mode} | {budget} | {a:.3f} | {b:.3f} | {fmt(b - a)} |")
    lines += ["", "Paired differences at B=40 (bootstrap over reviews), original vs memorization-adjusted:", ""]
    for (a_agent, a_mode), (b_agent, b_mode) in pairs:
        A = {r["review"]: r for r in groups.get((a_agent, a_mode, 40), [])}
        B = {r["review"]: r for r in groups.get((b_agent, b_mode, 40), [])}
        common = sorted(set(A) & set(B))
        if not common:
            continue
        o = boot([A[k]["recall"] - B[k]["recall"] for k in common])
        m = boot([adj(A[k]) - adj(B[k]) for k in common])
        lines.append(
            f"- {a_agent} {a_mode} − {b_agent} {b_mode}: {fmt(o[0])} [{fmt(o[1])}, {fmt(o[2])}] → "
            f"{fmt(m[0])} [{fmt(m[1])}, {fmt(m[2])}] (n={len(common)})"
        )
    lines.append("")


def main() -> None:
    lines = ["# Robustness checks", ""]
    t07_section(lines)
    closed_pairs = [
        (("sonnet-5.5", "hybrid"), ("bm25", "keyword")),
        (("sonnet-5.5", "hybrid"), ("specter2", "keyword")),
        (("sonnet-5.5", "hybrid"), ("haiku-5.5", "hybrid")),
        (("haiku-5.5", "hybrid"), ("bm25", "keyword")),
        (("qwen3-1.7b-grpo", "hybrid"), ("qwen3-32b", "hybrid")),
    ]
    memo_section(
        lines,
        "closed pool (8 reviews)",
        "v3-r2.jsonl",
        ["synergy-v3-api-r4", "synergy-v3-baselines-r3", "synergy-v3-dense-r3", "synergy-v3-small-trained-r4"],
        closed_pairs,
    )
    memo_section(
        lines,
        "transfer (24 reviews)",
        "transfer.jsonl",
        ["ext-transfer-api-r4", "ext-transfer-baselines-r3", "ext-transfer-small-r4"],
        closed_pairs,
    )
    Path(sys.argv[1]).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
