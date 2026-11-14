"""Why do live-web LLM agents miss gold studies? Usage: python scripts/web_miss_analysis.py OUTDIR

For every LLM-agent web run (repeat 0) and every gold study of its review, assign the first
matching cause:
  found              included by the agent
  rejected_read      the agent read the abstract and then excluded it
  rejected_unread    the agent excluded it from the title/snippet alone
  seen_undecided     shown to the agent (seed or tool result) but never included or excluded
  one_hop            never shown, but cites or is cited by a paper the agent included
                     (one expand away), using the corpus citation graph
  other_system       never shown to this agent, but some other web run on the review showed it
  never_surfaced     no web run on the review was ever shown it

Also compares properties of gold that some system found vs gold no system ever surfaced
(abstract present, year, citation degree, title length), and lists example rejections.
Gold labels are used only here, on the evaluator side.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from litsearch.store import CorpusStore

ROOT = Path("/scratch/users/kwx04/literature-search-agent")
SETS = [("eval (8 reviews)", "synergy-v3-web-r5", "synergy-v3"),
        ("transfer (24 reviews)", "ext-transfer-web-r5", "synergy-ext-transfer-v2")]
LLM_AGENTS = ("haiku-5.5", "sonnet-5.5")
CAUSES = ("found", "rejected_read", "rejected_unread", "seen_undecided", "one_hop", "other_system", "never_surfaced")


def run_facts(run):
    exposed = set(run.get("seed_ids") or [])
    read, excluded = set(), set()
    for ev in run["trace"]:
        exposed.update(ev.get("result_ids") or [])
        act = ev.get("action") or {}
        if str(act.get("kind", "")).lower().endswith("read") and act.get("paper_id"):
            read.add(act["paper_id"])
        excluded.update(ev.get("excluded_ids") or [])
    included = set(run["predictions"])
    return exposed, read, excluded - included, included


def neighbors(store, pid):
    p = store.papers.get(pid)
    return set(p.references) | set(p.citers) if p else set()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    L = ["# Why web agents miss gold studies", ""]
    examples = []
    for title, run_dir, data in SETS:
        store = CorpusStore.load(ROOT / "data" / data)
        runs = [json.loads(p.read_text()) for p in sorted((ROOT / "runs" / run_dir).glob("*.json"))]
        runs = [r for r in runs if r["matrix"].get("repeat", 0) == 0]
        facts = {id(r): run_facts(r) for r in runs}
        shown_any = defaultdict(set)  # review -> works shown to any web run
        for r in runs:
            shown_any[r["episode"]["review_id"]] |= facts[id(r)][0]
        table = defaultdict(Counter)
        hop_deg = defaultdict(list)  # (agent, mode, B) -> smallest neighbour list linking a one_hop gold
        for r in runs:
            if r["agent"] not in LLM_AGENTS:
                continue
            ep = r["episode"]
            rid = ep["review_id"]
            exposed, read, excluded, included = facts[id(r)]
            others = set().union(*(facts[id(o)][0] for o in runs if o is not r and o["episode"]["review_id"] == rid))
            hop = set().union(*(neighbors(store, i) for i in included)) if included else set()
            for g in store.review(rid).gold_ids:
                if g in included:
                    c = "found"
                elif g in excluded:
                    c = "rejected_read" if g in read else "rejected_unread"
                    p = store.papers.get(g)
                    examples.append((title, r["agent"], ep["mode"], ep["budget"], rid, c, p.title if p else g))
                elif g in exposed:
                    c = "seen_undecided"
                elif g in hop:
                    c = "one_hop"
                    hop_deg[(r["agent"], ep["mode"], ep["budget"])].append(
                        min(len(neighbors(store, i)) for i in included if g in neighbors(store, i)))
                elif g in others:
                    c = "other_system"
                else:
                    c = "never_surfaced"
                table[(r["agent"], ep["mode"], ep["budget"])][c] += 1
        L += [f"## {title}", "", "Share of gold studies by cause (summed over reviews).", "",
              "| agent | mode | B | n gold | " + " | ".join(CAUSES) + " |", "|---" * (4 + len(CAUSES)) + "|"]
        for key in sorted(table):
            n = sum(table[key].values())
            L.append(f"| {key[0]} | {key[1]} | {key[2]} | {n} | " + " | ".join(f"{table[key][c] / n:.2f}" for c in CAUSES) + " |")
        L += ["", "one_hop: size of the smallest citation neighbourhood (refs + citers) linking the gold study to an included paper;",
              "an expand returns at most 20, so large neighbourhoods are not one cheap step away.", "",
              "| agent | mode | B | n | median | share <= 20 | share <= 100 |", "|---|---|---|---|---|---|---|"]
        for key in sorted(hop_deg):
            d = hop_deg[key]
            L.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(d)} | {statistics.median(d):.0f} | "
                     f"{sum(x <= 20 for x in d) / len(d):.2f} | {sum(x <= 100 for x in d) / len(d):.2f} |")
        # What distinguishes gold no system surfaced from gold someone surfaced?
        feats = {"surfaced by some run": [], "never surfaced": []}
        for rid in shown_any:
            for g in store.review(rid).gold_ids:
                p = store.papers.get(g)
                if p is None:
                    continue
                feats["surfaced by some run" if g in shown_any[rid] else "never surfaced"].append(p)
        L += ["", "Gold studies surfaced by at least one web run vs by none:", "",
              "| group | n | has abstract | median year | median citation degree | median title words |", "|---|---|---|---|---|---|"]
        for name, ps in feats.items():
            if not ps:
                continue
            yrs = [p.publication_year for p in ps if p.publication_year]
            L.append(f"| {name} | {len(ps)} | {sum(bool(p.abstract) for p in ps) / len(ps):.2f} | "
                     f"{statistics.median(yrs) if yrs else '-'} | {statistics.median(len(p.references) + len(p.citers) for p in ps)} | "
                     f"{statistics.median(len(p.title.split()) for p in ps)} |")
        L.append("")
    L += ["## Example gold studies the agent rejected", ""]
    seen = set()
    for t, agent, mode, b, rid, c, ttl in examples:
        if (rid, ttl) in seen or len(seen) >= 25:
            continue
        seen.add((rid, ttl))
        L.append(f"- [{t.split()[0]}] {agent} {mode} B={b}, {rid}, {c}: {ttl}")
    (out / "web_miss_analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
