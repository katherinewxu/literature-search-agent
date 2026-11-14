"""Offline counterfactual: would relevance-ranked citation expansion surface more gold?

Usage: python scripts/expand_ranking_counterfactual.py OUTDIR

Live expansion returns a paper's references in list order and its citers by citation count,
cut to k (<= 20). This replays every expand the LLM web agents actually made (same paper,
same k, same cost) and asks how much gold the returned slice would hold under three orders:
  actual   the result_ids the live run really got (ground truth for the replay)
  sim      the default order re-created from the corpus graph (refs in stored order,
           citers by their citer count) to check the replay is faithful
  bm25     neighbours ranked by BM25 against the review question + eligibility criteria
           (text the agent already sees; no gold involved in the ranking)
Neighbourhoods come from the corpus citation graph, so works outside the corpus are missing;
the replay ignores which works were already shown earlier in the episode.
Gold labels are used only to score, on the evaluator side.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from litsearch.retrieval import BM25Retriever
from litsearch.store import CorpusStore

ROOT = Path("/scratch/users/kwx04/literature-search-agent")
SETS = [("eval (8 reviews)", "synergy-v3-web-r5", "synergy-v3"),
        ("transfer (24 reviews)", "ext-transfer-web-r5", "synergy-ext-transfer-v2")]
LLM_AGENTS = ("haiku-5.5", "sonnet-5.5")


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    L = ["# Counterfactual: relevance-ranked citation expansion", "",
         "Every expand the LLM web agents made is replayed with the same paper and k; columns count gold",
         "studies in the returned slices (union per run, summed over runs), and recall if each were included.", ""]
    for title, run_dir, data in SETS:
        store = CorpusStore.load(ROOT / "data" / data)
        bm25 = BM25Retriever(store.papers.values())
        runs = [json.loads(p.read_text()) for p in sorted((ROOT / "runs" / run_dir).glob("*.json"))]
        runs = [r for r in runs if r["matrix"].get("repeat", 0) == 0 and r["agent"] in LLM_AGENTS]
        score_cache = {}
        agg = defaultdict(lambda: defaultdict(float))
        coverage = []
        for r in runs:
            ep = r["episode"]
            rev = store.review(ep["review_id"])
            gold = rev.gold_ids
            if rev.id not in score_cache:
                score_cache[rev.id] = bm25.scores(f"{rev.question} {rev.eligibility_criteria}")
            scores = score_cache[rev.id]
            got = {"actual": set(), "sim": set(), "bm25": set()}
            n_exp = 0
            for ev in r["trace"]:
                act = ev.get("action") or {}
                kind = str(act.get("kind", "")).lower()
                if kind not in ("expand_refs", "expand_citers") or not act.get("paper_id"):
                    continue
                paper = store.papers.get(act["paper_id"])
                k = len(ev.get("result_ids") or [])
                if paper is None or k == 0:
                    continue
                n_exp += 1
                if kind == "expand_refs":
                    nbrs = [i for i in paper.references if i in store.papers]
                    sim = nbrs
                else:
                    nbrs = [i for i in paper.citers if i in store.papers]
                    sim = sorted(nbrs, key=lambda i: -len(store.papers[i].citers))
                total = len(paper.references) if kind == "expand_refs" else len(paper.citers)
                coverage.append(len(nbrs) / total if total else 1.0)
                ranked = sorted(nbrs, key=lambda i: -scores.get(i, 0.0))
                got["actual"] |= set(ev["result_ids"]) & gold
                got["sim"] |= set(sim[:k]) & gold
                got["bm25"] |= set(ranked[:k]) & gold
            key = (r["agent"], ep["mode"], ep["budget"])
            agg[key]["runs"] += 1
            agg[key]["expands"] += n_exp
            agg[key]["gold"] += len(gold)
            agg[key]["rec_found"] += len(set(r["predictions"]) & gold) / len(gold)
            for name, s in got.items():
                agg[key][name] += len(s)
                # recall if every gold in the slice (plus what the agent already found) were included
                agg[key][f"rec_{name}"] += len((set(r["predictions"]) & gold) | s) / len(gold)
        L += [f"## {title}", "",
              "| agent | mode | B | runs | expands/run | gold in slices: actual | sim | bm25 | recall: actual | upper bound actual | upper bound bm25 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for key in sorted(agg):
            a = agg[key]
            n = a["runs"]
            L.append(f"| {key[0]} | {key[1]} | {key[2]} | {n:.0f} | {a['expands'] / n:.1f} | {a['actual']:.0f} | {a['sim']:.0f} | "
                     f"{a['bm25']:.0f} | {a['rec_found'] / n:.3f} | {a['rec_actual'] / n:.3f} | {a['rec_bm25'] / n:.3f} |")
        if coverage:
            L += ["", f"Share of a paper's neighbours present in the corpus graph (median over expands): {statistics.median(coverage):.2f}", ""]
    (out / "expand_ranking_counterfactual.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
