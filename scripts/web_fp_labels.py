"""Which live-web inclusions already carry a human verdict? Usage: python scripts/web_fp_labels.py OUTDIR

Every SYNERGY review ships the full set of records its authors screened (``candidate_ids``),
each labelled included/excluded; the 8 eval reviews' raw CSVs also carry the title/abstract
stage label. A web "false positive" (included by the agent, not in gold) is therefore either
  * screened-excluded: the review authors saw it and said no (a real negative), or
  * unscreened: the review's search never retrieved it, so its eligibility is unknown.
Only the unscreened ones can hide recall that the gold set misses.

Writes OUTDIR/web_fp_labels.md (coverage tables) plus two JSONL files for the LLM-judge step:
  unscreened.jsonl         unique (review, work) pairs with no human label, with title/abstract
  screened_excluded.jsonl  unique (review, work) pairs labelled excluded (judge calibration)
Gold labels are read only here, on the evaluator side; nothing is fed back to an agent.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from litsearch.openalex import abstract_from_inverted, short_id

ROOT = Path("/scratch/users/kwx04/literature-search-agent")
SETS = [
    ("eval (8 reviews)", "synergy-v3-web-r5", "synergy-v3"),
    ("transfer (24 reviews)", "ext-transfer-web-r5", "synergy-ext-transfer-v2"),
]
csv.field_size_limit(10**9)


def load_reviews(data):
    out = {}
    for line in open(ROOT / "data" / data / "reviews.jsonl"):
        r = json.loads(line)
        out[r["id"]] = {"gold": set(r["gold_ids"]), "screened": set(r["candidate_ids"]),
                        "question": r.get("question") or r.get("title"), "criteria": r["eligibility_criteria"]}
    return out


def abstract_stage_labels():
    """review -> {work id: title/abstract-stage include (bool)} for reviews with raw CSVs."""
    out = {}
    for folder in ("raw", "raw_rest"):
        for path in sorted((ROOT / "synergy" / folder).glob("*.csv")):
            rows = {}
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    wid = short_id(row["openalex_id"].replace("openalex.org/w", "openalex.org/W"))
                    if wid and row.get("label_abstract_included") not in (None, ""):
                        rows[wid.upper()] = row["label_abstract_included"].strip() in ("1", "1.0", "True")
            out.setdefault(path.stem, {}).update(rows)
    return out


def web_cache_index(wanted):
    """id -> {title, abstract, year} for works seen in cached live-OpenAlex responses."""
    found = {}
    for path in (ROOT / "cache" / "openalex-web").glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        works = payload.get("results") if isinstance(payload, dict) and "results" in payload else [payload]
        for work in works or []:
            if not isinstance(work, dict):
                continue
            wid = short_id(work.get("id"))
            if wid in wanted and wid not in found:
                found[wid] = {"title": work.get("display_name") or work.get("title"),
                              "abstract": abstract_from_inverted(work.get("abstract_inverted_index")),
                              "year": work.get("publication_year"), "doi": work.get("doi")}
    return found


def store_index(data, wanted):
    found = {}
    for line in open(ROOT / "data" / data / "works.jsonl"):
        w = json.loads(line)
        if w["id"] in wanted:
            found[w["id"]] = {"title": w.get("title"), "abstract": w.get("abstract"),
                              "year": w.get("publication_year"), "doi": w.get("doi")}
    return found


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def main():
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    stage = abstract_stage_labels()
    L = ["# Live-web inclusions: how many already have a human label?", "",
         "Per cell: all inclusions, then the share that are gold, screened-and-excluded by the review",
         "authors, or never screened. `abs-incl` = screened-excluded records that passed the review's",
         "title/abstract stage (excluded only at full text; eval reviews only). Pooled over reviews.", ""]
    unscreened, excluded = {}, {}
    for title, run_dir, data in SETS:
        reviews = load_reviews(data)
        cells = defaultdict(Counter)
        per_review = defaultdict(Counter)
        for path in sorted((ROOT / "runs" / run_dir).glob("*.json")):
            run = json.loads(path.read_text())
            ep = run["episode"]
            if run["matrix"].get("repeat", 0) != 0:
                continue
            rv = reviews[ep["review_id"]]
            key = (run["agent"], ep["mode"], ep["budget"])
            for wid in run["predictions"]:
                c = cells[key]
                c["n"] += 1
                if wid in rv["gold"]:
                    c["gold"] += 1
                elif wid in rv["screened"]:
                    c["excl"] += 1
                    lab = stage.get(ep["review_id"], {}).get(wid)
                    if lab is not None:
                        c["excl_staged"] += 1
                        c["abs_incl"] += lab
                    excluded.setdefault((ep["review_id"], wid), {"abs_incl": lab, "data": data})
                    if run["agent"] != "openalex-search":
                        per_review[ep["review_id"]]["excl"] += 1
                else:
                    c["unscr"] += 1
                    if run["agent"] != "openalex-search":
                        unscreened.setdefault((ep["review_id"], wid), {"data": data, "agents": set()})["agents"].add(run["agent"])
                        per_review[ep["review_id"]]["unscr"] += 1
        L += [f"## {title}", "", "| agent | mode | B | inclusions | gold | screened-excluded | unscreened | abs-incl (of staged excl) |",
              "|---|---|---|---|---|---|---|---|"]
        tot = Counter()
        for key in sorted(cells):
            c = cells[key]
            if key[0] != "openalex-search":
                tot.update(c)
            L.append(f"| {key[0]} | {key[1]} | {key[2]} | {c['n']} | {pct(c['gold'], c['n'])} | {pct(c['excl'], c['n'])} | "
                     f"{pct(c['unscr'], c['n'])} | {c['abs_incl']}/{c['excl_staged']} |")
        nonGold = tot["n"] - tot["gold"]
        L += ["", f"LLM agents pooled: {tot['n']} inclusions, {nonGold} not gold; of those, "
              f"{pct(tot['excl'], nonGold)} screened-excluded (human negative) and {pct(tot['unscr'], nonGold)} unscreened.", "",
              "Per review (LLM agents, non-gold inclusions): screened-excluded / unscreened", ""]
        L += [", ".join(f"{r} {c['excl']}/{c['unscr']}" for r, c in sorted(per_review.items())), ""]

    for name, pairs, by_data in (("unscreened", unscreened, None), ("screened_excluded", excluded, None)):
        reviews_all = {d: load_reviews(d) for _, _, d in SETS}
        wanted = {wid for _, wid in pairs}
        meta = web_cache_index(wanted) if name == "unscreened" else {}
        for _, _, data in SETS:
            meta.update({k: v for k, v in store_index(data, wanted).items() if k not in meta})
        missing = 0
        with open(outdir / f"{name}.jsonl", "w") as fh:
            for (rid, wid), info in sorted(pairs.items()):
                m = meta.get(wid)
                if not m or not m.get("title"):
                    missing += 1
                    continue
                rv = reviews_all[info["data"]][rid]
                rec = {"review_id": rid, "work_id": wid, **m, "question": rv["question"], "criteria": rv["criteria"]}
                if name == "unscreened":
                    rec["agents"] = sorted(info["agents"])
                else:
                    rec["abs_incl"] = info["abs_incl"]
                fh.write(json.dumps(rec) + "\n")
        L.append(f"- {name}.jsonl: {len(pairs) - missing} unique (review, work) pairs written; {missing} without a cached title skipped.")
    (outdir / "web_fp_labels.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
