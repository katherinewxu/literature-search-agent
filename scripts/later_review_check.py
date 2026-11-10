"""LLM-free check of the eligibility judge: are judged-eligible records cited by later reviews?

Usage: python scripts/later_review_check.py JUDGEMENTS.jsonl OUT.md

For each SYNERGY review, fetch every OpenAlex work of type ``review`` that cites it (later
reviews and updates on the same topic) and pool their reference lists. Then compare, per
judged record group, the share that appears in that pool:
  gold, screened-excluded, unscreened judged eligible, unscreened judged ineligible.
Being cited by a later review signals relevance, not eligibility, so the comparison of interest
is unscreened-eligible vs unscreened-ineligible (same population, split only by the judge),
with gold and screened-excluded as reference points. No LLM is involved.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from litsearch.openalex import OpenAlexClient, short_id

ROOT = Path("/scratch/users/kwx04/literature-search-agent")
DATA = ["synergy-v3", "synergy-ext-transfer-v2"]


def later_reviews(client, review_paper_id, max_pages=5):
    out = []
    for page in range(1, max_pages + 1):
        res = client._get("works", {"filter": f"cites:{review_paper_id},type:review", "per-page": "200",
                                    "page": str(page), "select": "id,publication_year,referenced_works"})
        out += res.get("results", [])
        if len(out) >= res.get("meta", {}).get("count", 0):
            break
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    judg_path, out_md = Path(sys.argv[1]), Path(sys.argv[2])
    reviews = {}
    for d in DATA:
        for line in open(ROOT / "data" / d / "reviews.jsonl"):
            r = json.loads(line)
            reviews[r["id"]] = r
    client = OpenAlexClient(ROOT / "cache" / "openalex-web", delay_seconds=0.12)
    pool, n_later = {}, {}
    for rid, r in sorted(reviews.items()):
        lr = later_reviews(client, r["review_paper_id"])
        n_later[rid] = len(lr)
        pool[rid] = {short_id(w) for x in lr for w in (x.get("referenced_works") or [])}
    groups = defaultdict(lambda: [0, 0])
    per_review = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for j in map(json.loads, open(judg_path)):
        if j["eligible"] is None or not j["has_abstract"]:
            continue
        g = j["kind"] if j["kind"] != "unscreened" else ("unscreened, judged eligible" if j["eligible"] else "unscreened, judged ineligible")
        hit = j["work_id"] in pool[j["review_id"]]
        groups[g][0] += hit
        groups[g][1] += 1
        per_review[j["review_id"]][g][0] += hit
        per_review[j["review_id"]][g][1] += 1
    L = ["# Later-review citation check (no LLM)", "",
         f"Later reviews found: {sum(n_later.values())} citing reviews across {len(n_later)} SYNERGY reviews "
         f"({sum(1 for v in n_later.values() if v)} reviews have at least one).", "",
         "| group | cited by a later review | 95% CI | n |", "|---|---|---|---|"]
    for g in ("gold", "excluded", "unscreened, judged eligible", "unscreened, judged ineligible"):
        k, n = groups[g]
        lo, hi = wilson(k, n)
        L.append(f"| {g} | {k / n:.3f} | [{lo:.3f}, {hi:.3f}] | {n} |" if n else f"| {g} | - | - | 0 |")
    # Within-review comparison so a few heavily-cited reviews cannot drive the gap.
    diffs = []
    for rid, gs in per_review.items():
        a, b = gs["unscreened, judged eligible"], gs["unscreened, judged ineligible"]
        if a[1] >= 3 and b[1] >= 3 and n_later[rid]:
            diffs.append(a[0] / a[1] - b[0] / b[1])
    if diffs:
        pos = sum(d > 0 for d in diffs)
        L += ["", f"Within-review gap (eligible minus ineligible, reviews with >=3 of each and >=1 later review): "
              f"mean {sum(diffs) / len(diffs):+.3f} over {len(diffs)} reviews; positive in {pos}, negative in {sum(d < 0 for d in diffs)}."]
    L += ["", "Per review: later reviews | gold | excluded | unscr-eligible | unscr-ineligible (cited/n)", ""]
    for rid in sorted(per_review):
        gs = per_review[rid]
        cells = " | ".join(f"{gs[g][0]}/{gs[g][1]}" for g in ("gold", "excluded", "unscreened, judged eligible", "unscreened, judged ineligible"))
        L.append(f"- {rid}: {n_later[rid]} | {cells}")
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
