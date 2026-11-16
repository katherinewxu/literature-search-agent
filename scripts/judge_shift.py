"""Covariate-shift correction for the LLM judge's eligible share of never-screened records.

Usage: python scripts/judge_shift.py OUT.md

The judge is calibrated on records with human labels (gold studies and screened-out records the
agents included), but the target is records the review never screened. Those differ in two
observable ways, and this script corrects for each:

  similarity  how on-topic a record looks: BM25 of title+abstract against the review question
              and eligibility criteria, as a within-review percentile. Sensitivity and
              specificity are measured per similarity bin and applied to the unscreened records
              in that bin (stratified Rogan-Gladen).
  review mix  calibration records are concentrated in some reviews, unscreened records in
              others. Calibration records are reweighted so each review counts as much as it
              does among the unscreened records.

Estimates: naive (as in web_robust.py), similarity-stratified, review-reweighted, both, and
per-review Rogan-Gladen with each review's error rates shrunk toward the pooled rates (k=10
pseudo-records), which is the review-mix correction with less variance from tiny reviews.
CIs: cluster bootstrap over reviews (2,000 resamples), recomputing bins' rates and weights.
This removes only observable shift; whatever made the review's own search miss a record may
still change the judge's error rates. Labels are used on the evaluator side only.
"""
from __future__ import annotations

import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from web_judge import ROOT, WEBFP, all_records, load_reviews  # noqa: E402

JUDGES = ("judgements_gemini-3.8-flash.jsonl", "judgements_gpt-5.6-luna.jsonl")
DATA = ("synergy-v3", "synergy-ext-transfer-v2")
CUTS = (0.25, 0.5, 0.75)  # similarity-percentile bin edges
MIN_CAL = 15              # fewer calibration records of a class in a bin -> borrow the nearest bin's rate
STOP = set("the a an of and or in on for to with by from at as is are was were be been this that these those "
           "which who whom its it their there than then into over under between among within without not no "
           "study studies review systematic patients participants".split())
rng = random.Random(20261114)


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2 and t not in STOP]


def bm25_percentiles(recs, reviews):
    """Within-review percentile of BM25(question + criteria, title + abstract)."""
    docs = [Counter(tokens(f"{r['title']} {r.get('abstract') or ''}")) for r in recs]
    N = len(docs)
    df = Counter(t for d in docs for t in d)
    avg = statistics.fmean(sum(d.values()) for d in docs) or 1.0
    idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    scores = []
    for r, d in zip(recs, docs):
        rv = reviews[r["review_id"]]
        q = set(tokens(f"{rv.get('question') or rv.get('title')} {rv.get('eligibility_criteria') or ''}"))
        L = sum(d.values())
        scores.append(sum(idf[t] * d[t] * 2.2 / (d[t] + 1.2 * (0.25 + 0.75 * L / avg)) for t in q if t in d))
    by_rev = defaultdict(list)
    for i, r in enumerate(recs):
        by_rev[r["review_id"]].append(i)
    pct = [0.0] * N
    for idx in by_rev.values():
        order = sorted(idx, key=lambda i: scores[i])
        for rank, i in enumerate(order):
            pct[i] = (rank + 0.5) / len(order)
    return pct


def bin_of(p):
    return sum(p >= c for c in CUTS)


def rg(q, sens, spec):
    return min(1.0, max(0.0, (q + spec - 1) / (sens + spec - 1)))


def wmean(pairs):
    """pairs: (value, weight)."""
    tw = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / tw if tw else float("nan")


def estimate(rows, reweight, stratify):
    """rows: dicts with review, kind, y (judged eligible), bin. Returns (share, effective PPV)."""
    count = defaultdict(Counter)  # kind -> review -> n
    for r in rows:
        count[r["kind"]][r["review"]] += 1
    tot = {k: sum(c.values()) for k, c in count.items()}
    U = count["unscreened"]

    def w(r):
        if not reweight or r["kind"] == "unscreened":
            return 1.0
        c = count[r["kind"]]
        # review weight in this class = its share of unscreened records / its share of this class
        covered = sum(U[v] for v in c if U[v])
        if not U[r["review"]] or not covered:
            return 0.0
        return (U[r["review"]] / covered) / (c[r["review"]] / tot[r["kind"]])

    def rates(sub):
        sens = wmean([(r["y"], w(r)) for r in sub if r["kind"] == "gold"])
        spec = 1 - wmean([(r["y"], w(r)) for r in sub if r["kind"] == "excluded"])
        return sens, spec

    sens0, spec0 = rates(rows)
    bins = sorted({r["bin"] for r in rows}) if stratify else [None]
    share, yes_ppv, yes_n = 0.0, 0.0, 0
    n_unscr = sum(r["kind"] == "unscreened" for r in rows)
    def nearest(b, kind):
        """Rate for `kind` from the nearest bin with >= MIN_CAL calibration records (else pooled)."""
        for d in range(len(CUTS) + 1):
            for bb in sorted({b - d, b + d}):
                sub = [r for r in rows if r["bin"] == bb and r["kind"] == kind]
                if len(sub) >= MIN_CAL:
                    return rates(sub)[0 if kind == "gold" else 1]
        return sens0 if kind == "gold" else spec0

    for b in bins:
        sub = [r for r in rows if b is None or r["bin"] == b]
        un = [r["y"] for r in sub if r["kind"] == "unscreened"]
        if not un:
            continue
        if b is None:
            sens, spec = sens0, spec0
        else:
            sens, spec = nearest(b, "gold"), nearest(b, "excluded")
        if sens + spec - 1 < 0.1:
            sens, spec = sens0, spec0
        q = statistics.fmean(un)
        p = rg(q, sens, spec)
        share += p * len(un) / n_unscr
        ppv = sens * p / (sens * p + (1 - spec) * (1 - p)) if p > 0 else 0.0
        yes_ppv += ppv * sum(un)
        yes_n += sum(un)
    return share, (yes_ppv / yes_n if yes_n else float("nan"))


def shrunk(rows, k=10.0):
    """Per-review Rogan-Gladen with each review's sensitivity/specificity shrunk toward the pooled
    rates (k pseudo-records), weighted by the review's unscreened count."""
    pool = defaultdict(list)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pool[r["kind"]].append(r["y"])
        by[r["review"]][r["kind"]].append(r["y"])
    s0, sp0 = statistics.fmean(pool["gold"]), 1 - statistics.fmean(pool["excluded"])
    share, n_u, yes_ppv, yes_n = 0.0, 0, 0.0, 0
    for v in by.values():
        u = v["unscreened"]
        if not u:
            continue
        sens = (sum(v["gold"]) + k * s0) / (len(v["gold"]) + k)
        spec = (len(v["excluded"]) - sum(v["excluded"]) + k * sp0) / (len(v["excluded"]) + k)
        if sens + spec - 1 < 0.1:
            sens, spec = s0, sp0
        p = rg(statistics.fmean(u), sens, spec)
        share += p * len(u)
        n_u += len(u)
        if p > 0:
            yes_ppv += sens * p / (sens * p + (1 - spec) * (1 - p)) * sum(u)
        yes_n += sum(u)
    return share / n_u, (yes_ppv / yes_n if yes_n else float("nan"))


VARIANTS = (("naive (pooled calibration)", False, False), ("similarity-stratified", False, True),
            ("review-reweighted", True, False), ("both", True, True))


def main():
    out = Path(sys.argv[1])
    reviews = {}
    for d in DATA:
        reviews.update(load_reviews(d))
    recs = [r for r in all_records() if r.get("abstract")]
    pct = bm25_percentiles(recs, reviews)
    meta = {(r["review_id"], r["work_id"], r["kind"]): p for r, p in zip(recs, pct)}

    L = ["# Judge calibration under covariate shift", "",
         "Eligible share of never-screened (abstract-bearing) records the LLM agents included, with the",
         "judge's error rates corrected for differences between calibration and target records.",
         "Similarity = within-review percentile of BM25(review question + criteria, title + abstract);",
         f"bins at {', '.join(str(c) for c in CUTS)}. CI: cluster bootstrap over reviews.", ""]
    for fname in JUDGES:
        path = WEBFP / fname
        if not path.exists():
            L += [f"_{fname} not found_", ""]
            continue
        rows = []
        for j in map(json.loads, open(path)):
            key = (j["review_id"], j["work_id"], j["kind"])
            if j["eligible"] is None or not j["has_abstract"] or key not in meta:
                continue
            rows.append({"review": j["review_id"], "kind": j["kind"], "y": bool(j["eligible"]), "bin": bin_of(meta[key])})
        name = fname.split("_", 1)[1].rsplit(".", 1)[0]
        L += [f"## {name}", "", "Where the records sit (share of each class per similarity bin) and the judge's yes-rate:", "",
              "| similarity bin | gold n | gold yes | excluded n | excluded yes | unscreened n | unscreened yes |",
              "|---|---|---|---|---|---|---|"]
        labels = ["< .25", ".25-.50", ".50-.75", ">= .75"]
        for b in range(len(CUTS) + 1):
            cells = []
            for k in ("gold", "excluded", "unscreened"):
                ys = [r["y"] for r in rows if r["kind"] == k and r["bin"] == b]
                cells += [str(len(ys)), f"{statistics.fmean(ys):.2f}" if ys else "-"]
            L.append(f"| {labels[b]} | " + " | ".join(cells) + " |")
        by_rev = defaultdict(list)
        for r in rows:
            by_rev[r["review"]].append(r)
        revs = sorted(by_rev)
        L += ["", f"{len(revs)} reviews; {sum(r['kind'] == 'unscreened' for r in rows)} unscreened records.", "",
              "| estimate | eligible share | 95% CI | P(eligible \\| judged eligible) | 95% CI |", "|---|---|---|---|---|"]
        boots = defaultdict(list)
        for _ in range(2000):
            sample = []
            for i, rv in enumerate(rng.choices(revs, k=len(revs))):
                sample += [{**r, "review": f"{rv}#{i}"} for r in by_rev[rv]]
            for label, rw, st in VARIANTS:
                boots[label].append(estimate(sample, rw, st))
            boots["per-review, shrunk (k=10)"].append(shrunk(sample))
        for label, rw, st in VARIANTS + (("per-review, shrunk (k=10)", None, None),):
            s, ppv = estimate(rows, rw, st) if rw is not None else shrunk(rows)
            bs = sorted(x[0] for x in boots[label])
            bp = sorted(x[1] for x in boots[label] if not math.isnan(x[1]))
            ci = lambda v: f"[{v[int(.025 * len(v))]:.2f}, {v[int(.975 * len(v)) - 1]:.2f}]"
            L.append(f"| {label} | {s:.2f} | {ci(bs)} | {ppv:.2f} | {ci(bp)} |")
        L.append("")
    L += ["Reading: if the corrected shares sit inside the naive CI, the observable shift does not change",
          "the conclusion. The beyond-search counts in `web_robust.md` section 4 scale with",
          "P(eligible | judged eligible), so multiply them by (corrected PPV / naive PPV).",
          "Remaining assumption: given similarity and review, the judge errs on never-screened records",
          "as it does on screened ones. Only human labels on never-screened records can test that."]
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
