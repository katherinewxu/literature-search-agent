"""LLM eligibility judge for live-web inclusions, calibrated on records with human labels.

Usage:
  python scripts/web_judge.py judge  CONFIG.json  [--limit-batches N]   # paid; writes judgements.jsonl
  python scripts/web_judge.py report CONFIG.json  OUT.md               # free; tables from judgements

Records (from scripts/web_fp_labels.py output plus the web runs):
  gold        gold studies the LLM web agents found        -> should be judged eligible
  excluded    web inclusions the review authors screened and rejected (hard negatives)
  unscreened  web inclusions the review's own search never retrieved (label unknown)
All three are shuffled together, batched per review, and shown by position number (never by
OpenAlex id), so the judge cannot tell calibration records from unknown ones. The judge sees the
review question, criteria, and each record's title/abstract; never any label.

Sensitivity is measured on gold, specificity on the hard negatives, and the share of eligible
unscreened records is corrected with Rogan-Gladen. The report gives recall against the gold set
and against gold plus the judged-eligible pool (TREC-style pooling over every system).
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/scratch/users/kwx04/literature-search-agent")
WEBFP = ROOT / "analysis" / "webfp"
SETS = [("eval (8 reviews)", "synergy-v3-web-r5", "synergy-v3"), ("transfer (24 reviews)", "ext-transfer-web-r5", "synergy-ext-transfer-v2")]
LLM_AGENTS = ("haiku-5.5", "sonnet-5.5")

PROMPT = """You are an expert screener for a systematic review. Judge each record against the review's
eligibility criteria as the review authors would at title/abstract screening.

Review question: {question}

Eligibility criteria: {criteria}

Mark a record eligible only if it plausibly meets every criterion (population, intervention or
exposure, comparator, outcomes, study design, and any date, language or publication-type limits).
Topical relevance alone is not enough. If a record has no abstract, judge from the title and be
conservative. Answer with JSON only:
{{"decisions": [{{"n": 1, "eligible": true, "reason": "<= 12 words, name the failed criterion if not"}}, ...]}}
with one entry per record, in order.

{records}"""


def load_reviews(data):
    out = {}
    for line in open(ROOT / "data" / data / "reviews.jsonl"):
        r = json.loads(line)
        out[r["id"]] = r
    return out


def gold_records():
    """Unique (review, gold work) pairs that an LLM web agent included, with title/abstract."""
    recs = []
    for _, run_dir, data in SETS:
        reviews = load_reviews(data)
        pairs = set()
        for path in sorted((ROOT / "runs" / run_dir).glob("*.json")):
            run = json.loads(path.read_text())
            if run["agent"] not in LLM_AGENTS or run["matrix"].get("repeat", 0) != 0:
                continue
            rid = run["episode"]["review_id"]
            pairs.update((rid, w) for w in run["predictions"] if w in set(reviews[rid]["gold_ids"]))
        wanted = {w for _, w in pairs}
        meta = {}
        for line in open(ROOT / "data" / data / "works.jsonl"):
            w = json.loads(line)
            if w["id"] in wanted:
                meta[w["id"]] = w
        for rid, wid in sorted(pairs):
            m = meta[wid]
            recs.append({"review_id": rid, "work_id": wid, "kind": "gold", "title": m.get("title"),
                         "abstract": m.get("abstract"), "year": m.get("publication_year")})
    return recs


def all_records():
    recs = gold_records()
    for name, kind in (("screened_excluded", "excluded"), ("unscreened", "unscreened")):
        for line in open(WEBFP / f"{name}.jsonl"):
            r = json.loads(line)
            recs.append({"review_id": r["review_id"], "work_id": r["work_id"], "kind": kind, "title": r["title"],
                         "abstract": r.get("abstract"), "year": r.get("year")})
    return recs


def batches(recs, size, seed=0):
    by_review = defaultdict(list)
    for r in recs:
        by_review[r["review_id"]].append(r)
    rng = random.Random(seed)
    out = []
    for rid in sorted(by_review):
        items = by_review[rid]
        rng.shuffle(items)
        out += [items[i:i + size] for i in range(0, len(items), size)]
    rng.shuffle(out)  # so a --limit-batches pilot spans reviews
    return out


def judge(config_path, limit):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from litsearch.matrix import _backend

    config = json.loads(Path(config_path).read_text())
    backend = _backend(config["judge"])
    reviews = {}
    for _, _, data in SETS:
        reviews.update(load_reviews(data))
    out_path = WEBFP / config["out"]
    done = set()
    if out_path.exists():
        done = {(j["review_id"], j["work_id"]) for j in map(json.loads, open(out_path))}
    todo = batches(all_records(), int(config.get("batch", 8)))
    if limit:
        todo = todo[:limit]
    bad = 0
    with open(out_path, "a") as fh:
        for chunk in todo:
            if all((r["review_id"], r["work_id"]) in done for r in chunk):
                continue
            rv = reviews[chunk[0]["review_id"]]
            records = "\n\n".join(
                f"[{i}] {r['title']} ({r.get('year') or 'n.d.'})\n{(r.get('abstract') or '(no abstract)')[:1200]}"
                for i, r in enumerate(chunk, 1))
            prompt = PROMPT.format(question=rv.get("question") or rv.get("title"), criteria=rv["eligibility_criteria"][:2500], records=records)
            text = backend.complete([{"role": "system", "content": "You answer with JSON only."}, {"role": "user", "content": prompt}],
                                    temperature=0.0, seed=None)
            try:
                decisions = {int(d["n"]): d for d in json.loads(re.search(r"\{.*\}", text, re.S).group(0))["decisions"]}
            except (AttributeError, ValueError, KeyError, TypeError):
                decisions = {}
            for i, r in enumerate(chunk, 1):
                d = decisions.get(i)
                if d is None:
                    bad += 1
                fh.write(json.dumps({"review_id": r["review_id"], "work_id": r["work_id"], "kind": r["kind"],
                                     "eligible": None if d is None else bool(d.get("eligible")),
                                     "reason": None if d is None else d.get("reason"),
                                     "has_abstract": bool(r.get("abstract"))}) + "\n")
            fh.flush()
    print(f"judged {len(todo)} batches; {bad} records unparsed; usage {backend.usage}")


def report(config_path, out_md):
    config = json.loads(Path(config_path).read_text())
    J = {}
    for j in map(json.loads, open(WEBFP / config["out"])):
        if j["eligible"] is not None:
            J[(j["review_id"], j["work_id"])] = j
    rate = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    L = [f"# LLM judge on live-web inclusions ({config['judge']['model']})", ""]
    # The judge says yes to nearly every record without an abstract, so calibration and the pool use
    # abstract-bearing records only; unscreened records without an abstract never enter the pool.
    A = {k: j for k, j in J.items() if j["has_abstract"]}
    sens = rate([j["eligible"] for j in A.values() if j["kind"] == "gold"])
    spec = 1 - rate([j["eligible"] for j in A.values() if j["kind"] == "excluded"])
    q = rate([j["eligible"] for j in A.values() if j["kind"] == "unscreened"])
    p = min(1.0, max(0.0, (q + spec - 1) / (sens + spec - 1)))
    ppv = sens * p / (sens * p + (1 - spec) * (1 - p)) if p > 0 else 0.0
    n = lambda k: sum(1 for j in A.values() if j["kind"] == k)
    L += ["Calibration and pool use records with an abstract only.", "", "| quantity | value | n |", "|---|---|---|",
          f"| sensitivity (gold judged eligible) | {sens:.2f} | {n('gold')} |",
          f"| specificity (screened-excluded judged ineligible) | {spec:.2f} | {n('excluded')} |",
          f"| unscreened judged eligible (raw) | {q:.2f} | {n('unscreened')} |",
          f"| unscreened truly eligible (Rogan-Gladen) | {p:.2f} | |",
          f"| P(eligible \\| judged eligible), unscreened | {ppv:.2f} | |", ""]
    for kind in ("gold", "excluded", "unscreened"):
        for ab in (True, False):
            xs = [j["eligible"] for j in J.values() if j["kind"] == kind and j["has_abstract"] == ab]
            L.append(f"- {kind}, {'with' if ab else 'without'} abstract: {rate(xs):.2f} judged eligible (n={len(xs)})")
    L += ["", "Specificity is measured on hard negatives (records the review's own search retrieved), so it is",
          "probably lower than on unscreened records, which makes the Rogan-Gladen share conservative.", ""]
    for title, run_dir, data in SETS:
        reviews = load_reviews(data)
        runs = []
        for path in sorted((ROOT / "runs" / run_dir).glob("*.json")):
            run = json.loads(path.read_text())
            if run["matrix"].get("repeat", 0) == 0:
                runs.append(run)
        pool = defaultdict(set)  # judged-eligible unscreened works, pooled over every system
        for run in runs:
            rid = run["episode"]["review_id"]
            for w in run["predictions"]:
                j = A.get((rid, w))
                if j and j["kind"] == "unscreened" and j["eligible"]:
                    pool[rid].add(w)
        cells = defaultdict(list)
        for run in runs:
            ep = run["episode"]
            rid = ep["review_id"]
            gold = set(reviews[rid]["gold_ids"])
            pred = set(run["predictions"])
            tp = len(pred & gold)
            extra = len(pred & pool[rid])
            denom_pool = len(gold) + ppv * len(pool[rid])
            cells[(run["agent"], ep["mode"], ep["budget"])].append(
                (tp / len(gold), (tp + ppv * extra) / denom_pool if denom_pool else 0.0,
                 (tp + ppv * extra) / len(pred) if pred else 0.0, run["metrics"]["precision"]))
        L += [f"## {title}", "", f"Pool: {sum(len(v) for v in pool.values())} judged-eligible unscreened works, weighted by P(eligible|judged)={ppv:.2f}.", "",
              "| agent | mode | B | recall (gold) | recall (gold + pool) | precision (gold) | precision (adjusted) |", "|---|---|---|---|---|---|---|"]
        for key in sorted(cells):
            xs = cells[key]
            m = [sum(x[i] for x in xs) / len(xs) for i in range(4)]
            L.append(f"| {key[0]} | {key[1]} | {key[2]} | {m[0]:.3f} | {m[1]:.3f} | {m[3]:.3f} | {m[2]:.3f} |")
        L.append("")
    Path(out_md).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd, cfg = sys.argv[1], sys.argv[2]
    if cmd == "judge":
        lim = int(sys.argv[sys.argv.index("--limit-batches") + 1]) if "--limit-batches" in sys.argv else 0
        judge(cfg, lim)
    else:
        report(cfg, sys.argv[3])
