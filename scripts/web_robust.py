"""Reviewer-proofing for the live-web results. Usage: python scripts/web_robust.py OUT.md

1. Paired comparisons over reviews: mean difference, 95% bootstrap CI (10,000 resamples of
   reviews), and an exact two-sided sign test. Covers the v2 environment (eval and held-out
   transfer) and LLM agents vs the two non-LLM web baselines.
2. Judge sensitivity: the Rogan-Gladen eligible share of never-screened records across a grid of
   sensitivity/specificity, plus a bootstrap CI that resamples the calibration records.
3. Cost: LLM dollars and calls per episode and per gold study found, from the per-call usage that
   the response cache stores (recomputing each turn's cache key from the run's config).
4. Beyond the review's search: per episode, gold found vs never-screened records the judge calls
   eligible (expected count = judged-eligible x PPV). Abstract-level eligibility only.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/scratch/users/kwx04/literature-search-agent")
RUNS = ROOT / "runs"
WEBFP = ROOT / "analysis" / "webfp"
DATA = {"eval": "synergy-v3", "transfer": "synergy-ext-transfer-v2"}
rng = random.Random(20261113)


def gold_for(split):
    return {json.loads(l)["id"]: set(json.loads(l)["gold_ids"]) for l in open(ROOT / "data" / DATA[split] / "reviews.jsonl")}


GOLD = {s: gold_for(s) for s in DATA}


def load(run_dir, agent=None, mode=None, budget=None):
    out = []
    for p in sorted((RUNS / run_dir).glob("*.json")):
        r = json.loads(p.read_text())
        ep = r["episode"]
        if r["matrix"].get("repeat", 0):
            continue
        if (agent and r["agent"] != agent) or (mode and ep["mode"] != mode) or (budget and ep["budget"] != budget):
            continue
        out.append(r)
    return out


def recall_by_review(runs, split):
    g = GOLD[split]
    return {r["episode"]["review_id"]: len(set(r["predictions"]) & g[r["episode"]["review_id"]]) / len(g[r["episode"]["review_id"]]) for r in runs}


def boot_ci(diffs, n=10000):
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def sign_test(diffs):
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n, k = pos + neg, min(pos, neg)
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(label, a, b):
    """a, b: review -> recall; reports b - a."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return f"| {label} | 0 | - | - | - | - |"
    d = [b[k] - a[k] for k in keys]
    lo, hi = boot_ci(d)
    return (f"| {label} | {len(d)} | {statistics.fmean(d):+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
            f"{sum(x > 0 for x in d)}/{sum(x < 0 for x in d)} | {sign_test(d):.3f} |")


def section_paired():
    L = ["## 1. Paired comparisons over reviews (recall, B - A)", "",
         "CI: bootstrap over reviews; up/down: reviews where B beats / trails A; p: exact two-sided sign test.", "",
         "| comparison | reviews | mean diff | 95% CI | up/down | sign-test p |", "|---|---|---|---|---|---|"]
    rows = [
        ("eval: v2 - original, haiku citation B=10", "eval", ("synergy-v3-web-r5", "haiku-5.5", "citation", 10), ("synergy-v3-web-v2", "haiku-5.5", "citation", 10)),
        ("eval: v2 - original, haiku citation B=20", "eval", ("synergy-v3-web-r5", "haiku-5.5", "citation", 20), ("synergy-v3-web-v2", "haiku-5.5", "citation", 20)),
        ("eval: v2 - original, haiku citation B=40", "eval", ("synergy-v3-web-r5", "haiku-5.5", "citation", 40), ("synergy-v3-web-v2", "haiku-5.5", "citation", 40)),
        ("eval: v2 - original, sonnet citation B=40", "eval", ("synergy-v3-web-r5", "sonnet-5.5", "citation", 40), ("synergy-v3-web-v2", "sonnet-5.5", "citation", 40)),
        ("eval: v2 - original, haiku hybrid B=40", "eval", ("synergy-v3-web-r5", "haiku-5.5", "hybrid", 40), ("synergy-v3-web-v2", "haiku-5.5", "hybrid", 40)),
        ("HELD-OUT transfer: v2 - original, haiku citation B=40", "transfer", ("ext-transfer-web-citation-orig", "haiku-5.5", "citation", 40), ("ext-transfer-web-v2", "haiku-5.5", "citation", 40)),
        ("HELD-OUT transfer: v2 - original, haiku hybrid B=20", "transfer", ("ext-transfer-web-r5", "haiku-5.5", "hybrid", 20), ("ext-transfer-web-v2", "haiku-5.5", "hybrid", 20)),
        ("HELD-OUT transfer: v2 - original, haiku hybrid B=40", "transfer", ("ext-transfer-web-r5", "haiku-5.5", "hybrid", 40), ("ext-transfer-web-v2", "haiku-5.5", "hybrid", 40)),
        ("eval: sonnet hybrid - snowball hybrid, B=40", "eval", ("synergy-v3-web-snowball", "snowball-tuned", "hybrid", 40), ("synergy-v3-web-r5", "sonnet-5.5", "hybrid", 40)),
        ("eval: sonnet hybrid - OpenAlex search, B=40", "eval", ("synergy-v3-web-r5", "openalex-search", "keyword", 40), ("synergy-v3-web-r5", "sonnet-5.5", "hybrid", 40)),
        ("eval: haiku citation - snowball citation, B=40", "eval", ("synergy-v3-web-snowball", "snowball-tuned", "citation", 40), ("synergy-v3-web-r5", "haiku-5.5", "citation", 40)),
        ("transfer: haiku hybrid - snowball hybrid, B=40", "transfer", ("ext-transfer-web-snowball", "snowball-tuned", "hybrid", 40), ("ext-transfer-web-r5", "haiku-5.5", "hybrid", 40)),
        ("transfer: haiku hybrid - OpenAlex search, B=40", "transfer", ("ext-transfer-web-r5", "openalex-search", "keyword", 40), ("ext-transfer-web-r5", "haiku-5.5", "hybrid", 40)),
    ]
    for label, split, a, b in rows:
        L.append(paired(label, recall_by_review(load(*a), split), recall_by_review(load(*b), split)))
    # Mean recall of every web system, for the baseline table.
    L += ["", "Mean recall per web system (original environment unless noted):", "", "| split | system | mode | B | n | recall |", "|---|---|---|---|---|---|"]
    for split, dirs in (("eval", ["synergy-v3-web-r5", "synergy-v3-web-snowball", "synergy-v3-web-v2"]),
                        ("transfer", ["ext-transfer-web-r5", "ext-transfer-web-snowball", "ext-transfer-web-citation-orig", "ext-transfer-web-v2"])):
        for d in dirs:
            groups = defaultdict(list)
            for r in load(d):
                groups[(r["agent"], r["episode"]["mode"], r["episode"]["budget"])].append(r)
            for key in sorted(groups):
                rec = recall_by_review(groups[key], split)
                tag = " (v2 env)" if d.endswith("v2") else ""
                L.append(f"| {split} | {key[0]}{tag} | {key[1]} | {key[2]} | {len(rec)} | {statistics.fmean(rec.values()):.3f} |")
    return L


def rg(q, sens, spec):
    return min(1.0, max(0.0, (q + spec - 1) / (sens + spec - 1)))


def section_judge():
    L = ["", "## 2. Judge sensitivity (Rogan-Gladen eligible share of never-screened records)", ""]
    for fname in ("judgements_gemini-3.8-flash.jsonl", "judgements_gpt-5.6-luna.jsonl"):
        recs = [j for j in map(json.loads, open(WEBFP / fname)) if j["eligible"] is not None and j["has_abstract"]]
        by = defaultdict(list)
        for j in recs:
            by[j["kind"]].append(bool(j["eligible"]))
        sens, spec, q = statistics.fmean(by["gold"]), 1 - statistics.fmean(by["excluded"]), statistics.fmean(by["unscreened"])
        boots = []
        for _ in range(5000):
            s = statistics.fmean(rng.choices(by["gold"], k=len(by["gold"])))
            sp = 1 - statistics.fmean(rng.choices(by["excluded"], k=len(by["excluded"])))
            qq = statistics.fmean(rng.choices(by["unscreened"], k=len(by["unscreened"])))
            boots.append(rg(qq, s, sp))
        boots.sort()
        L += [f"**{fname.split('_', 1)[1].rsplit('.', 1)[0]}**: measured sensitivity {sens:.2f}, specificity {spec:.2f}, "
              f"unscreened yes-rate {q:.2f}; eligible share {rg(q, sens, spec):.2f}, bootstrap 95% CI "
              f"[{boots[125]:.2f}, {boots[4874]:.2f}] (calibration sampling only).", "",
              "If specificity on never-screened records differs from the screened-excluded estimate:", "",
              "| sensitivity \\ specificity | " + " | ".join(f"{x:.2f}" for x in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90)) + " |",
              "|---" * 7 + "|"]
        for s in (0.80, round(sens, 2), 0.95):
            L.append(f"| {s:.2f} | " + " | ".join(f"{rg(q, s, sp):.2f}" for sp in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90)) + " |")
        L.append("")
    L.append("Eligibility here is abstract-level: gold studies were included after full-text review, so these shares are upper bounds on full-text eligibility.")
    return L


def turn_cost(run, cfg_cache):
    """Sum cached per-call usage for a run's turns. Returns (usd, calls, found_in_cache)."""
    cfg_path = run["matrix"]["config"]
    if cfg_path not in cfg_cache:
        p = Path(cfg_path)
        p = p if p.is_absolute() else Path.home() / "literature-search-agent" / p
        cfg_cache[cfg_path] = json.loads(p.read_text()) if p.exists() else None
    cfg = cfg_cache[cfg_path]
    if not cfg:
        return None
    ep = run["episode"]
    agents = [a for a in cfg["agents"] if a.get("label") == run["agent"] and ep["mode"] in a.get("modes", cfg.get("modes", []))]
    if not agents:
        return None
    a = agents[0]
    usd, hits = 0.0, 0
    for t in run.get("agent_turns") or []:
        payload = {"model": a["model"], "messages": t["messages"], "temperature": float(a.get("temperature", 0.0)),
                   "max_tokens": int(a.get("max_tokens", 600)), **(a.get("extra_body") or {})}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        f = Path(a.get("cache_dir", "")) / f"{key}.json"
        if f.exists():
            u = json.loads(f.read_text()).get("usage") or {}
            c = float(u.get("cost") or 0.0)
            if not c and a.get("price_per_mtok"):
                c = (int(u.get("prompt_tokens") or 0) * a["price_per_mtok"][0] + int(u.get("completion_tokens") or 0) * a["price_per_mtok"][1]) / 1e6
            usd += c
            hits += 1
    return usd, len(run.get("agent_turns") or []), hits


def section_cost():
    L = ["", "## 3. Cost per episode and per gold study found (B=40, original environment)", "",
         "Dollars are recovered from the response cache; 'matched' is the share of turns whose cached usage was found.", "",
         "| split | system | mode | n | LLM calls/ep | $/episode | gold found/ep | $/gold found | wall s/ep | matched |", "|---|---|---|---|---|---|---|---|---|---|"]
    cfg_cache = {}
    for split, dirs in (("eval", ["synergy-v3-web-r5", "synergy-v3-web-snowball"]), ("transfer", ["ext-transfer-web-r5", "ext-transfer-web-snowball"])):
        groups = defaultdict(list)
        for d in dirs:
            for r in load(d, budget=40):
                groups[(r["agent"], r["episode"]["mode"])].append(r)
        for (agent, mode), runs in sorted(groups.items()):
            g = GOLD[split]
            found = [len(set(r["predictions"]) & g[r["episode"]["review_id"]]) for r in runs]
            costs = [turn_cost(r, cfg_cache) if r.get("agent_turns") else (0.0, 0, 0) for r in runs]
            costs = [c for c in costs if c is not None]
            usd = sum(c[0] for c in costs)
            calls = sum(c[1] for c in costs)
            matched = sum(c[2] for c in costs) / calls if calls else 1.0
            n = len(runs)
            per_gold = f"{usd / sum(found):.3f}" if sum(found) else "-"
            L.append(f"| {split} | {agent} | {mode} | {n} | {calls / n:.1f} | {usd / n:.3f} | {statistics.fmean(found):.1f} | {per_gold} | "
                     f"{statistics.fmean(r.get('runtime_seconds') or 0 for r in runs):.0f} | {matched:.2f} |")
    return L


def section_beyond():
    L = ["", "## 4. Beyond the review's own search (B=40, original environment, judge = Gemini)", "",
         "Per episode: gold found, and never-screened records included by the system that the judge calls eligible",
         "(expected eligible = judged-eligible x PPV). Abstract-level eligibility; records without an abstract are not counted.", ""]
    recs = {(j["review_id"], j["work_id"]): j for j in map(json.loads, open(WEBFP / "judgements_gemini-3.8-flash.jsonl"))}
    by = defaultdict(list)
    for j in recs.values():
        if j["eligible"] is not None and j["has_abstract"]:
            by[j["kind"]].append(bool(j["eligible"]))
    sens, spec, q = statistics.fmean(by["gold"]), 1 - statistics.fmean(by["excluded"]), statistics.fmean(by["unscreened"])
    p = rg(q, sens, spec)
    ppv = sens * p / (sens * p + (1 - spec) * (1 - p))
    L += [f"PPV used: {ppv:.2f}.", "", "| split | system | mode | n | gold found/ep | judged-eligible new/ep | expected eligible new/ep | new as % of gold found |", "|---|---|---|---|---|---|---|---|"]
    for split, d in (("eval", "synergy-v3-web-r5"), ("transfer", "ext-transfer-web-r5")):
        groups = defaultdict(list)
        for r in load(d, budget=40):
            groups[(r["agent"], r["episode"]["mode"])].append(r)
        for (agent, mode), runs in sorted(groups.items()):
            g = GOLD[split]
            found, new = [], []
            for r in runs:
                rid = r["episode"]["review_id"]
                found.append(len(set(r["predictions"]) & g[rid]))
                new.append(sum(1 for w in r["predictions"] if (j := recs.get((rid, w))) and j["kind"] == "unscreened" and j["has_abstract"] and j["eligible"]))
            f, nw = statistics.fmean(found), statistics.fmean(new)
            L.append(f"| {split} | {agent} | {mode} | {len(runs)} | {f:.1f} | {nw:.1f} | {nw * ppv:.1f} | {100 * nw * ppv / f if f else 0:.0f}% |")
    return L


def main():
    L = ["# Live-web robustness: paired tests, judge sensitivity, cost, beyond-search yield", ""]
    L += section_paired() + section_judge() + section_cost() + section_beyond()
    Path(sys.argv[1]).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
