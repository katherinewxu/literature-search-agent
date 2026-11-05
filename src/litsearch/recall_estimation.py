"""Can a search agent tell how much it missed? Post-hoc recall estimators for finished episodes.

Two estimators, both computable at the end of an episode without gold labels:

* ``saturation``: fit N(t) = A (1 - exp(-k t)) to the agent's cumulative inclusions over charged
  cost and report included / A (a yield-curve / knee-style estimate; no guarantee).
* ``sample``: screen (i) the included set S, (ii) a random sample of exposed-but-not-included
  papers X, and (iii) a random sample of never-exposed papers U in the agent's search universe,
  with an LLM screener. Estimated relevant total R = |S| p_S + |X| p_X + |U| p_U, recall = |S| p_S / R.
  A one-sided Clopper-Pearson upper bound on p_X and p_U gives a lower confidence bound on recall
  (in the spirit of the statistical stopping rules of Callaghan & Mueller-Hansen, 2020).

Gold labels are used only to score the estimators (true recall), and by the ``oracle`` screener,
which replaces the LLM with gold labels to separate sampling error from screening error.
"""

from __future__ import annotations

import json
import math
import random
import re
from typing import Any, Callable

from .environment import allowed_ids_for
from .store import CorpusStore

Screener = Callable[[str, list[str]], set[str]]  # (review_id, paper_ids) -> ids judged eligible


# -- statistics ----------------------------------------------------------------------
def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) Clopper-Pearson upper bound for a binomial proportion."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def saturation_estimate(run: dict[str, Any]) -> float | None:
    """included / asymptote of an exponential yield curve fit over charged cost."""
    points, spent, included = [], 0, 0
    for event in run["trace"]:
        spent += int(event.get("charged_cost") or 0)
        included = len(event.get("included_ids") or [])
        points.append((spent, included))
    if not points or included == 0 or spent == 0:
        return None
    best = None
    for k in [10 ** (x / 20) for x in range(-60, 21)]:  # k from 1e-3 to 10 per unit cost
        basis = [1 - math.exp(-k * t) for t, _ in points]
        denom = sum(b * b for b in basis)
        if denom == 0:
            continue
        a = sum(b * y for b, (_, y) in zip(basis, points)) / denom
        err = sum((a * b - y) ** 2 for b, (_, y) in zip(basis, points))
        if best is None or err < best[0]:
            best = (err, a)
    if best is None or best[1] <= 0:
        return None
    return min(1.0, included / best[1])


# -- sample-based estimator -------------------------------------------------------------
def sample_estimate(
    run: dict[str, Any],
    store: CorpusStore,
    screen: Screener,
    *,
    n_excluded: int = 30,
    n_unexposed: int = 80,
    max_included: int = 120,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    review_id = run["episode"]["review_id"]
    review = store.review(review_id)
    rng = random.Random(f"{seed}:{review_id}:{run['agent']}:{run['episode']['budget']}")
    included = set(run["predictions"])
    exposed: set[str] = set(run.get("seed_ids") or [])
    for event in run["trace"]:
        exposed.update(event.get("result_ids") or [])
    exposed |= included
    allowed = allowed_ids_for(store, review)
    excluded = sorted((exposed - included) & allowed)
    unexposed = sorted(allowed - exposed)

    def sample(ids: list[str], n: int) -> list[str]:
        return ids if len(ids) <= n else rng.sample(ids, n)

    s_ids, x_ids, u_ids = sample(sorted(included), max_included), sample(excluded, n_excluded), sample(unexposed, n_unexposed)
    s_set = screen(review_id, s_ids) if s_ids else set()
    x_set = screen(review_id, x_ids) if x_ids else set()
    u_set = screen(review_id, u_ids) if u_ids else set()
    s_yes, x_yes, u_yes = len(s_set), len(x_set), len(u_set)
    p_s = s_yes / len(s_ids) if s_ids else 0.0
    p_x = x_yes / len(x_ids) if x_ids else 0.0
    p_u = u_yes / len(u_ids) if u_ids else 0.0
    found = len(included) * p_s
    total = found + len(excluded) * p_x + len(unexposed) * p_u
    upper = found + len(excluded) * cp_upper(x_yes, len(x_ids), alpha / 2) + len(unexposed) * cp_upper(u_yes, len(u_ids), alpha / 2)
    gold = set(review.gold_ids)
    return {
        "recall_hat": found / total if total > 0 else None,
        "recall_lower": found / upper if upper > 0 else None,
        "sizes": {"included": len(included), "excluded": len(excluded), "unexposed": len(unexposed)},
        "screened_yes": {"included": [s_yes, len(s_ids)], "excluded": [x_yes, len(x_ids)], "unexposed": [u_yes, len(u_ids)]},
        "true_recall": len(included & gold) / max(1, len(gold)),
        # Per-stratum sampled ids, screener decisions and gold, for calibrating the screener.
        "strata": {
            name: {"n_total": total_n, "sampled": ids, "yes": sorted(chosen), "gold": sorted(set(ids) & gold)}
            for name, total_n, ids, chosen in (
                ("included", len(included), s_ids, s_set),
                ("excluded", len(excluded), x_ids, x_set),
                ("unexposed", len(unexposed), u_ids, u_set),
            )
        },
        # What the agent could still find vs. what lies outside its universe (should be ~0 here).
        "true_missed": {"excluded": len((exposed - included) & gold), "unexposed": len((allowed - exposed) & gold)},
    }


def oracle_screener(store: CorpusStore) -> Screener:
    def screen(review_id: str, ids: list[str]) -> set[str]:
        gold = store.review(review_id).gold_ids
        return {i for i in ids if i in gold}

    return screen


SCREEN_PROMPT = """You are screening records for a systematic review.

Review question: {question}

Eligibility criteria: {criteria}

For each record below decide whether it would be INCLUDED at title/abstract screening. When a
record has no abstract, decide from the title. Answer with JSON only: {{"include": ["id", ...]}}.

{records}"""


def llm_screener(store: CorpusStore, backend, batch: int = 10) -> Screener:
    def screen(review_id: str, ids: list[str]) -> set[str]:
        review = store.review(review_id)
        out: set[str] = set()
        for start in range(0, len(ids), batch):
            chunk = ids[start : start + batch]
            records = []
            for paper_id in chunk:
                paper = store.papers[paper_id]
                abstract = (paper.abstract or "(no abstract)")[:900]
                records.append(f"[{paper_id}] {paper.title} ({paper.publication_year})\n{abstract}")
            prompt = SCREEN_PROMPT.format(
                question=review.question, criteria=review.eligibility_criteria[:1500], records="\n\n".join(records)
            )
            text = backend.complete(
                [{"role": "system", "content": "You answer with JSON only."}, {"role": "user", "content": prompt}],
                temperature=0.0,
                seed=None,
            )
            match = re.search(r"\{.*\}", text, re.S)
            try:
                chosen = json.loads(match.group(0)).get("include", []) if match else []
            except json.JSONDecodeError:
                chosen = re.findall(r"W\d+", text)
            out.update(i for i in chosen if i in chunk)
        return out

    return screen
