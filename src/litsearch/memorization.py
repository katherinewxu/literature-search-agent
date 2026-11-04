"""Closed-book memorization probe.

The model gets exactly what the agent gets at the start of an episode (review question and
eligibility criteria) but no tools, and is asked to list the studies it believes the review
included. Generated titles are matched to the review's candidate pool by normalized-title
similarity; matched gold / |gold| is closed-book recall. Gold labels are used only by this
scorer, never in the prompt.

High closed-book recall on a review means the model can name its included studies from
memory, so tool-based recall on that review may be inflated by memorization.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .store import CorpusStore

PROMPT = """You are helping to reproduce a published systematic review WITHOUT any search tools.

Review question: {question}

Eligibility criteria: {criteria}

From memory only, list up to {n} primary studies that you believe were INCLUDED in this
review. If you do not know, give your best guesses of real studies that would qualify. Answer
with a JSON list only, e.g. [{{"title": "...", "year": 2015}}, ...]."""


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _tokens(value: str) -> set[str]:
    return {t for t in _norm(value).split() if len(t) > 2}


def title_match(a: str, b: str) -> float:
    """Max of character-level ratio and token Jaccard on normalized titles."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    ta, tb = _tokens(a), _tokens(b)
    jaccard = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    return max(SequenceMatcher(None, na, nb).ratio(), jaccard)


def parse_titles(text: str) -> list[str]:
    start, end = text.find("["), text.rfind("]")
    if start >= 0 and end > start:
        try:
            items = json.loads(text[start : end + 1])
            return [str(i.get("title", "")) if isinstance(i, dict) else str(i) for i in items if i]
        except json.JSONDecodeError:
            pass
    return [m.strip() for m in re.findall(r'"title"\s*:\s*"([^"]+)"', text)]


def score_review(store: CorpusStore, review_id: str, titles: list[str], threshold: float = 0.8) -> dict[str, Any]:
    review = store.review(review_id)
    pool = [store.papers[i] for i in (review.candidate_ids or store.papers) if i in store.papers]
    gold_hit: set[str] = set()
    pool_hit: set[str] = set()
    best_gold: list[float] = []  # per generated title, best similarity to any gold title
    gold_titles = [store.papers[i].title for i in review.gold_ids if i in store.papers]
    for title in titles:
        best_gold.append(max((title_match(title, g) for g in gold_titles), default=0.0))
        if len(_norm(title)) < 15:
            continue
        best, best_id = 0.0, None
        for paper in pool:
            score = title_match(title, paper.title)
            if score > best:
                best, best_id = score, paper.id
        if best_id and best >= threshold:
            pool_hit.add(best_id)
            if best_id in review.gold_ids:
                gold_hit.add(best_id)
    return {
        "review_id": review_id,
        "generated": len(titles),
        "matched_pool": len(pool_hit),
        "matched_gold": len(gold_hit),
        "gold": len(review.gold_ids),
        "closed_book_recall": len(gold_hit) / max(1, len(review.gold_ids)),
        "closed_book_precision": len(gold_hit) / max(1, len(titles)),
        "gold_hit_ids": sorted(gold_hit),
        "titles": titles,
        "best_gold_similarity": [round(x, 3) for x in best_gold],
    }


def run_probe(config: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    from .matrix import _backend

    store = CorpusStore.load(config["data_dir"])
    reviews = config.get("reviews") or sorted(store.reviews)
    n = int(config.get("n_titles", 50))
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for agent in config["agents"]:
        backend = _backend(agent)
        for review_id in reviews:
            review = store.review(review_id)
            prompt = PROMPT.format(question=review.question, criteria=review.eligibility_criteria[:1500], n=n)
            text = backend.complete(
                [{"role": "system", "content": "You answer with JSON only."}, {"role": "user", "content": prompt}],
                temperature=0.0,
                seed=None,
            )
            row = {"model": agent["label"], **score_review(store, review_id, parse_titles(text))}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in {"gold_hit_ids", "titles", "best_gold_similarity"}}), flush=True)
    output.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rows
