"""Open-graph condition: widen each review's universe beyond its screening pool.

SYNERGY pools are the reviews' own search results, so a closed-pool environment
hides every path that leaves the pool. Here each review gains the out-of-pool works
its visible pool cites (ranked by how many pool papers cite them, gold-blind),
fetched from OpenAlex with their own references, so citation chasing can cross
outside the pool and search faces realistic distractors. Gold sets are unchanged;
leakage filters (review paper, its citers, post-cutoff work) still apply.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from .data import _paper_from_openalex_work, _with_reverse_edges
from .environment import allowed_ids_for
from .openalex import OpenAlexBudgetExhausted, OpenAlexClient
from .store import CorpusStore

WORK_FIELDS = (
    "id,display_name,title,abstract_inverted_index,publication_year,publication_date,"
    "referenced_works,doi,primary_location"
)


def open_graph_candidates(store: CorpusStore, review, cap: int) -> list[str]:
    """Out-of-pool references of the visible pool, most-cited first (ties by id)."""
    visible = allowed_ids_for(store, review)
    counts: Counter[str] = Counter()
    for paper_id in visible:
        for reference_id in store.papers[paper_id].references:
            if reference_id not in review.candidate_ids:
                counts[reference_id] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [paper_id for paper_id, _ in ranked[:cap]]


def build_open_graph(
    store: CorpusStore,
    client: OpenAlexClient,
    *,
    cap_per_review: int = 5000,
    pool_multiple: float = 1.0,
    batch_size: int = 50,
) -> tuple[CorpusStore, dict[str, Any]]:
    """Return a store whose reviews' ``candidate_ids`` include bounded one-hop neighbors.

    Each review receives at most ``min(cap_per_review, pool_multiple * |pool|)`` new works.
    """
    wanted: dict[str, list[str]] = {}
    for review in store.reviews.values():
        cap = min(cap_per_review, int(pool_multiple * len(review.candidate_ids)))
        wanted[review.id] = open_graph_candidates(store, review, cap)
    missing = sorted({i for ids in wanted.values() for i in ids if i not in store.papers})
    papers = dict(store.papers)
    stats = {"requested": len(missing), "fetched": 0, "failed_batches": 0}
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset : offset + batch_size]
        try:
            works = client.works_batch(batch, select=WORK_FIELDS)
        except OpenAlexBudgetExhausted:
            raise
        except Exception:
            stats["failed_batches"] += 1
            continue
        for work in works:
            if not work.get("id"):
                continue
            paper = _paper_from_openalex_work(work)
            if paper.id not in papers:
                papers[paper.id] = paper
                stats["fetched"] += 1
    reviews = []
    per_review: dict[str, Any] = {}
    for review in store.reviews.values():
        added = [i for i in wanted[review.id] if i in papers]
        reviews.append(replace(review, candidate_ids=review.candidate_ids | frozenset(added)))
        per_review[review.id] = {
            "pool": len(review.candidate_ids),
            "requested": len(wanted[review.id]),
            "added": len(added),
        }
    stats["per_review"] = per_review
    return CorpusStore(_with_reverse_edges(papers).values(), reviews), stats
