from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .models import TraceEvent


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    recall: float
    precision: float
    f1: float
    aurc: float
    charged_cost: int
    unique_exposed: int
    unique_read: int
    included_count: int
    true_positives: int
    gold_count: int
    wss_at_95: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def precision_recall(included: set[str], gold: set[str] | frozenset[str]) -> tuple[float, float]:
    true_positives = len(included & set(gold))
    recall = true_positives / len(gold) if gold else 0.0
    precision = true_positives / len(included) if included else 0.0
    return precision, recall


def area_under_recall_cost(
    trace: Iterable[TraceEvent], gold: set[str] | frozenset[str], budget: int
) -> float:
    if budget <= 0:
        return 0.0
    events = list(trace)
    cost = 0
    prior_recall = 0.0
    area = 0.0
    for event in events:
        next_cost = min(budget, cost + event.charged_cost)
        area += prior_recall * (next_cost - cost)
        _, prior_recall = precision_recall(set(event.included_ids), gold)
        cost = next_cost
        if cost >= budget:
            break
    area += prior_recall * max(0, budget - cost)
    return area / budget


def compute_metrics(
    *,
    included: set[str],
    gold: set[str] | frozenset[str],
    trace: Iterable[TraceEvent],
    budget: int,
    exposed: set[str],
    read_ids: set[str],
    candidate_count: int,
) -> EpisodeMetrics:
    trace = list(trace)
    precision, recall = precision_recall(included, gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    wss = None
    if recall >= 0.95 and candidate_count > 0:
        wss = 0.95 - len(exposed) / candidate_count
    return EpisodeMetrics(
        recall=recall,
        precision=precision,
        f1=f1,
        aurc=area_under_recall_cost(trace, gold, budget),
        charged_cost=sum(event.charged_cost for event in trace),
        unique_exposed=len(exposed),
        unique_read=len(read_ids),
        included_count=len(included),
        true_positives=len(included & set(gold)),
        gold_count=len(gold),
        wss_at_95=wss,
    )

