from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from .models import (
    Action,
    ActionError,
    ActionKind,
    EpisodeSpec,
    Observation,
    Review,
    StepResult,
    TraceEvent,
)
from .retrieval import Retriever
from .store import CorpusStore


def allowed_ids_for(store: CorpusStore, review: Review) -> set[str]:
    """Papers an agent may see: the review's pool minus the review, its citers, and post-cutoff work."""
    values: set[str] = set()
    cutoff_date = review.search_cutoff_date
    if review.candidate_ids:
        papers = (store.papers[i] for i in review.candidate_ids if i in store.papers)
    else:
        papers = iter(store.papers.values())
    for paper in papers:
        if paper.id == review.review_paper_id:
            continue
        if paper.publication_year and paper.publication_year > review.search_cutoff_year:
            continue
        if cutoff_date and paper.publication_date and paper.publication_date > cutoff_date:
            continue
        if review.review_paper_id and review.review_paper_id in paper.references:
            continue
        values.add(paper.id)
    return values


class LiteratureSearchEnv:
    def __init__(self, store: CorpusStore, retriever: Retriever, spec: EpisodeSpec) -> None:
        self.store = store
        self.retriever = retriever
        self.spec = spec
        self.review = store.review(spec.review_id)
        self.remaining_budget = spec.budget
        self.included: set[str] = set()
        self.excluded: set[str] = set()
        self.exposed: set[str] = set()
        self.read_ids: set[str] = set()
        self.frontier: list[str] = []
        self.done = False
        self.action_count = 0
        self.last_result: dict[str, Any] = {"type": "reset", "papers": []}
        self.trace: list[TraceEvent] = []
        self.allowed_ids = self._allowed_ids()
        seed_query = f"{spec.question} {spec.eligibility_criteria}"
        seeds = retriever.search(seed_query, spec.seed_count, self.allowed_ids)
        self._add_frontier(seeds)
        self.exposed.update(seeds)
        self.seed_ids = tuple(seeds)
        self.seed_cost = max(1, math.ceil(len(seeds) / 10)) if seeds else 0
        self.remaining_budget -= self.seed_cost
        self.last_result = {"type": "seeds", "papers": self._metadata(seeds)}
        if self.seed_cost:
            self.trace.append(
                TraceEvent(
                    step=0,
                    action={"kind": "seed"},
                    charged_cost=self.seed_cost,
                    budget_after=self.remaining_budget,
                    result_ids=list(seeds),
                )
            )
        if self.remaining_budget <= 0:
            self.done = True

    def _allowed_ids(self) -> set[str]:
        return allowed_ids_for(self.store, self.review)

    def _metadata(self, ids: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
        return [self.store.papers[paper_id].metadata() for paper_id in ids if paper_id in self.store.papers]

    def _add_frontier(self, ids: list[str]) -> None:
        known = set(self.frontier)
        for paper_id in ids:
            if paper_id in self.allowed_ids and paper_id not in known:
                self.frontier.append(paper_id)
                known.add(paper_id)

    def observe(self) -> Observation:
        return Observation(
            question=self.spec.question,
            eligibility_criteria=self.spec.eligibility_criteria,
            remaining_budget=self.remaining_budget,
            mode=self.spec.mode,
            included_ids=tuple(sorted(self.included)),
            excluded_ids=tuple(sorted(self.excluded)),
            frontier=tuple(self._metadata(self.frontier[-50:])),
            last_result=self.last_result,
            action_count=self.action_count,
        )

    def _error(self, code: str, message: str, action: Action, cost: int = 0) -> StepResult:
        error = ActionError(code=code, message=message)
        self.last_result = {"type": "error", "code": code, "message": message}
        if cost:
            self.remaining_budget -= cost
        self.action_count += 1
        if self.remaining_budget <= 0:
            self.done = True
        self._record(action, cost, [], error)
        return StepResult(self.observe(), cost, self.remaining_budget, self.done, error)

    def _record(
        self, action: Action, cost: int, result_ids: list[str], error: ActionError | None = None
    ) -> None:
        self.trace.append(
            TraceEvent(
                step=len(self.trace) + 1,
                action=action.to_dict(),
                charged_cost=cost,
                budget_after=self.remaining_budget,
                result_ids=result_ids,
                included_ids=sorted(self.included),
                excluded_ids=sorted(self.excluded),
                error=asdict(error) if error else None,
            )
        )

    def step(self, action: Action) -> StepResult:
        action.validate()
        if self.done:
            return self._error("episode_done", "episode has already ended", action)
        if action.kind == ActionKind.SEARCH and self.spec.mode not in {"keyword", "hybrid"}:
            return self._error("tool_unavailable", "search is unavailable in citation mode", action)
        if action.kind in {ActionKind.EXPAND_REFS, ActionKind.EXPAND_CITERS} and self.spec.mode not in {
            "citation",
            "hybrid",
        }:
            return self._error("tool_unavailable", "citation expansion is unavailable", action)

        cost = self._prospective_cost(action)
        if cost > self.remaining_budget:
            return self._error(
                "insufficient_budget",
                f"action costs {cost}, only {self.remaining_budget} remains",
                action,
            )

        result_ids: list[str] = []
        if action.kind == ActionKind.SEARCH:
            k = min(action.k or 10, 10)
            ranked = self.retriever.search(
                action.query or "", min(len(self.allowed_ids), k + len(self.exposed)), self.allowed_ids
            )
            result_ids = [paper_id for paper_id in ranked if paper_id not in self.exposed][:k]
            self._add_frontier(result_ids)
            self.exposed.update(result_ids)
            self.last_result = {"type": "search", "papers": self._metadata(result_ids)}
        elif action.kind in {ActionKind.EXPAND_REFS, ActionKind.EXPAND_CITERS}:
            paper = self.store.paper(action.paper_id or "")
            if not paper or paper.id not in self.allowed_ids:
                return self._error("unknown_paper", "paper is absent or filtered", action)
            neighbors = paper.references if action.kind == ActionKind.EXPAND_REFS else paper.citers
            k = min(action.k or 20, 20)
            result_ids = [paper_id for paper_id in neighbors if paper_id in self.allowed_ids][:k]
            actual_cost = max(1, math.ceil(len(result_ids) / 10))
            cost = min(cost, actual_cost)
            self._add_frontier(result_ids)
            self.exposed.update(result_ids)
            self.last_result = {"type": action.kind.value, "papers": self._metadata(result_ids)}
        elif action.kind == ActionKind.READ:
            paper = self.store.paper(action.paper_id or "")
            if not paper or paper.id not in self.allowed_ids:
                return self._error("unknown_paper", "paper is absent or filtered", action)
            self.exposed.add(paper.id)
            self.read_ids.add(paper.id)
            result_ids = [paper.id]
            self.last_result = {
                "type": "read",
                "paper": {**paper.metadata(), "abstract": paper.abstract},
            }
        elif action.kind == ActionKind.INCLUDE:
            if action.paper_id not in self.exposed:
                return self._error("not_exposed", "paper must be exposed before inclusion", action)
            self.excluded.discard(action.paper_id or "")
            self.included.add(action.paper_id or "")
            self.last_result = {"type": "include", "paper_id": action.paper_id}
        elif action.kind == ActionKind.EXCLUDE:
            if action.paper_id not in self.exposed:
                return self._error("not_exposed", "paper must be exposed before exclusion", action)
            self.included.discard(action.paper_id or "")
            self.excluded.add(action.paper_id or "")
            self.last_result = {"type": "exclude", "paper_id": action.paper_id}
        elif action.kind == ActionKind.STOP:
            self.done = True
            self.last_result = {"type": "stop"}
        elif action.kind == ActionKind.NOOP:
            self.last_result = {"type": "agent_error", "message": "agent produced no valid action"}

        self.remaining_budget -= cost
        self.action_count += 1
        if self.remaining_budget <= 0:
            self.done = True
        self._record(action, cost, result_ids)
        return StepResult(self.observe(), cost, self.remaining_budget, self.done)

    def _prospective_cost(self, action: Action) -> int:
        if action.kind == ActionKind.SEARCH:
            return max(1, math.ceil(min(action.k or 10, 10) / 10))
        if action.kind in {ActionKind.EXPAND_REFS, ActionKind.EXPAND_CITERS}:
            return max(1, math.ceil(min(action.k or 20, 20) / 10))
        if action.kind == ActionKind.READ:
            return 1
        if action.kind == ActionKind.NOOP:
            return 1
        return 0

