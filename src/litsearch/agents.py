from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from .models import Action, ActionKind, Observation
from .retrieval import tokenize


class Agent(Protocol):
    name: str

    def act(self, observation: Observation) -> Action: ...


@dataclass
class RankedRetrievalAgent:
    """Include a ranked list in pages, representing a conventional IR baseline."""

    name: str = "retrieval"
    _queue: list[str] = field(default_factory=list, init=False)
    _seen: set[str] = field(default_factory=set, init=False)

    def act(self, observation: Observation) -> Action:
        result_type = observation.last_result.get("type")
        if result_type in {"seeds", "search"}:
            for paper in observation.last_result.get("papers", []):
                paper_id = paper["id"]
                if paper_id not in self._seen:
                    self._queue.append(paper_id)
                    self._seen.add(paper_id)
        if self._queue:
            return Action(ActionKind.INCLUDE, paper_id=self._queue.pop(0))
        if observation.remaining_budget >= 1 and observation.mode in {"keyword", "hybrid"}:
            return Action(ActionKind.SEARCH, query=observation.question, k=10)
        return Action(ActionKind.STOP)


@dataclass
class RandomAgent:
    """Unscreened random traversal: uniformly picks an available tool, includes every read."""

    seed: int = 0
    name: str = "random"
    _rng: random.Random = field(init=False)
    _read: set[str] = field(default_factory=set, init=False)
    _expanded: set[tuple[str, str]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def act(self, observation: Observation) -> Action:
        if observation.last_result.get("type") == "read":
            paper_id = observation.last_result["paper"]["id"]
            if paper_id not in observation.included_ids:
                return Action(ActionKind.INCLUDE, paper_id=paper_id)
        if observation.remaining_budget <= 0:
            return Action(ActionKind.STOP)
        frontier = [item["id"] for item in observation.frontier]
        options: list[Action] = [
            Action(ActionKind.READ, paper_id=paper_id)
            for paper_id in frontier
            if paper_id not in self._read
        ]
        if observation.mode in {"citation", "hybrid"}:
            for paper_id in dict.fromkeys(list(observation.included_ids) + frontier):
                for kind in (ActionKind.EXPAND_REFS, ActionKind.EXPAND_CITERS):
                    if (kind.value, paper_id) not in self._expanded:
                        options.append(Action(kind, paper_id=paper_id, k=10))
        if observation.mode in {"keyword", "hybrid"}:
            options.append(Action(ActionKind.SEARCH, query=observation.question, k=10))
        if not options:
            return Action(ActionKind.STOP)
        action = self._rng.choice(options)
        if action.kind == ActionKind.READ:
            self._read.add(action.paper_id or "")
        elif action.kind in {ActionKind.EXPAND_REFS, ActionKind.EXPAND_CITERS}:
            self._expanded.add((action.kind.value, action.paper_id or ""))
        return action


@dataclass
class SnowballAgent:
    """Gold-blind lexical screening plus snowballing from included papers.

    Hybrid mode searches first and then chases citations from the screened-in hits;
    citation mode chases from the environment seeds; keyword mode pages through search.
    """

    name: str = "snowball"
    include_threshold: float = 0.16
    _read: set[str] = field(default_factory=set, init=False)
    _expanded_refs: set[str] = field(default_factory=set, init=False)
    _expanded_citers: set[str] = field(default_factory=set, init=False)
    _pending_decision: str | None = field(default=None, init=False)
    _pending_relevant: bool = field(default=False, init=False)
    _searched: bool = field(default=False, init=False)

    def _relevance(self, observation: Observation) -> float:
        paper = observation.last_result.get("paper", {})
        document = set(tokenize(f"{paper.get('title', '')} {paper.get('abstract', '')}"))
        query = set(tokenize(f"{observation.question} {observation.eligibility_criteria}"))
        if not query:
            return 0.0
        return len(document & query) / len(query)

    def _search(self, observation: Observation) -> Action:
        self._searched = True
        return Action(ActionKind.SEARCH, query=observation.question, k=10)

    def act(self, observation: Observation) -> Action:
        if observation.last_result.get("type") == "read":
            self._pending_decision = observation.last_result["paper"]["id"]
            self._pending_relevant = self._relevance(observation) >= self.include_threshold
        if self._pending_decision:
            paper_id = self._pending_decision
            self._pending_decision = None
            return Action(
                ActionKind.INCLUDE if self._pending_relevant else ActionKind.EXCLUDE,
                paper_id=paper_id,
            )
        if observation.remaining_budget < 1:
            return Action(ActionKind.STOP)
        if observation.mode == "hybrid" and not self._searched:
            return self._search(observation)

        unread = [item["id"] for item in observation.frontier if item["id"] not in self._read]
        if unread:
            paper_id = unread[0]
            self._read.add(paper_id)
            return Action(ActionKind.READ, paper_id=paper_id)

        if observation.mode in {"citation", "hybrid"}:
            # Chase from screened-in papers; fall back to the frontier only before any inclusion.
            sources = list(observation.included_ids) or [item["id"] for item in observation.frontier]
            for paper_id in sources:
                if paper_id not in self._expanded_refs:
                    self._expanded_refs.add(paper_id)
                    return Action(ActionKind.EXPAND_REFS, paper_id=paper_id, k=10)
                if paper_id not in self._expanded_citers:
                    self._expanded_citers.add(paper_id)
                    return Action(ActionKind.EXPAND_CITERS, paper_id=paper_id, k=10)
        if observation.mode in {"keyword", "hybrid"}:
            return self._search(observation)
        return Action(ActionKind.STOP)
