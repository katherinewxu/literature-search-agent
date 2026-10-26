from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request
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


SYSTEM_PROMPT = """You are searching for every study eligible for a systematic review.
Choose exactly one tool action. Optimize recall while respecting the charged budget.
Do not use knowledge of gold labels. Reply with one JSON object and no prose.
Schema: {"kind":"search|expand_refs|expand_citers|read|include|exclude|stop",
"query":"...", "paper_id":"W...", "k":10}.
Only use tools allowed by the episode mode. Search returns at most 10 results;
citation expansion returns at most 20. Include/exclude/stop are free.
"""


@dataclass
class OpenAICompatibleAgent:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    timeout: int = 60
    name: str = "prompted"

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("MODEL_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        self.base_url = os.environ.get("MODEL_API_BASE", self.base_url).rstrip("/")
        self.model = os.environ.get("MODEL_NAME", self.model)

    def _request(self, observation: Observation) -> str:
        if not self.api_key:
            raise RuntimeError("set MODEL_API_KEY or OPENAI_API_KEY")
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(observation.to_dict(), sort_keys=True),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"model API returned {exc.code}: {detail[:500]}") from exc
        return result["choices"][0]["message"]["content"]

    @staticmethod
    def _parse(content: str) -> Action:
        content = content.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if fenced:
            content = fenced.group(1)
        elif not content.startswith("{"):
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError("model response does not contain a JSON object")
            content = match.group(0)
        return Action.from_dict(json.loads(content))

    def act(self, observation: Observation) -> Action:
        first_error: Exception | None = None
        for _ in range(2):
            try:
                return self._parse(self._request(observation))
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                first_error = first_error or exc
        raise ValueError(f"model emitted two malformed actions: {first_error}")

