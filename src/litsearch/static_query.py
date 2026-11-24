"""Non-adaptive LLM baseline: write queries once, run them, screen by title. No agent loop.

Answers "is the agent loop needed, or would an LLM-written query list do as well?". The same
LLM writes ``n_queries`` keyword queries from the review question and criteria in a single call,
before seeing any result. The agent then spends the budget issuing those queries round-robin
(repeated queries page deeper, cost 1 per 10 results). With ``screen=True`` every batch of
results (and the seeds) is screened by title in one LLM call, exactly the information the loop
agent screens on; with ``screen=False`` everything retrieved is included.
Search only: no citation expansion, no reading, no feedback from results into the queries.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .llm import ChatBackend, TurnRecord, _clip
from .models import Action, ActionKind, Observation

QUERY_SYSTEM = """You are an expert systematic reviewer planning a literature search in OpenAlex \
(keyword search over titles and abstracts). You will not see any results before the queries run."""

QUERY_PROMPT = """REVIEW QUESTION: {question}
ELIGIBILITY CRITERIA: {criteria}

Write {n} diverse keyword search queries that together should find every eligible study.
Vary synonyms, populations, interventions/exposures, outcomes and study designs. Keep each query
short (3-10 words, no boolean operators). Reply with JSON only: {{"queries": ["...", ...]}}"""

SCREEN_SYSTEM = """You are an expert systematic reviewer screening search results by title for a \
review. Include the ones likely eligible and exclude the rest."""

SCREEN_PROMPT = """REVIEW QUESTION: {question}
ELIGIBILITY CRITERIA: {criteria}

PAPERS:
{papers}

Reply with JSON only: {{"include": ["W..."]}} listing the ids of likely eligible papers (may be empty)."""


def _json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("no JSON object in reply")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("reply JSON is not an object")
    return value


@dataclass
class StaticQueryAgent:
    backend: ChatBackend
    name: str = "static-query"
    n_queries: int = 8
    screen: bool = True
    temperature: float = 0.0
    seed: int | None = None  # set when temperature > 0 so repeats are distinct samples (and cache entries)
    turns: list[TurnRecord] = field(default_factory=list, init=False)
    stats: dict[str, int] = field(default_factory=lambda: {"turns": 0, "parse_failures": 0, "queries": 0}, init=False)
    _queries: list[str] | None = field(default=None, init=False)
    _next: int = field(default=0, init=False)
    _queue: list[Action] = field(default_factory=list, init=False)
    _handled: set[int] = field(default_factory=set, init=False)

    def _call(self, system: str, user: str) -> dict[str, Any] | None:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        self.stats["turns"] += 1
        # API/transport errors propagate: the runner charges them and stops after three, so a
        # misconfigured model fails loudly instead of producing a silent zero-recall run.
        response = self.backend.complete(messages, temperature=self.temperature, seed=self.seed)
        try:
            parsed = _json(response)
        except Exception as exc:  # an unparseable reply costs nothing in the environment; skip it
            self.stats["parse_failures"] += 1
            self.turns.append(TurnRecord(messages, response, None, f"{type(exc).__name__}: {exc}"))
            return None
        self.turns.append(TurnRecord(messages, response, parsed))
        return parsed

    def _plan_queries(self, obs: Observation) -> None:
        parsed = self._call(QUERY_SYSTEM, QUERY_PROMPT.format(
            question=obs.question, criteria=_clip(obs.eligibility_criteria, 1500), n=self.n_queries))
        queries = [str(q).strip()[:300] for q in (parsed or {}).get("queries") or [] if str(q).strip()]
        self._queries = list(dict.fromkeys(queries))[: self.n_queries] or [obs.question[:300]]
        self.stats["queries"] = len(self._queries)

    def _screen(self, obs: Observation, papers: list[dict[str, Any]]) -> None:
        papers = [p for p in papers if p["id"] not in obs.included_ids]
        if not papers:
            return
        if not self.screen:
            keep = [p["id"] for p in papers]
        else:
            listing = "\n".join(f"- {p['id']} ({p.get('publication_year')}) {_clip(p.get('title') or '', 160)}" for p in papers)
            parsed = self._call(SCREEN_SYSTEM, SCREEN_PROMPT.format(
                question=obs.question, criteria=_clip(obs.eligibility_criteria, 1500), papers=listing))
            ids = {p["id"] for p in papers}
            keep = [str(i) for i in (parsed or {}).get("include") or [] if str(i) in ids]
        self._queue += [Action(ActionKind.INCLUDE, paper_id=i) for i in dict.fromkeys(keep)]

    def act(self, observation: Observation) -> Action:
        if self._queue:
            return self._queue.pop(0)
        result = observation.last_result
        # Screen each new batch of results once (action_count identifies the batch).
        if result.get("type") in {"seeds", "search"} and observation.action_count not in self._handled:
            self._handled.add(observation.action_count)
            self._screen(observation, list(result.get("papers") or []))
            if self._queue:
                return self._queue.pop(0)
        if self._queries is None:
            self._plan_queries(observation)
        if observation.remaining_budget >= 1 and observation.mode in {"keyword", "hybrid"}:
            query = self._queries[self._next % len(self._queries)]
            self._next += 1
            return Action(ActionKind.SEARCH, query=query, k=10)
        return Action(ActionKind.STOP)

    def summary(self) -> dict[str, Any]:
        return {"backend_model": self.backend.model, "temperature": self.temperature, "seed": self.seed,
                "n_queries": self.n_queries, "screen": self.screen, "queries": self._queries, **self.stats}
