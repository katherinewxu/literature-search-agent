"""Ablation wrapper: reject an agent's early STOP while budget remains.

Tests whether early stopping causes the agent's recall gap to the no-loop baseline. When the
wrapped LLM agent chooses STOP with budget left, the stop is rejected: a line is added to the
agent's action history ("stop rejected: N budget left ...") and the agent is asked again on the
same state. After ``max_refusals`` consecutive refused stops in one decision the stop is honoured,
so a model that cannot find anything to do is not trapped. Screening decisions are unchanged.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from .models import Action, ActionKind, Observation

NUDGE = ("stop rejected by the environment: {left} budget left. Keep searching until the budget "
         "is spent (try new queries: synonyms, populations, outcomes, study designs; or expand citations).")


class NoStopAgent:
    def __init__(self, inner, max_refusals: int = 2, name: str | None = None) -> None:
        self.inner = inner
        self.max_refusals = max_refusals
        self.name = name or f"{inner.name}-nostop"
        self.rejected = 0
        self.honoured_early = 0

    def __getattr__(self, item: str) -> Any:  # turns, record_turns, ... come from the inner agent
        return getattr(self.inner, item)

    def act(self, observation: Observation) -> Action:
        action = self.inner.act(observation)
        refusals = 0
        while action.kind == ActionKind.STOP and observation.remaining_budget >= 1:
            if refusals >= self.max_refusals:
                self.honoured_early += 1
                break
            refusals += 1
            self.rejected += 1
            self.inner._history.append(NUDGE.format(left=observation.remaining_budget))
            # Same state, but no new result to absorb (the last one was already absorbed).
            action = self.inner.act(dataclasses.replace(observation, last_result={"type": "nudge"}))
        return action

    def summary(self) -> dict[str, Any]:
        base = self.inner.summary() if hasattr(self.inner, "summary") else {}
        return {**base, "stops_rejected": self.rejected, "early_stops_honoured": self.honoured_early,
                "max_refusals": self.max_refusals}
