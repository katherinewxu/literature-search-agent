"""Ablation wrapper: force citation chasing into an agent's own trajectory.

Every ``every``-th charged action that the wrapped agent chooses is replaced by citation
expansion of the most recently included paper not yet expanded (alternating citers and refs).
The agent keeps its own screening decisions and sees the expansion results on its next turn,
so this measures whether citation chasing *from the agent's own inclusions* adds recall at a
fixed budget. If nothing is included yet, the agent's action is kept.
"""

from __future__ import annotations

from typing import Any

from .models import Action, ActionKind, Observation

_CHARGED = {ActionKind.SEARCH, ActionKind.EXPAND_REFS, ActionKind.EXPAND_CITERS, ActionKind.READ}


class ForcedExpansionAgent:
    def __init__(self, inner, every: int = 3, name: str | None = None) -> None:
        self.inner = inner
        self.every = every
        self.name = name or f"{inner.name}-forced{every}"
        self._charged = 0
        self._expanded: set[str] = set()
        self._order: list[str] = []
        self.forced = 0

    def __getattr__(self, item: str) -> Any:  # turns, record_turns, ... come from the inner agent
        return getattr(self.inner, item)

    def act(self, observation: Observation) -> Action:
        for paper_id in observation.included_ids:
            if paper_id not in self._order:
                self._order.append(paper_id)
        action = self.inner.act(observation)
        if action.kind not in _CHARGED or observation.mode != "hybrid":
            return action
        self._charged += 1
        if self._charged % self.every:
            return action
        target = next((p for p in reversed(self._order) if p not in self._expanded), None)
        if target is None:
            return action
        self._expanded.add(target)
        kind = ActionKind.EXPAND_CITERS if self.forced % 2 == 0 else ActionKind.EXPAND_REFS
        self.forced += 1
        history = getattr(self.inner, "_history", None)
        if history is not None:
            history.append(f"{kind.value} {target} (forced by the environment)")
        return Action(kind, paper_id=target, k=20)

    def summary(self) -> dict[str, Any]:
        base = self.inner.summary() if hasattr(self.inner, "summary") else {}
        return {**base, "forced_expansions": self.forced, "forced_every": self.every}
