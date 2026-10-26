from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal


class ActionKind(StrEnum):
    SEARCH = "search"
    EXPAND_REFS = "expand_refs"
    EXPAND_CITERS = "expand_citers"
    READ = "read"
    INCLUDE = "include"
    EXCLUDE = "exclude"
    STOP = "stop"
    NOOP = "noop"  # internal: charged agent/model failure, never advertised as a tool


SearchMode = Literal["citation", "keyword", "hybrid"]


@dataclass(frozen=True, slots=True)
class Paper:
    id: str
    title: str
    abstract: str = ""
    publication_year: int | None = None
    doi: str | None = None
    venue: str | None = None
    references: tuple[str, ...] = ()
    citers: tuple[str, ...] = ()
    publication_date: str | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "publication_year": self.publication_year,
            "doi": self.doi,
            "venue": self.venue,
        }


@dataclass(frozen=True, slots=True)
class Review:
    id: str
    question: str
    eligibility_criteria: str
    gold_ids: frozenset[str]
    search_cutoff_year: int
    review_paper_id: str | None = None
    domain: str = "unknown"
    # ISO date; when set, papers dated after it are hidden (finer than the year cutoff).
    search_cutoff_date: str | None = None
    # The review's screening universe; empty means the whole corpus is searchable.
    candidate_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    review_id: str
    question: str
    eligibility_criteria: str
    budget: int
    mode: SearchMode
    seed_count: int = 5

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.mode not in {"citation", "keyword", "hybrid"}:
            raise ValueError(f"unsupported mode: {self.mode}")
        if self.seed_count < 0:
            raise ValueError("seed_count cannot be negative")


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    query: str | None = None
    paper_id: str | None = None
    k: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Action":
        if not isinstance(value, dict):
            raise ValueError("action must be a JSON object")
        try:
            kind = ActionKind(str(value["kind"]))
        except (KeyError, ValueError) as exc:
            raise ValueError("action.kind is missing or invalid") from exc
        if kind == ActionKind.NOOP:
            raise ValueError("noop is reserved for internal error accounting")
        action = cls(
            kind=kind,
            query=value.get("query"),
            paper_id=value.get("paper_id"),
            k=value.get("k"),
        )
        action.validate()
        return action

    def validate(self) -> None:
        if self.kind == ActionKind.SEARCH and not (self.query and self.query.strip()):
            raise ValueError("search requires a non-empty query")
        if self.kind in {
            ActionKind.EXPAND_REFS,
            ActionKind.EXPAND_CITERS,
            ActionKind.READ,
            ActionKind.INCLUDE,
            ActionKind.EXCLUDE,
        } and not self.paper_id:
            raise ValueError(f"{self.kind} requires paper_id")
        if self.k is not None and self.k <= 0:
            raise ValueError("k must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(slots=True)
class Observation:
    question: str
    eligibility_criteria: str
    remaining_budget: int
    mode: SearchMode
    included_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    frontier: tuple[dict[str, Any], ...]
    last_result: dict[str, Any]
    action_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActionError:
    code: str
    message: str


@dataclass(slots=True)
class StepResult:
    observation: Observation
    charged_cost: int
    remaining_budget: int
    done: bool
    error: ActionError | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "observation": self.observation.to_dict(),
            "charged_cost": self.charged_cost,
            "remaining_budget": self.remaining_budget,
            "done": self.done,
            "error": asdict(self.error) if self.error else None,
        }
        return value


@dataclass(slots=True)
class TraceEvent:
    step: int
    action: dict[str, Any]
    charged_cost: int
    budget_after: int
    result_ids: list[str] = field(default_factory=list)
    included_ids: list[str] = field(default_factory=list)
    excluded_ids: list[str] = field(default_factory=list)
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

