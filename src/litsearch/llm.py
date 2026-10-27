"""Prompted LLM search agent shared by API/vLLM evaluation, SFT data, and RL rollouts.

One model call is one *turn*: the model screens any exposed-but-undecided papers
(free include/exclude) and chooses exactly one charged action or ``stop``. The
agent expands a turn into the environment's single-step actions, so the model is
called roughly once per unit of budget instead of once per screening decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .models import Action, ActionKind, Observation

SYSTEM_PROMPT = """You are an expert systematic reviewer searching a citation database for \
every study that meets a review's eligibility criteria. You work under a strict budget.

Each turn you see the papers you have been shown but not yet screened. Screen them by \
title (include the ones likely eligible, exclude the rest), then choose ONE next action.

Actions and their budget cost:
- search: keyword search, returns up to 10 unseen papers. Cost 1. Needs "query".
- expand_refs: papers cited by a paper you have seen. Cost 1 per 10 returned (max 20). Needs "paper_id".
- expand_citers: papers that cite a paper you have seen. Cost 1 per 10 returned (max 20). Needs "paper_id".
- read: show a paper's abstract to screen it more carefully. Cost 1. Needs "paper_id".
- stop: end the search. Free.
Including and excluding are free. Only use actions allowed in this episode.

Goal: include as many eligible studies as possible; avoid including ineligible ones.
Reply with ONE JSON object and nothing else, for example:
{"include": ["W123"], "exclude": ["W456", "W789"], "action": {"kind": "search", "query": "..."}}"""

MODE_TOOLS = {
    "keyword": ("search", "read", "stop"),
    "citation": ("expand_refs", "expand_citers", "read", "stop"),
    "hybrid": ("search", "expand_refs", "expand_citers", "read", "stop"),
}
CHARGED_KINDS = {
    "search": ActionKind.SEARCH,
    "expand_refs": ActionKind.EXPAND_REFS,
    "expand_citers": ActionKind.EXPAND_CITERS,
    "read": ActionKind.READ,
    "stop": ActionKind.STOP,
}


class ChatBackend(Protocol):
    model: str

    def complete(
        self, messages: list[dict[str, str]], *, temperature: float, seed: int | None
    ) -> str: ...


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class TurnRecord:
    messages: list[dict[str, str]]
    response: str
    parsed: dict[str, Any] | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "response": self.response,
            "parsed": self.parsed,
            "error": self.error,
        }


def parse_turn(text: str) -> dict[str, Any]:
    """Decode the first complete JSON object in the reply; ignores fences and trailing text."""
    decoder = json.JSONDecoder()
    value = None
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "action" in value:
            break
    if value is None:
        raise ValueError("response has no JSON object")
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    action = value.get("action")
    if isinstance(action, str):
        action = {"kind": action}
    if not isinstance(action, dict) or str(action.get("kind")) not in CHARGED_KINDS:
        raise ValueError("action.kind is missing or invalid")
    return {
        "include": [str(i) for i in value.get("include") or [] if isinstance(i, (str, int))],
        "exclude": [str(i) for i in value.get("exclude") or [] if isinstance(i, (str, int))],
        "action": action,
    }


@dataclass
class LLMSearchAgent:
    backend: ChatBackend
    name: str = "llm"
    temperature: float = 0.0
    seed: int | None = None
    max_undecided_shown: int = 40
    history_shown: int = 12
    record_turns: bool = True
    turns: list[TurnRecord] = field(default_factory=list, init=False)
    stats: dict[str, int] = field(
        default_factory=lambda: {"turns": 0, "parse_failures": 0, "invalid_ids": 0, "disallowed": 0},
        init=False,
    )
    _titles: dict[str, str] = field(default_factory=dict, init=False)
    _years: dict[str, Any] = field(default_factory=dict, init=False)
    _undecided: list[str] = field(default_factory=list, init=False)
    _decided: set[str] = field(default_factory=set, init=False)
    _history: list[str] = field(default_factory=list, init=False)
    _queue: list[Action] = field(default_factory=list, init=False)
    _last_type: str | None = field(default=None, init=False)
    _budget: int | None = field(default=None, init=False)

    # -- state tracking -------------------------------------------------------
    def _absorb(self, observation: Observation) -> None:
        result = observation.last_result
        kind = result.get("type")
        papers = list(result.get("papers") or [])
        if kind == "read" and result.get("paper"):
            papers = [result["paper"]]
        new = 0
        for paper in papers:
            paper_id = paper["id"]
            self._titles.setdefault(paper_id, paper.get("title") or "")
            self._years.setdefault(paper_id, paper.get("publication_year"))
            if paper_id not in self._decided and paper_id not in self._undecided:
                self._undecided.append(paper_id)
                new += 1
        if kind in {"search", "expand_refs", "expand_citers"} and self._history:
            self._history[-1] += f" -> {len(papers)} results, {new} new"
        elif kind == "error" and self._history:
            self._history[-1] += f" -> error: {result.get('message')}"

    def _decide(self, paper_id: str) -> None:
        self._decided.add(paper_id)
        if paper_id in self._undecided:
            self._undecided.remove(paper_id)

    # -- prompt ---------------------------------------------------------------
    def render(self, observation: Observation, read_paper: dict[str, Any] | None) -> list[dict[str, str]]:
        tools = MODE_TOOLS[observation.mode]
        lines = [
            f"REVIEW QUESTION: {observation.question}",
            f"ELIGIBILITY CRITERIA: {_clip(observation.eligibility_criteria, 1500)}",
            "",
            f"Episode mode: {observation.mode}. Allowed actions: {', '.join(tools)}.",
            f"Budget remaining: {observation.remaining_budget}"
            + (" (spent: screen what remains, then stop)" if observation.remaining_budget <= 0 else ""),
            "",
            f"INCLUDED SO FAR ({len(observation.included_ids)}):",
        ]
        for paper_id in list(observation.included_ids)[-15:]:
            lines.append(f"- {paper_id} | {_clip(self._titles.get(paper_id, ''), 120)}")
        lines += ["", "RECENT ACTIONS:"]
        lines += [f"- {entry}" for entry in self._history[-self.history_shown :]] or ["- (none yet)"]
        if read_paper:
            lines += [
                "",
                f"ABSTRACT OF {read_paper['id']} ({read_paper.get('publication_year')}): "
                f"{_clip(read_paper.get('title', ''), 200)}",
                _clip(read_paper.get("abstract") or "(no abstract available)", 1800),
            ]
        shown = self._undecided[: self.max_undecided_shown]
        hidden = len(self._undecided) - len(shown)
        lines += ["", f"UNSCREENED PAPERS ({len(self._undecided)}):"]
        for paper_id in shown:
            lines.append(f"- {paper_id} ({self._years.get(paper_id)}) {_clip(self._titles[paper_id], 160)}")
        if hidden > 0:
            lines.append(f"- ... {hidden} more, shown after you screen these")
        if not self._undecided:
            lines.append("- (none)")
        lines += ["", "Reply with one JSON object: include, exclude, action."]
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]

    # -- acting ---------------------------------------------------------------
    def _plan(self, observation: Observation, parsed: dict[str, Any]) -> list[Action]:
        actions: list[Action] = []
        known = set(self._undecided) | set(self._titles)
        for kind, ids in ((ActionKind.INCLUDE, parsed["include"]), (ActionKind.EXCLUDE, parsed["exclude"])):
            for paper_id in dict.fromkeys(ids):
                if paper_id not in known:
                    self.stats["invalid_ids"] += 1
                    continue
                if kind == ActionKind.INCLUDE and paper_id in observation.included_ids:
                    continue
                self._decide(paper_id)
                actions.append(Action(kind, paper_id=paper_id))
        action = parsed["action"]
        name = str(action["kind"])
        if observation.remaining_budget <= 0 and name != "stop":
            name = "stop"
        if name not in MODE_TOOLS[observation.mode]:
            # Same treatment as an unknown paper: keep the screening, charge the slip.
            self.stats["disallowed"] += 1
            self._history.append(f"{name} (not available in {observation.mode} mode; charged 1)")
            actions.append(Action(ActionKind.NOOP))
            return actions
        if name == "search":
            query = str(action.get("query") or "").strip() or observation.question
            actions.append(Action(ActionKind.SEARCH, query=query[:300], k=10))
            self._history.append(f'search "{_clip(query, 120)}"')
        elif name in {"expand_refs", "expand_citers", "read"}:
            paper_id = str(action.get("paper_id") or "")
            if paper_id not in known:
                # Keep the turn's screening decisions; the bad action is charged as a NOOP.
                self.stats["invalid_ids"] += 1
                self._history.append(f"{name} {paper_id!r} (invalid: paper not seen; charged 1)")
                actions.append(Action(ActionKind.NOOP))
                return actions
            k = None if name == "read" else 20
            actions.append(Action(CHARGED_KINDS[name], paper_id=paper_id, k=k))
            self._history.append(f"{name} {paper_id} ({_clip(self._titles.get(paper_id, ''), 60)})")
        else:
            actions.append(Action(ActionKind.STOP))
        return actions

    def act(self, observation: Observation) -> Action:
        if self._queue:
            return self._queue.pop(0)
        self._absorb(observation)
        read_paper = observation.last_result.get("paper") if observation.last_result.get("type") == "read" else None
        messages = self.render(observation, read_paper)
        self.stats["turns"] += 1
        response = ""
        try:
            response = self.backend.complete(messages, temperature=self.temperature, seed=self.seed)
            parsed = parse_turn(response)
            planned = self._plan(observation, parsed)
        except Exception as exc:  # parse/plan failures are charged as NOOP by the runner
            self.stats["parse_failures"] += 1
            if self.record_turns:
                self.turns.append(TurnRecord(messages, response, None, f"{type(exc).__name__}: {exc}"))
            self._history.append(f"(invalid reply: {_clip(str(exc), 80)})")
            raise
        if self.record_turns:
            self.turns.append(TurnRecord(messages, response, parsed))
        self._queue = planned[1:]
        return planned[0]

    def summary(self) -> dict[str, Any]:
        return {
            "backend_model": self.backend.model,
            "temperature": self.temperature,
            "seed": self.seed,
            **self.stats,
        }


class SpendCapReached(BaseException):
    """Raised (as BaseException, so episode loops cannot swallow it) once the ledger hits the cap."""


class SpendLedger:
    """Append-only USD ledger shared by every thread, process, and job.

    On NFS, ``flock`` degrades to per-process POSIX locks, so threads in one job are
    additionally serialized with an in-process lock; each record is one O_APPEND write.
    """

    _thread_locks: dict[str, threading.Lock] = {}
    _registry_lock = threading.Lock()

    def __init__(self, path: str | Path, cap_usd: float) -> None:
        self.path = Path(path)
        self.cap_usd = cap_usd
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        with SpendLedger._registry_lock:
            self._thread_lock = SpendLedger._thread_locks.setdefault(str(self.path.resolve()), threading.Lock())

    def _locked(self, fn):
        import fcntl

        with self._thread_lock:
            fd = os.open(self.path, os.O_RDWR | os.O_APPEND)
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX)
                try:
                    return fn(fd)
                finally:
                    fcntl.lockf(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _total(self, fd) -> float:
        with open(self.path, encoding="utf-8") as stream:
            total = 0.0
            for line in stream:
                try:
                    total += float(json.loads(line)["usd"]) if line.strip() else 0.0
                except (ValueError, KeyError):
                    raise RuntimeError(f"corrupt spend ledger line in {self.path}: {line[:80]!r}")
            return total

    def total(self) -> float:
        return self._locked(self._total)

    def check(self) -> None:
        spent = self.total()
        if spent >= self.cap_usd:
            raise SpendCapReached(f"spent ${spent:.2f} of ${self.cap_usd:.2f} cap")

    def add(self, usd: float, model: str, tokens: int) -> None:
        record = json.dumps({"t": time.time(), "usd": usd, "model": model, "tokens": tokens}) + "\n"
        self._locked(lambda fd: os.write(fd, record.encode()))


class OpenAIChatBackend:
    """OpenAI-compatible chat endpoint (OpenAI, vLLM, ...) with a content-addressed disk cache."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key_env: str = "MODEL_API_KEY",
        cache_dir: str | Path | None = None,
        max_tokens: int = 600,
        timeout: int = 180,
        retries: int = 4,
        extra_body: dict[str, Any] | None = None,
        ledger: SpendLedger | None = None,
        price_per_mtok: tuple[float, float] | None = None,
        system_suffix: str = "",
    ) -> None:
        self.model = model
        self.ledger = ledger
        # Fallback (input, output) USD per million tokens when the API omits usage.cost.
        self.price_per_mtok = price_per_mtok
        # Model-specific control tag appended to the system message (e.g. Qwen3's "/no_think").
        self.system_suffix = system_suffix
        self.base_url = (base_url or os.environ.get("MODEL_API_BASE") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.extra_body = extra_body or {}
        self.usage = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
        self._lock = threading.Lock()

    def complete(self, messages: list[dict[str, str]], *, temperature: float, seed: int | None) -> str:
        if self.system_suffix and messages and messages[0]["role"] == "system":
            messages = [{**messages[0], "content": messages[0]["content"] + self.system_suffix}, *messages[1:]]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        if seed is not None:
            payload["seed"] = seed
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache_path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cache_path and cache_path.exists():
            with self._lock:
                self.usage["cache_hits"] += 1
            return json.loads(cache_path.read_text(encoding="utf-8"))["content"]
        if self.ledger is not None:
            self.ledger.check()
        wire = dict(payload)
        if "openrouter.ai" in self.base_url:
            wire["usage"] = {"include": True}  # OpenRouter reports the request's USD cost
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(wire).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                last_error = RuntimeError(f"model API {exc.code}: {detail}")
                if exc.code < 500 and exc.code != 429:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
            time.sleep(2**attempt)
        else:
            raise RuntimeError(f"model API failed after {self.retries} attempts: {last_error}")
        if "choices" not in result:
            raise RuntimeError(f"model API returned no choices: {str(result)[:300]}")
        content = result["choices"][0]["message"].get("content") or ""
        usage = result.get("usage") or {}
        usd = float(usage.get("cost") or 0.0)
        if not usd and self.price_per_mtok:
            usd = (
                int(usage.get("prompt_tokens") or 0) * self.price_per_mtok[0]
                + int(usage.get("completion_tokens") or 0) * self.price_per_mtok[1]
            ) / 1e6
        with self._lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            self.usage["usd"] += usd
        if self.ledger is not None:
            self.ledger.add(usd, self.model, int(usage.get("total_tokens") or 0))
        if cache_path:
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"content": content, "usage": usage}), encoding="utf-8")
            tmp.replace(cache_path)
        return content
