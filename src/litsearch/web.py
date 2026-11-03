"""Open-web condition: the same agent protocol over the whole of OpenAlex (~250M works).

``search`` hits OpenAlex's relevance search, ``expand_refs`` returns a work's full
reference list, ``expand_citers`` returns its (most-cited) citers, all restricted to the
review's search cutoff date. Leakage filter: the review itself, works citing the review,
and works whose title equals the review's title are hidden. Gold labels never leave the
evaluator; they are only used to score.

Canonicalization: OpenAlex often has several records for one paper (preprint, published
version, duplicates). A fetched work whose DOI or normalized title matches a paper in the
frozen store is shown under the store's id, so a found gold paper is credited even when the
live index returns another record. The mapping uses every paper in the store (not just
gold), so it reveals nothing about labels.

Every response is cached by the cache-first OpenAlexClient, so a run can be replayed
offline; the live index changes over time, so record the run date.
"""

from __future__ import annotations

import re
import threading
from typing import Any

from .environment import LiteratureSearchEnv
from .models import Action, ActionKind, EpisodeSpec, Paper, StepResult
from .openalex import OpenAlexClient, abstract_from_inverted, short_id
from .store import CorpusStore

SELECT = (
    "id,doi,display_name,publication_year,publication_date,abstract_inverted_index,"
    "primary_location,referenced_works"
)
PAGE = 25
MAX_PAGES = 6


def _norm_title(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _norm_doi(value: str | None) -> str:
    value = (value or "").lower().strip()
    return value.removeprefix("https://doi.org/").removeprefix("doi:")


class WebStore:
    """Duck-typed CorpusStore over live OpenAlex, seeded with a frozen store's papers."""

    def __init__(self, base: CorpusStore, client: OpenAlexClient) -> None:
        self.base = base
        self.client = client
        self.reviews = base.reviews
        self.papers: dict[str, Paper] = dict(base.papers)
        self.alias: dict[str, str] = {}
        self._lock = threading.Lock()
        self._by_doi = {_norm_doi(p.doi): p.id for p in base.papers.values() if p.doi}
        self._by_title = {}
        for paper in base.papers.values():
            key = _norm_title(paper.title)
            if len(key) >= 25:
                self._by_title.setdefault(key, paper.id)

    def review(self, review_id: str):
        return self.base.review(review_id)

    def canonical(self, work: dict[str, Any]) -> str:
        work_id = short_id(work.get("id")) or ""
        if work_id in self.base.papers:
            return work_id
        doi = _norm_doi(work.get("doi"))
        if doi and doi in self._by_doi:
            return self._by_doi[doi]
        title = _norm_title(work.get("display_name") or work.get("title"))
        if len(title) >= 25 and title in self._by_title:
            return self._by_title[title]
        return work_id

    def add_work(self, work: dict[str, Any]) -> str:
        work_id = short_id(work.get("id")) or ""
        canonical = self.canonical(work)
        with self._lock:
            if canonical != work_id:
                self.alias[work_id] = canonical
            if canonical not in self.papers:
                doi = work.get("doi")
                source = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
                self.papers[canonical] = Paper(
                    id=canonical,
                    title=work.get("display_name") or work.get("title") or "Untitled",
                    abstract=abstract_from_inverted(work.get("abstract_inverted_index")),
                    publication_year=work.get("publication_year"),
                    doi=_norm_doi(doi) or None,
                    venue=source,
                    references=tuple(v for item in work.get("referenced_works") or [] if (v := short_id(item))),
                    publication_date=work.get("publication_date"),
                )
        return canonical

    def paper(self, paper_id: str) -> Paper | None:
        paper_id = self.alias.get(paper_id, paper_id)
        if paper_id in self.papers:
            return self.papers[paper_id]
        if not re.fullmatch(r"W\d+", paper_id or ""):
            return None
        works = self.client.works_batch([paper_id], select=SELECT)
        return self.papers.get(self.add_work(works[0])) if works else None

    def fetch_many(self, ids: list[str]) -> list[str]:
        """Ensure metadata for ``ids`` (batched); return canonical ids in input order."""
        missing = [i for i in ids if self.alias.get(i, i) not in self.papers]
        for start in range(0, len(missing), 50):
            for work in self.client.works_batch(missing[start : start + 50], select=SELECT):
                self.add_work(work)
        out: list[str] = []
        for paper_id in ids:
            canonical = self.alias.get(paper_id, paper_id)
            if canonical in self.papers:
                out.append(canonical)
        return out


class _WebAllowed:
    """Set-like visibility predicate over the unbounded universe."""

    def __init__(self, store: WebStore, review) -> None:
        self.store = store
        self.review = review
        self.review_title = _norm_title(review.question)

    def __contains__(self, paper_id: object) -> bool:
        paper = self.store.papers.get(self.store.alias.get(paper_id, paper_id))  # type: ignore[arg-type]
        if paper is None or paper.id == self.review.review_paper_id:
            return False
        if paper.publication_year and paper.publication_year > self.review.search_cutoff_year:
            return False
        cutoff = self.review.search_cutoff_date
        if cutoff and paper.publication_date and paper.publication_date > cutoff:
            return False
        if self.review.review_paper_id and self.review.review_paper_id in paper.references:
            return False
        if self.review_title and _norm_title(paper.title) == self.review_title:
            return False
        return True

    def __len__(self) -> int:
        return 10**9

    def __iter__(self):  # pragma: no cover - never enumerated
        raise TypeError("the open-web universe cannot be enumerated")


class WebLiteratureSearchEnv(LiteratureSearchEnv):
    def __init__(self, store: WebStore, spec: EpisodeSpec) -> None:
        self.web = store
        self._review = store.review(spec.review_id)
        cutoff = self._review.search_cutoff_date or f"{self._review.search_cutoff_year}-12-31"
        self._filter = f"to_publication_date:{cutoff}"
        if self._review.review_paper_id:
            self._filter += f",cites:!{self._review.review_paper_id}"
        self._query_pages: dict[str, int] = {}
        super().__init__(store, _SeedRetriever(self), spec)  # type: ignore[arg-type]

    def _allowed_ids(self):
        return _WebAllowed(self.web, self._review)

    def _search_ids(self, query: str, k: int) -> list[str]:
        """Next ``k`` unexposed, visible results for ``query`` (re-issuing pages deeper)."""
        query = (query or "").strip()[:300] or self.spec.question[:300]
        found: list[str] = []
        page = self._query_pages.get(query, 1)
        while len(found) < k and page <= MAX_PAGES:
            result = self.web.client._get(
                "works",
                {"search": query, "filter": self._filter, "per-page": str(PAGE), "page": str(page), "select": SELECT},
            )
            works = result.get("results", [])
            for work in works:
                paper_id = self.web.add_work(work)
                if paper_id in self.allowed_ids and paper_id not in self.exposed and paper_id not in found:
                    found.append(paper_id)
                    if len(found) == k:
                        break
            if len(found) < k:
                page += 1
            if len(works) < PAGE:
                break
        self._query_pages[query] = page
        return found

    def _neighbors(self, paper: Paper, kind: ActionKind, k: int) -> list[str]:
        if kind == ActionKind.EXPAND_REFS:
            ids = self.web.fetch_many(list(paper.references))
        else:
            source = paper.id
            result = self.web.client._get(
                "works",
                {
                    "filter": f"cites:{source},{self._filter}",
                    "sort": "cited_by_count:desc",
                    "per-page": "50",
                    "select": SELECT,
                },
            )
            ids = [self.web.add_work(work) for work in result.get("results", [])]
        return [i for i in ids if i in self.allowed_ids][:k]

    def step(self, action: Action) -> StepResult:
        tool = action.kind in {ActionKind.SEARCH, ActionKind.EXPAND_REFS, ActionKind.EXPAND_CITERS}
        if not tool or self.done:
            return super().step(action)
        action.validate()
        if action.kind == ActionKind.SEARCH and self.spec.mode not in {"keyword", "hybrid"}:
            return self._error("tool_unavailable", "search is unavailable in citation mode", action)
        if action.kind != ActionKind.SEARCH and self.spec.mode not in {"citation", "hybrid"}:
            return self._error("tool_unavailable", "citation expansion is unavailable", action)
        cost = self._prospective_cost(action)
        if self.exhausted and cost > 0:
            self.done = True
            return self._error("budget_exhausted", "budget is spent; only screening remains", action)
        if cost > self.remaining_budget:
            return self._error("insufficient_budget", f"action costs {cost}, only {self.remaining_budget} remains", action)
        if action.kind == ActionKind.SEARCH:
            result_ids = self._search_ids(action.query or "", min(action.k or 10, 10))
        else:
            paper = self.web.paper(action.paper_id or "")
            if not paper or paper.id not in self.allowed_ids:
                return self._error("unknown_paper", "paper is absent or filtered", action)
            result_ids = self._neighbors(paper, action.kind, min(action.k or 20, 20))
            cost = min(cost, max(1, -(-len(result_ids) // 10)))
        self._add_frontier(result_ids)
        self.exposed.update(result_ids)
        self.last_result = {"type": action.kind.value, "papers": self._metadata(result_ids)}
        self.remaining_budget -= cost
        self.action_count += 1
        if self.remaining_budget <= 0:
            self.exhausted = True
        self._record(action, cost, result_ids)
        return StepResult(self.observe(), cost, self.remaining_budget, self.done)


class _SeedRetriever:
    """Seeds for the open-web condition: OpenAlex search on the review question."""

    def __init__(self, env: WebLiteratureSearchEnv) -> None:
        self.env = env

    def search(self, query: str, k: int, allowed: Any) -> list[str]:
        env = self.env
        # The base __init__ has not yet set ``exposed``-dependent state beyond empty sets.
        env.exposed = getattr(env, "exposed", set())
        return env._search_ids(env.spec.question, k)
