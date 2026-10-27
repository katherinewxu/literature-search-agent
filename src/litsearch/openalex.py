from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .models import Paper
from .store import CorpusStore


def abstract_from_inverted(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positioned = [(position, token) for token, positions in index.items() for position in positions]
    return " ".join(token for _, token in sorted(positioned))


def short_id(value: str | None) -> str | None:
    if not value:
        return None
    return value.rstrip("/").rsplit("/", 1)[-1]


class OpenAlexBudgetExhausted(RuntimeError):
    """The daily OpenAlex budget for this IP/key is spent; retrying cannot help."""


class OfflineCacheMiss(RuntimeError):
    """Raised by an offline client on a cache miss; the URL is recorded in ``missing``."""


class OpenAlexClient:
    """Cache-first OpenAlex API client used only during artifact construction.

    Cache keys hash the request URL *without* credentials (``api_key``/``mailto``), so
    entries fetched from any machine or with any key are shared. ``offline=True``
    never touches the network and records each missing URL instead.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        email: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.openalex.org",
        delay_seconds: float = 0.1,
        offline: bool = False,
        retries: int = 5,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.email = email
        self.api_key = api_key if api_key is not None else os.environ.get("OPENALEX_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.delay_seconds = delay_seconds
        self.offline = offline
        self.retries = retries
        self.missing: list[str] = []

    def url_for(self, path: str, params: dict[str, str] | None = None) -> str:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        return url

    def cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.json"

    def fetch_url(self, url: str) -> Any:
        """Fetch one credential-free URL (cache first), adding credentials only on the wire."""
        cache_path = self.cache_path(url)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        if self.offline:
            self.missing.append(url)
            raise OfflineCacheMiss(url)
        credentials = {k: v for k, v in (("api_key", self.api_key), ("mailto", self.email)) if v}
        wire = url + (("&" if "?" in url else "?") + urllib.parse.urlencode(credentials) if credentials else "")
        request = urllib.request.Request(wire, headers={"User-Agent": "budgeted-litsearch/0.1"})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                if exc.code == 429 and "budget" in body.lower():
                    raise OpenAlexBudgetExhausted(body[:200]) from exc
                if exc.code != 429 and exc.code < 500:
                    raise
                time.sleep(float(exc.headers.get("Retry-After") or 2**attempt))
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"OpenAlex request failed after {self.retries} attempts")
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        tmp.replace(cache_path)
        time.sleep(self.delay_seconds)
        return result

    def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        return self.fetch_url(self.url_for(path, params))

    def work(self, paper_id: str) -> dict[str, Any]:
        return self._get(f"works/{paper_id}")

    def citers(self, paper_id: str, limit: int = 20) -> list[dict[str, Any]]:
        result = self._get(
            "works",
            {"filter": f"cites:{paper_id}", "per-page": str(min(limit, 200))},
        )
        return list(result.get("results", []))[:limit]


    def works_batch(self, paper_ids: list[str], select: str = "id,abstract_inverted_index") -> list[dict[str, Any]]:
        """Fetch up to 50 works in one request via the ``openalex_id`` OR-filter."""
        if len(paper_ids) > 50:
            raise ValueError("OpenAlex OR-filters accept at most 50 values")
        result = self._get(
            "works",
            {
                "filter": "openalex_id:" + "|".join(paper_ids),
                "per-page": "50",
                "select": select,
            },
        )
        return list(result.get("results", []))


def backfill_abstracts(
    store: CorpusStore, client: OpenAlexClient, *, batch_size: int = 50
) -> tuple[CorpusStore, dict[str, int]]:
    """Fill empty abstracts from the live OpenAlex API, leaving every other field frozen."""
    missing = sorted(paper_id for paper_id, paper in store.papers.items() if not paper.abstract)
    papers = dict(store.papers)
    stats = {"missing_before": len(missing), "returned": 0, "filled": 0, "failed_batches": 0}
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset : offset + batch_size]
        try:
            works = client.works_batch(batch)
        except OpenAlexBudgetExhausted:
            raise
        except Exception:
            stats["failed_batches"] += 1
            continue
        for work in works:
            paper_id = short_id(work.get("id"))
            if paper_id not in papers:
                continue
            stats["returned"] += 1
            abstract = abstract_from_inverted(work.get("abstract_inverted_index"))
            if abstract:
                papers[paper_id] = replace(papers[paper_id], abstract=abstract)
                stats["filled"] += 1
    stats["missing_after"] = sum(not paper.abstract for paper in papers.values())
    return CorpusStore(papers.values(), store.reviews.values()), stats


def paper_from_work(work: dict[str, Any], *, citers: Iterable[str] = ()) -> Paper:
    paper_id = short_id(work.get("id"))
    if not paper_id:
        raise ValueError("OpenAlex work has no id")
    doi = work.get("doi")
    if doi and doi.startswith("https://doi.org/"):
        doi = doi.removeprefix("https://doi.org/")
    source = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
    return Paper(
        id=paper_id,
        title=work.get("display_name") or work.get("title") or "Untitled",
        abstract=abstract_from_inverted(work.get("abstract_inverted_index")),
        publication_year=work.get("publication_year"),
        doi=doi,
        venue=source,
        references=tuple(
            value for item in work.get("referenced_works", []) if (value := short_id(item))
        ),
        citers=tuple(citers),
    )


def enrich_one_hop(
    store: CorpusStore,
    client: OpenAlexClient,
    *,
    max_seed_papers: int | None = None,
    references_per_work: int = 20,
    citers_per_work: int = 20,
) -> CorpusStore:
    """Enrich a deterministic prefix of corpus works and add bounded one-hop neighbors."""
    ids = sorted(store.papers)
    if max_seed_papers is not None:
        ids = ids[:max_seed_papers]
    papers = dict(store.papers)
    for paper_id in ids:
        work = client.work(paper_id)
        citer_works = client.citers(paper_id, citers_per_work)
        citer_ids = [value for item in citer_works if (value := short_id(item.get("id")))]
        enriched = paper_from_work(work, citers=citer_ids)
        existing = papers[paper_id]
        papers[paper_id] = replace(
            enriched,
            abstract=enriched.abstract or existing.abstract,
            references=enriched.references[:references_per_work],
        )
        for citer_work in citer_works:
            citer = paper_from_work(citer_work)
            papers.setdefault(citer.id, citer)
        for reference_id in papers[paper_id].references:
            if reference_id in papers:
                continue
            try:
                papers[reference_id] = paper_from_work(client.work(reference_id))
            except Exception:
                # Missing/deleted works do not invalidate the rest of the frozen build.
                continue
    return CorpusStore(papers.values(), store.reviews.values())

