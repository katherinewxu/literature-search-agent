from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
import re
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import Paper, Review
from .openalex import abstract_from_inverted
from .store import CorpusStore, normalize_openalex_id


def fixture_store() -> CorpusStore:
    papers = [
        Paper(
            "W1",
            "Mindfulness therapy prevents recurrent major depression",
            "Randomized trial of mindfulness cognitive therapy for relapse prevention in remitted depression.",
            2018,
            references=("W3", "W4"),
            citers=("W2", "W90"),
        ),
        Paper(
            "W2",
            "Cognitive therapy and relapse in depression",
            "A prospective controlled study of cognitive therapy after recovery from major depressive disorder.",
            2019,
            references=("W1", "W3"),
            citers=("W5",),
        ),
        Paper(
            "W3",
            "Prospective predictors of depressive relapse",
            "Longitudinal study measuring cognitive risk factors before recurrence of major depression.",
            2015,
            references=("W4",),
            citers=("W1", "W2"),
        ),
        Paper(
            "W4",
            "Seasonal affective symptoms in older adults",
            "Cross-sectional survey of seasonal symptoms in late life.",
            2014,
            citers=("W1", "W3"),
        ),
        Paper(
            "W5",
            "Maintenance treatment for bipolar disorder",
            "Pharmacological maintenance therapy in bipolar disorder.",
            2020,
            references=("W2",),
        ),
        Paper(
            "W6",
            "Recurrent depression after remission",
            "Clinical interview study of recurrence after remission and recovery from unipolar depression.",
            2017,
            references=("W3",),
        ),
        Paper("W7", "Rodent stress model", "Animal model of chronic stress.", 2016),
        Paper("W8", "Diabetes cohort outcomes", "Longitudinal metabolic outcomes.", 2019),
        Paper("W9", "Depression review protocol", "Protocol for evidence synthesis.", 2018),
        Paper("W10", "Adolescent mood follow-up", "Prospective adolescent mood cohort.", 2017),
        Paper(
            "W90",
            "Systematic review of predictors of depression relapse",
            "Review article.",
            2021,
            references=("W1", "W2", "W3", "W6"),
        ),
        Paper(
            "W91",
            "Later paper citing the systematic review",
            "Published after the review and forbidden.",
            2022,
            references=("W90",),
        ),
    ]
    review = Review(
        id="fixture_depression",
        question="Which prospective studies predict relapse after remission from major depression?",
        eligibility_criteria=(
            "Prospective longitudinal human studies of remitted unipolar major depression "
            "with predictors measured before clinically diagnosed relapse."
        ),
        gold_ids=frozenset({"W1", "W2", "W3", "W6"}),
        search_cutoff_year=2020,
        review_paper_id="W90",
        domain="medicine",
    )
    return CorpusStore(papers, [review])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first(row: dict[str, str], names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        if row.get(name):
            return row[name].strip()
    return default


def _openalex_ids(value: str) -> tuple[str, ...]:
    if not value or value.lower() in {"nan", "none"}:
        return ()
    parsed: Any
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = [part.strip() for part in value.split("|")]
    if not isinstance(parsed, (list, tuple)):
        return ()
    values = []
    for item in parsed:
        try:
            values.append(normalize_openalex_id(str(item)))
        except ValueError:
            continue
    return tuple(dict.fromkeys(values))


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _review_info_from_source_dir(directory: Path) -> dict[str, Any]:
    """Review fields from a SYNERGY source folder (metadata.json + metadata_publication.json)."""
    info: dict[str, Any] = {}
    meta_path = directory / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        criteria = _clean((meta.get("publication") or {}).get("eligibility_criteria"))
        if criteria:
            info["eligibility_criteria"] = criteria
    publication_path = directory / "metadata_publication.json"
    if publication_path.exists():
        publication = json.loads(publication_path.read_text(encoding="utf-8"))
        if _clean(publication.get("title")):
            info["title"] = _clean(publication["title"])
        if publication.get("publication_year"):
            info["publication_year"] = int(publication["publication_year"])
        if _clean(publication.get("publication_date")):
            info["publication_date"] = _clean(publication["publication_date"])
        if _clean(publication.get("id")):
            info["review_paper_id"] = _clean(publication["id"])
        topic = publication.get("primary_topic") or {}
        field_name = _clean((topic.get("field") or {}).get("display_name"))
        if field_name:
            info["domain"] = field_name
    return info


def _review_info_from_csv(path: Path) -> dict[str, dict[str, Any]]:
    """Review fields keyed by ``key`` from SYNERGY's exported ``review_metadata.csv``."""
    csv.field_size_limit(sys.maxsize)
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = _clean(row.get("key"))
            if not key:
                continue
            info: dict[str, Any] = {}
            for name in ("title", "eligibility_criteria", "publication_date"):
                if _clean(row.get(name)):
                    info[name] = _clean(row[name])
            # SYNERGY's research_question is inconsistent (some describe a companion
            # experiment), so the review title is the uniform episode question.
            year = _clean(row.get("publication_year"))
            if year:
                info["publication_year"] = int(float(year))
            if _clean(row.get("openalex_id")):
                info["review_paper_id"] = _clean(row["openalex_id"])
            if _clean(row.get("primary_topic_field")):
                info["domain"] = _clean(row["primary_topic_field"])
            result[key] = info
    return result


def load_review_metadata(
    keys: list[str] | tuple[str, ...],
    *,
    source_root: Path | None = None,
    metadata_csvs: list[Path] | tuple[Path, ...] = (),
    overrides: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Merge review metadata; later sources win: source folders, CSV exports, reviews.json."""
    merged: dict[str, dict[str, Any]] = {key: {} for key in keys}
    if source_root is not None:
        for key in keys:
            merged[key].update(_review_info_from_source_dir(source_root / key))
    for path in metadata_csvs:
        for key, info in _review_info_from_csv(path).items():
            if key in merged:
                merged[key].update(info)
    if overrides is not None and overrides.exists():
        for key, info in json.loads(overrides.read_text(encoding="utf-8")).items():
            if key in merged:
                merged[key].update(info)
    return merged


def _make_review(
    review_id: str,
    info: dict[str, Any],
    gold: set[str],
    candidates: set[str],
    domains: Counter[str],
) -> Review:
    publication_date = info.get("publication_date")
    if "publication_year" in info:
        publication_year = int(info["publication_year"])
    elif publication_date:
        publication_year = int(str(publication_date)[:4])
    else:
        # Conservative fallback: the key suffix is the publication year, so search the year before.
        inferred = re.search(r"(?:19|20)\d{2}", review_id)
        publication_year = int(inferred.group(0)) if inferred else datetime.now(UTC).year
        info = {"search_cutoff_year": publication_year - 1, **info}
    cutoff_date = info.get("search_cutoff_date", publication_date)
    cutoff_year = int(info.get("search_cutoff_year", publication_year))
    review_paper_id = info.get("review_paper_id")
    if review_paper_id:
        review_paper_id = normalize_openalex_id(str(review_paper_id))
    return Review(
        id=review_id,
        question=info.get("question") or info.get("title") or review_id.replace("_", " "),
        eligibility_criteria=info.get("eligibility_criteria", "Not provided."),
        gold_ids=frozenset(gold),
        search_cutoff_year=cutoff_year,
        review_paper_id=review_paper_id,
        domain=info.get("domain", domains.most_common(1)[0][0] if domains else "unknown"),
        search_cutoff_date=cutoff_date,
        candidate_ids=frozenset(candidates),
    )


def _with_reverse_edges(papers: dict[str, Paper]) -> dict[str, Paper]:
    citers: dict[str, set[str]] = {paper_id: set() for paper_id in papers}
    for paper in papers.values():
        for reference_id in paper.references:
            if reference_id in citers:
                citers[reference_id].add(paper.id)
    return {
        paper_id: replace(paper, citers=tuple(sorted(citers[paper_id])))
        for paper_id, paper in papers.items()
    }


def build_from_synergy_csv(source: str | Path) -> CorpusStore:
    """Build from an unpacked SYNERGY directory without importing its package.

    One CSV per review is expected. Review metadata comes from any
    ``metadata/review_metadata.csv`` exports below ``source``; overrides can be
    supplied in ``reviews.json``.
    """
    source = Path(source)
    review_csvs = [
        path for path in sorted(source.glob("**/*.csv")) if path.parent.name != "metadata"
    ]
    keys = [path.stem for path in review_csvs]
    metadata = load_review_metadata(
        keys,
        metadata_csvs=sorted(source.glob("**/metadata/review_metadata.csv")),
        overrides=source / "reviews.json",
    )
    papers: dict[str, Paper] = {}
    reviews: list[Review] = []
    csv.field_size_limit(sys.maxsize)
    for csv_path in review_csvs:
        review_id = csv_path.stem
        gold: set[str] = set()
        candidates: set[str] = set()
        domains: Counter[str] = Counter()
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                raw_id = _first(row, ("id", "openalex_id", "openalex"))
                if not raw_id:
                    continue
                try:
                    paper_id = normalize_openalex_id(raw_id)
                except ValueError:
                    continue
                candidates.add(paper_id)
                title = _first(row, ("title", "primary_title"), "Untitled")
                abstract = _first(row, ("abstract", "abstract_text"))
                year_text = _first(row, ("publication_year", "year"))
                year = int(year_text) if year_text.isdigit() else None
                papers.setdefault(
                    paper_id,
                    Paper(
                        id=paper_id,
                        title=title,
                        abstract=abstract,
                        publication_year=year,
                        doi=_first(row, ("doi",)) or None,
                        venue=_first(row, ("journal_name", "venue")) or None,
                        references=_openalex_ids(_first(row, ("referenced_works",))),
                        publication_date=_first(row, ("publication_date",)) or None,
                    ),
                )
                domain = _first(row, ("primary_topic_domain", "domain"))
                if domain:
                    domains[domain] += 1
                label = _first(row, ("label_included", "included", "label")).lower()
                if label in {"1", "true", "included", "yes"}:
                    gold.add(paper_id)
        if not gold:
            continue
        reviews.append(_make_review(review_id, metadata[review_id], gold, candidates, domains))
    if not reviews:
        raise ValueError(f"no labeled review CSV files found under {source}")
    return CorpusStore(_with_reverse_edges(papers).values(), reviews)


def _paper_from_openalex_work(work: dict[str, Any]) -> Paper:
    source = ((work.get("primary_location") or {}).get("source") or {})
    index = work.get("abstract_inverted_index_cleaned") or work.get("abstract_inverted_index")
    return Paper(
        id=normalize_openalex_id(work["id"]),
        title=_clean(work.get("title") or work.get("display_name")) or "Untitled",
        abstract=abstract_from_inverted(index),
        publication_year=work.get("publication_year"),
        doi=_clean(work.get("doi")) or None,
        venue=_clean(source.get("display_name")) or None,
        references=_openalex_ids(json.dumps(work.get("referenced_works") or [])),
        publication_date=_clean(work.get("publication_date")) or None,
    )


def build_from_synergy_source(
    source_root: str | Path,
    review_keys: list[str],
    *,
    metadata_csvs: list[Path] | tuple[Path, ...] = (),
    overrides: Path | None = None,
) -> tuple[CorpusStore, dict[str, Any]]:
    """Build from SYNERGY source folders (labels.csv + works_*.zip), without OA filtering.

    The ``synergy_dataset`` exporter yields only open-access works with an abstract,
    which silently drops most candidates and gold studies. Reading the source keeps
    every labeled record; works without abstracts keep an empty abstract.
    Returns the store and a per-review ingestion report.
    """
    source_root = Path(source_root)
    metadata = load_review_metadata(
        review_keys, source_root=source_root, metadata_csvs=metadata_csvs, overrides=overrides
    )
    papers: dict[str, Paper] = {}
    reviews: list[Review] = []
    report: dict[str, Any] = {}
    for key in review_keys:
        directory = source_root / key
        labels: dict[str, int] = {}
        with (directory / "labels.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    paper_id = normalize_openalex_id(_clean(row.get("openalex_id")))
                except ValueError:
                    continue
                included = _clean(row.get("label_included")) in {"1", "1.0", "true", "True"}
                labels[paper_id] = max(labels.get(paper_id, 0), int(included))
        found: set[str] = set()
        domains: Counter[str] = Counter()
        for archive in sorted(directory.glob("works_*.zip")):
            with zipfile.ZipFile(archive) as bundle:
                for name in sorted(bundle.namelist()):
                    for work in json.loads(bundle.read(name)):
                        paper = _paper_from_openalex_work(work)
                        if paper.id not in labels:
                            continue
                        found.add(paper.id)
                        papers.setdefault(paper.id, paper)
                        topic = work.get("primary_topic") or {}
                        domain = _clean((topic.get("field") or {}).get("display_name"))
                        if domain:
                            domains[domain] += 1
        gold = {paper_id for paper_id, label in labels.items() if label and paper_id in found}
        missing_gold = sorted(
            paper_id for paper_id, label in labels.items() if label and paper_id not in found
        )
        report[key] = {
            "labeled_records": len(labels),
            "labeled_gold": sum(labels.values()),
            "works_found": len(found),
            "gold_found": len(gold),
            "missing_gold": missing_gold,
        }
        if not gold:
            continue
        reviews.append(_make_review(key, metadata[key], gold, found, domains))
    if not reviews:
        raise ValueError(f"no labeled reviews found under {source_root} for {review_keys}")
    return CorpusStore(_with_reverse_edges(papers).values(), reviews), report


def deterministic_split(reviews: list[Review]) -> dict[str, list[str]]:
    """Hash-ordered review-level split (~46/23/31). Per-domain splitting left the test
    split empty when, as in SYNERGY, most domains contain a single review."""
    ordered = sorted(reviews, key=lambda review: hashlib.sha256(review.id.encode()).hexdigest())
    result: dict[str, list[str]] = {"train": [], "development": [], "test": []}
    for index, review in enumerate(ordered):
        fraction = index / max(1, len(ordered))
        split = "train" if fraction < 0.46 else "development" if fraction < 0.69 else "test"
        result[split].append(review.id)
    return result


def write_artifacts(
    store: CorpusStore,
    output: str | Path,
    source: str,
    *,
    ingestion_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output)
    store.save(output)
    splits = deterministic_split(list(store.reviews.values()))
    (output / "splits.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    hashed = ["works.jsonl", "reviews.jsonl", "splits.json"]
    if ingestion_report is not None:
        (output / "ingestion_report.json").write_text(
            json.dumps(ingestion_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        hashed.append("ingestion_report.json")
    manifest = {
        "schema_version": 1,
        "built_at": datetime.now(UTC).isoformat(),
        "source": source,
        "paper_count": len(store.papers),
        "review_count": len(store.reviews),
        "files": {name: _sha256(output / name) for name in hashed},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _normalized_title(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def audit_store(store: CorpusStore, *, min_gold: int = 15) -> dict[str, Any]:
    """Leakage checks (hard issues) plus data-quality statistics and warnings per review."""
    from .environment import allowed_ids_for

    issues: list[dict[str, str]] = []
    warnings: list[dict[str, Any]] = []
    reviews: dict[str, Any] = {}
    for review in sorted(store.reviews.values(), key=lambda item: item.id):
        for paper_id in sorted(review.gold_ids - set(store.papers)):
            issues.append({"review_id": review.id, "code": "missing_gold", "paper_id": paper_id})
        allowed = allowed_ids_for(store, review)
        for paper_id in allowed:
            paper = store.papers[paper_id]
            leak = None
            if paper_id == review.review_paper_id:
                leak = "review_paper_visible"
            elif review.review_paper_id and review.review_paper_id in paper.references:
                leak = "review_citer_visible"
            elif paper.publication_year and paper.publication_year > review.search_cutoff_year:
                leak = "post_cutoff_visible"
            if leak:
                issues.append({"review_id": review.id, "code": leak, "paper_id": paper_id})
        pool = [store.papers[i] for i in (review.candidate_ids or store.papers) if i in store.papers]
        gold = [store.papers[i] for i in sorted(review.gold_ids) if i in store.papers]
        hidden_gold = sorted(paper.id for paper in gold if paper.id not in allowed)
        for paper_id in hidden_gold:
            warnings.append({"review_id": review.id, "code": "post_cutoff_gold", "paper_id": paper_id})
        if len(review.gold_ids) < min_gold:
            warnings.append(
                {"review_id": review.id, "code": "few_gold", "gold_count": len(review.gold_ids)}
            )
        pool_ids = {paper.id for paper in pool}
        def linked(paper: Paper, targets: set[str]) -> bool:
            return any(i in targets and i != paper.id for i in paper.references + paper.citers)
        gold_ids = {paper.id for paper in gold}
        titles = Counter(_normalized_title(paper.title) for paper in pool if paper.title)
        reviews[review.id] = {
            "domain": review.domain,
            "search_cutoff_year": review.search_cutoff_year,
            "search_cutoff_date": review.search_cutoff_date,
            "has_question": review.question != review.id.replace("_", " "),
            "has_eligibility_criteria": review.eligibility_criteria != "Not provided.",
            "review_paper_id": review.review_paper_id,
            "candidates": len(pool),
            "visible_candidates": len(allowed),
            "gold": len(review.gold_ids),
            "visible_gold": len(gold_ids & allowed),
            "prevalence": len(review.gold_ids) / max(1, len(pool)),
            "empty_abstract_rate": sum(not paper.abstract for paper in pool) / max(1, len(pool)),
            "gold_empty_abstract_rate": sum(not paper.abstract for paper in gold) / max(1, len(gold)),
            "missing_doi_rate": sum(not paper.doi for paper in pool) / max(1, len(pool)),
            "missing_year_rate": sum(not paper.publication_year for paper in pool) / max(1, len(pool)),
            "duplicate_titles": sum(count - 1 for count in titles.values() if count > 1),
            "candidate_edge_coverage": sum(linked(p, pool_ids) for p in pool) / max(1, len(pool)),
            "gold_edge_coverage": sum(linked(p, pool_ids) for p in gold) / max(1, len(gold)),
            "gold_linked_to_gold": sum(linked(p, gold_ids) for p in gold) / max(1, len(gold)),
            "mean_in_pool_references": sum(
                sum(i in pool_ids for i in paper.references) for paper in pool
            ) / max(1, len(pool)),
        }
    return {
        "ok": not issues,
        "paper_count": len(store.papers),
        "review_count": len(store.reviews),
        "issues": issues,
        "warnings": warnings,
        "reviews": reviews,
    }
