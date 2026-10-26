from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .models import Paper, Review


def normalize_openalex_id(value: str) -> str:
    value = value.strip().rstrip("/")
    if value.startswith("https://openalex.org/"):
        value = value.rsplit("/", 1)[-1]
    if not value or value[0].upper() != "W" or not value[1:].isdigit():
        raise ValueError(f"invalid OpenAlex work id: {value!r}")
    return "W" + value[1:]


class CorpusStore:
    """Immutable in-memory view over a built corpus.

    JSONL is the portable interchange format. The data builder can also emit
    Parquet when the optional data dependencies are installed.
    """

    def __init__(self, papers: Iterable[Paper], reviews: Iterable[Review]) -> None:
        self.papers = {paper.id: paper for paper in papers}
        self.reviews = {review.id: review for review in reviews}
        if len(self.papers) == 0:
            raise ValueError("corpus contains no papers")

    def paper(self, paper_id: str) -> Paper | None:
        return self.papers.get(paper_id)

    def review(self, review_id: str) -> Review:
        try:
            return self.reviews[review_id]
        except KeyError as exc:
            raise KeyError(f"unknown review: {review_id}") from exc

    @classmethod
    def load(cls, directory: str | Path) -> "CorpusStore":
        directory = Path(directory)
        papers: list[Paper] = []
        with (directory / "works.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                row["references"] = tuple(row.get("references", ()))
                row["citers"] = tuple(row.get("citers", ()))
                papers.append(Paper(**row))
        reviews: list[Review] = []
        with (directory / "reviews.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                row["gold_ids"] = frozenset(row["gold_ids"])
                row["candidate_ids"] = frozenset(row.get("candidate_ids", ()))
                reviews.append(Review(**row))
        return cls(papers, reviews)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "works.jsonl").open("w", encoding="utf-8") as stream:
            for paper in sorted(self.papers.values(), key=lambda item: item.id):
                stream.write(json.dumps(asdict(paper), sort_keys=True) + "\n")
        with (directory / "reviews.jsonl").open("w", encoding="utf-8") as stream:
            for review in sorted(self.reviews.values(), key=lambda item: item.id):
                row = asdict(review)
                row["gold_ids"] = sorted(review.gold_ids)
                row["candidate_ids"] = sorted(review.candidate_ids)
                stream.write(json.dumps(row, sort_keys=True) + "\n")

