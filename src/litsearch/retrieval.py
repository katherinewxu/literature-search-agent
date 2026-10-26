from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Protocol

from .models import Paper


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Retriever(Protocol):
    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[str]: ...


class BM25Retriever:
    """Okapi BM25 over title (double-weighted) and abstract via an inverted index."""

    def __init__(self, papers: Iterable[Paper], k1: float = 1.5, b: float = 0.75) -> None:
        self.papers = {paper.id: paper for paper in papers}
        self.k1 = k1
        self.b = b
        self.doc_lengths: dict[str, int] = {}
        self.postings: dict[str, list[tuple[str, int]]] = {}
        for paper in self.papers.values():
            tokens = tokenize(f"{paper.title} {paper.title} {paper.abstract}")
            self.doc_lengths[paper.id] = len(tokens)
            for term, frequency in Counter(tokens).items():
                self.postings.setdefault(term, []).append((paper.id, frequency))
        self.avg_length = sum(self.doc_lengths.values()) / max(1, len(self.doc_lengths))
        count = len(self.papers)
        self.idf = {
            term: math.log(1.0 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self.postings.items()
        }

    def scores(self, query: str) -> dict[str, float]:
        result: dict[str, float] = {}
        avg_length = max(self.avg_length, 1)
        for term, query_frequency in Counter(tokenize(query)).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            weight = query_frequency * idf * (self.k1 + 1)
            for paper_id, frequency in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_lengths[paper_id] / avg_length)
                result[paper_id] = result.get(paper_id, 0.0) + weight * frequency / (frequency + norm)
        return result

    def score(self, query: str, paper_id: str) -> float:
        return self.scores(query).get(paper_id, 0.0)

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[str]:
        scored = self.scores(query).items()
        if allowed_ids is not None:
            scored = [(paper_id, score) for paper_id, score in scored if paper_id in allowed_ids]
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:k]
        return [paper_id for paper_id, score in ranked if score > 0]


class Specter2Retriever:
    """Optional dense retriever loaded only when ML dependencies are present."""

    def __init__(self, papers: Iterable[Paper], cache_path: str | None = None) -> None:
        try:
            import faiss  # type: ignore
            import numpy as np
            import torch
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the 'ml' extra to use SPECTER2") from exc

        self._faiss = faiss
        self._np = np
        self._torch = torch
        paper_list = list(papers)
        self.ids = sorted(paper.id for paper in paper_list)
        paper_map = {paper.id: paper for paper in paper_list}
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self.model.load_adapter("allenai/specter2", source="hf", set_active=True)
        self.model.eval()
        cache = None if cache_path is None else __import__("pathlib").Path(cache_path)
        ids_cache = None if cache is None else cache.with_suffix(cache.suffix + ".ids.json")
        if cache is not None and cache.exists() and ids_cache is not None and ids_cache.exists():
            import json

            self.index = faiss.read_index(str(cache))
            self.ids = json.loads(ids_cache.read_text(encoding="utf-8"))
        else:
            texts = [
                paper_map[paper_id].title
                + self.tokenizer.sep_token
                + paper_map[paper_id].abstract
                for paper_id in self.ids
            ]
            vectors = self._encode(texts)
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
            if cache is not None and ids_cache is not None:
                import json

                cache.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(cache))
                ids_cache.write_text(json.dumps(self.ids) + "\n", encoding="utf-8")

    def _encode(self, texts: list[str]):
        batches = []
        with self._torch.inference_mode():
            for offset in range(0, len(texts), 32):
                inputs = self.tokenizer(
                    texts[offset : offset + 32],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
                vector = self.model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
                norm = self._np.linalg.norm(vector, axis=1, keepdims=True).clip(min=1e-8)
                batches.append(vector / norm)
        return self._np.ascontiguousarray(self._np.concatenate(batches).astype("float32"))

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[str]:
        query_vector = self._encode([query])
        _, indices = self.index.search(query_vector, min(len(self.ids), max(k * 5, k)))
        values = [self.ids[index] for index in indices[0] if index >= 0]
        if allowed_ids is not None:
            values = [paper_id for paper_id in values if paper_id in allowed_ids]
        return values[:k]

