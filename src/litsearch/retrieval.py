from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Protocol

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


class PooledRetrievers:
    """Lazily builds one retriever per review over that review's visible pool.

    Scoring statistics (BM25 IDF, the dense candidate set) then depend only on the
    review's own search universe, not on which other reviews share the store.
    """

    def __init__(self, factory) -> None:
        self.factory = factory
        self._cache: dict[str, Retriever] = {}
        self._lock = threading.Lock()

    def for_pool(self, key: str, papers: Iterable[Paper]) -> Retriever:
        with self._lock:
            if key not in self._cache:
                self._cache[key] = self.factory(list(papers))
            return self._cache[key]


SPECTER2_BASE = "allenai/specter2_base"
SPECTER2_DOC_ADAPTER = "allenai/specter2"
SPECTER2_QUERY_ADAPTER = "allenai/specter2_adhoc_query"


class Specter2Encoder:
    """SPECTER2 encoder: proximity adapter for papers, ad-hoc query adapter for queries."""

    def __init__(self, adapter: str, device: str | None = None) -> None:
        try:
            import numpy as np
            import torch
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the 'ml' extra to use SPECTER2") from exc
        self._np = np
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(SPECTER2_BASE)
        self.model = AutoAdapterModel.from_pretrained(SPECTER2_BASE)
        name = self.model.load_adapter(adapter, source="hf", load_as=adapter.rsplit("/", 1)[-1])
        self.model.set_active_adapters(name)
        if not self.model.active_adapters:
            raise RuntimeError(f"SPECTER2 adapter {adapter} failed to activate")
        self.adapter_name = name
        self.model.to(self.device).eval()

    def encode(self, texts: list[str], batch_size: int = 64, log_every: int = 0):
        import time

        batches = []
        started = time.time()
        with self._torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                if log_every and offset % log_every < batch_size:
                    rate = offset / max(time.time() - started, 1e-9)
                    print(f"encoded {offset}/{len(texts)} ({rate:.1f}/s)", flush=True)
                inputs = self.tokenizer(
                    texts[offset : offset + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    return_token_type_ids=False,
                ).to(self.device)
                vector = self.model(**inputs).last_hidden_state[:, 0, :].float().cpu().numpy()
                norm = self._np.linalg.norm(vector, axis=1, keepdims=True).clip(min=1e-8)
                batches.append(vector / norm)
        return self._np.concatenate(batches).astype("float32")


def paper_text(paper: Paper, sep: str = " [SEP] ") -> str:
    return paper.title + sep + paper.abstract


def build_specter2_index(papers: Iterable[Paper], output: str | Path) -> dict[str, int]:
    """Encode every paper once; writes ``<output>.npy`` and ``<output>.ids.json``."""
    import numpy as np

    output = Path(output)
    paper_list = sorted(papers, key=lambda paper: paper.id)
    encoder = Specter2Encoder(SPECTER2_DOC_ADAPTER)
    print(f"adapter {encoder.adapter_name} active on {encoder.device}", flush=True)
    vectors = encoder.encode(
        [paper_text(p, encoder.tokenizer.sep_token) for p in paper_list], log_every=2048
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output.with_suffix(".npy"), vectors)
    output.with_suffix(".ids.json").write_text(
        json.dumps([paper.id for paper in paper_list]) + "\n", encoding="utf-8"
    )
    return {"papers": len(paper_list), "dim": int(vectors.shape[1])}


_QUERY_ENCODER = None
_QUERY_ENCODER_LOCK = threading.Lock()


def _shared_query_encoder():
    """One query encoder per process; the global lock also serializes the ``adapters``
    lazy import, which is not thread-safe."""
    global _QUERY_ENCODER
    with _QUERY_ENCODER_LOCK:
        if _QUERY_ENCODER is None:
            _QUERY_ENCODER = Specter2Encoder(SPECTER2_QUERY_ADAPTER)
        return _QUERY_ENCODER


class DenseRetriever:
    """Exact inner-product search over precomputed SPECTER2 vectors (no ANN index)."""

    def __init__(self, index_path: str | Path, encoder=None) -> None:
        import numpy as np

        index_path = Path(index_path)
        self._np = np
        self.ids: list[str] = json.loads(index_path.with_suffix(".ids.json").read_text())
        self.vectors = np.load(index_path.with_suffix(".npy"), mmap_mode="r")
        self.row = {paper_id: index for index, paper_id in enumerate(self.ids)}
        self._encoder = encoder
        self._lock = threading.Lock()
        self._query_cache: dict[str, Any] = {}

    def _query_vector(self, query: str):
        with self._lock:
            if query not in self._query_cache:
                if self._encoder is None:
                    self._encoder = _shared_query_encoder()
                with _QUERY_ENCODER_LOCK:
                    self._query_cache[query] = self._encoder.encode([query])[0]
            return self._query_cache[query]

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[str]:
        if allowed_ids is None:
            rows = self._np.arange(len(self.ids))
        else:
            rows = self._np.array(sorted(self.row[i] for i in allowed_ids if i in self.row))
        if len(rows) == 0:
            return []
        scores = self.vectors[rows] @ self._query_vector(query)
        top = self._np.argsort(-scores, kind="stable")[:k]
        return [self.ids[rows[index]] for index in top]
