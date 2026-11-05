from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .agents import RandomAgent, RankedRetrievalAgent, SnowballAgent
from .environment import LiteratureSearchEnv, allowed_ids_for
from .models import EpisodeSpec
from .retrieval import BM25Retriever, DenseRetriever, PooledRetrievers
from .runner import run_episode, write_run
from .store import CorpusStore


def _make_agent(name: str, config: dict[str, Any], seed: int, backends: dict[str, Any]):
    label = config.get("label", name)
    if name == "snowball":
        return SnowballAgent(name=label, include_threshold=float(config.get("include_threshold", 0.16)))
    if name == "random":
        return RandomAgent(seed=seed, name=label)
    if name == "retrieval":
        return RankedRetrievalAgent(name=config.get("label", "retrieval"))
    if name == "llm":
        from .llm import LLMSearchAgent

        agent = LLMSearchAgent(
            backend=backends[config["label"]],
            name=config["label"],
            temperature=float(config.get("temperature", 0.0)),
            seed=seed if float(config.get("temperature", 0.0)) > 0 else None,
            record_turns=bool(config.get("record_turns", True)),
        )
        if config.get("force_expand_every"):
            from .forced import ForcedExpansionAgent

            agent = ForcedExpansionAgent(agent, every=int(config["force_expand_every"]), name=config["label"])
        return agent
    raise ValueError(f"unsupported matrix agent: {name}")


def _backend(config: dict[str, Any]):
    from .llm import OpenAIChatBackend, SpendLedger

    ledger = None
    if config.get("spend_ledger"):
        ledger = SpendLedger(config["spend_ledger"], float(config["spend_cap_usd"]))
    return OpenAIChatBackend(
        config["model"],
        ledger=ledger,
        base_url=config.get("base_url") or os.environ.get("MODEL_API_BASE"),
        api_key_env=config.get("api_key_env", "MODEL_API_KEY"),
        cache_dir=config.get("cache_dir"),
        max_tokens=int(config.get("max_tokens", 600)),
        extra_body=config.get("extra_body"),
        price_per_mtok=tuple(config["price_per_mtok"]) if config.get("price_per_mtok") else None,
        system_suffix=config.get("system_suffix", ""),
    )


def matrix_cells(config_path: Path) -> tuple[dict[str, Any], CorpusStore, list[dict[str, Any]]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    store = CorpusStore.load(config["data_dir"])
    split_path = Path(config["data_dir"]) / "splits.json"
    splits = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else {}
    review_ids = config.get("reviews") or splits.get(config.get("split", "test")) or sorted(store.reviews)
    cells: list[dict[str, Any]] = []
    for review_id in review_ids:
        for agent_config in config["agents"]:
            for mode in agent_config.get("modes", config.get("modes", ["hybrid"])):
                for budget in agent_config.get("budgets", config.get("budgets", [10, 20, 40])):
                    for repeat in range(int(agent_config.get("repeats", 1))):
                        cells.append(
                            {
                                "review_id": review_id,
                                "agent_config": agent_config,
                                "mode": mode,
                                "budget": int(budget),
                                "repeat": repeat,
                            }
                        )
    return config, store, cells


class _Retrievers:
    """Retriever lookup by (name, review). ``index_scope: pool`` builds per-review indexes."""

    def __init__(self, config: dict[str, Any], store: CorpusStore) -> None:
        self.store = store
        self.scope = config.get("index_scope", "corpus")
        self.dense_index = config.get("dense_index")
        self._corpus_bm25: BM25Retriever | None = None
        self._pooled = PooledRetrievers(BM25Retriever)
        self._dense: DenseRetriever | None = None

    def get(self, name: str, review) -> Any:
        if name == "bm25":
            if self.scope == "pool":
                allowed = allowed_ids_for(self.store, review)
                return self._pooled.for_pool(review.id, (self.store.papers[i] for i in sorted(allowed)))
            if self._corpus_bm25 is None:
                self._corpus_bm25 = BM25Retriever(self.store.papers.values())
            return self._corpus_bm25
        if name == "specter2":
            if not self.dense_index:
                raise ValueError("set dense_index in the matrix config to use specter2")
            if self._dense is None:
                self._dense = DenseRetriever(self.dense_index)
            return self._dense  # allowed_ids filtering makes it pool-scoped
        raise ValueError(f"unsupported retriever: {name}")


def run_matrix(
    config_path: Path,
    index: int | None = None,
    *,
    workers: int = 1,
    shard: tuple[int, int] | None = None,
    skip_existing: bool = False,
) -> list[Path]:
    config, store, cells = matrix_cells(config_path)
    if index is not None:
        if index < 0 or index >= len(cells):
            raise IndexError(f"matrix index {index} outside [0, {len(cells)})")
        cells = [cells[index]]
    elif shard is not None:
        cells = cells[shard[0] :: shard[1]]
    retrievers = _Retrievers(config, store)
    web_store = None
    if config.get("universe") == "openalex":
        from .openalex import OpenAlexClient
        from .web import WebStore

        web_store = WebStore(store, OpenAlexClient(config["openalex_cache"], delay_seconds=0.12))
    backends = {
        agent["label"]: _backend(agent) for agent in config["agents"] if agent["name"] == "llm"
    }
    output_dir = Path(config["output_dir"])

    def output_for(cell: dict[str, Any]) -> Path:
        label = cell["agent_config"].get("label", cell["agent_config"]["name"])
        return output_dir / (
            f"{cell['review_id']}__{label}__{cell['mode']}__b{cell['budget']}__r{cell['repeat']}.json"
        )

    def run_cell(cell: dict[str, Any]) -> Path:
        output = output_for(cell)
        if skip_existing and output.exists():
            return output
        review = store.review(cell["review_id"])
        agent_config = cell["agent_config"]
        seed = int(config.get("seed", 329)) + cell["repeat"]
        spec = EpisodeSpec(
            review.id, review.question, review.eligibility_criteria, cell["budget"], cell["mode"]
        )
        if web_store is not None:
            from .web import WebLiteratureSearchEnv

            env = WebLiteratureSearchEnv(web_store, spec)
        else:
            env = LiteratureSearchEnv(store, retrievers.get(agent_config.get("retriever", "bm25"), review), spec)
        run = run_episode(env, _make_agent(agent_config["name"], agent_config, seed, backends))
        run["matrix"] = {
            "config": str(config_path),
            "label": agent_config.get("label", agent_config["name"]),
            "repeat": cell["repeat"],
            "index_scope": retrievers.scope,
            "retriever": "openalex-live" if web_store is not None else agent_config.get("retriever", "bm25"),
            "universe": config.get("universe", "pool"),
            "data_dir": config["data_dir"],
        }
        return write_run(run, output)

    if workers <= 1:
        return [run_cell(cell) for cell in cells]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_cell, cells))
