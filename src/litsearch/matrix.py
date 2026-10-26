from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agents import OpenAICompatibleAgent, RandomAgent, RankedRetrievalAgent, SnowballAgent
from .environment import LiteratureSearchEnv
from .models import EpisodeSpec
from .retrieval import BM25Retriever, Specter2Retriever
from .runner import run_episode, write_run
from .store import CorpusStore


def _make_agent(name: str, config: dict[str, Any], seed: int):
    if name == "snowball":
        return SnowballAgent(include_threshold=float(config.get("include_threshold", 0.16)))
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "retrieval":
        return RankedRetrievalAgent(name=config.get("label", "retrieval"))
    if name == "prompted":
        return OpenAICompatibleAgent(model=config.get("model", "configured-model"))
    raise ValueError(f"unsupported matrix agent: {name}")


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
                for budget in config.get("budgets", [10, 20, 40]):
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


def run_matrix(config_path: Path, index: int | None = None) -> list[Path]:
    config, store, cells = matrix_cells(config_path)
    if index is not None:
        if index < 0 or index >= len(cells):
            raise IndexError(f"matrix index {index} outside [0, {len(cells)})")
        cells = [cells[index]]
    retrievers: dict[str, Any] = {"bm25": BM25Retriever(store.papers.values())}
    output_dir = Path(config["output_dir"])
    outputs: list[Path] = []
    for cell in cells:
        review = store.review(cell["review_id"])
        agent_config = cell["agent_config"]
        name = agent_config["name"]
        output_name = agent_config.get("label", name)
        retriever_name = agent_config.get("retriever", "bm25")
        if retriever_name not in retrievers:
            if retriever_name != "specter2":
                raise ValueError(f"unsupported retriever: {retriever_name}")
            retrievers[retriever_name] = Specter2Retriever(
                store.papers.values(), cache_path=agent_config.get("index_path")
            )
        repeat = cell["repeat"]
        seed = int(config.get("seed", 329)) + repeat
        spec = EpisodeSpec(
            review.id,
            review.question,
            review.eligibility_criteria,
            cell["budget"],
            cell["mode"],
        )
        run = run_episode(
            LiteratureSearchEnv(store, retrievers[retriever_name], spec),
            _make_agent(name, agent_config, seed),
        )
        output = output_dir / (
            f"{review.id}__{output_name}__{cell['mode']}__b{cell['budget']}__r{repeat}.json"
        )
        outputs.append(write_run(run, output))
    return outputs

