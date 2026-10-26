from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents import OpenAICompatibleAgent, RandomAgent, RankedRetrievalAgent, SnowballAgent
from .data import (
    audit_store,
    build_from_synergy_csv,
    build_from_synergy_source,
    fixture_store,
    write_artifacts,
)
from .demo import write_html_replay
from .environment import LiteratureSearchEnv
from .evaluation import evaluate_runs
from .models import EpisodeSpec
from .openalex import OpenAlexClient, backfill_abstracts, enrich_one_hop
from .retrieval import BM25Retriever, Specter2Retriever
from .runner import run_episode, write_run
from .store import CorpusStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="litsearch")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    build = data_commands.add_parser("build")
    build.add_argument(
        "--source", choices=("fixture", "synergy", "synergy-source"), default="fixture"
    )
    build.add_argument("--input", type=Path, help="CSV folder, or SYNERGY source root")
    build.add_argument("--reviews", help="comma-separated review keys (synergy-source)")
    build.add_argument(
        "--review-metadata", type=Path, action="append", default=[],
        help="exported review_metadata.csv; repeatable",
    )
    build.add_argument("--overrides", type=Path, help="reviews.json with explicit overrides")
    build.add_argument("--output", type=Path, required=True)
    audit = data_commands.add_parser("audit")
    audit.add_argument("--data", type=Path, required=True)
    enrich = data_commands.add_parser("enrich")
    enrich.add_argument("--data", type=Path, required=True)
    enrich.add_argument("--output", type=Path, required=True)
    enrich.add_argument("--cache", type=Path, required=True)
    enrich.add_argument("--email")
    enrich.add_argument("--max-seed-papers", type=int)
    backfill = data_commands.add_parser("backfill-abstracts")
    backfill.add_argument("--data", type=Path, required=True)
    backfill.add_argument("--output", type=Path, required=True)
    backfill.add_argument("--cache", type=Path, required=True)
    backfill.add_argument("--email")

    run = commands.add_parser("run")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--review", required=True)
    run.add_argument(
        "--agent", choices=("random", "retrieval", "snowball", "prompted"), default="snowball"
    )
    run.add_argument("--retriever", choices=("bm25", "specter2"), default="bm25")
    run.add_argument("--mode", choices=("citation", "keyword", "hybrid"), default="hybrid")
    run.add_argument("--budget", type=int, choices=(10, 20, 40), default=20)
    run.add_argument("--model", default="configured-model")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", type=Path, required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)

    report = commands.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--output", type=Path)

    matrix = commands.add_parser("matrix")
    matrix.add_argument("--config", type=Path, required=True)
    matrix.add_argument("--index", type=int, help="run one zero-based matrix cell")
    matrix.add_argument("--count", action="store_true", help="print the number of matrix cells")

    demo = commands.add_parser("demo")
    demo_commands = demo.add_subparsers(dest="demo_command", required=True)
    demo_build = demo_commands.add_parser("build")
    demo_build.add_argument("--trace", type=Path, required=True)
    demo_build.add_argument("--output", type=Path, required=True)

    train_sft = commands.add_parser("train-sft")
    train_sft.add_argument("--config", type=Path, required=True)
    train_grpo = commands.add_parser("train-grpo")
    train_grpo.add_argument("--config", type=Path, required=True)
    return parser


def _agent(name: str, model: str, seed: int):
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "snowball":
        return SnowballAgent()
    if name == "retrieval":
        return RankedRetrievalAgent()
    return OpenAICompatibleAgent(model=model)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "data":
        if args.data_command == "build":
            report = None
            if args.source == "fixture":
                store = fixture_store()
            elif args.input is None:
                raise SystemExit(f"--input is required for --source {args.source}")
            elif args.source == "synergy":
                store = build_from_synergy_csv(args.input)
            else:
                if not args.reviews:
                    raise SystemExit("--reviews is required for --source synergy-source")
                store, report = build_from_synergy_source(
                    args.input,
                    [key.strip() for key in args.reviews.split(",") if key.strip()],
                    metadata_csvs=args.review_metadata,
                    overrides=args.overrides,
                )
            if args.output.exists() and any(args.output.iterdir()):
                raise SystemExit(f"refusing to overwrite non-empty {args.output}")
            manifest = write_artifacts(store, args.output, args.source, ingestion_report=report)
            print(json.dumps(manifest, indent=2))
            return 0
        if args.data_command == "enrich":
            store = CorpusStore.load(args.data)
            enriched = enrich_one_hop(
                store,
                OpenAlexClient(args.cache, email=args.email),
                max_seed_papers=args.max_seed_papers,
            )
            print(json.dumps(write_artifacts(enriched, args.output, "openalex-enriched"), indent=2))
            return 0
        if args.data_command == "backfill-abstracts":
            if args.output.exists() and any(args.output.iterdir()):
                raise SystemExit(f"refusing to overwrite non-empty {args.output}")
            store, stats = backfill_abstracts(
                CorpusStore.load(args.data), OpenAlexClient(args.cache, email=args.email)
            )
            report_path = args.data / "ingestion_report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            report["abstract_backfill"] = stats
            manifest = write_artifacts(
                store, args.output, f"openalex-abstracts:{args.data}", ingestion_report=report
            )
            print(json.dumps({"backfill": stats, "manifest": manifest}, indent=2))
            return 0
        report = audit_store(CorpusStore.load(args.data))
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 2

    if args.command == "run":
        store = CorpusStore.load(args.data)
        review = store.review(args.review)
        retriever = (
            BM25Retriever(store.papers.values())
            if args.retriever == "bm25"
            else Specter2Retriever(store.papers.values())
        )
        spec = EpisodeSpec(
            review_id=review.id,
            question=review.question,
            eligibility_criteria=review.eligibility_criteria,
            budget=args.budget,
            mode=args.mode,
        )
        result = run_episode(
            LiteratureSearchEnv(store, retriever, spec),
            _agent(args.agent, args.model, args.seed),
        )
        write_run(result, args.output)
        print(json.dumps(result["metrics"], indent=2))
        return 0

    if args.command == "evaluate":
        paths = sorted(
            path
            for path in args.run_dir.rglob("*.json")
            if not path.name.startswith(("summary", "manifest", "splits"))
        )
        summary = evaluate_runs(paths)
        rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0

    if args.command == "report":
        from .report import build_report

        rendered = build_report(args.run_dir)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0

    if args.command == "matrix":
        from .matrix import matrix_cells, run_matrix

        if args.count:
            _, _, cells = matrix_cells(args.config)
            print(len(cells))
            return 0
        outputs = run_matrix(args.config, index=args.index)
        print(json.dumps({"completed": len(outputs), "outputs": [str(p) for p in outputs]}, indent=2))
        return 0

    if args.command == "demo":
        path = write_html_replay(args.trace, args.output)
        print(path)
        return 0

    if args.command in {"train-sft", "train-grpo"}:
        from .training import run_training

        return run_training(args.command.removeprefix("train-"), args.config)
    return 1


if __name__ == "__main__":
    sys.exit(main())

