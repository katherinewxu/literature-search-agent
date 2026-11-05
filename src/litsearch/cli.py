from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents import RandomAgent, RankedRetrievalAgent, SnowballAgent
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
from .retrieval import BM25Retriever, DenseRetriever, build_specter2_index
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
    build.add_argument(
        "--split", help="assign every review to this one split (e.g. train, transfer)"
    )
    audit = data_commands.add_parser("audit")
    audit.add_argument("--data", type=Path, required=True)
    enrich = data_commands.add_parser("enrich")
    enrich.add_argument("--data", type=Path, required=True)
    enrich.add_argument("--output", type=Path, required=True)
    enrich.add_argument("--cache", type=Path, required=True)
    enrich.add_argument("--email")
    enrich.add_argument("--max-seed-papers", type=int)
    open_graph = data_commands.add_parser("open-graph", help="add bounded out-of-pool neighbors")
    open_graph.add_argument("--data", type=Path, required=True)
    open_graph.add_argument("--output", type=Path, required=True)
    open_graph.add_argument("--cache", type=Path, required=True)
    open_graph.add_argument("--list-missing", type=Path, help="offline: write uncached request URLs here and exit")
    open_graph.add_argument("--allow-partial", action="store_true", help="write output despite failed batches")
    open_graph.add_argument("--cap-per-review", type=int, default=5000)
    open_graph.add_argument("--pool-multiple", type=float, default=1.0)
    fetch = data_commands.add_parser("fetch-urls", help="fill an OpenAlex cache from a URL list")
    fetch.add_argument("--urls", type=Path, required=True)
    fetch.add_argument("--cache", type=Path, required=True)
    backfill = data_commands.add_parser("backfill-abstracts")
    backfill.add_argument("--data", type=Path, required=True)
    backfill.add_argument("--output", type=Path, required=True)
    backfill.add_argument("--cache", type=Path, required=True)
    backfill.add_argument("--list-missing", type=Path, help="offline: write uncached request URLs here and exit")
    backfill.add_argument("--allow-partial", action="store_true", help="write output despite failed batches")
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
    run.add_argument("--dense-index", type=Path, help="SPECTER2 index prefix (with --retriever specter2)")
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
    matrix.add_argument("--workers", type=int, default=1, help="concurrent episodes (LLM runs)")
    matrix.add_argument("--shard", help="I/N: run every N-th cell starting at I")
    matrix.add_argument("--skip-existing", action="store_true")

    index = commands.add_parser("index", help="encode a store with SPECTER2")
    index.add_argument("--data", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True, help="prefix; writes .npy and .ids.json")

    demo = commands.add_parser("demo")
    demo_commands = demo.add_subparsers(dest="demo_command", required=True)
    demo_build = demo_commands.add_parser("build")
    demo_build.add_argument("--trace", type=Path, required=True)
    demo_build.add_argument("--output", type=Path, required=True)
    demo_build.add_argument("--data", type=Path, help="frozen store: adds titles and gold markers")
    demo_build.add_argument("--estimates", type=Path, help="recall_estimates.jsonl: adds the agent's recall estimate")
    demo_build.add_argument("--openalex-cache", type=Path, help="titles for live-web works outside the store")
    demo_build.add_argument("--label")

    train_sft = commands.add_parser("train-sft")
    train_sft.add_argument("--config", type=Path, required=True)
    train_grpo = commands.add_parser("train-grpo")
    train_grpo.add_argument("--config", type=Path, required=True)
    collect = commands.add_parser("collect-sft", help="rejection-sample teacher turns for SFT")
    collect.add_argument("--run-dir", type=Path, action="append", required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--keep-fraction", type=float, default=0.5)

    analyze = commands.add_parser("analyze", help="curves, exposure, reachability, strategy, misses")
    analyze.add_argument("--data", type=Path, required=True)
    analyze.add_argument("--run-dir", type=Path, action="append", required=True)
    analyze.add_argument("--open-data", type=Path, help="open-graph store for reachability")
    analyze.add_argument("--output", type=Path, required=True)

    probe = commands.add_parser("probe-memory", help="closed-book recall of gold studies (no tools)")
    probe.add_argument("--config", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    return parser


def _agent(name: str, model: str, seed: int):
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "snowball":
        return SnowballAgent()
    if name == "retrieval":
        return RankedRetrievalAgent()
    from .llm import LLMSearchAgent, OpenAIChatBackend

    return LLMSearchAgent(OpenAIChatBackend(model), name="prompted")


def _stop_on_failures(args, client: OpenAlexClient, stats: dict) -> bool:
    """Offline listing or failed batches: report and write no (partial) artifact."""
    if args.list_missing is not None:
        args.list_missing.parent.mkdir(parents=True, exist_ok=True)
        args.list_missing.write_text("".join(url + "\n" for url in client.missing), encoding="utf-8")
        print(json.dumps({"missing_urls": len(client.missing), "written": str(args.list_missing)}))
        return True
    if stats.get("failed_batches") and not args.allow_partial:
        print(json.dumps({"error": "failed batches; output not written", **stats}, default=str))
        return True
    return False


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
            splits = {args.split: sorted(store.reviews)} if args.split else None
            manifest = write_artifacts(
                store, args.output, args.source, ingestion_report=report, splits=splits
            )
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
        if args.data_command == "fetch-urls":
            client = OpenAlexClient(args.cache)
            urls = [line.strip() for line in args.urls.read_text().splitlines() if line.strip()]
            fetched = 0
            for url in urls:
                if not url.startswith(client.base_url + "/"):
                    raise SystemExit(f"refusing non-OpenAlex URL: {url[:80]}")
                cached = client.cache_path(url).exists()
                client.fetch_url(url)
                fetched += not cached
            print(json.dumps({"urls": len(urls), "fetched": fetched}))
            return 0
        if args.data_command == "open-graph":
            from .open_graph import build_open_graph

            if args.output.exists() and any(args.output.iterdir()):
                raise SystemExit(f"refusing to overwrite non-empty {args.output}")
            client = OpenAlexClient(args.cache, offline=args.list_missing is not None)
            store, stats = build_open_graph(
                CorpusStore.load(args.data),
                client,
                cap_per_review=args.cap_per_review,
                pool_multiple=args.pool_multiple,
            )
            if _stop_on_failures(args, client, stats):
                return 3
            report_path = args.data / "ingestion_report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            report["open_graph"] = stats
            split_path = args.data / "splits.json"
            manifest = write_artifacts(
                store,
                args.output,
                f"open-graph:{args.data}",
                ingestion_report=report,
                splits=json.loads(split_path.read_text()) if split_path.exists() else None,
            )
            print(json.dumps({"open_graph": stats, "manifest": manifest}, indent=2))
            return 0
        if args.data_command == "backfill-abstracts":
            if args.output.exists() and any(args.output.iterdir()):
                raise SystemExit(f"refusing to overwrite non-empty {args.output}")
            client = OpenAlexClient(args.cache, email=args.email, offline=args.list_missing is not None)
            store, stats = backfill_abstracts(CorpusStore.load(args.data), client)
            if _stop_on_failures(args, client, stats):
                return 3
            report_path = args.data / "ingestion_report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            report["abstract_backfill"] = stats
            split_path = args.data / "splits.json"
            manifest = write_artifacts(
                store,
                args.output,
                f"openalex-abstracts:{args.data}",
                ingestion_report=report,
                splits=json.loads(split_path.read_text()) if split_path.exists() else None,
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
            else DenseRetriever(args.dense_index)
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
        shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
        outputs = run_matrix(
            args.config,
            index=args.index,
            workers=args.workers,
            shard=shard,
            skip_existing=args.skip_existing,
        )
        print(json.dumps({"completed": len(outputs), "outputs": [str(p) for p in outputs]}, indent=2))
        return 0

    if args.command == "index":
        if args.output.with_suffix(".npy").exists():
            raise SystemExit(f"refusing to overwrite {args.output}.npy")
        stats = build_specter2_index(CorpusStore.load(args.data).papers.values(), args.output)
        print(json.dumps(stats, indent=2))
        return 0

    if args.command == "demo":
        path = write_html_replay(
            args.trace, args.output, data_dir=args.data, estimates=args.estimates,
            label=args.label, openalex_cache=args.openalex_cache,
        )
        print(path)
        return 0

    if args.command == "collect-sft":
        from .rl import collect_sft

        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        print(json.dumps(collect_sft(args.run_dir, args.output, keep_fraction=args.keep_fraction), indent=2))
        return 0

    if args.command == "analyze":
        from .analysis_report import write_analysis

        print(write_analysis(args.data, args.run_dir, args.output, open_data=args.open_data))
        return 0

    if args.command == "probe-memory":
        from .memorization import run_probe

        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        run_probe(json.loads(args.config.read_text(encoding="utf-8")), args.output)
        return 0

    if args.command in {"train-sft", "train-grpo"}:
        from .training import run_training

        return run_training(args.command.removeprefix("train-"), args.config)
    return 1


if __name__ == "__main__":
    sys.exit(main())

