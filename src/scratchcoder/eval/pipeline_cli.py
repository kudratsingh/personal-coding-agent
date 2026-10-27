"""Command-line entry points of the evaluation pipeline (generate, execute, seal, score)."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

COMMANDS = (
    "eval-generate",
    "eval-execute",
    "eval-seal",
    "eval-baseline",
    "eval-test-baseline",
    "scorecard",
)


def _print(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def add_parsers(subparsers: Any) -> None:
    from scratchcoder.eval.protocols import PROTOCOLS, SUITES

    generate = subparsers.add_parser(
        "eval-generate",
        help="generate a hashed completions file for one suite (runs no generated code)",
    )
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--tokenizer", type=Path, required=True)
    generate.add_argument("--suite", choices=SUITES, required=True)
    generate.add_argument("--split", help="dev or test for the pool; eval (default) otherwise")
    generate.add_argument("--tasks", type=Path, required=True, help="pool JSONL or release file")
    generate.add_argument("--protocol", choices=PROTOCOLS, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--device", default="cpu", help="cpu (default), cuda, cuda:N or mps")
    generate.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    generate.add_argument("--batch-size", type=int, default=64)
    generate.add_argument("--samples", type=int, default=1)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--top-k", type=int)
    generate.add_argument("--seed", type=int, default=0)
    execute = subparsers.add_parser(
        "eval-execute",
        help="execute a pool completions file in the sandbox into a resumable results file",
    )
    execute.add_argument("--completions", type=Path, required=True)
    execute.add_argument("--tasks", type=Path, required=True, help="the pool task file")
    execute.add_argument("--results", type=Path, required=True, help="created, or continued")
    execute.add_argument(
        "--tokenizer", type=Path, help="required for chat-v1 completions and the test split"
    )
    execute.add_argument("--workers", type=int, default=8)
    execute.add_argument("--seal-ledger", type=Path, help="test split only")
    execute.add_argument("--unseal", type=Path, help="test split only: the unseal record")
    seal = subparsers.add_parser(
        "eval-seal", help="record pool test-split completions in the seal ledger"
    )
    seal.add_argument("--completions", type=Path, required=True)
    seal.add_argument("--ledger", type=Path, required=True)
    baseline = subparsers.add_parser(
        "eval-baseline",
        help="constant-answer baseline of the pool dev split (the test split is sealed)",
    )
    baseline.add_argument("--tasks", type=Path, required=True, help="pool dev JSONL")
    baseline.add_argument("--output", type=Path, required=True, help="new JSON file")
    baseline.add_argument("--workers", type=int, default=8)
    test_baseline = subparsers.add_parser(
        "eval-test-baseline",
        help="retry the logged test-split baseline after every unsealed read has finished",
    )
    test_baseline.add_argument("--tasks", type=Path, required=True, help="the pool test JSONL")
    test_baseline.add_argument("--seal-ledger", type=Path, required=True)
    test_baseline.add_argument("--unseal", type=Path, required=True, help="the unseal record")
    test_baseline.add_argument("--workers", type=int, default=8)
    score = subparsers.add_parser(
        "scorecard",
        help="build the scorecard from results files, benchmark artifacts and ARC reports",
    )
    score.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help="name a checkpoint, e.g. sft=<sha256>; repeat for each",
    )
    score.add_argument(
        "--inputs",
        type=Path,
        help="a scorecard-inputs file: the names, the reuses and every input, recorded once",
    )
    score.add_argument("--results", type=Path, action="append", default=[])
    score.add_argument("--benchmark", type=Path, action="append", default=[])
    score.add_argument("--arc", type=Path, action="append", default=[], help="eval-choice report")
    score.add_argument("--arc-tasks", type=Path, help="ARC-Easy choice JSONL or Parquet directory")
    score.add_argument("--pool-baseline", type=Path, action="append", default=[])
    score.add_argument(
        "--pair",
        action="append",
        metavar="FIRST:SECOND",
        help="compare two named checkpoints (default: sft:base, rl:sft, more-sft:sft, rl:more-sft)",
    )
    score.add_argument("--output", type=Path, required=True, help="new JSON file")
    score.add_argument("--markdown", type=Path, help="also write a Markdown summary")
    score.add_argument("--bootstrap-samples", type=int, default=2000)
    score.add_argument("--bootstrap-seed", type=int, default=0)


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.command == "eval-generate":
        from scratchcoder.eval.completions import Sampling
        from scratchcoder.eval.generate import generate_completions

        split = args.split or ("eval" if args.suite != "pool" else None)
        if split is None:
            parser.error("--split dev or --split test is required for the pool")
        header, digest = generate_completions(
            checkpoint=args.checkpoint,
            tokenizer_path=args.tokenizer,
            suite=args.suite,
            split=split,
            tasks_path=args.tasks,
            protocol=args.protocol,
            output=args.output,
            sampling=Sampling(
                samples=args.samples,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
            ),
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
        )
        _print(
            {
                "completions": str(args.output),
                "sha256": digest,
                "suite": header.suite,
                "split": header.split,
                "protocol": header.protocol,
                "protocol_sha256": header.protocol_sha256,
                "rows": header.rows,
                "generation_seconds": header.generation_seconds,
            }
        )
        return 0
    if args.command == "eval-execute":
        from scratchcoder.eval.executor import execute_completions
        from scratchcoder.tokenizer.bpe import ByteBPE

        summary = execute_completions(
            args.completions,
            args.tasks,
            args.results,
            tokenizer=None if args.tokenizer is None else ByteBPE.load(args.tokenizer),
            workers=args.workers,
            seal_ledger=args.seal_ledger,
            unseal=args.unseal,
        )
        _print(summary)
        return 0 if summary["unresolved"] == 0 and summary["pending"] == 0 else 3
    if args.command == "eval-seal":
        from scratchcoder.eval.seal import seal_completions

        _print(seal_completions(args.completions, args.ledger))
        return 0
    if args.command == "eval-baseline":
        from scratchcoder.eval.baseline import pool_baseline_report, save_baseline

        if args.output.exists():
            parser.error("--output must be a new file")
        report = pool_baseline_report(args.tasks, split="dev", workers=args.workers)
        save_baseline(args.output, report)
        _print({"baseline": str(args.output), **report["summary"]})
        return 0
    if args.command == "eval-test-baseline":
        from scratchcoder.eval.executor import run_test_baseline

        outcome = run_test_baseline(args.tasks, args.seal_ledger, args.unseal, workers=args.workers)
        _print(outcome)
        return 0 if outcome["recorded"] else 3
    if args.command == "scorecard":
        from scratchcoder.common.records import atomic_json
        from scratchcoder.eval.scorecard import build_scorecard, render_markdown

        for path in (args.output, args.markdown):
            if path is not None and path.exists():
                parser.error(f"{path} already exists")
        from scratchcoder.eval.scorecard import load_inputs

        checkpoints: dict[str, str] = {}
        reuses: dict[str, str] = {}
        reasons: dict[str, str] = {}
        results, benchmarks, arcs = list(args.results), list(args.benchmark), list(args.arc)
        if args.inputs is not None:
            recorded = load_inputs(args.inputs)
            checkpoints.update(recorded.checkpoints)
            reuses.update(recorded.reuses)
            reasons.update(recorded.reasons)
            results += recorded.results
            benchmarks += recorded.benchmarks
            arcs += recorded.arc_reports
        for item in args.checkpoint:
            name, _, digest = item.partition("=")
            if not name or not digest or name in checkpoints:
                parser.error("--checkpoint takes NAME=SHA256, each name once")
            checkpoints[name] = digest
        if not checkpoints:
            parser.error("name the checkpoints with --checkpoint or --inputs")
        pairs = None
        if args.pair:
            pairs = []
            for item in args.pair:
                first, _, second = item.partition(":")
                named = {*checkpoints, *reuses}
                if first not in named or second not in named:
                    parser.error("--pair takes FIRST:SECOND of named checkpoints")
                pairs.append((first, second))
        card = build_scorecard(
            checkpoints=checkpoints,
            results=results,
            benchmarks=benchmarks,
            arc_reports=arcs,
            arc_tasks=args.arc_tasks,
            pool_baselines=args.pool_baseline,
            pairs=pairs,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            reuses=reuses,
            reuse_reasons=reasons,
        )
        atomic_json(args.output, card)
        if args.markdown is not None:
            args.markdown.write_text(render_markdown(card) + "\n", encoding="utf-8")
        _print({"scorecard": str(args.output), "rows": len(card["rows"]), "leak": card["leak"]})
        return 0
    raise ValueError(f"unknown pipeline command {args.command}")
