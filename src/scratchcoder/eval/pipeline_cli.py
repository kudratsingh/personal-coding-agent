"""Command-line entry points of the evaluation pipeline (generate, execute, seal, score)."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

COMMANDS = ("eval-generate", "eval-baseline")


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
    baseline = subparsers.add_parser(
        "eval-baseline",
        help="constant-answer baseline of the pool dev split (the test split is sealed)",
    )
    baseline.add_argument("--tasks", type=Path, required=True, help="pool dev JSONL")
    baseline.add_argument("--output", type=Path, required=True, help="new JSON file")
    baseline.add_argument("--workers", type=int, default=8)


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
    if args.command == "eval-baseline":
        from scratchcoder.eval.baseline import pool_baseline_report, save_baseline

        if args.output.exists():
            parser.error("--output must be a new file")
        report = pool_baseline_report(args.tasks, split="dev", workers=args.workers)
        save_baseline(args.output, report)
        _print({"baseline": str(args.output), **report["summary"]})
        return 0
    raise ValueError(f"unknown pipeline command {args.command}")
