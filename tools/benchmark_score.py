"""Score one public-benchmark completions file on a disposable runner (GitHub Actions).

The ``benchmark-score`` workflow downloads the completions file through a short-lived
link, checks its SHA-256, and runs this tool. It refuses a completions file whose SHA-256
differs from ``--sha256`` or whose suite is not a public benchmark, downloads the pinned
release, runs the canonical preflight, scores every attempt and every constant answer in
the sandbox, and writes ``benchmark-<suite>-<sha256>.json`` into ``--output``. It never
handles pool tasks: their hidden tests stay on the Mac.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from scratchcoder.common.records import atomic_json
from scratchcoder.data.manifest import download
from scratchcoder.eval.benchmarks import (
    benchmark_execution,
    execution_record,
    release_manifest,
    results_filename,
    score_benchmark,
)
from scratchcoder.eval.completions import read_completions, validate_against_tasks
from scratchcoder.eval.protocols import BENCHMARK_SUITES
from scratchcoder.sandbox.runner import Result, run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=BENCHMARK_SUITES, required=True)
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="the completions file's SHA-256")
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--release-path", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="fetch the pinned release")
    parser.add_argument("--image-id", default="default", help="EvalPlus image ID, or default")
    parser.add_argument("--output", type=Path, required=True, help="a new directory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-constant-baseline", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must be a new directory")
    completions = read_completions(args.completions, sha256=args.sha256)
    if completions.header.suite != args.suite or completions.header.split != "eval":
        parser.error("the completions file is not for this public benchmark suite")
    if args.download:
        download(release_manifest(args.suite), args.release_path)
    tasks = validate_against_tasks(completions, args.release_path)
    runtime, limits = benchmark_execution(args.suite, args.image_id)

    def execute(code: str, tests: str, mode: str) -> Result:
        return run(code, tests, limits, mode=mode, runtime=runtime)

    report = score_benchmark(
        completions,
        tasks,
        execute=execute,
        workers=args.workers,
        constant_baseline=not args.no_constant_baseline,
    )
    report["request_id"] = args.request_id
    report["execution"] = execution_record(runtime, limits)
    args.output.mkdir(parents=True)
    path = args.output / results_filename(args.suite, completions.sha256)
    atomic_json(path, report)
    summary = {
        key: report.get(key)
        for key in ("suite", "status", "completions_sha256", "request_id", "canonical", "summary")
    }
    baseline = report.get("constant_baseline")
    if baseline is not None:
        summary["constant_baseline"] = {
            key: baseline[key] for key in ("full_pass_share", "covered", "uncovered")
        }
    print("BENCHMARK_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
