"""Explicit pinned release validation on a deliberately built disposable runtime."""

import argparse
import json
from pathlib import Path

from scratchcoder.data.evalplus import read_release, release
from scratchcoder.data.manifest import download
from scratchcoder.eval.harness import EvaluationConfig
from scratchcoder.eval.suites import evaluate_suite
from scratchcoder.sandbox.runner import Limits, Runtime, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["humaneval-plus", "mbpp-plus"], required=True)
    parser.add_argument("--release-path", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--memory-mb", type=int, default=4096)
    parser.add_argument(
        "--exclude-find-zero",
        action="store_true",
        help="Score 163 HumanEval+ tasks, excluding task 32: known numerical-tolerance failure",
    )
    parser.add_argument(
        "--correct-find-zero-oracle",
        action="store_true",
        help="Explicitly opt into the versioned, fingerprint-bound HumanEval/32 correction",
    )
    args = parser.parse_args()
    if args.correct_find_zero_oracle and args.suite != "humaneval-plus":
        parser.error("find-zero correction is only available for HumanEval+")
    if args.exclude_find_zero and args.suite != "humaneval-plus":
        parser.error("task-32 exclusion is only available for HumanEval+")
    if args.exclude_find_zero and args.correct_find_zero_oracle:
        parser.error("task-32 exclusion cannot be combined with an oracle correction")
    if args.output.exists():
        parser.error("output must be a new directory")
    specification = release(args.suite)
    runtime = Runtime(args.image_id, True)
    limits = Limits(timeout_seconds=60, memory_mb=args.memory_mb, input_bytes=2 * 1024 * 1024)
    if args.download:
        download(specification.manifest, args.release_path)
    tasks = read_release(args.release_path, specification)
    try:
        reports = evaluate_suite(
            tasks,
            specification,
            execute=lambda code, tests, mode: run(code, tests, limits, mode=mode, runtime=runtime),
            runtime=runtime,
            limits=limits,
            output=args.output,
            checkpoint=args.checkpoint,
            tokenizer_path=args.tokenizer,
            config=EvaluationConfig(max_new_tokens=args.max_new_tokens),
            correct_find_zero_oracle=args.correct_find_zero_oracle,
            exclude_find_zero=args.exclude_find_zero,
        )
    finally:
        # Preserve safe completed records even when canonical correctness blocks model scoring.
        for name in ("canonical", "checkpoint"):
            path = args.output / name / "report.json"
            if path.exists():
                print(
                    name.upper()
                    + "_REPORT="
                    + json.dumps(json.loads(path.read_text()), sort_keys=True),
                    flush=True,
                )
    return 0 if reports else 1


if __name__ == "__main__":
    raise SystemExit(main())
