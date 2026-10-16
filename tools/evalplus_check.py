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
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    specification = release(args.suite)
    runtime = Runtime(args.image_id, True)
    limits = Limits(timeout_seconds=60, memory_mb=1024, input_bytes=2 * 1024 * 1024)
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
