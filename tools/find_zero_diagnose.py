"""Inspect the pinned task-32 canonical failure inside the deliberate sandbox only."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.evalplus import read_release, release
from scratchcoder.data.manifest import download
from scratchcoder.eval.find_zero_diagnostics import diagnose_find_zero
from scratchcoder.sandbox.runner import Limits, Runtime, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-path", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file")
    spec = release("humaneval-plus")
    runtime = Runtime(args.image_id, True)
    limits = Limits(timeout_seconds=60, memory_mb=4096, input_bytes=2 * 1024 * 1024)
    if args.download:
        download(spec.manifest, args.release_path)
    tasks = read_release(args.release_path, spec)
    task = next(task for task in tasks if task.id == "humaneval/32")
    commit, dirty = git_state()
    report = diagnose_find_zero(
        task,
        spec,
        execute=lambda code, tests, mode: run(code, tests, limits, mode=mode, runtime=runtime),
    )
    report.update(
        schema_version=1,
        created_at=utc_now(),
        git_commit=commit,
        dirty=dirty,
        runtime=asdict(runtime),
        limits=asdict(limits),
    )
    atomic_json(args.output, report)
    print("FIND_ZERO_REPORT=" + json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
