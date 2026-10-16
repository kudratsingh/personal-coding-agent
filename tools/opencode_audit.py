"""Explicit prefix-only source pilot; never exports or admits training tasks."""

import argparse
import json
import resource
import sys
from dataclasses import asdict
from pathlib import Path

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.arc import arc_splits, read_arc_easy
from scratchcoder.data.evalplus import read_release, release
from scratchcoder.data.manifest import download
from scratchcoder.data.opencode import audit_opencode, opencode_sample, read_opencode_sample
from scratchcoder.sandbox.runner import Limits, Runtime, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file")
    specification = opencode_sample()
    benchmarks = []
    manifests = {}
    for name in ("humaneval-plus", "mbpp-plus"):
        spec = release(name)
        path = args.data_directory / f"{name}.jsonl.gz"
        if args.download:
            download(spec.manifest, path)
        benchmarks.extend(read_release(path, spec))
        manifests[name] = asdict(spec.manifest)
    if args.download:
        download(specification.manifest, args.data_directory / specification.filename)
        for arc in arc_splits().values():
            download(arc.manifest, args.data_directory / "arc" / arc.filename)
    choices, choice_metadata = read_arc_easy(args.data_directory / "arc")
    rows = read_opencode_sample(args.data_directory / specification.filename, specification)
    runtime, limits = Runtime(), Limits()
    commit, dirty = git_state()
    report = audit_opencode(
        rows,
        specification,
        benchmarks,
        choices,
        execute=lambda code, tests, mode: run(code, tests, limits, mode=mode, runtime=runtime),
    )
    report.update(
        schema_version=1,
        created_at=utc_now(),
        git_commit=commit,
        dirty=dirty,
        benchmark_manifests=manifests,
        choice_source=choice_metadata,
        runtime=asdict(runtime),
        limits=asdict(limits),
        driver_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    )
    atomic_json(args.output, report)
    print("OPENCODE_AUDIT_REPORT=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
