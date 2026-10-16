"""Explicit real-source MBPP reference audit using disposable isolation only."""

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
from scratchcoder.data.mbpp import mbpp_release, read_mbpp
from scratchcoder.data.mbpp_audit import audit_mbpp
from scratchcoder.sandbox.runner import Limits, Runtime, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file")
    specification = mbpp_release()
    benchmarks = []
    benchmark_manifests = {}
    for name in ("humaneval-plus", "mbpp-plus"):
        spec = release(name)
        path = args.data_directory / f"{name}.jsonl.gz"
        if args.download:
            download(spec.manifest, path)
        benchmarks.extend(read_release(path, spec))
        benchmark_manifests[name] = asdict(spec.manifest)
    if args.download:
        download(specification.manifest, args.data_directory / "mbpp.jsonl")
        for spec_arc in arc_splits().values():
            download(spec_arc.manifest, args.data_directory / "arc" / spec_arc.filename)
    choices, choice_metadata = read_arc_easy(args.data_directory / "arc")
    rows = read_mbpp(args.data_directory / "mbpp.jsonl", specification)
    runtime = Runtime()
    limits = Limits()
    commit, dirty = git_state()
    report = audit_mbpp(
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
        benchmark_manifests=benchmark_manifests,
        choice_source=choice_metadata,
        runtime=asdict(runtime),
        limits=asdict(limits),
        driver_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    )
    atomic_json(args.output, report)
    print("MBPP_AUDIT_REPORT=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
