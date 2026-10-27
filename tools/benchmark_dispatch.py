"""Score one public-benchmark completions file on GitHub Actions, from the Mac.

The completions file (HumanEval, HumanEval+ or MBPP+ only) must already be in R2 under
``--key``, uploaded with its SHA-256 as object metadata. This tool:

1. validates the local file as public-benchmark completions for ``--suite`` and checks
   that the R2 object has the same size and SHA-256;
2. creates a presigned GET link for exactly that one object, expiring in at most an hour;
3. dispatches the ``benchmark-score`` workflow with ``gh``, passing the link, the SHA-256,
   the suite and a fresh request ID, then finds the run by its name;
4. waits for the run, downloads the ``benchmark-results-<sha256>`` artifact and checks that
   the results name the same completions SHA-256, suite and request ID.

Run it as a module from the repository root (it imports its sibling ``tools.r2_store``)::

    uv run --locked python -m tools.benchmark_dispatch --completions he.jsonl \\
        --suite humaneval --key evaluation/he.jsonl --repo owner/name --output runs/he-results

The link appears in the public repository's run metadata, which is why it names one
object, expires quickly and may only point at public-benchmark completions. R2 credentials
stay on the Mac: they sign the link here and are removed from ``gh``'s environment. Pool
completions are refused: their hidden tests never leave the Mac.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from scratchcoder.eval.benchmarks import RESULTS_KIND, artifact_name, results_filename
from scratchcoder.eval.completions import Completions, read_completions
from scratchcoder.eval.protocols import BENCHMARK_SUITES
from tools.r2_store import Credentials, bounded_store, presigned_get

WORKFLOW = "benchmark-score.yml"
MAX_EXPIRES_SECONDS = 3600
"""The longest a dispatched link may stay valid; the workflow downloads it first."""
HIDDEN_ENVIRONMENT_PREFIXES = ("R2_", "AWS_", "RUNPOD_")


class Runner(Protocol):
    def __call__(
        self, arguments: Sequence[str], environment: Mapping[str, str]
    ) -> subprocess.CompletedProcess[str]: ...


class ObjectStore(Protocol):
    def head(self, key: str) -> dict[str, str] | None: ...


def _run(
    arguments: Sequence[str], environment: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments), env=dict(environment), capture_output=True, text=True, check=True
    )


def gh_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` without R2, AWS and Runpod settings or anything named like a secret."""
    return {
        name: value
        for name, value in environ.items()
        if not name.startswith(HIDDEN_ENVIRONMENT_PREFIXES) and "SECRET" not in name.upper()
    }


def check_public_completions(path: Path, suite: str) -> Completions:
    if suite not in BENCHMARK_SUITES:
        raise ValueError("only HumanEval, HumanEval+ and MBPP+ completions may be dispatched")
    completions = read_completions(path)
    header = completions.header
    if header.suite != suite or header.split != "eval":
        raise ValueError("the file is not public-benchmark completions for this suite")
    return completions


def check_object(store: ObjectStore, key: str, *, sha256: str, size: int) -> None:
    """The R2 object must exist with the local file's size and SHA-256 metadata."""
    headers = store.head(key)
    if headers is None:
        raise ValueError(f"{key} is not in R2; upload it first")
    if headers.get("content-length") != str(size):
        raise ValueError(f"{key}: the R2 object's size differs from the local file")
    if headers.get("x-amz-meta-sha256") != sha256:
        raise ValueError(f"{key}: the R2 object's SHA-256 metadata differs from the local file")


def _json(runner: Runner, arguments: Sequence[str], environment: Mapping[str, str]) -> Any:
    return json.loads(runner(arguments, environment).stdout or "null")


def find_run(
    runner: Runner,
    environment: Mapping[str, str],
    *,
    repo: str,
    workflow: str,
    title: str,
    timeout_seconds: float,
    poll_seconds: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> int:
    deadline = clock() + timeout_seconds
    while True:
        runs = _json(
            runner,
            [
                "gh",
                "run",
                "list",
                "--repo",
                repo,
                "--workflow",
                workflow,
                "--event",
                "workflow_dispatch",
                "--limit",
                "50",
                "--json",
                "databaseId,displayTitle,status,conclusion",
            ],
            environment,
        )
        matches = [run for run in runs or [] if run.get("displayTitle") == title]
        if len(matches) > 1:
            raise RuntimeError(f"more than one run is named {title!r}")
        if matches:
            return int(matches[0]["databaseId"])
        if clock() >= deadline:
            raise TimeoutError(f"no run named {title!r} appeared")
        sleep(poll_seconds)


def wait_for_run(
    runner: Runner,
    environment: Mapping[str, str],
    *,
    repo: str,
    run_id: int,
    timeout_seconds: float,
    poll_seconds: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> str:
    deadline = clock() + timeout_seconds
    while True:
        view = _json(
            runner,
            ["gh", "run", "view", str(run_id), "--repo", repo, "--json", "status,conclusion"],
            environment,
        )
        if view.get("status") == "completed":
            return str(view.get("conclusion"))
        if clock() >= deadline:
            raise TimeoutError(f"run {run_id} did not finish in time")
        sleep(poll_seconds)


def dispatch(
    *,
    completions_path: Path,
    suite: str,
    key: str,
    repo: str,
    output: Path,
    credentials: Credentials,
    store: ObjectStore,
    ref: str = "main",
    workflow: str = WORKFLOW,
    expires_seconds: int = MAX_EXPIRES_SECONDS,
    runner: Runner = _run,
    environ: Mapping[str, str] = os.environ,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    poll_seconds: float = 15.0,
    find_timeout_seconds: float = 300.0,
    run_timeout_seconds: float = 160 * 60.0,
) -> dict[str, Any]:
    """Dispatch, wait, download and verify; return a summary (the link is never returned)."""
    if type(expires_seconds) is not int or not 1 <= expires_seconds <= MAX_EXPIRES_SECONDS:
        raise ValueError("the link must expire within 3600 seconds")
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    completions = check_public_completions(completions_path, suite)
    digest = completions.sha256
    check_object(store, key, sha256=digest, size=completions_path.stat().st_size)
    signed_at = now()
    url = presigned_get(credentials, key, expires_seconds=expires_seconds, now=signed_at)
    request_id = uuid4().hex
    environment = gh_environment(environ)
    runner(
        [
            "gh",
            "workflow",
            "run",
            workflow,
            "--repo",
            repo,
            "--ref",
            ref,
            "-f",
            f"suite={suite}",
            "-f",
            f"completions_url={url}",
            "-f",
            f"completions_sha256={digest}",
            "-f",
            f"request_id={request_id}",
        ],
        environment,
    )
    title = f"Benchmark {suite} {request_id}"
    run_id = find_run(
        runner,
        environment,
        repo=repo,
        workflow=workflow,
        title=title,
        timeout_seconds=find_timeout_seconds,
        poll_seconds=poll_seconds,
        sleep=sleep,
        clock=clock,
    )
    conclusion = wait_for_run(
        runner,
        environment,
        repo=repo,
        run_id=run_id,
        timeout_seconds=run_timeout_seconds,
        poll_seconds=poll_seconds,
        sleep=sleep,
        clock=clock,
    )
    output.mkdir(parents=True)
    runner(
        [
            "gh",
            "run",
            "download",
            str(run_id),
            "--repo",
            repo,
            "--name",
            artifact_name(digest),
            "--dir",
            str(output),
        ],
        environment,
    )
    path = output / results_filename(suite, digest)
    if not path.is_file():
        raise RuntimeError(f"run {run_id} ({conclusion}) uploaded no {path.name}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("kind") != RESULTS_KIND
        or report.get("completions_sha256") != digest
        or report.get("suite") != suite
        or report.get("request_id") != request_id
    ):
        raise RuntimeError("the downloaded results do not name this completions file and request")
    if conclusion != "success" or report.get("status") != "completed":
        raise RuntimeError(f"run {run_id} ended {conclusion}: {report.get('status')}")
    return {
        "suite": suite,
        "completions_sha256": digest,
        "request_id": request_id,
        "run_id": run_id,
        "conclusion": conclusion,
        "results": str(path),
        "link_expires_at": (signed_at + dt.timedelta(seconds=expires_seconds)).isoformat(),
        "summary": report.get("summary"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.benchmark_dispatch", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--suite", choices=BENCHMARK_SUITES, required=True)
    parser.add_argument("--key", required=True, help="the completions file's R2 object key")
    parser.add_argument("--repo", required=True, help="owner/name of the public repository")
    parser.add_argument("--ref", default="main")
    parser.add_argument("--output", type=Path, required=True, help="a new directory")
    parser.add_argument("--expires-seconds", type=int, default=MAX_EXPIRES_SECONDS)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)
    if not 1 <= args.expires_seconds <= MAX_EXPIRES_SECONDS:
        parser.error("--expires-seconds must be from 1 to 3600")
    credentials = Credentials.from_environment()
    summary = dispatch(
        completions_path=args.completions,
        suite=args.suite,
        key=args.key,
        repo=args.repo,
        ref=args.ref,
        output=args.output,
        credentials=credentials,
        store=bounded_store(credentials),
        expires_seconds=args.expires_seconds,
        poll_seconds=args.poll_seconds,
    )
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
