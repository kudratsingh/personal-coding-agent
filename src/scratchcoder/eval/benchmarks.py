"""Score public-benchmark completions (HumanEval, HumanEval+, MBPP+) where they run.

The benchmarks are public, so they are scored on a disposable GitHub Actions runner rather
than on the Mac (``.github/workflows/benchmark-score.yml`` runs ``tools/benchmark_score.py``,
which calls :func:`score_benchmark`). Each suite runs through the project's existing path
for it:

- HumanEval (164): the reference gate's protocol of record, the pinned stdlib image with
  the runner's default limits.
- HumanEval+ (163, task 32 excluded) and MBPP+ (378): the EvalPlus path, the locally built
  NumPy image with a 60 s budget, 4 GiB memory and a 2 MiB input cap.

Before any completion is interpreted, every canonical solution must pass (the canonical
preflight); a failure blocks model scoring. Each attempt's program is the one stored in the
completions file (the protocol's extraction, computed when it was generated); for the base
protocol it is also re-derived from the completion text and must agree. Host failures are
retried and never scored. Every task's constant-answer programs
(:func:`scratchcoder.eval.baseline.benchmark_constant_programs`) are scored the same way, so
the suite's constant baseline sits beside the model's score.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

from scratchcoder.common.records import git_state, utc_now
from scratchcoder.data.evalplus import release
from scratchcoder.data.humaneval import manifest as humaneval_manifest
from scratchcoder.data.manifest import ManifestEntry
from scratchcoder.eval.baseline import DEFINITION, benchmark_constant_programs
from scratchcoder.eval.completions import Completions
from scratchcoder.eval.executor import CONTEXT_LIMIT, NO_PROGRAM, program_sha256
from scratchcoder.eval.harness import HOST_RETRIES
from scratchcoder.eval.parallel import Call, Execute, Outcome, run_calls
from scratchcoder.eval.protocols import BASE_BENCHMARK, BENCHMARK_SUITES, HUMANEVAL, protocol_for
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.reward import audit_attempt
from scratchcoder.sandbox.runner import Limits, Runtime

RESULTS_KIND = "benchmark-results"
SCHEMA_VERSION = 1
EVALPLUS_LIMITS = Limits(timeout_seconds=60, memory_mb=4096, input_bytes=2 * 1024 * 1024)


def artifact_name(completions_sha256: str) -> str:
    """The Actions artifact that holds the results for one completions file."""
    return f"benchmark-results-{completions_sha256}"


def results_filename(suite: str, completions_sha256: str) -> str:
    return f"benchmark-{suite}-{completions_sha256}.json"


def release_manifest(suite: str) -> ManifestEntry:
    """The pinned release a benchmark suite is read from."""
    if suite == HUMANEVAL:
        return humaneval_manifest()
    if suite not in BENCHMARK_SUITES:
        raise ValueError("not a benchmark suite")
    return release(suite).manifest


def benchmark_execution(suite: str, image_id: str | None = None) -> tuple[Runtime, Limits]:
    """The runtime and limits a suite is scored under (see the module docstring)."""
    if suite == HUMANEVAL:
        if image_id not in (None, "default"):
            raise ValueError("HumanEval runs on the pinned default image")
        return Runtime(), Limits()
    if suite not in BENCHMARK_SUITES:
        raise ValueError("not a benchmark suite")
    if image_id is None or image_id == "default":
        raise ValueError("HumanEval+ and MBPP+ need the locally built EvalPlus image ID")
    return Runtime(image_id, site_packages=True), EVALPLUS_LIMITS


def _candidate(completions: Completions, task: Task, row: Any) -> str | None:
    header = completions.header
    if row.stop_reason == CONTEXT_LIMIT:
        if row.program is not None:
            raise ValueError(f"{row.key}: a context-limited attempt has no program")
        return None
    if header.protocol == BASE_BENCHMARK:
        derived = protocol_for(header.protocol, header.suite).candidate(
            task, tokens=row.tokens, completion=row.completion
        )
        if derived != row.program:
            raise ValueError(
                f"{row.key}: the stored program differs from the protocol's extraction"
            )
    program: str | None = row.program
    return program


def score_benchmark(
    completions: Completions,
    tasks: Sequence[Task],
    *,
    execute: Execute,
    workers: int = 4,
    retries: int = HOST_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
    constant_baseline: bool = True,
) -> dict[str, Any]:
    """Canonical preflight, then the completions, then the constant answers."""
    header = completions.header
    if header.suite not in BENCHMARK_SUITES:
        raise ValueError("only HumanEval, HumanEval+ and MBPP+ completions are scored here")
    by_id = {task.id: task for task in tasks}
    if sorted(by_id) != completions.task_ids:
        raise ValueError("the tasks differ from the completions' tasks")
    started = time.monotonic()
    seconds = 0.0
    commit, dirty = git_state()
    report: dict[str, Any] = {
        "kind": RESULTS_KIND,
        "schema_version": SCHEMA_VERSION,
        "suite": header.suite,
        "completions_sha256": completions.sha256,
        "completions": header.to_json(),
        "excluded_tasks": [dict(item) for item in header.excluded_tasks],
        "release_sha256": header.task_file_sha256,
        "host_retries": retries,
        "git_commit": commit,
        "dirty": dirty,
        "created_at": utc_now(),
    }
    canonical: dict[str, Outcome] = {}

    def keep_canonical(outcome: Outcome) -> None:
        nonlocal seconds
        seconds += outcome.elapsed_seconds
        canonical[str(outcome.key)] = outcome

    run_calls(
        [
            Call(task.id, task.solution or "", task.hidden_tests, task.mode)
            for task in sorted(tasks, key=lambda item: item.id)
        ],
        execute,
        on_outcome=keep_canonical,
        workers=workers,
        retries=retries,
        sleep=sleep,
    )
    failed = sorted(task_id for task_id, outcome in canonical.items() if not outcome.passed)
    report["canonical"] = {
        "tasks": len(tasks),
        "passed": len(tasks) - len(failed),
        "failed_tasks": failed,
        "unresolved_tasks": sorted(
            key for key, outcome in canonical.items() if not outcome.resolved
        ),
    }
    if failed:
        report.update(status="canonical_failed", sandbox_seconds=seconds)
        report["wall_seconds"] = time.monotonic() - started
        return report
    programs = {
        row.key: _candidate(completions, by_id[row.task_id], row) for row in completions.rows
    }
    attempts: dict[tuple[str, int], dict[str, Any]] = {}

    def entry(key: tuple[str, int], status: str, outcome: Outcome | None) -> dict[str, Any]:
        row = completions.row(*key)
        program = programs[key]
        success = status == "passed"
        return {
            "task_id": row.task_id,
            "sample": row.sample,
            "status": status,
            "passed": success,
            "resolved": outcome is None or outcome.resolved,
            "stop_reason": row.stop_reason,
            "program_sha256": None if program is None else program_sha256(program),
            "audit_flags": list(audit_attempt(program)) if success and program else None,
            "host_failures": 0 if outcome is None else outcome.host_failures,
            "elapsed_seconds": 0.0 if outcome is None else outcome.elapsed_seconds,
        }

    calls = []
    for row in completions.rows:
        program = programs[row.key]
        if program is None:
            status = CONTEXT_LIMIT if row.stop_reason == CONTEXT_LIMIT else NO_PROGRAM
            attempts[row.key] = entry(row.key, status, None)
            continue
        task = by_id[row.task_id]
        calls.append(Call(row.key, program, task.hidden_tests, task.mode))

    def keep_attempt(outcome: Outcome) -> None:
        nonlocal seconds
        seconds += outcome.elapsed_seconds
        key = outcome.key
        assert isinstance(key, tuple)
        status = outcome.status if outcome.resolved else "unresolved"
        attempts[key] = entry(key, status, outcome)

    run_calls(
        calls, execute, on_outcome=keep_attempt, workers=workers, retries=retries, sleep=sleep
    )
    ordered = [attempts[row.key] for row in completions.rows]
    resolved = [item for item in ordered if item["resolved"]]
    report["attempts"] = ordered
    report["summary"] = {
        "tasks": len(tasks),
        "attempts": len(ordered),
        "resolved_attempts": len(resolved),
        "unresolved_attempts": len(ordered) - len(resolved),
        "passed_attempts": sum(item["passed"] for item in ordered),
        "audit_flagged_passes": sum(bool(item["audit_flags"]) for item in ordered),
    }
    if constant_baseline:
        report["constant_baseline"], extra = _constant_baseline(
            tasks, header.suite, execute=execute, workers=workers, retries=retries, sleep=sleep
        )
        seconds += extra
    report.update(status="completed", sandbox_seconds=seconds)
    report["wall_seconds"] = time.monotonic() - started
    return report


def _constant_baseline(
    tasks: Sequence[Task],
    suite: str,
    *,
    execute: Execute,
    workers: int,
    retries: int,
    sleep: Callable[[float], None],
) -> tuple[dict[str, Any], float]:
    """Per task, whether any constant answer passes the suite's tests (pass/fail scoring)."""
    programs = {task.id: benchmark_constant_programs(task, suite) for task in tasks}
    outcomes: dict[tuple[str, int], Outcome] = {}
    seconds = 0.0

    def keep(outcome: Outcome) -> None:
        nonlocal seconds
        seconds += outcome.elapsed_seconds
        key = outcome.key
        assert isinstance(key, tuple)
        outcomes[key] = outcome

    by_id = {task.id: task for task in tasks}
    run_calls(
        [
            Call((task_id, index), code, by_id[task_id].hidden_tests, by_id[task_id].mode)
            for task_id, entries in sorted(programs.items())
            for index, (_, code) in enumerate(entries)
        ],
        execute,
        on_outcome=keep,
        workers=workers,
        retries=retries,
        sleep=sleep,
    )
    rows: list[dict[str, Any]] = []
    for task_id, entries in sorted(programs.items()):
        results = [outcomes[(task_id, index)] for index in range(len(entries))]
        passing = [
            label for (label, _), outcome in zip(entries, results, strict=True) if outcome.passed
        ]
        if not entries:
            rows.append(
                {"task_id": task_id, "passed": None, "answer": "none", "reason": "no entry point"}
            )
        elif passing:
            rows.append({"task_id": task_id, "passed": True, "answer": passing[0], "reason": ""})
        elif any(not outcome.resolved for outcome in results):
            reason = "a constant-answer call stayed unresolved"
            rows.append({"task_id": task_id, "passed": None, "answer": "none", "reason": reason})
        else:
            rows.append({"task_id": task_id, "passed": False, "answer": "none", "reason": ""})
    covered = [row for row in rows if row["passed"] is not None]
    share = sum(row["passed"] for row in covered) / len(covered) if covered else None
    return (
        {
            "definition": DEFINITION,
            "scoring": "pass/fail per task: the best constant answer passes or not",
            "tasks": len(rows),
            "covered": len(covered),
            "uncovered": len(rows) - len(covered),
            "mean_reward": share,
            "full_pass_share": share,
            "executed_calls": sum(len(entries) for entries in programs.values()),
            "per_task": rows,
        },
        seconds,
    )


def execution_record(runtime: Runtime, limits: Limits) -> dict[str, Any]:
    return {"runtime": asdict(runtime), "limits": asdict(limits)}
