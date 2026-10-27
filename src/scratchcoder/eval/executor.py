"""Execute a pool completions file on the Mac, in parallel, into a resumable results file.

Each attempt is scored by the evaluation runner's protocol of record
(:func:`scratchcoder.eval.harness.isolated_execution`: the extracted program with the
task's hidden tests, one fresh container per call) after the protocol's extraction.
chat-v1 takes the first closed python block of the generated token IDs; with none, the
attempt scores ``no_program`` without running. The base protocols cut the completion at
their stop strings. A context-limited attempt scores ``context_limit`` without running.

Host failures (Docker unavailable, a runner error, any result with ``host_stalled``) are
retried up to three times with bounded backoff and never scored. An attempt still failing
is written as ``unresolved``; a later run of the same results file retries it. Every
passing attempt also records the reward channel's static audit
(:func:`scratchcoder.sandbox.reward.audit_attempt`).

The results file is append-only JSON Lines keyed by (task ID, sample). Its header names
the completions file's SHA-256; running again with the same results path continues where
an interrupted run stopped. The only rewrite ever made is dropping a torn last line that an
interruption left half-written. A run holds an exclusive lock on its results file, so two
runs can never append to the same file at once.

A resumed run must find the execution environment it started under: the runner and
supervisor sources, the image and the deadline rule are compared with the results header,
and a mismatch is refused, so a window of verdicts never mixes two runners.

The pool test split is sealed: see :mod:`scratchcoder.eval.seal`. Its read is checked
against the ledger and the unseal record before the task file is opened, and it needs the
tokenizer, so the one read is never spent on completions whose prompts and tokens were not
verified.
"""

import fcntl
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any
from uuid import uuid4

from scratchcoder.common.records import git_state, utc_now
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.baseline import pool_constant_baseline, save_baseline
from scratchcoder.eval.completions import (
    Completions,
    CompletionsHeader,
    prompt_sha256,
    read_completions,
    validate_against_tasks,
)
from scratchcoder.eval.export import ExportError, refuse_exported, task_file_identity
from scratchcoder.eval.harness import HOST_RETRIES, isolated_execution
from scratchcoder.eval.parallel import Call, Execute, Outcome, run_calls
from scratchcoder.eval.protocols import CHAT, POOL, load_suite_tasks, protocol_for
from scratchcoder.eval.seal import SealError, SealLedger, UnsealRecord
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.reward import audit_attempt
from scratchcoder.sandbox.runner import IMAGE
from scratchcoder.tokenizer.bpe import ByteBPE

RESULTS_FORMAT = "scratchcoder-results"
SCHEMA_VERSION = 1
NO_PROGRAM = "no_program"
CONTEXT_LIMIT = "context_limit"
DEADLINE_RULE = "function: 2 s; I/O: 2 s + 0.25 s per case, at most 60 s; 128 MiB"
SANDBOX = Path(__file__).resolve().parents[1] / "sandbox"


def program_sha256(program: str) -> str:
    return hashlib.sha256(program.encode("utf-8", errors="surrogatepass")).hexdigest()


@dataclass(frozen=True)
class AttemptResult:
    task_id: str
    sample: int
    status: str
    passed: bool
    source: str
    written_by: str
    stop_reason: str
    program_sha256: str | None
    elapsed_seconds: float
    host_failures: int
    audit_flags: tuple[str, ...] | None
    """The static audit's flags for a passing attempt (``()`` when none); ``None`` when the
    attempt did not pass and was not audited."""

    @property
    def key(self) -> tuple[str, int]:
        return self.task_id, self.sample


@dataclass(frozen=True)
class ResultsFile:
    path: Path
    header: dict[str, Any]
    attempts: dict[tuple[str, int], AttemptResult]
    unresolved: dict[tuple[str, int], dict[str, Any]]
    """Attempts whose latest record is ``unresolved`` (no scored result yet)."""
    windows: list[dict[str, Any]]
    sha256: str
    torn_bytes: int

    @property
    def completions_header(self) -> CompletionsHeader:
        return CompletionsHeader.from_json(self.header["completions"])

    @property
    def complete(self) -> bool:
        return not self.unresolved and len(self.attempts) == self.header["attempts"]

    @property
    def sandbox_seconds(self) -> float:
        return float(sum(window.get("sandbox_seconds", 0.0) for window in self.windows))


def read_results(path: Path) -> ResultsFile:
    """Parse a results file. A torn final line (from an interruption) is ignored and counted."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    lines = data.splitlines(keepends=True)
    torn = 0
    if lines and not lines[-1].endswith(b"\n"):
        torn = len(lines.pop())
    if not lines:
        raise ValueError("empty results file")
    header = json.loads(lines[0])
    if (
        not isinstance(header, dict)
        or header.get("kind") != "header"
        or header.get("format") != RESULTS_FORMAT
        or header.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("not a results file")
    attempts: dict[tuple[str, int], AttemptResult] = {}
    unresolved: dict[tuple[str, int], dict[str, Any]] = {}
    windows: list[dict[str, Any]] = []
    for number, line in enumerate(lines[1:], 2):
        record = json.loads(line)
        kind = record.get("kind") if isinstance(record, dict) else None
        if kind == "attempt":
            fields = {key: value for key, value in record.items() if key != "kind"}
            flags = fields["audit_flags"]
            fields["audit_flags"] = None if flags is None else tuple(flags)
            result = AttemptResult(**fields)
            if result.key in attempts:
                raise ValueError(f"line {number}: attempt {result.key} scored twice")
            attempts[result.key] = result
            unresolved.pop(result.key, None)
        elif kind == "unresolved":
            key = (record["task_id"], record["sample"])
            if key in attempts:
                raise ValueError(f"line {number}: unresolved after a scored result")
            unresolved[key] = record
        elif kind in ("window", "window_end"):
            windows.append(record)
        else:
            raise ValueError(f"line {number}: unknown results record")
    return ResultsFile(path, header, attempts, unresolved, windows, digest, torn)


class _Appender:
    """Append one JSON record per line, flushed and synced, to an open results file."""

    def __init__(self, stream: IO[bytes]) -> None:
        self.stream = stream

    def write(self, record: Mapping[str, Any]) -> None:
        self.stream.write(json.dumps(record, sort_keys=True, allow_nan=False).encode() + b"\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())


def _derive_programs(
    completions: Completions, tasks: Mapping[str, Task], tokenizer: ByteBPE | None
) -> dict[tuple[str, int], str | None]:
    """Apply the protocol's extraction to every attempt and require the stored program.

    With the tokenizer, each attempt's prompt hash, prompt length and decoded completion
    are checked too, so a file whose stored fields disagree with its tokens is refused.
    """
    header = completions.header
    definition = protocol_for(header.protocol, header.suite)
    if tokenizer is not None and tokenizer.sha256 != header.tokenizer_sha256:
        raise ValueError("the tokenizer differs from the one that generated the completions")
    if definition.name == CHAT and tokenizer is None:
        raise ValueError("chat-v1 extraction reads token IDs; supply the tokenizer")
    prompts: dict[str, list[int]] = {}
    programs: dict[tuple[str, int], str | None] = {}
    for row in completions.rows:
        task = tasks[row.task_id]
        if tokenizer is not None:
            if row.task_id not in prompts:
                prompts[row.task_id] = definition.prompt_ids(tokenizer, task)
            prompt = prompts[row.task_id]
            if row.prompt_tokens != len(prompt) or row.prompt_sha256 != prompt_sha256(prompt):
                raise ValueError(f"{row.key}: the prompt differs from the protocol's prompt")
            if row.completion != tokenizer.decode(row.tokens, skip_special=True):
                raise ValueError(f"{row.key}: the completion text differs from its tokens")
        if row.stop_reason == CONTEXT_LIMIT:
            program = None
        else:
            program = definition.candidate(
                task, tokens=row.tokens, completion=row.completion, decoder=tokenizer
            )
        if program != row.program:
            raise ValueError(
                f"{row.key}: the stored program differs from the protocol's extraction"
            )
        programs[row.key] = program
    return programs


ENVIRONMENT_KEYS = ("image", "deadline_rule", "runner_sha256", "program_sha256")
"""The parts of the execution record a resumed run must find unchanged."""


def _execution_record(execute: Execute, retries: int) -> dict[str, Any]:
    name = f"{getattr(execute, '__module__', '?')}.{getattr(execute, '__qualname__', '?')}"
    return {
        "execute": name,
        "image": IMAGE,
        "deadline_rule": DEADLINE_RULE,
        "runner_sha256": file_sha256(SANDBOX / "runner.py"),
        "program_sha256": file_sha256(SANDBOX / "program.py"),
        "host_retries": retries,
        "host_backoff": "2 s doubling, at most 30 s",
        "audit": "scratchcoder.sandbox.reward.audit_attempt on passing attempts",
    }


def _new_header(
    completions: Completions,
    tasks: Mapping[str, Task],
    *,
    execute: Execute,
    retries: int,
    tokenizer: ByteBPE | None,
    unseal: UnsealRecord | None,
) -> dict[str, Any]:
    commit, dirty = git_state()
    return {
        "kind": "header",
        "format": RESULTS_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "results_id": uuid4().hex,
        "completions_sha256": completions.sha256,
        "completions": completions.header.to_json(),
        "task_file_sha256": completions.header.task_file_sha256,
        "attempts": completions.header.rows,
        "tasks": [
            {"task_id": task.id, "source": task.source, "written_by": task.written_by}
            for task in sorted(tasks.values(), key=lambda item: item.id)
        ],
        "execution": _execution_record(execute, retries),
        "extraction_checked_with_tokenizer": tokenizer is not None,
        "unseal_sha256": None if unseal is None else unseal.sha256,
        "git_commit": commit,
        "dirty": dirty,
        "created_at": utc_now(),
    }


def execute_completions(
    completions_path: Path,
    tasks_path: Path,
    results_path: Path,
    *,
    tokenizer: ByteBPE | None = None,
    workers: int = 8,
    execute: Execute = isolated_execution,
    retries: int = HOST_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
    seal_ledger: Path | None = None,
    unseal: Path | None = None,
    max_consecutive_unresolved: int | None = None,
) -> dict[str, Any]:
    """Execute (or continue executing) a pool completions file; return a summary.

    The pool test split additionally needs ``seal_ledger`` and ``unseal``; its read is
    logged before anything runs, and its constant-answer baseline is computed inside the
    same step (once per test task file) and recorded in the ledger.
    """
    completions = read_completions(completions_path)
    header = completions.header
    if header.suite != POOL:
        raise ValueError("benchmark completions are scored by the GitHub Actions workflow")
    sealed = header.split == "test"
    if sealed and (seal_ledger is None or unseal is None):
        raise SealError("the pool test split is sealed: give the seal ledger and an unseal record")
    if not sealed and unseal is not None:
        raise ValueError("an unseal record applies to the test split only")
    if sealed and tokenizer is None:
        raise ValueError(
            "the pool test split needs the tokenizer: its one read must not be spent on "
            "completions whose prompts and tokens were not checked"
        )
    record = UnsealRecord.load(unseal) if sealed and unseal is not None else None
    ledger = SealLedger(seal_ledger) if sealed and seal_ledger is not None else None
    if ledger is not None and record is not None:
        # Refuse before the sealed task file is opened, hashed or parsed.
        ledger.check_read(completions, record)
    tasks = {task.id: task for task in validate_against_tasks(completions, tasks_path)}
    programs = _derive_programs(completions, tasks, tokenizer)
    environment = _execution_record(execute, retries)
    created = not results_path.exists()
    if created:
        results_header = _new_header(
            completions,
            tasks,
            execute=execute,
            retries=retries,
            tokenizer=tokenizer,
            unseal=record,
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("xb") as stream:
            _Appender(stream).write(results_header)
        previous = read_results(results_path)
    else:
        previous = read_results(results_path)
        if previous.header["completions_sha256"] != completions.sha256:
            raise ValueError("the results file belongs to other completions")
        changed = [
            key
            for key in ENVIRONMENT_KEYS
            if previous.header.get("execution", {}).get(key) != environment[key]
        ]
        if changed:
            raise ValueError(
                "the results file was written under another execution environment "
                f"({', '.join(changed)} differ); a window must not mix two runners, "
                "so start a new results file"
            )
        if record is not None and previous.header["unseal_sha256"] != record.sha256:
            raise SealError("the results file was opened under another unseal record")
    results_id = previous.header["results_id"]
    ledger_head = None
    if ledger is not None and record is not None:
        try:
            ledger.open_read(completions, record, results_id=results_id, results_path=results_path)
        except SealError:
            if created:
                results_path.unlink()
            raise
        ledger_head = ledger.head()
    started, sandbox_seconds = time.monotonic(), 0.0
    resolved = unresolved = passed = flagged = 0
    commit, dirty = git_state()
    with results_path.open("ab") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another eval-execute is writing this results file") from error
        previous = read_results(results_path)
        if previous.torn_bytes:
            with results_path.open("r+b") as torn:
                torn.truncate(results_path.stat().st_size - previous.torn_bytes)
        pending = [row for row in completions.rows if row.key not in previous.attempts]
        out = _Appender(stream)
        out.write(
            {
                "kind": "window",
                "started_at": utc_now(),
                "workers": workers,
                "pending": len(pending),
                "git_commit": commit,
                "dirty": dirty,
                **({} if ledger_head is None else {"ledger": ledger_head}),
            }
        )

        def record_result(
            key: tuple[str, int], status: str, outcome: Outcome | None = None
        ) -> None:
            nonlocal resolved, passed, flagged
            row = completions.row(*key)
            task = tasks[row.task_id]
            program = programs[key]
            success = status == "passed"
            flags = audit_attempt(program) if success and program is not None else None
            out.write(
                {
                    "kind": "attempt",
                    **asdict(
                        AttemptResult(
                            task_id=row.task_id,
                            sample=row.sample,
                            status=status,
                            passed=success,
                            source=task.source,
                            written_by=task.written_by,
                            stop_reason=row.stop_reason,
                            program_sha256=None if program is None else program_sha256(program),
                            elapsed_seconds=0.0 if outcome is None else outcome.elapsed_seconds,
                            host_failures=0 if outcome is None else outcome.host_failures,
                            audit_flags=flags,
                        )
                    ),
                }
            )
            resolved += 1
            passed += success
            flagged += bool(flags)

        calls = []
        for row in pending:
            program = programs[row.key]
            if program is None:
                status = CONTEXT_LIMIT if row.stop_reason == CONTEXT_LIMIT else NO_PROGRAM
                record_result(row.key, status)
                continue
            task = tasks[row.task_id]
            calls.append(Call(row.key, program, task.hidden_tests, task.mode))

        def collect(outcome: Outcome) -> None:
            nonlocal unresolved, sandbox_seconds
            sandbox_seconds += outcome.elapsed_seconds
            key = outcome.key
            assert isinstance(key, tuple)
            if outcome.resolved:
                record_result(key, outcome.status, outcome)
                return
            unresolved += 1
            out.write(
                {
                    "kind": "unresolved",
                    "task_id": key[0],
                    "sample": key[1],
                    "last_status": outcome.status,
                    "host_failures": outcome.host_failures,
                    "elapsed_seconds": outcome.elapsed_seconds,
                }
            )

        limit = max_consecutive_unresolved or max(16, 4 * workers)
        try:
            run_calls(
                calls,
                execute,
                on_outcome=collect,
                workers=workers,
                retries=retries,
                sleep=sleep,
                max_consecutive_unresolved=limit,
            )
        finally:
            out.write(
                {
                    "kind": "window_end",
                    "ended_at": utc_now(),
                    "resolved": resolved,
                    "unresolved": unresolved,
                    "passed": passed,
                    "audit_flagged": flagged,
                    "wall_seconds": time.monotonic() - started,
                    "sandbox_seconds": sandbox_seconds,
                }
            )
    final = read_results(results_path)
    baseline_path = None
    if ledger is not None and final.complete:
        ledger.complete_read(completions.sha256, results_id=results_id, results_sha256=final.sha256)
    if ledger is not None and final.complete:
        baseline_path = _test_baseline(
            ledger,
            tasks,
            header.task_file_sha256,
            results_id=results_id,
            execute=execute,
            workers=workers,
            retries=retries,
            sleep=sleep,
        )
    return {
        "results": str(results_path),
        "results_sha256": final.sha256,
        "completions_sha256": completions.sha256,
        "attempts": header.rows,
        "scored": len(final.attempts),
        "passed": sum(result.passed for result in final.attempts.values()),
        "unresolved": len(final.unresolved),
        "pending": header.rows - len(final.attempts) - len(final.unresolved),
        "audit_flagged_passes": sum(bool(result.audit_flags) for result in final.attempts.values()),
        "this_window": {
            "resolved": resolved,
            "unresolved": unresolved,
            "sandbox_seconds": sandbox_seconds,
            "wall_seconds": time.monotonic() - started,
        },
        "test_baseline": None if baseline_path is None else str(baseline_path),
    }


def _test_baseline(
    ledger: SealLedger,
    tasks: Mapping[str, Task],
    task_file_sha256: str,
    *,
    results_id: str,
    execute: Execute,
    workers: int,
    retries: int,
    sleep: Callable[[float], None],
) -> Path | None:
    """The test split's constant-answer baseline, computed once, inside a logged read.

    It is computed after a read whose attempts are all resolved, and logged only if no
    constant-answer call stayed unresolved; otherwise it is left for a later read or for
    :func:`run_test_baseline` and None is returned (an uncovered task for another reason is
    reported, not retried). A baseline file that an interruption saved without its ledger
    event is checked against the task file and recorded, not computed again."""
    existing = ledger.baseline_event(task_file_sha256)
    if existing is not None:
        path = Path(existing["path"])
        if not path.exists() or file_sha256(path) != existing["sha256"]:
            raise SealError("the logged test baseline file is missing or changed")
        return path
    path = ledger.path.with_name(f"pool-test-baseline-{task_file_sha256[:16]}.json")
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(saved, dict)
            or saved.get("split") != "test"
            or saved.get("task_file_sha256") != task_file_sha256
        ):
            raise SealError(f"{path} is not the test baseline of this task file")
    else:
        report = pool_constant_baseline(
            list(tasks.values()),
            execute=execute,
            workers=workers,
            retries=retries,
            sleep=sleep,
            allow_test=True,
        )
        if any("unresolved" in item["reason"] for item in report["summary"]["uncovered_tasks"]):
            return None
        commit, dirty = git_state()
        report.update(
            split="test",
            task_file_sha256=task_file_sha256,
            git_commit=commit,
            dirty=dirty,
            created_at=utc_now(),
            computed_in_results_id=results_id,
        )
        save_baseline(path, report)
    ledger.record_baseline(
        task_file_sha256=task_file_sha256,
        path=path,
        sha256=file_sha256(path),
        results_id=results_id,
    )
    return path


def run_test_baseline(
    tasks_path: Path,
    seal_ledger: Path,
    unseal: Path,
    *,
    execute: Execute = isolated_execution,
    workers: int = 8,
    retries: int = HOST_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Compute and log the test split's constant-answer baseline after every read finished.

    This is the retry path for a baseline that :func:`execute_completions` could not log
    (a constant-answer call stayed unresolved, or the ledger write failed). It is gated by
    the ledger: the unseal record must be the one the reads opened under, every completions
    file it lists must have a completed read, and the task file must be the one those files
    were generated for. Nothing is read from the task file before that gate passes. A
    baseline already logged is returned unchanged.
    """
    record = UnsealRecord.load(unseal)
    ledger = SealLedger(seal_ledger)
    finished = ledger.finished_unseal(record)
    expected = finished["task_file_sha256"]
    try:
        # The sealed files may name the full task file or its Pod export, derived again here;
        # the export itself, with placeholder tests, is never the file to run.
        task_file_identity(tasks_path, expected)
        tasks, _ = load_suite_tasks("pool", "test", tasks_path)
        refuse_exported(tasks, tasks_path)
    except ExportError as error:
        raise SealError(
            "the task file is not the test task file the sealed completions name"
        ) from error
    path = _test_baseline(
        ledger,
        {task.id: task for task in tasks},
        expected,
        results_id=finished["results_id"],
        execute=execute,
        workers=workers,
        retries=retries,
        sleep=sleep,
    )
    return {
        "test_baseline": None if path is None else str(path),
        "task_file_sha256": expected,
        "recorded": path is not None,
    }
