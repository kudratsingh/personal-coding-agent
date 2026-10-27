"""Parallel sandbox calls: host failures are retried, bounded and never scored."""

import json
import threading
from pathlib import Path

import pytest
from eval_fixtures import pool_task, save_task_file

from scratchcoder.cli import main
from scratchcoder.eval.parallel import (
    OVERSIZED,
    Call,
    HostUnavailableError,
    Outcome,
    resolve_call,
    run_calls,
)
from scratchcoder.sandbox.runner import Result


def test_a_stall_is_retried_with_backoff_then_resolved() -> None:
    results = iter(
        [
            Result("timeout", host_stalled=True, elapsed_seconds=12),
            Result("failed", elapsed_seconds=1),
        ]
    )
    waits: list[float] = []
    outcome = resolve_call(
        Call("k", "c", "t", "function"), lambda *_: next(results), sleep=waits.append
    )
    assert outcome.resolved and outcome.status == "failed" and not outcome.passed
    assert outcome.host_failures == 1 and outcome.elapsed_seconds == 13 and waits == [2.0]


def test_a_persistent_host_failure_is_unresolved_not_failed() -> None:
    calls: list[int] = []
    waits: list[float] = []

    def unavailable(*_: str) -> Result:
        calls.append(1)
        return Result("unavailable")

    outcome = resolve_call(Call("k", "c", "t", "function"), unavailable, sleep=waits.append)
    assert not outcome.resolved and outcome.status == "unavailable"
    assert len(calls) == 4 and waits == [2.0, 4.0, 8.0] and outcome.host_failures == 4
    stalled = resolve_call(
        Call("k", "c", "t", "function"),
        lambda *_: Result("timeout", host_stalled=True),
        retries=0,
        sleep=waits.append,
    )
    assert not stalled.resolved and stalled.status == "host_stalled"


def test_an_input_the_runner_refuses_is_the_candidates_failure() -> None:
    def refuse(*_: str) -> Result:
        raise ValueError("oversized input")

    outcome = resolve_call(Call("k", "c", "t", "function"), refuse)
    assert outcome.resolved and outcome.status == OVERSIZED and not outcome.passed


def test_run_calls_delivers_every_outcome_on_the_calling_thread() -> None:
    caller = threading.get_ident()
    seen: list[Outcome] = []

    def collect(outcome: Outcome) -> None:
        assert threading.get_ident() == caller
        seen.append(outcome)

    calls = [Call(index, "pass" if index % 2 else "fail", "t", "function") for index in range(25)]
    count = run_calls(
        calls,
        lambda code, *_: Result("passed" if code == "pass" else "failed"),
        on_outcome=collect,
        workers=3,
    )
    assert count == 25 and sorted(outcome.key for outcome in seen) == list(range(25))  # type: ignore[type-var]
    assert sum(outcome.passed for outcome in seen) == 12


def test_a_down_host_stops_new_calls_after_a_streak() -> None:
    started: list[object] = []
    seen: list[Outcome] = []

    def down(code: str, *_: str) -> Result:
        started.append(code)
        return Result("unavailable")

    with pytest.raises(HostUnavailableError):
        run_calls(
            (Call(index, str(index), "t", "function") for index in range(1000)),
            down,
            on_outcome=seen.append,
            workers=2,
            retries=0,
            sleep=lambda _: None,
            max_consecutive_unresolved=5,
        )
    assert 5 <= len(seen) < 20 and len(started) == len(seen)
    assert not any(outcome.resolved for outcome in seen)


def test_eval_baseline_cli_on_a_dev_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = save_task_file(
        tmp_path / "dev.jsonl", [pool_task("t/1", hidden_tests="assert add_one(5) == 0")]
    )
    output = tmp_path / "baseline.json"
    assert main(["eval-baseline", "--tasks", str(tasks), "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["mean_reward"] == 1.0 and printed["covered"] == 1
    saved = json.loads(output.read_text())
    assert saved["split"] == "dev" and saved["per_task"][0]["answer"] == "constant False"
    with pytest.raises(SystemExit):
        main(["eval-baseline", "--tasks", str(tasks), "--output", str(output)])
    test_file = save_task_file(tmp_path / "test.jsonl", [pool_task("t/2", split="test")])
    with pytest.raises(ValueError, match="only dev"):
        main(["eval-baseline", "--tasks", str(test_file), "--output", str(tmp_path / "t.json")])
