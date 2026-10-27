import hashlib
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from scratchcoder.eval import harness
from scratchcoder.eval.harness import (
    ContextLimitError,
    EvaluationConfig,
    attempt_seed,
    evaluate,
    execution_limits,
    host_backoff,
    is_host_failure,
    isolated_execution,
    save_report,
)
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.runner import Limits, Result


def tasks_fixture() -> list[Task]:
    return [
        Task(
            id=str(i),
            prompt="Implement a function",
            public_tests="public-test",
            hidden_tests="hidden-test",
            source=f"synthetic-{i}",
            license="fixture",
            written_by="human" if i == 0 else "model",
            split="test",
        )
        for i in range(2)
    ]


def test_harness_metrics_lineage_and_scorecard(tmp_path: Path) -> None:
    report = evaluate(
        tasks_fixture(),
        lambda task, _: "correct" if task.id == "0" else "incorrect",
        config=EvaluationConfig(bootstrap_samples=100),
        checkpoint_hash="fixture-checkpoint",
        tokenizer_hash="fixture-tokenizer",
        execute=lambda code, tests, mode: Result("passed" if code == "correct" else "failed"),
    )
    assert report["metrics"]["mean_pass_rate"] == 0.5
    assert report["metrics"]["greedy_pass_at_1"] == 0.5
    assert report["metrics"]["by_authorship"] == {"human": 1.0, "model": 0.0}
    assert report["checkpoint_hash"] == "fixture-checkpoint"
    scorecard = tmp_path / "scorecard.jsonl"
    save_report(report, tmp_path / "evaluation", scorecard=scorecard)
    assert json.loads(scorecard.read_text())["task_count"] == 2
    saved = json.loads((tmp_path / "evaluation/report.json").read_text())
    assert saved["task_hash"] == report["task_hash"]
    assert "hidden-test" not in scorecard.read_text()


def test_unavailable_isolation_does_not_produce_a_score() -> None:
    with pytest.raises(RuntimeError, match="aborted"):
        evaluate(
            tasks_fixture(),
            lambda *_: "code",
            config=EvaluationConfig(),
            checkpoint_hash="checkpoint",
            tokenizer_hash="tokenizer",
            execute=lambda *_: Result("unavailable"),
        )


def test_context_limit_counts_as_model_limit_and_training_tasks_rejected() -> None:
    def too_long(task: Task, seed: int) -> str:
        raise ContextLimitError("too long")

    report = evaluate(
        tasks_fixture(),
        too_long,
        config=EvaluationConfig(bootstrap_samples=10),
        checkpoint_hash="checkpoint",
        tokenizer_hash="tokenizer",
    )
    assert report["metrics"]["mean_pass_rate"] == 0
    assert report["per_task"][0]["attempts"][0]["status"] == "context_limit"
    with pytest.raises(ValueError, match="training"):
        evaluate(
            [replace(tasks_fixture()[0], split="train")],
            too_long,
            config=EvaluationConfig(),
            checkpoint_hash="checkpoint",
            tokenizer_hash="tokenizer",
        )


def test_task_order_does_not_change_seeds_or_scores() -> None:
    run_evaluation = partial(
        evaluate,
        config=EvaluationConfig(bootstrap_samples=10),
        checkpoint_hash="c",
        tokenizer_hash="t",
        execute=lambda *_: Result("passed"),
    )
    first = run_evaluation(tasks_fixture(), lambda *_: "code")
    second = run_evaluation(list(reversed(tasks_fixture())), lambda *_: "code")
    assert first["per_task"] == second["per_task"]


def test_attempt_seeds_follow_the_recorded_formula() -> None:
    digest = int(hashlib.sha256(b"authored/7").hexdigest()[:16], 16)
    assert attempt_seed(5, "authored/7", 2) == (5 + digest + 2) % (2**63 - 1)
    assert attempt_seed(5, "authored/7", 2) == 6715044113498020353
    report = evaluate(
        tasks_fixture(),
        lambda *_: "code",
        config=EvaluationConfig(samples=3, temperature=1.0, seed=4, bootstrap_samples=10),
        checkpoint_hash="c",
        tokenizer_hash="t",
        execute=lambda *_: Result("passed"),
    )
    for row in report["per_task"]:
        seeds = [attempt["seed"] for attempt in row["attempts"]]
        assert seeds == [attempt_seed(4, row["id"], index) for index in range(3)]


def test_host_stalled_results_are_retried_never_scored() -> None:
    # A stall reports status "timeout"; before the fix it was scored as a candidate timeout.
    stalled = Result("timeout", host_stalled=True)
    results = iter([stalled, stalled, Result("passed")])
    waits: list[float] = []
    report = evaluate(
        tasks_fixture()[:1],
        lambda *_: "code",
        config=EvaluationConfig(bootstrap_samples=10),
        checkpoint_hash="c",
        tokenizer_hash="t",
        execute=lambda *_: next(results),
        sleep=waits.append,
    )
    assert report["per_task"][0]["attempts"][0]["status"] == "passed"
    assert report["metrics"]["greedy_pass_at_1"] == 1.0
    assert waits == [2.0, 4.0]


def test_a_stall_that_never_clears_aborts_instead_of_scoring_a_timeout() -> None:
    calls: list[str] = []
    waits: list[float] = []

    def stalls(code: str, tests: str, mode: str) -> Result:
        calls.append(code)
        return Result("timeout", host_stalled=True)

    with pytest.raises(RuntimeError, match="aborted"):
        evaluate(
            tasks_fixture()[:1],
            lambda *_: "code",
            config=EvaluationConfig(),
            checkpoint_hash="c",
            tokenizer_hash="t",
            execute=stalls,
            sleep=waits.append,
        )
    assert len(calls) == 1 + harness.HOST_RETRIES
    assert waits == [2.0, 4.0, 8.0]


def test_host_failure_classification_and_backoff() -> None:
    for status in ("passed", "failed", "timeout", "output_limit", "resource_limit"):
        assert not is_host_failure(Result(status))
    assert is_host_failure(Result("timeout", host_stalled=True))
    assert is_host_failure(Result("unavailable"))
    assert is_host_failure(Result("infrastructure_error"))
    assert [host_backoff(n) for n in range(1, 7)] == [2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
    with pytest.raises(ValueError):
        host_backoff(0)


def test_execution_limits_follow_the_pool_verification_deadline_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert execution_limits("assert f(1) == 2", "function") == Limits()
    cases = json.dumps([{"input": str(i), "output": str(i)} for i in range(99)])
    # 2 s plus 0.25 s per case: the deadline the pool's I/O references passed under.
    assert execution_limits(cases, "io") == Limits(timeout_seconds=26.75)
    single = json.dumps([{"input": "", "output": ""}])
    assert execution_limits(single, "io").timeout_seconds == 2.25
    seen: list[tuple[str, str, Limits | None, str]] = []

    def fake_run(code: str, tests: str, limits: Limits | None = None, *, mode: str) -> Result:
        seen.append((code, tests, limits, mode))
        return Result("passed")

    monkeypatch.setattr(harness, "run", fake_run)
    assert isolated_execution("print(1)", cases, "io").passed
    assert seen == [("print(1)", cases, Limits(timeout_seconds=26.75), "io")]
