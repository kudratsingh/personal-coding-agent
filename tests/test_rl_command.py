"""Tests for the local RL screening commands and the attempt/reward JSON forms."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.cli import main
from scratchcoder.eval.tasks import Task
from scratchcoder.rl import command
from scratchcoder.rl.command import read_candidates, screen_attempts
from scratchcoder.rl.rewards import NO_PROGRAM, Attempt, AttemptReward, LocalRewardSource
from scratchcoder.sandbox.reward import COMPLETED, RewardResult


def task(identifier: str, hidden: str, *, split: str = "train", source: str = "s1") -> Task:
    return Task(
        id=identifier,
        prompt="Compute f.",
        public_tests="assert f(3) == 11",
        hidden_tests=hidden,
        source=source,
        license="fixture",
        written_by="human",
        split=split,
        solution="def f(x):\n    return 2 * x + 5",
    )


TASKS = [
    task("a", "assert f(6) == 17\nassert f(7) == 19"),
    task("b", "assert f(1) == 0\nassert f(2) == 0", source="s2"),
    task("c", "x = f(1)\nassert x == 7", source="s2"),
]


def write(tmp_path: Path, tasks: list[Task], listed: list[tuple[str, str]]) -> tuple[Path, Path]:
    tasks_path, candidates_path = tmp_path / "train.jsonl", tmp_path / "candidates.jsonl"
    tasks_path.write_text("".join(json.dumps(asdict(t)) + "\n" for t in tasks))
    candidates_path.write_text(
        "".join(json.dumps({"id": i, "source": s}) + "\n" for i, s in listed)
    )
    return tasks_path, candidates_path


def test_read_candidates_checks_split_source_and_presence(tmp_path: Path) -> None:
    tasks_path, candidates_path = write(tmp_path, TASKS, [("c", "s2"), ("a", "s1")])
    assert [t.id for t in read_candidates(tasks_path, candidates_path)] == ["c", "a"]
    for listed in ([("a", "s2")], [("zzz", "s1")], [("a", "s1"), ("a", "s1")]):
        tasks_path, candidates_path = write(tmp_path, TASKS, listed)
        with pytest.raises(ValueError):
            read_candidates(tasks_path, candidates_path)
    tasks_path, candidates_path = write(
        tmp_path, [task("d", "assert f(6) == 17", split="dev")], [("d", "s1")]
    )
    with pytest.raises(ValueError, match="not a train task"):
        read_candidates(tasks_path, candidates_path)


def test_rl_prescreen_command_prints_the_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks_path, candidates_path = write(tmp_path, TASKS, [("a", "s1"), ("b", "s2"), ("c", "s2")])
    assert (
        main(["rl-prescreen", "--tasks", str(tasks_path), "--candidates", str(candidates_path)])
        == 0
    )
    counts = json.loads(capsys.readouterr().out)
    assert counts["candidates"] == 3 and counts["parseable"] == 2
    assert counts["constant_baseline"] == 1 and counts["eligible_before_equal_rewards"] == 1
    assert counts["by_source"]["s2"]["not_parseable"] == 1


def test_attempt_and_reward_json_round_trip() -> None:
    attempt = Attempt("a", 3, None, 7)
    assert Attempt.from_json(json.loads(json.dumps(attempt.to_json()))) == attempt
    reward = AttemptReward(0.5, COMPLETED, ("inspect",), 1.25, 2)
    assert AttemptReward.from_json(json.loads(json.dumps(reward.to_json()))) == reward
    with pytest.raises(ValueError):
        Attempt.from_json({"task_id": "a", "sample": 0, "code": None})
    with pytest.raises(ValueError):
        AttemptReward.from_json({**reward.to_json(), "extra": 1})


def test_screen_attempts_scores_and_writes_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sandbox is replaced by a fake score; the record carries every reward."""

    def fake_score(item: Task, code: str) -> RewardResult:
        return RewardResult(COMPLETED, (len(code) % 3) / 2, 1, 2)

    def local(tasks: Any, **options: Any) -> LocalRewardSource:
        return LocalRewardSource(tasks, score=fake_score, **options)

    monkeypatch.setattr(command, "LocalRewardSource", local)
    tasks_path, candidates_path = write(tmp_path, TASKS, [("a", "s1"), ("b", "s2"), ("c", "s2")])
    attempts_path = tmp_path / "attempts.jsonl"
    rows = [Attempt(t.id, s, None if s == 7 else "x" * s, s) for t in TASKS for s in range(8)]
    attempts_path.write_text("".join(json.dumps(row.to_json()) + "\n" for row in rows))
    output = tmp_path / "screening.json"
    summary = screen_attempts(
        tasks_path=tasks_path,
        candidates_path=candidates_path,
        attempts_path=attempts_path,
        output=output,
        timeout=60.0,
        floor=1,
    )
    assert summary["outcome"] == "kept" and summary["kept"] == 1
    record = json.loads(output.read_text())
    assert record["kept"] == ["a"] and len(record["rewards"]) == 24
    assert record["rewards"][7]["status"] == NO_PROGRAM
    reasons = {row["id"]: row["reason"] for row in record["tasks"]}
    assert reasons == {"a": None, "b": "constant_baseline", "c": "not_parseable"}
    with pytest.raises(ValueError, match="already exists"):
        screen_attempts(
            tasks_path=tasks_path,
            candidates_path=candidates_path,
            attempts_path=attempts_path,
            output=output,
            timeout=60.0,
        )
