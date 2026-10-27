"""Tests for RL screening: the keep rule, its record, the floor and the pre-screen counts."""

import hashlib
import json

import pytest

from scratchcoder.eval.tasks import Task
from scratchcoder.rl.fakes import FakeRewardSource
from scratchcoder.rl.rewards import NO_PROGRAM, Attempt, AttemptReward
from scratchcoder.rl.screening import (
    REASONS,
    hash_order,
    prescreen,
    require_train_tasks,
    score_screening,
    screen,
)
from scratchcoder.sandbox.reward import COMPLETED


def task(
    identifier: str, hidden: str, *, public: str = "assert f(3) == 11", source: str = "s1"
) -> Task:
    return Task(
        id=identifier,
        prompt="Compute f.",
        public_tests=public,
        hidden_tests=hidden,
        source=source,
        license="fixture",
        written_by="human",
        split="train",
        solution="def f(x):\n    return 2 * x + 5",
    )


ELIGIBLE = "assert f(6) == 17\nassert f(7) == 19"  # no constant, echo or public value passes
CONSTANT = "assert f(1) == 0\nassert f(2) == 0\nassert f(5) == 15"  # constant 0 earns 2/3
UNPARSEABLE = "x = f(1)\nassert x == 7"  # a statement other than assert or import


def attempts_for(item: Task, rewards: list[float]) -> tuple[list[Attempt], list[AttemptReward]]:
    attempts = [Attempt(item.id, sample, f"code{sample}", 4) for sample in range(len(rewards))]
    return attempts, [AttemptReward(value, COMPLETED) for value in rewards]


def scored(
    items: list[tuple[Task, list[float]]],
) -> tuple[list[Task], list[Attempt], list[AttemptReward]]:
    tasks, attempts, rewards = [], [], []
    for item, values in items:
        tasks.append(item)
        more, replies = attempts_for(item, values)
        attempts += more
        rewards += replies
    return tasks, attempts, rewards


UNEQUAL = [0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_the_keep_rule_and_its_drop_reasons() -> None:
    tasks, attempts, rewards = scored(
        [
            (task("keep", ELIGIBLE), UNEQUAL),
            (task("equal", ELIGIBLE), [0.0] * 8),
            (task("constant", CONSTANT), UNEQUAL),
            (task("unparseable", UNPARSEABLE), UNEQUAL),
        ]
    )
    result = screen(tasks, attempts, rewards, floor=1)
    reasons = {row.id: row.reason for row in result.tasks}
    assert reasons == {
        "keep": None,
        "equal": "equal_rewards",
        "constant": "constant_baseline",
        "unparseable": "not_parseable",
    }
    assert result.outcome == "kept" and result.kept == ("keep",) and result.eligible == ("keep",)
    constant = next(row for row in result.tasks if row.id == "constant")
    # False == 0 in Python, and False comes first among the constant answers.
    assert constant.baseline == pytest.approx(2 / 3)
    assert constant.baseline_answer == "constant False"
    keep = next(row for row in result.tasks if row.id == "keep")
    assert keep.baseline == 0.0 and keep.rewards == tuple(UNEQUAL)
    assert keep.statuses == (COMPLETED,) * 8 and keep.parseable
    unparseable = next(row for row in result.tasks if row.id == "unparseable")
    assert unparseable.baseline is None and not unparseable.parseable


def test_the_threshold_is_a_setting_and_drops_at_or_above_it() -> None:
    tasks, attempts, rewards = scored([(task("constant", CONSTANT), UNEQUAL)])
    assert screen(tasks, attempts, rewards, floor=1).tasks[0].reason == "constant_baseline"
    exact = screen(tasks, attempts, rewards, floor=1, threshold=2 / 3)
    assert exact.tasks[0].reason == "constant_baseline"  # >= threshold drops
    above = screen(tasks, attempts, rewards, floor=1, threshold=0.7)
    assert above.kept == ("constant",)


def test_kept_tasks_follow_id_hash_order_and_the_cap() -> None:
    names = [f"t{index}" for index in range(12)]
    tasks, attempts, rewards = scored([(task(name, ELIGIBLE), UNEQUAL) for name in names])
    ordered = sorted(names, key=lambda name: hash_order(name))
    result = screen(tasks, attempts, rewards, cap=5, floor=1)
    assert result.kept == tuple(ordered[:5])
    assert result.eligible == tuple(ordered)
    assert [row.id for row in result.tasks] == ordered
    assert {row.id for row in result.tasks if row.reason == "over_cap"} == set(ordered[5:])
    # The key is sha256("rl-candidate-v1|" + id).
    assert hash_order("t0") == hashlib.sha256(b"rl-candidate-v1|t0").digest()


def test_fewer_than_the_floor_returns_a_floor_result_without_a_task_list() -> None:
    tasks, attempts, rewards = scored([(task(f"t{i}", ELIGIBLE), UNEQUAL) for i in range(3)])
    result = screen(tasks, attempts, rewards, floor=4)
    assert result.outcome == "floor" and result.kept is None and len(result.eligible) == 3
    assert screen(tasks, attempts, rewards, floor=3).outcome == "kept"
    assert screen(tasks, attempts, rewards).outcome == "floor"  # default floor 320


def test_the_record_counts_by_source_and_reason_and_serializes() -> None:
    tasks, attempts, rewards = scored(
        [
            (task("a", ELIGIBLE, source="s1"), UNEQUAL),
            (task("b", ELIGIBLE, source="s2"), [1.0] * 8),
            (task("c", CONSTANT, source="s2"), UNEQUAL),
            (task("d", UNPARSEABLE, source="s1"), UNEQUAL),
        ]
    )
    result = screen(tasks, attempts, rewards, floor=1)
    counts = result.counts()
    assert counts["candidates"] == 4 and counts["parseable"] == 3
    assert counts["baseline_at_or_above_threshold"] == 1
    assert counts["eligible_before_equal_rewards"] == 2 and counts["kept"] == 1
    assert counts["by_reason"] == {
        "kept": 1,
        "not_parseable": 1,
        "constant_baseline": 1,
        "equal_rewards": 1,
        "over_cap": 0,
    }
    assert (
        counts["by_source"]["s1"]["kept"] == 1 and counts["by_source"]["s1"]["not_parseable"] == 1
    )
    assert counts["by_source"]["s2"]["equal_rewards"] == 1
    record = json.loads(json.dumps(result.to_json(), allow_nan=False))
    assert record["outcome"] == "kept" and record["kept"] == ["a"]
    assert record["rule"]["threshold"] == 0.5 and record["rule"]["reasons"] == list(REASONS)
    assert {row["id"]: len(row["rewards"]) for row in record["tasks"]} == dict.fromkeys("abcd", 8)


def test_screen_rejects_incomplete_or_foreign_attempts() -> None:
    item = task("a", ELIGIBLE)
    attempts, rewards = attempts_for(item, UNEQUAL[:7])
    with pytest.raises(ValueError):
        screen([item], attempts, rewards)
    attempts, rewards = attempts_for(task("other", ELIGIBLE), UNEQUAL)
    with pytest.raises(ValueError):
        screen([item], attempts, rewards)
    attempts, rewards = attempts_for(item, UNEQUAL)
    with pytest.raises(ValueError):
        screen([item], attempts + attempts[:1], rewards + rewards[:1])
    with pytest.raises(ValueError):
        screen([item], attempts, rewards, threshold=1.5)


def test_score_screening_sends_numbered_batches_in_order() -> None:
    attempts = [Attempt("a", index % 8, None if index % 3 == 0 else "xx", 1) for index in range(20)]
    fake = FakeRewardSource()
    rewards = score_screening(fake, attempts, chunk=8)
    assert fake.calls == [(0, 8), (1, 8), (2, 4)]
    assert [reward.status == NO_PROGRAM for reward in rewards] == [
        attempt.code is None for attempt in attempts
    ]


def test_prescreen_counts_parseable_and_high_baseline_tasks_by_source() -> None:
    tasks = [
        task("a", ELIGIBLE, source="s1"),
        task("b", CONSTANT, source="s1"),
        task("c", UNPARSEABLE, source="s2"),
        task("d", ELIGIBLE, source="s2"),
    ]
    counts = prescreen(tasks)
    assert counts["candidates"] == 4 and counts["parseable"] == 3
    assert counts["not_parseable"] == 1 and counts["constant_baseline"] == 1
    assert counts["eligible_before_equal_rewards"] == 2
    assert counts["by_source"]["s1"] == {
        "candidates": 2,
        "not_parseable": 0,
        "parseable": 2,
        "constant_baseline": 1,
        "eligible_before_equal_rewards": 1,
    }


def test_require_train_tasks_names_every_kind_of_offender() -> None:
    good = task("good", ELIGIBLE)
    baselines = require_train_tasks([good, task("const", CONSTANT)])
    assert baselines["good"].parseable and baselines["const"].reward == pytest.approx(2 / 3)
    held_out = Task(**{**good.__dict__, "id": "sealed", "split": "test"})
    bare = Task(**{**good.__dict__, "id": "bare", "solution": None})
    messy = task("messy", UNPARSEABLE)
    require_train_tasks([bare])  # a missing reference matters only to the control
    with pytest.raises(ValueError, match="1 not train-split tasks, e.g. sealed"):
        require_train_tasks([good, held_out])
    with pytest.raises(ValueError, match="1 unparseable by the reward channel, e.g. messy"):
        require_train_tasks([good, messy])
    with pytest.raises(ValueError, match="without a reference solution, e.g. bare"):
        require_train_tasks([good, bare], references=True)
    with pytest.raises(ValueError) as combined:
        require_train_tasks([held_out, messy, bare], references=True)
    assert all(word in str(combined.value) for word in ("sealed", "messy", "bare"))
