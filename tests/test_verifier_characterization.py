"""Characterization of known verifier gaps: these tests pin today's behaviour.

Each test passes against the current code and documents a limitation, not a desired
behaviour. If a gap is closed, the matching test should fail and be rewritten to assert the
new rule. Every executor here is a fake returning canned results; no task, reference or
generated code is ever run.
"""

from dataclasses import replace
from pathlib import Path

from scratchcoder.data.pool import build_pool
from scratchcoder.eval.tasks import Task
from scratchcoder.rl.rewards import Attempt, AttemptReward
from scratchcoder.rl.screening import screen
from scratchcoder.sandbox.reward import degenerate_baseline
from scratchcoder.sandbox.runner import Result
from tools.task_pool import call_key, verify

REFERENCE = "def add_one(value):\n    return value + 1\n"
PROMPT = "Write add_one(value), which returns the value plus one."
PUBLIC = "assert add_one(1) == 2\n"
HASH = "a" * 40


def _task(identifier: str, hidden: str, *, prompt: str = PROMPT) -> Task:
    return Task(
        id=identifier,
        prompt=prompt,
        public_tests=PUBLIC,
        hidden_tests=hidden,
        source="fixture/source",
        license="MIT",
        written_by="human",
        split="train",
        entry_point="add_one",
        solution=REFERENCE,
        source_revision=HASH,
    )


def test_u1_zero_check_hidden_tests_are_admitted() -> None:
    """Characterization of a known gap, not desired behaviour.

    Hidden tests of just ``pass`` contain no check at all. ``Task`` accepts them and
    ``build_pool`` admits the task whenever the executor reports the reference as passing
    (an empty test file always passes). Admission has no minimum-check rule: today a source
    is protected only by the adapter's assert rule (``scratchcoder.data.adapters``), so a
    task that reaches the pool by another route is not caught here.
    """
    task = _task("fixture/zero-check", "pass\n")
    assert task.hidden_tests == "pass\n"
    calls: list[tuple[str, str, str]] = []

    def execute(code: str, tests: str, mode: str) -> Result:
        calls.append((code, tests, mode))
        return Result("passed", "", 0, 0.01)

    kept, report = build_pool([task], [], execute=execute)
    assert [item.id for item in kept] == ["fixture/zero-check"]
    assert report["kept"] == 1
    assert report.get("invalid_solution", 0) == 0
    assert (REFERENCE, "pass\n", "function") in calls


def test_u2_single_run_cache_admits_a_flaky_first_pass(tmp_path: Path) -> None:
    """Characterization of a known gap, not desired behaviour.

    ``verify`` runs each distinct (code, tests, mode) once and caches the verdict. The fake
    executor passes the first run of every key and fails every later run, as a flaky task
    would. The first pass is recorded and admitted, and a second ``verify`` on the same
    cache executes nothing, so the flakiness is never observed.
    """
    runs: dict[str, int] = {}
    calls = [(REFERENCE, PUBLIC, "function"), (REFERENCE, "assert add_one(2) == 3\n", "function")]

    def execute(code: str, tests: str, mode: str) -> Result:
        key = call_key(code, tests, mode)
        runs[key] = runs.get(key, 0) + 1
        return Result("passed" if runs[key] == 1 else "failed", "", 0, 0.01)

    cache = tmp_path / "verdicts.jsonl"

    def deadline(tests: str, mode: str) -> float:
        return 10.0

    first, counters = verify(
        calls, cache, execute=execute, deadline=deadline, workers=1, progress=lambda text: None
    )
    assert counters["executed"] == 2
    assert all(result.passed for result in first.values())
    assert sum(runs.values()) == 2

    second, counters = verify(
        calls, cache, execute=execute, deadline=deadline, workers=1, progress=lambda text: None
    )
    assert counters["executed"] == 0
    assert counters["cached"] == 2
    assert sum(runs.values()) == 2
    assert all(result.passed for result in second.values())


def test_u3_screening_keeps_a_task_whose_rewards_differ_on_identical_programs() -> None:
    """Characterization of a known gap, not desired behaviour.

    The keep rule looks only at whether the 8 screening rewards are all equal. Here every
    attempt has the same program text yet the rewards differ, which can only come from a
    flaky or nondeterministic verifier. ``screen`` keeps the task: no input to the rule
    (it sees rewards and the task, not program equality) could tell this from real skill
    differences between samples.
    """
    task = replace(
        _task("fixture/flaky", "assert add_one(10) == 11\nassert add_one(20) == 21\n"),
        public_tests="assert add_one(1) == 2\n",
    )
    attempts = [Attempt(task.id, index, REFERENCE, 10) for index in range(8)]
    values = (0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5)
    rewards = [AttemptReward(value, "passed" if value == 1.0 else "failed") for value in values]
    assert len({attempt.code for attempt in attempts}) == 1
    result = screen([task], attempts, rewards, floor=1)
    assert result.kept == (task.id,)
    assert result.tasks[0].reason is None


def test_u4a_baseline_misses_a_task_with_candidate_independent_checks() -> None:
    """Characterization of a known gap, not desired behaviour.

    Hidden tests that never call the candidate's function (``abs(-1) == 1``) pass for any
    program. The constant-answer baseline only credits checks whose call is a direct call
    of a candidate-defined function, so it scores 0.0 and screening cannot see that the
    task pays every program in full.
    """
    task = _task("fixture/independent", "assert abs(-1) == 1\nassert len('ab') == 2\n")
    best = degenerate_baseline(task)
    assert best is not None
    assert best.reward == 0.0
    assert best.answer == "none"


def test_u4b_baseline_misses_a_none_returning_method_check() -> None:
    """Characterization of a known gap, not desired behaviour.

    ``Logger().log('a') is None`` is satisfied by a class whose method returns nothing, so
    a trivial class earns full reward. The baseline applies answers only to direct calls of
    a candidate function, so this check counts as failed and the baseline is 0.0.
    """
    task = _task(
        "fixture/none-method",
        "assert Logger().log('a') is None\nassert Logger().log('b') is None\n",
        prompt="Write a Logger class whose log method records a message.",
    )
    best = degenerate_baseline(task)
    assert best is not None
    assert best.reward == 0.0
    assert best.answer == "none"
