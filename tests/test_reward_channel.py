"""Tests for the out-of-process RL reward channel.

The unit tests drive the host comparison, parsing, audit and status classification with fake
containers and a fake ``docker`` binary; they need no Docker and never run candidate code.
The tests at the bottom are gated behind ``SCRATCHCODER_SANDBOX_TESTS=1`` and prove against
the real pinned container that no forgery, malformed report, signal, ``/proc`` write, flood,
over-budget sleep, fork bomb or memory blow-up earns more reward than correct values or
turns a candidate failure into a host failure, and that the I/O fork server agrees with the
evaluation path of record.
"""

import ast
import base64
import dataclasses
import json
import math
import os
import shutil
import stat
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox import budget, reward, reward_program, runner
from scratchcoder.sandbox.reward import (
    COMPLETED,
    FUNCTION_BUDGET_SECONDS,
    HOST_ERROR,
    HOST_TIMEOUT,
    MALFORMED_REPORT,
    NOT_PARSEABLE,
    OVERSIZED,
    RESOURCE_KILL,
    UNAVAILABLE,
    AuditTally,
    Check,
    ContainerOutcome,
    RewardResult,
    attempt_budget,
    audit_attempt,
    budget_for,
    check_passes,
    degenerate_baseline,
    is_parseable,
    parse_function_checks,
    parse_io_cases,
    run_reward_container,
    score_attempt,
    score_with_host_retries,
)
from scratchcoder.sandbox.reward_program import LineSplitter, admit_row
from scratchcoder.sandbox.runner import Limits, command
from tools import task_pool

POOL_SPEC = Path(__file__).resolve().parents[1] / "configs" / "task-pool-rehearsal.json"


def function_task(hidden: str, public: str = "assert callable(f)") -> Task:
    return Task(
        id="t/fn",
        prompt="p",
        public_tests=public,
        hidden_tests=hidden,
        source="s",
        license="MIT",
        written_by="model",
        split="train",
        mode="function",
    )


def io_task(cases: list[dict[str, str]]) -> Task:
    return Task(
        id="t/io",
        prompt="p",
        public_tests=json.dumps(cases[:1]),
        hidden_tests=json.dumps(cases),
        source="s",
        license="MIT",
        written_by="human",
        split="train",
        mode="io",
    )


# --------------------------------------------------------------------------------------
# Parsing.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hidden",
    [
        "assert f(1) == 2",
        "assert f() == [1, 2, 3]",
        "assert f() == {'a': 1, 'b': 2}",
        "assert f() == (1, 2)",
        "assert f() == {1, 2, 3}",
        "assert f() is None",
        "assert f() is True",
        "assert f() is False",
        "assert not f()",
        "assert 7 == f()",
        "assert f() == -3.5",
        "assert f() == set()",
        "import math\nassert f() == 1",
    ],
)
def test_supported_forms_parse(hidden: str) -> None:
    assert parse_function_checks(hidden) is not None
    assert is_parseable(function_task(hidden))


@pytest.mark.parametrize(
    "hidden",
    [
        "assert f() != 2",
        "assert f() < 2",
        "assert f() in [1, 2]",
        "assert 1 <= f() <= 2",
        # A bare ``assert CALL`` has a known passing outcome (truthy), which a candidate could
        # steer through its own objects or literals visible in the call source.
        "assert f()",
        "assert all(x > 0 for x in f())",
        "assert f() == g()",
        "assert f() == object()",
        "assert f().startswith('x')",
        "x = 1\nassert f() == 1",
        "not a test at all (",
        # A comparison inside CALL would carry the expected value into the container.
        "assert (f(1) == 5) is True",
        "assert (f(1) == 5) == True",
        "assert not (f(1) != 5)",
        "assert not f(1) == 5",
        "assert f([x for x in range(3) if x > 1]) == [2]",
        "assert (f(1) and g(1)) == 1",
        # ``is`` accepts only the None/True/False singletons, by identity.
        "assert f() is 0",
        "assert f() is 1",
        "assert f() is 1.0",
        "assert f() is 0.0",
        "assert None is None",
        "assert not 0",
        "assert True",
        "assert f() and g()",
        # Expected values the driver could never report.
        "assert f() == 1j",
        "assert f() == ...",
    ],
)
def test_unsupported_forms_are_not_parseable(hidden: str) -> None:
    assert parse_function_checks(hidden) is None
    assert not is_parseable(function_task(hidden))


def test_imports_go_to_setup_and_assertions_to_checks() -> None:
    parsed = parse_function_checks("import math\nfrom collections import Counter\nassert f() == 1")
    assert parsed is not None
    assert "import math" in parsed.setup
    assert "from collections import Counter" in parsed.setup
    assert parsed.total == 1


def test_expected_values_never_appear_in_the_call_sources() -> None:
    parsed = parse_function_checks("assert f(1) == 424242\nassert 636363 == f(2)")
    assert parsed is not None
    assert [check.call for check in parsed.checks] == ["f(1)", "f(2)"]
    assert all("424242" not in check.call and "636363" not in check.call for check in parsed.checks)


def test_an_expected_value_too_large_for_one_result_row_is_not_parseable() -> None:
    hidden = f"assert f() == {'x' * 70000!r}"
    assert parse_function_checks(hidden) is None
    assert parse_function_checks(hidden, limits=Limits(output_bytes=1024 * 1024)) is not None


def test_what_enters_the_container_must_fit_the_input_limit() -> None:
    calls = "\n".join(f"assert f({'1' * 100}, {index}) == {index}" for index in range(2000))
    assert not is_parseable(function_task(calls))
    big_input = json.dumps([{"input": "x" * 200000, "output": "1"}])
    assert parse_io_cases(big_input) is None
    # The inputs together must fit as well: every one of them enters the container.
    many_inputs = json.dumps([{"input": "x" * 70000, "output": "1"}] * 2)
    assert parse_io_cases(many_inputs) is None
    assert parse_io_cases(many_inputs, limits=Limits(input_bytes=150000)) is not None
    # Expected outputs stay on the host, so many large ones are fine.
    big_outputs = json.dumps([{"input": "", "output": "y" * 60000}] * 10)
    assert parse_io_cases(big_outputs) is not None


def test_well_formed_io_tasks_are_parseable() -> None:
    assert is_parseable(io_task([{"input": "1", "output": "2"}]))
    parsed = parse_io_cases(
        json.dumps([{"input": "1", "output": "2"}, {"input": "", "output": ""}])
    )
    assert parsed is not None and parsed.total == 2


@pytest.mark.parametrize(
    "hidden",
    [
        "not json",
        json.dumps({"input": "1", "output": "2"}),
        json.dumps([]),
        json.dumps([{"input": "1"}]),
        json.dumps([{"input": 1, "output": "2"}]),
        json.dumps([{"input": "1", "output": "2"}] * 101),
        # An expected output beyond the per-case output cap can never be matched.
        json.dumps([{"input": "", "output": "x" * 70000}]),
    ],
)
def test_malformed_or_unmatchable_io_tasks_are_not_parseable(hidden: str) -> None:
    assert parse_io_cases(hidden) is None
    task = io_task([{"input": "1", "output": "2"}])
    try:
        candidate = dataclasses.replace(task, hidden_tests=hidden)
    except ValueError:
        return  # the task loader already rejects this shape
    assert not is_parseable(candidate)


def test_io_output_cap_follows_the_limits() -> None:
    hidden = json.dumps([{"input": "", "output": "x" * 70000}])
    assert parse_io_cases(hidden, limits=Limits(output_bytes=1024 * 1024)) is not None


# --------------------------------------------------------------------------------------
# Attempt budgets.
# --------------------------------------------------------------------------------------


def pool_verification() -> dict[str, Any]:
    verification: dict[str, Any] = json.loads(POOL_SPEC.read_text(encoding="utf-8"))["verification"]
    return verification


def test_the_shared_rule_holds_the_pool_specs_values() -> None:
    verification = pool_verification()
    assert verification["timeout_seconds"] == budget.IO_BASE_SECONDS == 2.0
    assert verification["io_seconds_per_case"] == budget.IO_SECONDS_PER_CASE == 0.25
    assert budget.io_seconds(0) == 2.0
    assert budget.io_seconds(24) == 8.0
    assert budget.io_seconds(100) == 27.0
    assert budget.io_seconds(232) == 60.0
    assert budget.io_seconds(4, base_seconds=1.0, seconds_per_case=0.5) == 3.0
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            budget.io_seconds(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("count", range(1, 101))
def test_pool_verification_and_reward_channel_give_an_io_task_the_same_deadline(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = io_task([{"input": f"{n}\n", "output": str(n)} for n in range(count)])
    pool_limits = task_pool.limits_for(pool_verification(), task.hidden_tests, "io")
    assert budget_for(task) == pool_limits.timeout_seconds
    seen: list[tuple[dict[str, Any], Limits]] = []

    def fake_container(payload: dict[str, Any], *, limits: Limits) -> ContainerOutcome:
        seen.append((payload, limits))
        return ContainerOutcome(HOST_TIMEOUT, None)

    monkeypatch.setattr(reward, "run_reward_container", fake_container)
    result = score_attempt(task, "print(input())")
    payload, limits = seen[0]
    # The same deadline, and the same container limits (CPU seconds and memory) as the pool's.
    assert payload["budget"] == result.budget_seconds == pool_limits.timeout_seconds
    assert limits == pool_limits
    # Every case may use what the budget has left, as in the pool verification's I/O program.
    assert payload["per_case"] is None


def test_function_tasks_keep_the_eight_second_budget() -> None:
    seen: list[dict[str, Any]] = []

    def capture(payload: dict[str, Any]) -> ContainerOutcome:
        seen.append(payload)
        return ContainerOutcome(COMPLETED, function_report([row(0, 1)]))

    task = function_task("assert f() == 1")
    result = score_attempt(task, "c", container=capture)
    assert budget_for(task) == FUNCTION_BUDGET_SECONDS == 8.0
    assert result.budget_seconds == seen[0]["budget"] == 8.0
    assert seen[0]["per_call"] == reward.PER_CALL_SECONDS == 4.0
    parsed = parse_function_checks("assert f() == 1")
    assert parsed is not None and attempt_budget(parsed) == 8.0


def test_explicit_budgets_are_used_and_recorded() -> None:
    seen: list[dict[str, Any]] = []

    def capture(payload: dict[str, Any]) -> ContainerOutcome:
        seen.append(payload)
        return ContainerOutcome(UNAVAILABLE, None)

    task = io_task([{"input": "1", "output": "2"}])
    result = score_attempt(task, "c", budget=3.5, per_case=1.0, container=capture)
    assert seen[0]["budget"] == result.budget_seconds == 3.5 and seen[0]["per_case"] == 1.0
    unparseable = score_attempt(function_task("assert f() != 1"), "c", container=capture)
    assert unparseable.budget_seconds is None and len(seen) == 1
    assert budget_for(function_task("assert f() != 1")) is None


# --------------------------------------------------------------------------------------
# Fake-container host-comparison logic.
# --------------------------------------------------------------------------------------


def wire(value: object) -> object:
    """Encode a plain value exactly as the honest driver would, for fakes."""
    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        return ["float", value]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", list(value)]
    if isinstance(value, list):
        return ["list", [wire(item) for item in value]]
    if isinstance(value, tuple):
        return ["tuple", [wire(item) for item in value]]
    if isinstance(value, set):
        return ["set", [wire(item) for item in value]]
    if isinstance(value, frozenset):
        return ["frozenset", [wire(item) for item in value]]
    if isinstance(value, dict):
        return ["dict", [[wire(k), wire(v)] for k, v in value.items()]]
    raise TypeError("unsupported in fake")


def b64(data: bytes | str) -> str:
    return base64.b64encode(data.encode() if isinstance(data, str) else data).decode()


def row(index: int, value: object) -> str:
    return b64(json.dumps({"i": index, "ok": True, "v": wire(value)}))


def function_report(rows: list[Any], **extra: Any) -> dict[str, Any]:
    return {
        "status": "ran",
        "mode": "function",
        "n": len(rows),
        "rows": rows,
        "child_rc": 0,
        "output_limit": False,
        **extra,
    }


def report_container(
    report: dict[str, Any], *, exit_code: int | None = 0
) -> Callable[[dict[str, Any]], ContainerOutcome]:
    return lambda payload: ContainerOutcome(COMPLETED, report, exit_code)


def function_container(values: list[object]) -> Callable[[dict[str, Any]], ContainerOutcome]:
    """A fake container reporting ``values[i]`` for call ``i`` (an exception value = an error)."""
    rows: list[Any] = []
    for index, item in enumerate(values):
        if isinstance(item, BaseException):
            rows.append(b64(json.dumps({"i": index, "ok": False, "e": "boom"})))
        else:
            rows.append(row(index, item))
    return report_container(function_report(rows))


def io_container(stdouts: list[str | None]) -> Callable[[dict[str, Any]], ContainerOutcome]:
    cases: list[dict[str, Any]] = []
    for text in stdouts:
        if text is None:
            cases.append({"ran": False, "e": "case_timeout"})
        else:
            cases.append({"ran": True, "rc": 0, "out": b64(text)})
    return report_container({"status": "ran", "mode": "io", "n": len(cases), "cases": cases})


def test_correct_values_earn_full_reward() -> None:
    task = function_task("assert f() == 10\nassert f() == 10")
    result = score_attempt(task, "code", container=function_container([10, 10]))
    assert result.status == COMPLETED
    assert result.reward == 1.0
    assert (result.passed, result.total) == (2, 2)
    assert result.outcomes == (True, True)


def test_partial_values_earn_the_right_fraction() -> None:
    task = function_task("assert f() == 1\nassert f() == 2\nassert f() == 3\nassert f() == 4")
    result = score_attempt(task, "code", container=function_container([1, 2, 99, 4]))
    assert result.passed == 3 and result.total == 4
    assert result.reward == 0.75
    assert result.outcomes == (True, True, False, True)


def test_denominator_comes_from_the_task_file_not_the_container() -> None:
    task = function_task("assert f() == 1\nassert f() == 2")
    # More rows than the task has checks, or fewer, is a malformed report: nothing passes.
    longer = function_report([row(0, 1), row(1, 2), row(2, 2)])
    shorter = function_report([row(0, 1)])
    for report in (longer, shorter):
        result = score_attempt(task, "code", container=report_container(report))
        assert result.total == 2 and result.passed == 0 and result.reward == 0.0


def test_a_row_must_carry_its_own_index() -> None:
    task = function_task("assert f() == 1\nassert f() == 2")
    result = score_attempt(
        task, "code", container=report_container(function_report([row(1, 1), row(1, 2)]))
    )
    assert result.outcomes == (False, True)


def reward_for(hidden: str, value: object) -> float | None:
    return score_attempt(function_task(hidden), "c", container=function_container([value])).reward


# (expected literal, reported built-in value) pairs, including numeric-tower, NaN, signed
# zero, infinity, container-type and hash-equal key cases.
EQUALITY_CASES: list[tuple[str, object]] = [
    ("2", 2.0),
    ("2", 2),
    ("2.0", 2),
    ("1", True),
    ("0", False),
    ("True", 1),
    ("True", 1.0),
    ("False", 0),
    ("1", 1.0000000000000002),
    ("100000000000000000000", 1e20),
    ("0.30000000000000004", 0.1 + 0.2),
    ("0.3", 0.1 + 0.2),
    ("0.0", -0.0),
    ("-0.0", 0),
    ("1e999", math.inf),
    ("-1e999", -math.inf),
    ("1e999", math.nan),
    ("0.0", math.nan),
    ("[1.0]", [math.nan]),
    ("(1, 2)", [1, 2]),
    ("[1, 2]", [1, 2]),
    ("[1, 2]", [1.0, 2.0]),
    ("[1, 2]", [2, 1]),
    ("{1, 2, 3}", {3, 1, 2}),
    ("{1, 2}", frozenset({1.0, 2})),
    ("{0.0}", {-0.0}),
    ("{'b': 2, 'a': 1}", {"a": 1, "b": 2}),
    ("{1: 'a'}", {1.0: "a"}),
    ("{1: 'a'}", {True: "a"}),
    ("{'a': [1, (2, 3)]}", {"a": [1.0, (2, 3.0)]}),
    ("'a'", b"a"),
    ("b'a'", b"a"),
    ("None", 0),
    ("[]", ()),
    ("set()", {}),
    ("{}", {}),
]


@pytest.mark.parametrize("literal,value", EQUALITY_CASES)
def test_equality_is_decided_as_the_assertion_itself_would(literal: str, value: object) -> None:
    # The reference: Python's own ``==`` between the same built-in values, which is what the
    # evaluation runner's in-process ``assert CALL == LITERAL`` computes for them.
    expected = ast.literal_eval(literal)
    for hidden in (f"assert f() == {literal}", f"assert {literal} == f()"):
        assert reward_for(hidden, value) == (1.0 if value == expected else 0.0), hidden


@pytest.mark.parametrize(
    "hidden,value,passes",
    [
        # Identity forms stay identity: an equal value that is not the singleton fails.
        ("assert f() is True", True, True),
        ("assert f() is True", 1, False),
        ("assert f() is True", 1.0, False),
        ("assert f() is False", False, True),
        ("assert f() is False", 0, False),
        ("assert f() is None", None, True),
        ("assert f() is None", 0, False),
        ("assert f() is None", [], False),
        ("assert None is f()", None, True),
        # Truthiness for negated calls.
        ("assert not f()", [], True),
        ("assert not f()", "", True),
        ("assert not f()", 0.0, True),
        ("assert not f()", -0.0, True),
        ("assert not f()", [1], False),
        ("assert not f()", math.nan, False),
        ("assert not f()", None, True),
    ],
)
def test_identity_and_truthiness_forms(hidden: str, value: object, passes: bool) -> None:
    assert reward_for(hidden, value) == (1.0 if passes else 0.0)


def test_reconstruction_yields_only_built_in_types() -> None:
    reported: dict[str, object] = {"ok": True}
    wire_value = [
        "dict",
        [[["tuple", [["int", 1], ["float_s", "nan"]]], ["set", [["bytes", [97]]]]]],
    ]
    value = reward._value_from_wire(wire_value)
    seen: list[type] = []

    def walk(item: object) -> None:
        seen.append(type(item))
        if isinstance(item, dict):
            for key, inner in item.items():
                walk(key)
                walk(inner)
        elif isinstance(item, (list, tuple, set, frozenset)):
            for inner in item:
                walk(inner)

    walk(value)
    assert set(seen) <= {dict, tuple, int, float, set, bytes}
    # A set element or dict key Python could not hash is a malformed report, not a crash.
    for bad in (["set", [["list", []]]], ["dict", [[["list", []], ["int", 1]]]]):
        reported["v"] = bad
        assert not check_passes(Check("f()", "eq", set()), reported)


def test_unserialisable_and_errored_calls_fail() -> None:
    # An error sentinel (what an always-equal object or crash produces) never passes.
    task = function_task("assert f() == 1\nassert f() == 2")
    result = score_attempt(task, "c", container=function_container([1, ValueError()]))
    assert result.passed == 1 and result.reward == 0.5


def test_unreported_and_timed_out_calls_fail_and_count_as_unfinished() -> None:
    task = function_task("assert f() == 1\nassert f() == 2\nassert f() == 3")
    timed_out = b64(json.dumps({"i": 1, "ok": False, "e": "call_timeout"}))
    report = function_report([row(0, 1), timed_out, None])
    result = score_attempt(task, "c", container=report_container(report))
    assert result.passed == 1 and result.total == 3
    assert result.unfinished == 2


def test_forged_wrong_values_cannot_beat_correct() -> None:
    # The host compares against the secret expected value; a blind forge of a wrong value fails.
    task = function_task("assert f() == 424242")
    forged = score_attempt(task, "c", container=function_container([111111])).reward
    correct = score_attempt(task, "c", container=function_container([424242])).reward
    assert forged == 0.0 and correct == 1.0


DEEP: list[Any] = ["list", []]
for _ in range(100):
    DEEP = ["list", [DEEP]]


@pytest.mark.parametrize(
    "value",
    [
        ["int"],
        ["int", 1, 2],
        ["int", True],
        ["int", 1.0],
        ["bool", 1],
        ["float", 1],
        ["float_s", "x"],
        ["float_s", 1],
        ["str", 5],
        ["bytes", 10**20],
        ["bytes", 10**8],
        ["bytes", [256]],
        ["bytes", [-1]],
        ["bytes", "ab"],
        ["list", 5],
        ["list", [["int"]]],
        ["dict", [[["int", 1]]]],
        ["dict", [5]],
        ["unknown", 1],
        [1, 2],
        [],
        "int",
        None,
        5,
        {"k": 1},
        DEEP,
    ],
)
def test_malformed_wire_values_fail_without_raising(value: object) -> None:
    # The candidate can write any row to the result pipe; decoding it must never crash or
    # allocate on the host, and never pass.
    for hidden in ("assert f() == 1", "assert not f()"):
        report = function_report([b64(json.dumps({"i": 0, "ok": True, "v": value}))])
        result = score_attempt(function_task(hidden), "c", container=report_container(report))
        assert result.status == COMPLETED and result.reward == 0.0


@pytest.mark.parametrize(
    "raw",
    [
        "!!! not base64 !!!",
        b64(b"[" * 100000),
        b64(b"\xff\xfe"),
        b64(json.dumps([0, 1])),
        5,
        ["nested"],
    ],
)
def test_malformed_rows_fail_without_raising(raw: object) -> None:
    report = function_report([raw])
    result = score_attempt(
        function_task("assert f() == 1"), "c", container=report_container(report)
    )
    assert result.status == COMPLETED and result.reward == 0.0


@pytest.mark.parametrize("reported", [None, [], "x", {"ok": 1, "v": ["int", 1]}, {"ok": True}])
def test_check_passes_rejects_malformed_reports(reported: object) -> None:
    assert not check_passes(Check("f()", "eq", 1), reported)


def test_io_fraction_and_unrun() -> None:
    task = io_task(
        [
            {"input": "1", "output": "2"},
            {"input": "2", "output": "3"},
            {"input": "3", "output": "4"},
        ]
    )
    result = score_attempt(task, "c", container=io_container(["2\n", "wrong", None]))
    assert result.passed == 1 and result.total == 3
    assert result.unfinished == 1
    assert result.outcomes == (True, False, False)


@pytest.mark.parametrize(
    "case",
    [
        {"ran": True, "out": "!!!"},
        {"ran": True, "out": 5},
        {"ran": 1, "out": b64("2")},
        {"ran": False, "out": b64("2")},
        "not a dict",
    ],
)
def test_malformed_io_rows_fail(case: object) -> None:
    report = {"status": "ran", "mode": "io", "n": 1, "cases": [case]}
    task = io_task([{"input": "1", "output": "2"}])
    result = score_attempt(task, "c", container=report_container(report))
    assert result.status == COMPLETED and result.reward == 0.0


def test_io_stdout_is_compared_after_strip_with_replacement_decoding() -> None:
    task = io_task([{"input": "", "output": "\ufffd"}, {"input": "", "output": " a b "}])
    report = {
        "status": "ran",
        "mode": "io",
        "n": 2,
        "cases": [{"ran": True, "out": b64(b"\xff\n")}, {"ran": True, "out": b64("a b\n\n")}],
    }
    assert score_attempt(task, "c", container=report_container(report)).reward == 1.0


# --------------------------------------------------------------------------------------
# Host statuses and retries.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [HOST_TIMEOUT, UNAVAILABLE, HOST_ERROR])
def test_host_failures_never_score_a_reward(status: str) -> None:
    task = function_task("assert f() == 1")
    result = score_attempt(
        task, "c", container=lambda payload: ContainerOutcome(status, None, elapsed_seconds=1.0)
    )
    assert result.status == status
    assert result.reward is None
    assert result.is_host_failure


@pytest.mark.parametrize("status", [RESOURCE_KILL, MALFORMED_REPORT])
def test_candidate_attributed_failures_score_zero_not_a_host_failure(status: str) -> None:
    task = function_task("assert f() == 1")
    result = score_attempt(task, "c", container=lambda payload: ContainerOutcome(status, None))
    assert result.status == status
    assert result.reward == 0.0
    assert not result.is_host_failure


def test_a_completed_report_without_the_ran_status_is_malformed() -> None:
    result = score_attempt(
        function_task("assert f() == 1"),
        "c",
        container=lambda payload: ContainerOutcome(COMPLETED, None),
    )
    assert result.status == MALFORMED_REPORT and result.reward == 0.0


def test_out_of_memory_evidence_labels_a_resource_kill_with_the_partial_reward() -> None:
    task = function_task("assert f() == 1\nassert f() == 2")
    rows = [row(0, 1), None]
    by_events = score_attempt(
        task, "c", container=report_container(function_report(rows, oom_kills=1))
    )
    assert by_events.status == RESOURCE_KILL and by_events.reward == 0.5
    # With a complete report, only in-container evidence labels a resource kill.
    late_kill = score_attempt(
        task, "c", container=report_container(function_report(rows), exit_code=137)
    )
    assert late_kill.status == COMPLETED and late_kill.reward == 0.5
    clean = score_attempt(task, "c", container=report_container(function_report(rows, oom_kills=0)))
    assert clean.status == COMPLETED


def test_not_parseable_returns_none_reward_without_running() -> None:
    ran = False

    def guard(payload: dict[str, Any]) -> ContainerOutcome:
        nonlocal ran
        ran = True
        return ContainerOutcome(COMPLETED, function_report([]))

    result = score_attempt(function_task("assert f() != 1"), "c", container=guard)
    assert result.status == NOT_PARSEABLE and result.reward is None
    assert not ran


def test_process_level_failures_fail_every_function_check() -> None:
    # One process runs the candidate and every call, as in the evaluation program, where
    # output beyond the cap or a nonzero exit fails the whole run.
    task = function_task("assert f() == 1\nassert f() == 2")
    rows = [row(0, 1), row(1, 2)]

    def scored(**extra: Any) -> RewardResult:
        return score_attempt(task, "c", container=report_container(function_report(rows, **extra)))

    assert scored().reward == 1.0
    flood = scored(output_limit=True)
    assert flood.status == COMPLETED and flood.reward == 0.0 and flood.detail == "output limit"
    for status in (1, -9, None, "0"):
        exited = scored(child_rc=status)
        assert exited.reward == 0.0 and exited.detail == f"exit status {status}", status
    # A report without the output flag is not trusted to have checked it.
    unflagged = function_report(rows)
    del unflagged["output_limit"]
    assert score_attempt(task, "c", container=report_container(unflagged)).reward == 0.0
    # A process the budget ended before every call reported keeps the calls that did.
    cut = function_report([row(0, 1), None], child_rc=-9)
    partial = score_attempt(task, "c", container=report_container(cut))
    assert partial.reward == 0.5 and partial.unfinished == 1 and partial.detail == ""


def test_an_oversized_attempt_scores_zero_without_starting_a_container() -> None:
    ran = False

    def guard(payload: dict[str, Any]) -> ContainerOutcome:
        nonlocal ran
        ran = True
        return ContainerOutcome(COMPLETED, function_report([row(0, 1)]))

    limit = Limits().input_bytes
    task = function_task("assert f() == 1")
    huge = "def f():\n    return 1\n#" + "x" * limit
    result = score_attempt(task, huge, container=guard)
    assert not ran
    assert result.status == OVERSIZED and result.reward == 0.0 and not result.is_host_failure
    assert result.total == 1 and result.outcomes == (False,) and result.audit_flags == ()
    # The call sources (or case inputs) count toward the limit with the code.
    calls = function_task("assert f('" + "y" * (limit // 2) + "') == 1")
    assert score_attempt(calls, "z" * (limit // 2), container=guard).status == OVERSIZED
    cases = io_task([{"input": "y" * (limit // 2), "output": "1"}])
    assert score_attempt(cases, "z" * (limit // 2 + 1), container=guard).status == OVERSIZED
    assert not ran
    # Just inside the limit still runs.
    assert score_attempt(task, "def f():\n    return 1", container=guard).reward == 1.0
    # A host failure is never reported for size: retries do not run it again.
    retried = score_with_host_retries(task, huge, sleep=lambda _: None, container=guard)
    assert retried.status == OVERSIZED and retried.reward == 0.0


def test_the_supervisor_reports_spawn_failures_only_before_any_candidate_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nothing here can start a process: the supervisor's spawning points are replaced.
    def no_memory(data: bytes) -> int:
        raise BlockingIOError("EAGAIN")

    monkeypatch.setattr(reward_program, "_memory_file", no_memory)
    function_payload = {
        "mode": "function",
        "code": "c",
        "setup": "",
        "calls": ["f()"],
        "per_call": 1.0,
        "budget": 1.0,
        "output_bytes": 65536,
        "row_bytes": 65536,
        "site_packages": False,
    }
    report = reward_program.run_report(function_payload)
    assert report == {"status": "spawn_failed", "error_type": "BlockingIOError"}
    # A malformed payload is the host's own fault too, and nothing of the candidate ran.
    assert reward_program.run_report({"mode": "function"})["status"] == "spawn_failed"

    class NoForkServer:
        def __init__(self, payload: dict[str, Any]) -> None:
            raise OSError("ENOMEM")

    io_payload = {
        "mode": "io",
        "code": "c",
        "cases": ["1", "2"],
        "per_case": None,
        "budget": 1.0,
        "output_bytes": 65536,
        "site_packages": False,
    }
    monkeypatch.setattr(reward_program, "_Zygote", NoForkServer)
    assert reward_program.run_report(io_payload)["status"] == "spawn_failed"

    class BrokenAfterFirstRequest:
        built = 0

        def __init__(self, payload: dict[str, Any]) -> None:
            BrokenAfterFirstRequest.built += 1
            if BrokenAfterFirstRequest.built > 1:
                raise OSError("ENOMEM")
            self.pid = 0

        def run_case(self, text: str, cap: int, case_deadline: float) -> dict[str, Any]:
            # The request reached the fork server, so a candidate may have run.
            reward_program._mark_candidate_started()
            raise ConnectionError("fork server exited")

        def close(self) -> None:
            pass

    monkeypatch.setattr(reward_program, "_Zygote", BrokenAfterFirstRequest)
    report = reward_program.run_report(io_payload)
    # Once a candidate may have run, a failed fork server only fails the cases: scored.
    assert report["status"] == "ran"
    assert report["cases"] == [{"ran": False, "e": "runner"}, {"ran": False, "e": "runner"}]


def test_the_supervisors_own_overrun_stays_far_inside_the_host_slack() -> None:
    # The supervisor's longest wait past the budget, plus several seconds for the container
    # to start and stop under emulation, is still short of the host's stall deadline.
    assert reward_program.OVERRUN_SECONDS == 7.0
    assert reward_program.OVERRUN_SECONDS + 5.0 <= reward.HOST_SLACK_SECONDS


# --------------------------------------------------------------------------------------
# Degenerate-answer baseline (host side; runs no code).
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hidden,public,reward_value,answer",
    [
        ("assert f(1) is None\nassert f(2) is None\nassert f(3) == 4", "", 2 / 3, "constant None"),
        ("assert not f(1)\nassert f(2) == 0\nassert f(3) == []", "", 2 / 3, "constant False"),
        ("assert f(1) == 7\nassert f(2) == 8", "assert f(0) == 7", 0.5, "public example 7"),
        ("assert f([1]) == [1]\nassert f([2, 1]) == [1, 2]", "", 0.5, "echo"),
        ("assert f(1) == 2.0\nassert f(2) == 2", "assert f(5) == 2", 1.0, "public example 2"),
        # Only direct calls of a candidate function take the answer: ``len(f(3))`` needs the
        # candidate's value, and ``floor`` is the tests' own import.
        ("assert len(f(3)) == 0\nassert f(1) == -1", "", 0.5, "constant -1"),
        (
            "from math import floor\nassert floor(0.5) == 0\nassert f() == ''",
            "",
            0.5,
            "constant ''",
        ),
        ("assert Stack().pop() is None", "", 0.0, "none"),
    ],
)
def test_degenerate_baseline_for_function_tasks(
    hidden: str, public: str, reward_value: float, answer: str
) -> None:
    baseline = degenerate_baseline(function_task(hidden, public=public or "assert True"))
    assert baseline is not None
    assert baseline.reward == pytest.approx(reward_value) and baseline.answer == answer
    assert baseline.total == hidden.count("assert")


def test_degenerate_baseline_for_io_tasks() -> None:
    cases = [{"input": "3\n", "output": "YES\n"}, {"input": "1\n", "output": "YES"}]
    cases += [{"input": "2\n", "output": "NO"}]
    yes = degenerate_baseline(io_task(cases))
    assert yes is not None and (yes.passed, yes.total, yes.answer) == (2, 3, "constant 'YES'")
    # The public example's output (the first case here) is known to the policy.
    shown = [{"input": "9\n", "output": "42"}] * 3 + [{"input": "8\n", "output": "41"}]
    copied = degenerate_baseline(io_task(shown))
    assert copied is not None and copied.reward == 0.75 and copied.answer == "public example '42'"
    echo = [{"input": "x\n", "output": "y"}, {"input": "abc\n", "output": "abc"}]
    echo += [{"input": "q\n", "output": "q"}]
    echoed = degenerate_baseline(io_task(echo))
    assert echoed is not None and echoed.answer == "echo" and echoed.passed == 2
    hidden = io_task([{"input": "1\n", "output": "424242"}])
    unshown = dataclasses.replace(hidden, public_tests=json.dumps([{"input": "", "output": "7"}]))
    solved = degenerate_baseline(unshown)
    assert solved is not None and solved.reward == 0.0 and solved.answer == "none"
    assert degenerate_baseline(function_task("assert f() != 1")) is None


def test_the_payload_carries_no_expected_values() -> None:
    seen: list[dict[str, Any]] = []

    def capture(payload: dict[str, Any]) -> ContainerOutcome:
        seen.append(payload)
        return ContainerOutcome(COMPLETED, function_report([None]))

    score_attempt(function_task("assert f(1) == 424242"), "c", container=capture)
    score_attempt(io_task([{"input": "7", "output": "636363"}]), "c", container=capture)
    assert "424242" not in json.dumps(seen[0]) and "636363" not in json.dumps(seen[1])


def test_retries_recover_a_transient_host_failure() -> None:
    calls = {"n": 0}

    def flaky(payload: dict[str, Any]) -> ContainerOutcome:
        calls["n"] += 1
        if calls["n"] < 3:
            return ContainerOutcome(HOST_TIMEOUT, None)
        return ContainerOutcome(COMPLETED, function_report([row(0, 1)]))

    result = score_with_host_retries(
        function_task("assert f() == 1"), "c", attempts=3, sleep=lambda _: None, container=flaky
    )
    assert result.status == COMPLETED and result.reward == 1.0 and calls["n"] == 3


def test_retries_give_up_with_a_host_failure_never_zero() -> None:
    result = score_with_host_retries(
        function_task("assert f() == 1"),
        "c",
        attempts=3,
        sleep=lambda _: None,
        container=lambda payload: ContainerOutcome(UNAVAILABLE, None),
    )
    assert result.reward is None and result.is_host_failure


def test_candidate_attributed_failures_are_not_retried() -> None:
    calls = {"n": 0}

    def disrupted(payload: dict[str, Any]) -> ContainerOutcome:
        calls["n"] += 1
        return ContainerOutcome(MALFORMED_REPORT, None)

    result = score_with_host_retries(
        function_task("assert f() == 1"), "c", sleep=lambda _: None, container=disrupted
    )
    assert result.reward == 0.0 and calls["n"] == 1


# --------------------------------------------------------------------------------------
# Container classification with a fake docker binary (no Docker, no candidate code).
# --------------------------------------------------------------------------------------

FAKE_DOCKER = r"""
import base64, json, os, sys, time

args = sys.argv[1:]
if "rm" in args:
    with open(os.environ["FAKE_DOCKER_LOG"], "a") as handle:
        handle.write(" ".join(args) + "\n")
    sys.exit(0)
mode = os.environ["FAKE_DOCKER_MODE"]
payload = json.load(sys.stdin)
nonce = payload.get("nonce", "")
row = json.dumps({"i": 0, "ok": True, "v": ["int", 1]}).encode()
report = {
    "status": "ran",
    "mode": "function",
    "n": 1,
    "rows": [base64.b64encode(row).decode()],
    "child_rc": 0,
    "output_limit": False,
}


def frame(kind, **extra):
    sys.stdout.write(json.dumps({"frame": kind, "nonce": nonce, **extra}) + "\n")
    sys.stdout.flush()


if mode == "stall_silent":
    time.sleep(60)
elif mode == "stall_partial":
    sys.stdout.write("partial output without a newline")
    sys.stdout.flush()
    time.sleep(60)
elif mode == "stall_after_started":
    frame("started")
    time.sleep(60)
elif mode == "exit_125":
    print("docker: Error response from daemon: No such image")
    sys.exit(125)
elif mode == "daemon_down":
    print("failed to connect to the docker API at unix:///var/run/docker.sock")
    sys.exit(1)
elif mode == "exit_1":
    print("something else went wrong")
    sys.exit(1)
elif mode == "exit_137_before_start":
    sys.exit(137)
elif mode == "exit_137_after_start":
    frame("started")
    sys.exit(137)
elif mode == "junk_after_start":
    frame("started")
    print("{{{ not a frame")
    sys.exit(0)
elif mode == "supervisor_error":
    frame("started")
    frame("final", report={"status": "supervisor_error", "error_type": "OSError"})
    sys.exit(0)
elif mode == "spawn_failed":
    frame("started")
    frame("final", report={"status": "spawn_failed", "error_type": "BlockingIOError"})
    sys.exit(0)
elif mode == "no_started_frame":
    print(json.dumps(report))
    sys.exit(0)
elif mode == "wrong_nonce":
    nonce = "0" * 32
    frame("started")
    frame("final", report=report)
    sys.exit(0)
elif mode == "final_without_start":
    frame("final", report=report)
    sys.exit(0)
elif mode == "rejected":
    print(json.dumps({"frame": "rejected", "error_type": "KeyError"}))
    sys.exit(0)
elif mode == "valid_with_noise":
    print("noise before the frames")
    frame("started")
    print("x" * 300000)
    print('{"frame": "final", "nonce": "forged", "report": {}}')
    frame("final", report=report)
    sys.exit(0)
elif mode == "valid_then_137":
    frame("started")
    frame("final", report=report)
    sys.exit(137)
"""


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], str]:
    script = tmp_path / "fake_docker.py"
    script.write_text(FAKE_DOCKER, encoding="utf-8")
    wrapper = tmp_path / "docker"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(tmp_path / "rm.log"))
    monkeypatch.setattr(reward, "HOST_SLACK_SECONDS", 0.6)

    def use(mode: str) -> str:
        monkeypatch.setenv("FAKE_DOCKER_MODE", mode)
        return str(wrapper)

    return use


def fake_payload() -> dict[str, Any]:
    return {"mode": "function", "calls": ["f()"], "row_bytes": 65536, "budget": 0.2}


@pytest.mark.parametrize(
    "mode,status",
    [
        ("stall_silent", HOST_TIMEOUT),
        ("stall_partial", HOST_TIMEOUT),
        ("stall_after_started", HOST_TIMEOUT),
        ("exit_125", UNAVAILABLE),
        ("daemon_down", UNAVAILABLE),
        ("exit_1", HOST_ERROR),
        ("exit_137_before_start", HOST_ERROR),
        ("no_started_frame", HOST_ERROR),
        ("wrong_nonce", HOST_ERROR),
        ("final_without_start", HOST_ERROR),
        ("rejected", HOST_ERROR),
        ("exit_137_after_start", RESOURCE_KILL),
        ("junk_after_start", MALFORMED_REPORT),
        ("supervisor_error", MALFORMED_REPORT),
        # The supervisor could not start the candidate's first process: nothing ran.
        ("spawn_failed", HOST_ERROR),
        ("valid_with_noise", COMPLETED),
        ("valid_then_137", COMPLETED),
    ],
)
def test_container_outcomes_are_classified_from_the_framing(
    fake_docker: Callable[[str], str], tmp_path: Path, mode: str, status: str
) -> None:
    started = time.monotonic()
    outcome = run_reward_container(fake_payload(), limits=Limits(), docker=fake_docker(mode))
    assert outcome.host_status == status
    # A stalled CLI is killed at the deadline (0.2 s budget + 0.6 s slack), never waited out.
    assert time.monotonic() - started < 10
    assert (outcome.report is not None) == (status == COMPLETED)
    # Every invocation removes its own container.
    log = (tmp_path / "rm.log").read_text(encoding="utf-8")
    assert "rm -f scratchcoder-reward-" in log


def test_fake_docker_end_to_end_scoring(fake_docker: Callable[[str], str]) -> None:
    def container(mode: str) -> Callable[[dict[str, Any]], ContainerOutcome]:
        binary = fake_docker(mode)
        return lambda payload: run_reward_container(payload, limits=Limits(), docker=binary)

    task = function_task("assert f() == 1")
    assert score_attempt(task, "c", container=container("valid_with_noise")).reward == 1.0
    late_kill = score_attempt(task, "c", container=container("valid_then_137"))
    assert late_kill.status == COMPLETED and late_kill.reward == 1.0
    lost = score_attempt(task, "c", container=container("exit_137_after_start"))
    assert lost.status == RESOURCE_KILL and lost.reward == 0.0
    junk = score_attempt(task, "c", container=container("junk_after_start"))
    assert junk.status == MALFORMED_REPORT and junk.reward == 0.0
    stalled = score_attempt(task, "c", budget=0.2, container=container("stall_silent"))
    assert stalled.status == HOST_TIMEOUT and stalled.reward is None


def test_the_host_stall_deadline_follows_the_budget(fake_docker: Callable[[str], str]) -> None:
    payload = {**fake_payload(), "budget": 1.0}
    started = time.monotonic()
    outcome = run_reward_container(payload, limits=Limits(), docker=fake_docker("stall_silent"))
    elapsed = time.monotonic() - started
    assert outcome.host_status == HOST_TIMEOUT
    # Declared at the budget plus the (shortened) slack of 0.6 s, not before.
    assert 1.6 <= outcome.elapsed_seconds and elapsed < 10


def test_a_missing_docker_binary_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    outcome = run_reward_container(fake_payload(), limits=Limits())
    assert outcome.host_status == UNAVAILABLE


def test_the_reward_command_keeps_every_isolation_flag() -> None:
    default = command("scratchcoder-x", Limits(), docker="docker")
    rewarded = command("scratchcoder-x", Limits(), docker="docker", supervisor="SUPERVISOR")
    assert rewarded[:-1] == default[:-1]
    assert rewarded[-1] == "SUPERVISOR"
    assert default[-1] == Path(runner.__file__).with_name("program.py").read_text(encoding="utf-8")


def test_the_supervisor_sources_compile() -> None:
    compile(reward.REWARD_SUPERVISOR, "<reward_program>", "exec")
    compile(reward_program.DRIVER, "<driver>", "exec")
    compile(reward_program.ZYGOTE, "<zygote>", "exec")


def test_the_cleanup_sweep_only_ever_runs_as_the_container_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``kill`` is replaced by a recorder, so even a broken guard could not signal anything.
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(os, "waitpid", lambda pid, flags: (0, 0))
    reward_program._kill_others(set())
    reward_program._kill_others({12345})
    if os.getpid() != 1:
        assert killed == []


# --------------------------------------------------------------------------------------
# The evaluation runner's host watchdog.
# --------------------------------------------------------------------------------------

FAKE_RUNNER_DOCKER = r"""
import json, os, sys, time

if "rm" in sys.argv[1:]:
    sys.exit(0)
sys.stdin.read()
if os.environ["FAKE_DOCKER_MODE"] == "stall":
    time.sleep(60)
print(json.dumps({"status": "timeout", "stdout": "", "exit_code": -9}))
"""


@pytest.mark.parametrize("mode,stalled", [("stall", True), ("supervisor_timeout", False)])
def test_runner_marks_host_stalls_without_changing_the_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, stalled: bool
) -> None:
    script = tmp_path / "fake_docker.py"
    script.write_text(FAKE_RUNNER_DOCKER, encoding="utf-8")
    wrapper = tmp_path / "docker"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_DOCKER_MODE", mode)
    monkeypatch.setattr(runner, "HOST_WATCHDOG_SECONDS", 0.5)
    started = time.monotonic()
    result = runner.run("x = 1", "assert x == 1", Limits(timeout_seconds=0.1))
    assert result.status == "timeout"
    assert result.host_stalled is stalled
    assert time.monotonic() - started < 10


def test_the_pool_verification_retries_what_the_runner_marks_stalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pool's elapsed-time stall signature uses the runner's own watchdog constant, and a
    # result the runner marks stalled is retried whatever its elapsed time.
    assert task_pool.HOST_DEADLINE_SLACK == runner.HOST_WATCHDOG_SECONDS
    monkeypatch.setattr(task_pool, "RETRY_DELAY_SECONDS", 0.0)
    stalls = {"left": 1}

    def execute(code: str, tests: str, mode: str) -> runner.Result:
        if stalls["left"]:
            stalls["left"] -= 1
            return runner.Result("timeout", "", -9, 0.5, host_stalled=True)
        return runner.Result("passed")

    found, counters = task_pool.verify(
        [("a", "b", "function")],
        tmp_path / "cache.jsonl",
        execute=execute,
        deadline=lambda tests, mode: 2.0,
        workers=1,
        progress=lambda message: None,
    )
    assert found[task_pool.call_key("a", "b", "function")].passed
    assert counters["retried_calls"] == 1


# --------------------------------------------------------------------------------------
# Supervisor stream helpers (pure; they never run candidate code).
# --------------------------------------------------------------------------------------


def test_line_splitter_drops_only_the_oversized_line() -> None:
    lines: list[bytes] = []
    splitter = LineSplitter(10, lines.append)
    splitter.feed(b"short\n" + b"x" * 50)
    splitter.feed(b"x" * 50 + b"\nafter\npart")
    splitter.feed(b"ial")
    splitter.finish()
    assert lines == [b"short", b"after", b"partial"]


def test_admit_row_keeps_the_first_well_framed_row_per_index() -> None:
    rows: dict[int, bytes] = {}
    first = json.dumps({"i": 0, "ok": True, "v": ["int", 1]}).encode()
    for line in [
        b"[" * 60000,
        b"junk",
        json.dumps({"i": 7, "ok": True}).encode(),
        json.dumps({"i": True, "ok": True}).encode(),
        first,
        json.dumps({"i": 0, "ok": True, "v": ["int", 2]}).encode(),
    ]:
        admit_row(rows, 2, line)
    assert rows == {0: first}


# --------------------------------------------------------------------------------------
# Static audit.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,flag",
    [
        ("import sys\nsys._getframe(0)", "_getframe"),
        ("x = f.__code__.co_consts", "co_consts"),
        ("import inspect", "inspect"),
        ("import sys\nsys.excepthook = None", "excepthook"),
        ("import atexit", "atexit"),
        ("import os\nos._exit(0)", "os_exit"),
        ("from os import _exit", "os_exit"),
        ("class A:\n def __eq__(self, o): return True", "eq_override"),
        ("import builtins", "builtins"),
        ("__builtins__['x'] = 1", "builtins"),
        ("open('/proc/self/status')", "proc_path"),
        ("print(__file__)", "dunder_file"),
        ("import sys\nsys.modules.clear()", "sys_modules"),
        ("import gc", "gc"),
        ("import ctypes", "ctypes"),
        ("import signal", "signal"),
        ("open('/tmp/x', 'w')", "tmp_path"),
        # Evasions the first audit missed.
        ("class E:\n    __eq__ = lambda s, o: True", "eq_override"),
        ("E = type('E', (), {'__eq__': lambda s, o: True})", "eq_override"),
        ("class E:\n    async def __eq__(self, o):\n        return True", "eq_override"),
        ("class E:\n    def __ne__(self, o):\n        return False", "eq_override"),
        ("m = __import__('inspect')", "inspect"),
        ("m = __import__('ctypes')", "ctypes"),
        ("m = __import__('atexit')", "atexit"),
        ("import importlib\nimportlib.import_module('gc')", "gc"),
        ("from importlib import import_module as im\nim('signal')", "signal"),
        ("import importlib\nimportlib.import_module('ins' + 'pect')", "inspect"),
        ("name = input()\n__import__(name)", "dynamic_import"),
        ("import sys as s\ns.modules.clear()", "sys_modules"),
        ("from sys import modules", "sys_modules"),
        ("__import__('sys').modules.clear()", "sys_modules"),
        ("import sys\ngetattr(sys, 'modules')", "sys_modules"),
        ("x = 'a'\nopen(f'/tmp/{x}', 'w')", "tmp_path"),
        ("import pathlib\npathlib.Path('/tmp/x').write_text('1')", "tmp_path"),
        ("import sys\ngetattr(sys, '_get' + 'frame')(0)", "_getframe"),
        ("import os\ngetattr(os, '_ex' + 'it')(0)", "os_exit"),
        ("open('/pr' + 'oc/1/status')", "proc_path"),
        ("import os\nos.kill(1, 2)", "os_kill"),
        ("import os as o\no.killpg(0, 9)", "os_kill"),
        ("from os import kill\nkill(1, 2)", "os_kill"),
        # Evasions a review found: implicit concatenation, joined pieces, built code, frame
        # walks, trace hooks, marshal and the main module.
        ("C = type('C', (), {'__e' 'q__': lambda s, o: True})", "eq_override"),
        ("C = type('C', (), {'\\x5f_eq__': lambda s, o: True})", "eq_override"),
        ("import os\ngetattr(os, ''.join(['_e', 'xit']))(0)", "os_exit"),
        ("exec(bytes.fromhex('41').decode())", "dynamic_code"),
        ("eval(input())", "dynamic_code"),
        ("code = compile(text, 'x', 'exec')", "dynamic_code"),
        ("exec('import gc')", "gc"),
        ("exec('import os\\nos._exit(0)')", "os_exit"),
        (
            "try:\n    1 / 0\nexcept Exception as e:\n"
            "    g = e.__traceback__.tb_frame.f_back.f_globals",
            "frame_access",
        ),
        ("import sys\nframes = sys._current_frames()", "frame_access"),
        ("def g():\n    yield 1\nx = g().gi_frame.f_locals", "frame_access"),
        ("import sys\nsys.settrace(lambda *a: None)", "trace_hook"),
        ("import threading\nthreading.setprofile(print)", "trace_hook"),
        ("import sys\nsys.addaudithook(print)", "trace_hook"),
        ("import sys\nsys.monitoring.use_tool_id(0, 'x')", "trace_hook"),
        ("import marshal", "marshal"),
        ("m = __import__('mar' + 'shal')", "marshal"),
        ("import __main__", "main_module"),
        ("from __main__ import calls", "main_module"),
        ("import traceback\ntraceback.walk_stack(None)", "traceback"),
        ("import pdb", "debugger"),
    ],
)
def test_audit_flags_risky_patterns(code: str, flag: str) -> None:
    assert flag in audit_attempt(code)


@pytest.mark.parametrize(
    "code",
    [
        "def add(a, b):\n    return a + b",
        "class R:\n    def __enter__(self):\n        return self\n"
        "    def __exit__(self, *a):\n        return False",
        "def can_exit(state):\n    return state.done",
        "def find_exit(grid):\n    return grid",
        "path = '/process/items'",
        "import sys\nsys.exit(0)",
        "import sys\ndata = sys.stdin.read()",
        "x = getattr(obj, name)",
        "import os\nprint(os.path.join('a', 'b'))",
        # ``re.compile`` and ``str.join`` are not the code builtins; constant code is audited.
        "import re\npattern = re.compile(r'\\d+')",
        "print(', '.join(['a', 'b']))",
        "value = eval('1 + 2')",
        "frame = {'f_back_count': 1}",
        "def evaluate(expr):\n    return expr",
    ],
)
def test_audit_does_not_flag_ordinary_code(code: str) -> None:
    assert audit_attempt(code) == ()


def test_audit_survives_syntax_errors_and_deep_nesting() -> None:
    # Text scanning still flags obvious tokens even when the code does not parse.
    assert "os_exit" in audit_attempt("os._exit(0) this is not valid python (")
    assert audit_attempt("x = " + "+".join(["'a'"] * 5000)) == ()
    assert audit_attempt("(" * 10000) == ()


def test_audit_tally_tracks_the_rewarded_denominator() -> None:
    tally = AuditTally()
    tally.observe(())
    tally.observe(("os_exit",))
    tally.observe(("os_exit", "co_consts"), rewarded=True)
    tally.observe((), rewarded=True)
    assert tally.attempts == 4 and tally.flagged == 2
    assert tally.flag_counts["os_exit"] == 2
    assert tally.rate == 0.5
    assert tally.rewarded == 2 and tally.rewarded_flagged == 1
    assert tally.rewarded_rate == 0.5


def test_audit_tally_observes_results() -> None:
    tally = AuditTally()
    tally.observe_result(RewardResult(COMPLETED, 0.5, audit_flags=("gc",)))
    tally.observe_result(RewardResult(COMPLETED, 0.0, audit_flags=("gc",)))
    tally.observe_result(RewardResult(HOST_ERROR, None))
    assert tally.attempts == 3 and tally.rewarded == 1 and tally.rewarded_rate == 1.0


def test_reward_result_shape() -> None:
    result = RewardResult(COMPLETED, 0.5, 1, 2)
    assert result.reward == 0.5 and not result.is_host_failure
    assert result.unfinished == 0 and result.outcomes == ()


# --------------------------------------------------------------------------------------
# Docker-backed tests (opt-in; disposable worker only).
# --------------------------------------------------------------------------------------

integration = pytest.mark.skipif(
    os.environ.get("SCRATCHCODER_SANDBOX_TESTS") != "1",
    reason="requires explicitly enabled disposable Docker worker",
)

# Hidden expectations a blind forge cannot guess: specific, non-trivial literals.
ABUSE_HIDDEN = (
    "assert secret(2) == 424242\n"
    "assert secret(3) == 636363\n"
    "assert secret(10) == 2121210\n"
    "assert secret(0) == 0\n"
)
CORRECT = "def secret(n):\n    return n * 212121"
PARTIAL = "def secret(n):\n    return n * 212121 if n < 10 else -1"


def live(task: Task, code: str, **kwargs: Any) -> RewardResult:
    """Score in the real container; a host failure is a test failure, never a skip."""
    result = score_attempt(task, code, **kwargs)
    assert not result.is_host_failure, result
    return result


@integration
def test_live_correct_and_partial_rewards() -> None:
    full = live(function_task(ABUSE_HIDDEN), CORRECT)
    assert full.status == COMPLETED and full.reward == 1.0
    part = live(function_task(ABUSE_HIDDEN), PARTIAL)
    assert part.status == COMPLETED and part.passed == 3 and part.reward == 0.75


@integration
@pytest.mark.parametrize(
    "code",
    [
        # Early exit before producing any result.
        "import os\ndef secret(n):\n    return n * 212121\nos._exit(0)",
        # Exception from an excepthook / atexit cannot forge a pass.
        "import sys, atexit\n"
        "sys.excepthook = lambda *a: print('nope')\n"
        "atexit.register(lambda: print('nope'))\n"
        "def secret(n):\n    raise SystemExit(0)",
        # Always-equal object is unserialisable.
        "class E:\n    def __eq__(self, o):\n        return True\ndef secret(n):\n    return E()",
        # An always-equal dict subclass is reported by its (empty) stored items.
        "class D(dict):\n    def __eq__(self, o):\n        return True\n"
        "def secret(n):\n    return D()",
        # An object that only pretends to be an int.
        "class Fake:\n    @property\n    def __class__(self):\n        return int\n"
        "    def __eq__(self, o):\n        return True\ndef secret(n):\n    return Fake()",
        # Blindly forge result lines to every open descriptor (does not know the expected values).
        "import os, json\n"
        "def secret(n):\n    return 0\n"
        "line = (json.dumps({'i': 0, 'ok': True, 'v': ['int', 999]}) + chr(10)).encode()\n"
        "for fd in range(3, 64):\n"
        "    try:\n"
        "        os.write(fd, line)\n"
        "    except OSError:\n        pass",
        # Flood stdout.
        "def secret(n):\n    return 0\nwhile True:\n    print('x' * 4096, flush=True)",
        # Sleep past the per-attempt budget.
        "import time\ndef secret(n):\n    return n * 212121\ntime.sleep(30)",
        # Fork bomb (contained by the pid limit).
        "import os\ndef secret(n):\n    return 0\nwhile True:\n    os.fork()",
        # Read /proc.
        "def secret(n):\n    return open('/proc/1/status').read() and 0",
        # Close every output and keep running.
        "import os, time\nfor fd in range(1, 64):\n    try:\n        os.close(fd)\n"
        "    except OSError:\n        pass\ntime.sleep(30)",
    ],
)
def test_live_abuse_cannot_exceed_correct_reward(code: str) -> None:
    result = live(function_task(ABUSE_HIDDEN), code)
    assert result.reward is not None
    assert result.reward < 1.0


@integration
def test_live_malformed_and_deep_rows_are_scored_not_host_failures() -> None:
    code = (
        "import os, json\n"
        "def secret(n):\n    return 0\n"
        "junk = [b'[' * 200000 + b'\\n',\n"
        "        (json.dumps({'i': 0, 'ok': True, 'v': ['int']}) + '\\n').encode(),\n"
        "        (json.dumps({'i': 1, 'ok': True, 'v': ['bytes', 10 ** 20]}) + '\\n').encode(),\n"
        "        (json.dumps({'i': 2, 'ok': True, 'v': ['bytes', 10 ** 8]}) + '\\n').encode()]\n"
        "for fd in range(3, 64):\n"
        "    for line in junk:\n"
        "        try:\n"
        "            os.write(fd, line)\n"
        "        except OSError:\n"
        "            pass\n"
    )
    result = live(function_task(ABUSE_HIDDEN), code)
    # Only ``secret(0) == 0`` can pass for a function returning 0.
    assert result.status == COMPLETED and result.reward is not None and result.reward <= 0.25


@integration
@pytest.mark.parametrize(
    "prelude",
    [
        # PID 1 ignores the signal, so the honest result still arrives.
        "import os, signal\nos.kill(1, signal.SIGINT)\nos.kill(1, signal.SIGTERM)\n",
        # PID 1 is non-dumpable, so its /proc entries refuse the candidate.
        "try:\n    open('/proc/1/fd/1', 'w').write('junk\\n')\n    BLOCKED = False\n"
        "except OSError:\n    BLOCKED = True\n",
        "try:\n    open('/proc/1/oom_score_adj', 'w').write('1000')\n    BLOCKED = False\n"
        "except OSError:\n    BLOCKED = True\n",
    ],
)
def test_live_pid_one_is_out_of_the_candidates_reach(prelude: str) -> None:
    guard = "BLOCKED" if "BLOCKED" in prelude else "True"
    code = prelude + f"def secret(n):\n    return n * 212121 if {guard} else -1\n"
    result = live(function_task(ABUSE_HIDDEN), code)
    assert result.status == COMPLETED and result.reward == 1.0


@integration
def test_live_subclasses_are_reported_as_their_base_type() -> None:
    hidden = (
        "assert counts('aab') == {'a': 2, 'b': 1}\n"
        "assert point() == (1, 2)\n"
        "assert ordered() == {'x': 1}\n"
        "assert defaults() == {'k': [1]}\n"
        "assert color() == 3\n"
        "assert label() == 'hi'\n"
    )
    code = (
        "from collections import Counter, OrderedDict, defaultdict, namedtuple\n"
        "import enum\n"
        "P = namedtuple('P', 'x y')\n"
        "class C(enum.IntEnum):\n    RED = 3\n"
        "class S(str):\n    pass\n"
        "def counts(s):\n    return Counter(s)\n"
        "def point():\n    return P(1, 2)\n"
        "def ordered():\n    return OrderedDict(x=1)\n"
        "def defaults():\n    d = defaultdict(list)\n    d['k'].append(1)\n    return d\n"
        "def color():\n    return C.RED\n"
        "def label():\n    return S('hi')\n"
    )
    result = live(function_task(hidden), code)
    assert result.status == COMPLETED and result.reward == 1.0


# (hidden assertion, candidate code, whether the evaluation runner's in-process assertion
# passes it, whether the reward channel passes it). The two agree on built-in values; the rows
# where they differ are the documented residuals: a value outside the reported types fails in
# reward mode (a candidate-defined type, ``complex``, ``Decimal``, ``Fraction``, ``deque``),
# and a subclass is judged by its base value, not by its own ``__eq__``.
LIVE_EQUALITY: list[tuple[str, str, bool, bool]] = [
    ("assert half(4) == 2", "def half(n):\n    return n / 2", True, True),
    ("assert is_even(4) == 1", "def is_even(n):\n    return n % 2 == 0", True, True),
    ("assert is_even(4) == True", "def is_even(n):\n    return 1", True, True),
    ("assert is_even(4) is True", "def is_even(n):\n    return 1", False, False),
    ("assert pair() == (1, 2)", "def pair():\n    return [1, 2]", False, False),
    ("assert pair() == [1, 2]", "def pair():\n    return [1.0, 2]", True, True),
    ("assert nan() == 0.0", "def nan():\n    return float('nan')", False, False),
    ("assert zero() == 0", "def zero():\n    return -0.0", True, True),
    ("assert big() == 1e999", "def big():\n    return float('inf')", True, True),
    ("assert not empty()", "def empty():\n    return 0.0", True, True),
    ("assert keys() == {1: 'a'}", "def keys():\n    return {True: 'a'}", True, True),
    # A bytearray is reported as bytes, a keys view as a frozenset: same ``==`` either way.
    ("assert raw() == b'ab'", "def raw():\n    return bytearray(b'ab')", True, True),
    ("assert raws() == [b'a']", "def raws():\n    return [bytearray(b'a')]", True, True),
    ("assert keys() == {1, 2}", "def keys():\n    return {1: 0, 2: 0}.keys()", True, True),
    ("assert not keys()", "def keys():\n    return {}.keys()", True, True),
    ("assert keys() == [1]", "def keys():\n    return {1: 0}.keys()", False, False),
    # Residuals: values the encoding does not carry.
    (
        "assert point() == (1, 2)",
        "class P:\n    def __eq__(self, other):\n        return other == (1, 2)\n"
        "def point():\n    return P()",
        True,
        False,
    ),
    ("assert c() == 2", "def c():\n    return complex(2, 0)", True, False),
    (
        "assert d() == 1.5",
        "from decimal import Decimal\ndef d():\n    return Decimal('1.5')",
        True,
        False,
    ),
    (
        "assert q() == 0.5",
        "from fractions import Fraction\ndef q():\n    return Fraction(1, 2)",
        True,
        False,
    ),
    ("assert not d()", "from collections import deque\ndef d():\n    return deque()", True, False),
    # Residuals: a subclass is judged by its base value, either way.
    (
        "assert s() == 'abc'",
        "class S(str):\n    def __eq__(self, o):\n        return str.lower(self) == str.lower(o)\n"
        "    __hash__ = str.__hash__\ndef s():\n    return S('ABC')",
        True,
        False,
    ),
    (
        "assert n() == 5",
        "class N(int):\n    def __eq__(self, o):\n        return False\n"
        "    __hash__ = int.__hash__\ndef n():\n    return N(5)",
        False,
        True,
    ),
]


@integration
def test_live_equality_matches_the_in_process_assertion() -> None:
    def verdicts(item: tuple[str, str, bool, bool]) -> tuple[str, float | None, bool, bool, bool]:
        hidden, code, evaluation, rewarded = item
        result = live(function_task(hidden), code).reward
        return hidden, result, runner.run(code, hidden).passed, evaluation, rewarded

    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(verdicts, LIVE_EQUALITY))
    for hidden, result, evaluated, evaluation, rewarded in rows:
        assert evaluated is evaluation, hidden
        assert result == (1.0 if rewarded else 0.0), hidden


# A candidate that rewinds stdin to read the driver's configuration (the call sources, and a
# literal written inside one of them) finds it empty, as in the evaluation program.
REWIND_STDIN = (
    "import json, os, re, sys\n"
    "try:\n"
    "    os.lseek(0, 0, os.SEEK_SET)\n"
    "    calls = json.loads(sys.stdin.read())['calls']\n"
    "    NUMS = [float(x) for c in calls for x in re.findall(r'\\d+\\.\\d+', c)]\n"
    "except Exception:\n"
    "    NUMS = [0.0]\n"
    "def f(x):\n"
    "    return NUMS[0]\n"
)

# (hidden tests, candidate code, whether the evaluation runner passes it): the candidate runs
# as the main script ``/tmp/candidate.py`` with an empty stdin, and the hidden tests' imports
# come after it, in both paths.
LIVE_MAIN_SCRIPT: list[tuple[str, str, bool]] = [
    (
        "assert db.get('a') == 1",
        "class D(dict):\n    pass\nif __name__ == '__main__':\n    db = D(a=1)",
        True,
    ),
    (
        "assert where() == '/tmp'",
        "import os\ndef where():\n    return os.path.dirname(os.path.abspath(__file__))",
        True,
    ),
    ("import math\nassert f() == 3", "X = math.floor(3.5)\ndef f():\n    return X", False),
    ("from math import floor\nassert floor(2.5) == 2", "def floor(x):\n    return 99", True),
    ("assert f() == 3", "def f():\n    return __builtins__.len([1, 2, 3])", True),
    (
        "assert f() == 3",
        "import sys\ndef helper():\n    return 3\n"
        "def f():\n    return getattr(sys.modules['__main__'], 'helper')()",
        True,
    ),
    ("assert f() == '/tmp/candidate.py'", "import sys\ndef f():\n    return sys.argv[0]", True),
    (
        "assert f() == 'def g(): pass\\n'",
        "import inspect\ndef g(): pass\ndef f():\n    return inspect.getsource(g)",
        True,
    ),
    (
        "assert f() == 1",
        "import pickle\nclass A:\n    x = 1\n"
        "def f():\n    return pickle.loads(pickle.dumps(A())).x",
        True,
    ),
    ("assert f() == ''", "import sys\ndef f():\n    return sys.stdin.read()", True),
    ("assert f(1) - 2.5 == 0", REWIND_STDIN, False),
]


@integration
def test_live_candidate_runs_as_the_main_script_as_in_evaluation() -> None:
    def verdicts(item: tuple[str, str, bool]) -> tuple[str, float | None, bool, bool]:
        hidden, code, expected = item
        rewarded = live(function_task(hidden), code).reward
        return hidden, rewarded, runner.run(code, hidden).passed, expected

    with ThreadPoolExecutor(4) as pool:
        for hidden, rewarded, evaluated, expected in pool.map(verdicts, LIVE_MAIN_SCRIPT):
            assert evaluated is expected, hidden
            assert rewarded == (1.0 if expected else 0.0), hidden


# Process-level failures the evaluation program fails as a whole: output beyond the cap, a
# nonzero exit status, and a process that never ends. The first program is the control.
LIVE_PROCESS: list[tuple[str, bool]] = [
    ("def f():\n    print('x' * 1000)\n    return 1", True),
    ("def f():\n    print('x' * 70000)\n    return 1", False),
    ("import atexit, os\natexit.register(lambda: os._exit(1))\ndef f():\n    return 1", False),
    (
        "import threading, time\nthreading.Thread(target=lambda: time.sleep(100)).start()\n"
        "def f():\n    return 1",
        False,
    ),
]


@integration
def test_live_process_level_failures_fail_every_check_as_in_evaluation() -> None:
    hidden = "assert f() == 1"

    def verdicts(item: tuple[str, bool]) -> tuple[str, RewardResult, bool, bool]:
        code, expected = item
        result = live(function_task(hidden), code, budget=3.0)
        return code, result, runner.run(code, hidden).passed, expected

    with ThreadPoolExecutor(4) as pool:
        for code, result, evaluated, expected in pool.map(verdicts, LIVE_PROCESS):
            assert evaluated is expected, code
            assert result.status == COMPLETED, code
            assert result.reward == (1.0 if expected else 0.0), code


@integration
def test_live_resource_kill_is_distinct_from_timeout() -> None:
    task = function_task("assert secret(1) == 1")
    result = live(
        task,
        "def secret(n):\n    return n\nx = bytearray(1024 * 1024 * 1024)",
        limits=Limits(timeout_seconds=6.0, memory_mb=64),
    )
    assert result.status == RESOURCE_KILL
    assert result.reward == 0.0
    timed_out = live(task, "def secret(n):\n    return n\nwhile True:\n    pass", budget=2.0)
    assert timed_out.status == COMPLETED and timed_out.reward == 0.0


THREE_CASES = [{"input": f"{n}\n", "output": str(n + 1)} for n in (1, 2, 3)]


@integration
def test_live_io_correct_partial_and_budget() -> None:
    cases = [{"input": "1\n", "output": "2"}, {"input": "5\n", "output": "6"}]
    good = live(io_task(cases), "print(int(input()) + 1)")
    assert good.status == COMPLETED and good.reward == 1.0
    half = live(io_task(cases), "n = int(input())\nprint(n + 1 if n < 5 else 0)")
    assert half.passed == 1 and half.total == 2


@integration
def test_live_io_large_correct_outputs_earn_full_reward() -> None:
    cases = [{"input": f"{n}\n", "output": str(n) * 60000} for n in range(10)]
    result = live(io_task(cases), "n = input().strip()\nprint(n * 60000)")
    assert result.status == COMPLETED and result.reward == 1.0


@integration
@pytest.mark.parametrize(
    "code,count",
    [
        ("while True:\n    print('x' * 4096, flush=True)", 3),
        ("import sys\nsys.stdout.buffer.write(b'\\xff' * 60000)", 3),
        ("import sys\nsys.stdout.buffer.write(b'\\xff' * 70000)", 3),
        ("print('1 ' * 40000)", 20),
    ],
)
def test_live_io_output_abuse_scores_zero_not_a_host_failure(code: str, count: int) -> None:
    cases = [{"input": f"{n}\n", "output": str(n + 1)} for n in range(count)]
    result = live(io_task(cases), code)
    assert result.status == COMPLETED and result.reward == 0.0


@integration
@pytest.mark.parametrize(
    "first_case",
    [
        # A fork bomb: the case times out and the sweep clears the pid limit.
        "    while True:\n        try:\n            os.fork()\n        except OSError:\n"
        "            pass\n",
        # Output closed while the process keeps running.
        "    print(2, flush=True)\n    os.close(1)\n    os.close(2)\n    time.sleep(30)\n",
        # The fork server is killed; the next case gets a new one.
        "    os.kill(os.getppid(), signal.SIGKILL)\n    time.sleep(1)\n",
        # /tmp is filled; inputs are not on /tmp, so later cases still start.
        "    for i in range(32):\n        try:\n"
        "            open(f'/tmp/fill{i}', 'wb').write(b'x' * 1048000)\n"
        "        except OSError:\n            break\n    raise SystemExit(1)\n",
    ],
)
def test_live_io_one_case_cannot_spoil_the_next(first_case: str) -> None:
    # An explicit per-case limit, so a case that hangs leaves time for the next ones.
    code = "import os, signal, time\nn = int(input())\nif n == 1:\n" + first_case + "print(n + 1)\n"
    result = live(io_task(THREE_CASES), code, budget=6.0, per_case=1.5)
    assert result.status == COMPLETED
    assert result.outcomes == (False, True, True)


@integration
def test_live_io_cases_share_the_pool_verifications_deadline() -> None:
    # 40 cases: 2 s + 0.25 s x 40 = 12 s for the whole attempt. One case needing 5 s (more
    # than any fixed per-case limit of 4 s would allow) still passes, as it does in the pool
    # verification; a case that never ends uses up the budget and leaves the rest unrun.
    cases = [{"input": f"{n}\n", "output": str(n + 1)} for n in range(40)]
    task = io_task(cases)
    slow = "import time\nn = int(input())\nif n == 0:\n    time.sleep(5)\nprint(n + 1)"
    result = live(task, slow)
    assert result.budget_seconds == 12.0
    assert result.status == COMPLETED and result.reward == 1.0 and result.unfinished == 0
    pool_limits = task_pool.limits_for(pool_verification(), task.hidden_tests, "io")
    assert runner.run(slow, task.hidden_tests, pool_limits, mode="io").passed
    # The container's CPU limit equals the budget by default, so a busy loop can be ended by
    # the CPU limit a moment before the wall deadline instead of by it; the reward side lifts
    # the CPU limit to make the outcome exact, and the pool verification may report either
    # ``timeout`` or ``failed``.
    hang = "n = int(input())\nwhile n == 0:\n    pass\nprint(n + 1)"
    hung = live(task, hang, limits=Limits(timeout_seconds=30.0))
    assert hung.budget_seconds == 12.0
    assert hung.status == COMPLETED and hung.reward == 0.0 and hung.unfinished == 40
    assert not runner.run(hang, task.hidden_tests, pool_limits, mode="io").passed


@integration
def test_live_io_a_hundred_correct_cases_are_not_cut_by_the_old_flat_budget() -> None:
    # About 0.1 s of work per case is about 10 s in all: over the old flat 8 s budget, inside
    # the rule's 27 s for 100 cases.
    cases = [{"input": f"{n}\n", "output": str(n + 1)} for n in range(100)]
    code = "import time\nn = int(input())\ntime.sleep(0.1)\nprint(n + 1)"
    result = live(io_task(cases), code)
    assert result.budget_seconds == 27.0
    assert result.status == COMPLETED and result.reward == 1.0 and result.unfinished == 0
    assert result.elapsed_seconds > 8.0


@integration
def test_live_io_signals_to_pid_one_are_ignored() -> None:
    code = "import os, signal\nos.kill(1, signal.SIGINT)\nprint(int(input()) + 1)"
    result = live(io_task(THREE_CASES), code)
    assert result.status == COMPLETED and result.reward == 1.0


@integration
@pytest.mark.parametrize(
    "code",
    [
        # Stop the fork server, then finish: its exit-status answer never comes.
        "import os, signal\nos.kill(os.getppid(), signal.SIGSTOP)\nprint(int(input()) + 1)",
        # Stop the fork server and keep the case running.
        "import os, signal, time\nos.kill(os.getppid(), signal.SIGSTOP)\ntime.sleep(100)",
    ],
)
def test_live_a_stopped_fork_server_cannot_push_an_attempt_into_a_host_stall(code: str) -> None:
    # The supervisor bounds its own waits past the budget (``OVERRUN_SECONDS``), so the
    # attempt is scored well before the host's stall deadline of budget + 15 s.
    result = live(io_task(THREE_CASES), code)
    assert result.status == COMPLETED and result.reward == 0.0
    assert result.budget_seconds is not None
    overrun = result.elapsed_seconds - result.budget_seconds
    assert overrun < reward_program.OVERRUN_SECONDS + 5.0, overrun


@integration
def test_live_io_files_in_tmp_persist_across_cases_as_in_the_pool_verification() -> None:
    # Cases share the container's /tmp in both paths; no expected value is there to find.
    code = (
        "import os\npath = '/tmp/n'\n"
        "n = int(open(path).read()) if os.path.exists(path) else 0\n"
        "open(path, 'w').write(str(n + 1))\nprint(n)"
    )
    task = io_task([{"input": "", "output": str(n)} for n in range(5)])
    assert live(task, code).reward == 1.0
    pool_limits = task_pool.limits_for(pool_verification(), task.hidden_tests, "io")
    assert runner.run(code, task.hidden_tests, pool_limits, mode="io").passed


@integration
def test_live_io_out_of_memory_case_is_a_resource_kill() -> None:
    code = "n = int(input())\nif n == 1:\n    x = bytearray(1024 * 1024 * 1024)\nprint(n + 1)"
    result = live(
        io_task(THREE_CASES), code, budget=6.0, limits=Limits(timeout_seconds=6.0, memory_mb=64)
    )
    assert result.status == RESOURCE_KILL
    assert result.outcomes == (False, True, True)


@integration
def test_live_io_many_cases_fit_the_budget_under_parallel_load() -> None:
    # 99 interpreter starts used to take a whole 8 s budget under emulation; the fork server
    # makes a correct 99-case solution earn 1.0 even within 8 s (the rule allows 26.75 s), also
    # with 8 attempts running at once.
    cases = [{"input": f"{n}\n", "output": str(n + 1)} for n in range(99)]
    task = io_task(cases)
    with ThreadPoolExecutor(8) as pool:
        results = list(
            pool.map(lambda _: live(task, "print(int(input()) + 1)", budget=8.0), range(8))
        )
    for result in results:
        assert result.status == COMPLETED and result.reward == 1.0 and result.unfinished == 0


# Programs whose per-case I/O verdicts must match the evaluation path of record exactly.
TWO_CASES = [{"input": "3\n", "output": "6"}, {"input": "5\n", "output": "10"}]
ONE_EMPTY = [{"input": "", "output": "/tmp/candidate.py"}]
EQUIVALENCE = {
    "double": ("print(int(input()) * 2)", TWO_CASES),
    "read_all": ("import sys\nprint(sum(map(int, sys.stdin.read().split())) * 2)", TWO_CASES),
    "main_guard": (
        "def main():\n    print(int(input()) * 2)\nif __name__ == '__main__':\n    main()",
        TWO_CASES,
    ),
    "argv": ("import sys\nprint(sys.argv[0])", ONE_EMPTY),
    "dunder_file": ("print(__file__)", ONE_EMPTY),
    "exit_zero": ("import sys\nprint(int(input()) * 2)\nsys.exit(0)", TWO_CASES),
    "exit_three": ("print(int(input()) * 2)\nraise SystemExit(3)", TWO_CASES),
    "exit_message": ("import sys\nprint(int(input()) * 2)\nsys.exit('bye')", TWO_CASES),
    "raises": ("print(int(input()) * 2)\nraise ValueError('x')", TWO_CASES),
    "atexit": (
        "import atexit\natexit.register(lambda: print('bye'))\nprint(int(input()) * 2)",
        [{"input": "3\n", "output": "6\nbye"}],
    ),
    "thread": (
        "import threading, sys\nsys.setrecursionlimit(10000)\nthreading.stack_size(1 << 26)\n"
        "def run():\n    print(int(input()) * 2)\nthreading.Thread(target=run).start()",
        TWO_CASES,
    ),
    "no_newline": ("import sys\nsys.stdout.write(str(int(input()) * 2))", TWO_CASES),
    "stderr": ("import sys\nprint(int(input()) * 2, file=sys.stderr)", TWO_CASES),
    "unicode": ("print('\u00e9' + input().strip())", [{"input": "a\n", "output": "\u00e9a"}]),
    "invalid_utf8": (
        "import sys\nsys.stdout.buffer.write(b'\\xff\\n')",
        [{"input": "", "output": "\ufffd"}],
    ),
    "exit_builtin": ("print(int(input()) * 2)\nexit()", TWO_CASES),
    "syntax_error": ("print(", TWO_CASES),
    "os_write": ("import os\nos.write(1, (str(int(input()) * 2) + '\\n').encode())", TWO_CASES),
    "loop": ("while True:\n    pass", TWO_CASES[:1]),
    "eof": ("a = input()\nb = input()\nprint(int(a) * 2)", TWO_CASES),
    "open_zero": ("print(int(open(0).read()) * 2)", TWO_CASES),
    "close_stdout": ("import sys\nprint(int(input()) * 2)\nsys.stdout.close()", TWO_CASES),
    "recursion": (
        "def f(n):\n    return 0 if n == 0 else 1 + f(n - 1)\nprint(f(900) - 900 + 6)",
        TWO_CASES[:1],
    ),
    "coding_cookie": ("# -*- coding: latin-1 -*-\nprint(int(input()) * 2)", TWO_CASES),
    "os_exit_flushed": (
        "import os, sys\nprint(int(input()) * 2)\nsys.stdout.flush()\nos._exit(0)",
        TWO_CASES,
    ),
    "os_exit_unflushed": ("import os\nprint(int(input()) * 2)\nos._exit(0)", TWO_CASES),
    "input_prompt": ("print(int(input('n? ')) * 2)", TWO_CASES),
    # Output that only reaches stdout during interpreter shutdown (a common TACO fast-I/O
    # idiom, an unclosed writer, a re-wrapped stream) and other shutdown paths.
    "fastio_atexit": (
        "import sys\nfrom io import StringIO\nfrom atexit import register\n"
        "sys.stdout = StringIO()\n"
        "register(lambda: sys.__stdout__.write(sys.stdout.getvalue()))\n"
        "print(int(input()) * 2)",
        TWO_CASES,
    ),
    "unclosed_writer": (
        "out = open(1, 'w', closefd=False)\nout.write(str(int(input()) * 2))",
        TWO_CASES,
    ),
    "rewrapped_stdout": (
        "import io, sys\nsys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')\n"
        "print(int(input()) * 2)",
        TWO_CASES,
    ),
    "buffer_write": (
        "import sys\nsys.stdout.buffer.write(str(int(input()) * 2).encode())",
        TWO_CASES,
    ),
    "daemon_thread": (
        "import threading, time\ndef run():\n    time.sleep(0.2)\n    print('late')\n"
        "threading.Thread(target=run, daemon=True).start()\nprint(int(input()) * 2)",
        TWO_CASES,
    ),
    "stdout_none": (
        "import sys\nprint(int(input()) * 2)\nsys.stdout.flush()\nsys.stdout = None",
        TWO_CASES,
    ),
    "stderr_broken": ("import sys\nprint(int(input()) * 2)\nsys.stderr = 5", TWO_CASES),
    "keyboard_interrupt": ("print(int(input()) * 2)\nraise KeyboardInterrupt", TWO_CASES),
    "exit_256": ("import sys\nprint(int(input()) * 2)\nsys.exit(256)", TWO_CASES),
}


@integration
def test_live_io_fork_server_matches_the_evaluation_path_of_record() -> None:
    def verdicts(item: tuple[str, tuple[str, list[dict[str, str]]]]) -> tuple[str, Any, Any]:
        name, (code, cases) = item
        task = io_task(cases)
        rewarded = live(task, code).outcomes
        evaluated = tuple(
            runner.run(code, json.dumps([case]), Limits(timeout_seconds=4.0), mode="io").passed
            for case in cases
        )
        # The pool verification runs all cases in one call under the same deadline rule.
        pool_limits = task_pool.limits_for(pool_verification(), task.hidden_tests, "io")
        whole = runner.run(code, task.hidden_tests, pool_limits, mode="io").passed
        return name, rewarded, (evaluated, whole)

    with ThreadPoolExecutor(4) as pool:
        for name, rewarded, (evaluated, whole) in pool.map(verdicts, EQUIVALENCE.items()):
            assert rewarded == evaluated, name
            assert all(rewarded) == whole, name
    # An output beyond the cap fails both ways: the evaluation path reports output_limit, and
    # the reward channel never admits a task whose expected output could not fit.
    oversized = [{"input": "", "output": "x" * 70000}]
    assert not runner.run("print('x' * 70000)", json.dumps(oversized), mode="io").passed
    assert not is_parseable(io_task(oversized))
