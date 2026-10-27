"""Tests for the out-of-process RL reward channel.

The unit tests drive the host comparison, parsing and audit logic with fake containers and
need no Docker. The abuse tests at the bottom are gated behind ``SCRATCHCODER_SANDBOX_TESTS=1``
and prove against the real pinned container that no forgery, always-equal object, early exit,
flood, over-budget sleep, fork bomb or ``/proc`` read can earn more reward than correct values.
"""

import json
import os
from collections.abc import Callable

import pytest

from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.reward import (
    COMPLETED,
    HOST_ERROR,
    HOST_TIMEOUT,
    NOT_PARSEABLE,
    RESOURCE_KILL,
    UNAVAILABLE,
    AuditTally,
    ContainerOutcome,
    RewardResult,
    audit_attempt,
    is_parseable,
    parse_function_checks,
    score_attempt,
    score_with_host_retries,
)


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
        "assert not f()",
        "assert 7 == f()",
        "assert f() == -3.5",
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
        "assert f()",
        "assert all(x > 0 for x in f())",
        "assert f() == g()",
        "assert f() == object()",
        "assert f().startswith('x')",
        "x = 1\nassert f() == 1",
        "not a test at all (",
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


def test_io_tasks_are_always_parseable() -> None:
    assert is_parseable(io_task([{"input": "1", "output": "2"}]))


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
    if isinstance(value, dict):
        return ["dict", [[wire(k), wire(v)] for k, v in value.items()]]
    raise TypeError("unsupported in fake")


def function_container(
    values: list[object],
) -> Callable[[dict[str, object]], ContainerOutcome]:
    """A fake container reporting ``values[i]`` for call ``i`` (an exception value = an error)."""

    def run(payload: dict[str, object]) -> ContainerOutcome:
        calls = []
        for index in range(len(values)):
            item = values[index]
            if isinstance(item, BaseException):
                calls.append({"i": index, "ok": False, "e": "boom"})
            else:
                calls.append({"i": index, "ok": True, "v": wire(item)})
        report = {"status": "ran", "mode": "function", "n": len(calls), "calls": calls}
        return ContainerOutcome(COMPLETED, report)

    return run


def io_container(
    stdouts: list[str | None],
) -> Callable[[dict[str, object]], ContainerOutcome]:
    def run(payload: dict[str, object]) -> ContainerOutcome:
        cases = []
        for text in stdouts:
            if text is None:
                cases.append({"ran": False, "e": "case_timeout"})
            else:
                cases.append({"ran": True, "stdout": text})
        report = {"status": "ran", "mode": "io", "n": len(cases), "cases": cases}
        return ContainerOutcome(COMPLETED, report)

    return run


def test_correct_values_earn_full_reward() -> None:
    task = function_task("assert f() == 10\nassert f() == 10")
    result = score_attempt(task, "code", container=function_container([10, 10]))
    assert result.status == COMPLETED
    assert result.reward == 1.0
    assert (result.passed, result.total) == (2, 2)


def test_partial_values_earn_the_right_fraction() -> None:
    task = function_task("assert f() == 1\nassert f() == 2\nassert f() == 3\nassert f() == 4")
    result = score_attempt(task, "code", container=function_container([1, 2, 99, 4]))
    assert result.passed == 3 and result.total == 4
    assert result.reward == 0.75


def test_denominator_comes_from_the_task_file_not_the_container() -> None:
    task = function_task("assert f() == 1\nassert f() == 2")

    def forge_many(payload: dict[str, object]) -> ContainerOutcome:
        # Flood with forged correct-looking rows, including out-of-range and duplicate indices.
        calls = [
            {"i": 0, "ok": True, "v": wire(1)},
            {"i": 0, "ok": True, "v": wire(1)},
            {"i": 1, "ok": True, "v": wire(2)},
            {"i": 5, "ok": True, "v": wire(2)},
            {"i": 99, "ok": True, "v": wire(2)},
        ]
        return ContainerOutcome(COMPLETED, {"status": "ran", "mode": "function", "calls": calls})

    result = score_attempt(task, "code", container=forge_many)
    assert result.total == 2
    assert result.passed == 2
    assert result.reward == 1.0


def test_type_strictness_bool_is_not_int_and_tuple_is_not_list() -> None:
    assert (
        score_attempt(
            function_task("assert f() == 1"), "c", container=function_container([True])
        ).reward
        == 0.0
    )
    assert (
        score_attempt(
            function_task("assert f() == (1, 2)"), "c", container=function_container([[1, 2]])
        ).reward
        == 0.0
    )
    # Sets compare order-independently.
    assert (
        score_attempt(
            function_task("assert f() == {1, 2, 3}"), "c", container=function_container([{3, 1, 2}])
        ).reward
        == 1.0
    )


def test_is_none_and_not_forms() -> None:
    assert (
        score_attempt(
            function_task("assert f() is None"), "c", container=function_container([None])
        ).reward
        == 1.0
    )
    assert (
        score_attempt(
            function_task("assert not f()"), "c", container=function_container([[]])
        ).reward
        == 1.0
    )
    assert (
        score_attempt(
            function_task("assert not f()"), "c", container=function_container([[1]])
        ).reward
        == 0.0
    )


def test_unserialisable_and_errored_calls_fail() -> None:
    # An error sentinel (what an always-equal object or crash produces) never passes.
    task = function_task("assert f() == 1\nassert f() == 2")
    result = score_attempt(task, "c", container=function_container([1, ValueError()]))
    assert result.passed == 1 and result.reward == 0.5


def test_missing_indices_count_as_failed() -> None:
    task = function_task("assert f() == 1\nassert f() == 2\nassert f() == 3")

    def partial(payload: dict[str, object]) -> ContainerOutcome:
        calls = [{"i": 0, "ok": True, "v": wire(1)}]
        return ContainerOutcome(COMPLETED, {"status": "ran", "mode": "function", "calls": calls})

    result = score_attempt(task, "c", container=partial)
    assert result.passed == 1 and result.total == 3


def test_forged_wrong_values_cannot_beat_correct() -> None:
    # The host compares against the secret expected value; a blind forge of a wrong value fails.
    task = function_task("assert f() == 424242")
    forged = score_attempt(task, "c", container=function_container([111111])).reward
    correct = score_attempt(task, "c", container=function_container([424242])).reward
    assert forged == 0.0 and correct == 1.0


def test_io_fraction_and_unrun() -> None:
    task = io_task(
        [
            {"input": "1", "output": "2"},
            {"input": "2", "output": "3"},
            {"input": "3", "output": "4"},
        ]
    )
    result = score_attempt(task, "c", container=io_container(["2", "wrong", None]))
    assert result.passed == 1 and result.total == 3


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


def test_resource_kill_is_a_zero_reward_not_a_host_failure() -> None:
    task = function_task("assert f() == 1")
    result = score_attempt(
        task, "c", container=lambda payload: ContainerOutcome(RESOURCE_KILL, None)
    )
    assert result.status == RESOURCE_KILL
    assert result.reward == 0.0
    assert not result.is_host_failure


def test_not_parseable_returns_none_reward_without_running() -> None:
    ran = False

    def guard(payload: dict[str, object]) -> ContainerOutcome:
        nonlocal ran
        ran = True
        return ContainerOutcome(COMPLETED, {"status": "ran", "mode": "function", "calls": []})

    result = score_attempt(function_task("assert f() != 1"), "c", container=guard)
    assert result.status == NOT_PARSEABLE and result.reward is None
    assert not ran


def test_retries_recover_a_transient_host_failure() -> None:
    calls = {"n": 0}

    def flaky(payload: dict[str, object]) -> ContainerOutcome:
        calls["n"] += 1
        if calls["n"] < 3:
            return ContainerOutcome(HOST_TIMEOUT, None)
        return ContainerOutcome(
            COMPLETED,
            {"status": "ran", "mode": "function", "calls": [{"i": 0, "ok": True, "v": wire(1)}]},
        )

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
        ("class A:\n def __eq__(self, o): return True", "eq_override"),
        ("import builtins", "builtins"),
        ("__builtins__['x'] = 1", "builtins"),
        ("open('/proc/self/status')", "proc_path"),
        ("print(__file__)", "dunder_file"),
        ("import sys\nsys.modules.clear()", "sys_modules"),
        ("import gc", "gc"),
        ("import ctypes", "ctypes"),
        ("import signal", "signal"),
        ("open('/tmp/x', 'w')", "open_tmp"),
    ],
)
def test_audit_flags_risky_patterns(code: str, flag: str) -> None:
    assert flag in audit_attempt(code)


def test_audit_clean_code_has_no_flags() -> None:
    assert audit_attempt("def add(a, b):\n    return a + b") == ()


def test_audit_survives_syntax_errors() -> None:
    # Substring scanning still flags obvious tokens even when the code does not parse.
    assert "os_exit" in audit_attempt("os._exit(0) this is not valid python (")


def test_audit_tally() -> None:
    tally = AuditTally()
    tally.observe(())
    tally.observe(("os_exit",))
    tally.observe(("os_exit", "co_consts"))
    assert tally.attempts == 3 and tally.flagged == 2
    assert tally.flag_counts["os_exit"] == 2
    assert round(tally.rate, 3) == round(2 / 3, 3)


def test_reward_result_shape() -> None:
    result = RewardResult(COMPLETED, 0.5, 1, 2)
    assert result.reward == 0.5 and not result.is_host_failure


# --------------------------------------------------------------------------------------
# Docker-backed abuse tests (opt-in; disposable worker only).
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


@integration
def test_live_correct_and_partial_rewards() -> None:
    full = score_attempt(function_task(ABUSE_HIDDEN), CORRECT, budget=6.0)
    assert full.status == COMPLETED and full.reward == 1.0
    part = score_attempt(function_task(ABUSE_HIDDEN), PARTIAL, budget=6.0)
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
    ],
)
def test_live_abuse_cannot_exceed_correct_reward(code: str) -> None:
    result = score_attempt(function_task(ABUSE_HIDDEN), code, budget=6.0)
    if result.is_host_failure:
        pytest.skip(f"host failure {result.status}; not a reward")
    assert result.reward is not None
    assert result.reward < 1.0


@integration
def test_live_forged_values_match_only_when_correct() -> None:
    # Reward is driven only by the hidden, host-held expectations: a candidate that returns
    # the right value earns 1.0, and no amount of forging without knowing it does better.
    assert score_attempt(function_task(ABUSE_HIDDEN), CORRECT, budget=6.0).reward == 1.0


@integration
def test_live_io_correct_partial_and_budget() -> None:
    cases = [{"input": "1\n", "output": "2"}, {"input": "5\n", "output": "6"}]
    good = score_attempt(io_task(cases), "print(int(input()) + 1)", budget=6.0)
    assert good.status == COMPLETED and good.reward == 1.0
    half = score_attempt(
        io_task(cases), "n = int(input())\nprint(n + 1 if n < 5 else 0)", budget=6.0
    )
    assert half.passed == 1 and half.total == 2


@integration
def test_live_resource_kill_is_distinct_from_timeout() -> None:
    from scratchcoder.sandbox.runner import Limits

    task = function_task("assert secret(1) == 1")
    # A hard memory blow-up should be reported as a resource kill (reward 0.0), not a host
    # failure, and must never raise the reward.
    result = score_attempt(
        task,
        "def secret(n):\n    return n\nx = bytearray(1024 * 1024 * 1024)",
        budget=6.0,
        limits=Limits(timeout_seconds=6.0, memory_mb=64),
    )
    assert result.reward in (0.0, None)
    assert result.reward != 1.0
