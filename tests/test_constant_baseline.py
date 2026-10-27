"""Reference checks for the constant-answer baselines (fake execution; no code runs here)."""

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from eval_fixtures import io_task, pool_task, save_task_file

from scratchcoder.eval.baseline import (
    CHECK_MARK,
    arc_constant_baseline,
    benchmark_constant_programs,
    candidate_names,
    constant_function_program,
    counted_passes,
    counting_tests,
    function_answers,
    io_answers,
    pool_baseline_report,
    pool_constant_baseline,
)
from scratchcoder.eval.choices import ChoiceTask
from scratchcoder.sandbox.reward import FUNCTION_DEGENERATE_ANSWERS, degenerate_baseline
from scratchcoder.sandbox.runner import Result

UNPARSEABLE = (
    # Non-literal expected values make the reward channel refuse the task.
    "assert sq(0) == [i for i in range(0)]\n"
    "assert sq(1) == [i * i for i in range(1, 2)]\n"
    "assert sq(3) == [x * x for x in (1, 2, 3)]\n"
)


def answer_of(code: str) -> str:
    """The expression the first constant function returns (what a fake sandbox inspects)."""
    found = re.search(r"\n    return (.*)\n", code)
    assert found is not None
    return found[1]


def fake_sandbox(verdicts: dict[str, list[int]]) -> Callable[[str, str, str], Result]:
    """Pretend to run counting tests: answer source -> indices of the asserts that hold."""

    def execute(code: str, tests: str, mode: str) -> Result:
        assert mode == "function" and CHECK_MARK in tests
        holds = verdicts.get(answer_of(code), [])
        total = tests.count("except BaseException:\n    print(")
        lines = [f"{CHECK_MARK} {index} {int(index in holds)}" for index in range(total)]
        return Result("passed", stdout="\n".join(lines) + "\n", elapsed_seconds=0.5)

    return execute


def test_parseable_tasks_use_the_reward_channel_on_the_host() -> None:
    tasks = [
        pool_task("t/1", hidden_tests="assert add_one(2) == 0\nassert add_one(5) == 6"),
        io_task("t/2", hidden_tests=json.dumps([{"input": "1", "output": "YES"}] * 2)),
    ]

    def never(code: str, tests: str, mode: str) -> Result:
        raise AssertionError("parseable tasks run no code")

    report = pool_constant_baseline(tasks, execute=never)
    rows = {row["task_id"]: row for row in report["per_task"]}
    for task in tasks:
        expected = degenerate_baseline(task)
        assert expected is not None
        assert rows[task.id]["reward"] == expected.reward
        assert rows[task.id]["method"] == "reward-channel"
    assert (
        rows["t/1"]["reward"] == 0.5 and rows["t/1"]["answer"] == "constant False"
    )  # False == 0 in Python
    assert rows["t/2"]["reward"] == 1.0 and rows["t/2"]["answer"] == "constant 'YES'"
    summary = report["summary"]
    assert summary["mean_reward"] == 0.75 and summary["full_pass_share"] == 0.5
    assert summary["uncovered"] == 0 and report["executed_calls"] == 0


def test_unparseable_tasks_execute_constant_programs_per_answer() -> None:
    task = pool_task("t/u", hidden_tests=UNPARSEABLE, public_tests="assert sq(2) == [1, 4]")
    assert degenerate_baseline(task) is None
    calls: list[str] = []
    sandbox = fake_sandbox({"[]": [0], "[1, 4]": [], "args[0]": []})

    def execute(code: str, tests: str, mode: str) -> Result:
        calls.append(code)
        return sandbox(code, tests, mode)

    report = pool_constant_baseline([task], execute=execute, workers=2)
    row = report["per_task"][0]
    assert row["method"] == "executed" and row["total"] == 3
    assert row["passed"] == 1 and row["reward"] == pytest.approx(1 / 3)
    assert row["answer"] == "constant []"
    # 11 fixed values, the one public example value, and echo; one call each.
    assert len(calls) == len(FUNCTION_DEGENERATE_ANSWERS) + 2 == report["executed_calls"]
    assert {answer_of(code) for code in calls} >= {"None", "[1, 4]", "args[0]"}
    assert report["sandbox_seconds"] == pytest.approx(0.5 * len(calls))


def test_uncoverable_tasks_are_counted_never_dropped() -> None:
    no_call = pool_task("t/a", hidden_tests="assert RESULT == [i for i in range(2)]")
    stalled = pool_task("t/b", hidden_tests=UNPARSEABLE)
    waits: list[float] = []

    def execute(code: str, tests: str, mode: str) -> Result:
        return Result("timeout", host_stalled=True)

    report = pool_constant_baseline([no_call, stalled], execute=execute, sleep=waits.append)
    rows = {row["task_id"]: row for row in report["per_task"]}
    assert rows["t/a"]["method"] == "uncovered" and "no candidate function" in rows["t/a"]["reason"]
    assert rows["t/b"]["method"] == "uncovered" and "unresolved" in rows["t/b"]["reason"]
    summary = report["summary"]
    assert summary["tasks"] == 2 and summary["covered"] == 0 and summary["uncovered"] == 2
    assert summary["mean_reward"] is None
    assert [item["task_id"] for item in summary["uncovered_tasks"]] == ["t/a", "t/b"]
    assert waits and max(waits) <= 30


def test_unparseable_io_tasks_compare_the_constant_output_with_every_case() -> None:
    long_output = "y" * 70000  # Longer than the output cap: the reward channel refuses it.
    cases = [{"input": "1", "output": "NO"}, {"input": "2", "output": long_output}]
    task = io_task("t/io", hidden_tests=json.dumps(cases))
    assert degenerate_baseline(task) is None

    def execute(code: str, tests: str, mode: str) -> Result:
        assert mode == "io" and json.loads(tests) == cases
        return Result("failed")

    report = pool_constant_baseline([task], execute=execute)
    row = report["per_task"][0]
    assert row["method"] == "executed" and row["passed"] == 1 and row["total"] == 2
    assert row["answer"] == "constant 'NO'"
    # A sandbox pass that the output comparison contradicts leaves the task uncovered.
    contradicted = pool_constant_baseline([task], execute=lambda *_: Result("passed"))
    assert contradicted["per_task"][0]["method"] == "uncovered"


def test_the_test_split_is_refused_outside_the_unseal_step(tmp_path: Path) -> None:
    task = pool_task("t/1", split="test")
    with pytest.raises(PermissionError, match="unseal"):
        pool_constant_baseline([task])
    assert pool_constant_baseline([task], allow_test=True)["summary"]["covered"] == 1
    path = save_task_file(tmp_path / "test.jsonl", [task])
    with pytest.raises(PermissionError, match="unseal"):
        pool_baseline_report(path, split="test")
    dev = save_task_file(tmp_path / "dev.jsonl", [pool_task("t/2")])
    report = pool_baseline_report(dev, split="dev")
    assert report["split"] == "dev" and len(report["task_file_sha256"]) == 64


def test_program_and_test_builders_by_hand() -> None:
    assert candidate_names("import math\nassert f(1) == g(len([2]))\nassert math.floor(h(1))") == [
        "f",
        "g",
        "h",
    ]
    assert candidate_names("def helper(x): return x\nassert helper(f(1)) == 1") == ["f"]
    answers = function_answers([[1, 4], [1, 4], 0, float("inf")])
    labels = [answer.label for answer in answers]
    assert labels[-2:] == ["public example [1, 4]", "echo"]  # Duplicates, 0 and inf skipped.
    program = constant_function_program(["f", "g"], answers[0])
    assert program == (
        "def f(*args, **kwargs):\n    return None\n\n\n"
        "def g(*args, **kwargs):\n    return None\n\n\n"
    )
    tests, total = counting_tests("import math\nassert f(1) == 1\nassert f(2) == 2\n")
    assert total == 2
    assert tests.startswith("try:\n    import math\nexcept BaseException:\n    pass\n")
    assert f"print({CHECK_MARK!r}, 1, 1, flush=True)" in tests
    stdout = f"noise\n{CHECK_MARK} 0 1\n{CHECK_MARK} 0 0\n{CHECK_MARK} 2 1\n{CHECK_MARK} 1 0\n"
    assert counted_passes(stdout, 2) == 1  # First report per index; index 2 is out of range.
    io_labels = [answer.label for answer in io_answers(io_task("t/1"))]
    assert io_labels[0] == "constant ''" and io_labels[-2:] == ["public example '6'", "echo"]


def test_benchmark_constant_programs() -> None:
    prompt = (
        "import math\n\n\ndef double(x):\n    '''Double it.\n    >>> double(2)\n    4\n    '''\n"
    )
    task = pool_task(
        "humaneval/0", prompt=prompt, split="eval", entry_point="double", public_tests="x"
    )
    programs = dict(benchmark_constant_programs(task, "humaneval"))
    assert set(programs) >= {"constant None", "public example 4", "echo"}
    assert programs["public example 4"] == (
        prompt + "    pass\n\n\ndef double(*args, **kwargs):\n    return 4\n\n\n"
    )
    mbpp = pool_task(
        "mbpp/2",
        split="eval",
        public_tests="import math\nassert set(shared((1, 2), (2, 3))) == set((2,))",
    )
    mbpp_programs = dict(benchmark_constant_programs(mbpp, "mbpp-plus"))
    assert mbpp_programs["echo"] == (
        "import math\n\ndef shared(*args, **kwargs):\n    return args[0]\n\n\n"
    )
    assert benchmark_constant_programs(pool_task("he/1", split="eval"), "humaneval") == []


def test_arc_baseline_from_labels_alone() -> None:
    def item(identifier: str, labels: tuple[str, ...], answer: str) -> ChoiceTask:
        return ChoiceTask(
            id=identifier,
            question="q",
            labels=labels,
            choices=tuple(f"c{label}" for label in labels),
            answer=answer,
            source="allenai/ai2_arc",
            source_revision="a" * 40,
            license="CC-BY-SA-4.0",
        )

    tasks = [
        item("1", ("A", "B", "C", "D"), "B"),
        item("2", ("A", "B", "C", "D"), "B"),
        item("3", ("A", "B", "C"), "A"),
        item("4", ("1", "2", "3", "4", "5"), "5"),
    ]
    report = arc_constant_baseline(tasks)
    assert report["best_position"] == 1 and report["best_position_accuracy"] == 0.5
    assert report["accuracy_by_position"] == {"0": 0.25, "1": 0.5, "2": 0.0, "3": 0.0, "4": 0.25}
    assert report["chance"] == pytest.approx((0.25 + 0.25 + 1 / 3 + 0.2) / 4)
