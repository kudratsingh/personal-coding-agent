from dataclasses import replace

import pytest

from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.runner import Result
from tools.baselines.humaneval import (
    completion_diagnostics,
    failure_category,
    normalize,
    truncate_completion,
    validate_references,
)


def test_humaneval_mapping_and_stop_words() -> None:
    rows = [
        {
            "task_id": "synthetic/0",
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
            "canonical_solution": "    return a+b\n",
            "test": "def check(candidate): assert candidate(1,2) == 3",
        }
    ]
    task = normalize(rows)[0]
    assert task.split == "eval"
    assert task.hidden_tests.endswith("check(add)")
    assert task.solution == "def add(a, b):\n    return a+b\n"
    assert "candidate(1,2)" not in task.public_prompt()
    assert truncate_completion("    return a+b\n\ndef other(): pass") == "    return a+b\n"


def test_reference_validation_uses_solution_and_fails_closed() -> None:
    task = Task(
        id="fixture",
        prompt="question",
        public_tests="public",
        hidden_tests="hidden-test",
        source="synthetic",
        license="fixture",
        written_by="human",
        split="test",
    )
    tasks = [replace(task, solution="known answer")]

    def execute(code: str, tests: str, mode: str) -> Result:
        assert code == "known answer" and tests == "hidden-test" and mode == "function"
        return Result("passed")

    report = validate_references(tasks, execute=execute)
    assert report["metrics"]["greedy_pass_at_1"] == 1
    with pytest.raises(RuntimeError, match="aborted"):
        validate_references(tasks, execute=lambda *_: Result("unavailable"))
    with pytest.raises(ValueError, match="missing"):
        validate_references([task], execute=execute)


def test_failure_categories_never_retain_raw_output() -> None:
    assert failure_category(Result("failed", "private-test\nSyntaxError: secret")) == "SyntaxError"
    assert failure_category(Result("failed", "secret")) == "other_failure"
    assert failure_category(Result("timeout", "secret")) == "timeout"


@pytest.mark.parametrize(
    "completion",
    [
        "    return 3\n",
        "    return 3\n\ndef other(): pass",
        "\n# host-secret",
        "    return (",
        "    return 'host-secret'\nclass Ignore: pass",
        "\x00",
    ],
)
def test_generation_diagnostics_observe_existing_stops_without_source_leak(completion: str) -> None:
    import json

    retained = truncate_completion(completion)
    diagnostics = completion_diagnostics(
        "def fixture():\n", completion, [7, 8], budget=2, eos_token_id=9
    )
    assert diagnostics["retained_characters"] == len(retained)
    assert diagnostics["decoded_characters"] == len(completion)
    assert diagnostics["budget_reached"] and diagnostics["budget_without_eos"]
    assert not diagnostics["eos_seen"]
    assert diagnostics["empty_retained_completion"] == (not retained.strip())
    assert "host-secret" not in json.dumps(diagnostics)
    assert truncate_completion(completion) == retained  # instrumentation never changes the source.


def test_generation_budget_eos_and_first_stop_accounting() -> None:
    row = completion_diagnostics(
        "def fixture():\n",
        "    return 3\n# stop\ndef later(): pass",
        [8, 9],
        budget=2,
        eos_token_id=9,
    )
    assert row["budget_reached"] and row["eos_seen"] and not row["budget_without_eos"]
    assert row["candidate_syntax_valid"] and row["stop_index"] == 4
    short = completion_diagnostics(
        "def fixture():\n", "    return 3", [9], budget=2, eos_token_id=9
    )
    assert not short["budget_reached"] and short["eos_seen"]
    invalid = completion_diagnostics(
        "def fixture():\n", "    return (", [], budget=2, eos_token_id=None
    )
    assert not invalid["candidate_syntax_valid"]
    for tokens in ([True], [-1], [1, 2, 3]):
        with pytest.raises(ValueError, match="generation"):
            completion_diagnostics("", "", tokens, budget=2, eos_token_id=None)
