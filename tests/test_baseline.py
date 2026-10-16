from dataclasses import replace

import pytest

from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.runner import Result
from tools.baselines.humaneval import (
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
