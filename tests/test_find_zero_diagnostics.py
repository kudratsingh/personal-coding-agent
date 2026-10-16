import hashlib
import json
import os
from dataclasses import replace

import pytest

from scratchcoder.data import evalplus_corrections
from scratchcoder.data.evalplus import origin_task, release
from scratchcoder.eval.find_zero_diagnostics import MARKER, diagnose_find_zero
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.runner import Result, Runtime, run


@pytest.fixture
def task(monkeypatch: pytest.MonkeyPatch) -> Task:
    task = origin_task(
        {
            "task_id": "HumanEval/32",
            "entry_point": "find_zero",
            "prompt": "def find_zero(xs):\n",
            "canonical_solution": "    if xs[1] == 0: raise ZeroDivisionError\n    return 0.0\n",
            "test": (
                "def _poly(xs,x): return sum(c*x**i for i,c in enumerate(xs))\n"
                "def check(candidate):\n"
                "    inputs = [([0,1],), ([1,1],), ([1,0],)]\n"
                "    results = [0,-1,0]\n"
                "    for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
                "        assert _poly(*candidate(*inp), inp) <= 0.0001\n"
            ),
        },
        release("humaneval-plus"),
    )
    monkeypatch.setattr(
        evalplus_corrections,
        "FIND_ZERO_ORIGINAL_SHA256",
        hashlib.sha256(task.hidden_tests.encode()).hexdigest(),
    )
    return task


def test_diagnostic_accounting_guards_and_no_stdout_leak(task: Task) -> None:
    observed = {
        "case_count": 3,
        "passed": 1,
        "residual_failures": 1,
        "exceptions": {"ZeroDivisionError": 1},
        "failure_examples": [],
    }

    def execute(code: str, tests: str, mode: str) -> Result:
        assert code == task.solution and mode == "function"
        assert "inputs = [([0,1],), ([1,1],), ([1,0],)]" in tests
        assert evalplus_corrections.CORRECTED_ASSERTION not in tests
        return Result("passed", MARKER + json.dumps(observed))

    report = diagnose_find_zero(task, release("humaneval-plus"), execute=execute)
    assert report["diagnostic_only"] and report["benchmark_acceptance"] is False
    assert report["status"] == "completed" and report["input_count"] == 3
    assert report["input_domain"]["zero_highest_coefficients"] == 1
    assert report["observations"] == observed
    assert task.hidden_tests not in json.dumps(report)
    failed = diagnose_find_zero(
        task,
        release("humaneval-plus"),
        execute=lambda *_: Result("failed", "TypeError: host-secret"),
    )
    assert failed["failure_category"] == "TypeError"
    assert "host-secret" not in json.dumps(failed)
    for stdout in (
        "no marker",
        MARKER + json.dumps(dict(observed, case_count=2)),
        MARKER + json.dumps(observed) + "\n" + MARKER + json.dumps(observed),
        MARKER + json.dumps(dict(observed, failure_examples=[{"raw": "host-secret"}])),
    ):
        with pytest.raises(ValueError):
            diagnose_find_zero(
                task,
                release("humaneval-plus"),
                execute=lambda *_, value=stdout: Result("passed", value),
            )
    with pytest.raises(ValueError, match="SHA-256"):
        diagnose_find_zero(
            replace(task, hidden_tests=task.hidden_tests + "\n"),
            release("humaneval-plus"),
            execute=execute,
        )


@pytest.mark.skipif(
    not os.environ.get("SCRATCHCODER_NUMPY_IMAGE"),
    reason="requires an explicitly built disposable NumPy image",
)
def test_diagnostic_all_cases_run_only_inside_sandbox(task: Task) -> None:
    runtime = Runtime(os.environ["SCRATCHCODER_NUMPY_IMAGE"], True)
    report = diagnose_find_zero(
        task,
        release("humaneval-plus"),
        execute=lambda code, tests, mode: run(code, tests, runtime=runtime, mode=mode),
    )
    assert report["status"] == "completed"
    observations = report["observations"]
    assert observations["case_count"] == 3 and observations["passed"] == 1
    assert observations["residual_failures"] == 1
    assert observations["exceptions"] == {"ZeroDivisionError": 1}
    assert observations["failure_examples"][0]["candidate_residual"] == "1.0"
    assert observations["failure_examples"][0]["release_expected_root_residual"] == "0"
