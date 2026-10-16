import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scratchcoder.data import evalplus_corrections as correction
from scratchcoder.data.evalplus import origin_task, release
from scratchcoder.eval import suites
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.runner import Limits, Result, Runtime, run


@pytest.fixture
def task(monkeypatch: pytest.MonkeyPatch) -> Task:
    # Authored fixture, not copied benchmark inputs. Both roots are valid.
    test = (
        "def _poly(coefficients, x):\n"
        "    return sum(c*x**i for i,c in enumerate(coefficients))\n"
        "def check(candidate):\n"
        "    inputs = [([-1,0,1],)]\n"
        "    for inp in inputs:\n"
        f"        {correction.BROKEN_ASSERTION}\n"
    )
    result = origin_task(
        {
            "task_id": "HumanEval/32",
            "entry_point": "find_zero",
            "prompt": "def find_zero(coefficients):\n",
            "canonical_solution": "    return 1.0\n",
            "test": test,
        },
        release("humaneval-plus"),
    )
    monkeypatch.setattr(
        correction,
        "FIND_ZERO_ORIGINAL_SHA256",
        hashlib.sha256(result.hidden_tests.encode()).hexdigest(),
    )
    return result


def test_correction_changes_only_one_expression_and_records_fingerprints(task: Task) -> None:
    corrected, record = correction.correct_find_zero([task], release("humaneval-plus"))
    expected = task.hidden_tests.replace(
        correction.BROKEN_ASSERTION, correction.CORRECTED_ASSERTION
    )
    assert corrected == [replace(task, hidden_tests=expected)]
    # Exact reverse comparison covers every input literal and all unrelated script bytes.
    assert (
        expected.replace(correction.CORRECTED_ASSERTION, correction.BROKEN_ASSERTION)
        == task.hidden_tests
    )
    assert record["version"] == "find-zero-residual-v1"
    assert record["task_id"] == task.id
    assert (
        record["original_hidden_script_sha256"]
        == hashlib.sha256(task.hidden_tests.encode()).hexdigest()
    )
    assert record["effective_hidden_script_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    assert (
        record["effective_task_sha256"]
        == hashlib.sha256(json.dumps(asdict(corrected[0]), sort_keys=True).encode()).hexdigest()
    )


def test_correction_rejects_unreviewed_inputs(task: Task, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = release("humaneval-plus")
    for other in (release("mbpp-plus"), replace(spec, task_count=1)):
        with pytest.raises(ValueError, match="exact pinned"):
            correction.correct_find_zero([task], other)
    for tasks in ([], [task, task], [replace(task, id="humaneval/31")]):
        with pytest.raises(ValueError, match="exactly one task"):
            correction.correct_find_zero(tasks, spec)
    for altered in (
        replace(task, source_revision="a" * 40),
        replace(task, source="other/source"),
        replace(task, entry_point="other"),
        replace(task, split="train"),
    ):
        with pytest.raises(ValueError, match="provenance"):
            correction.correct_find_zero([altered], spec)
    with pytest.raises(ValueError, match="SHA-256"):
        correction.correct_find_zero([replace(task, hidden_tests=task.hidden_tests + "\n")], spec)
    for script in (
        task.hidden_tests + "\n" + correction.BROKEN_ASSERTION,
        task.hidden_tests.replace(correction.BROKEN_ASSERTION, "assert True"),
    ):
        monkeypatch.setattr(
            correction, "FIND_ZERO_ORIGINAL_SHA256", hashlib.sha256(script.encode()).hexdigest()
        )
        with pytest.raises(ValueError, match="exactly one original assertion"):
            correction.correct_find_zero([replace(task, hidden_tests=script)], spec)


def test_suite_default_unchanged_and_corrected_metadata_shared(
    task: Task, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = release("humaneval-plus")
    tasks = [replace(task, id=f"humaneval/{index}") for index in range(spec.task_count)]
    original = suites.evaluate_suite(
        tasks,
        spec,
        execute=lambda *_: Result("passed"),
        runtime=Runtime(),
        limits=Limits(),
        output=tmp_path / "original",
    )["canonical"]
    assert original["benchmark"]["oracle_policy"] == "unmodified-OriginFmt-tests"
    assert original["benchmark"]["oracle_corrections"] == []
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test-only")
    monkeypatch.setattr(
        suites, "load_checkpoint", lambda *_: (None, SimpleNamespace(sha256="fixture"))
    )
    monkeypatch.setattr(suites, "checkpoint_generator", lambda *_: lambda task, seed: "fixture")
    executed = []

    def execute(code: str, tests: str, mode: str) -> Result:
        executed.append(tests)
        return Result("passed")

    reports = suites.evaluate_suite(
        tasks,
        spec,
        execute=execute,
        runtime=Runtime(),
        limits=Limits(),
        output=tmp_path / "corrected",
        checkpoint=checkpoint,
        tokenizer_path=tmp_path / "tokenizer.json",
        correct_find_zero_oracle=True,
    )
    for report in reports.values():
        assert report["task_hash"] != original["task_hash"]
        assert report["benchmark"]["oracle_policy"] == "OriginFmt-with-find-zero-residual-v1"
        assert report["benchmark"]["oracle_corrections"][0]["task_id"] == "humaneval/32"
        assert task.hidden_tests not in json.dumps(report)
    assert (
        reports["canonical"]["benchmark"]["oracle_corrections"]
        == reports["checkpoint"]["benchmark"]["oracle_corrections"]
    )
    assert sum(correction.CORRECTED_ASSERTION in tests for tests in executed) == 2


@pytest.mark.skipif(
    not os.environ.get("SCRATCHCODER_NUMPY_IMAGE"),
    reason="requires an explicitly built disposable NumPy image",
)
def test_residual_oracle_accepts_distinct_roots_not_wrong_root_in_sandbox(task: Task) -> None:
    runtime = Runtime(os.environ["SCRATCHCODER_NUMPY_IMAGE"], True)
    corrected, _ = correction.correct_find_zero([task], release("humaneval-plus"))
    for root in (-1.0, 1.0):
        code = f"def find_zero(coefficients): return {root!r}"
        assert run(code, corrected[0].hidden_tests, runtime=runtime).passed
        assert run(code, task.hidden_tests, runtime=runtime).status == "failed"
    # Negative residual must not be accepted without abs().
    assert (
        run(
            "def find_zero(coefficients): return 0.0", corrected[0].hidden_tests, runtime=runtime
        ).status
        == "failed"
    )
