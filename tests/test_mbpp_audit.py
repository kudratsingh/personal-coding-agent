import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data.adapters import adapt_row
from scratchcoder.data.mbpp import MbppRelease, mbpp_release, read_mbpp
from scratchcoder.data.mbpp_audit import audit_mbpp
from scratchcoder.eval.choices import ChoiceTask
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.runner import Result


def _row(identifier: int) -> dict[str, Any]:
    return {
        "task_id": identifier,
        "text": "Authored fixture alpha: add one to an integer",
        "code": "def alpha(value): return value+1",
        "test_setup_code": "",
        "test_list": ["assert alpha(1)==2", "assert alpha(3)==4", "assert alpha(5)==6"],
        "challenge_test_list": ["assert alpha(999)==1000"],
    }


def _spec(rows: list[dict[str, Any]]) -> MbppRelease:
    raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    original = mbpp_release()
    return replace(
        original,
        task_count=len(rows),
        manifest=replace(
            original.manifest,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        ),
    )


def _protection() -> tuple[list[Task], list[ChoiceTask]]:
    task = Task(
        id="MBPP/2",
        prompt="Find repeated suffix words using the independently specified beta rule",
        public_tests="assert beta('one') == 1",
        hidden_tests="assert beta('two') == 2",
        source="fixture",
        source_revision="a" * 40,
        license="fixture",
        written_by="human",
        split="eval",
    )
    choice = ChoiceTask(
        id="fixture-choice",
        question="Which authored color fixture is green?",
        labels=("A", "B"),
        choices=("green", "purple"),
        answer="A",
        source="fixture",
        source_revision="a" * 40,
        license="fixture",
    )
    return [task], [choice]


def test_mbpp_pins_and_bounded_reader(tmp_path: Path) -> None:
    original = mbpp_release()
    original.validate()
    assert original.task_count == 974 and original.manifest.purpose == "eval"
    with pytest.raises(ValueError):
        replace(original, manifest=replace(original.manifest, purpose="train")).validate()
    rows = [_row(1), _row(2)]
    spec = _spec(rows)
    path = tmp_path / "fixture.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert read_mbpp(path, spec) == rows
    with pytest.raises(ValueError, match="count"):
        read_mbpp(path, replace(spec, task_count=3))
    with pytest.raises(ValueError, match="SHA-256"):
        read_mbpp(path, replace(spec, manifest=replace(spec.manifest, sha256="a" * 64)))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="size"):
        read_mbpp(path, spec)


@pytest.mark.parametrize(
    "change",
    [
        {"task_id": True},
        {"task_id": 1001},
        {"code": ""},
        {"test_list": []},
        {"challenge_test_list": "not-a-list"},
        {"test_setup_code": None},
        {"unreviewed": "x"},
    ],
)
def test_mbpp_reader_rejects_unreviewed_schema(tmp_path: Path, change: dict[str, Any]) -> None:
    row = dict(_row(1), **change)
    path = tmp_path / "fixture.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="schema"):
        read_mbpp(path, _spec([row]))


def test_mbpp_challenge_assertions_are_not_silently_discarded() -> None:
    row = _row(1)
    spec = _spec([row])
    task = adapt_row(spec.manifest.source, row, spec.manifest, split="eval")
    assert "999" not in task.public_tests and "999" in task.hidden_tests
    assert len(row["test_list"]) == 3  # adapter did not mutate the original arrays.
    with pytest.raises(ValueError, match="arrays"):
        adapt_row(
            spec.manifest.source, dict(row, challenge_test_list=None), spec.manifest, split="eval"
        )


def test_mbpp_audit_excludes_ids_then_preserves_challenge_tests_and_accounting() -> None:
    rows = [_row(1), _row(2), dict(_row(3), test_setup_code="unsupported=1")]
    rows[1]["test_setup_code"] = "must-not-be-normalized=1"
    benchmarks, choices = _protection()
    calls = []

    def execute(code: str, tests: str, mode: str) -> Result:
        assert code == rows[0]["code"] and mode == "function"
        calls.append(tests)
        return Result("passed")

    report = audit_mbpp(rows, _spec(rows), benchmarks, choices, execute=execute)
    assert (
        report["training_admission"] is False and report["complete_final_holdout_coverage"] is False
    )
    assert report["source_rows"] == 3
    assert report["source_test_counts"] == {"primary": 9, "challenge": 3}
    assert report["intake_counts"] == {
        "seen": 3,
        "normalized_for_pool_checks": 1,
        "benchmark_id_excluded": 1,
        "unsupported_assertion_schema": 1,
    }
    assert report["audit_kept_ids"] == ["mbpp/1"] and len(calls) == 2
    assert "999" in calls[1] and "999" not in calls[0]
    assert report["audit_split_summary"]["split:eval:total"] == 1
    assert rows[0]["code"] not in json.dumps(report)
    with pytest.raises(RuntimeError, match="isolation"):
        audit_mbpp(rows, _spec(rows), benchmarks, choices, execute=lambda *_: Result("unavailable"))


def test_mbpp_audit_choice_protection_aborts_before_execution() -> None:
    benchmarks, choices = _protection()
    rows = [dict(_row(1), text="Prefix " + choices[0].question + " suffix"), _row(2)]

    def forbidden(*_: str) -> Result:
        raise AssertionError("protected candidate must never execute")

    report = audit_mbpp(rows, _spec(rows), benchmarks, choices, execute=forbidden)
    assert report["intake_counts"]["choice_contamination"] == 1
    assert report["audit_kept_ids"] == []


def test_mbpp_audit_rejects_missing_protected_ids_before_execution() -> None:
    rows = [_row(1)]
    benchmarks, choices = _protection()
    with pytest.raises(ValueError, match="protected MBPP IDs"):
        audit_mbpp(rows, _spec(rows), benchmarks, choices, execute=lambda *_: Result("passed"))
