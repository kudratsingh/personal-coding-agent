import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data.opencode import (
    OpenCodeSample,
    audit_opencode,
    opencode_sample,
    read_opencode_sample,
)
from scratchcoder.eval.choices import ChoiceTask
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.runner import Result


def row(identifier: str = "1") -> dict[str, Any]:
    return {
        "id": identifier,
        "input": "Authored alpha fixture that increments its integer argument",
        "output": "def alpha(value): return value+1",
        "domain": "generic",
        "generation_algorithm": "synthetic-fixture",
        "llm_judgement": "{}",
        "unit_tests": json.dumps(["assert alpha(1)==2", "assert alpha(9)==10"]),
        "tests_execution_status": '["pass", "pass"]',
        "average_test_score": 1.0,
    }


def protection() -> tuple[list[Task], list[ChoiceTask]]:
    benchmark = Task(
        id="humaneval/0",
        prompt="Distinct protected benchmark prompt with beta sorting rules",
        public_tests="assert beta('word') == 1",
        hidden_tests="assert beta('other') == 2",
        source="fixture",
        source_revision="a" * 40,
        license="fixture",
        written_by="human",
        split="eval",
    )
    choice = ChoiceTask(
        id="choice",
        question="Which independently authored answer describes the green fixture?",
        labels=("A", "B"),
        choices=("green", "purple"),
        answer="A",
        source="fixture",
        source_revision="a" * 40,
        license="fixture",
    )
    return [benchmark], [choice]


def test_pin_and_eval_only_bounds() -> None:
    spec = opencode_sample()
    spec.validate()
    assert spec.sample_rows == 200 and spec.file_rows == 100000
    assert spec.manifest.size_bytes == 133476478
    assert (
        spec.manifest.sha256 == "342757e0c6b706c8f68cf0a867ead5c417d0273726453dbcfc934f5c3c6ca891"
    )
    for changed in (
        replace(spec, sample_rows=201),
        replace(spec, sample_rows=True),
        replace(spec, file_rows=0),
        replace(spec, filename="../private.parquet"),
        replace(spec, manifest=replace(spec.manifest, purpose="train")),
        replace(spec, manifest=replace(spec.manifest, size_bytes=161 * 1024 * 1024)),
    ):
        with pytest.raises(ValueError):
            changed.validate()


def test_audit_accounting_protection_no_raw_output_or_training() -> None:
    benchmarks, choices = protection()
    rows = [row(str(index)) for index in range(5)]
    rows[1]["unit_tests"] = '["raise RuntimeError(1)"]'
    rows[2]["input"] = choices[0].question
    rows[3]["input"] = benchmarks[0].prompt
    rows[4]["input"] = "Count digits under an unrelated authored gamma fixture specification"
    rows[4]["output"] = "def alpha(value): return 99"
    calls = []

    def execute(code: str, tests: str, mode: str) -> Result:
        calls.append(tests)
        assert mode == "function"
        return Result("failed", stdout="PRIVATE_RAW_STDOUT") if "99" in code else Result("passed")

    report = audit_opencode(
        rows,
        replace(opencode_sample(), sample_rows=5),
        benchmarks,
        choices,
        execute=execute,
    )
    assert report["intake_counts"] == {
        "seen": 5,
        "normalized_for_pool_checks": 3,
        "unsupported_assertion_schema": 1,
        "choice_contamination": 1,
    }
    assert report["pool_counts"]["kept"] == report["pool_counts"]["invalid_solution"] == 1
    assert report["pool_counts"]["contamination_or_duplicate"] == 1
    assert report["audit_kept_ids"] == ["nvidia/OpenCodeInstruct/0"]
    assert report["audit_split_summary"]["split:eval:authorship:model"] == 1
    assert len(calls) == 4
    assert report["training_admission"] is False
    assert report["complete_final_holdout_coverage"] is False
    assert report["representative_source_sample"] is False
    assert rows[0]["output"] not in json.dumps(report) and "PRIVATE_RAW_STDOUT" not in json.dumps(
        report
    )
    assert "9" not in calls[0] and "9" in calls[1]  # Hidden assertion not stripped.


@pytest.mark.parametrize("status", ["unavailable", "unknown"])
def test_missing_isolation_aborts(status: str) -> None:
    with pytest.raises(RuntimeError, match="isolation"):
        audit_opencode(
            [row()],
            replace(opencode_sample(), sample_rows=1),
            *protection(),
            execute=lambda *_: Result(status),
        )


def test_missing_protection_and_duplicate_ids_fail_before_execution() -> None:
    def forbidden(*_: str) -> Result:
        raise AssertionError("invalid input executed")

    with pytest.raises(ValueError, match="protection"):
        audit_opencode(
            [row()], replace(opencode_sample(), sample_rows=1), [], [], execute=forbidden
        )
    with pytest.raises(ValueError, match="unique"):
        audit_opencode(
            [row(), row()],
            replace(opencode_sample(), sample_rows=2),
            *protection(),
            execute=forbidden,
        )


def write_fixture(
    path: Path, rows: list[dict[str, Any]], *, groups: bool = False
) -> OpenCodeSample:
    arrow = pytest.importorskip("pyarrow", reason="requires the locked evaluation extra")
    parquet = pytest.importorskip("pyarrow.parquet")
    parquet.write_table(arrow.Table.from_pylist(rows), path, row_group_size=1 if groups else None)
    original = opencode_sample()
    return replace(
        original,
        file_rows=len(rows),
        sample_rows=min(2, len(rows)),
        manifest=replace(
            original.manifest,
            size_bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ),
    )


def test_bounded_reader_preserves_exact_prefix_and_pins(tmp_path: Path) -> None:
    rows = [row(str(index)) for index in range(3)]
    path = tmp_path / "fixture.parquet"
    spec = write_fixture(path, rows)
    assert read_opencode_sample(path, spec) == rows[:2]
    with pytest.raises(ValueError, match="count"):
        read_opencode_sample(path, replace(spec, file_rows=4))
    with pytest.raises(ValueError, match="SHA-256"):
        read_opencode_sample(path, replace(spec, manifest=replace(spec.manifest, sha256="a" * 64)))
    with pytest.raises(ValueError, match="size"):
        read_opencode_sample(path, replace(spec, manifest=replace(spec.manifest, size_bytes=1)))


@pytest.mark.parametrize(
    "change",
    [
        {"id": None},
        {"id": ""},
        {"input": "x" * 128001},
        {"output": ""},
        {"average_test_score": float("nan")},
        {"average_test_score": 1.1},
        {"average_test_score": "1.0"},
        {"unreviewed_field": "fixture"},
    ],
)
def test_unreviewed_parquet_rows_fail(tmp_path: Path, change: dict[str, Any]) -> None:
    path = tmp_path / "fixture.parquet"
    spec = write_fixture(path, [dict(row(), **change)])
    with pytest.raises(ValueError):
        read_opencode_sample(path, spec)


def test_groups_batch_and_unique_id_bounds(tmp_path: Path) -> None:
    path = tmp_path / "fixture.parquet"
    spec = write_fixture(path, [row(str(index)) for index in range(9)], groups=True)
    with pytest.raises(ValueError, match="bounds"):
        read_opencode_sample(path, spec)
    spec = write_fixture(path, [row(), row()])
    with pytest.raises(ValueError, match="unique"):
        read_opencode_sample(path, spec)
    rows = [dict(row(str(index)), input="x" * 120000) for index in range(40)]
    spec = replace(write_fixture(path, rows), sample_rows=40)
    with pytest.raises(ValueError, match="batch"):
        read_opencode_sample(path, spec)
