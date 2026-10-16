"""Tiny-only tests for the bounded, network-free CPU reference tool."""

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from scratchcoder.model.gpt import expected_parameters
from tools import cpu_resume_check as check


def test_tiny_check_exercises_both_precisions_and_preserves_threads(tmp_path: Path) -> None:
    threads = torch.get_num_threads()
    report = check.run_checks(
        tmp_path / "run",
        size="tiny",
        seeds=[7],
        precisions=["float32", "bfloat16"],
        steps=4,
        progress=lambda _: None,
    )
    assert report["status"] == "completed" and report["planned_cases"] == 2
    assert len(report["cases"]) == 2 and report["threads"] == 1
    assert torch.get_num_threads() == threads
    assert all(case["exact_resume"] and case["split_step"] == 2 for case in report["cases"])
    assert report["cases"][0]["source_hash"] == report["cases"][1]["source_hash"]


@pytest.mark.parametrize(
    "change",
    [
        {"size": "large"},
        {"seeds": []},
        {"seeds": [7, 7]},
        {"seeds": [-1]},
        {"seeds": [True]},
        {"seeds": [2**63]},
        {"seeds": list(range(9))},
        {"precisions": []},
        {"precisions": ["float32", "float32"]},
        {"precisions": ["float16"]},
        {"steps": True},
        {"steps": 1},
        {"steps": 17},
    ],
)
def test_invalid_requests_fail_before_output(tmp_path: Path, change: dict[str, Any]) -> None:
    values: dict[str, Any] = {"size": "tiny", "seeds": [7], "precisions": ["float32"], "steps": 4}
    values.update(change)
    with pytest.raises(ValueError):
        check.run_checks(tmp_path / "run", **values)
    assert not list(tmp_path.iterdir())


def test_existing_output_is_not_modified(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        check.run_checks(tmp_path, size="tiny", seeds=[7], precisions=["float32"], steps=4)
    assert not list(tmp_path.iterdir())


def test_failed_comparison_leaves_failed_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("authored mismatch")

    monkeypatch.setattr(check, "run_case", fail)
    with pytest.raises(AssertionError):
        check.run_checks(
            tmp_path / "run",
            size="tiny",
            seeds=[7],
            precisions=["float32"],
            steps=4,
            progress=lambda _: None,
        )
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["status"] == "failed" and report["error_type"] == "AssertionError"
    assert report["cases"] == [] and "authored mismatch" not in str(report)


@pytest.mark.parametrize(
    "actual", [torch.tensor([1.01]), torch.tensor([float("nan")]), [1.0], (1,), {"x": 2}]
)
def test_exact_comparator_detects_state_changes(actual: Any) -> None:
    with pytest.raises(AssertionError):
        check.assert_identical(actual, torch.tensor([1.0]))


def test_nano_preset_matches_planned_shape_without_allocating_weights() -> None:
    config = check.configuration("nano", "float32", 7, 8)
    assert (
        config.model.width,
        config.model.layers,
        config.model.heads,
        config.model.context,
        config.model.vocab_size,
    ) == (256, 4, 4, 256, 32768)
    assert expected_parameters(config.model) == 11_536_640


@pytest.mark.parametrize(
    "actual,reference", [({"x": 2}, {"x": 1}), ([1.0], [1]), ((1,), [1]), (True, 1)]
)
def test_exact_comparator_checks_nested_values_and_types(actual: Any, reference: Any) -> None:
    with pytest.raises(AssertionError):
        check.assert_identical(actual, reference)


def test_optimized_python_cannot_disable_scalar_verification() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "from tools.cpu_resume_check import assert_identical; "
            "assert_identical({'step': 3}, {'step': 4})",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0 and "AssertionError" in result.stderr
