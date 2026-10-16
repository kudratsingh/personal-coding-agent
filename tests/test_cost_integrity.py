"""Pure estimate validation; no pricing lookup, provisioning or payment authority."""

import json
import math
import subprocess
import sys
from typing import Any

import pytest

from scratchcoder.common.cost import estimate


@pytest.mark.parametrize(
    "change",
    [
        {"tokens": True},
        {"tokens": 1.5},
        {"tokens": math.nan},
        {"tokens": math.inf},
        {"tokens": "10"},
        {"gpus": True},
        {"gpus": 1.5},
        {"gpus": math.nan},
        {"gpus": 0},
        {"tokens_per_second_per_gpu": True},
        {"tokens_per_second_per_gpu": "1"},
        {"dollars_per_gpu_hour": True},
        {"dollars_per_gpu_hour": "1"},
        {"dollars_per_gpu_hour": math.nan},
        {"slowdown": True},
        {"slowdown": "1"},
        {"layer_parameters": True, "reference_parameters": 1},
        {"layer_parameters": 1.5, "reference_parameters": 1},
        {"layer_parameters": math.nan, "reference_parameters": 1},
        {"layer_parameters": 1, "reference_parameters": math.inf},
        {"layer_parameters": 1},
        {"reference_parameters": 1},
    ],
)
def test_invalid_estimate_inputs_raise_value_error(change: dict[str, Any]) -> None:
    fields: dict[str, Any] = {
        "tokens": 100,
        "tokens_per_second_per_gpu": 10,
        "dollars_per_gpu_hour": 1,
    }
    fields.update(change)
    with pytest.raises(ValueError):
        estimate(**fields)


@pytest.mark.parametrize(
    "change",
    [
        {"tokens": 10**1000},
        {"tokens_per_second_per_gpu": 10**1000},
        {"tokens": 10**308, "tokens_per_second_per_gpu": 1e-308},
        {"tokens": 10**308, "dollars_per_gpu_hour": 1e308},
        {"layer_parameters": 10**1000, "reference_parameters": 1},
        {"layer_parameters": 1, "reference_parameters": 10**1000},
        {"gpus": 10**1000},
        {"tokens_per_second_per_gpu": 1e308, "slowdown": 1e-308},
        {"dollars_per_gpu_hour": 5e-324},
    ],
)
def test_unrepresentable_results_reject_instead_of_nan_inf_or_spurious_zero(
    change: dict[str, Any],
) -> None:
    fields: dict[str, Any] = {
        "tokens": 100,
        "tokens_per_second_per_gpu": 10,
        "dollars_per_gpu_hour": 1,
    }
    fields.update(change)
    with pytest.raises(ValueError):
        estimate(**fields)


def test_zero_tokens_and_zero_price_remain_valid() -> None:
    empty = estimate(0, 100, 2, gpus=2)
    assert empty.gpu_hours == empty.wall_hours == empty.dollars == 0
    free = estimate(360000, 100, 0, gpus=2)
    assert free.gpu_hours == 1 and free.wall_hours == 0.5 and free.dollars == 0


@pytest.mark.parametrize("tokens", [0, 1, 360000, 10**10])
def test_regular_estimate_stays_equal_to_original_formula(tokens: int) -> None:
    result = estimate(
        tokens,
        123.5,
        1.25,
        gpus=4,
        slowdown=1.2,
        layer_parameters=235000000,
        reference_parameters=110000000,
    )
    hours = tokens / 123.5 / 3600 * 1.2 * (235000000 / 110000000)
    assert result.gpu_hours == hours
    assert result.wall_hours == hours / 4
    assert result.dollars == hours * 1.25


def test_two_stage_cli_rejects_total_overflow_without_printing_json() -> None:
    # Each stage is finite, but their total is not.
    tokens = str(10**308)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.cost",
            "--stage1-tokens",
            tokens,
            "--stage2-tokens",
            tokens,
            "--tokens-per-second-per-gpu",
            "1",
            "--slowdown",
            "3600",
            "--dollars-per-gpu-hour",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2 and result.stdout == ""
    assert "finite" in result.stderr


def test_two_stage_cli_valid_json_is_unchanged() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.cost",
            "--stage1-tokens",
            "360000",
            "--stage2-tokens",
            "180000",
            "--tokens-per-second-per-gpu",
            "100",
            "--dollars-per-gpu-hour",
            "2",
            "--gpus",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["total"] == {"gpu_hours": 1.5, "wall_hours": 0.75, "dollars": 3.0}
