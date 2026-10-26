"""Rehearsal time, cost and storage fit: hand-checked arithmetic and the D-35 numbers."""

import json
from pathlib import Path

import pytest

from tools import rehearsal_ops
from tools.pilot_feasibility import (
    MEASURED_UNTRAINED_CHECKPOINT_BYTES,
    REHEARSAL_DISCLAIMER,
    RehearsalOptions,
    RehearsalPlan,
    build_rehearsal_report,
    default_rehearsal_plans,
    main,
    rehearsal_time_budget,
    rehearsal_total_hourly,
)

SMALL = RehearsalOptions(
    reference_tokens_per_second=131_072.0,
    minimum_speed_fraction=0.5,
    startup_seconds=100,
    compile_seconds_per_stage=10,
    validation_speed_factor=2.0,
    checkpoint_save_seconds=5,
    lifetime_seconds=5_000,
    self_destruct_margin_seconds=100,
    finalization_reserve_seconds=200,
    checkpoint_interval_seconds=600,
)
PLAN = RehearsalPlan("B", (("mixed", 1_000),), 2_000_000)


def test_time_budget_adds_every_phase_by_hand() -> None:
    reference = rehearsal_time_budget(PLAN, SMALL, SMALL.reference_tokens_per_second)
    validation_rate = 2 * 131_072
    # Full validations at updates 250, 500 and 750 and at the stage end.
    running = 1_000 + 2 * 2_000_000 / validation_rate + 4 * 2_000_000 / validation_rate + 10
    assert reference["training_seconds"] == pytest.approx(1_000)
    assert reference["interval_validations"] == 2
    assert reference["full_validations"] == 4
    assert reference["checkpoints"] == 2  # One 600-second save during training, one at the end.
    assert reference["total_seconds"] == pytest.approx(100 + running + 2 * 5 + 200)
    assert reference["usable_lifetime_seconds"] == 4_900 and reference["fits"]
    assert reference["latest_startup_seconds"] == pytest.approx(4_900 - (running + 10 + 200))
    floor = rehearsal_time_budget(PLAN, SMALL, SMALL.floor_tokens_per_second)
    assert floor["training_seconds"] == pytest.approx(2_000)
    assert floor["checkpoints"] == 4
    late = rehearsal_time_budget(PLAN, SMALL, SMALL.floor_tokens_per_second, startup_seconds=2_700)
    assert not late["fits"] and late["margin_seconds"] < 0


def test_d35_runs_fit_at_the_reference_speed_and_the_quote_bounds_spend() -> None:
    report = build_rehearsal_report(default_rehearsal_plans(), RehearsalOptions(), source="S")
    assert report["quoted_total_hourly"] == 0.60
    assert report["floor_tokens_per_second"] == pytest.approx(0.85 * 56_479)
    assert report["two_runs_wall_bound_usd"] == pytest.approx(15.0)
    for run, updates in (("A", 15_260), ("B", 15_259)):
        row = report["runs"][run]
        assert row["updates"] == updates and row["tokens_per_update"] == 131_072
        assert row["reference"]["fits"]
        assert row["floor"]["margin_seconds"] < row["reference"]["margin_seconds"]
        assert row["time_fit_at_floor"] is row["floor"]["fits"]
        assert row["cost"]["wall_bound_usd"] == pytest.approx(7.5)
        assert row["cost"]["fits"] and row["cost"]["wall_within_cost_stop"]
        assert row["container_disk"]["fits"]
    # Both full streams at each quarter and end of every stage: 8 passes for A, 4 for B.
    assert report["runs"]["A"]["reference"]["full_validations"] == 8
    assert report["runs"]["B"]["reference"]["full_validations"] == 4
    assert report["r2_storage"]["monthly_usd"] < 1
    assert report["disclaimer"] == REHEARSAL_DISCLAIMER
    # The upload rate the receipt requires: one label of at least the measured checkpoint
    # size in the 360 s drain, with a factor-2 margin; the last 420 s are the SSH pull's.
    upload = report["r2_upload"]
    assert upload["checkpoint_bytes"] >= MEASURED_UNTRAINED_CHECKPOINT_BYTES
    assert upload["drain_seconds"] == 360 and upload["pull_reserve_seconds"] == 420
    assert upload["required_upload_bytes_per_second"] == pytest.approx(
        upload["checkpoint_bytes"] / 360 * 2
    )
    assert (
        upload["required_upload_bytes_per_second"] == rehearsal_ops.REQUIRED_UPLOAD_BYTES_PER_SECOND
    )


def test_a_higher_quote_breaks_the_cost_stop() -> None:
    assert rehearsal_total_hourly(0.59, 40) == 0.60
    assert rehearsal_total_hourly(0.61, 40) == 0.62
    report = build_rehearsal_report(
        default_rehearsal_plans(), RehearsalOptions(quoted_gpu_hourly=0.61), source="S"
    )
    for row in report["runs"].values():
        assert row["cost"]["wall_bound_usd"] == pytest.approx(7.75)
        assert not row["cost"]["fits"] and not row["cost"]["wall_within_cost_stop"]


@pytest.mark.parametrize(
    "change",
    [
        {"minimum_speed_fraction": 1.5},
        {"reference_tokens_per_second": 0},
        {"startup_seconds": -1},
        {"self_destruct_margin_seconds": 45_000},
        {"validation_interval_updates": 0},
    ],
)
def test_options_are_validated(change: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RehearsalOptions(**change)  # type: ignore[arg-type]


def test_cli_prints_the_report_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "feasibility.json"
    assert main(["rehearsal", "--quoted-gpu-hourly", "0.59", "--output", str(output)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("# Rehearsal feasibility (D-35)")
    assert REHEARSAL_DISCLAIMER in printed and "| A | 15,260 |" in printed
    assert "1 label of 1.65 GB in the 360 s drain with a 2x margin needs 9.17 MB/s" in printed
    saved = json.loads(output.read_text())
    assert saved["kind"] == "rehearsal-feasibility" and set(saved["runs"]) == {"A", "B"}
    with pytest.raises(SystemExit):
        main(["rehearsal", "--binding-sha256", "ab" * 32])
