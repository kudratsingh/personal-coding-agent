"""Run 2's cost projection: the proposal's table at the planning speeds, with the Mac twice as
slow and at run 1's measured rates, against the ceiling fixed at the gate."""

import json
from pathlib import Path

import pytest

from tools.pilot_feasibility import (
    BRIDGE_ROUND_TRIP_SECONDS,
    main,
    posttrain2_report,
    render_posttrain2_markdown,
    run1_measured_rates,
)

TEMPLATE = Path(__file__).resolve().parents[1] / "configs" / "rehearsal-a40-s-post-2.json"


def test_run_two_fits_its_ceiling_at_the_planning_speeds() -> None:
    report = posttrain2_report(TEMPLATE, ceiling_usd=5.80)
    planned = report["scenarios"]["planning speeds"]
    rows = planned["rows"]
    hours = {key: row["hours"] for key, row in rows.items()}
    assert hours["setup"] == pytest.approx(0.25)
    assert hours["rl_rollouts"] == pytest.approx(100 * 256 * 256 / 10_000 / 3600)  # 0.18
    assert hours["rl_training"] == pytest.approx(100 * 256 * 1_024 / 56_765 / 3600)  # 0.13
    assert hours["rl_reward_waits"] == pytest.approx(100 * 48 / 3600)  # 1.33
    generation, wait = hours["rl_dev_checks"]
    assert generation == pytest.approx(6 * 1_800 * 512 / 5_000 / 3600)  # 0.31
    assert wait == pytest.approx(6 * 300 / 3600)  # 0.50
    assert hours["control_training"] == pytest.approx(hours["rl_rollouts"] + hours["rl_training"])
    assert hours["control_dev_checks"] == pytest.approx(
        [5 * 1_800 * 512 / 5_000 / 3600, 5 * 300 / 3600]
    )
    assert planned["evaluation_tokens"] == 2 * 18_792_960  # 37.6M tokens at cap.
    assert hours["evaluation"] == pytest.approx((14 * 20 + 2 * 18_792_960 / 5_000) / 3600)
    assert planned["total_hours"] == pytest.approx(6.17, abs=0.01)
    assert planned["total_usd"] == pytest.approx(3.70, abs=0.01)
    assert planned["fits_lifetime"] and planned["fits_ceiling"]
    assert report["wait_margin_usd"] == pytest.approx(10_889.5 / 3600 * 0.6)
    slow = report["scenarios"]["planning speeds, Mac twice as slow"]
    assert slow["total_usd"] == pytest.approx(5.05, abs=0.01) and slow["fits_ceiling"]
    assert report["limits"]["watchdog_cost_stop_usd"] == 5.79


def test_the_template_needs_a_ceiling_to_price_against() -> None:
    with pytest.raises(ValueError, match="--ceiling-usd"):
        posttrain2_report(TEMPLATE)


def run1_records(tmp_path: Path) -> tuple[Path, Path, Path]:
    report = {
        "kind": "posttrain-run-report",
        "run": "1",
        "probe": {
            "measured_tokens_per_second": {
                "1": 90.0,
                "64": 7_000.0,
                "128": 9_000.0,
                "256": 14_000.0,
            }
        },
        "rates": {"generation_tokens_per_second": 6_000.0, "arc_seconds_per_checkpoint": 40.0},
        "sft": {"slots_per_second": 60_000.0},
    }
    screening = {
        "format": "rl-screening-v1",
        "rewards": [{"task_id": "t", "sample": index, "seconds": 2.0} for index in range(8)]
        + [{"task_id": "u", "sample": index, "seconds": 0.0} for index in range(8)],
    }
    dev = [{"kind": "header"}] + [{"kind": "attempt", "elapsed_seconds": 1.2} for _ in range(5)]
    paths = (tmp_path / "report.json", tmp_path / "screening.json", tmp_path / "dev.jsonl")
    paths[0].write_text(json.dumps(report))
    paths[1].write_text(json.dumps(screening))
    paths[2].write_text("".join(json.dumps(row) + "\n" for row in dev))
    return paths


def test_run_ones_records_give_the_measured_rates(tmp_path: Path) -> None:
    report, screening, dev = run1_records(tmp_path)
    rates = run1_measured_rates(report, screening, dev_results=[dev])
    # The evaluation files ran at 6/7 of the probe's batch-64 rate: batch 256 is slowed alike.
    assert rates["generation_tokens_per_second"] == {"64": 6_000.0, "256": pytest.approx(12_000.0)}
    assert rates["training_slots_per_second"] == 60_000.0
    assert rates["reward_wait_seconds_per_update"] == pytest.approx(256 * 1.0 / 8 + 5.5)
    assert rates["dev_wait_seconds_per_check"] == pytest.approx(1_800 * 1.2 / 8 + 5.5)
    assert BRIDGE_ROUND_TRIP_SECONDS == 2 + 3 * 1 + 0.5
    projected = posttrain2_report(TEMPLATE, ceiling_usd=5.80, measured=rates)
    measured = projected["scenarios"]["run 1's measured rates"]
    assert measured["rows"]["rl_reward_waits"]["seconds"] == pytest.approx(100 * (32 + 5.5))
    assert measured["rows"]["rl_rollouts"]["seconds"] == pytest.approx(100 * 256 * 256 / 12_000)
    assert measured["total_usd"] < projected["scenarios"]["planning speeds"]["total_usd"] + 1
    text = render_posttrain2_markdown(projected)
    assert "## run 1's measured rates" in text and "| **Total** |" in text


def test_the_cli_prints_the_proposals_table_for_run_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "feasibility-2.json"
    argv = ["posttrain", "--specification", str(TEMPLATE), "--ceiling-usd", "5.80"]
    assert main([*argv, "--quoted-gpu-hourly", "0.59", "--output", str(output)]) == 0
    text = capsys.readouterr().out
    assert "| **Total** | **6.17** | **$3.70** |" in text
    assert "| RL dev checks: generation / Mac wait | 0.31 / 0.50 |" in text
    assert json.loads(output.read_text())["run"] == "2"
    report, screening, _ = run1_records(tmp_path)
    assert main([*argv, "--run1-report", str(report), "--screening", str(screening)]) == 0
    assert "## run 1's measured rates" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([*argv, "--run1-report", str(report)])  # Both records, or neither.
