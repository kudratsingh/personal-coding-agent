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
from tools.posttrain_ops import estimate_limit_usd

TEMPLATE = Path(__file__).resolve().parents[1] / "configs" / "rehearsal-a40-s-post-2.json"
RAW = json.loads(TEMPLATE.read_text())
BATCHES = {
    "generation_batch": RAW["generation"]["batch_size"],
    "rl_batches": (RAW["rl"]["generation_batch"], RAW["rl"]["dev_batch_size"]),
}
"""The batches run 2 generates at, from its template: the evaluation's (run 1's, 128) and RL's
rollouts' and dev checks' (256)."""


def test_run_two_fits_its_ceiling_at_the_planning_speeds() -> None:
    report = posttrain2_report(TEMPLATE, ceiling_usd=5.80)
    planned = report["scenarios"]["planning speeds"]
    rows = planned["rows"]
    hours = {key: row["hours"] for key, row in rows.items()}
    assert hours["setup"] == pytest.approx(0.25)
    assert hours["rl_rollouts"] == pytest.approx(100 * 256 * 256 / 9_268 / 3600)  # 0.20
    assert hours["rl_training"] == pytest.approx(100 * 256 * 1_024 / 56_765 / 3600)  # 0.13
    assert hours["rl_reward_waits"] == pytest.approx(100 * 48 / 3600)  # 1.33
    generation, wait = hours["rl_dev_checks"]
    assert generation == pytest.approx(6 * 1_800 * 512 / 9_268 / 3600)  # 0.17 at batch 256
    assert wait == pytest.approx(6 * 300 / 3600)  # 0.50
    assert hours["control_training"] == pytest.approx(hours["rl_rollouts"] + hours["rl_training"])
    assert hours["control_dev_checks"] == pytest.approx(
        [5 * 1_800 * 512 / 9_268 / 3600, 5 * 300 / 3600]
    )
    assert planned["evaluation_tokens"] == 2 * 18_792_960  # 37.6M tokens at cap.
    assert hours["evaluation"] == pytest.approx((14 * 20 + 2 * 18_792_960 / 8_279) / 3600)
    assert planned["total_hours"] == pytest.approx(5.11, abs=0.01)
    assert planned["total_usd"] == pytest.approx(3.07, abs=0.01)
    assert planned["fits_lifetime"] and planned["fits_ceiling"]
    assert report["wait_margin_usd"] == pytest.approx(11_746.5 / 3600 * 0.6)
    slow = report["scenarios"]["planning speeds, Mac twice as slow"]
    assert slow["total_usd"] == pytest.approx(4.42, abs=0.01) and slow["fits_ceiling"]
    assert report["limits"]["watchdog_cost_stop_usd"] == 5.79


def test_the_template_needs_a_ceiling_to_price_against() -> None:
    with pytest.raises(ValueError, match="--ceiling-usd"):
        posttrain2_report(TEMPLATE)


def run1_records(tmp_path: Path) -> tuple[Path, Path, Path]:
    report = {
        "kind": "posttrain-run-report",
        "run": "1",
        "specification": {"generation": {"batch_size": 128}},
        "probe": {
            "measured_tokens_per_second": {
                "1": 90.0,
                "64": 5_000.0,
                "128": 9_000.0,
                "256": 12_000.0,
            }
        },
        "rates": {"generation_tokens_per_second": 7_200.0, "arc_seconds_per_checkpoint": 40.0},
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
    rates = run1_measured_rates(report, screening, dev_results=[dev], **BATCHES)
    # The evaluation files ran at 0.8 of the probe's rate at their batch, 128: batch 256 is
    # slowed alike, and batch 64, at which nothing in run 2 generates, is left out.
    assert rates["generation_tokens_per_second"] == {"128": 7_200.0, "256": pytest.approx(9_600.0)}
    assert rates["training_slots_per_second"] == 60_000.0
    assert rates["reward_wait_seconds_per_update"] == pytest.approx(256 * 1.0 / 8 + 5.5)
    assert rates["dev_wait_seconds_per_check"] == pytest.approx(1_800 * 1.2 / 8 + 5.5)
    assert BRIDGE_ROUND_TRIP_SECONDS == 2 + 3 * 1 + 0.5
    projected = posttrain2_report(TEMPLATE, ceiling_usd=5.80, measured=rates)
    measured = projected["scenarios"]["run 1's measured rates"]
    assert measured["rows"]["rl_reward_waits"]["seconds"] == pytest.approx(100 * (32 + 5.5))
    assert measured["rows"]["rl_rollouts"]["seconds"] == pytest.approx(100 * 256 * 256 / 9_600)
    assert measured["rows"]["evaluation"]["seconds"] == pytest.approx(
        14 * 20 + 2 * 18_792_960 / 7_200
    )
    assert measured["total_usd"] < projected["scenarios"]["planning speeds"]["total_usd"] + 1
    text = render_posttrain2_markdown(projected)
    assert "## run 1's measured rates" in text and "| **Total** |" in text
    # The gate's estimate is this scenario's total, and the stop it gives is derived as the
    # worker derives it: the lower of the ceiling and 120% of the estimate.
    assert projected["gate_estimate_usd"] == measured["total_usd"]
    assert f"**The gate's estimate: ${measured['total_usd']:.4f}**" in text
    stopped = measured["limits_at_this_estimate"]
    assert stopped["limit_usd"] == estimate_limit_usd(5.80, measured["total_usd"])
    assert stopped["ceiling_usd"] == 5.80 and stopped["gate_estimate_usd"] == measured["total_usd"]
    assert measured["fits_lifetime_at_this_estimate"]
    assert posttrain2_report(TEMPLATE, ceiling_usd=5.80)["gate_estimate_usd"] is None
    # The launch rule prices the planning rates whatever run 1 measured: a fast Mac lowers the
    # estimate, and so the stop, below what the planning-rate need and the setup fit in.
    fast = dict(rates, reward_wait_seconds_per_update=24.0, dev_wait_seconds_per_check=150.0)
    quick = posttrain2_report(TEMPLATE, ceiling_usd=5.80, measured=fast)
    launch = quick["scenarios"]["run 1's measured rates"]["launch_at_this_estimate"]
    assert launch["planning_seconds_needed"] == pytest.approx(17_495, abs=1)
    assert launch["setup_allowance_seconds"] < launch["startup_seconds"] == 900
    assert launch["fits"] is False
    assert "**the gate refuses it**" in render_posttrain2_markdown(quick)


def test_run_ones_rates_must_be_at_the_batches_run_two_generates_at(tmp_path: Path) -> None:
    """Every endpoint's evaluation shares one generation batch, so a run 1 report generated at
    another batch is refused, as is one whose probe lacks a batch run 2 generates at."""
    report, screening, _ = run1_records(tmp_path)
    raw = json.loads(report.read_text())
    for change, message in (
        ({"specification": {"generation": {"batch_size": 64}}}, "at batch 64, run 2's"),
        ({"specification": {}}, "at batch None, run 2's"),
        ({"probe": {"measured_tokens_per_second": {"64": 5_000.0, "256": 1e4}}}, "batch 128$"),
        ({"probe": {"measured_tokens_per_second": {"128": 9_000.0}}}, "batch 256$"),
    ):
        report.write_text(json.dumps({**raw, **change}))
        with pytest.raises(ValueError, match=message):
            run1_measured_rates(report, screening, **BATCHES)


def test_the_projection_names_the_template_it_priced() -> None:
    import hashlib

    report = posttrain2_report(TEMPLATE, ceiling_usd=5.80)
    assert report["specification_sha256"] == hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    planned = report["scenarios"]["planning speeds"]
    # At the planning speeds the estimate would be $3.0658 and the stop $3.67, a 22,020 s wall.
    assert planned["limits_at_this_estimate"]["limit_usd"] == 3.67
    assert planned["limits_at_this_estimate"]["wall_seconds"] == 22_020
    launch = planned["launch_at_this_estimate"]
    assert launch["lifetime_seconds"] == 21_840 and launch["fits"] is True
    assert launch["setup_allowance_seconds"] == pytest.approx(21_840 - 17_495, abs=1)
    assert report["limits"]["ceiling_usd"] == 5.80 and report["limits"]["wall_seconds"] == 34_800


def test_the_cli_prints_the_proposals_table_for_run_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "feasibility-2.json"
    argv = ["posttrain", "--specification", str(TEMPLATE), "--ceiling-usd", "5.80"]
    assert main([*argv, "--quoted-gpu-hourly", "0.59", "--output", str(output)]) == 0
    text = capsys.readouterr().out
    assert "| **Total** | **5.11** | **$3.07** |" in text
    assert "| RL dev checks: generation / Mac wait | 0.17 / 0.50 |" in text
    assert json.loads(output.read_text())["run"] == "2"
    report, screening, _ = run1_records(tmp_path)
    assert main([*argv, "--run1-report", str(report), "--screening", str(screening)]) == 0
    assert "## run 1's measured rates" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([*argv, "--run1-report", str(report)])  # Both records, or neither.
