"""The run-1 cost projection: hours and dollars per stage against the $3.00 ceiling."""

import json
from pathlib import Path

import pytest

from tools.pilot_feasibility import main, posttrain_report, render_posttrain_markdown

SPEC = Path(__file__).resolve().parents[1] / "configs" / "rehearsal-a40-s-post-1.json"


def test_run_one_fits_its_ceiling_at_the_planning_speeds() -> None:
    report = posttrain_report(SPEC)
    planned = report["scenarios"]["1x generation"]
    assert planned["generation_tokens_per_second"] == 8_279  # The batch-128 floor.
    stages = planned["stages"]
    assert list(stages) == [
        "setup",
        "probe",
        "arc-base",
        "eval-base",
        "sft",
        "arc-sft",
        "eval-sft",
        "screen",
        "finalization",
    ]
    assert stages["eval-base"]["hours"] == pytest.approx((140 + 18_612_480 / 8_279) / 3600)
    assert stages["screen"]["hours"] == pytest.approx((20 + 2_048_000 / 8_279) / 3600)
    assert planned["total_hours"] == pytest.approx(2.21, abs=0.01)
    assert planned["total_usd"] == pytest.approx(1.33, abs=0.01)
    assert planned["fits_lifetime"] and planned["fits_ceiling"]
    assert report["sft"] == {
        "train_rows": 3_390,
        "updates": 106,
        "token_slots": 106 * 32_768,
        "training_slots_per_second": 56_765.0,
    }
    assert report["limits"]["worker_cost_stop_usd"] == 3.0
    assert report["r2_upload"]["required_upload_bytes_per_second"] == pytest.approx(
        3.0556e6, rel=1e-3
    )
    slow = report["scenarios"]["0.5x generation"]
    assert slow["probe_stops_run"] and slow["spend_if_probe_stops_usd"] < 0.4
    # At half the floors the run would still fit the ceiling, but it never runs: the probe
    # stops it first.
    assert slow["fits_ceiling"] and slow["total_usd"] == pytest.approx(2.12, abs=0.01)


def test_the_cli_prints_the_table_and_writes_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "feasibility.json"
    assert main(["posttrain", "--quoted-gpu-hourly", "0.59", "--output", str(output)]) == 0
    text = capsys.readouterr().out
    assert "| **Total** | **2.21** | **$1.33** |" in text
    assert "the probe stops the run" in text
    assert json.loads(output.read_text())["kind"] == "posttrain-feasibility"
    assert render_posttrain_markdown(posttrain_report(SPEC, quoted_gpu_hourly=0.59)) == text


def test_the_documents_quote_the_projections_probe_stop_cost() -> None:
    """The runbook and the operations page give the probe-stop spend the tool prints."""
    root = SPEC.parents[1]
    spend = posttrain_report(SPEC)["scenarios"]["0.5x generation"]["spend_if_probe_stops_usd"]
    figure = f"at most about ${spend:.2f}"
    assert figure == "at most about $0.39"
    for path in (root / "POSTTRAIN_RUNBOOK.md", root / "docs" / "posttrain-runs.md"):
        text = path.read_text()
        assert text.count(figure) >= 2, path.name
        assert "$0.25" not in text and "$0.30 in" not in text, path.name
