"""Run 1b: its approved specification, its parser rules and its worker's format-gate stage.

Run 1b re-runs SFT from run B's ``mixed-final`` after run 1's SFT weights failed the format
gate: the special rows redrawn and trained at 10x, two passes over the rebuilt data, the
gate on the Pod before anything reads the weights, then ARC, evaluation and screening, at a
$1.50 ceiling. Run 2 then refuses SFT weights without a passing gate record of their own.
"""

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from tools import posttrain_approved, posttrain_ops
from tools.pilot_feasibility import posttrain_report
from tools.posttrain_readiness import APPROVED_CHECKS, MAC_COPIES
from tools.posttrain_run import (
    FORMAT_GATE_STOP,
    SFT_GATE_ARTIFACT,
    Rates,
    Specification,
    check_sft_gate,
    launch_fit,
    stage_seconds,
)
from tools.posttrain_smoke import SMOKE_POD_ID_1B, failing_gate, smoke_1b
from tools.rehearsal_run import quarter_updates

ROOT = Path(__file__).resolve().parents[1]
SPEC_1B = ROOT / "configs" / "rehearsal-a40-s-post-1b.json"


def approved() -> dict[str, Any]:
    return posttrain_approved.run1b_specification()


def test_the_tracked_specification_is_the_approved_run_1b() -> None:
    raw = json.loads(SPEC_1B.read_text())
    assert raw == approved()
    posttrain_approved.check_run1b(raw)
    # Run 1's specification is untouched by run 1b.
    run1 = json.loads(posttrain_approved.SPECIFICATION_PATH.read_text())
    assert run1 == posttrain_approved.run1_specification()
    posttrain_approved.check_run1(run1)


def test_run_1b_is_the_owners_approval() -> None:
    spec = Specification.load(SPEC_1B)
    assert spec.run == "1b"
    limits = spec.ceiling
    assert (limits.ceiling_usd, limits.worker_cost_stop_usd) == (1.5, 1.5)
    assert (limits.watchdog_cost_stop_usd, limits.wall_seconds) == (1.49, 9_000)
    assert limits.self_destruct_seconds == 8_820
    assert posttrain_ops.run_prefix("1b", "pod1") == "posttrain-runs/1b/pod1/"
    assert [stage.name for stage in spec.stages] == [
        "probe",
        "sft",
        "format-gate",
        "arc-sft",
        "eval-sft",
        "screen",
    ]
    sft = spec.sft
    assert sft["passes"] == 2 and spec.sft_updates == 212 and spec.rows_per_update == 32
    assert (sft["learning_rate"], sft["warmup_fraction"], sft["min_lr_ratio"]) == (6e-5, 0.03, 0.1)
    assert max(1, round(0.03 * 212)) == 6
    assert [update for _, update in quarter_updates(212)] == [53, 106, 159]
    assert sft["special_rows_init"] == "ordinary-mean-noise"
    assert sft["special_rows_lr_scale"] == 10.0 and sft["special_rows_seed"] == 0
    gate = spec.stage("format-gate").settings
    assert (gate["row_count"], gate["smoke_prompts"], gate["benchmark_prompts"]) == (32, 16, 8)
    assert gate["benchmarks"] == ["humaneval", "mbpp-plus"] and gate["checkpoint"] == "sft"
    # Run 1's probe, floors, planning rates, generation batch and protocols.
    run1 = posttrain_approved.run1_specification()
    for section in ("model", "probe", "generation", "screening", "r2"):
        assert spec.raw[section] == run1[section], section
    assert {key: value for key, value in spec.planning.items() if key != "format_gate_seconds"} == (
        run1["planning"]
    )
    eval_sft = next(stage for stage in run1["stages"] if stage["name"] == "eval-sft")
    assert spec.raw["stages"][4] == eval_sft


def test_run_1b_reads_the_rebuilt_sft_data_and_run_1s_other_inputs() -> None:
    inputs = approved()["inputs"]
    run1 = posttrain_approved.INPUTS
    assert set(inputs) == set(run1) - {"a-stage1-final", "a-stage2-final"}
    for name in ("sft-train", "sft-code-dev"):
        assert inputs[name]["sha256"] != run1[name]["sha256"]
        assert inputs[name] == {**run1[name], **posttrain_approved.SFT_DATA_1B[name]}
    for name in set(inputs) - {"sft-train", "sft-code-dev"}:
        assert inputs[name] == run1[name], name
    assert approved()["sft"]["train_rows"] == 3_390
    assert approved()["sft"]["planned_dev_tokens"] == 813_765


def test_the_readiness_receipt_knows_run_1b() -> None:
    assert APPROVED_CHECKS["1b"] is posttrain_approved.check_run1b
    assert MAC_COPIES["1b"] == ("b-mixed-final",)


@pytest.mark.parametrize(
    "change",
    [
        lambda raw: raw["sft"].update(passes=1),
        lambda raw: raw["sft"].update(special_rows_lr_scale=1.0),
        lambda raw: raw["limits"].update(ceiling_usd=3.0),
        lambda raw: raw["stages"][2].update(row_count=4),
        lambda raw: raw["planning"].update(format_gate_seconds=1),
        lambda raw: raw["inputs"]["sft-train"].update(sha256="0" * 64),
        lambda raw: raw.update(run="1"),
    ],
)
def test_check_run_1b_refuses_any_change(change: Any) -> None:
    raw = approved()
    change(raw)
    with pytest.raises(ValueError):
        posttrain_approved.check_run1b(raw)


def test_the_pool_test_pin_is_filled_only_at_the_go() -> None:
    raw = approved()
    assert raw["inputs"]["pool-test"]["sha256"] is None
    raw["inputs"]["pool-test"].update(size_bytes=10, sha256="ab" * 32)
    posttrain_approved.check_run1b(raw)


def parse(change: Any) -> None:
    raw = approved()
    change(raw)
    Specification.from_dict(raw)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda raw: (raw["stages"].pop(2), raw["planning"].pop("format_gate_seconds")),
            "through the format gate",
        ),
        (
            lambda raw: raw["stages"].insert(2, raw["stages"].pop(3)),
            "before the format gate",
        ),
        (lambda raw: raw["stages"].insert(3, copy.deepcopy(raw["stages"][2])), "unique"),
        (lambda raw: raw["stages"][2].update(checkpoint="b-mixed-final"), "selected weights"),
        (lambda raw: raw["stages"][2].update(benchmarks=["humaneval-plus"]), "benchmark smoke"),
        (lambda raw: raw["stages"][2].update(rows="sft-train", tasks="pool-test-x"), "input"),
        (lambda raw: raw["planning"].pop("format_gate_seconds"), "planning"),
        (lambda raw: raw["sft"].pop("special_rows_seed"), "sft"),
        (lambda raw: raw["sft"].update(special_rows_init="zeros"), "special_rows_init"),
    ],
)
def test_the_parser_puts_the_gate_before_every_use_of_the_weights(change: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse(change)


def test_a_second_gate_stage_is_refused() -> None:
    def twice(raw: dict[str, Any]) -> None:
        second = copy.deepcopy(raw["stages"][2])
        second["name"] = "format-gate-again"
        raw["stages"].insert(3, second)

    with pytest.raises(ValueError, match="one format-gate stage"):
        parse(twice)


def test_run_1s_specification_still_parses_without_gate_or_special_rows() -> None:
    spec = Specification.from_dict(posttrain_approved.run1_specification())
    assert "special_rows_init" not in spec.sft and spec.sft_updates == 106
    assert "format_gate_seconds" not in spec.planning
    assert Rates.planning(spec).format_gate_seconds is None


def test_the_projection_and_launch_rule_price_the_gate() -> None:
    spec = Specification.load(SPEC_1B)
    rates = Rates.planning(spec)
    assert stage_seconds(spec, spec.stage("format-gate"), rates) == 20 + 120
    planned = posttrain_report(SPEC_1B, quoted_gpu_hourly=0.59)["scenarios"]["1x generation"]
    assert round(planned["total_hours"], 2) == 1.54 and round(planned["total_usd"], 2) == 0.92
    assert planned["fits_lifetime"] and planned["fits_ceiling"]
    fit = launch_fit(spec, remaining_seconds=spec.ceiling.self_destruct_seconds)
    assert fit["fits_at_planning"]
    assert math.isclose(fit["planning_seconds_needed"], planned["total_seconds"] - 1_080)


def test_the_smoke_passes_the_real_gate_and_runs_every_stage(tmp_path: Path) -> None:
    summary = smoke_1b(tmp_path / "smoke")
    assert summary["status"] == "completed" and summary["objects_outside_prefix"] == []
    assert set(summary["stages"].values()) == {"completed"}
    gate = summary["format_gate"]
    assert gate["passed"] is True and gate["failed_checks"] == []
    assert gate["checkpoint_sha256"] == summary["sft_selected"]["sha256"]
    assert set(gate["benchmark_smoke"]) == {"humaneval", "mbpp-plus"}


def test_a_failing_gate_stops_the_run_before_anything_reads_the_weights(tmp_path: Path) -> None:
    summary = smoke_1b(tmp_path / "smoke", changes=failing_gate)
    assert summary["status"] == "stopped" and summary["stop_reason"] == FORMAT_GATE_STOP
    assert summary["stages"] == {
        "probe": "completed",
        "sft": "completed",
        "format-gate": "stopped",
        "arc-sft": "skipped",
        "eval-sft": "skipped",
        "screen": "skipped",
    }
    assert summary["objects_outside_prefix"] == [] and summary["format_gate"]["passed"] is False
    report = json.loads((tmp_path / "smoke" / "run" / "report.json").read_text())
    assert report["r2"]["prefix"] == f"posttrain-runs/1b/{SMOKE_POD_ID_1B}/"


# Run 2 refuses SFT weights without a passing gate record of their own -------------------------

WEIGHTS = "aa" * 32


def report_with(root: Path, *, run: str, gate: dict[str, Any] | None) -> tuple[Path, Path]:
    """A verified-looking SFT run report listing its weights and, optionally, a gate record."""
    prefix = posttrain_ops.run_prefix(run, "pod1")
    gate_path = root / "gate.json"
    rows = [
        {
            "label": "sft-weights",
            "name": "checkpoint.pt",
            "key": f"{prefix}sft-weights/checkpoint.pt",
            "sha256": WEIGHTS,
            "bytes": 10,
        }
    ]
    if gate is not None:
        gate_path.write_text(json.dumps(gate))
        from scratchcoder.data.shards import file_sha256

        rows.append(
            {
                "label": "format-gate",
                "name": "gate.json",
                "key": f"{prefix}{SFT_GATE_ARTIFACT}",
                "sha256": file_sha256(gate_path),
                "bytes": gate_path.stat().st_size,
            }
        )
    report = {
        "kind": posttrain_ops.REPORT_KIND,
        "run": run,
        "pod_id": "pod1",
        "status": "completed",
        "artifacts": rows,
    }
    path = root / "report.json"
    path.write_text(json.dumps(report))
    return path, gate_path


def gate_record(*, passed: bool = True, checkpoint: str = WEIGHTS) -> dict[str, Any]:
    from scratchcoder.eval.format_gate import FORMAT

    return {"format": FORMAT, "passed": passed, "inputs": {"checkpoint_sha256": checkpoint}}


def test_run_2s_gate_takes_a_passing_record_of_exactly_the_weights(tmp_path: Path) -> None:
    report, gate = report_with(tmp_path, run="1b", gate=gate_record())
    pin = posttrain_approved.sft_gate_pin(report, gate)
    assert pin["key"] == "posttrain-runs/1b/pod1/format-gate/gate.json"
    assert pin["checkpoint_sha256"] == WEIGHTS and pin["passed"] is True


@pytest.mark.parametrize(
    ("run", "record", "match"),
    [
        ("1", None, "no format-gate record"),
        ("1b", gate_record(passed=False), "failed the format gate"),
        ("1b", gate_record(checkpoint="bb" * 32), "selected weights"),
        ("2", gate_record(), "SFT run"),
    ],
)
def test_run_2s_gate_refuses_weights_without_a_passing_record(
    tmp_path: Path, run: str, record: dict[str, Any] | None, match: str
) -> None:
    report, gate = report_with(tmp_path, run=run, gate=record)
    if record is None:
        gate.write_text("{}")
    with pytest.raises(ValueError, match=match):
        posttrain_approved.sft_gate_pin(report, gate)


def test_run_2s_gate_refuses_a_copy_with_other_bytes(tmp_path: Path) -> None:
    report, gate = report_with(tmp_path, run="1b", gate=gate_record())
    gate.write_text(json.dumps(gate_record()) + " ")
    with pytest.raises(ValueError, match="not the format-gate record"):
        posttrain_approved.sft_gate_pin(report, gate)


def filled_gate_spec(**pin: Any) -> dict[str, Any]:
    raw = posttrain_approved.run2_specification()
    key = "posttrain-runs/1b/pod1/sft-weights/checkpoint.pt"
    raw["inputs"]["sft-weights"].update(key=key, size_bytes=10, sha256=WEIGHTS)
    raw["sft_gate"] = {
        "key": "posttrain-runs/1b/pod1/format-gate/gate.json",
        "size_bytes": 5,
        "sha256": "cc" * 32,
        "checkpoint_sha256": WEIGHTS,
        "passed": True,
        **pin,
    }
    return raw


def test_the_run_2_template_leaves_the_gate_pin_null_and_the_filled_one_is_checked() -> None:
    template = posttrain_approved.run2_specification()
    assert template["sft_gate"] == dict.fromkeys(posttrain_approved.SFT_GATE_KEYS_TEMPLATE)
    assert check_sft_gate(template) == template["sft_gate"]
    assert check_sft_gate(filled_gate_spec())["passed"] is True
    tracked = json.loads(posttrain_approved.SPECIFICATION_PATH_2.read_text())
    assert tracked == template


@pytest.mark.parametrize(
    ("pin", "match"),
    [
        ({"passed": False}, "did not pass"),
        ({"checkpoint_sha256": "bb" * 32}, "other weights"),
        ({"key": "posttrain-runs/1/pod1/format-gate/gate.json"}, "beside the SFT weights"),
        ({"size_bytes": 0}, "positive size"),
    ],
)
def test_run_2_refuses_a_filled_gate_pin_that_is_not_a_pass(
    pin: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        check_sft_gate(filled_gate_spec(**pin))
