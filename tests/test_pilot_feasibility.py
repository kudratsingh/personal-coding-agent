import json
from pathlib import Path

import pytest

from tools.pilot_feasibility import (
    DEFAULT_ACTIVATION_GIB_PER_SEQUENCE,
    DEFAULT_PEAK_TFLOPS,
    DEFAULT_USABLE_GPU_MEMORY_GIB,
    GIB,
    GPU_DISCLAIMER,
    MEASURED_UNTRAINED_CHECKPOINT_BYTES,
    FeasibilityOptions,
    PilotSpecification,
    accumulation_steps,
    build_report,
    checkpoint_bytes_estimate,
    disk_estimate,
    finalization_seconds,
    flops_per_token,
    implied_tflops,
    main,
    memory_verdict,
    microbatch_table,
    model_flops_utilization,
    model_parameters,
    network_volume_decision,
    outcome_label,
    recommended_container_disk_gb,
    render_markdown,
    required_throughput,
    schedule,
    static_bytes,
    total_updates,
    transfer_seconds,
)

SPEC_PATH = Path(__file__).resolve().parent.parent / "configs" / "pilot-a40-s.json"
PARAMETERS = 135_285_504


@pytest.fixture(scope="module")
def spec() -> PilotSpecification:
    return PilotSpecification.load(SPEC_PATH)


@pytest.fixture(scope="module")
def flops_token(spec: PilotSpecification) -> int:
    return flops_per_token(PARAMETERS, spec.model.layers, spec.model.context, spec.model.width)


def test_default_specification_parameters_and_static_bytes(spec: PilotSpecification) -> None:
    assert model_parameters(spec) == PARAMETERS
    assert static_bytes(PARAMETERS) == 16 * 135_285_504
    assert spec.arms == 3
    assert spec.steps == 500
    assert spec.screen_steps == 100
    assert spec.tokens_per_update == 131072


def test_checkpoint_estimate_tracks_measured_untrained_checkpoint() -> None:
    estimate = checkpoint_bytes_estimate(PARAMETERS)
    assert estimate > PARAMETERS * 12
    assert MEASURED_UNTRAINED_CHECKPOINT_BYTES == 1_623_502_465
    assert abs(estimate / MEASURED_UNTRAINED_CHECKPOINT_BYTES - 1) < 0.001


def test_total_updates_counts_every_screen_and_the_continuation(spec: PilotSpecification) -> None:
    assert total_updates(spec) == 3 * 100 + (500 - 100) == 700


def test_required_throughputs_for_screens_and_full_pilot(spec: PilotSpecification) -> None:
    screens = required_throughput(spec.tokens_per_update, 300, spec.training_seconds)
    full = required_throughput(spec.tokens_per_update, 700, spec.training_seconds)
    assert screens == pytest.approx(10922.67, abs=0.005)
    assert full == pytest.approx(25486.22, abs=0.005)


@pytest.mark.parametrize(("microbatch", "expected"), [(16, 8), (32, 4), (8, 16), (4, 32)])
def test_accumulation_steps_divide_the_update_exactly(microbatch: int, expected: int) -> None:
    assert accumulation_steps(131072, microbatch, 1024) == expected


@pytest.mark.parametrize("microbatch", [3, 5, 24, 1000])
def test_indivisible_microbatch_is_rejected(microbatch: int) -> None:
    with pytest.raises(ValueError, match="not a multiple"):
        accumulation_steps(131072, microbatch, 1024)


def test_memory_verdict_thresholds() -> None:
    usable = 100.0
    assert memory_verdict(50.0, usable) == "likely fits"
    assert memory_verdict(84.999, usable) == "likely fits"
    assert memory_verdict(85.0, usable) == "marginal"
    assert memory_verdict(100.0, usable) == "marginal"
    assert memory_verdict(100.001, usable) == "likely OOM"


def test_microbatch_table_with_default_proxy(spec: PilotSpecification) -> None:
    table = microbatch_table(
        spec, PARAMETERS, DEFAULT_ACTIVATION_GIB_PER_SEQUENCE, DEFAULT_USABLE_GPU_MEMORY_GIB
    )
    rows = {row.microbatch: row for row in table}
    assert list(rows) == [16, 32, 8, 4]
    assert rows[4].verdict == "likely fits" and rows[4].fits
    assert rows[8].verdict == "likely fits" and rows[8].fits
    assert rows[16].verdict in {"marginal", "likely OOM"}
    assert rows[32].verdict == "likely OOM" and not rows[32].fits
    assert rows[8].accumulation_steps == 16
    assert rows[8].static_bytes == 16 * PARAMETERS
    assert rows[8].activation_proxy_bytes == pytest.approx(8 * 2.85 * GIB)
    assert rows[8].proxy_total_bytes == pytest.approx(16 * PARAMETERS + 8 * 2.85 * GIB)
    # The analytical output-layer term is reported but never added to the fit decision.
    assert rows[8].output_layer_analytical_bytes == 8 * 1024 * 65536 * 14
    assert rows[8].proxy_total_bytes == rows[8].static_bytes + rows[8].activation_proxy_bytes


def test_flops_per_token_and_utilization(flops_token: int) -> None:
    assert flops_token == 6 * PARAMETERS + 12 * 12 * 1024 * 768
    assert implied_tflops(25486.22, flops_token) == pytest.approx(23.57, abs=0.01)
    assert model_flops_utilization(25486.22, flops_token, DEFAULT_PEAK_TFLOPS) == pytest.approx(
        0.1575, abs=0.001
    )


def test_schedule_at_required_throughput_completes_the_pilot(
    spec: PilotSpecification, flops_token: int
) -> None:
    rate = required_throughput(spec.tokens_per_update, 700, spec.training_seconds)
    plan = schedule(spec, rate, probe_microbatches=[8, 4], flops_token=flops_token)
    assert plan.updates_total == 700
    assert plan.updates_reached == 700
    assert plan.screens_completed == 3
    assert plan.outcome == "complete 500"
    assert plan.training_deadline_seconds == 3600
    assert plan.training_seconds_used == pytest.approx(3600, abs=1e-6)
    assert plan.probe_seconds == pytest.approx(20 * (8 + 4) * 1024 / rate)
    assert plan.probe_within_limit
    assert plan.fits_pod and plan.fits_cost_cap


def test_schedule_at_5000_tokens_per_second(spec: PilotSpecification, flops_token: int) -> None:
    plan = schedule(spec, 5000, probe_microbatches=[8, 4], flops_token=flops_token)
    assert plan.seconds_per_update == pytest.approx(26.2144)
    assert plan.updates_reached == 137
    assert plan.screens_completed == 1
    assert plan.outcome == "no complete screen"
    assert plan.training_seconds_used == pytest.approx(137 * 26.2144)
    assert plan.finalization_seconds == pytest.approx(finalization_seconds(1_623_502_465))
    assert plan.transfer_seconds == pytest.approx(649.4, abs=0.01)
    assert plan.total_pod_seconds == pytest.approx(
        600
        + plan.probe_seconds
        + plan.training_seconds_used
        + plan.finalization_seconds
        + plan.transfer_seconds
    )
    assert plan.total_pod_seconds < 5400 and plan.fits_pod
    assert plan.cost_dollars == pytest.approx(plan.total_pod_seconds / 3600 * 0.6)
    assert plan.fits_cost_cap


def test_schedule_outcomes_at_screen_boundaries(spec: PilotSpecification, flops_token: int) -> None:
    screens_only = required_throughput(spec.tokens_per_update, 300, spec.training_seconds)
    plan = schedule(spec, screens_only, probe_microbatches=[8, 4], flops_token=flops_token)
    assert plan.updates_reached == 300
    assert plan.outcome == "screens only (no winner continuation)"
    partial = schedule(spec, 25000, probe_microbatches=[8, 4], flops_token=flops_token)
    assert partial.updates_reached == 686
    assert partial.outcome == "partial continuation to step 486"
    assert outcome_label(299, 3, 100, 500, 700) == "no complete screen"
    assert outcome_label(301, 3, 100, 500, 700) == "partial continuation to step 101"
    assert outcome_label(700, 3, 100, 500, 700) == "complete 500"


def test_schedule_deadline_is_bounded_by_pod_budget(flops_token: int) -> None:
    raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    raw["limits"]["pod_seconds"] = 3000
    tight = PilotSpecification.from_mapping(raw)
    plan = schedule(tight, 40000, probe_microbatches=[8, 4], flops_token=flops_token)
    assert plan.training_deadline_seconds == pytest.approx(3000 - 600 - plan.probe_seconds - 900)
    assert plan.updates_reached == 455 < 700
    assert plan.outcome == "partial continuation to step 255"
    # The 900 s reserve exceeds actual finalization + transfer, so the pod still fits.
    assert plan.total_pod_seconds <= 3000 and plan.fits_pod


def test_slow_transfer_breaks_the_pod_budget(spec: PilotSpecification, flops_token: int) -> None:
    plan = schedule(
        spec, 40000, probe_microbatches=[8, 4], flops_token=flops_token, transfer_mbps=1
    )
    assert plan.transfer_seconds == pytest.approx(12988.02, abs=0.01)
    assert plan.total_pod_seconds > 5400
    assert not plan.fits_pod
    assert plan.fits_cost_cap


def test_schedule_rejects_bad_throughput(spec: PilotSpecification, flops_token: int) -> None:
    for rate in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            schedule(spec, rate, probe_microbatches=[4], flops_token=flops_token)


def test_finalization_and_transfer_arithmetic() -> None:
    assert finalization_seconds(1_000_000_000) == pytest.approx(12.0)
    assert transfer_seconds(1_000_000_000, 20) == pytest.approx(400.0)
    assert transfer_seconds(MEASURED_UNTRAINED_CHECKPOINT_BYTES, 20) == pytest.approx(
        649.40, abs=0.01
    )


def test_disk_estimate_recommends_twenty_gigabytes() -> None:
    disk = disk_estimate(3)
    assert disk.shard_bytes == 66_030_000 * 2
    assert disk.validation_bytes == 32 * 1025 * 2
    assert disk.dataset_bytes == 132_060_000 + 65_600 + 3_000_000
    assert disk.checkpoint_copies == 5
    assert disk.export_bytes == 0
    assert disk.checkpoint_total_bytes == 5 * 1_623_502_465
    assert disk.total_bytes == disk.dataset_bytes + disk.checkpoint_total_bytes
    assert disk.recommended_container_disk_gb == 20
    assert recommended_container_disk_gb(0) == 20
    assert recommended_container_disk_gb(17_000_000_000) == 30
    assert recommended_container_disk_gb(16_000_000_000) == 20


def test_network_volume_is_not_required() -> None:
    volume = network_volume_decision()
    assert volume.size_gb == 30
    assert volume.monthly_cost_dollars == pytest.approx(2.10)
    assert volume.billing == "billed hourly until deleted"
    assert volume.required is False
    assert "Not required" in volume.conclusion
    assert "135 MB" in volume.conclusion
    assert "Mac-initiated" in volume.conclusion


def test_options_validation() -> None:
    with pytest.raises(ValueError, match="usable_gpu_memory_gib"):
        FeasibilityOptions(gpu_memory_gib=48, usable_gpu_memory_gib=50)
    with pytest.raises(ValueError, match="throughputs"):
        FeasibilityOptions(throughputs=(1000.0, 0.0))
    with pytest.raises(ValueError, match="startup_seconds"):
        FeasibilityOptions(startup_seconds=-1)


def test_specification_rejects_malformed_input() -> None:
    raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    del raw["limits"]["approved_cost_cap"]
    with pytest.raises(ValueError, match="approved_cost_cap"):
        PilotSpecification.from_mapping(raw)
    raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    raw["microbatches"] = [16, 0]
    with pytest.raises(ValueError, match="microbatches"):
        PilotSpecification.from_mapping(raw)


def test_report_and_markdown_carry_the_disclaimer(spec: PilotSpecification) -> None:
    report = build_report(spec, FeasibilityOptions(), str(SPEC_PATH))
    assert report["disclaimer"] == GPU_DISCLAIMER
    assert report["probe_microbatches"] == [8, 4]
    assert [item["updates"] for item in report["requirements"]] == [300, 700]
    assert report["requirements"][1]["schedule"]["outcome"] == "complete 500"
    assert [item["throughput"] for item in report["scenarios"]] == [
        5000,
        10000,
        15000,
        25000,
        40000,
    ]
    markdown = render_markdown(report)
    for heading in (
        "## Static memory",
        "## Microbatch table",
        "## Throughput requirements",
        "## Scenario table",
        "## Disk",
        "## Storage decision",
    ):
        assert heading in markdown
    assert GPU_DISCLAIMER in markdown
    assert "10,922.67" in markdown and "25,486.22" in markdown
    assert "not a CUDA measurement" in markdown


def test_cli_writes_json_and_prints_disclaimer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "feasibility.json"
    assert main(["--specification", str(SPEC_PATH), "--output", str(output)]) == 0
    stdout = capsys.readouterr().out
    assert GPU_DISCLAIMER in stdout
    assert "| 4 | 32 |" in stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["disclaimer"] == GPU_DISCLAIMER
    assert report["model"]["parameters"] == PARAMETERS
    assert report["static_memory"]["static_bytes"] == 16 * PARAMETERS
    assert report["static_memory"]["gpu_memory_gib"] == 48.0
    assert report["static_memory"]["usable_gpu_memory_gib"] == 44.5
    assert report["disk"]["recommended_container_disk_gb"] == 20
    assert report["scenarios"][0]["outcome"] == "no complete screen"
    assert report["scenarios"][-1]["outcome"] == "complete 500"


def test_cli_rejects_unreadable_specification(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit):
        main(["--specification", str(missing)])
