from dataclasses import replace

import pytest

from scratchcoder.train.pilot import (
    Limits,
    PilotStop,
    accumulation_steps,
    choose_microbatch,
    cost_per_billion,
    throughput,
)


def limits() -> Limits:
    return Limits(300, 3600, 5400, 0.6, 0.9, 5, 1.2, 0.5, 300, 600, 500)


def test_effective_batch_and_actual_quoted_cost_reference() -> None:
    assert [accumulation_steps(size, 1024, 131072) for size in (8, 16, 32)] == [16, 8, 4]
    assert cost_per_billion(10000, 0.59) == pytest.approx(16.3888888889)
    with pytest.raises(ValueError, match="exact"):
        accumulation_steps(3, 1024, 131072)
    with pytest.raises(ValueError):
        accumulation_steps(True, 1024, 131072)


def test_throughput_uses_total_tokens_over_total_seconds_after_warmup() -> None:
    rows = [
        {"step": 1.0, "tokens": 100.0, "step_seconds": 100.0},
        {"step": 2.0, "tokens": 200.0, "step_seconds": 2.0},
        {"step": 3.0, "tokens": 200.0, "step_seconds": 4.0},
    ]
    assert throughput(rows, warmup_steps=1) == {
        "tokens": 400.0,
        "step_seconds": 6.0,
        "tokens_per_second": 400 / 6,
    }
    with pytest.raises(ValueError, match="strictly"):
        throughput([rows[1], rows[1]])
    with pytest.raises(ValueError, match="post-warmup"):
        throughput(rows, warmup_steps=3)


def test_largest_fit_is_measured_and_incomplete_probes_never_select() -> None:
    rows = [
        dict(
            microbatch=size,
            status="completed",
            updates=20,
            tokens_per_second=10.0,
            peak_allocated_bytes=100.0,
        )
        for size in (16, 32, 8)
    ]
    assert choose_microbatch(rows, (16, 32, 8), 20) == 32
    rows[1] = dict(microbatch=32, status="oom")
    assert choose_microbatch(rows, (16, 32, 8), 20) == 16
    with pytest.raises(PilotStop, match="inventory"):
        choose_microbatch(rows[:2], (16, 32, 8), 20)
    rows[0]["updates"] = 19
    with pytest.raises(PilotStop, match="fixed step"):
        choose_microbatch(rows, (16, 32, 8), 20)


def test_wall_deadline_overrides_state_and_cost_caps_are_independent() -> None:
    policy = limits()
    assert policy.stop_reason(pod_elapsed=5399) is None
    assert policy.stop_reason(pod_elapsed=5400) == "pod_wall_limit"
    assert policy.stop_reason(pod_elapsed=100, idle_elapsed=600) == "idle_gpu_limit"
    expensive = replace(policy, quoted_total_hourly=36)
    assert expensive.cost_stop == 1.08
    assert expensive.stop_reason(pod_elapsed=108) == "cost_limit"
    assert replace(policy, approved_cost_cap=0.5).cost_stop == 0.5
    with_starting_cost = replace(expensive, starting_cost_usd=0.5)
    assert with_starting_cost.stop_reason(pod_elapsed=59) == "cost_limit"


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_limits_fail_before_a_run(bad: float) -> None:
    with pytest.raises(ValueError):
        replace(limits(), pod_seconds=bad)


def test_training_deadline_is_the_earlier_of_budget_and_reserve() -> None:
    policy = replace(limits(), finalization_reserve_seconds=900)
    # Training starts 600 s after pod creation: the 3600 s budget ends at 4200 s,
    # before the reserve deadline at 5400 - 900 = 4500 s.
    assert policy.training_deadline(pod_created=0, training_started=600) == 4200
    # Training starts late: the reserve deadline binds instead.
    assert policy.training_deadline(pod_created=0, training_started=1500) == 4500
    with pytest.raises(ValueError):
        policy.training_deadline(pod_created=10, training_started=5)
    with pytest.raises(ValueError, match="inconsistent"):
        replace(limits(), finalization_reserve_seconds=5200)  # No room left for probes.
    with pytest.raises(ValueError):
        replace(limits(), finalization_reserve_seconds=-1)


def test_skipped_and_incomplete_probes_never_select_but_do_not_abort() -> None:
    rows = [
        dict(
            microbatch=16,
            status="completed",
            updates=20,
            tokens_per_second=10.0,
            peak_allocated_bytes=100.0,
        ),
        dict(microbatch=32, status="oom", updates=0),
        dict(microbatch=8, status="incomplete", updates=7),
        dict(microbatch=4, status="skipped", updates=0),
    ]
    assert choose_microbatch(rows, (16, 32, 8, 4), 20) == 16
    rows[0]["status"] = "incomplete"
    with pytest.raises(PilotStop, match="no measured microbatch fits"):
        choose_microbatch(rows, (16, 32, 8, 4), 20)
    rows[0]["status"] = "crashed"
    with pytest.raises(PilotStop, match="fixed step"):
        choose_microbatch(rows, (16, 32, 8, 4), 20)


def test_headroom_budget_excludes_probes_that_barely_fit() -> None:
    rows = [
        dict(
            microbatch=size,
            status="completed",
            updates=20,
            tokens_per_second=10.0,
            peak_allocated_bytes=float(size) * 1e9,
            peak_reserved_bytes=float(size) * 1.1e9,
        )
        for size in (16, 8, 4)
    ]
    assert choose_microbatch(rows, (16, 8, 4), 20, max_peak_bytes=48e9) == 16
    assert choose_microbatch(rows, (16, 8, 4), 20, max_peak_bytes=10e9) == 8
    assert rows[0]["exceeds_headroom"] is True and "exceeds_headroom" not in rows[1]
    with pytest.raises(PilotStop, match="no measured microbatch fits"):
        choose_microbatch(rows, (16, 8, 4), 20, max_peak_bytes=1e9)
    with pytest.raises(ValueError):
        choose_microbatch(rows, (16, 8, 4), 20, max_peak_bytes=0)
