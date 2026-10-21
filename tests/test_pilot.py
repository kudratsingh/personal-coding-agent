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
