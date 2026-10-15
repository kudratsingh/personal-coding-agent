import pytest

from scratchcoder.common.cost import estimate


def test_compute_units_and_parallelism() -> None:
    result = estimate(360000, 100, 2.0, gpus=2)
    assert result.gpu_hours == 1
    assert result.wall_hours == 0.5
    assert result.dollars == 2


def test_reference_scaling_and_slowdown() -> None:
    result = estimate(360000, 100, 2, layer_parameters=50, reference_parameters=100, slowdown=2)
    assert result.gpu_hours == 1


@pytest.mark.parametrize("throughput", [0, -1, float("nan"), float("inf")])
def test_bad_rates_are_rejected(throughput: float) -> None:
    with pytest.raises(ValueError):
        estimate(10, throughput, 1)
