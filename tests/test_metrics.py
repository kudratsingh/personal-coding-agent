import math
from fractions import Fraction
from itertools import combinations

import pytest

from scratchcoder.eval.metrics import (
    accuracy,
    bits_per_byte,
    mean_pass_rate,
    paired_bootstrap,
    pass_at_k,
)


def test_pass_at_k_matches_exhaustive_combinatorics() -> None:
    for n in range(1, 9):
        for correct in range(n + 1):
            for k in range(1, n + 1):
                subsets = list(combinations(range(n), k))
                successes = sum(any(index < correct for index in subset) for subset in subsets)
                expected = float(Fraction(successes, len(subsets)))
                assert pass_at_k(n, correct, k) == pytest.approx(expected, abs=1e-14)


def test_task_weighting_accuracy_and_bits_by_hand() -> None:
    assert mean_pass_rate([[True], [False, True]]) == 0.75
    assert accuracy(["A", "B", "A"], ["A", "C", "A"]) == pytest.approx(2 / 3)
    assert bits_per_byte(8 * math.log(2), 4) == 2


def test_paired_bootstrap_reuses_task_indices_and_seed() -> None:
    before = [0.1, 0.2, 0.3, 0.4]
    after = [value + 0.25 for value in before]
    result = paired_bootstrap(before, after, samples=100, seed=7)
    assert result.mean == pytest.approx(0.25)
    assert result.lower == pytest.approx(0.25)
    assert result.upper == pytest.approx(0.25)
    assert result == paired_bootstrap(before, after, samples=100, seed=7)


def test_invalid_metric_inputs_fail() -> None:
    with pytest.raises(ValueError):
        pass_at_k(1, 1, 2)
    with pytest.raises(ValueError):
        mean_pass_rate([])
    with pytest.raises(ValueError):
        paired_bootstrap([1], [math.nan])
