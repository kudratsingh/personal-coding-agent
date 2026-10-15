"""From-scratch metrics with task-level, seeded uncertainty estimates."""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


def pass_at_k(n: int, correct: int, k: int) -> float:
    if not 1 <= k <= n or not 0 <= correct <= n:
        raise ValueError("require 1 <= k <= n and 0 <= correct <= n")
    if n - correct < k:
        return 1.0
    # -expm1(sum(log probabilities)) avoids cancellation for very small pass rates.
    return -math.expm1(sum(math.log1p(-correct / (n - i)) for i in range(k)))


def mean_pass_rate(attempts: Sequence[Sequence[bool]]) -> float:
    if not attempts or any(not row for row in attempts):
        raise ValueError("every task needs at least one attempt")
    return sum(sum(row) / len(row) for row in attempts) / len(attempts)


def accuracy(predictions: Sequence[object], references: Sequence[object]) -> float:
    if not predictions or len(predictions) != len(references):
        raise ValueError("nonempty aligned predictions/references required")
    return sum(left == right for left, right in zip(predictions, references, strict=True)) / len(
        predictions
    )


def bits_per_byte(total_negative_log_likelihood: float, byte_count: int) -> float:
    if byte_count <= 0 or not 0 <= total_negative_log_likelihood < float("inf"):
        raise ValueError("finite nonnegative NLL and positive byte count required")
    return total_negative_log_likelihood / math.log(2) / byte_count


@dataclass(frozen=True)
class Interval:
    mean: float
    lower: float
    upper: float
    samples: int
    confidence: float


def _quantile(ordered: list[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_bootstrap(
    before: Sequence[float],
    after: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    if not before or len(before) != len(after) or samples < 2 or not 0 < confidence < 1:
        raise ValueError("invalid aligned scores or bootstrap settings")
    if any(not math.isfinite(value) for value in (*before, *after)):
        raise ValueError("scores must be finite")
    differences = [right - left for left, right in zip(before, after, strict=True)]
    rng = random.Random(seed)
    estimates = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(samples)
    )
    tail = (1 - confidence) / 2
    return Interval(
        sum(differences) / len(differences),
        _quantile(estimates, tail),
        _quantile(estimates, 1 - tail),
        samples,
        confidence,
    )
