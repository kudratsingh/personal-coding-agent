"""Finite scaling, filter semantics and the counter-based draw of handwritten sampling."""

import math
from typing import Any

import pytest
import torch
from torch import Tensor

from scratchcoder.engine.generation import (
    SAMPLER_VERSION,
    inverse_cdf,
    philox4x32,
    sample,
    sampling_log_probs,
    sampling_uniforms,
)

MASK = 0xFFFFFFFF


@pytest.mark.parametrize(
    "values,dtype,temperature,relative",
    [
        ([1.0, 2.0], torch.float32, 1e-40, [0.0, 1.0]),
        ([-1e308, 1e308], torch.float64, 1e308, [math.exp(-2), 1.0]),
        ([-1e-300, 0.0, 1e-300], torch.float64, 1e-300, [math.exp(-2), math.exp(-1), 1.0]),
        ([-5e-324, 0.0, 5e-324], torch.float64, 5e-324, [math.exp(-2), math.exp(-1), 1.0]),
        ([1e300, 2e300], torch.float64, 1e300, [math.exp(-1), 1.0]),
        ([2.0, 2.0, 1.0], torch.float32, 1e-40, [1.0, 1.0, 0.0]),
    ],
)
def test_extreme_finite_scaling_matches_hand_computed_probabilities(
    values: list[float], dtype: torch.dtype, temperature: float, relative: list[float]
) -> None:
    logits = torch.tensor([values], dtype=dtype)
    before = logits.clone()
    probabilities = sampling_log_probs(logits, temperature=temperature).exp()
    assert torch.equal(logits, before)
    assert bool(torch.isfinite(probabilities).all())
    expected = torch.tensor([[weight / sum(relative) for weight in relative]], dtype=torch.float64)
    torch.testing.assert_close(probabilities.double(), expected, atol=1e-12, rtol=1e-12)
    token = sample(logits, temperature=temperature, seed=3)
    assert relative[int(token)] > 0


def legacy_distribution(
    logits: Tensor, temperature: float, top_k: int | None, top_p: float
) -> Tensor:
    """Pre-counter-sampler filtering, returned as probabilities in vocabulary order."""
    filtered = logits.float() / temperature
    if top_k is not None:
        values, indices = filtered.topk(min(top_k, filtered.shape[-1]), dim=-1)
        filtered = torch.full_like(filtered, -math.inf).scatter(-1, indices, values)
    values, indices = filtered.sort(dim=-1, descending=True)
    probabilities = values.softmax(dim=-1)
    removed = probabilities.cumsum(dim=-1) - probabilities >= top_p
    sorted_probabilities = values.masked_fill(removed, -math.inf).softmax(dim=-1)
    return torch.zeros_like(sorted_probabilities).scatter(-1, indices, sorted_probabilities)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("temperature", [0.1, 1.0, 10.0])
def test_filtering_semantics_match_the_previous_sampler(
    dtype: torch.dtype, temperature: float
) -> None:
    logits = torch.randn((4, 17), generator=torch.Generator().manual_seed(19)).to(dtype)
    for top_k in (None, 1, 5, 100):
        for top_p in (1.0, 0.95, 0.2):
            expected = legacy_distribution(logits, temperature, top_k, top_p)
            actual = sampling_log_probs(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            ).exp()
            torch.testing.assert_close(actual.double(), expected.double(), atol=1e-6, rtol=1e-5)
            assert torch.equal(actual > 0, expected > 0) or top_p == 1.0


def test_greedy_is_the_first_maximal_logit() -> None:
    logits = torch.tensor([[1.0, 3.0, 3.0], [5.0, -1.0, 0.0]])
    assert sample(logits, temperature=0).tolist() == [[1], [0]]
    point = sampling_log_probs(logits, temperature=0)
    assert point.tolist() == [[-math.inf, 0.0, -math.inf], [0.0, -math.inf, -math.inf]]


def philox_reference(counter: tuple[int, ...], key: tuple[int, int]) -> tuple[int, ...]:
    """Scalar Philox4x32-10 written from the published round function."""
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for index in range(10):
        if index:
            k0, k1 = (k0 + 0x9E3779B9) & MASK, (k1 + 0xBB67AE85) & MASK
        first, second = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (
            (second >> 32) ^ c1 ^ k0,
            second & MASK,
            (first >> 32) ^ c3 ^ k1,
            first & MASK,
        )
    return c0, c1, c2, c3


@pytest.mark.parametrize(
    "counter,key,expected",
    [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        ((MASK,) * 4, (MASK, MASK), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
)
def test_philox_matches_published_known_answers(
    counter: tuple[int, ...], key: tuple[int, int], expected: tuple[int, ...]
) -> None:
    assert philox_reference(counter, key) == expected
    words = philox4x32(
        (
            torch.tensor([counter[0]]),
            torch.tensor([counter[1]]),
            torch.tensor([counter[2]]),
            torch.tensor([counter[3]]),
        ),
        (torch.tensor([key[0]]), torch.tensor([key[1]])),
    )
    assert tuple(int(word) for word in words) == expected


def test_uniforms_depend_only_on_seed_and_step() -> None:
    seeds = torch.tensor([0, 7, 2**63 - 1, 123456789012345])
    table = sampling_uniforms(seeds, 6)
    assert table.dtype == torch.float64 and table.shape == (4, 6)
    assert bool(((table >= 0) & (table < 1)).all())
    assert torch.equal(sampling_uniforms(seeds[1:2], 6)[0], table[1])
    assert torch.equal(sampling_uniforms(seeds, 2, first=3), table[:, 3:5])
    for row, seed in enumerate(seeds.tolist()):
        for step in range(6):
            words = philox_reference((step, 0, 0, 0), (seed & MASK, seed >> 32))
            bits = (words[0] >> 5) * 2**26 + (words[1] >> 6)
            assert table[row, step].item() == bits / 2**53
    many = sampling_uniforms(torch.arange(2000), 50)
    assert float(many.mean()) == pytest.approx(0.5, abs=0.003)
    assert float(many.var()) == pytest.approx(1 / 12, abs=0.002)
    with pytest.raises(ValueError):
        sampling_uniforms(seeds, 2, first=2**32 - 1)


def test_inverse_cdf_selects_by_hand_computed_cumulative_mass() -> None:
    weights = torch.tensor([0.2, 0.3, 0.5]).log()
    cases = [(0.0, 0), (0.1, 0), (0.2, 1), (0.49, 1), (0.5, 2), (0.999, 2), (1 - 2**-53, 2)]
    for uniform, expected in cases:
        token = inverse_cdf(weights, torch.tensor(uniform, dtype=torch.float64))
        assert token.tolist() == [expected]
    removed = torch.tensor([[-math.inf, 0.0, -math.inf, 0.0, -math.inf]])
    for uniform in (0.0, 0.25, 0.5, 0.75, 1 - 2**-53):
        token = inverse_cdf(removed, torch.tensor([uniform], dtype=torch.float64))
        assert token.tolist() == [[1 if uniform < 0.5 else 3]]


def test_scan_rounding_cannot_select_a_removed_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # A parallel scan may round a removed token's running sum above its predecessor's.
    original = Tensor.cumsum

    def rounded(self: Tensor, *args: Any, **options: Any) -> Tensor:
        return original(self, *args, **options) + torch.tensor([0.0, 1e-9, 0.0])

    monkeypatch.setattr(Tensor, "cumsum", rounded)
    weights = torch.tensor([0.0, -math.inf, 0.0])
    uniform = torch.tensor(0.5 + 0.5e-9, dtype=torch.float64)
    assert inverse_cdf(weights, uniform, filtered=False).tolist() == [1]
    assert inverse_cdf(weights, uniform).tolist() == [2]


def test_seeded_draws_follow_the_filtered_distribution() -> None:
    logits = torch.tensor([0.5, 2.0, -1.0, 1.0, 0.0, -3.0])
    for settings in ({}, {"top_k": 3}, {"top_p": 0.8}, {"temperature": 0.5, "top_k": 4}):
        options: dict[str, Any] = {"temperature": 1.0, **settings}
        probabilities = sampling_log_probs(logits, **options).exp()
        tokens = sample(logits.expand(20000, 6), seed=11, step=2, **options)[:, 0]
        frequencies = torch.bincount(tokens, minlength=6).double() / 20000
        torch.testing.assert_close(frequencies, probabilities.double(), atol=0.012, rtol=0)
        assert bool((probabilities[tokens] > 0).all())


def test_sampler_version_is_recorded_text() -> None:
    assert SAMPLER_VERSION == "philox4x32-10-inverse-cdf-v1"


@pytest.mark.parametrize(
    "settings",
    [
        {"temperature": True},
        {"temperature": "1"},
        {"temperature": None},
        {"temperature": -1},
        {"temperature": float("inf")},
        {"temperature": float("nan")},
        {"top_p": True},
        {"top_p": "1"},
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_k": True},
        {"top_k": 0},
        {"top_k": 1.5},
        {"top_k": "1"},
        {"seed": -1},
        {"seed": 2**63},
        {"seed": True},
        {"seed": [1, 2, 3]},
        {"step": -1},
        {"step": 2**32},
    ],
)
def test_invalid_controls_are_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        sample(torch.tensor([[1.0, 2.0]]), **settings)


@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor(1.0),
        torch.empty(0),
        torch.empty(0, 2),
        torch.empty(1, 0),
        torch.ones(1, 1, 2),
        torch.ones(2, dtype=torch.long),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.complex64),
        torch.ones(2).to_sparse(),
        torch.tensor([math.nan, 1.0]),
        torch.tensor([math.inf, 1.0]),
    ],
)
def test_invalid_logits_are_rejected(logits: Tensor) -> None:
    with pytest.raises(ValueError):
        sample(logits)
    with pytest.raises(ValueError):
        sampling_log_probs(logits, temperature=1.0)


@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_extreme_temperature_respects_top_filters_and_is_seeded(shape: tuple[int, ...]) -> None:
    logits = torch.tensor([1.0, 2.0, 2.0]).expand(shape)
    for settings in ({"top_k": 1}, {"top_p": 0.1}, {"top_k": 2, "top_p": 1.0}):
        first = sample(logits, temperature=1e-40, seed=7, **settings)
        second = sample(logits, temperature=1e-40, seed=7, **settings)
        assert torch.equal(first, second)
        assert first.shape == (*shape[:-1], 1)
        assert bool((first >= 1).all())
