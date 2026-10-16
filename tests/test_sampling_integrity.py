"""Finite scaling and unchanged ordinary seeded behavior of handwritten sampling."""

import math
from typing import Any

import pytest
import torch
from torch import Tensor

from scratchcoder.engine.generation import sample


@pytest.mark.parametrize(
    "values,dtype,temperature,relative",
    [
        ([1.0, 2.0], torch.float32, 1e-40, [1.0, 0.0]),
        ([-1e308, 1e308], torch.float64, 1e308, [1.0, math.exp(-2)]),
        ([-1e-300, 0.0, 1e-300], torch.float64, 1e-300, [1.0, math.exp(-1), math.exp(-2)]),
        ([-5e-324, 0.0, 5e-324], torch.float64, 5e-324, [1.0, math.exp(-1), math.exp(-2)]),
        ([1e300, 2e300], torch.float64, 1e300, [1.0, math.exp(-1)]),
        ([2.0, 2.0, 1.0], torch.float32, 1e-40, [1.0, 1.0, 0.0]),
    ],
)
def test_extreme_finite_scaling_matches_hand_computed_probabilities(
    values: list[float],
    dtype: torch.dtype,
    temperature: float,
    relative: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Tensor] = []

    def choose(probabilities: Tensor, count: int, *, generator: Any = None) -> Tensor:
        observed.append(probabilities.detach().clone())
        return torch.zeros((*probabilities.shape[:-1], count), dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", choose)
    logits = torch.tensor([values], dtype=dtype)
    before = logits.clone()
    result = sample(logits, temperature=temperature)
    assert result.shape == (1, 1)
    assert torch.equal(logits, before)
    assert len(observed) == 1 and bool(torch.isfinite(observed[0]).all())
    expected = torch.tensor([[weight / sum(relative) for weight in relative]], dtype=torch.float64)
    torch.testing.assert_close(observed[0].double(), expected, atol=1e-12, rtol=1e-12)


def legacy_sample(
    logits: Tensor, temperature: float, top_k: int | None, top_p: float, seed: int
) -> Tensor:
    """Pre-fix expression, retained only as the ordinary finite-path reference."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    filtered = logits.float() / temperature
    if top_k is not None:
        values, indices = filtered.topk(min(top_k, filtered.shape[-1]), dim=-1)
        filtered = torch.full_like(filtered, -math.inf).scatter(-1, indices, values)
    values, indices = filtered.sort(dim=-1, descending=True)
    probabilities = values.softmax(dim=-1)
    removed = probabilities.cumsum(dim=-1) - probabilities >= top_p
    values = values.masked_fill(removed, -math.inf)
    selected = torch.multinomial(
        values.softmax(dim=-1), 1, generator=torch.Generator().manual_seed(seed)
    )
    return indices.gather(-1, selected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("temperature", [0.0, 0.1, 1.0, 10.0])
def test_ordinary_seeded_path_matches_pre_fix_reference(
    dtype: torch.dtype, temperature: float
) -> None:
    logits = torch.randn((4, 17), generator=torch.Generator().manual_seed(19)).to(dtype)
    for top_k in (None, 1, 5, 100):
        for top_p in (1.0, 0.95, 0.2):
            for seed in (0, 7, 23):
                expected = legacy_sample(logits, temperature, top_k, top_p, seed)
                actual = sample(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=torch.Generator().manual_seed(seed),
                )
                assert torch.equal(actual, expected)


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
    ],
)
def test_invalid_controls_reject_before_consuming_rng(settings: dict[str, Any]) -> None:
    generator = torch.Generator().manual_seed(7)
    before = generator.get_state().clone()
    with pytest.raises(ValueError):
        sample(torch.tensor([[1.0, 2.0]]), generator=generator, **settings)
    assert torch.equal(generator.get_state(), before)


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
def test_invalid_logits_reject_before_consuming_rng(logits: Tensor) -> None:
    generator = torch.Generator().manual_seed(7)
    before = generator.get_state().clone()
    with pytest.raises(ValueError):
        sample(logits, generator=generator)
    assert torch.equal(generator.get_state(), before)


@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_extreme_temperature_respects_top_filters_and_is_seeded(shape: tuple[int, ...]) -> None:
    logits = torch.tensor([1.0, 2.0, 2.0]).expand(shape)
    for settings in ({"top_k": 1}, {"top_p": 0.1}, {"top_k": 2, "top_p": 1.0}):
        first = sample(
            logits, temperature=1e-40, generator=torch.Generator().manual_seed(7), **settings
        )
        second = sample(
            logits, temperature=1e-40, generator=torch.Generator().manual_seed(7), **settings
        )
        assert torch.equal(first, second)
        assert bool((first >= 1).all())
