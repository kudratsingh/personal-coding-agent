"""Constructor inputs must be owned and compatible with native checkpoint validation."""

from typing import Any

import pytest
import torch

from scratchcoder.optim.adamw import AdamW


def test_caller_mask_and_betas_cannot_change_subsequent_updates() -> None:
    mask = [True, False]
    betas: Any = [0.9, 0.95]
    parameters = [torch.ones(2, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    optimizer = AdamW(parameters, learning_rate=0.1, weight_decay=0.5, decay_mask=mask, betas=betas)
    mask[:] = [False, True]
    betas[:] = [0.0, 0.0]
    assert optimizer.decay_mask == [True, False] and optimizer.betas == (0.9, 0.95)
    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    torch.testing.assert_close(parameters[0], torch.full_like(parameters[0], 0.95), atol=0, rtol=0)
    torch.testing.assert_close(parameters[1], torch.ones_like(parameters[1]), atol=0, rtol=0)


@pytest.mark.parametrize(
    "settings",
    [
        {"learning_rate": True},
        {"learning_rate": "0.1"},
        {"learning_rate": -1},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"learning_rate": 10**1000},
        {"eps": False},
        {"eps": "1e-8"},
        {"eps": 0},
        {"eps": float("nan")},
        {"eps": float("inf")},
        {"weight_decay": True},
        {"weight_decay": "0.1"},
        {"weight_decay": -1},
        {"weight_decay": float("inf")},
        {"betas": None},
        {"betas": []},
        {"betas": [0.9]},
        {"betas": [0.9, 0.95, 0.99]},
        {"betas": [True, 0.95]},
        {"betas": [0.9, 1.0]},
        {"betas": [-0.1, 0.95]},
        {"betas": [0.9, float("nan")]},
        {"betas": ["0.9", 0.95]},
        {"decay_mask": [1]},
        {"decay_mask": ["false"]},
        {"decay_mask": []},
        {"decay_mask": [False, True]},
        {"decay_mask": "x"},
    ],
)
def test_invalid_constructors_raise_value_error(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        AdamW([torch.ones(2, requires_grad=True)], **settings)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"learning_rate": 0},
        {"betas": (0, 0), "weight_decay": 0},
        {"betas": [0.8, 0.9], "decay_mask": [False]},
        {"decay_mask": (True,)},
    ],
)
def test_valid_constructor_state_roundtrips_and_continues_exactly(
    dtype: torch.dtype, settings: dict[str, Any]
) -> None:
    actual = torch.tensor([1.0, -2.0], dtype=dtype, requires_grad=True)
    first = AdamW([actual], **settings)
    actual.grad = torch.tensor([0.1, 0.2], dtype=dtype)
    first.step()
    expected = actual.detach().clone().requires_grad_()
    second = AdamW([expected])
    second.load_state_dict(first.state_dict())
    for index in range(10):
        for parameter in (actual, expected):
            parameter.grad = torch.tensor([index / 10, -0.2], dtype=dtype)
        first.step()
        second.step()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_empty_constructor_remains_supported() -> None:
    optimizer = AdamW([], learning_rate=0, betas=(0, 0), decay_mask=[])
    optimizer.step()
    optimizer.load_state_dict(optimizer.state_dict())
    assert optimizer.parameters == optimizer.steps == optimizer.decay_mask == []
