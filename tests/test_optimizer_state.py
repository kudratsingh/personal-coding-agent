from copy import deepcopy
from typing import Any, cast

import pytest
import torch

from scratchcoder.optim.adamw import AdamW


def optimizer() -> AdamW:
    return AdamW([torch.tensor([1.0, 2.0], requires_grad=True)])


@pytest.mark.parametrize("field,value", [("steps", [-1]), ("v", [torch.tensor([-1.0, 0.0])])])
def test_malformed_state_is_not_accepted(field: str, value: object) -> None:
    target = optimizer()
    invalid = target.state_dict()
    invalid[field] = value
    with pytest.raises(ValueError):
        target.load_state_dict(invalid)


def test_malformed_second_moment_does_not_mutate_first() -> None:
    target = optimizer()
    before = deepcopy(target.state_dict())
    invalid = target.state_dict()
    invalid["m"] = [torch.ones(2)]
    invalid["v"] = [torch.zeros(3)]
    with pytest.raises(ValueError):
        target.load_state_dict(invalid)
    torch.testing.assert_close(target.m[0], before["m"][0], atol=0, rtol=0)


def test_exported_decay_mask_is_not_aliased() -> None:
    target = optimizer()
    state = target.state_dict()
    state["decay_mask"][0] = False
    assert target.decay_mask == [True]


def assert_same_state(left: dict[str, Any], right: dict[str, Any]) -> None:
    for name in ("m", "v"):
        for actual, expected in zip(left[name], right[name], strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for name in set(left) - {"m", "v"}:
        assert left[name] == right[name]


@pytest.mark.parametrize(
    "field,value",
    [
        ("m", None),
        ("m", [1]),
        ("m", [torch.zeros(3)]),
        ("m", [torch.zeros(2, dtype=torch.float64)]),
        ("m", [torch.full((2,), float("nan"))]),
        ("v", [torch.full((2,), float("inf"))]),
        ("v", [torch.empty(2, device="meta")]),
        ("v", [torch.zeros(2).to_sparse()]),
        ("steps", []),
        ("steps", [True]),
        ("steps", [1.5]),
        ("decay_mask", []),
        ("decay_mask", [1]),
        ("decay_mask", ["false"]),
        ("lr", -1),
        ("lr", True),
        ("lr", float("nan")),
        ("lr", float("inf")),
        ("lr", "0.001"),
        ("lr", 10**1000),
        ("eps", 0),
        ("eps", False),
        ("weight_decay", -1),
        ("weight_decay", float("inf")),
        ("betas", None),
        ("betas", [0.9]),
        ("betas", [0.9, 1.0]),
        ("betas", [True, 0.9]),
        ("betas", [0.9, float("nan")]),
    ],
)
def test_rejected_state_preserves_next_update(field: str, value: Any) -> None:
    target, control = optimizer(), optimizer()
    for instance in (target, control):
        instance.parameters[0].grad = torch.tensor([0.1, -0.2])
        instance.step()
    invalid = target.state_dict()
    invalid[field] = value
    before = target.state_dict()
    with pytest.raises(ValueError):
        target.load_state_dict(invalid)
    assert_same_state(target.state_dict(), before)
    for _ in range(3):
        for instance in (target, control):
            instance.parameters[0].grad = torch.tensor([-0.3, 0.4])
            instance.step()
        torch.testing.assert_close(target.parameters[0], control.parameters[0], atol=0, rtol=0)
        assert_same_state(target.state_dict(), control.state_dict())


@pytest.mark.parametrize("change", ["missing", "extra", "not-dict"])
def test_state_schema_rejected_without_mutation(change: str) -> None:
    target = optimizer()
    before = target.state_dict()
    invalid: Any = target.state_dict()
    if change == "missing":
        del invalid["m"]
    elif change == "extra":
        invalid["unknown"] = 0
    else:
        invalid = None
    with pytest.raises(ValueError, match="state fields"):
        target.load_state_dict(invalid)
    assert_same_state(target.state_dict(), before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_roundtrip_and_caller_copy_independence(dtype: torch.dtype) -> None:
    parameters = [torch.tensor([1.0, 2.0], dtype=dtype), torch.tensor([3.0], dtype=dtype)]
    first = AdamW(parameters, learning_rate=0.01, decay_mask=[True, False])
    for step in range(5):
        parameters[0].grad = torch.full_like(parameters[0], 0.1)
        parameters[1].grad = None if step % 2 == 0 else torch.full_like(parameters[1], -0.2)
        first.step()
    assert first.steps == [5, 2]
    resumed = AdamW([value.clone() for value in parameters], learning_rate=0.5)
    saved = first.state_dict()
    resumed.load_state_dict(saved)
    assert_same_state(first.state_dict(), resumed.state_dict())
    saved["m"][0].fill_(99)
    saved["v"][0].fill_(99)
    saved["decay_mask"][0] = False
    saved["steps"][0] = 999
    assert_same_state(first.state_dict(), resumed.state_dict())
    for step in range(20):
        for instance in (first, resumed):
            instance.parameters[0].grad = torch.full_like(parameters[0], 0.3)
            instance.parameters[1].grad = None if step % 2 else torch.full_like(parameters[1], -0.4)
            instance.step()
        for actual, expected in zip(first.parameters, resumed.parameters, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert_same_state(first.state_dict(), resumed.state_dict())


def test_empty_optimizer_roundtrip() -> None:
    target = AdamW([])
    target.load_state_dict(target.state_dict())
    target.step()
    assert target.steps == []


def test_loaded_moments_are_detached() -> None:
    target = optimizer()
    state = target.state_dict()
    state["m"][0].requires_grad_()
    state["v"][0].requires_grad_()
    target.load_state_dict(state)
    assert not target.m[0].requires_grad and not target.v[0].requires_grad


def test_copy_failure_does_not_commit_partial_state(monkeypatch: pytest.MonkeyPatch) -> None:
    target = optimizer()
    before = target.state_dict()
    incoming = target.state_dict()
    incoming["m"][0].fill_(0.25)
    incoming["lr"] = 0.5
    original: Any = torch.Tensor.to
    calls = 0

    def fail_second_copy(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("authored transfer failure")
        return cast(torch.Tensor, original(tensor, *args, **kwargs))

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", fail_second_copy)
        with pytest.raises(RuntimeError, match="authored transfer failure"):
            target.load_state_dict(incoming)
    assert calls == 2
    assert_same_state(target.state_dict(), before)
