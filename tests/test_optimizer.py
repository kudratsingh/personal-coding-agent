import math

import pytest
import torch
from torch.nn import functional as F

from scratchcoder.optim.adamw import AdamW, clip_gradients, cross_entropy, learning_rate


def test_adamw_matches_pytorch_for_100_steps() -> None:
    torch.manual_seed(1)
    actual = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    reference = actual.detach().clone().requires_grad_()
    optimizer = AdamW([actual], learning_rate=0.01, weight_decay=0.2)
    trusted = torch.optim.AdamW([reference], lr=0.01, betas=(0.9, 0.95), weight_decay=0.2)
    for _ in range(100):
        gradient = torch.randn_like(actual)
        actual.grad, reference.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        trusted.step()
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)


def test_loss_and_gradient_match_reference() -> None:
    logits = (torch.randn(2, 3, 7) * 50).requires_grad_()
    targets = torch.randint(0, 7, (2, 3))
    actual = cross_entropy(logits, targets)
    reference = F.cross_entropy(logits.reshape(-1, 7), targets.flatten())
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits)[0], torch.autograd.grad(reference, logits)[0]
    )


def test_schedule_and_clipping_by_hand() -> None:
    values = [
        learning_rate(step, total_steps=7, warmup_steps=2, peak=1, minimum_ratio=0.2)
        for step in range(7)
    ]
    assert values[0] == 0.5
    assert values[1] == values[2] == 1
    assert values[4] == pytest.approx(0.6)
    assert values[6] == pytest.approx(0.2)
    parameter = torch.zeros(2, requires_grad=True)
    parameter.grad = torch.tensor([3.0, 4.0])
    assert clip_gradients([parameter], 1) == 5
    assert float(parameter.grad.norm()) == pytest.approx(1, abs=1e-6)
    parameter.grad[0] = math.inf
    with pytest.raises(FloatingPointError):
        clip_gradients([parameter], 1)


def explicit_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """The earlier three-copy formulation, kept as a numerical reference."""
    shifted = logits.float() - logits.float().amax(dim=-1, keepdim=True)
    log_normalizer = shifted.exp().sum(dim=-1).log()
    chosen = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - chosen).mean()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cross_entropy_matches_explicit_formulation_with_one_saved_copy(
    dtype: torch.dtype,
) -> None:
    from torch.autograd.graph import saved_tensors_hooks

    torch.manual_seed(3)
    batch, time, vocabulary = 2, 16, 4096
    logits = (torch.randn(batch, time, vocabulary) * 8).to(dtype).requires_grad_(True)
    targets = torch.randint(0, vocabulary, (batch, time))
    reference = explicit_cross_entropy(logits, targets)
    reference.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
    assert logits.grad is not None
    reference_gradient = logits.grad.clone()
    logits.grad = None
    storages: dict[int, int] = {}

    def pack(value: torch.Tensor) -> torch.Tensor:
        storages[value.untyped_storage().data_ptr()] = value.untyped_storage().nbytes()
        return value

    with saved_tensors_hooks(pack, lambda value: value):
        actual = cross_entropy(logits, targets)
    actual.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(logits.grad, reference_gradient, atol=1e-7, rtol=1e-5)
    full_size = batch * time * vocabulary
    saved_full_tensors = [size for size in storages.values() if size >= full_size]
    assert sum(saved_full_tensors) <= 4 * full_size + full_size * logits.element_size()
    assert len(saved_full_tensors) <= 2  # fp32 log-probabilities plus the leaf itself
