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
