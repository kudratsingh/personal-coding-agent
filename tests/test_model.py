import pytest
import torch
from torch.nn import functional as F

from scratchcoder.common.config import ModelConfig
from scratchcoder.model.gpt import GPT, causal_attention, expected_parameters, rms_norm, rotary


def test_attention_outputs_and_gradients_match_reference() -> None:
    torch.manual_seed(3)
    values = [torch.randn(2, 4, 7, 8, requires_grad=True) for _ in range(3)]
    actual = causal_attention(*values)
    reference = F.scaled_dot_product_attention(values[0], values[1], values[2], is_causal=True)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
    gradients = torch.autograd.grad(actual.square().sum(), values, retain_graph=True)
    expected = torch.autograd.grad(reference.square().sum(), values)
    for left, right in zip(gradients, expected, strict=True):
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)


def test_norm_rotary_and_parameter_counts() -> None:
    x = torch.randn(2, 4, 7, 8)
    weight = torch.ones(8)
    torch.testing.assert_close(rms_norm(x, weight), F.rms_norm(x, (8,), weight, eps=1e-6))
    torch.testing.assert_close(rotary(x).square().sum(-1), x.square().sum(-1))
    config = ModelConfig(width=32, layers=2, heads=4, context=16)
    assert GPT(config).parameter_count() == expected_parameters(config)


def test_causality_and_shape() -> None:
    torch.manual_seed(0)
    model = GPT(ModelConfig(width=32, layers=2, heads=4, context=16))
    ids = torch.randint(0, 512, (2, 8))
    changed = ids.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % 512
    torch.testing.assert_close(model(ids)[:, :5], model(changed)[:, :5], atol=0, rtol=0)
    assert model(ids).shape == (2, 8, 512)
    with pytest.raises(ValueError):
        model(torch.ones(2, 17, dtype=torch.long))
