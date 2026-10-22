import math
from dataclasses import replace
from typing import cast

import pytest
import torch
from torch.nn import functional as F

from scratchcoder.common.config import ModelConfig
from scratchcoder.model.gpt import (
    GPT,
    Block,
    causal_attention,
    expected_parameters,
    rms_norm,
    rotary,
)


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


def test_fused_attention_matches_reference_outputs_and_gradients() -> None:
    from scratchcoder.model.gpt import fused_causal_attention

    torch.manual_seed(5)
    values = [torch.randn(2, 4, 16, 8, requires_grad=True) for _ in range(3)]
    reference = causal_attention(*values)
    reference.square().sum().backward()  # type: ignore[no-untyped-call]
    reference_gradients = [value.grad.clone() for value in values if value.grad is not None]
    for value in values:
        value.grad = None
    fused = fused_causal_attention(*values)
    fused.square().sum().backward()  # type: ignore[no-untyped-call]
    torch.testing.assert_close(fused, reference, atol=1e-6, rtol=1e-5)
    for value, expected in zip(values, reference_gradients, strict=True):
        torch.testing.assert_close(value.grad, expected, atol=1e-5, rtol=1e-4)
    with pytest.raises(ValueError, match="equal query and key"):
        fused_causal_attention(values[0], values[1][:, :, :8], values[2][:, :, :8])


@pytest.mark.parametrize("autocast", [False, True])
def test_fused_model_path_matches_reference_model_path(autocast: bool) -> None:
    from scratchcoder.optim.adamw import cross_entropy

    config = ModelConfig(vocab_size=300, width=32, layers=2, heads=4, context=16)
    torch.manual_seed(11)
    reference = GPT(config)
    fused = GPT(config)
    fused.load_state_dict(reference.state_dict())
    fused.set_fused_attention(True)
    assert fused.fused_attention is True and reference.fused_attention is False
    ids = torch.randint(0, 300, (3, 16))
    targets = torch.randint(0, 300, (3, 16))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=autocast):
        reference_loss = cross_entropy(reference(ids), targets)
        fused_loss = cross_entropy(fused(ids), targets)
    reference_loss.backward()  # type: ignore[no-untyped-call]
    fused_loss.backward()  # type: ignore[no-untyped-call]
    atol = rtol = 2e-2 if autocast else 1e-5
    torch.testing.assert_close(fused_loss, reference_loss, atol=atol, rtol=rtol)
    for left, right in zip(fused.parameters(), reference.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        error = float((left.grad - right.grad).norm())
        scale = float(right.grad.norm())
        assert error <= (0.05 if autocast else 1e-4) * max(scale, 1e-12)
    # Future tokens still cannot influence earlier logits through the fused kernel.
    changed = ids.clone()
    changed[:, 10:] = (changed[:, 10:] + 1) % 300
    with torch.no_grad():
        torch.testing.assert_close(fused(changed)[:, :10], fused(ids)[:, :10], atol=0, rtol=0)
    # Cached generation always uses the reference path and is unaffected by the flag.
    with torch.no_grad():
        cached, _ = fused.forward_cached(ids)
        torch.testing.assert_close(cached, reference(ids), atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError):
        fused.set_fused_attention(1)  # type: ignore[arg-type]


def test_residual_projections_use_depth_scaled_initialization() -> None:
    torch.manual_seed(2)
    config = ModelConfig(vocab_size=300, width=256, layers=8, heads=4, context=16)
    model = GPT(config)
    residual_std = 0.02 / math.sqrt(2 * config.layers)
    for module in model.blocks:
        block = cast(Block, module)
        assert float(block.attention.output.weight.std()) == pytest.approx(residual_std, rel=0.05)
        assert float(block.down.weight.std()) == pytest.approx(residual_std, rel=0.05)
        assert float(block.attention.qkv.weight.std()) == pytest.approx(0.02, rel=0.05)
        assert float(block.up.weight.std()) == pytest.approx(0.02, rel=0.05)
    assert float(model.embedding.weight.std()) == pytest.approx(0.02, rel=0.05)


def test_qk_norm_adds_only_shared_head_gains_and_keeps_causality() -> None:
    config = ModelConfig(vocab_size=300, width=32, layers=2, heads=4, context=16, qk_norm=True)
    model = GPT(config)
    assert model.parameter_count() == expected_parameters(config)
    plain = expected_parameters(
        ModelConfig(vocab_size=300, width=32, layers=2, heads=4, context=16)
    )
    assert expected_parameters(config) - plain == 2 * 2 * 8  # two gains of head width per layer
    pilot = ModelConfig(vocab_size=65536, width=768, layers=12, heads=12, context=1024)
    assert expected_parameters(pilot) == 135_285_504
    assert expected_parameters(replace(pilot, qk_norm=True)) == 135_285_504 + 12 * 2 * 64
    ids = torch.randint(0, 300, (2, 16))
    changed = ids.clone()
    changed[:, 9:] = (changed[:, 9:] + 1) % 300
    with torch.no_grad():
        torch.testing.assert_close(model(changed)[:, :9], model(ids)[:, :9], atol=0, rtol=0)
        model.set_fused_attention(True)
        fused = model(ids)
        model.set_fused_attention(False)
        torch.testing.assert_close(fused, model(ids), atol=1e-5, rtol=1e-5)
        cached, _ = model.forward_cached(ids)
        torch.testing.assert_close(cached, model(ids), atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="qk_norm"):
        ModelConfig(qk_norm=1)  # type: ignore[arg-type]
