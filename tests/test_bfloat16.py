from pathlib import Path

import pytest
import torch

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.data.loader import MixedLoader
from scratchcoder.model.gpt import GPT
from scratchcoder.optim.adamw import cross_entropy
from scratchcoder.train.trainer import Trainer


@pytest.mark.parametrize("seed", [0, 7, 23])
def test_cpu_bfloat16_matches_float32_reference(seed: int) -> None:
    torch.manual_seed(seed)
    model = GPT(ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=8))
    inputs = torch.arange(16).reshape(2, 8)
    targets = (inputs + 1) % 272
    reference_logits = model(inputs)
    reference_loss = cross_entropy(reference_logits, targets)
    reference_loss.backward()  # type: ignore[no-untyped-call]
    gradients = []
    for parameter in model.parameters():
        assert parameter.grad is not None
        gradients.append(parameter.grad.detach().clone())
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual_logits = model(inputs)
        actual_loss = cross_entropy(actual_logits, targets)
    assert actual_logits.dtype == torch.bfloat16
    assert actual_loss.dtype == torch.float32
    torch.testing.assert_close(actual_logits.float(), reference_logits, atol=0.005, rtol=0.03)
    torch.testing.assert_close(actual_loss, reference_loss, atol=0.001, rtol=0.001)
    actual_loss.backward()  # type: ignore[no-untyped-call]
    for parameter, reference in zip(model.parameters(), gradients, strict=True):
        assert parameter.dtype == torch.float32
        assert bool(torch.isfinite(parameter).all())
        assert parameter.grad is not None
        assert parameter.grad.dtype == torch.float32
        assert bool(torch.isfinite(parameter.grad).all())
        norm = float(reference.norm())
        error = float((parameter.grad - reference).norm())
        assert error <= 0.03 * norm if norm else error == 0


def trainer(root: Path, name: str, *, accumulation: int = 1, memorize: bool = False) -> Trainer:
    config = RunConfig(
        model=ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=8),
        train=TrainConfig(
            steps=100 if memorize else 6,
            sequence_length=8,
            batch_size=2,
            accumulation_steps=accumulation,
            precision="bfloat16",
            learning_rate=0.02 if memorize else 0.0006,
        ),
        data_mix={"web": 1.0},
    )
    source = [[1] * 32] if memorize else [[1, 2, 3, 4] * 8, [4, 3, 2, 1] * 8]
    loader = MixedLoader(
        {"web": source}, {"web": 1.0}, sequence_length=8, batch_size=2, seed=config.train.seed
    )
    return Trainer(
        config,
        loader,
        run_id=name,
        run_root=root,
        data_hash="authored-synthetic",
        tokenizer_hash="authored-synthetic",
        device="cpu",
    )


@pytest.mark.parametrize("accumulation", [1, 2])
def test_cpu_bfloat16_checkpoint_resume_is_exact(tmp_path: Path, accumulation: int) -> None:
    full = trainer(tmp_path, "full", accumulation=accumulation)
    expected = full.train()
    partial = trainer(tmp_path, "partial", accumulation=accumulation)
    assert partial.train(stop_after=3) == expected[:3]
    checkpoint = tmp_path / "partial.pt"
    partial.checkpoint(checkpoint)
    resumed = trainer(tmp_path, "resumed", accumulation=accumulation)
    resumed.resume(checkpoint)
    assert resumed.train() == expected[3:]
    assert resumed.loader.state_dict() == full.loader.state_dict()
    assert resumed.optimizer.steps == full.optimizer.steps
    for actual, reference in zip(resumed.model.parameters(), full.model.parameters(), strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    for name in ("m", "v"):
        for actual, reference in zip(
            getattr(resumed.optimizer, name), getattr(full.optimizer, name), strict=True
        ):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_cpu_bfloat16_memorizes_one_batch(tmp_path: Path) -> None:
    instance = trainer(tmp_path, "memorize", memorize=True)
    losses = instance.train()
    assert losses[-1] < 0.05
    assert losses[-1] < losses[0] / 50
    assert all(parameter.dtype == torch.float32 for parameter in instance.model.parameters())
