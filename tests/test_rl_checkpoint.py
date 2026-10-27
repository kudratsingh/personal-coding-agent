"""Weights-only checkpoints: bit-for-bit round trips and compatibility with existing loaders."""

from pathlib import Path

import pytest
import torch

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.data.sft import SFTLoader, SFTRow
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.model.gpt import GPT
from scratchcoder.rl.checkpoint import (
    compare_weights,
    file_sha256,
    load_weights,
    read_weights,
    save_weights,
)
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.train.trainer import Trainer

CONFIG = ModelConfig(vocab_size=272, width=16, layers=2, heads=2, context=32)


def trained_model() -> GPT:
    """A model whose weights are not the initialization, with awkward float values."""
    torch.manual_seed(1)
    model = GPT(CONFIG)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 1e-3).mul_(1.0 + 1e-7)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_saving_and_loading_reproduces_the_weights_bit_for_bit(
    dtype: torch.dtype, tmp_path: Path
) -> None:
    model = trained_model().to(dtype)
    path = tmp_path / "weights.pt"
    digest = save_weights(path, model, tokenizer_hash="tok", run_id="rl-run", step=12)
    assert digest == file_sha256(path)
    loaded = load_weights(path, tokenizer_hash="tok", expected_sha256=digest)
    original = model.state_dict()
    restored = loaded.state_dict()
    assert original.keys() == restored.keys()
    for name, value in original.items():
        assert restored[name].dtype == value.dtype
        assert torch.equal(restored[name].view(torch.uint8), value.view(torch.uint8)), name
    state = read_weights(path)
    assert state["format"] == "weights-v1" and state["step"] == 12 and state["run_id"] == "rl-run"


def test_the_same_weights_saved_twice_compare_equal_though_their_files_differ(
    tmp_path: Path,
) -> None:
    """The run 2 rule: an endpoint is the SFT weights when its tensors equal them bit for bit,
    whatever its file's SHA-256 (two saves of one model record another run ID and step)."""
    model = trained_model()
    first, second = tmp_path / "sft.pt", tmp_path / "rl-check-0000.pt"
    save_weights(first, model, tokenizer_hash="tok", run_id="posttrain-1-sft", step=106)
    save_weights(second, load_weights(first), tokenizer_hash="tok", run_id="posttrain-2-rl", step=0)
    assert file_sha256(first) != file_sha256(second)
    same = compare_weights(first, second)
    assert same.same and same.first_difference is None
    assert same.tensors == len(model.state_dict())
    # One bit of one weight differs: not the same weights.
    with torch.no_grad():
        weight = model.embedding.weight
        weight.view(torch.int32)[0, 0] ^= 1
    changed = tmp_path / "changed.pt"
    save_weights(changed, model, tokenizer_hash="tok", run_id="posttrain-2-rl", step=12)
    found = compare_weights(first, changed)
    assert not found.same and found.first_difference == "embedding.weight"
    # Another dtype with equal values is another model file: not the same weights either.
    halved = tmp_path / "bf16.pt"
    save_weights(
        halved, load_weights(first).to(torch.bfloat16), tokenizer_hash="t", run_id="r", step=0
    )
    assert not compare_weights(first, halved).same
    # NaN and -0.0 compare as stored, bit for bit.
    with torch.no_grad():
        weight.view(torch.int32)[0, 0] ^= 1
        weight[0, 1] = float("nan")
        weight[0, 2] = -0.0
    odd = tmp_path / "odd.pt"
    save_weights(odd, model, tokenizer_hash="tok", run_id="a", step=1)
    again = tmp_path / "odd-again.pt"
    save_weights(again, load_weights(odd), tokenizer_hash="tok", run_id="b", step=2)
    assert compare_weights(odd, again).same
    with torch.no_grad():
        weight[0, 2] = 0.0
    zero = tmp_path / "zero.pt"
    save_weights(zero, model, tokenizer_hash="tok", run_id="c", step=3)
    assert not compare_weights(odd, zero).same


def test_altered_bytes_and_identities_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    digest = save_weights(path, trained_model(), tokenizer_hash="tok", run_id="r", step=0)
    with pytest.raises(ValueError):
        read_weights(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError):
        load_weights(path, tokenizer_hash="other")
    assert load_weights(path, expected_sha256=digest) is not None
    for kwargs in (
        {"tokenizer_hash": "", "run_id": "r", "step": 0},
        {"tokenizer_hash": "t", "run_id": "r", "step": -1},
    ):
        with pytest.raises(ValueError):
            save_weights(tmp_path / "bad.pt", trained_model(), **kwargs)  # type: ignore[arg-type]
    assert not (tmp_path / "bad.pt").exists()


def test_a_weights_checkpoint_is_a_valid_trainer_parent(tmp_path: Path) -> None:
    model = trained_model()
    parent = tmp_path / "parent.pt"
    save_weights(parent, model, tokenizer_hash="tok", run_id="rl-run", step=3)
    rows = [SFTRow("a", "code", (256, 1, 2, 3), (False, False, True, True))]
    loader = SFTLoader(rows, sequence_length=8, batch_size=1, accumulation_steps=1)
    run = RunConfig(
        model=CONFIG,
        train=TrainConfig(steps=1, sequence_length=8, batch_size=1, warmup_steps=0),
        parent_checkpoint=str(parent),
    )
    trainer = Trainer(
        run, loader, run_id="child", run_root=tmp_path, data_hash="d", tokenizer_hash="tok"
    )
    for name, value in model.state_dict().items():
        assert torch.equal(trainer.model.state_dict()[name], value)
    assert trainer.record.parent_run_id == "rl-run"


def test_a_weights_checkpoint_loads_for_evaluation(tmp_path: Path) -> None:
    tokenizer = ByteBPE()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    model = trained_model()
    path = tmp_path / "weights.pt"
    save_weights(path, model, tokenizer_hash=tokenizer.sha256, run_id="rl-run", step=0)
    loaded, _ = load_checkpoint(path, tokenizer_path)
    for name, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)
