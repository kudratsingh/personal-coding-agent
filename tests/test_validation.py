import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.data.loader import MixedLoader
from scratchcoder.data.shards import TokenShard, write_shard
from scratchcoder.model.gpt import GPT
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.train.trainer import Trainer
from scratchcoder.train.validation import ValidationWindow, shard_windows, validate


def test_validation_matches_uniform_reference_and_restores_model() -> None:
    model = GPT(ModelConfig(vocab_size=272, width=16, heads=2, layers=1, context=16))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state()
    metrics = validate(model, (ValidationWindow((256, 97, 98), 2), ValidationWindow((256, 99), 1)))
    assert metrics["loss"] == pytest.approx(math.log(272), abs=1e-6)
    assert metrics["bits_per_byte"] == pytest.approx(math.log2(272), abs=1e-6)
    assert metrics["tokens"] == metrics["bytes"] == 3
    assert model.training and all(parameter.grad is None for parameter in model.parameters())
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in weights.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    model.eval()
    with pytest.raises(ValueError, match="context"):
        validate(model, (ValidationWindow(tuple([97] * 18), 17),))
    assert not model.training


def test_raw_bytes_not_lossy_utf8_decoding(tmp_path: Path) -> None:
    tokenizer = ByteBPE()
    path = tmp_path / "unicode.bin"
    write_shard(path, [tokenizer.encode("🐍")], vocab_size=272, tokenizer_hash=tokenizer.sha256)
    windows = shard_windows(
        TokenShard(path, tokenizer_hash=tokenizer.sha256), tokenizer, documents=1, sequence_length=1
    )
    assert windows[0].target_bytes == 1
    assert len(tokenizer.decode(windows[0].tokens[1:]).encode()) == 3


def test_periodic_validation_preserves_training_and_checkpoint_identity(tmp_path: Path) -> None:
    config = RunConfig(
        model=ModelConfig(vocab_size=272, width=16, heads=2, layers=1, context=16),
        train=TrainConfig(steps=3, sequence_length=4, batch_size=1, validation_interval=2),
    )
    windows: dict[str, tuple[ValidationWindow, ...]] = {
        "web": (ValidationWindow((256, 1, 2, 3), 3),),
        "python": (ValidationWindow((256, 4, 5), 2),),
    }

    def trainer(
        name: str, data: dict[str, tuple[ValidationWindow, ...]] | None, options: RunConfig = config
    ) -> Trainer:
        return Trainer(
            options,
            MixedLoader(
                {"web": [[256, 1, 2, 3] * 8]}, {"web": 1.0}, sequence_length=4, batch_size=1, seed=0
            ),
            run_id=name,
            run_root=tmp_path,
            data_hash="data",
            tokenizer_hash="tokens",
            validation=data,
        )

    validated = trainer("validated", windows)
    losses = validated.train()
    reference = trainer(
        "reference", None, replace(config, train=replace(config.train, validation_interval=0))
    )
    assert losses == reference.train()
    for left, right in zip(validated.model.parameters(), reference.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert validated.loader.state_dict() == reference.loader.state_dict()
    events = [
        json.loads(line)
        for line in (tmp_path / "validated/events-rank0.jsonl").read_text().splitlines()
    ]
    validation_events = [event["fields"] for event in events if event["event"] == "validation"]
    assert [event["step"] for event in validation_events] == [0, 2, 3]
    assert "validation/python/bits_per_byte" in validation_events[0]["metrics"]
    checkpoint = tmp_path / "checkpoint.pt"
    validated.checkpoint(checkpoint)
    trainer("same", windows).resume(checkpoint)
    changed = dict(windows, web=(ValidationWindow((256, 7, 8, 9), 3),))
    with pytest.raises(ValueError, match="validation identity"):
        trainer("different", changed).resume(checkpoint)
    with pytest.raises(ValueError, match="supplied together"):
        trainer("missing", None)
