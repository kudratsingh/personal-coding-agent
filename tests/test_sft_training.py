"""Reference checks for assistant-token loss normalization and the assistant-only dev loss."""

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.nn import functional as F

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.data.loader import BatchSource, MixedLoader
from scratchcoder.data.onepass import IGNORE
from scratchcoder.data.sft import SFTLoader, SFTRow
from scratchcoder.model.gpt import GPT
from scratchcoder.optim.adamw import AdamW, clip_gradients, cross_entropy, learning_rate
from scratchcoder.train.trainer import Trainer
from scratchcoder.train.validation import assistant_loss, token_weighted_loss

VOCAB = 272
F_, T_ = False, True


def rows() -> list[SFTRow]:
    """Five conversations of 3 to 9 tokens with 1 to 5 assistant targets: 13 in all."""
    return [
        SFTRow("a", "code", (256, 1, 2, 3, 4), (F_, F_, F_, T_, T_)),
        SFTRow("b", "language", (256, 5, 6), (F_, F_, T_)),
        SFTRow(
            "c", "code", (256, 7, 8, 9, 10, 11, 12, 13, 14), (F_, F_, F_, F_, T_, T_, T_, T_, T_)
        ),
        SFTRow("d", "language", (256, 15, 16, 17), (F_, T_, F_, T_)),
        SFTRow("e", "code", (256, 18, 19, 20, 21, 22), (F_, F_, T_, T_, T_, F_)),
    ]


def config(*, steps: int = 1, **train: Any) -> RunConfig:
    options: dict[str, Any] = {
        "steps": steps,
        "sequence_length": 8,
        "batch_size": 2,
        "accumulation_steps": 2,
        "gradient_clip": 1e6,
        "loss_normalization": "update_targets",
    }
    options.update(train)
    return RunConfig(
        model=ModelConfig(vocab_size=VOCAB, width=16, layers=1, heads=2, context=16),
        train=TrainConfig(**options),
    )


def sft_loader(items: list[SFTRow] | None = None, **options: Any) -> SFTLoader:
    return SFTLoader(
        rows() if items is None else items,
        sequence_length=8,
        batch_size=2,
        accumulation_steps=2,
        **options,
    )


def make_trainer(run: RunConfig, loader: BatchSource, root: Path, name: str) -> Trainer:
    return Trainer(
        run, loader, run_id=name, run_root=root, data_hash="authored", tokenizer_hash="authored"
    )


def step_metrics(root: Path, name: str) -> list[dict[str, Any]]:
    events = [
        json.loads(line)
        for line in (root / name / "events-rank0.jsonl").read_text().split("\n")
        if line
    ]
    return [row["fields"]["metrics"] for row in events if row["event"] == "metrics"]


def direct_mean_loss(model: GPT, items: list[SFTRow]) -> torch.Tensor:
    """Mean cross-entropy over every assistant target, one unpadded row at a time."""
    total = torch.zeros(())
    count = 0
    for item in items:
        ids = torch.tensor([item.ids])
        targets = ids[:, 1:].masked_fill(~torch.tensor([item.mask[1:]]), IGNORE)
        total = total + F.cross_entropy(
            model(ids[:, :-1]).reshape(-1, VOCAB),
            targets.flatten(),
            ignore_index=IGNORE,
            reduction="sum",
        )
        count += item.assistant_tokens
    return total / count


def test_update_gradient_is_the_mean_cross_entropy_over_assistant_tokens(tmp_path: Path) -> None:
    # Five rows of 2 x 2 = 4 per update: update 1 holds four rows over two microbatches.
    run = config()
    loader = sft_loader()
    assert loader.planned_updates == 2
    trainer = make_trainer(run, loader, tmp_path, "sft")
    losses = trainer.train()
    first = loader.reading_order()[:4]
    torch.manual_seed(run.train.seed)
    model = GPT(run.model)
    expected = direct_mean_loss(model, first)
    expected.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
    for (name, reference), actual in zip(
        model.named_parameters(), trainer.model.parameters(), strict=True
    ):
        assert reference.grad is not None and actual.grad is not None, name
        torch.testing.assert_close(actual.grad, reference.grad, rtol=1e-5, atol=1e-7)
    assert losses == [pytest.approx(float(expected.detach()), rel=1e-6)]
    assert step_metrics(tmp_path, "sft")[0]["tokens"] == sum(
        item.assistant_tokens for item in first
    )


def test_the_divisor_spans_microbatches_not_each_microbatch(tmp_path: Path) -> None:
    # Unequal target counts per microbatch tell an update mean from a mean of means.
    run = config()
    loader = sft_loader()
    trainer = make_trainer(run, loader, tmp_path, "sft")
    trainer.train()
    first = loader.reading_order()[:4]
    counts = [
        sum(item.assistant_tokens for item in first[:2]),
        sum(item.assistant_tokens for item in first[2:]),
    ]
    assert counts[0] != counts[1]
    torch.manual_seed(run.train.seed)
    model = GPT(run.model)
    mean_of_means = (direct_mean_loss(model, first[:2]) + direct_mean_loss(model, first[2:])) / 2
    mean_of_means.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
    gaps = [
        float((actual.grad - reference.grad).norm())
        for reference, actual in zip(model.parameters(), trainer.model.parameters(), strict=True)
        if actual.grad is not None and reference.grad is not None
    ]
    assert max(gaps) > 1e-4


def test_capacity_normalization_is_the_same_gradient_scaled_by_count_over_capacity(
    tmp_path: Path,
) -> None:
    targets = make_trainer(config(), sft_loader(), tmp_path, "targets")
    targets.train()
    capacity = make_trainer(
        config(loss_normalization="capacity"), sft_loader(), tmp_path, "capacity"
    )
    capacity.train()
    count = step_metrics(tmp_path, "targets")[0]["tokens"]
    assert step_metrics(tmp_path, "capacity")[0]["tokens"] == count
    scale = count / (2 * 8 * 2)
    for left, right in zip(targets.model.parameters(), capacity.model.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(right.grad, left.grad * scale, rtol=1e-5, atol=1e-9)


def test_a_stop_inside_an_update_rolls_back_and_resumes_exactly(tmp_path: Path) -> None:
    run = config(steps=2)
    reference = make_trainer(run, sft_loader(), tmp_path, "reference")
    expected = reference.train()
    interrupted = make_trainer(run, sft_loader(), tmp_path, "interrupted")
    calls = 0

    def stop() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 7  # Before the second update's second microbatch is read.

    assert interrupted.train(should_stop=stop) == expected[:1]
    assert interrupted.loader.state_dict()["rows"] == 4
    interrupted.checkpoint(tmp_path / "partial.pt")
    resumed = make_trainer(run, sft_loader(), tmp_path, "resumed")
    resumed.resume(tmp_path / "partial.pt")
    assert resumed.train() == expected[1:]
    for left, right in zip(reference.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_a_checkpoint_saved_before_the_option_existed_still_resumes(tmp_path: Path) -> None:
    run = RunConfig(
        model=ModelConfig(vocab_size=VOCAB, width=16, layers=1, heads=2, context=16),
        train=TrainConfig(steps=2, sequence_length=8, batch_size=2, accumulation_steps=2),
    )
    documents = {"web": [[1, 2, 3, 4, 5, 6, 7] * 3], "py": [[10, 11, 12] * 6]}

    def mixed() -> MixedLoader:
        return MixedLoader(
            documents, {"web": 0.5, "py": 0.5}, sequence_length=8, batch_size=2, seed=3
        )

    trainer = make_trainer(run, mixed(), tmp_path, "older")
    trainer.train(should_stop=lambda: trainer.step >= 1)
    trainer.checkpoint(tmp_path / "older.pt")
    state = torch.load(tmp_path / "older.pt", weights_only=False)
    del state["config"]["train"]["loss_normalization"]
    torch.save(state, tmp_path / "older.pt")
    (tmp_path / "older.pt.sha256").unlink(missing_ok=True)
    resumed = make_trainer(run, mixed(), tmp_path, "resumed")
    resumed.resume(tmp_path / "older.pt")
    assert resumed.step == 1
    targets = RunConfig(
        model=run.model, train=replace(run.train, loss_normalization="update_targets")
    )
    with pytest.raises(ValueError, match="configuration differs"):
        make_trainer(targets, mixed(), tmp_path, "changed").resume(tmp_path / "older.pt")


def test_update_targets_rejects_z_loss_and_empty_updates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="z-loss"):
        config(z_loss=1e-4)
    with pytest.raises(ValueError, match="loss_normalization"):
        config(loss_normalization="rows")

    class Empty:
        def next_batch(self) -> tuple[list[list[int]], list[list[int]], list[str]]:
            return [[256] * 8] * 2, [[IGNORE] * 8] * 2, ["", ""]

        def state_dict(self) -> dict[str, Any]:
            return {}

        def load_state_dict(self, state: dict[str, Any]) -> None:
            pass

    with pytest.raises(ValueError, match="no real targets"):
        make_trainer(config(), Empty(), tmp_path, "empty").train()


def test_default_pretraining_update_is_bitwise_the_written_out_capacity_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new option leaves pretraining untouched: the default config is ``capacity``."""

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("an unmasked pretraining batch left the existing path")

    monkeypatch.setattr("scratchcoder.train.trainer.cross_entropy_sum", forbidden)
    monkeypatch.setattr(Trainer, "_target_normalized", forbidden)
    run = RunConfig(
        model=ModelConfig(vocab_size=VOCAB, width=16, layers=1, heads=2, context=16),
        train=TrainConfig(steps=3, sequence_length=8, batch_size=2, accumulation_steps=2),
    )
    assert run.train.loss_normalization == "capacity"
    documents = {"web": [[1, 2, 3, 4, 5, 6, 7] * 3, [8, 9] * 9], "py": [[10, 11, 12] * 6]}

    def mixed() -> MixedLoader:
        return MixedLoader(
            documents, {"web": 0.5, "py": 0.5}, sequence_length=8, batch_size=2, seed=3
        )

    trainer = make_trainer(run, mixed(), tmp_path, "pretraining")
    losses = trainer.train()
    source = mixed()
    options = run.train
    torch.manual_seed(options.seed)
    model = GPT(run.model)
    named = list(model.named_parameters())
    optimizer = AdamW(
        [parameter for _, parameter in named],
        learning_rate=options.learning_rate,
        betas=(options.beta1, options.beta2),
        eps=options.adam_epsilon,
        weight_decay=options.weight_decay,
        decay_mask=[p.ndim >= 2 and name != "embedding.weight" for name, p in named],
    )
    expected = []
    for step in range(options.steps):
        optimizer.zero_grad()
        value = 0.0
        for _ in range(options.accumulation_steps):
            inputs, targets, _ = source.next_batch()
            nll = cross_entropy(model(torch.tensor(inputs)), torch.tensor(targets))
            (nll / options.accumulation_steps).backward()  # type: ignore[no-untyped-call]
            value += float(nll.detach()) / options.accumulation_steps
        clip_gradients(model.parameters(), options.gradient_clip)
        optimizer.lr = learning_rate(
            step,
            total_steps=options.steps,
            warmup_steps=options.warmup_steps,
            peak=options.learning_rate,
            minimum_ratio=options.min_lr_ratio,
        )
        optimizer.step()
        expected.append(value)
    assert losses == expected
    for left, right in zip(trainer.model.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert [row["tokens"] for row in step_metrics(tmp_path, "pretraining")] == [32, 32, 32]


def test_assistant_dev_loss_is_the_token_weighted_mean_over_assistant_targets() -> None:
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=VOCAB, width=16, layers=1, heads=2, context=16))
    items = rows()
    with torch.no_grad():
        expected = float(direct_mean_loss(model, items))
    for batch_size in (1, 2, 5):
        result = assistant_loss(model, iter(items), batch_size=batch_size)
        assert result["loss"] == pytest.approx(expected, rel=1e-6)
        assert result["tokens"] == 13 and result["rows"] == 5
    parts = [assistant_loss(model, items[:2]), assistant_loss(model, items[2:])]
    assert token_weighted_loss(parts) == pytest.approx(expected, rel=1e-6)
    assert model.training
    bf16 = assistant_loss(model, items, autocast_bf16=True)
    assert math.isfinite(bf16["loss"]) and abs(bf16["loss"] - expected) < 0.05 * expected


def test_assistant_dev_loss_rejects_bad_rows() -> None:
    model = GPT(ModelConfig(vocab_size=VOCAB, width=16, layers=1, heads=2, context=4))
    with pytest.raises(ValueError, match="context"):
        assistant_loss(model, rows())
    with pytest.raises(ValueError, match="no assistant targets"):
        assistant_loss(model, [SFTRow("z", "code", (256, 1), (F_, F_))])
    with pytest.raises(ValueError):
        token_weighted_loss([])
