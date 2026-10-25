"""Reference checks for the exact one-pass packed loader and the masked final update."""

import hashlib
import json
import math
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch.nn import functional as F

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.data.onepass import (
    CHAIN_START,
    IGNORE,
    ORDER_DTYPE,
    OnePassLoader,
    chain_digest,
    combine_digests,
    multiset_digest,
    order_digest,
    pass_order,
)
from scratchcoder.data.shards import TokenShard, file_sha256, write_shard
from scratchcoder.model.gpt import GPT
from scratchcoder.optim.adamw import (
    AdamW,
    clip_gradients,
    cross_entropy,
    cross_entropy_sum,
    learning_rate,
)
from scratchcoder.train.trainer import Trainer

BOS, VOCAB = 256, 272
Shards = list[tuple[str, TokenShard, str]]


def shard(tmp_path: Path, name: str, documents: Sequence[Sequence[int]]) -> tuple[TokenShard, str]:
    path = tmp_path / f"{name}.bin"
    write_shard(path, documents, vocab_size=VOCAB, tokenizer_hash="authored", bos_id=BOS)
    return TokenShard(path, tokenizer_hash="authored", vocab_size=VOCAB, bos_id=BOS), file_sha256(
        path
    )


def authored(tmp_path: Path) -> Shards:
    """Two streams; with BOS the documents are 4, 3 (web) and 2, 5, 4 (py) tokens."""
    web, web_sha = shard(tmp_path, "web", [[1, 2, 3], [4, 5]])
    py, py_sha = shard(tmp_path, "py", [[6], [7, 8, 9, 10], [11, 12, 13]])
    return [("web", web, web_sha), ("py", py, py_sha)]


def manual_order(pairs: Sequence[tuple[int, int]]) -> npt.NDArray[np.void]:
    order = np.empty(len(pairs), dtype=ORDER_DTYPE)
    order["shard"] = [ordinal for ordinal, _ in pairs]
    order["index"] = [index for _, index in pairs]
    return order


# (shard ordinal, index): py[1], web[0], py[0], web[1], py[2]
HAND_ORDER = [(1, 1), (0, 0), (1, 0), (0, 1), (1, 2)]


def hand_loader(tmp_path: Path, *, batch_size: int = 2, accumulation: int = 2) -> OnePassLoader:
    return OnePassLoader(
        authored(tmp_path),
        manual_order(HAND_ORDER),
        sequence_length=4,
        batch_size=batch_size,
        accumulation_steps=accumulation,
    )


def drain(loader: OnePassLoader) -> list[tuple[list[list[int]], list[list[int]], list[str]]]:
    batches = []
    while not loader.exhausted:
        batches.append(loader.next_batch())
    return batches


def test_hand_computed_packing_ragged_window_padding_and_exhaustion(tmp_path: Path) -> None:
    # Stream: sentinel, then py[1] web[0] py[0] web[1] py[2]:
    # 256 | 256 7 8 9 10 | 256 1 2 3 | 256 6 | 256 4 5 | 256 11 12 13  (18 shard tokens)
    loader = hand_loader(tmp_path)
    assert (loader.total_targets, loader.total_windows, loader.planned_updates) == (18, 5, 2)
    pad, ignored = [BOS] * 4, [IGNORE] * 4
    assert drain(loader) == [
        ([[256, 256, 7, 8], [9, 10, 256, 1]], [[256, 7, 8, 9], [10, 256, 1, 2]], ["py", "py"]),
        ([[2, 3, 256, 6], [256, 4, 5, 256]], [[3, 256, 6, 256], [4, 5, 256, 11]], ["web", "web"]),
        # The ragged last window: two real targets, then BOS inputs / IGNORE targets.
        ([[11, 12, 256, 256], pad], [[12, 13, IGNORE, IGNORE], ignored], ["py", ""]),
        # The rest of the final update is fully padded, keeping the shape fixed.
        ([pad, pad], [ignored, ignored], ["", ""]),
    ]
    assert loader.exhausted and loader.emitted_windows == 5 and loader.emitted_rows == 8
    with pytest.raises(RuntimeError, match="exhausted: all 2 planned updates"):
        loader.next_batch()


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_every_token_is_a_target_once_and_the_sentinel_only_an_input(
    tmp_path: Path, seed: int
) -> None:
    shards = authored(tmp_path)
    order = pass_order([(sha, len(opened)) for _, opened, sha in shards], seed)
    loader = OnePassLoader(shards, order, sequence_length=3, batch_size=2, accumulation_steps=2)
    documents = [
        shards[int(o)][1][int(i)] for o, i in zip(order["shard"], order["index"], strict=True)
    ]
    stream = [BOS] + [token for document in documents for token in document]
    inputs: list[int] = []
    targets: list[int] = []
    for batch_inputs, batch_targets, _ in drain(loader):
        for row_inputs, row_targets in zip(batch_inputs, batch_targets, strict=True):
            real = [i for i, target in enumerate(row_targets) if target != IGNORE]
            assert real == list(range(len(real)))  # Padding only ever follows real targets.
            assert all(token == BOS for token in row_inputs[len(real) :])
            inputs.extend(row_inputs[: len(real)])
            targets.extend(row_targets[: len(real)])
    assert targets == stream[1:]  # Every shard token, in order, exactly once.
    assert inputs == stream[:-1]  # The sentinel is the first input and never a target.
    assert loader.ticket()["targets"] == len(targets) == 18


@pytest.mark.parametrize(
    ("documents", "length", "batch", "accumulation", "windows", "updates"),
    [
        ([[1, 2, 3], [4, 5], [6], [7, 8, 9, 10], [11, 12, 13]], 4, 2, 2, 5, 2),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], 4, 2, 2, 4, 1),  # 16: no padding
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12, 13]], 4, 1, 1, 5, 5),
        ([[1, 2, 3], [4, 5], [6], [7, 8, 9, 10], [11, 12, 13]], 5, 3, 1, 4, 2),
        ([[]], 4, 2, 3, 1, 1),  # A document that is only its BOS (a quota cut).
    ],
)
def test_planned_updates_are_the_ceiling_of_windows_over_rows_per_update(
    tmp_path: Path,
    documents: list[list[int]],
    length: int,
    batch: int,
    accumulation: int,
    windows: int,
    updates: int,
) -> None:
    opened, sha = shard(tmp_path, "only", documents)
    order = pass_order([(sha, len(opened))], 3)
    loader = OnePassLoader(
        [("web", opened, sha)],
        order,
        sequence_length=length,
        batch_size=batch,
        accumulation_steps=accumulation,
    )
    total = sum(len(document) + 1 for document in documents)
    assert loader.total_targets == total
    assert loader.total_windows == windows == -(-total // length)
    assert loader.planned_updates == updates == -(-windows // (batch * accumulation))
    batches = drain(loader)
    assert len(batches) == updates * accumulation
    shapes = {(len(rows), len(row)) for rows, _, _ in batches for row in rows}
    assert shapes == {(batch, length)}
    real = sum(target != IGNORE for _, targets, _ in batches for row in targets for target in row)
    assert real == total
    padded_rows = sum(all(t == IGNORE for t in row) for _, targets, _ in batches for row in targets)
    assert padded_rows == updates * batch * accumulation - windows
    with pytest.raises(RuntimeError, match="exhausted"):
        loader.next_batch()


def expected_digests(
    shards: Shards, order: npt.NDArray[np.void], documents: dict[str, list[list[int]]]
) -> tuple[str, str]:
    """Multiset and order chain from the authored token lists, without the loader."""
    hashes = {}
    for _, opened, sha in shards:
        for index, content in enumerate(documents[opened.path.stem]):
            tokens = [BOS, *content]
            hashes[(sha, index)] = hashlib.sha256(
                struct.pack(f"<{len(tokens)}H", *tokens)
            ).hexdigest()
    chain = CHAIN_START
    for ordinal, index in zip(order["shard"], order["index"], strict=True):
        sha = shards[int(ordinal)][2]
        chain = chain_digest(chain, sha, int(index), hashes[(sha, int(index))])
    return multiset_digest(hashes.values()), chain


def test_ticket_digests_and_counts_by_hand(tmp_path: Path) -> None:
    loader = hand_loader(tmp_path)
    shards = authored(tmp_path / "again")
    documents = {"web": [[1, 2, 3], [4, 5]], "py": [[6], [7, 8, 9, 10], [11, 12, 13]]}
    multiset, chain = expected_digests(shards, manual_order(HAND_ORDER), documents)
    loader.next_batch()
    partial = loader.ticket()
    # Two windows: py[1] (5 tokens) is complete and web[0] has 3 of its 4 tokens read.
    assert partial["complete"] is False
    assert (partial["documents"], partial["targets"], partial["windows"]) == (1, 8, 2)
    assert partial["per_stream"] == {
        "py": {"documents": 1, "tokens": 5},
        "web": {"documents": 0, "tokens": 3},
    }
    drain(loader)
    assert loader.ticket() == {
        "complete": True,
        "documents": 5,
        "targets": 18,
        "windows": 5,
        "updates": 2,
        "multiset_sha256": multiset,
        "order_chain_sha256": chain,
        "order_sha256": order_digest(manual_order(HAND_ORDER)),
        "per_stream": {"py": {"documents": 3, "tokens": 11}, "web": {"documents": 2, "tokens": 7}},
    }


def test_stage_tickets_compose_to_the_mixed_pass(tmp_path: Path) -> None:
    documents = {
        "s1-web-a": [[1, 2], [3, 4, 5], [6]],
        "s1-web-b": [[7, 8, 9, 10, 11], [12]],
        "s2-web": [[13, 14], [15]],
        "s2-py-a": [[16, 17, 18], [19, 20], [21, 22, 23, 24]],
        "s2-py-b": [[25], [26, 27]],
    }
    streams = {
        "s1-web-a": "stage1-web",
        "s1-web-b": "stage1-web",
        "s2-web": "stage2-web",
        "s2-py-a": "stage2-python",
        "s2-py-b": "stage2-python",
    }
    opened = {name: shard(tmp_path, name, content) for name, content in documents.items()}

    def run(names: list[str], seed: int) -> dict[str, Any]:
        shards = [(streams[name], *opened[name]) for name in names]
        order = pass_order([(sha, len(item)) for _, item, sha in shards], seed)
        loader = OnePassLoader(shards, order, sequence_length=4, batch_size=2, accumulation_steps=2)
        drain(loader)
        ticket = loader.ticket()
        assert ticket["complete"]
        assert (ticket["multiset_sha256"], ticket["order_chain_sha256"]) == expected_digests(
            shards, order, documents
        )
        return ticket

    stage1 = run(["s1-web-a", "s1-web-b"], seed=11)
    stage2 = run(["s2-web", "s2-py-a", "s2-py-b"], seed=12)
    mixed = run(list(documents), seed=13)
    assert (
        combine_digests(stage1["multiset_sha256"], stage2["multiset_sha256"])
        == (mixed["multiset_sha256"])
    )
    assert {**stage1["per_stream"], **stage2["per_stream"]} == mixed["per_stream"]
    assert mixed["per_stream"] == {
        "stage1-web": {"documents": 5, "tokens": 17},
        "stage2-python": {"documents": 5, "tokens": 17},
        "stage2-web": {"documents": 2, "tokens": 5},
    }
    assert stage1["documents"] + stage2["documents"] == mixed["documents"] == 12
    assert stage1["targets"] + stage2["targets"] == mixed["targets"] == 39
    # The same documents in another order: same multiset, different order chain.
    other = run(list(documents), seed=14)
    assert other["multiset_sha256"] == mixed["multiset_sha256"]
    assert other["order_chain_sha256"] != mixed["order_chain_sha256"]


def test_resume_from_any_saved_state_matches_the_uninterrupted_pass(tmp_path: Path) -> None:
    shards = authored(tmp_path)
    order = pass_order([(sha, len(opened)) for _, opened, sha in shards], 5)

    def fresh() -> OnePassLoader:
        return OnePassLoader(shards, order, sequence_length=3, batch_size=1, accumulation_steps=2)

    reference = fresh()
    expected = drain(reference)
    offsets = []
    for cut in range(len(expected) + 1):
        partial = fresh()
        for _ in range(cut):
            partial.next_batch()
        state = json.loads(json.dumps(partial.state_dict()))  # Primitives survive JSON.
        offsets.append(state["offset"])
        resumed = OnePassLoader(
            authored(tmp_path / f"reopened-{cut}"),
            order,
            sequence_length=3,
            batch_size=1,
            accumulation_steps=2,
        )
        resumed.load_state_dict(state)
        assert resumed.state_dict() == partial.state_dict()
        assert drain(resumed) == expected[cut:]
        assert resumed.ticket() == reference.ticket()
    assert any(offsets) and offsets[0] == 0  # Some cuts fall inside a document.


@pytest.mark.parametrize(
    "change",
    [
        {"signature": "0" * 64},
        {"rows": 3},
        {"rows": 999},
        {"windows": 1},
        {"cursor": 0},
        {"offset": 1},
        {"sentinel": True},
        {"multiset_sha256": "xyz"},
        {"order_chain_sha256": 7},
        {"rows": True},
        {"extra": 1},
    ],
)
def test_rejected_state_leaves_the_loader_unchanged(tmp_path: Path, change: dict[str, Any]) -> None:
    loader = hand_loader(tmp_path)
    loader.next_batch()
    before = loader.state_dict()
    with pytest.raises(ValueError):
        loader.load_state_dict({**before, **change})
    assert loader.state_dict() == before
    reference = hand_loader(tmp_path / "reference")
    reference.next_batch()
    assert loader.next_batch() == reference.next_batch()
    empty = hand_loader(tmp_path / "empty")
    with pytest.raises(ValueError, match="empty before the first document"):
        empty.load_state_dict({**empty.state_dict(), "multiset_sha256": "1" * 64})


def test_construction_rejects_mismatched_shards_and_orders(tmp_path: Path) -> None:
    shards = authored(tmp_path)
    good = manual_order(HAND_ORDER)
    with pytest.raises(ValueError, match="differ from the supplied SHA-256"):
        OnePassLoader(
            [shards[0], (shards[1][0], shards[1][1], "f" * 64)],
            good,
            sequence_length=4,
            batch_size=1,
            accumulation_steps=1,
        )
    for order in (
        manual_order(HAND_ORDER[:-1]),
        manual_order([(1, 1), (1, 1), (1, 0), (0, 1), (1, 2)]),
        manual_order([(1, 1), (0, 0), (1, 0), (0, 2), (1, 2)]),
        manual_order([(2, 0), (0, 0), (1, 0), (0, 1), (1, 2)]),
        good.astype([("shard", "<u4"), ("index", "<u4")]),
    ):
        with pytest.raises(ValueError, match="order"):
            OnePassLoader(shards, order, sequence_length=4, batch_size=1, accumulation_steps=1)
    with pytest.raises(ValueError, match="distinct"):
        OnePassLoader(
            [shards[0], ("again", shards[0][1], shards[0][2])],
            manual_order([(0, 0), (0, 1), (1, 0), (1, 1)]),
            sequence_length=4,
            batch_size=1,
            accumulation_steps=1,
        )
    with pytest.raises(ValueError, match="positive integer"):
        OnePassLoader(shards, good, sequence_length=0, batch_size=1, accumulation_steps=1)


def test_documents_without_bos_or_with_foreign_tokens_are_rejected_without_advancing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "split.bin"
    write_shard(path, [[1, 2], [300]], vocab_size=512, tokenizer_hash="authored", bos_id=BOS)
    manifest = path.with_suffix(".bin.json")
    metadata = json.loads(manifest.read_text())
    # [256, 1, 2, 256, 300] re-declared as [256], [1, 2], [256, 300] with a 272 vocabulary.
    metadata["documents"] = [[0, 1], [1, 2], [3, 2]]
    metadata["vocab_size"] = VOCAB
    manifest.write_text(json.dumps(metadata))
    opened = TokenShard(path, tokenizer_hash="authored", vocab_size=VOCAB, bos_id=BOS)
    for pairs, message in (
        ([(0, 1), (0, 0), (0, 2)], "does not start with BOS"),
        ([(0, 2), (0, 0), (0, 1)], "outside vocabulary"),
    ):
        loader = OnePassLoader(
            [("web", opened, file_sha256(path))],
            manual_order(pairs),
            sequence_length=8,
            batch_size=1,
            accumulation_steps=1,
        )
        before = loader.state_dict()
        with pytest.raises(ValueError, match=message):
            loader.next_batch()
        assert loader.state_dict() == before


def test_masked_loss_sums_real_targets_and_matches_a_trusted_reference() -> None:
    torch.manual_seed(4)
    logits = (torch.randn(2, 5, 11) * 6).requires_grad_(True)
    targets = torch.randint(0, 11, (2, 5))
    targets[0, 3:] = IGNORE
    targets[1, 0] = IGNORE
    total, count = cross_entropy_sum(logits, targets)
    reference = F.cross_entropy(
        logits.reshape(-1, 11), targets.flatten(), ignore_index=IGNORE, reduction="sum"
    )
    assert int(count) == 7
    torch.testing.assert_close(total, reference)
    torch.testing.assert_close(
        torch.autograd.grad(total, logits)[0], torch.autograd.grad(reference, logits)[0]
    )
    unmasked = torch.randint(0, 11, (2, 5))
    plain_total, plain_count = cross_entropy_sum(logits, unmasked)
    assert int(plain_count) == 10
    torch.testing.assert_close(plain_total / 10, cross_entropy(logits, unmasked))
    with pytest.raises(ValueError, match="shape mismatch"):
        cross_entropy_sum(logits, unmasked[:, :3])


def test_masked_loss_with_every_target_ignored_is_a_differentiable_zero() -> None:
    logits = torch.randn(2, 3, 7, requires_grad=True)
    total, count = cross_entropy_sum(logits, torch.full((2, 3), IGNORE))
    assert float(total.detach()) == 0.0 and int(count) == 0 and total.requires_grad
    total.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
    assert logits.grad is not None and not bool(logits.grad.any())


def test_masked_loss_keeps_one_saved_fp32_copy() -> None:
    from torch.autograd.graph import saved_tensors_hooks

    batch, time, vocabulary = 2, 16, 4096
    logits = (torch.randn(batch, time, vocabulary) * 8).to(torch.bfloat16).requires_grad_(True)
    targets = torch.randint(0, vocabulary, (batch, time))
    targets[:, time // 2 :] = IGNORE
    storages: dict[int, int] = {}

    def pack(value: torch.Tensor) -> torch.Tensor:
        storages[value.untyped_storage().data_ptr()] = value.untyped_storage().nbytes()
        return value

    with saved_tensors_hooks(pack, lambda value: value):
        total, _ = cross_entropy_sum(logits, targets)
    total.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
    full_size = batch * time * vocabulary
    saved_full_tensors = [size for size in storages.values() if size >= full_size]
    assert sum(saved_full_tensors) <= 4 * full_size + full_size * logits.element_size()
    assert len(saved_full_tensors) <= 2


def tiny_config(*, steps: int, accumulation: int, **train: Any) -> RunConfig:
    return RunConfig(
        model=ModelConfig(vocab_size=VOCAB, width=16, layers=1, heads=2, context=16),
        train=TrainConfig(
            steps=steps,
            sequence_length=4,
            batch_size=2,
            accumulation_steps=accumulation,
            **train,
        ),
    )


def make_trainer(config: RunConfig, loader: OnePassLoader, root: Path, name: str) -> Trainer:
    return Trainer(
        config, loader, run_id=name, run_root=root, data_hash="authored", tokenizer_hash="authored"
    )


def step_metrics(root: Path, name: str) -> list[dict[str, Any]]:
    events = [
        json.loads(line)
        for line in (root / name / "events-rank0.jsonl").read_text().split("\n")
        if line
    ]
    return [row["fields"]["metrics"] for row in events if row["event"] == "metrics"]


def test_padded_final_update_matches_a_hand_computed_sum_over_capacity(tmp_path: Path) -> None:
    # One update of 2 rows x 4 microbatches x 4 positions = 32 target slots: two full
    # microbatches, the ragged window plus a padded row, and a fully padded microbatch.
    config = tiny_config(steps=1, accumulation=4, gradient_clip=1e6)
    loader = hand_loader(tmp_path / "data", accumulation=4)
    assert loader.planned_updates == 1
    trainer = make_trainer(config, loader, tmp_path, "padded")
    losses = trainer.train()

    windows = hand_loader(tmp_path / "copy", batch_size=1, accumulation=1)
    rows = [windows.next_batch() for _ in range(windows.total_windows)]
    x = torch.tensor([inputs[0] for inputs, _, _ in rows])
    y = torch.tensor([targets[0] for _, targets, _ in rows])
    torch.manual_seed(config.train.seed)
    model = GPT(config.model)
    summed = F.cross_entropy(
        model(x).reshape(-1, VOCAB), y.flatten(), ignore_index=IGNORE, reduction="sum"
    )
    (summed / 32).backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
    for (name, expected), actual in zip(
        model.named_parameters(), trainer.model.parameters(), strict=True
    ):
        assert expected.grad is not None and actual.grad is not None, name
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-8)
    assert losses == [pytest.approx(float(summed.detach()) / 18, rel=1e-6)]
    assert step_metrics(tmp_path, "padded")[0]["tokens"] == 18
    with pytest.raises(RuntimeError, match="exhausted"):
        trainer.loader.next_batch()


def test_bfloat16_padded_update_tracks_float32(tmp_path: Path) -> None:
    gradients: dict[str, list[torch.Tensor]] = {}
    for precision in ("float32", "bfloat16"):
        config = tiny_config(steps=1, accumulation=4, gradient_clip=1e6, precision=precision)
        loader = hand_loader(tmp_path / precision, accumulation=4)
        trainer = make_trainer(config, loader, tmp_path, precision)
        (loss,) = trainer.train()
        assert math.isfinite(loss) and step_metrics(tmp_path, precision)[0]["tokens"] == 18
        grads = [parameter.grad for parameter in trainer.model.parameters()]
        assert all(grad is not None and grad.dtype == torch.float32 for grad in grads)
        gradients[precision] = [grad.clone() for grad in grads if grad is not None]
    for reference, actual in zip(gradients["float32"], gradients["bfloat16"], strict=True):
        assert float((actual - reference).norm()) <= 0.03 * float(reference.norm())


def test_unpadded_batches_take_the_existing_path_bitwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("the masked loss ran for an unpadded batch")

    monkeypatch.setattr("scratchcoder.train.trainer.cross_entropy_sum", forbidden)
    config = tiny_config(steps=2, accumulation=2)
    # 32 shard tokens = 8 full windows = exactly 2 updates: no IGNORE anywhere.
    documents = [[1, 2, 3], [4, 5, 6, 7, 8], [9], [10, 11, 12, 13, 14, 15], [16, 17], [18]]
    documents += [[19, 20, 21, 22, 23], [24]]

    def loader(name: str) -> OnePassLoader:
        opened, sha = shard(tmp_path / name, "even", documents)
        order = pass_order([(sha, len(opened))], 2)
        return OnePassLoader(
            [("web", opened, sha)], order, sequence_length=4, batch_size=2, accumulation_steps=2
        )

    trainer = make_trainer(config, loader("trained"), tmp_path, "unpadded")
    losses = trainer.train()
    assert [row["tokens"] for row in step_metrics(tmp_path, "unpadded")] == [16, 16]

    # The pre-existing update, written out: mean loss / accumulation per microbatch.
    source = loader("replayed")
    options = config.train
    torch.manual_seed(options.seed)
    model = GPT(config.model)
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


def test_stop_inside_the_masked_update_rolls_back_and_resumes_exactly(tmp_path: Path) -> None:
    config = tiny_config(steps=2, accumulation=2)
    reference = make_trainer(config, hand_loader(tmp_path / "a"), tmp_path, "reference")
    expected = reference.train()
    metrics = step_metrics(tmp_path, "reference")
    assert [row["tokens"] for row in metrics] == [16, 2]  # The final update has 2 targets.
    interrupted = make_trainer(config, hand_loader(tmp_path / "b"), tmp_path, "interrupted")
    calls = 0

    def stop() -> bool:
        nonlocal calls
        calls += 1
        # Four checks per update; the seventh follows the final update's masked microbatch.
        return calls >= 7

    assert interrupted.train(should_stop=stop) == expected[:1]
    assert interrupted.step == 1 and interrupted.loader.state_dict()["rows"] == 4
    interrupted.checkpoint(tmp_path / "partial.pt")
    resumed = make_trainer(config, hand_loader(tmp_path / "c"), tmp_path, "resumed")
    resumed.resume(tmp_path / "partial.pt")
    assert resumed.train() == expected[1:]
    for left, right in zip(reference.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert isinstance(resumed.loader, OnePassLoader) and isinstance(reference.loader, OnePassLoader)
    assert resumed.loader.ticket() == reference.loader.ticket()


def test_masked_updates_reject_z_loss_and_extra_steps(tmp_path: Path) -> None:
    with_z = make_trainer(
        tiny_config(steps=1, accumulation=4, z_loss=1e-4),
        hand_loader(tmp_path / "z", accumulation=4),
        tmp_path,
        "with-z",
    )
    with pytest.raises(ValueError, match="z-loss"):
        with_z.train()
    too_long = make_trainer(
        tiny_config(steps=3, accumulation=2), hand_loader(tmp_path / "long"), tmp_path, "long"
    )
    with pytest.raises(RuntimeError, match="exhausted"):
        too_long.train()
    assert too_long.step == 2
