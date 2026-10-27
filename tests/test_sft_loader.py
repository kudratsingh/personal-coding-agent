"""Reference checks for SFT rows and the padded, seeded, resumable SFT batch source."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data.chat import BOS, Message, render_conversation
from scratchcoder.data.onepass import IGNORE
from scratchcoder.data.sft import (
    SFTLoader,
    SFTRow,
    loss_spans,
    loss_tokens,
    read_rows,
    row_from_json,
    row_json,
    spans_mask,
    write_rows,
)
from scratchcoder.tokenizer.bpe import ByteBPE

F, T = False, True


def row(name: str, ids: list[int], mask: list[bool], source: str = "code") -> SFTRow:
    return SFTRow(name, source, tuple(ids), tuple(mask))


def authored() -> list[SFTRow]:
    return [
        row("a", [256, 1, 2, 3], [F, F, T, T]),
        row("b", [256, 4, 5], [F, F, T], "language"),
        row("c", [256, 6, 7, 8, 9], [F, T, F, T, T]),
        row("d", [256, 10, 11, 12, 13, 14], [F, F, F, T, T, T]),  # 5 pairs > L = 4
        row("e", [256, 15, 16], [F, F, F]),  # No assistant targets.
    ]


def by_hand_order(names: list[str], seed: int, epoch: int = 0) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        digest = hashlib.sha256(f"{seed}|{epoch}|{name}".encode()).digest()
        return int.from_bytes(digest[:8], "big"), name

    return sorted(names, key=key)


def loader(rows: list[SFTRow] | None = None, **options: Any) -> SFTLoader:
    settings: dict[str, Any] = {"sequence_length": 4, "batch_size": 2, "accumulation_steps": 1}
    settings.update(options)
    return SFTLoader(authored() if rows is None else rows, **settings)


def drain(source: SFTLoader) -> list[tuple[list[list[int]], list[list[int]], list[str]]]:
    batches = []
    while not source.exhausted:
        batches.append(source.next_batch())
    return batches


def test_rows_pad_to_length_with_ignore_outside_the_mask_by_hand() -> None:
    source = loader()
    assert source.dropped == {"no_assistant_targets": 1, "over_length": 1}
    assert source.planned_updates == 2 and source.planned_rows == 4
    expected = {
        "a": ([256, 1, 2, BOS], [IGNORE, 2, 3, IGNORE]),
        "b": ([256, 4, BOS, BOS], [IGNORE, 5, IGNORE, IGNORE]),
        "c": ([256, 6, 7, 8], [6, IGNORE, 8, 9]),
    }
    order = by_hand_order(["a", "b", "c"], seed=0)
    batches = drain(source)
    inputs = [line for batch in batches for line in batch[0]]
    targets = [line for batch in batches for line in batch[1]]
    sources = [name for batch in batches for name in batch[2]]
    for position, name in enumerate(order):
        assert (inputs[position], targets[position]) == expected[name]
    assert inputs[3] == [BOS] * 4 and targets[3] == [IGNORE] * 4 and sources[3] == ""
    assert sources[:3] == ["language" if name == "b" else "code" for name in order]
    assert source.total_targets == 2 + 1 + 3
    with pytest.raises(RuntimeError, match="exhausted"):
        source.next_batch()


def test_loss_tokens_are_the_targets_the_loader_emits_that_are_not_ignore() -> None:
    # The count the SFT data build caps and records is the trainer's real-target count.
    kept = [item for item in authored() if item.id in {"a", "b", "c"}]
    for item in kept:
        _, targets, _ = loader([item], batch_size=1).next_batch()
        assert sum(target != IGNORE for target in targets[0]) == loss_tokens(item)
        assert loss_tokens(item) == sum(item.mask)  # The first token is never a target.
    assert [loss_tokens(item) for item in authored()] == [2, 1, 3, 3, 0]
    assert loader().total_targets == sum(loss_tokens(item) for item in kept)


def test_order_is_the_seeded_hash_order_and_changes_with_seed_and_pass() -> None:
    names = [f"row-{index}" for index in range(12)]
    rows = [row(name, [256, 1, 2], [F, F, T]) for name in names]

    def read(source: SFTLoader) -> list[int]:
        return [len(batch[0]) for batch in drain(source)]

    for seed in (0, 1):
        source = loader(rows, batch_size=1, seed=seed, passes=2, sequence_length=2)
        emitted = [item.id for item in source.reading_order()]
        assert emitted == by_hand_order(names, seed, 0) + by_hand_order(names, seed, 1)
        assert read(source) == [1] * 24
    assert by_hand_order(names, 0) != by_hand_order(names, 1)
    assert loader(rows, seed=0).order_sha256 != loader(rows, seed=1).order_sha256


def test_resume_from_every_microbatch_boundary_matches_the_uninterrupted_pass() -> None:
    rows = [row(f"r{index}", [256, index + 1, 2, 3], [F, T, F, T]) for index in range(7)]
    options = {"batch_size": 2, "accumulation_steps": 2}
    reference = loader(rows, **options)
    expected = drain(reference)
    assert len(expected) == 4  # 7 rows, 4 per update: 2 updates, the last one padded.
    for cut in range(len(expected) + 1):
        first = loader(rows, **options)
        consumed = [first.next_batch() for _ in range(cut)]
        state = json.loads(json.dumps(first.state_dict()))
        second = loader(rows, **options)
        second.load_state_dict(state)
        assert consumed + drain(second) == expected
        assert second.ticket() == reference.ticket()


@pytest.mark.parametrize(
    "change",
    [{"rows": 1}, {"rows": 10}, {"rows": -2}, {"rows": True}, {"signature": "0" * 64}, {"x": 1}],
)
def test_rejected_state_leaves_the_loader_unchanged(change: dict[str, Any]) -> None:
    source = loader()
    source.next_batch()
    before = source.state_dict()
    state = {**before, **change}
    with pytest.raises(ValueError):
        source.load_state_dict(state)
    assert source.state_dict() == before


def test_signature_binds_rows_shape_seed_and_passes() -> None:
    base = loader().signature
    assert loader(seed=1).signature != base
    assert loader(passes=2).signature != base
    assert loader(batch_size=1).signature != base
    changed = authored()
    changed[0] = row("a", [256, 1, 2, 9], [F, F, T, T])
    assert loader(changed).signature != base
    assert loader().signature == base


def test_ticket_counts_rows_targets_and_sources() -> None:
    source = loader()
    source.next_batch()
    ticket = source.ticket()
    first_two = by_hand_order(["a", "b", "c"], seed=0)[:2]
    targets = {"a": 2, "b": 1, "c": 3}
    assert ticket["rows"] == 2 and ticket["targets"] == sum(targets[n] for n in first_two)
    assert ticket["updates"] == 1 and not ticket["complete"]
    drain(source)
    assert source.ticket()["complete"] and source.ticket()["targets"] == 6
    assert source.ticket()["per_source"] == {
        "code": {"rows": 2, "targets": 5},
        "language": {"rows": 1, "targets": 1},
    }


def test_construction_validation() -> None:
    with pytest.raises(ValueError, match="unique"):
        loader([row("a", [256, 1], [F, T]), row("a", [256, 2], [F, T])])
    with pytest.raises(ValueError, match="at least one row"):
        loader([row("e", [256, 15, 16], [F, F, F])])
    with pytest.raises(ValueError, match="positive"):
        loader(passes=0)
    with pytest.raises(ValueError):
        SFTRow("", "code", (256, 1), (False, True))
    with pytest.raises(ValueError):
        SFTRow("x", "code", (256, -1), (False, True))


def test_a_conversation_of_l_plus_one_tokens_fits_exactly() -> None:
    fitting = row("fit", [256, 1, 2, 3, 4], [F, F, T, T, T])
    source = loader([fitting])
    inputs, targets, _ = source.next_batch()
    assert inputs[0] == [256, 1, 2, 3] and targets[0] == [IGNORE, 2, 3, 4]
    assert source.dropped == {}


def test_loss_spans_round_trip_and_files(tmp_path: Path) -> None:
    mask = (F, T, T, F, F, T, F, T)
    assert loss_spans(mask) == [[1, 3], [5, 6], [7, 8]]
    assert spans_mask([[1, 3], [5, 6], [7, 8]], 8) == mask
    bad_spans: list[Any] = [[[2, 1]], [[0, 9]], [[3, 4], [2, 5]], [[1]], [["0", 1]]]
    for bad in bad_spans:
        with pytest.raises(ValueError):
            spans_mask(bad, 8)
    rendered = render_conversation(ByteBPE(), [Message("user", "hi"), Message("assistant", "yo")])
    item = SFTRow.of("conv", "language", rendered)
    assert row_from_json(json.loads(json.dumps(row_json(item)))) == item
    write_rows(tmp_path / "rows.jsonl", [item, *authored()])
    assert list(read_rows(tmp_path / "rows.jsonl")) == [item, *authored()]
    with pytest.raises(FileExistsError):
        write_rows(tmp_path / "rows.jsonl", [item])
    (tmp_path / "dupe.jsonl").write_text(2 * (json.dumps(row_json(item)) + "\n"))
    with pytest.raises(ValueError, match="duplicate"):
        list(read_rows(tmp_path / "dupe.jsonl"))
    assert (
        item.sha256()
        == hashlib.sha256(
            json.dumps(row_json(item), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
