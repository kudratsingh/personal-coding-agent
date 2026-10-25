"""Reference checks for whole-document validation windows and batched validation."""

import hashlib
import json
import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from scratchcoder.common.config import ModelConfig
from scratchcoder.data.shards import TokenShard, write_shard
from scratchcoder.model.gpt import GPT
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.train.validation import (
    ValidationWindow,
    document_windows,
    iter_document_windows,
    selection_sha256,
    validate,
    validate_batched,
)

# One merge: ID 272 is the two raw bytes "hi"; IDs 256-271 are specials with no bytes.
TOKENIZER = ByteBPE(((104, 105),))
# write_shard prepends BOS 256 to each document.
DOCUMENTS = [[272, 32, 97, 98, 99], [], [257, 258], [100, 257, 101, 102]]


class CountingShard(TokenShard):
    reads: int = 0

    def __getitem__(self, index: int) -> list[int]:
        self.reads += 1
        return super().__getitem__(index)


def shard(tmp_path: Path, documents: Sequence[Sequence[int]] = DOCUMENTS) -> CountingShard:
    path = tmp_path / "validation.bin"
    if not path.exists():
        write_shard(
            path, documents, vocab_size=TOKENIZER.vocab_size, tokenizer_hash=TOKENIZER.sha256
        )
    return CountingShard(path, tokenizer_hash=TOKENIZER.sha256, vocab_size=TOKENIZER.vocab_size)


def windows_of(
    source: TokenShard, *, sequence_length: int, target_tokens: int | None = None
) -> list[tuple[tuple[int, ...], int]]:
    return [
        (window.tokens, window.target_bytes)
        for window in document_windows(
            source, TOKENIZER, sequence_length=sequence_length, target_tokens=target_tokens
        )
    ]


def test_hand_computed_windows_and_raw_byte_counts(tmp_path: Path) -> None:
    source = shard(tmp_path)
    # Starts 0, 2, 4 of [256, 272, 32, 97, 98, 99]; the empty document has no target;
    # the all-special window has no bytes and is skipped; 257 is scored with 0 bytes.
    assert windows_of(source, sequence_length=2) == [
        ((256, 272, 32), 3),
        ((32, 97, 98), 2),
        ((98, 99), 1),
        ((256, 100, 257), 1),
        ((257, 101, 102), 2),
    ]
    assert windows_of(source, sequence_length=4) == [
        ((256, 272, 32, 97, 98), 5),
        ((98, 99), 1),
        ((256, 100, 257, 101, 102), 3),
    ]
    # Exactly L + 1 tokens: the would-be one-token window [99] is dropped.
    assert windows_of(source, sequence_length=5)[0] == ((256, 272, 32, 97, 98, 99), 6)
    assert len(windows_of(source, sequence_length=5)) == 2


def test_target_tokens_cut_is_exact_and_truncates_the_last_window(tmp_path: Path) -> None:
    source = shard(tmp_path)
    assert windows_of(source, sequence_length=2, target_tokens=3) == [
        ((256, 272, 32), 3),
        ((32, 97), 1),
    ]
    assert windows_of(source, sequence_length=2, target_tokens=4) == [
        ((256, 272, 32), 3),
        ((32, 97, 98), 2),
    ]
    # Five targets from the first document, the skipped window adds none, then one more.
    assert windows_of(source, sequence_length=2, target_tokens=6)[3:] == [((256, 100), 1)]
    assert len(windows_of(source, sequence_length=2, target_tokens=9)) == 5
    with pytest.raises(ValueError, match="fewer targets"):
        document_windows(source, TOKENIZER, sequence_length=2, target_tokens=10)
    for target in range(1, 10):
        cut = document_windows(source, TOKENIZER, sequence_length=3, target_tokens=target)
        assert sum(len(window.tokens) - 1 for window in cut) == target


def test_every_token_after_bos_is_a_target_once_in_document_order(tmp_path: Path) -> None:
    rng = random.Random(0)
    # Raw bytes and the merge only, so a leading 256 always marks a document start.
    documents = [
        [rng.choice([*range(256), 272]) for _ in range(rng.randrange(0, 40))] for _ in range(30)
    ]
    source = shard(tmp_path, documents)
    for length in (1, 2, 3, 7, 16, 64):
        rebuilt: list[list[int]] = []
        for window in iter_document_windows(
            source, TOKENIZER, sequence_length=length, target_tokens=None
        ):
            assert 2 <= len(window.tokens) <= length + 1
            if window.tokens[0] == 256:
                rebuilt.append(list(window.tokens))
            else:
                assert window.tokens[0] == rebuilt[-1][-1]
                rebuilt[-1].extend(window.tokens[1:])
        assert rebuilt == [[256, *document] for document in documents if document]


def test_windows_are_lazy_deterministic_and_validated(tmp_path: Path) -> None:
    source = shard(tmp_path)
    iterator = iter_document_windows(source, TOKENIZER, sequence_length=2, target_tokens=None)
    assert source.reads == 0
    next(iterator)
    assert source.reads == 1
    source.reads = 0
    document_windows(source, TOKENIZER, sequence_length=2, target_tokens=2)
    assert source.reads == 1
    first = document_windows(shard(tmp_path), TOKENIZER, sequence_length=2, target_tokens=None)
    assert first == document_windows(
        shard(tmp_path), TOKENIZER, sequence_length=2, target_tokens=None
    )
    # Limits are rejected when the stream is created, not at its first window.
    for length, target in ((0, None), (True, None), (2, 0), (2, True), (2, -1)):
        with pytest.raises(ValueError, match="positive"):
            iter_document_windows(source, TOKENIZER, sequence_length=length, target_tokens=target)
    with pytest.raises(ValueError, match="vocabulary"):
        iter_document_windows(source, ByteBPE(), sequence_length=2, target_tokens=None)
    empty = shard(tmp_path / "empty", [[], [257]])
    with pytest.raises(ValueError, match="empty"):
        document_windows(empty, TOKENIZER, sequence_length=2, target_tokens=None)


def test_selection_hash_is_the_recorded_json_identity_computed_incrementally(
    tmp_path: Path,
) -> None:
    windows = document_windows(shard(tmp_path), TOKENIZER, sequence_length=2, target_tokens=None)
    reference = hashlib.sha256(
        json.dumps([asdict(window) for window in windows], sort_keys=True).encode()
    ).hexdigest()
    assert selection_sha256(windows) == reference
    assert selection_sha256(iter(windows)) == reference
    assert selection_sha256(
        iter_document_windows(shard(tmp_path), TOKENIZER, sequence_length=2, target_tokens=None)
    ) == (reference)
    assert selection_sha256(()) == hashlib.sha256(b"[]").hexdigest()
    changed_bytes = (ValidationWindow(windows[0].tokens, 4), *windows[1:])
    changed_token = (ValidationWindow((256, 272, 33), 3), *windows[1:])
    assert len({reference, selection_sha256(changed_bytes), selection_sha256(changed_token)}) == 3
    assert selection_sha256(windows[::-1]) != reference


def tiny_model(seed: int = 0, *, fused: bool = False) -> GPT:
    torch.manual_seed(seed)
    model = GPT(
        ModelConfig(vocab_size=TOKENIZER.vocab_size, width=16, heads=2, layers=2, context=16)
    )
    model.set_fused_attention(fused)
    return model


def random_windows(tmp_path: Path, length: int = 8) -> tuple[ValidationWindow, ...]:
    rng = random.Random(1)
    documents = [
        [rng.randrange(TOKENIZER.vocab_size) for _ in range(rng.randrange(1, 30))]
        for _ in range(12)
    ]
    return document_windows(
        shard(tmp_path, documents), TOKENIZER, sequence_length=length, target_tokens=None
    )


def test_uniform_logits_give_the_hand_computed_loss_with_padding() -> None:
    model = tiny_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    windows = (ValidationWindow((256, 97, 98, 99), 3), ValidationWindow((256, 272), 2))
    metrics = validate_batched(model, windows, batch_size=2)
    assert metrics["loss"] == pytest.approx(math.log(273), rel=1e-6)
    assert metrics["bits_per_byte"] == pytest.approx(4 * math.log2(273) / 5, rel=1e-6)
    assert (metrics["tokens"], metrics["bytes"]) == (4.0, 5.0)


@pytest.mark.parametrize("fused", [False, True])
def test_batched_validation_matches_unbatched_reference(tmp_path: Path, fused: bool) -> None:
    model = tiny_model(fused=fused)
    windows = random_windows(tmp_path)
    assert len({len(window.tokens) for window in windows}) > 3
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    reference = validate(model, windows)
    single = validate_batched(model, windows, batch_size=1)
    for metric in ("loss", "bits_per_byte"):
        assert single[metric] == pytest.approx(reference[metric], rel=1e-6, abs=0)
    assert (single["tokens"], single["bytes"]) == (reference["tokens"], reference["bytes"])
    for batch_size in (2, 3, 16, len(windows) + 5):
        batched = validate_batched(model, windows, batch_size=batch_size)
        for metric in ("loss", "bits_per_byte"):
            assert batched[metric] == pytest.approx(single[metric], rel=1e-5, abs=0)
        assert (batched["tokens"], batched["bytes"]) == (single["tokens"], single["bytes"])
        assert validate_batched(model, iter(windows), batch_size=batch_size) == batched
    assert model.training and all(parameter.grad is None for parameter in model.parameters())
    for key, value in weights.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def test_bfloat16_autocast_stays_close_to_float32(tmp_path: Path) -> None:
    model = tiny_model()
    windows = random_windows(tmp_path)
    exact = validate_batched(model, windows, batch_size=4)
    autocast = validate_batched(model, windows, batch_size=4, autocast_bf16=True)
    # The flag must change the arithmetic; nearly uniform tiny-model logits keep it close.
    assert autocast["loss"] != exact["loss"]
    assert autocast["loss"] == pytest.approx(exact["loss"], rel=1e-3)
    assert (autocast["tokens"], autocast["bytes"]) == (exact["tokens"], exact["bytes"])


def test_on_batch_runs_after_each_lazily_read_batch(tmp_path: Path) -> None:
    windows = random_windows(tmp_path)[:7]
    pulled: list[int] = []
    seen: list[int] = []

    def stream() -> Iterator[ValidationWindow]:
        for window in windows:
            pulled.append(1)
            yield window

    validate_batched(
        tiny_model(), stream(), batch_size=3, on_batch=lambda: seen.append(len(pulled))
    )
    assert seen == [3, 6, 7]
    calls: list[None] = []
    validate_batched(tiny_model(), windows, batch_size=16, on_batch=lambda: calls.append(None))
    assert len(calls) == 1


def test_batched_validation_rejects_bad_inputs_and_restores_mode() -> None:
    model = tiny_model()
    for size in (0, True, 1.0):
        with pytest.raises(ValueError, match="batch size"):
            validate_batched(model, (), batch_size=size)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boolean"):
        validate_batched(model, (), autocast_bf16=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        validate_batched(model, ())
    model.eval()
    with pytest.raises(ValueError, match="context"):
        validate_batched(model, (ValidationWindow(tuple([97] * 18), 17),))
    assert not model.training
    model.train()
    with pytest.raises(ValueError, match="context/vocabulary"):
        validate_batched(model, (ValidationWindow((256, 273), 1),))
    assert model.training
