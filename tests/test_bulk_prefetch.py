"""Bounded producer consumption, not a byte/RSS guarantee, for spawn workers."""

from collections.abc import Iterator
from multiprocessing import active_children
from typing import Any

import pytest

from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.tokenizer.bulk import encode_many


def test_workers_do_not_consume_the_whole_source_when_consumer_pauses() -> None:
    seen: list[int] = []

    def documents() -> Iterator[str]:
        for index in range(1000):
            seen.append(index)
            yield f"document {index} 🐍"

    tokenizer = ByteBPE()
    encoded = encode_many(tokenizer, documents(), workers=2)
    try:
        assert next(encoded) == tokenizer.encode("document 0 🐍")
        assert len(seen) == 32
        for index in range(1, 32):
            assert next(encoded) == tokenizer.encode(f"document {index} 🐍")
        assert len(seen) == 32
        assert next(encoded) == tokenizer.encode("document 32 🐍")
        assert len(seen) == 64
        assert list(encoded) == [tokenizer.encode(f"document {i} 🐍") for i in range(33, 1000)]
    finally:
        encoded.close()


def test_serial_path_remains_one_document_lazy() -> None:
    seen: list[int] = []

    def documents() -> Iterator[str]:
        for index in range(10):
            seen.append(index)
            yield str(index)

    encoded = encode_many(ByteBPE(), documents())
    try:
        assert seen == []
        assert next(encoded) == [48]
        assert seen == [0]
    finally:
        encoded.close()


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("documents", [[], ["one"], [f"row {i}" for i in range(35)]])
def test_short_empty_and_partial_windows_preserve_order(workers: int, documents: list[str]) -> None:
    tokenizer = ByteBPE()
    assert list(encode_many(tokenizer, documents, workers=workers)) == [
        tokenizer.encode(text) for text in documents
    ]


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2", None])
def test_invalid_worker_counts_do_not_consume_inputs(workers: Any) -> None:
    seen: list[str] = []

    def documents() -> Iterator[str]:
        seen.append("consumed")
        yield "one"

    with pytest.raises(ValueError):
        list(encode_many(ByteBPE(), documents(), workers=workers))
    assert seen == []


def test_early_close_terminates_own_workers_without_exhausting_source() -> None:
    previous = {child.pid for child in active_children()}
    seen: list[int] = []

    def documents() -> Iterator[str]:
        for index in range(1000):
            seen.append(index)
            yield str(index)

    encoded = encode_many(ByteBPE(), documents(), workers=2)
    try:
        next(encoded)
        assert {child.pid for child in active_children()} - previous
        assert len(seen) == 32
    finally:
        encoded.close()
    assert not ({child.pid for child in active_children()} - previous)


def test_producer_error_cleans_up_pool() -> None:
    previous = {child.pid for child in active_children()}

    def documents() -> Iterator[str]:
        yield from (str(index) for index in range(35))
        raise RuntimeError("authored producer failure")

    with pytest.raises(RuntimeError, match="authored producer failure"):
        list(encode_many(ByteBPE(), documents(), workers=2))
    assert not ({child.pid for child in active_children()} - previous)


def test_worker_error_cleans_up_pool() -> None:
    previous = {child.pid for child in active_children()}
    malformed: Any = [123]
    with pytest.raises(TypeError):
        list(encode_many(ByteBPE(), malformed, workers=2))
    assert not ({child.pid for child in active_children()} - previous)
