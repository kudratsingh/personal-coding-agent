"""Cached rank-based encoding and optional process-parallel tokenization."""

from collections.abc import Iterable, Iterator
from functools import lru_cache
from multiprocessing import get_context

from scratchcoder.tokenizer.bpe import CHUNKS, FIRST_MERGE, ByteBPE, merge_pair


class FastEncoder:
    def __init__(self, tokenizer: ByteBPE) -> None:
        self.ranks = {pair: FIRST_MERGE + rank for rank, pair in enumerate(tokenizer.merges)}
        self.chunk = lru_cache(maxsize=65536)(self._encode_chunk)

    def _encode_chunk(self, text: str) -> tuple[int, ...]:
        ids = list(text.encode())
        while len(ids) > 1:
            candidates = {
                (left, right): self.ranks[(left, right)]
                for left, right in zip(ids, ids[1:], strict=False)
                if (left, right) in self.ranks
            }
            if not candidates:
                break
            pair = min(candidates, key=lambda item: candidates[item])
            ids = merge_pair(ids, pair, candidates[pair])
        return tuple(ids)

    def encode(self, text: str) -> list[int]:
        return [token for chunk in CHUNKS.findall(text) for token in self.chunk(chunk)]


_worker_encoder: FastEncoder | None = None


def _initialize(tokenizer: ByteBPE) -> None:
    global _worker_encoder
    _worker_encoder = FastEncoder(tokenizer)


def _encode(text: str) -> list[int]:
    if _worker_encoder is None:
        raise RuntimeError("worker encoder was not initialized")
    return _worker_encoder.encode(text)


def encode_many(
    tokenizer: ByteBPE, texts: Iterable[str], *, workers: int = 1
) -> Iterator[list[int]]:
    """Yield ordered documents with bounded work queues and no special-token injection."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        encoder = FastEncoder(tokenizer)
        yield from map(encoder.encode, texts)
    else:
        with get_context("spawn").Pool(
            workers, initializer=_initialize, initargs=(tokenizer,)
        ) as pool:
            yield from pool.imap(_encode, texts, chunksize=16)
