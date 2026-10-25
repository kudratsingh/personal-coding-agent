"""Deterministic one-pass document order and order-independent token digests.

A rehearsal run reads every document of its streams exactly once, in an order that
anyone can recompute from the shards' content hashes and one seed. Each document's
sort key is the first eight bytes of ``sha256(f"{seed}|{shard_sha256}|{index}")``,
read as a big-endian integer, with (shard, index) breaking ties. Sorting by a hash is
stable across Python and NumPy versions, unlike a library shuffle.

``multiset_digest`` adds the documents' token SHA-256 values modulo 2**256, so two
runs that read the same documents in different orders get the same digest.

``OnePassLoader`` packs the documents in that order into fixed training windows so
that every shard token is a next-token target exactly once. See
``docs/walkthroughs/one-pass-loader.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from scratchcoder.data.shards import TokenShard, file_sha256

ORDER_DTYPE = np.dtype([("shard", "<u2"), ("index", "<u4")])
MODULUS = 1 << 256
# Target value for a padded position; the masked loss skips it.
IGNORE = -100
CHAIN_START = "0" * 64
_HEX64 = re.compile(r"[0-9a-f]{64}")


def pass_order(shards: Sequence[tuple[str, int]], seed: int) -> npt.NDArray[np.void]:
    """The document order for (shard SHA-256, document count) pairs and a seed.

    Returns a structured array of (shard ordinal, document index) in reading order.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("the order seed must be a nonnegative integer")
    if len(shards) >= 1 << 16 or len({sha for sha, _ in shards}) != len(shards):
        raise ValueError("shards must be distinct and fewer than 65,536")
    keys: list[tuple[int, int, int]] = []
    for ordinal, (sha256, documents) in enumerate(shards):
        if type(documents) is not int or documents < 0 or documents >= 1 << 32:
            raise ValueError("document counts must be nonnegative 32-bit integers")
        prefix = f"{seed}|{sha256}|".encode()
        keys.extend(
            (
                int.from_bytes(hashlib.sha256(prefix + str(index).encode()).digest()[:8], "big"),
                ordinal,
                index,
            )
            for index in range(documents)
        )
    keys.sort()
    order = np.empty(len(keys), dtype=ORDER_DTYPE)
    order["shard"] = [ordinal for _, ordinal, _ in keys]
    order["index"] = [index for _, _, index in keys]
    return order


def order_digest(order: npt.NDArray[np.void]) -> str:
    """SHA-256 of the order's little-endian (shard, index) records."""
    return hashlib.sha256(np.ascontiguousarray(order, dtype=ORDER_DTYPE).tobytes()).hexdigest()


def multiset_digest(document_digests: Iterable[str]) -> str:
    """Order-independent digest: the sum of the documents' SHA-256 values mod 2**256."""
    total = 0
    for digest in document_digests:
        total = (total + int(digest, 16)) % MODULUS
    return f"{total:064x}"


def combine_digests(*digests: str) -> str:
    """The multiset digest of the union of disjoint document sets."""
    return multiset_digest(digests)


def document_sha256(tokens: npt.NDArray[np.uint16]) -> str:
    """SHA-256 of one document's uint16 little-endian bytes, its leading BOS included.

    This is the ``tokens_sha256`` field of a build's ``documents.jsonl``.
    """
    return hashlib.sha256(np.ascontiguousarray(tokens, dtype="<u2").tobytes()).hexdigest()


def chain_digest(previous: str, shard_sha256: str, index: int, document: str) -> str:
    """One link of the order chain: SHA-256 of ``"previous|shard|index|document"``.

    The chain starts at ``CHAIN_START``. Folding each document in reading order
    gives a digest that changes if any two documents swap places.
    """
    return hashlib.sha256(f"{previous}|{shard_sha256}|{index}|{document}".encode()).hexdigest()


class OnePassLoader:
    """Every document of the given shards exactly once, packed into fixed windows.

    The token stream is one sentinel BOS followed by the documents' own tokens in
    ``order`` (each document already starts with its BOS). Window ``k`` has inputs
    ``stream[kL:kL+L]`` and targets ``stream[kL+1:kL+L+1]``, so every shard token is
    a target exactly once and the sentinel is only an input. The last window may
    have fewer than ``L`` targets: its remaining inputs are BOS and its remaining
    targets ``IGNORE``. Rows after the last window, up to the end of the final
    update, are entirely padding, so every microbatch has the same shape.

    ``shards`` holds (stream name, opened shard, shard SHA-256) triples in the order
    that was given to ``pass_order``, so the order's ordinals name the same shards.
    """

    def __init__(
        self,
        shards: Sequence[tuple[str, TokenShard, str]],
        order: npt.NDArray[np.void],
        *,
        sequence_length: int,
        batch_size: int,
        accumulation_steps: int,
    ) -> None:
        for label, value in (
            ("sequence_length", sequence_length),
            ("batch_size", batch_size),
            ("accumulation_steps", accumulation_steps),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if not shards or len(shards) >= 1 << 16:
            raise ValueError("a one-pass loader needs between 1 and 65,535 shards")
        for item in shards:
            if (
                not isinstance(item, tuple)
                or len(item) != 3
                or not isinstance(item[0], str)
                or not item[0]
                or not isinstance(item[1], TokenShard)
                or not isinstance(item[2], str)
                or not _HEX64.fullmatch(item[2])
            ):
                raise ValueError("shards must be (stream name, TokenShard, SHA-256) triples")
        opened = [shard for _, shard, _ in shards]
        self._streams = [name for name, _, _ in shards]
        self._hashes = [sha256 for _, _, sha256 in shards]
        if len(set(self._hashes)) != len(self._hashes):
            raise ValueError("shards must be distinct")
        bos_ids = {shard.bos_id for shard in opened}
        vocabularies = {shard.vocab_size for shard in opened}
        bos = next(iter(bos_ids))
        if len(bos_ids) != 1 or bos is None or len(vocabularies) != 1:
            raise ValueError("shards must declare one shared BOS and vocabulary")
        self.bos_id: int = bos
        self.vocab_size: int = next(iter(vocabularies))
        for shard, sha256 in zip(opened, self._hashes, strict=True):
            if file_sha256(shard.path) != sha256:
                raise ValueError(f"{shard.path.name}: bytes differ from the supplied SHA-256")
        self.length = sequence_length
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps

        # Per-document data lives in NumPy arrays (26 bytes a document), not tuples.
        counts = np.array([len(shard) for shard in opened], dtype=np.int64)
        self._base = np.zeros(len(opened), dtype=np.int64)
        self._base[1:] = np.cumsum(counts)[:-1]
        table = np.concatenate(
            [np.asarray(shard.offsets, dtype=np.int64).reshape(-1, 2) for shard in opened]
        )
        self._starts = np.ascontiguousarray(table[:, 0])
        self._lengths = np.ascontiguousarray(table[:, 1])
        del table
        self._documents = int(counts.sum())
        if (
            not isinstance(order, np.ndarray)
            or order.dtype != ORDER_DTYPE
            or order.ndim != 1
            or order.size != self._documents
        ):
            raise ValueError("the order must be a pass_order array over these shards")
        ordinals = order["shard"].astype(np.int64)
        indices = order["index"].astype(np.int64)
        if self._documents and int(ordinals.max()) >= len(opened):
            raise ValueError("the order names a shard outside the shard list")
        if (indices >= counts[ordinals]).any():
            raise ValueError("the order names a document outside its shard")
        self._ids = self._base[ordinals] + indices
        seen = np.zeros(self._documents, dtype=bool)
        seen[self._ids] = True
        if not seen.all():
            raise ValueError("the order must list every document exactly once")
        del ordinals, indices, seen
        self._ordinals = np.array(order["shard"], dtype=np.uint16)
        self.order_sha256 = order_digest(order)
        self._tokens: list[npt.NDArray[np.uint16]] = [
            np.memmap(shard.path, dtype="<u2", mode="r")
            if shard.path.stat().st_size
            else np.zeros(0, dtype="<u2")
            for shard in opened
        ]

        self.total_targets = int(self._lengths.sum())
        if self.total_targets == 0:
            raise ValueError("a one-pass loader needs at least one document")
        self.total_windows = -(-self.total_targets // sequence_length)
        self.rows_per_update = batch_size * accumulation_steps
        self.planned_updates = -(-self.total_windows // self.rows_per_update)
        self.planned_rows = self.planned_updates * self.rows_per_update
        signature = {
            "format": "onepass-1",
            "length": sequence_length,
            "batch": batch_size,
            "accumulation": accumulation_steps,
            "bos": self.bos_id,
            "vocab": self.vocab_size,
            "shards": [
                [name, sha256, len(shard), shard.identity]
                for name, shard, sha256 in zip(self._streams, opened, self._hashes, strict=True)
            ],
            "order": self.order_sha256,
        }
        self.signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        # Reading position: (cursor, offset) is the next target token; carry is the
        # token before it, which becomes the next window's first input.
        self._cursor = 0
        self._offset = 0
        self._carry = self.bos_id
        self._windows = 0
        self._rows = 0
        self._multiset = 0
        self._chain = CHAIN_START

    @property
    def emitted_windows(self) -> int:
        return self._windows

    @property
    def emitted_rows(self) -> int:
        return self._rows

    @property
    def exhausted(self) -> bool:
        """True once every planned update's microbatches have been emitted."""
        return self._rows == self.planned_rows

    def _document(self, position: int) -> tuple[int, int, npt.NDArray[np.uint16]]:
        """(shard ordinal, index in shard, tokens) of the document at ``position``."""
        ordinal = int(self._ordinals[position])
        identity = int(self._ids[position])
        start = int(self._starts[identity])
        tokens = self._tokens[ordinal][start : start + int(self._lengths[identity])]
        return ordinal, identity - int(self._base[ordinal]), tokens

    def _fill(self, row: npt.NDArray[np.int64]) -> int:
        """Copy up to ``L`` next targets into ``row[1:]``; return how many were copied."""
        filled = 0
        while filled < self.length and self._cursor < self._documents:
            ordinal, index, tokens = self._document(self._cursor)
            if self._offset == 0 and int(tokens[0]) != self.bos_id:
                raise ValueError(f"document {index} of shard {ordinal} does not start with BOS")
            take = min(self.length - filled, tokens.size - self._offset)
            chunk = tokens[self._offset : self._offset + take]
            if int(chunk.max()) >= self.vocab_size:
                raise ValueError(f"document {index} of shard {ordinal}: token outside vocabulary")
            row[1 + filled : 1 + filled + take] = chunk
            filled += take
            self._offset += take
            if self._offset == tokens.size:
                digest = document_sha256(tokens)
                self._multiset = (self._multiset + int(digest, 16)) % MODULUS
                self._chain = chain_digest(self._chain, self._hashes[ordinal], index, digest)
                self._cursor += 1
                self._offset = 0
        return filled

    def next_batch(self) -> tuple[list[list[int]], list[list[int]], list[str]]:
        """One ``batch_size x L`` microbatch; padded rows have the source ``""``."""
        if self.exhausted:
            raise RuntimeError(
                f"one-pass loader exhausted: all {self.planned_updates} planned updates "
                f"({self.planned_rows} rows) were already emitted"
            )
        saved = self._position()
        window = np.full((self.batch_size, self.length + 1), self.bos_id, dtype=np.int64)
        counts = np.zeros(self.batch_size, dtype=np.int64)
        sources: list[str] = []
        try:
            for row in range(self.batch_size):
                if self._windows == self.total_windows:
                    sources.append("")
                    continue
                # A row is credited to the stream of its first target.
                sources.append(self._streams[int(self._ordinals[self._cursor])])
                window[row, 0] = self._carry
                counts[row] = self._fill(window[row])
                self._carry = int(window[row, counts[row]])
                self._windows += 1
        except BaseException:
            # A rejected document leaves the loader where this microbatch began.
            self._restore(saved)
            raise
        self._rows += self.batch_size
        padded = np.arange(self.length) >= counts[:, None]
        inputs = np.where(padded, self.bos_id, window[:, :-1])
        targets = np.where(padded, IGNORE, window[:, 1:])
        return inputs.tolist(), targets.tolist(), sources

    def _position(self) -> tuple[int, int, int, int, int, int, str]:
        return (
            self._cursor,
            self._offset,
            self._carry,
            self._windows,
            self._rows,
            self._multiset,
            self._chain,
        )

    def _restore(self, saved: tuple[int, int, int, int, int, int, str]) -> None:
        (
            self._cursor,
            self._offset,
            self._carry,
            self._windows,
            self._rows,
            self._multiset,
            self._chain,
        ) = saved

    def state_dict(self) -> dict[str, Any]:
        """Small primitives only; the trainer copies this before every update."""
        return {
            "signature": self.signature,
            "rows": self._rows,
            "windows": self._windows,
            "cursor": self._cursor,
            "offset": self._offset,
            "sentinel": self._windows == 0,
            "multiset_sha256": f"{self._multiset:064x}",
            "order_chain_sha256": self._chain,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        fields = {
            "signature",
            "rows",
            "windows",
            "cursor",
            "offset",
            "sentinel",
            "multiset_sha256",
            "order_chain_sha256",
        }
        if not isinstance(state, dict) or set(state) != fields:
            raise ValueError("invalid one-pass loader state fields")
        if state["signature"] != self.signature:
            raise ValueError("loader configuration differs from saved state")
        rows, windows = state["rows"], state["windows"]
        cursor, offset = state["cursor"], state["offset"]
        if any(type(value) is not int or value < 0 for value in (rows, windows, cursor, offset)):
            raise ValueError("one-pass loader counters must be nonnegative integers")
        if (
            rows > self.planned_rows
            or rows % self.batch_size
            or windows != min(rows, self.total_windows)
        ):
            raise ValueError("one-pass loader rows and windows are inconsistent")
        if (cursor, offset) != self._locate(min(windows * self.length, self.total_targets)):
            raise ValueError("one-pass loader position disagrees with its window count")
        if type(state["sentinel"]) is not bool or state["sentinel"] != (windows == 0):
            raise ValueError("one-pass loader sentinel flag disagrees with its window count")
        multiset, chain = state["multiset_sha256"], state["order_chain_sha256"]
        if not all(
            isinstance(value, str) and _HEX64.fullmatch(value) for value in (multiset, chain)
        ):
            raise ValueError("one-pass loader digests must be SHA-256 hex strings")
        if cursor == 0 and (multiset != "0" * 64 or chain != CHAIN_START):
            raise ValueError("one-pass loader digests must be empty before the first document")
        carry = self._token_before(cursor, offset)
        # Every check passed; only now does the live position change.
        self._restore((cursor, offset, carry, windows, rows, int(multiset, 16), chain))

    def _locate(self, consumed: int) -> tuple[int, int]:
        """(cursor, offset) of the next target after the first ``consumed`` targets."""
        if consumed == self.total_targets:
            return self._documents, 0
        ends = np.cumsum(self._lengths[self._ids])
        cursor = int(np.searchsorted(ends, consumed, side="right"))
        return cursor, consumed - (int(ends[cursor - 1]) if cursor else 0)

    def _token_before(self, cursor: int, offset: int) -> int:
        if cursor == 0 and offset == 0:
            return self.bos_id  # The sentinel.
        if offset:
            return int(self._document(cursor)[2][offset - 1])
        return int(self._document(cursor - 1)[2][-1])

    def ticket(self) -> dict[str, Any]:
        """What has been read so far; ``complete`` is true only once exhausted.

        ``documents`` and both digests cover the documents whose every token has
        been a target; ``targets`` and per-stream ``tokens`` also count the tokens
        of a partly read document.
        """
        ordinals = self._ordinals[: self._cursor].astype(np.intp)
        documents = np.bincount(ordinals, minlength=len(self._streams))
        tokens = np.zeros(len(self._streams), dtype=np.int64)
        np.add.at(tokens, ordinals, self._lengths[self._ids[: self._cursor]])
        if self._offset:
            tokens[int(self._ordinals[self._cursor])] += self._offset
        per_stream = {name: {"documents": 0, "tokens": 0} for name in sorted(set(self._streams))}
        for ordinal, name in enumerate(self._streams):
            per_stream[name]["documents"] += int(documents[ordinal])
            per_stream[name]["tokens"] += int(tokens[ordinal])
        targets = min(self._windows * self.length, self.total_targets)
        return {
            "complete": self.exhausted,
            "documents": self._cursor,
            "targets": targets,
            "windows": self._windows,
            "updates": self._rows // self.rows_per_update,
            "multiset_sha256": f"{self._multiset:064x}",
            "order_chain_sha256": self._chain,
            "order_sha256": self.order_sha256,
            "per_stream": per_stream,
        }
