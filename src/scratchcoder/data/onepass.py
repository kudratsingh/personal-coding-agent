"""Deterministic one-pass document order and order-independent token digests.

A rehearsal run reads every document of its streams exactly once, in an order that
anyone can recompute from the shards' content hashes and one seed. Each document's
sort key is the first eight bytes of ``sha256(f"{seed}|{shard_sha256}|{index}")``,
read as a big-endian integer, with (shard, index) breaking ties. Sorting by a hash is
stable across Python and NumPy versions, unlike a library shuffle.

``multiset_digest`` adds the documents' token SHA-256 values modulo 2**256, so two
runs that read the same documents in different orders get the same digest.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt

ORDER_DTYPE = np.dtype([("shard", "<u2"), ("index", "<u4")])
MODULUS = 1 << 256


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
