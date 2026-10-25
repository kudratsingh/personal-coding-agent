"""Reference checks for the one-pass document order and multiset digests."""

import hashlib

import numpy as np
import pytest

from scratchcoder.data.onepass import combine_digests, multiset_digest, order_digest, pass_order


def test_order_is_a_hash_keyed_permutation_recomputable_by_hand() -> None:
    shards = [("a" * 64, 3), ("b" * 64, 2)]
    order = pass_order(shards, seed=17)
    pairs = [(int(o), int(i)) for o, i in zip(order["shard"], order["index"], strict=True)]
    assert sorted(pairs) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

    def key(ordinal: int, index: int) -> tuple[int, int, int]:
        text = f"17|{shards[ordinal][0]}|{index}".encode()
        return int.from_bytes(hashlib.sha256(text).digest()[:8], "big"), ordinal, index

    assert pairs == [p[1:] for p in sorted(key(*pair) for pair in pairs)]
    assert order_digest(order) == order_digest(pass_order(shards, seed=17))
    assert order_digest(order) != order_digest(pass_order(shards, seed=18))
    with pytest.raises(ValueError):
        pass_order([("a" * 64, 1), ("a" * 64, 1)], seed=1)


def test_multiset_digest_ignores_order_and_composes_over_disjoint_sets() -> None:
    digests = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(5)]
    assert multiset_digest(digests) == multiset_digest(reversed(digests))
    assert combine_digests(multiset_digest(digests[:2]), multiset_digest(digests[2:])) == (
        multiset_digest(digests)
    )
    assert multiset_digest([]) == "0" * 64
    assert np.dtype(pass_order([("c" * 64, 1)], 0).dtype).names == ("shard", "index")
