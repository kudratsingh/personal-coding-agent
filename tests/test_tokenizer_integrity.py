"""Tokenizer ownership and format invariants, independent of the BPE trainer."""

import json
import pickle
from pathlib import Path
from typing import cast

import pytest

from scratchcoder.tokenizer.bpe import FIRST_MERGE, ByteBPE
from scratchcoder.tokenizer.bulk import FastEncoder


def construct(merges: object) -> ByteBPE:
    """Exercise runtime callers that do not pass the annotated immutable type."""
    return ByteBPE(cast(tuple[tuple[int, int], ...], merges))


def test_constructor_owns_outer_and_inner_merge_containers(tmp_path: Path) -> None:
    pairs = [[97, 98], [FIRST_MERGE, 99]]
    tokenizer = construct(pairs)
    fast = FastEncoder(tokenizer)
    original = tokenizer.serialized()
    identity = tokenizer.sha256
    expected = [FIRST_MERGE + 1, 32, FIRST_MERGE]
    assert tokenizer.encode("abc ab") == fast.encode("abc ab") == expected

    pairs[0][0] = 98
    pairs[1] = [98, 99]
    pairs.append([1, 2])
    assert tokenizer.merges == ((97, 98), (FIRST_MERGE, 99))
    assert tokenizer.serialized() == original
    assert tokenizer.sha256 == identity
    assert tokenizer.encode("abc ab") == fast.encode("abc ab") == expected
    assert tokenizer.decode(expected) == "abc ab"
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    assert ByteBPE.load(path) == pickle.loads(pickle.dumps(tokenizer)) == tokenizer


def test_tuple_outer_with_mutable_pair_is_also_owned() -> None:
    pair = [97, 98]
    tokenizer = construct((pair,))
    pair[0] = 99
    assert tokenizer.merges == ((97, 98),)
    assert tokenizer.encode("ab") == [FIRST_MERGE]
    assert FastEncoder(tokenizer).encode("ab") == [FIRST_MERGE]


def test_mutable_outer_table_cannot_diverge_from_existing_fast_encoder() -> None:
    pairs = [(97, 98)]
    tokenizer = construct(pairs)
    fast = FastEncoder(tokenizer)
    identity = tokenizer.sha256
    assert fast.encode("ab") == [FIRST_MERGE]  # Populate the chunk cache before mutation.
    pairs[0] = (98, 99)
    assert tokenizer.sha256 == identity
    assert fast.encode("ab") == tokenizer.encode("ab") == [FIRST_MERGE]
    assert fast.encode("bc") == tokenizer.encode("bc") == [98, 99]


def test_format_one_hash_and_list_tuple_equivalence_are_unchanged(tmp_path: Path) -> None:
    tokenizer = ByteBPE(((97, 98), (FIRST_MERGE, 99)))
    # Captured before this ownership fix; serialization bytes are the identity contract.
    assert tokenizer.sha256 == "65e498e2341f240326a2b5fbcd38647d15f0457f78d7d96e318397869b3d5848"
    assert construct([[97, 98], [FIRST_MERGE, 99]]).serialized() == tokenizer.serialized()
    path = tmp_path / "legacy.json"
    path.write_text(tokenizer.serialized() + "\n", encoding="utf-8")
    assert ByteBPE.load(path).sha256 == tokenizer.sha256


@pytest.mark.parametrize(
    "merges",
    [
        None,
        True,
        1,
        "ab",
        b"ab",
        {},
        {"pair": [97, 98]},
        [None],
        [True],
        [1],
        ["ab"],
        [b"ab"],
        [{97: 1, 98: 2}],
        [()],
        [(97,)],
        [(97, 98, 99)],
        [(True, 98)],
        [(97.0, 98)],
        [(-1, 98)],
        [(256, 98)],
        [(FIRST_MERGE, 98)],
        [(97, 98), (97, 98)],
        [(97, 98), (FIRST_MERGE + 1, 99)],
    ],
)
def test_invalid_runtime_merge_tables_raise_value_error(merges: object) -> None:
    with pytest.raises(ValueError, match="merge"):
        construct(merges)


def test_vocabulary_ceiling_applies_to_constructor_and_loader(tmp_path: Path) -> None:
    # Each pair introduces the next ID; test structure only, never expand token bytes.
    pairs = [(0, 0)] + [(FIRST_MERGE + index - 1, 0) for index in range(1, 65536 - FIRST_MERGE)]
    tokenizer = ByteBPE(tuple(pairs))
    assert tokenizer.vocab_size == 65536
    path = tmp_path / "maximum.json"
    tokenizer.save(path)
    assert ByteBPE.load(path).sha256 == tokenizer.sha256
    pairs.append((65535, 0))
    with pytest.raises(ValueError, match="vocabulary"):
        construct(pairs)
    raw = json.loads(tokenizer.serialized())
    raw["merges"].append([65535, 0])
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="vocabulary"):
        ByteBPE.load(path)
