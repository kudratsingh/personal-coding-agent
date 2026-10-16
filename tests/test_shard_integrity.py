import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from scratchcoder.data.shards import TokenShard, write_shard


def test_shard_bytes_match_independent_little_endian_reference(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    write_shard(path, [[1, 2], [3]], vocab_size=300, tokenizer_hash="fixture")
    # [256, 1, 2, 256, 3], explicitly rendered without the writer's struct helper.
    assert path.read_bytes() == bytes([0, 1, 1, 0, 2, 0, 0, 1, 3, 0])
    shard = TokenShard(path, tokenizer_hash="fixture")
    assert [shard[index] for index in range(len(shard))] == [[256, 1, 2], [256, 3]]


def test_writer_rejects_floating_vocabulary_before_creating_files(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    with pytest.raises(ValueError):
        write_shard(path, [[1]], vocab_size=300.0, tokenizer_hash="fixture")  # type: ignore[arg-type]
    assert not path.exists()


def test_iterator_file_exists_error_does_not_leave_partial_shard(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"

    def documents() -> Iterator[Sequence[int]]:
        yield [1, 2]
        raise FileExistsError("authored iterator failure")

    with pytest.raises(FileExistsError, match="iterator"):
        write_shard(path, documents(), vocab_size=300, tokenizer_hash="fixture")
    assert not path.exists()
    assert not path.with_suffix(".bin.json").exists()


def test_exclusive_open_error_preserves_existing_binary(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"existing-owner-fixture")
    with pytest.raises(FileExistsError):
        write_shard(path, [[1]], vocab_size=300, tokenizer_hash="fixture")
    assert path.read_bytes() == b"existing-owner-fixture"


def test_reader_rejects_hash_consistent_out_of_vocabulary_ids(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    manifest = write_shard(path, [[60000]], vocab_size=65536, tokenizer_hash="fixture")
    metadata = json.loads(manifest.read_text())
    metadata["vocab_size"] = 300
    manifest.write_text(json.dumps(metadata))
    shard = TokenShard(path, tokenizer_hash="fixture")
    with pytest.raises(ValueError, match="vocabulary"):
        shard[0]


@pytest.mark.parametrize(
    "change",
    [
        {"version": True},
        {"vocab_size": 300.0},
        {"vocab_size": 1},
        {"vocab_size": 65537},
        {"bos_id": True},
        {"bos_id": None},
        {"bos_id": 300},
        {"documents": [[0.0, 2]]},
        {"documents": [[False, 2]]},
        {"documents": [[0]]},
        {"documents": "invalid"},
        {"sha256": "invalid"},
        {"unreviewed": True},
    ],
)
def test_reader_rejects_malformed_semantic_metadata(
    tmp_path: Path, change: dict[str, object]
) -> None:
    path = tmp_path / "fixture.bin"
    manifest = write_shard(path, [[1]], vocab_size=300, tokenizer_hash="fixture")
    metadata = json.loads(manifest.read_text())
    metadata.update(change)
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        TokenShard(path, tokenizer_hash="fixture")


def test_custom_bos_and_expected_tokenizer_properties(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    manifest = write_shard(path, [[1, 2]], vocab_size=300, bos_id=257, tokenizer_hash="fixture")
    assert json.loads(manifest.read_text())["bos_id"] == 257
    assert TokenShard(path, tokenizer_hash="fixture")[0] == [257, 1, 2]
    assert TokenShard(path, tokenizer_hash="fixture", vocab_size=300, bos_id=257)[0] == [257, 1, 2]
    for expected in ({"vocab_size": 301}, {"bos_id": 256}, {"bos_id": True}):
        with pytest.raises(ValueError):
            TokenShard(path, tokenizer_hash="fixture", **expected)
    metadata = json.loads(manifest.read_text())
    metadata["bos_id"] = 256
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="BOS"):
        TokenShard(path, tokenizer_hash="fixture")[0]


def test_legacy_header_identity_and_explicit_bos_context_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    manifest = write_shard(path, [[1, 2]], vocab_size=300, bos_id=257, tokenizer_hash="fixture")
    metadata = json.loads(manifest.read_text())
    del metadata["bos_id"]
    manifest.write_text(json.dumps(metadata))
    old_identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    shard = TokenShard(path, tokenizer_hash="fixture")
    assert shard.identity == old_identity and shard.bos_id is None
    assert shard[0] == [257, 1, 2]
    assert TokenShard(path, tokenizer_hash="fixture", bos_id=257)[0] == [257, 1, 2]
    with pytest.raises(ValueError, match="BOS"):
        TokenShard(path, tokenizer_hash="fixture", bos_id=256)[0]
