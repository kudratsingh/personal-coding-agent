import json
from collections import Counter
from pathlib import Path

import pytest

from scratchcoder.data.loader import MixedLoader
from scratchcoder.data.shards import TokenShard, write_shard


def test_shard_roundtrip_and_integrity(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    write_shard(path, [[1, 2], [3, 4]], vocab_size=300, tokenizer_hash="tokenizer")
    shard = TokenShard(path, tokenizer_hash="tokenizer")
    assert len(shard) == 2
    assert shard[0] == [256, 1, 2]
    assert shard[1] == [256, 3, 4]
    with pytest.raises(FileExistsError):
        write_shard(path, [], vocab_size=300, tokenizer_hash="tokenizer")
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        TokenShard(path, tokenizer_hash="tokenizer")


def test_loader_resume_roundtrip_and_source_shares() -> None:
    sources = {"web": [[256, 10] * 20], "python": [[256, 20] * 20]}
    options = dict(sequence_length=8, batch_size=16, seed=17)
    first = MixedLoader(sources, {"web": 0.1, "python": 0.9}, **options)
    for _ in range(7):
        first.next_batch()
    state = json.loads(json.dumps(first.state_dict()))
    second = MixedLoader(sources, {"web": 0.1, "python": 0.9}, **options)
    second.load_state_dict(state)
    assert [first.next_batch() for _ in range(5)] == [second.next_batch() for _ in range(5)]
    counts: Counter[str] = Counter()
    for _ in range(1000):
        counts.update(first.next_batch()[2])
    assert abs(counts["python"] / counts.total() - 0.9) < 0.01


def test_ranks_partition_documents_without_overlap() -> None:
    sources = {"web": [[i] * 20 for i in range(10)]}
    loaders = [
        MixedLoader(
            sources, {"web": 1.0}, sequence_length=4, batch_size=2, seed=2, rank=rank, world_size=2
        )
        for rank in range(2)
    ]
    assert set(loaders[0].indices["web"]).isdisjoint(loaders[1].indices["web"])
    assert set(loaders[0].indices["web"]) | set(loaders[1].indices["web"]) == set(range(10))


def test_loader_rejects_incompatible_resume() -> None:
    source = {"web": [[256, 1, 2]]}
    first = MixedLoader(source, {"web": 1.0}, sequence_length=2, batch_size=1, seed=1)
    second = MixedLoader(source, {"web": 1.0}, sequence_length=3, batch_size=1, seed=1)
    with pytest.raises(ValueError):
        second.load_state_dict(first.state_dict())
    changed = MixedLoader(
        {"web": [[256, 9, 9]]}, {"web": 1.0}, sequence_length=2, batch_size=1, seed=1
    )
    with pytest.raises(ValueError):
        changed.load_state_dict(first.state_dict())
