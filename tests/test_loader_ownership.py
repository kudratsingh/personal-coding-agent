"""Loader configuration ownership without copying corpus contents."""

import json

import pytest

from scratchcoder.data.loader import MixedLoader


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("wire_format", ["native", "json"])
@pytest.mark.parametrize("mutation", ["shares", "replace", "clear", "extra"])
def test_caller_mapping_changes_preserve_batches_and_resume(
    rank: int, wire_format: str, mutation: str
) -> None:
    sources = {
        "web": [[256, index, 10, 11] for index in range(4)],
        "code": [[256, index, 20, 21] for index in range(4)],
    }
    shares = {"web": 0.5, "code": 0.5}
    options = dict(sequence_length=3, batch_size=4, seed=7, rank=rank, world_size=2)
    actual = MixedLoader(sources, shares, **options)
    expected = MixedLoader(dict(sources), dict(shares), **options)
    for _ in range(3):
        assert actual.next_batch() == expected.next_batch()
    if mutation == "shares":
        shares.update(web=1.0, code=0.0)
    elif mutation == "replace":
        sources["web"] = [[256, 99, 99, 99]]
    elif mutation == "clear":
        sources.clear()
        shares.clear()
    else:
        sources["extra"] = [[256, 99]]
        shares["extra"] = 1.0
    assert actual.signature == expected.signature
    assert actual.state_dict() == expected.state_dict()
    assert [actual.next_batch() for _ in range(20)] == [expected.next_batch() for _ in range(20)]
    state = actual.state_dict()
    if wire_format == "json":
        state = json.loads(json.dumps(state))
    expected.load_state_dict(state)
    assert [actual.next_batch() for _ in range(20)] == [expected.next_batch() for _ in range(20)]


def test_source_objects_are_not_deep_copied_and_legacy_signature_is_unchanged() -> None:
    sources = {"web": [[256, 1, 2, 3]], "code": [[256, 4, 5, 6]]}
    shares = {"web": 0.5, "code": 0.5}
    loader = MixedLoader(sources, shares, sequence_length=2, batch_size=4, seed=7)
    assert loader.sources is not sources
    assert loader.shares is not shares
    assert loader.sources["web"] is sources["web"]
    assert loader.sources["code"] is sources["code"]
    assert loader.signature == "4d4efe87f5c211ca581df86ed3ee07025c54c6eceb5ae58c38bda7ba68aba71d"
    assert loader.next_batch() == (
        [[256, 4], [5, 6], [256, 1], [256, 4]],
        [[4, 5], [6, 256], [1, 2], [4, 5]],
        ["code", "code", "web", "code"],
    )
