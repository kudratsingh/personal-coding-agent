import json
import random
from copy import deepcopy
from typing import Any

import pytest

from scratchcoder.data.loader import MixedLoader


def loader(seed: int = 7) -> MixedLoader:
    return MixedLoader(
        {"python": [[256, 1, 2] * 40], "web": [[256, 3, 4] * 40]},
        {"python": 0.7, "web": 0.3},
        sequence_length=4,
        batch_size=2,
        seed=seed,
    )


@pytest.mark.parametrize("mutation", ["negative_position", "float_token", "partial_mutation"])
def test_malformed_state_rejected_without_mutation(mutation: str) -> None:
    target = loader()
    target.next_batch()
    before = target.state_dict()
    invalid = deepcopy(before)
    if mutation == "negative_position":
        invalid["positions"]["web"] = -1
    elif mutation == "float_token":
        invalid["buffers"]["python"] = [1.5]
    else:
        other = loader(seed=19)
        invalid["rng"] = other.state_dict()["rng"]
        invalid["buffers"] = None
    with pytest.raises(ValueError):
        target.load_state_dict(invalid)
    assert target.state_dict() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("positions", {}),
        ("positions", {"python": 0, "web": 0, "unknown": 0}),
        ("positions", {"python": 0, "web": True}),
        ("positions", {"python": 0, "web": 1.5}),
        ("positions", []),
        ("buffers", {}),
        ("buffers", {"python": [], "web": [], "unknown": []}),
        ("buffers", {"python": [], "web": [-1]}),
        ("buffers", {"python": [], "web": [True]}),
        ("buffers", {"python": [], "web": "12"}),
        ("buffers", {"python": [], "web": (1, 2)}),
        ("rng", None),
        ("rng", []),
        ("rng", [True, [], None]),
        ("rng", [2, [], None]),
        ("rng", [3, [], None]),
        ("rng", [3, [0] * 624 + [625], None]),
        ("rng", [3, [-1] * 624 + [0], None]),
        ("rng", [3, [2**32] * 624 + [0], None]),
        ("rng", [3, [True] * 624 + [0], None]),
        ("rng", [3, [0.5] * 624 + [0], None]),
        ("rng", [3, [0] * 624 + [0], float("nan")]),
        ("rng", [3, [0] * 624 + [0], float("inf")]),
        ("rng", [3, [0] * 624 + [0], "cache"]),
    ],
)
def test_invalid_fields_preserve_state_and_continuation(field: str, value: Any) -> None:
    target, control = loader(), loader()
    target.next_batch()
    control.next_batch()
    before = target.state_dict()
    invalid = deepcopy(before)
    invalid[field] = value
    with pytest.raises(ValueError):
        target.load_state_dict(invalid)
    assert target.state_dict() == before
    assert [target.next_batch() for _ in range(5)] == [control.next_batch() for _ in range(5)]


@pytest.mark.parametrize("invalid", [None, [], {}, {"extra": 1}])
def test_invalid_top_level_state(invalid: Any) -> None:
    target = loader()
    before = target.state_dict()
    with pytest.raises(ValueError, match="state fields"):
        target.load_state_dict(invalid)
    assert target.state_dict() == before


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_exact_state_fields(change: str) -> None:
    target = loader()
    before = target.state_dict()
    invalid = deepcopy(before)
    if change == "missing":
        del invalid["rng"]
    else:
        invalid["unexpected"] = 1
    with pytest.raises(ValueError):
        target.load_state_dict(invalid)
    assert target.state_dict() == before


@pytest.mark.parametrize("json_roundtrip", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_valid_native_and_json_resume_preserves_reference_batches(
    json_roundtrip: bool, rank: int
) -> None:
    sources = {"python": [[256, i] * 80 for i in range(4)], "web": [[256, i] for i in range(4)]}
    options = dict(sequence_length=4, batch_size=3, seed=7, rank=rank, world_size=2)
    first = MixedLoader(sources, {"python": 0.5, "web": 0.5}, **options)
    for _ in range(30):
        first.next_batch()
    state = first.state_dict()
    assert max(state["positions"].values()) > 2  # Several epochs are valid.
    assert len(state["buffers"]["python"]) > first.length  # Whole-document tail.
    if json_roundtrip:
        state = json.loads(json.dumps(state))
    second = MixedLoader(sources, {"python": 0.5, "web": 0.5}, **options)
    second.load_state_dict(state)
    assert second.state_dict() == first.state_dict()
    expected = [first.next_batch() for _ in range(20)]
    assert [second.next_batch() for _ in range(20)] == expected


def test_state_copies_and_finite_native_gaussian_cache() -> None:
    target = loader()
    target.next_batch()
    state = target.state_dict()
    native_rng = random.Random(7)
    native_rng.gauss(0, 1)
    state["rng"] = native_rng.getstate()
    target.load_state_dict(state)
    before = deepcopy(target.state_dict())
    state["positions"]["web"] = 10**6
    state["buffers"]["python"].append(999)
    assert target.state_dict() == before
    exported = target.state_dict()
    exported["positions"]["web"] += 1
    exported["buffers"]["python"].append(999)
    assert target.state_dict() == before
