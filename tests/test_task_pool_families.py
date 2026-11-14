"""Family grouping in the pool split assigner (``tools.task_pool.assign``); no pyarrow needed.

The build-level checks (a null option builds the same pool; families stay together in a
full fixture build) live in ``tests/test_task_pool.py``, which needs the evaluation extra.
"""

import json
from pathlib import Path

import pytest

from scratchcoder.eval.tasks import Task
from tools import task_pool

CONFIG = Path(__file__).resolve().parents[1] / "configs/task-pool-rehearsal.json"


def test_the_rehearsal_spec_has_no_family_grouping_and_still_loads() -> None:
    spec = task_pool.load_spec(CONFIG)
    assert "family_grouping" not in spec["splits"]


def test_family_grouping_is_validated_in_the_spec(tmp_path: Path) -> None:
    value = json.loads(CONFIG.read_text())
    path = tmp_path / "spec.json"
    for setting, accepted in (
        ("task-clusters-v0", False),
        (None, True),
        ("task-clusters-v1", True),
    ):
        value["splits"]["family_grouping"] = setting
        path.write_text(json.dumps(value))
        if accepted:
            assert task_pool.load_spec(path)["splits"]["family_grouping"] == setting
        else:
            with pytest.raises(ValueError, match="family_grouping"):
                task_pool.load_spec(path)


def pool_task(
    identifier: str, source: str, *, split: str = "train", solution: str = "x = 1"
) -> Task:
    human = source in {task_pool.DATASETS["taco"], task_pool.DATASETS["mbpp"]}
    return Task(
        id=identifier,
        prompt=f"prompt {identifier}",
        public_tests="assert True",
        hidden_tests="assert True",
        source=source,
        license="CC-BY-4.0",
        written_by="human" if human else "model",
        split=split,
        solution=solution,
    )


def assignment_fixture() -> list[Task]:
    taco, mbpp = task_pool.DATASETS["taco"], task_pool.DATASETS["mbpp"]
    opencode, acecode = task_pool.DATASETS["opencode"], task_pool.DATASETS["acecode"]
    return [
        *(pool_task(f"taco-test-{i}", taco, split="test") for i in range(3)),
        *(pool_task(f"taco-train-{i}", taco) for i in range(2)),
        *(pool_task(f"mbpp-{i}", mbpp) for i in range(4)),
        *(pool_task(f"oc-{i}", opencode) for i in range(10)),
        *(pool_task(f"ac-{i}", acecode) for i in range(3)),
    ]


SPLITS = {"dev": 4, "test": 4, "human_holdout_minimum": 4, "model_holdout_sources": ["opencode"]}
HISTORICAL = {
    "dev": ["oc-1", "oc-2", "taco-test-0", "taco-test-1"],
    "test": ["mbpp-1", "oc-4", "oc-7", "taco-test-2"],
}
"""The assignment of ``assignment_fixture`` by ``tools/task_pool.py`` as of main ``b1bc3e6``,
before family grouping existed (computed with that file, not with this branch)."""


def test_without_family_grouping_the_historical_assignment_is_unchanged() -> None:
    for splits in (SPLITS, {**SPLITS, "family_grouping": None}):
        result, rule = task_pool.assign(assignment_fixture(), splits)
        found = {name: sorted(t.id for t in result if t.split == name) for name in ("dev", "test")}
        assert found == HISTORICAL
        assert "family_grouping" not in rule
    with pytest.raises(ValueError, match="needs splits.family_grouping"):
        task_pool.assign(assignment_fixture(), SPLITS, families={"oc-1": "f"})


def test_family_grouping_keeps_every_family_in_one_split() -> None:
    families = {
        # A TACO test task's family pulls its training-side members into the same holdout.
        "taco-test-0": "f-taco",
        "taco-train-0": "f-taco",
        "oc-3": "f-taco",
        # A model family with a member from a source that may not be held out stays in train.
        "oc-1": "f-mixed",
        "ac-0": "f-mixed",
        # A model-only family moves as one unit.
        "oc-2": "f-pair",
        "oc-5": "f-pair",
    }
    splits = {**SPLITS, "dev": 6, "test": 5, "family_grouping": "task-clusters-v1"}
    result, rule = task_pool.assign(assignment_fixture(), splits, families=families)
    split_of = {t.id: t.split for t in result}
    for family in set(families.values()):
        assert len({split_of[i] for i, f in families.items() if f == family}) == 1
    assert split_of["taco-test-0"] == split_of["taco-train-0"] == split_of["oc-3"] != "train"
    assert split_of["oc-1"] == split_of["ac-0"] == "train"
    assert sum(t.split == "dev" for t in result) == 6
    assert sum(t.split == "test" for t in result) == 5
    assert all(split_of[f"taco-test-{i}"] != "train" for i in range(3))
    grouping = rule["family_grouping"]
    assert grouping["version"] == "task-clusters-v1" and grouping["families"] == 3
    assert grouping["holdout_members_pulled_in_by_source"] == {task_pool.DATASETS["taco"]: 1}
    assert grouping["model_families_skipped"] >= 1
    with pytest.raises(ValueError, match="exceed"):
        task_pool.assign(
            assignment_fixture(),
            {**splits, "dev": 1},
            families={f"taco-test-{i}": "f" for i in range(3)},
        )
