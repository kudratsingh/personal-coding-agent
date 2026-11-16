"""Reference copies of the two functions :mod:`scratchcoder.agent.synth.dev_family` ports.

``dev_family_ids`` is copied from the pilot intake tool (``tools/pilot_intake.py``) and
``scan`` from the holdout integrity scan (``scratchcoder.data.integrity_scan``), both on
their review branches. Only the imports are gathered here; ``scan`` raises ``ValueError``
for its sealed-file error and leaves out its aggregate counts (it returns an empty mapping
in their place). The decision logic is unchanged. Tests compare the port's decisions with
these on small fixtures; nothing here is used outside the tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from scratchcoder.data.clusters import (
    DEFAULT_RULES,
    UNION_SIGNALS,
    ClusterRules,
    Item,
    UnionFind,
    find_edges,
)
from scratchcoder.eval.tasks import Task

# -- tools/pilot_intake.py ----------------------------------------------------------------


def _item(task: Task, split: str) -> Item:
    return Item(
        task.id, split, task.source, task.prompt, task.solution, task.mode, task.entry_point
    )


def dev_family_ids(
    candidates: Sequence[Task],
    dev: Sequence[Task],
    train: Sequence[Task],
    range_bridges: Sequence[Task] = (),
) -> set[str]:
    """Candidates in one task family with any dev task.

    Pool train and every other adapted row of the range (``range_bridges``: rows dropped for
    another reason) join families as bridges, as in the pool build, which assigned families
    before it dropped paraphrases. A bridge row whose ID repeats a pool or range ID is
    scanned under an internal ID and joined to the item that holds that ID."""
    items = [
        *(_item(task, "dev") for task in dev),
        *(_item(task, "train") for task in train),
        *(_item(task, "pilot") for task in candidates),
    ]
    position = {item.id: index for index, item in enumerate(items)}
    if len(position) != len(items):
        raise ValueError("dev, pool train and candidate IDs must be unique")
    joins: list[tuple[int, int]] = []
    for number, task in enumerate(range_bridges):
        if task.id in position:
            joins.append((position[task.id], len(items)))
            items.append(_item(replace(task, id=f"\x00range-{number}"), "bridge"))
        else:
            position[task.id] = len(items)
            items.append(_item(task, "bridge"))
    joined = UnionFind(len(items))
    for edge in find_edges(items):
        if edge.union:
            joined.union(edge.a, edge.b)
    for a, b in joins:
        joined.union(a, b)
    dev_roots = {joined.find(index) for index, item in enumerate(items) if item.split == "dev"}
    return {
        item.id
        for index, item in enumerate(items)
        if item.split == "pilot" and joined.find(index) in dev_roots
    }


# -- scratchcoder.data.integrity_scan -----------------------------------------------------

EXCLUDING_SIGNALS = (*UNION_SIGNALS, "prompt_named_paraphrase")
FLAG_SIGNAL = "prompt_named_flag"
CHAIN_SIGNAL = "family_chain"
SAME_ID_SIGNAL = "same_id"


@dataclass(frozen=True)
class SubjectResult:
    id: str
    label: str
    status: str
    signals: tuple[str, ...]


def scan(
    test: Sequence[Item],
    subjects: Mapping[str, Sequence[Item]],
    rules: ClusterRules = DEFAULT_RULES,
    *,
    bridges: Mapping[str, Sequence[Item]] | None = None,
) -> tuple[list[SubjectResult], dict[str, Any]]:
    bridges = bridges or {}
    test_index = {item.id: index for index, item in enumerate(test)}
    if len(test_index) != len(test):
        raise ValueError("the sealed task file repeats a task ID")
    others = [("subject", label, item) for label, items in subjects.items() for item in items]
    others += [("bridge", label, item) for label, items in bridges.items() for item in items]
    other_ids = [item.id for _, _, item in others]
    if len(set(other_ids)) != len(other_ids):
        raise ValueError("subject and bridge IDs must be unique across all sets")
    items = list(test)
    same_id: list[tuple[int, int]] = []
    for number, (_, _, item) in enumerate(others):
        if item.id in test_index:
            same_id.append((test_index[item.id], len(items)))
            items.append(replace(item, id=f"\x00same-id-{number}"))
        else:
            items.append(item)
    first_other = len(test)
    edges = find_edges(items, rules)
    joined = UnionFind(len(items))
    for edge in edges:
        if edge.union:
            joined.union(edge.a, edge.b)
    for test_position, other_position in same_id:
        joined.union(test_position, other_position)
    test_roots = {joined.find(index) for index in range(first_other)}
    direct: dict[int, set[str]] = {}
    for edge in edges:
        a_test, b_test = edge.a < first_other, edge.b < first_other
        if a_test == b_test:
            continue  # test-test or other-other: only chains matter, via the union-find
        other = edge.b if a_test else edge.a
        direct.setdefault(other, set()).update(name for name in edge.signals if name != "evidence")
    for _, other_position in same_id:
        direct.setdefault(other_position, set()).add(SAME_ID_SIGNAL)
    results: list[SubjectResult] = []
    for offset, (kind, label, item) in enumerate(others):
        index = first_other + offset
        in_family = joined.find(index) in test_roots
        if kind == "bridge":
            continue  # bridges join families; they are never judged or counted
        names = direct.get(index, set())
        signals = set(names)
        if in_family and not names & {*UNION_SIGNALS, SAME_ID_SIGNAL}:
            signals.add(CHAIN_SIGNAL)
        if in_family or names & {*EXCLUDING_SIGNALS, SAME_ID_SIGNAL}:
            status = "excluded"
        elif FLAG_SIGNAL in names:
            status = "flagged"
        else:
            status = "clear"
        results.append(SubjectResult(item.id, label, status, tuple(sorted(signals))))
    return results, {}
