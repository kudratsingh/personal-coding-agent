"""Holdout integrity scan: which candidate tasks share a family with a sealed test task.

Static text and AST signals only. Nothing here generates, executes or scores anything. The
rules are the frozen task-family rules of :mod:`scratchcoder.data.clusters`
(``task-clusters-v1``), applied between the sealed test split and the *subjects* (tasks
that might be trained on). A subject is:

``excluded``
    when it is in the same family (union-signal connected component: equal reference AST,
    equal reference AST up to renaming, equal upstream ID, or prompt shingle Jaccard at or
    above 0.5) as any test task, directly or through other subjects; or when it shares a
    defined function or class name with a test task at prompt Jaccard at or above 0.25 (the
    pool build's named-paraphrase drop rule); or when its ID equals a test task's ID;
``flagged``
    when it is not excluded but shares a defined name with a test task at prompt Jaccard at
    or above 0.10 (the rules' flag level, never merged into a family). The scan cannot
    adjudicate a flag without showing the test task, so the owner decides what a flag means;
``clear``
    otherwise.

Families are connected components over the test tasks, the subjects and any *bridge* tasks
(tasks that are not judged, such as the pool's train and dev splits), so a subject joined to
a test task through a bridge is excluded too.

The result names subjects only, each with its status. It never carries a test task's ID,
prompt, reference, tests, signal key or any per-test value, and it does not say which signal
excluded a subject: signal names appear only in aggregate counts.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from scratchcoder.data.clusters import (
    DEFAULT_RULES,
    FLAG_SIGNALS,
    RULES_VERSION,
    UNION_SIGNALS,
    ClusterRules,
    Item,
    UnionFind,
    find_edges,
)
from scratchcoder.eval.tasks import Task

SCAN_FORMAT = "holdout-integrity-scan-v1"
STATUSES = ("excluded", "flagged", "clear")
EXCLUDING_SIGNALS = (*UNION_SIGNALS, "prompt_named_paraphrase")
"""Signals that exclude a subject on a direct edge to a test task."""
FLAG_SIGNAL = "prompt_named_flag"
CHAIN_SIGNAL = "family_chain"
"""Recorded when a subject is in a test task's family only through other subjects."""
SAME_ID_SIGNAL = "same_id"
assert FLAG_SIGNAL in FLAG_SIGNALS and RULES_VERSION == DEFAULT_RULES.version


class SealedContentError(ValueError):
    """The sealed task file could not be parsed. Its message never quotes the file."""


def item_of(task: Task) -> Item:
    return Item(
        task.id, task.split, task.source, task.prompt, task.solution, task.mode, task.entry_point
    )


def parse_sealed_tasks(data: bytes, *, split: str = "test") -> list[Task]:
    """The tasks of a sealed JSONL file's bytes, every one of ``split``.

    Errors name a line number at most, never an ID or any text from the file."""
    tasks: list[Task] = []
    seen: set[str] = set()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise SealedContentError("the sealed task file is not UTF-8") from None
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            task = Task(**json.loads(line))
        except (ValueError, TypeError):
            raise SealedContentError(f"sealed task file line {number} is not a task") from None
        if task.split != split:
            raise SealedContentError(f"sealed task file line {number} is not a {split} task")
        if task.id in seen:
            raise SealedContentError(f"sealed task file line {number} repeats a task ID")
        seen.add(task.id)
        tasks.append(task)
    return tasks


@dataclass(frozen=True)
class SubjectResult:
    """One subject's status. ``signals`` stay in memory: outputs carry only the status, so a
    subject's line never says which property of an unnamed test task it matched."""

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
    """Each subject's status, and aggregate counts.

    ``bridges`` are tasks that are not judged (for example pool train and dev tasks) but
    take part in the family scan, so a subject joined to a test task's family only through
    one of them is still excluded. Subject and bridge IDs must be unique among themselves.
    A subject or bridge whose ID equals a test task's ID is scanned under an internal ID and
    joined to that test task's family, so tasks reached only through it are found too."""
    bridges = bridges or {}
    test_index = {item.id: index for index, item in enumerate(test)}
    if len(test_index) != len(test):
        raise SealedContentError("the sealed task file repeats a task ID")
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
    touched: set[int] = set()
    for edge in edges:
        a_test, b_test = edge.a < first_other, edge.b < first_other
        if a_test == b_test:
            continue  # test-test or other-other: only chains matter, via the union-find
        other, test_position = (edge.b, edge.a) if a_test else (edge.a, edge.b)
        direct.setdefault(other, set()).update(name for name in edge.signals if name != "evidence")
        touched.add(test_position)
    for test_position, other_position in same_id:
        direct.setdefault(other_position, set()).add(SAME_ID_SIGNAL)
        touched.add(test_position)
    results: list[SubjectResult] = []
    bridge_counts: dict[str, Counter[str]] = {label: Counter() for label in bridges}
    subject_roots: set[int] = set()
    for offset, (kind, label, item) in enumerate(others):
        index = first_other + offset
        in_family = joined.find(index) in test_roots
        if kind == "bridge":
            bridge_counts[label]["in_test_family" if in_family else "outside"] += 1
            continue
        subject_roots.add(joined.find(index))
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
    counts = aggregates(results, labels=list(subjects), test_tasks=len(test))
    counts["test_tasks_with_direct_subject_or_bridge_edge"] = len(touched)
    counts["test_tasks_in_a_subject_family"] = sum(
        joined.find(index) in subject_roots for index in range(first_other)
    )
    counts["bridges"] = {
        label: {"in_test_family": tally["in_test_family"], "total": sum(tally.values())}
        for label, tally in sorted(bridge_counts.items())
    }
    return results, counts


def aggregates(
    results: Sequence[SubjectResult], *, labels: Sequence[str], test_tasks: int
) -> dict[str, Any]:
    by_label: dict[str, Counter[str]] = {label: Counter() for label in labels}
    signals: dict[str, Counter[str]] = {label: Counter() for label in labels}
    for result in results:
        by_label[result.label][result.status] += 1
        signals[result.label].update(result.signals)
    return {
        "test_tasks_scanned": test_tasks,
        "subjects": {
            label: {status: counts.get(status, 0) for status in STATUSES}
            | {"total": sum(counts.values())}
            for label, counts in sorted(by_label.items())
        },
        "subjects_by_signal": {
            label: dict(sorted(counts.items())) for label, counts in sorted(signals.items())
        },
    }
