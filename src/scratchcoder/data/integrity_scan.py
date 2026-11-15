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

The result names subjects only. It never carries a test task's ID, prompt, reference,
tests, signal key or any per-test value: each subject gets the *names* of the signals that
tied it to the test split, and the aggregates are counts.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scratchcoder.data.clusters import (
    DEFAULT_RULES,
    FLAG_SIGNALS,
    RULES_VERSION,
    UNION_SIGNALS,
    ClusterRules,
    Item,
    components,
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
    id: str
    label: str
    status: str
    signals: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "status": self.status, "signals": self.signals}


def scan(
    test: Sequence[Item],
    subjects: Mapping[str, Sequence[Item]],
    rules: ClusterRules = DEFAULT_RULES,
) -> tuple[list[SubjectResult], dict[str, Any]]:
    """Each subject's status, and aggregate counts. Subject IDs must be unique across labels.

    A subject whose ID equals a test task's ID is excluded without entering the family scan
    (the scan needs unique IDs)."""
    test_ids = {item.id for item in test}
    if len(test_ids) != len(test):
        raise SealedContentError("the sealed task file repeats a task ID")
    labelled = [(label, item) for label, items in subjects.items() for item in items]
    subject_ids = [item.id for _, item in labelled]
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("subject IDs must be unique across all subject sets")
    same_id = {item.id for _, item in labelled if item.id in test_ids}
    scanned = [(label, item) for label, item in labelled if item.id not in same_id]
    items = [*test, *(item for _, item in scanned)]
    first_subject = len(test)
    edges = find_edges(items, rules)
    groups = components(len(items), edges)
    has_test = [False] * len(items)
    for members in groups:
        touched = any(index < first_subject for index in members)
        for index in members:
            has_test[index] = touched
    direct: dict[int, set[str]] = {}
    touched_tests: set[int] = set()
    for edge in edges:
        a_test, b_test = edge.a < first_subject, edge.b < first_subject
        if a_test == b_test:
            continue  # test-test or subject-subject: only chains matter, via components
        subject, test_index = (edge.b, edge.a) if a_test else (edge.a, edge.b)
        names = {name for name in edge.signals if name != "evidence"}
        direct.setdefault(subject, set()).update(names)
        touched_tests.add(test_index)
    position = {item.id: first_subject + offset for offset, (_, item) in enumerate(scanned)}
    results: list[SubjectResult] = []
    for label, item in labelled:
        if item.id in same_id:
            results.append(SubjectResult(item.id, label, "excluded", (SAME_ID_SIGNAL,)))
            continue
        index = position[item.id]
        names = direct.get(index, set())
        signals = set(names)
        if has_test[index] and not names & set(UNION_SIGNALS):
            signals.add(CHAIN_SIGNAL)
        if has_test[index] or names & set(EXCLUDING_SIGNALS):
            status = "excluded"
        elif FLAG_SIGNAL in names:
            status = "flagged"
        else:
            status = "clear"
        results.append(SubjectResult(item.id, label, status, tuple(sorted(signals))))
    return results, aggregates(results, test_tasks=len(test), touched_tests=len(touched_tests))


def aggregates(
    results: Sequence[SubjectResult], *, test_tasks: int, touched_tests: int
) -> dict[str, Any]:
    by_label: dict[str, Counter[str]] = {}
    signals: dict[str, Counter[str]] = {}
    for result in results:
        by_label.setdefault(result.label, Counter())[result.status] += 1
        for name in result.signals:
            signals.setdefault(result.label, Counter())[name] += 1
    return {
        "test_tasks_scanned": test_tasks,
        "test_tasks_with_any_subject_edge": touched_tests,
        "subjects": {
            label: {status: counts.get(status, 0) for status in STATUSES}
            | {"total": sum(counts.values())}
            for label, counts in sorted(by_label.items())
        },
        "subjects_by_signal": {
            label: dict(sorted(counts.items())) for label, counts in sorted(signals.items())
        },
    }
