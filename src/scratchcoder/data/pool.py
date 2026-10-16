"""Prompt near-duplicate rejection and verified-reference pool admission."""

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import replace

from scratchcoder.data.adapters import canonical_id
from scratchcoder.data.filters import ContaminationIndex, words
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.runner import Result


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    tokens = words(text)
    size = min(5, len(tokens))
    return (
        frozenset(tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1))
        if size
        else frozenset()
    )


class SimilarityIndex:
    """Exact Jaccard check after a shared-order prefix candidate filter."""

    def __init__(self, threshold: float = 0.85) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("invalid similarity threshold")
        self.threshold = threshold
        self.items: list[frozenset[tuple[str, ...]]] = []
        self.prefixes: dict[tuple[str, ...], set[int]] = defaultdict(set)

    def _prefix(self, value: frozenset[tuple[str, ...]]) -> list[tuple[str, ...]]:
        return sorted(value)[: len(value) - math.ceil(self.threshold * len(value)) + 1]

    def contains(self, text: str) -> bool:
        value = shingles(text)
        candidates = set().union(*(self.prefixes.get(item, set()) for item in self._prefix(value)))
        return any(
            len(value & self.items[index]) / max(len(value | self.items[index]), 1)
            >= self.threshold
            for index in candidates
        )

    def add(self, text: str) -> None:
        value = shingles(text)
        for item in self._prefix(value):
            self.prefixes[item].add(len(self.items))
        self.items.append(value)


def build_pool(
    tasks: Iterable[Task], benchmarks: Iterable[Task], *, execute: Callable[[str, str, str], Result]
) -> tuple[list[Task], dict[str, int]]:
    """Admit only passing references; unavailable isolation aborts admission."""
    protected = list(benchmarks)
    index = SimilarityIndex()
    for task in protected:
        index.add(task.prompt)
    contamination = ContaminationIndex(
        part
        for task in protected
        for part in (task.prompt, task.solution or "", task.public_tests, task.hidden_tests)
    )
    benchmark_ids = {canonical_id(task.id) for task in protected}
    report: Counter[str] = Counter()

    def count(task: Task, reason: str) -> None:
        report[reason] += 1
        report[f"source:{task.source}:{reason}"] += 1

    kept = []
    seen_ids: set[str] = set()
    # Priority for explicitly held-out source tasks; never let train copies displace them.
    ordered = sorted(tasks, key=lambda item: (item.split == "train", item.source, item.id))
    for task in ordered:
        count(task, "seen")
        if task.id in seen_ids:
            raise ValueError("duplicate task IDs before pool split")
        seen_ids.add(task.id)
        if (
            canonical_id(task.id) in benchmark_ids
            or index.contains(task.prompt)
            or any(
                contamination.contains(part)
                for part in (task.prompt, task.solution or "", task.public_tests, task.hidden_tests)
            )
        ):
            count(task, "contamination_or_duplicate")
            continue
        if task.solution is None:
            count(task, "missing_solution")
            continue
        public = execute(task.solution, task.public_tests, task.mode)
        hidden = execute(task.solution, task.hidden_tests, task.mode)
        allowed = {"passed", "failed", "timeout", "output_limit", "resource_limit"}
        if public.status not in allowed or hidden.status not in allowed:
            raise RuntimeError("cannot verify tasks without working isolation")
        if not public.passed or not hidden.passed:
            count(task, "invalid_solution")
            continue
        index.add(task.prompt)
        # Explicit held-out tasks retain their split. Hash allocation never changes it to train.
        kept.append(task)
        count(task, "kept")
        report[f"authorship:{task.written_by}:kept"] += 1
    return kept, dict(report)


def split_summary(tasks: Iterable[Task]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for task in tasks:
        counts[f"split:{task.split}:total"] += 1
        counts[f"split:{task.split}:authorship:{task.written_by}"] += 1
        counts[f"split:{task.split}:source:{task.source}"] += 1
    return dict(counts)


def assign_splits(tasks: Iterable[Task], *, dev_size: int, test_size: int) -> list[Task]:
    """Allocate exact local pool sizes after deduplication, preserving fixed holdouts."""
    if any(type(size) is not int or size < 0 for size in (dev_size, test_size)):
        raise ValueError("split sizes must be nonnegative integers")
    items = list(tasks)
    if len({task.id for task in items}) != len(items):
        raise ValueError("task IDs must be unique")
    fixed = [task for task in items if task.split != "train"]
    if (
        sum(task.split == "dev" for task in fixed) > dev_size
        or sum(task.split == "test" for task in fixed) > test_size
    ):
        raise ValueError("fixed holdouts exceed requested exact split sizes")
    candidates = sorted((task for task in items if task.split == "train"), key=lambda task: task.id)
    # Use a content-independent stable identifier hash for deterministic selection.
    import hashlib

    candidates.sort(key=lambda task: hashlib.sha256(task.id.encode()).digest())
    dev_remaining = max(0, dev_size - sum(task.split == "dev" for task in fixed))
    test_remaining = max(0, test_size - sum(task.split == "test" for task in fixed))
    if len(candidates) < dev_remaining + test_remaining:
        raise ValueError("not enough verified tasks for requested holdouts")
    return fixed + [
        replace(
            task,
            split="dev"
            if i < dev_remaining
            else "test"
            if i < dev_remaining + test_remaining
            else "train",
        )
        for i, task in enumerate(candidates)
    ]
