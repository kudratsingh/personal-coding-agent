import random
from dataclasses import replace
from pathlib import Path

import pytest

from scratchcoder.data.pool import SimilarityIndex, assign_splits, build_pool, shingles
from scratchcoder.eval.tasks import Task, load_tasks, map_source_row, save_tasks
from scratchcoder.sandbox.runner import Result


def fixture_task(identifier: str = "one") -> Task:
    return Task(
        id=identifier,
        prompt=f"Compute the value for identifier {identifier} using the supplied rule",
        public_tests="assert solve(1) == 2",
        hidden_tests="assert solve(17) == 18",
        source="synthetic",
        license="fixture",
        written_by="human",
        split="train",
        solution="def solve(value): return value + 1",
    )


def test_task_roundtrip_and_prompt_boundary(tmp_path: Path) -> None:
    task = fixture_task()
    path = tmp_path / "tasks.jsonl"
    save_tasks(path, [task])
    assert list(load_tasks(path)) == [task]
    assert task.hidden_tests not in task.public_prompt()
    assert task.solution is not None and task.solution not in task.public_prompt()
    mapped = map_source_row(
        {"text": task.prompt},
        {"prompt": "text"},
        metadata={key: value for key, value in vars(task).items() if key != "prompt"},
    )
    assert mapped == task


def test_near_duplicate_prompt_rejection() -> None:
    index = SimilarityIndex()
    index.add("Return the sum of every even integer in the supplied list of values")
    assert index.contains("Return the sum of every even integer in the supplied list of values!")
    assert not index.contains("Find the longest common prefix in a collection of strings")


def test_pool_verifies_references_and_excludes_benchmark_ids() -> None:
    task = fixture_task()
    other = replace(
        fixture_task("other"),
        prompt="Reverse the characters in an arbitrary Unicode string",
        public_tests="assert reverse('ab') == 'ba'",
        hidden_tests="assert reverse('cd') == 'dc'",
        solution="def reverse(text): return text[::-1]",
    )
    kept, report = build_pool([task, other], [task], execute=lambda *_: Result("passed"))
    assert kept == [other]
    assert report["contamination_or_duplicate"] == 1
    with pytest.raises(RuntimeError):
        build_pool([other], [], execute=lambda *_: Result("unavailable"))


def test_exact_split_sizes_stable_order_and_fixed_holdouts() -> None:
    tasks = [fixture_task(str(i)) for i in range(10)] + [
        replace(fixture_task("fixed"), split="test")
    ]
    first = assign_splits(tasks, dev_size=2, test_size=3)
    second = assign_splits(reversed(tasks), dev_size=2, test_size=3)
    assert {task.id: task.split for task in first} == {task.id: task.split for task in second}
    assert sum(task.split == "dev" for task in first) == 2
    assert sum(task.split == "test" for task in first) == 3
    assert next(task for task in first if task.id == "fixed").split == "test"
    with pytest.raises(ValueError, match="exceed"):
        assign_splits([replace(fixture_task(), split="test")], dev_size=0, test_size=0)
    with pytest.raises(ValueError, match="integer"):
        assign_splits(tasks, dev_size=True, test_size=1)


def test_similarity_prefix_filter_matches_brute_force_reference() -> None:
    rng = random.Random(8)
    for threshold in (0.5, 0.85, 1.0):
        prompts = [
            " ".join(rng.choices(["alpha", "beta", "gamma", "delta", "epsilon"], k=20))
            for _ in range(30)
        ]
        index = SimilarityIndex(threshold)
        seen: list[frozenset[tuple[str, ...]]] = []
        for prompt in prompts + prompts:
            target = shingles(prompt)
            expected = any(
                len(target & item) / max(len(target | item), 1) >= threshold for item in seen
            )
            assert index.contains(prompt) == expected
            index.add(prompt)
            seen.append(target)


def test_unknown_isolation_outcome_aborts_pool() -> None:
    with pytest.raises(RuntimeError, match="isolation"):
        build_pool([fixture_task()], [], execute=lambda *_: Result("unknown"))
