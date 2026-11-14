"""Reference tests for task-family candidate clusters (``scratchcoder.data.clusters``).

Fixtures are synthetic, structure-preserving stand-ins written for these tests; no dataset
text is copied. Brute-force references check the prefix-filtered pair search and the
union-find components.
"""

import itertools
import random
from typing import Any

import pytest

from scratchcoder.data import clusters
from scratchcoder.data.clusters import Item
from scratchcoder.eval.tasks import Task
from tools.task_pool import HoldoutParaphrases

# A stand-in for a counting pair: same reference code, differently worded prompts.
COUNT_A = '''def count_matching_ends(text):
    """Count substrings that start and end with the same character."""
    total = 0
    for i in range(len(text)):
        for j in range(i, len(text)):
            if text[i] == text[j]:
                total += 1
    return total
'''
COUNT_B = """def count_matching_ends(text):
    # Every substring whose first and last characters agree.
    total = 0
    for i in range(len(text)):

        for j in range(i, len(text)):
            if text[i] == text[j]:
                total += 1
    return total
"""
# A stand-in for the two shape-class pairs: one class, two methods.
SHAPE = '''class Rectangle:
    """A rectangle."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)
'''


def item(identifier: str, split: str, prompt: str, solution: str | None, **extra: Any) -> Item:
    return Item(identifier, split, "fixture/source", prompt, solution, **extra)


def test_solution_key_ignores_layout_comments_and_docstrings_but_not_operators() -> None:
    assert clusters.solution_key(COUNT_A) == clusters.solution_key(COUNT_B)
    assert clusters.solution_key(COUNT_A) != clusters.solution_key(COUNT_A.replace("==", "!="))
    assert clusters.solution_key(SHAPE) == clusters.solution_key(
        SHAPE.replace('"""A rectangle."""', "")
    )
    assert clusters.solution_key("def broken(:\n") is None
    assert clusters.solution_key(None) is None and clusters.solution_key("  ") is None
    # Invalid escape sequences still parse; the warning is not an error.
    assert clusters.solution_key("import re\nPATTERN = re.compile('\\d+')\n") is not None


def test_canonical_key_matches_renamings_only() -> None:
    first = (
        "def total(values, scale=2):\n    acc = 0\n    for v in values:\n"
        "        acc += v\n    return acc * scale\n"
    )
    renamed = (
        "def summed(xs, factor=2):\n    s = 0\n    for x in xs:\n"
        "        s += x\n    return s * factor\n"
    )
    annotated = (
        "def summed(xs: list[int], factor: int = 2) -> int:\n    s: int = 0\n"
        "    for x in xs:\n        s += x\n    return s * factor\n"
    )
    assert clusters.solution_key(first) != clusters.solution_key(renamed)
    assert clusters.canonical_key(first) == clusters.canonical_key(renamed)
    assert clusters.canonical_key(first) == clusters.canonical_key(annotated)
    # Constants, builtins and operators are kept.
    assert clusters.canonical_key(first) != clusters.canonical_key(first.replace("2", "3"))
    assert clusters.canonical_key("def f(a):\n    return len(a)\n") != clusters.canonical_key(
        "def f(a):\n    return max(a)\n"
    )
    # A keyword naming a local parameter is renamed with it; other keywords are kept.
    call = "def f(a, b):\n    return a - b\nprint(f(a=1, b=2), sorted([1], key=abs))\n"
    other = "def g(x, y):\n    return x - y\nprint(g(x=1, y=2), sorted([1], key=abs))\n"
    assert clusters.canonical_key(call) == clusters.canonical_key(other)
    assert clusters.canonical_key(call) != clusters.canonical_key(other.replace("key=", "reverse="))


def test_defined_and_top_level_names() -> None:
    source = (
        "def main():\n    pass\n\nclass Box:\n    def __init__(self):\n"
        "        pass\n    def open(self):\n        pass\n"
    )
    assert clusters.defined_names(source) == {"Box", "open"}
    assert clusters.top_level_names(source) == ("main", "Box")
    assert clusters.defined_names("def (") == frozenset()


@pytest.mark.parametrize("threshold", [0.1, 0.25, 0.5, 0.85, 1.0])
def test_prefix_filtered_pairs_equal_brute_force(threshold: float) -> None:
    generator = random.Random(7)
    sets = [frozenset(generator.sample(range(14), generator.randint(0, 7))) for _ in range(60)]
    expected = {
        (i, j)
        for i, j in itertools.combinations(range(len(sets)), 2)
        if sets[i] and sets[j] and clusters.jaccard(sets[i], sets[j]) >= threshold
    }
    found = clusters.similar_pairs(sets, threshold)
    assert {(i, j) for i, j, _ in found} == expected
    assert all(score == clusters.jaccard(sets[i], sets[j]) for i, j, score in found)
    with pytest.raises(ValueError):
        clusters.similar_pairs(sets, 0)


def test_components_equal_a_breadth_first_reference() -> None:
    generator = random.Random(3)
    size = 40
    pairs = [sorted(generator.sample(range(size), 2)) for _ in range(30)]
    edges = [clusters.Edge(a, b, {"solution_ast": "k"}) for a, b in pairs]
    edges.append(clusters.Edge(0, 39, {"prompt_named_flag": 0.2}))
    neighbours: dict[int, set[int]] = {index: set() for index in range(size)}
    for edge in edges:
        if edge.union:
            neighbours[edge.a].add(edge.b)
            neighbours[edge.b].add(edge.a)
    seen: set[int] = set()
    expected = []
    for start in range(size):
        if start in seen:
            continue
        group, frontier = {start}, [start]
        while frontier:
            for nxt in neighbours[frontier.pop()] - group:
                group.add(nxt)
                frontier.append(nxt)
        seen |= group
        expected.append(sorted(group))
    assert clusters.components(size, edges) == expected
    with pytest.raises(ValueError, match="flag-only"):
        clusters.components(size, edges, ("prompt_named_flag",))


def test_regression_pairs_the_build_rules_miss_are_joined() -> None:
    dev_count = item(
        "dev/count", "dev", "Count the substrings of a string whose two ends match.", COUNT_A
    )
    train_count = item(
        "train/count",
        "train",
        "Given text, return how many contiguous pieces begin and finish with one letter.",
        COUNT_B,
    )
    dev_shape = item(
        "dev/shape", "dev", "Implement a class with width, height, area and perimeter.", SHAPE
    )
    train_shape = item(
        "train/shape",
        "train",
        "Design a geometric type storing two side lengths; expose its surface and border length.",
        SHAPE.replace('"""A rectangle."""', '"""Axis-aligned box."""'),
        roles=("sft_train",),
    )
    changed = item(
        "train/changed",
        "train",
        "Report how many unequal character pairs a word holds.",
        COUNT_A.replace("==", "!="),
    )
    items = [dev_count, train_count, dev_shape, train_shape, changed]

    # (a) The v1 holdout-paraphrase rule keeps both training tasks.
    def as_task(entry: Item) -> Task:
        return Task(
            id=entry.id,
            prompt=entry.prompt,
            public_tests="assert True",
            hidden_tests="assert True",
            source=entry.source,
            license="CC-BY-4.0",
            written_by="model",
            split=entry.split,
            solution=entry.solution,
        )

    paraphrases = HoldoutParaphrases([as_task(dev_count), as_task(dev_shape)])
    assert not paraphrases.matches(as_task(train_count))
    assert not paraphrases.matches(as_task(train_shape))
    # (b, c) The cluster scan joins each pair; a one-operator change stays apart.
    edges = clusters.find_edges(items)
    groups = [[items[i].id for i in members] for members in clusters.components(len(items), edges)]
    assert ["dev/count", "train/count"] in groups
    assert ["dev/shape", "train/shape"] in groups
    assert ["train/changed"] in groups
    joined = next(edge for edge in edges if (edge.a, edge.b) == (2, 3))
    assert "solution_ast" in joined.signals and "canonical_ast" in joined.signals
    assert joined.signals["evidence"]["shared_names"] == ["Rectangle", "area", "perimeter"]
    assert joined.signals["evidence"]["same_top_level_names"] is True
    family = clusters.family_map(items)
    assert family["dev/shape"] == family["train/shape"] != family["dev/count"]


def test_flag_edges_are_recorded_but_never_join() -> None:
    items = [
        item(
            "a",
            "dev",
            "Write fibonacci(n) that returns the nth fibonacci number for a given integer n",
            "def fibonacci(n):\n    return n\n",
        ),
        item(
            "b",
            "train",
            "Write fibonacci(n) that returns a list of fibonacci numbers below n",
            "def fibonacci(n):\n    return [n]\n",
        ),
    ]
    edges = clusters.find_edges(items)
    assert len(edges) == 1 and not edges[0].union
    assert set(edges[0].signals) - {"evidence"} <= set(clusters.FLAG_SIGNALS)
    assert clusters.components(2, edges) == [[0], [1]]


def test_cluster_records_start_pending_and_mixed_modes_start_quarantined() -> None:
    items = [
        item("d", "dev", "Add two numbers.", "print(sum(map(int, input().split())))\n", mode="io"),
        item("t", "train", "Add two numbers.", "print(sum(map(int, input().split())))\n"),
    ]
    edges = clusters.find_edges(items)
    record = clusters.cluster_record(items, [0, 1], edges)
    assert record["adjudication"]["status"] == "unclear_quarantined"
    assert "function task and an I/O task" in record["uncertainty"][0]
    assert record["behavior"] == [clusters.pending_behavior("d", "t")]
    same = [
        items[0],
        Item("t", "train", "fixture/source", "Add two numbers.", items[1].solution, "io"),
    ]
    record = clusters.cluster_record(same, [0, 1], clusters.find_edges(same))
    assert record["adjudication"] == clusters.pending_adjudication()
    assert record["cluster_id"] == clusters.cluster_id(["t", "d"])


def manifest_with(record: dict[str, Any]) -> dict[str, Any]:
    return {"format": clusters.MANIFEST_FORMAT, "clusters": [record]}


def test_adjudication_needs_matching_behaviour_a_reviewer_and_kept_uncertainty() -> None:
    items = [
        item("d", "dev", "Sum.", "def s(a):\n    return sum(a)\n"),
        item("t", "train", "Total.", "def s(a):\n    return sum(a)\n"),
    ]
    record = clusters.cluster_record(items, [0, 1], clusters.find_edges(items))
    manifest = manifest_with(record)
    clusters.validate_manifest(manifest)
    decision = {"reviewer": "owner", "decided_at": "2026-11-13", "rationale": "same contract"}
    with pytest.raises(ValueError, match="behaviour result"):
        clusters.adjudicate(
            manifest, record["cluster_id"], status="confirmed_equivalent", **decision
        )
    with pytest.raises(ValueError, match="state the uncertainty"):
        clusters.adjudicate(
            manifest, record["cluster_id"], status="unclear_quarantined", **decision
        )
    with pytest.raises(ValueError, match="pending"):
        clusters.adjudicate(manifest, record["cluster_id"], status="pending", **decision)
    manifest = manifest_with(clusters.cluster_record(items, [0, 1], clusters.find_edges(items)))
    manifest["clusters"][0]["behavior"][0]["outcome"] = "equivalent"
    done = clusters.adjudicate(
        manifest, record["cluster_id"], status="confirmed_equivalent", **decision
    )
    assert done["adjudication"]["status"] == "confirmed_equivalent"
    unclear = manifest_with(clusters.cluster_record(items, [0, 1], clusters.find_edges(items)))
    clusters.adjudicate(
        unclear,
        record["cluster_id"],
        status="unclear_quarantined",
        uncertainty="tests are too weak to tell",
        **decision,
    )
    assert unclear["clusters"][0]["uncertainty"] == ["tests are too weak to tell"]


def test_manifest_validation_refuses_inconsistent_records() -> None:
    items = [
        item("d", "dev", "Sum.", "def s(a):\n    return sum(a)\n"),
        item("t", "train", "Total.", "def s(a):\n    return sum(a)\n"),
    ]
    record = clusters.cluster_record(items, [0, 1], clusters.find_edges(items))
    bad = [
        ("cluster_id", "tc-0000000000000000", "does not match"),
        ("members", [*record["members"], {**record["members"][0], "id": "x", "split": "test"}], ""),
    ]
    for key, value, _ in bad:
        broken = manifest_with({**record, key: value})
        with pytest.raises(ValueError):
            clusters.validate_manifest(broken)
    pending_with_reviewer = {**record, "adjudication": {**record["adjudication"], "reviewer": "x"}}
    with pytest.raises(ValueError, match="no reviewer"):
        clusters.validate_manifest(manifest_with(pending_with_reviewer))
    quiet = {**record, "adjudication": {**record["adjudication"], "status": "unclear_quarantined"}}
    with pytest.raises(ValueError, match="uncertainty"):
        clusters.validate_manifest(manifest_with(quiet))
    with pytest.raises(ValueError, match="format"):
        clusters.validate_manifest({"format": "other", "clusters": []})
    with pytest.raises(ValueError, match="unique"):
        clusters.find_edges([items[0], items[0]])
