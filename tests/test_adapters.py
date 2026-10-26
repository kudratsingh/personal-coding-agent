import json
from dataclasses import replace

import pytest

from scratchcoder.data.adapters import adapt_row, adapt_taco_io_row, assertion_partitions
from scratchcoder.data.manifest import ManifestEntry
from scratchcoder.data.pool import build_pool, split_summary
from scratchcoder.sandbox.runner import Result


def provenance(source: str) -> ManifestEntry:
    return ManifestEntry(
        source=source,
        revision="a" * 40,
        url="https://example.invalid/synthetic",
        license="MIT",
        attribution="Synthetic schema fixture",
        sha256="b" * 64,
        size_bytes=0,
        license_verified=True,
    )


def test_opencode_and_acecode_mapping() -> None:
    tests = ["assert add(1,2) == 3", "assert add(4,5) == 9"]
    source = "nvidia/OpenCodeInstruct"
    task = adapt_row(
        source,
        {
            "id": 1,
            "input": "Implement add",
            "output": "def add(a,b): return a+b",
            "unit_tests": json.dumps(tests),
        },
        provenance(source),
    )
    assert "4, 5" not in task.public_tests
    assert "4, 5" in task.hidden_tests
    assert task.source_revision == "a" * 40
    source = "TIGER-Lab/AceCode-87K"
    task = adapt_row(
        source,
        {
            "id": 2,
            "question": "Implement add",
            "test_cases": tests,
            "inferences": [
                {"completion": "wrong", "pass_rate": 0},
                {"completion": "```python\ndef add(a,b): return a+b\n```", "pass_rate": 1.0},
            ],
        },
        provenance(source),
    )
    assert task.solution is not None and task.solution.startswith("def add")


def test_mbpp_plus_id_exclusion_and_setup() -> None:
    source = "google-research-datasets/mbpp"
    task = adapt_row(
        source,
        {
            "task_id": 17,
            "text": "Implement add",
            "code": "def add(a,b): return a+b",
            "test_setup_code": "import math",
            "test_list": ["assert add(1,2)==3", "assert add(4,5)==9"],
        },
        provenance(source),
    )
    assert task.id == "mbpp/17"
    assert "import math" in task.public_tests and "import math" in task.hidden_tests
    benchmark = replace(task, id="MBPP/17", split="eval")
    kept, report = build_pool([task], [benchmark], execute=lambda *_: Result("passed"))
    assert not kept and report["contamination_or_duplicate"] == 1


def test_unsupported_tests_and_missing_terms_fail() -> None:
    with pytest.raises(ValueError, match="only imports"):
        assertion_partitions(["def check(): pass", "assert True"])
    with pytest.raises(ValueError):
        assertion_partitions(["assert True", "assert True"])
    with pytest.raises(ValueError, match="only imports"):
        assertion_partitions(["assert add(1,2)==3", "assert add(3,4)==7"], setup="answer=42")
    source = "nvidia/OpenCodeInstruct"
    with pytest.raises(ValueError, match="accept"):
        adapt_row(source, {}, replace(provenance(source), terms_required=True))


def test_humaneval_adapter_is_evaluation_only() -> None:
    source = "openai/human-eval"
    row = {
        "task_id": "HumanEval/0",
        "prompt": "def add(a,b):\n",
        "entry_point": "add",
        "test": "def check(candidate): assert candidate(1,2) == 3",
        "canonical_solution": "    return a+b\n",
    }
    entry = replace(provenance(source), purpose="eval")
    task = adapt_row(source, row, entry, split="eval")
    assert task.hidden_tests.endswith("check(add)")
    with pytest.raises(ValueError, match="evaluation-only"):
        adapt_row(source, row, entry, split="train")


def taco_fixture() -> dict[str, str]:
    return {
        "question": "Add two integers from stdin and print their sum",
        "difficulty": "EASY",
        "starter_code": "",
        "url": "https://example.invalid/problem/1",
        "picture_num": "0",
        "solutions": json.dumps(["a,b=map(int,input().split());print(a+b)"]),
        "input_output": json.dumps(
            {"inputs": ["1 2\n", "1 2\n", "4 5\n"], "outputs": ["3\n", "3", "9\n"]}
        ),
    }


def test_taco_io_dedup_stable_id_holdout_and_partitioning() -> None:
    source = provenance("BAAI/TACO")
    row = taco_fixture()
    task = adapt_taco_io_row(row, source, upstream_split="test", split="test")
    assert task.mode == "io" and task.written_by == "human"
    assert len(json.loads(task.public_tests)) == len(json.loads(task.hidden_tests)) == 1
    assert "4 5" not in task.public_prompt()
    assert adapt_taco_io_row(row, source, upstream_split="train", split="train").id == task.id
    with pytest.raises(ValueError, match="cannot become training"):
        adapt_taco_io_row(row, source, upstream_split="test", split="train")
    with pytest.raises(ValueError, match="evaluation-only"):
        adapt_taco_io_row(
            row, replace(source, purpose="eval"), upstream_split="train", split="train"
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("difficulty", "HARD"),
        ("starter_code", "class Solution:"),
        ("picture_num", "1"),
        ("solutions", "[]"),
        ("input_output", '{"inputs": ["1", "1"], "outputs": ["2", "3"]}'),
        ("input_output", '{"inputs": ["1", "2"], "outputs": ["2", "3"], "fn_name": "solve"}'),
    ],
)
def test_taco_rejects_unsupported_or_conflicting_rows(field: str, value: str) -> None:
    row = dict(taco_fixture(), **{field: value})
    with pytest.raises(ValueError):
        adapt_taco_io_row(row, provenance("BAAI/TACO"), upstream_split="train", split="train")


def test_taco_null_picture_count_is_figure_free_unless_a_placeholder_remains() -> None:
    source = provenance("BAAI/TACO")
    row: dict[str, str | None] = dict(taco_fixture(), picture_num=None)
    assert adapt_taco_io_row(row, source, upstream_split="train", split="train").mode == "io"
    question = taco_fixture()["question"]
    for marker in ("[Image]", "<image>", "<img src=x>", "![fig](a.png)"):
        with pytest.raises(ValueError, match="standalone"):
            adapt_taco_io_row(
                dict(row, question=question + " " + marker),
                source,
                upstream_split="train",
                split="train",
            )


def test_taco_case_limit_rejects_by_default_and_truncation_keeps_upstream_order() -> None:
    source = provenance("BAAI/TACO")
    inputs = [f"{i} {i}\n" for i in range(150)]
    outputs = [f"{2 * i}\n" for i in range(150)]
    row = dict(taco_fixture(), input_output=json.dumps({"inputs": inputs, "outputs": outputs}))
    with pytest.raises(ValueError, match="unsupported TACO"):
        adapt_taco_io_row(row, source, upstream_split="test", split="test")
    task = adapt_taco_io_row(row, source, upstream_split="test", split="test", truncate_cases=True)
    public, hidden = json.loads(task.public_tests), json.loads(task.hidden_tests)
    assert len(public) + len(hidden) == 100
    assert [case["input"] for case in public + hidden] == inputs[:100]


def test_pool_drop_reasons_per_source_and_split_authorship_counts() -> None:
    task = adapt_taco_io_row(
        taco_fixture(), provenance("BAAI/TACO"), upstream_split="test", split="test"
    )
    missing = replace(
        task, id="other", prompt="Reverse the characters in a supplied string", solution=None
    )
    kept, report = build_pool([task, missing], [], execute=lambda *_: Result("passed"))
    assert kept == [task]
    assert report["source:BAAI/TACO:seen"] == 2
    assert report["source:BAAI/TACO:kept"] == report["source:BAAI/TACO:missing_solution"] == 1
    assert split_summary(kept)["split:test:authorship:human"] == 1


@pytest.mark.parametrize(
    "sentence",
    [
        "If there are several answers, print any of them.",
        "Output any valid permutation.",
        "Your answer is accepted if its absolute or relative error does not exceed 1e-6.",
        "There may be multiple possible answers.",
    ],
)
def test_taco_tasks_needing_a_checker_are_rejected(sentence: str) -> None:
    row = dict(taco_fixture(), question=taco_fixture()["question"] + ". " + sentence)
    with pytest.raises(ValueError, match="checker"):
        adapt_taco_io_row(row, provenance("BAAI/TACO"), upstream_split="train", split="train")


def test_taco_figure_wording_counts_when_the_picture_count_is_unrecorded() -> None:
    question = taco_fixture()["question"] + ". The grid is shown in the figure below."
    source = provenance("BAAI/TACO")
    with pytest.raises(ValueError, match="standalone"):
        adapt_taco_io_row(
            dict(taco_fixture(), question=question, picture_num=None),
            source,
            upstream_split="train",
            split="train",
        )
    recorded = dict(taco_fixture(), question=question)
    assert adapt_taco_io_row(recorded, source, upstream_split="train", split="train").mode == "io"
