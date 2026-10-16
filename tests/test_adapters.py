import json
from dataclasses import replace

import pytest

from scratchcoder.data.adapters import adapt_row, assertion_partitions
from scratchcoder.data.manifest import ManifestEntry
from scratchcoder.data.pool import build_pool
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
