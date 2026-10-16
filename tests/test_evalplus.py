import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data.evalplus import EvalPlusRelease, origin_task, read_release, release


def human_row() -> dict[str, Any]:
    return {
        "task_id": "HumanEval/0",
        "prompt": "def add(a,b):\n",
        "canonical_solution": "    return a+b\n",
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(12,30)==42\n",
    }


def mbpp_row() -> dict[str, Any]:
    return {
        "task_id": 2,
        "prompt": "Find common elements",
        "code": "def common(a,b): return set(a)&set(b)",
        "source_file": "fixture",
        "test_imports": ["import math"],
        "test_list": ["assert common([1],[1])=={1}"],
        "test": "assert set(common([2,3],[3,2]))=={2,3}\n",
    }


def test_origin_format_preserves_hidden_oracles_and_original_ids() -> None:
    row = human_row()
    task = origin_task(row, release("humaneval-plus"))
    assert task.hidden_tests == row["test"] + "\ncheck(add)\n"
    assert row["test"] not in task.public_prompt() and "12,30" not in task.public_prompt()
    assert task.solution == row["prompt"] + row["canonical_solution"]
    row = mbpp_row()
    task = origin_task(row, release("mbpp-plus"))
    assert task.id == "mbpp/2" and task.hidden_tests == row["test"]
    assert task.solution is not None and task.solution.startswith("import math\n")
    assert task.split == "eval" and "2,3" not in task.public_prompt()


def test_legacy_oracle_schema_and_training_provenance_rejected() -> None:
    spec = release("humaneval-plus")
    with pytest.raises(ValueError, match="schema"):
        origin_task(dict(human_row(), plus_input=[]), spec)
    with pytest.raises(ValueError, match="evaluation-only"):
        origin_task(
            human_row(),
            replace(spec, manifest=replace(spec.manifest, purpose="train", license="MIT")),
        )
    with pytest.raises(ValueError, match="integer"):
        origin_task(dict(mbpp_row(), task_id=True), release("mbpp-plus"))
    with pytest.raises(ValueError, match="only imports"):
        origin_task(
            dict(mbpp_row(), test_imports=["print('must never execute')"]), release("mbpp-plus")
        )


def fixture_release(tmp_path: Path, rows: list[dict[str, Any]]) -> tuple[Path, EvalPlusRelease]:
    raw = gzip.compress("\n".join(json.dumps(row) for row in rows).encode())
    path = tmp_path / "fixture.gz"
    path.write_bytes(raw)
    spec = release("humaneval-plus")
    return path, replace(
        spec,
        manifest=replace(
            spec.manifest, size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
        ),
        task_count=len(rows),
    )


def test_bounded_release_hash_size_count_and_unique_ids(tmp_path: Path) -> None:
    path, spec = fixture_release(tmp_path, [human_row()])
    assert len(read_release(path, spec)) == 1
    with pytest.raises(ValueError, match="expanded"):
        read_release(path, spec, expanded_limit=10)
    with pytest.raises(ValueError, match="SHA-256"):
        read_release(path, replace(spec, manifest=replace(spec.manifest, sha256="0" * 64)))
    with pytest.raises(ValueError, match="count"):
        read_release(path, replace(spec, task_count=2))
    path, spec = fixture_release(tmp_path, [human_row(), human_row()])
    with pytest.raises(ValueError, match="unique"):
        read_release(path, spec)
