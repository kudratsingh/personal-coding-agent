"""Docker-backed: completions generated from a Pod export execute against the full file (opt-in).

Run with ``SCRATCHCODER_SANDBOX_TESTS=1``. Authored tasks only: the reference programs pass
their real hidden tests in the evaluation runner, and the placeholder tests never run.
"""

import os
from pathlib import Path

import pytest
from eval_fixtures import TOKENIZER, io_task, pool_task, save_task_file, write_fixture_completions

from scratchcoder.eval.executor import execute_completions, read_results
from scratchcoder.eval.export import write_export

pytestmark = pytest.mark.skipif(
    os.environ.get("SCRATCHCODER_SANDBOX_TESTS") != "1",
    reason="requires an explicitly enabled disposable Docker worker",
)


def test_references_pass_their_real_hidden_tests_from_an_export(tmp_path: Path) -> None:
    tasks = [pool_task("t/1"), io_task("t/2", cases=4)]
    full = save_task_file(tmp_path / "dev.jsonl", tasks)
    shipped = tmp_path / "pod" / "dev-export.jsonl"
    write_export("pool", tasks_path=full, output=shipped, record=tmp_path / "record.json")
    completions = tmp_path / "completions.jsonl"
    write_fixture_completions(completions, shipped, tasks, lambda task, _: task.solution)
    summary = execute_completions(
        completions, full, tmp_path / "results.jsonl", tokenizer=TOKENIZER
    )
    assert summary["scored"] == 2 and summary["passed"] == 2 and summary["unresolved"] == 0
    results = read_results(tmp_path / "results.jsonl")
    assert {result.status for result in results.attempts.values()} == {"passed"}
