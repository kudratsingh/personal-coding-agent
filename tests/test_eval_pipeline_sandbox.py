"""Docker-backed checks of the Mac executor and the constant baseline (opt-in).

Run with ``SCRATCHCODER_SANDBOX_TESTS=1``. Authored dev-shaped tasks always run; with
``SCRATCHCODER_POOL_DEV`` pointing at the pool's dev JSONL, a few real dev tasks (one per
source, short references) run too. Every program here goes through the evaluation runner.
"""

import json
import os
from pathlib import Path

import pytest
from eval_fixtures import TOKENIZER, io_task, pool_task, save_task_file, write_fixture_completions

from scratchcoder.eval.baseline import pool_constant_baseline
from scratchcoder.eval.completions import Sampling
from scratchcoder.eval.executor import execute_completions, read_results
from scratchcoder.eval.tasks import Task, load_tasks

pytestmark = pytest.mark.skipif(
    os.environ.get("SCRATCHCODER_SANDBOX_TESTS") != "1",
    reason="requires an explicitly enabled disposable Docker worker",
)


def wrong_program(task: Task) -> str:
    return "print(0)\n" if task.mode == "io" else "def placeholder():\n    return None\n"


def run_references_and_wrong_answers(tmp_path: Path, tasks: list[Task]) -> dict[str, list[str]]:
    tasks_path = save_task_file(tmp_path / "dev.jsonl", tasks)
    completions = tmp_path / "completions.jsonl"

    write_fixture_completions(
        completions,
        tasks_path,
        tasks,
        lambda task, sample: (task.solution or "") if sample == 0 else wrong_program(task),
        sampling=Sampling(samples=2, temperature=1.0),
    )
    results = tmp_path / "results.jsonl"
    summary = execute_completions(completions, tasks_path, results, tokenizer=TOKENIZER, workers=4)
    assert summary["unresolved"] == 0 and summary["scored"] == 2 * len(tasks)
    parsed = read_results(results)
    statuses: dict[str, list[str]] = {}
    for task in tasks:
        statuses[task.id] = [parsed.attempts[(task.id, sample)].status for sample in (0, 1)]
    return statuses


def test_authored_dev_shaped_tasks_through_the_real_runner(tmp_path: Path) -> None:
    # 40 I/O cases: the flat 2 s default could not run them all; the deadline rule gives 12 s.
    tasks = [
        pool_task("dev/function"),
        io_task("dev/io", cases=40),
        pool_task(
            "dev/class",
            hidden_tests="assert Counter().add(2).total == 2",
            prompt="Write a Counter class.",
            public_tests="assert Counter().add(1).total == 1",
            solution=(
                "class Counter:\n    def __init__(self):\n        self.total = 0\n"
                "    def add(self, n):\n        self.total += n\n        return self"
            ),
        ),
    ]
    statuses = run_references_and_wrong_answers(tmp_path, tasks)
    assert statuses == {task.id: ["passed", "failed"] for task in tasks}


def test_executed_constant_baseline_through_the_real_runner() -> None:
    # Non-literal expectations: the reward channel cannot parse these, so constants run.
    task = pool_task(
        "dev/unparseable",
        hidden_tests=(
            "assert sq(0) == [i for i in range(0)]\n"
            "assert sq(1) == [i * i for i in range(1, 2)]\n"
            "assert sq(2) == [x * x for x in (1, 2)]\n"
        ),
        public_tests="assert sq(2) == [1, 4]",
    )
    report = pool_constant_baseline([task], workers=4)
    row = report["per_task"][0]
    assert row["method"] == "executed" and row["total"] == 3
    # Returning [] passes the first assert; returning the public [1, 4] passes the third.
    assert row["passed"] == 1 and report["summary"]["uncovered"] == 0


def short_real_dev_tasks() -> list[Task]:
    path = os.environ.get("SCRATCHCODER_POOL_DEV")
    if not path:
        pytest.skip("SCRATCHCODER_POOL_DEV is not set")
    chosen: dict[str, Task] = {}
    for task in load_tasks(Path(path)):
        if task.split != "dev":
            raise ValueError("SCRATCHCODER_POOL_DEV must be the dev split")
        if task.source not in chosen and task.solution and len(task.solution.encode()) <= 480:
            chosen[task.source] = task
    return list(chosen.values())


def test_real_dev_tasks_reference_pass_and_wrong_answers_fail(tmp_path: Path) -> None:
    tasks = short_real_dev_tasks()
    assert {task.mode for task in tasks} == {"function", "io"}
    io_cases = [len(json.loads(task.hidden_tests)) for task in tasks if task.mode == "io"]
    assert io_cases and max(io_cases) > 8
    statuses = run_references_and_wrong_answers(tmp_path, tasks)
    assert statuses == {task.id: ["passed", "failed"] for task in tasks}
