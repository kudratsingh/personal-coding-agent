import os
from pathlib import Path

import pytest

from scratchcoder.data.pool import build_pool
from scratchcoder.eval.harness import EvaluationConfig, evaluate, isolated_execution, save_report
from scratchcoder.eval.tasks import Task

pytestmark = pytest.mark.skipif(
    os.environ.get("SCRATCHCODER_SANDBOX_TESTS") != "1",
    reason="requires an explicitly enabled disposable Docker worker",
)


def test_reference_pool_and_scorecard_through_real_isolation(tmp_path: Path) -> None:
    task = Task(
        id="synthetic-add",
        prompt="Implement add(a, b)",
        public_tests="assert add(1,2) == 3",
        hidden_tests="assert add(12,30) == 42",
        source="synthetic",
        license="fixture",
        written_by="human",
        split="test",
        solution="def add(a,b): return a+b",
    )
    kept, _ = build_pool([task], [], execute=isolated_execution)
    assert kept == [task]
    report = evaluate(
        kept,
        lambda *_: "def add(a,b): return a+b",
        config=EvaluationConfig(bootstrap_samples=10),
        checkpoint_hash="synthetic-candidate",
        tokenizer_hash="synthetic-tokenizer",
    )
    assert report["metrics"]["greedy_pass_at_1"] == 1
    save_report(report, tmp_path / "evaluation", scorecard=tmp_path / "scorecard.jsonl")
    failed = evaluate(
        kept,
        lambda *_: "def add(a,b): return 0",
        config=EvaluationConfig(bootstrap_samples=10),
        checkpoint_hash="synthetic-failure",
        tokenizer_hash="synthetic-tokenizer",
    )
    assert failed["metrics"]["greedy_pass_at_1"] == 0
