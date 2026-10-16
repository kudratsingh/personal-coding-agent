import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from scratchcoder.common.config import ModelConfig
from scratchcoder.data.pool import build_pool
from scratchcoder.eval.command import evaluate_checkpoint
from scratchcoder.eval.harness import EvaluationConfig, evaluate, isolated_execution, save_report
from scratchcoder.eval.tasks import Task, save_tasks
from scratchcoder.model.gpt import GPT
from scratchcoder.tokenizer.bpe import ByteBPE

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


def test_local_checkpoint_code_evaluation_end_to_end(tmp_path: Path) -> None:
    tokenizer = ByteBPE()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    config = ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=256)
    model = GPT(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"model": asdict(config)},
            "tokenizer_hash": tokenizer.sha256,
        },
        checkpoint,
    )
    tasks = tmp_path / "tasks.jsonl"
    save_tasks(
        tasks,
        [
            Task(
                id="synthetic",
                prompt="def add(a,b):\n",
                public_tests="assert add(1,2)==3",
                hidden_tests="assert add(2,3)==5",
                source="synthetic",
                license="fixture",
                written_by="human",
                split="eval",
                entry_point="add",
            )
        ],
    )
    evaluate_checkpoint(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_path,
        tasks_path=tasks,
        output=tmp_path / "evaluation",
        scorecard=tmp_path / "scores.jsonl",
        config=EvaluationConfig(max_new_tokens=4, bootstrap_samples=10),
    )
    report = json.loads((tmp_path / "evaluation/report.json").read_text())
    assert report["metrics"]["greedy_pass_at_1"] == 0
    assert report["per_task"][0]["attempts"][0]["status"] == "failed"
