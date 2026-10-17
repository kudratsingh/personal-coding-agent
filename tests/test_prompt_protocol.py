import json
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch

from scratchcoder.common.config import ModelConfig
from scratchcoder.eval import command
from scratchcoder.eval.harness import EvaluationConfig, evaluate
from scratchcoder.eval.prompt import BASE_STOPS, base_prompt, prompt_protocol, truncate_completion
from scratchcoder.eval.tasks import Task, save_tasks
from scratchcoder.model.gpt import GPT
from scratchcoder.sandbox.runner import Result
from scratchcoder.tokenizer.bpe import ByteBPE


def fixture_task() -> Task:
    return Task(
        id="authored/0",
        prompt='def fixture():\n    """authored question"""\n \t\n',
        public_tests="public_marker\n",
        hidden_tests="hidden_marker",
        source="synthetic",
        license="fixture",
        written_by="human",
        split="eval",
        entry_point="fixture",
    )


def setup_generator(
    monkeypatch: pytest.MonkeyPatch, completion: str
) -> tuple[GPT, ByteBPE, list[str]]:
    tokenizer = ByteBPE()
    model = GPT(ModelConfig(vocab_size=272, width=16, heads=2, layers=1, context=256))
    seen: list[str] = []

    def generated(model: GPT, prompts: torch.Tensor, **options: Any) -> torch.Tensor:
        seen.append(tokenizer.decode(prompts[0].tolist()))
        tokens = tokenizer.encode(completion)
        assert len(tokens) <= options["max_new_tokens"]
        return torch.cat((prompts, torch.tensor([tokens])), dim=1)

    monkeypatch.setattr(command, "generate", generated)
    return model, tokenizer, seen


def test_project_base_generation_uses_stripped_raw_prompt_and_same_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion = "\n    return 3\n\ndef later(): pass"
    model, tokenizer, seen = setup_generator(monkeypatch, completion)
    task = fixture_task()
    candidate = command.checkpoint_generator(model, tokenizer, EvaluationConfig(max_new_tokens=64))(
        task, 7
    )
    assert seen == [task.prompt.rstrip()]
    assert "public_marker" not in seen[0] and "hidden_marker" not in seen[0]
    assert candidate == task.prompt.rstrip() + "\n    return 3\n"
    assert candidate == base_prompt(task.prompt) + truncate_completion(completion)


def test_explicit_instruction_mode_preserves_legacy_public_prompt_and_fences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion = "```python\ndef fixture(): return 3\n```"
    model, tokenizer, seen = setup_generator(monkeypatch, completion)
    task = fixture_task()
    candidate = command.checkpoint_generator(
        model, tokenizer, EvaluationConfig(max_new_tokens=64, prompt_mode="instruction")
    )(task, 7)
    assert seen == [task.public_prompt()]
    assert candidate == "def fixture(): return 3\n"
    assert prompt_protocol("instruction")["prompt_stripped"] is False


def test_project_report_and_scorecard_persist_the_prompt_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, tokenizer, _ = setup_generator(monkeypatch, "\n    return 3")
    monkeypatch.setattr(command, "load_checkpoint", lambda *_: (model, tokenizer))
    monkeypatch.setattr(command, "evaluate", partial(evaluate, execute=lambda *_: Result("passed")))
    checkpoint = tmp_path / "authored.pt"
    checkpoint.write_bytes(b"authored checkpoint identity fixture, not deserialized")
    tasks = tmp_path / "tasks.jsonl"
    save_tasks(tasks, [fixture_task()])
    command.evaluate_checkpoint(
        checkpoint=checkpoint,
        tokenizer_path=tmp_path / "unused-tokenizer.json",
        tasks_path=tasks,
        output=tmp_path / "evaluation",
        scorecard=tmp_path / "scorecard.jsonl",
        config=EvaluationConfig(max_new_tokens=64, bootstrap_samples=10),
    )
    for path in (tmp_path / "evaluation/report.json", tmp_path / "scorecard.jsonl"):
        report = json.loads(path.read_text())
        assert report["prompt_stripped"] is True
        assert report["protocol"] == prompt_protocol()["protocol"]
        assert report["configuration"]["prompt_mode"] == "base"


def test_base_protocol_preserves_leading_whitespace_and_original_stop_semantics() -> None:
    assert base_prompt("  def fixture():\n\t\u2003") == "  def fixture():"
    assert tuple(prompt_protocol()["protocol"]["stop_strings"]) == BASE_STOPS
    assert truncate_completion("\ndef later(): pass") == ""  # No stop-at-zero change.
