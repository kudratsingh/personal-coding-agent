import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from scratchcoder.cli import main
from scratchcoder.common.config import ModelConfig
from scratchcoder.data.manifest import ManifestEntry
from scratchcoder.eval.choices import ChoiceTask, adapt_arc_row, evaluate_choices, load_choices
from scratchcoder.eval.harness import ContextLimitError
from scratchcoder.model.gpt import GPT
from scratchcoder.tokenizer.bpe import ByteBPE


def fixture_task() -> ChoiceTask:
    return ChoiceTask(
        id="fixture/1",
        question="Which number is even?",
        labels=("A", "B"),
        choices=("3", "4"),
        answer="B",
        source="synthetic",
        source_revision="a" * 40,
        license="fixture",
    )


def test_choice_metrics_ties_order_hash_and_key_isolation() -> None:
    first = fixture_task()
    second = replace(first, id="fixture/2", answer="A")
    assert first.public_prompt() == replace(first, answer="A").public_prompt()

    def score(prompt: str, labels: list[str]) -> list[float]:
        assert prompt.endswith("Answer:") and labels == [" A", " B"]
        return [-1.0, -1.0]  # deterministic first-listed tie, not answer-aware

    report = evaluate_choices(
        [second, first], score, checkpoint_hash="model", tokenizer_hash="tok", bootstrap_samples=20
    )
    assert report["metrics"]["accuracy"] == 0.5
    assert report["metrics"]["confidence_interval"]["mean"] == 0.5
    assert [row["prediction"] for row in report["per_task"]] == ["A", "A"]
    repeated = evaluate_choices(
        [first, second], score, checkpoint_hash="model", tokenizer_hash="tok", bootstrap_samples=20
    )
    assert repeated["task_hash"] == report["task_hash"]
    assert repeated["per_task"] == report["per_task"]


@pytest.mark.parametrize(
    "changes",
    [
        {"labels": ("A", "A")},
        {"answer": "C"},
        {"split": "train"},
        {"source_revision": "main"},
        {"choices": ("one",)},
        {"labels": ("A\n", "B")},
    ],
)
def test_choice_schema_rejects_invalid_fields(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(fixture_task(), **changes)  # type: ignore[arg-type]


@pytest.mark.parametrize("scores", [[0.0], [float("nan"), 0.0], [0.0, float("inf")]])
def test_choice_scorer_rejects_bad_results(scores: list[float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_choices(
            [fixture_task()], lambda *_: scores, checkpoint_hash="m", tokenizer_hash="t"
        )


def test_choice_context_limit_is_counted_not_truncated() -> None:
    def score(prompt: str, labels: list[str]) -> list[float]:
        raise ContextLimitError("oversized")

    report = evaluate_choices([fixture_task()], score, checkpoint_hash="m", tokenizer_hash="t")
    assert report["metrics"]["accuracy"] == 0
    assert report["per_task"][0]["status"] == "context_limit"


def test_choice_loading_and_arc_mapping_require_eval_provenance(tmp_path: Path) -> None:
    task = fixture_task()
    path = tmp_path / "choices.jsonl"
    path.write_text(json.dumps(asdict(task)) + "\n")
    assert list(load_choices(path)) == [task]
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate"):
        list(load_choices(path))
    provenance = ManifestEntry(
        source="allenai/ai2_arc",
        revision="a" * 40,
        url="https://example.invalid/data",
        license="CC-BY-SA-4.0",
        attribution="synthetic mapping fixture",
        sha256="b" * 64,
        size_bytes=0,
        purpose="eval",
        license_verified=True,
    )
    row = {
        "id": "fixture",
        "question": task.question,
        "choices": {"label": list(task.labels), "text": list(task.choices)},
        "answerKey": "B",
    }
    assert adapt_arc_row(row, provenance).public_prompt() == task.public_prompt()
    with pytest.raises(ValueError):
        adapt_arc_row(row, replace(provenance, purpose="train"))


def test_choice_checkpoint_cli_end_to_end(tmp_path: Path) -> None:
    tokenizer = ByteBPE.train(["synthetic fixture"], vocab_size=272)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    config = ModelConfig(vocab_size=272, context=256, width=16, layers=1, heads=2)
    model = GPT(config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"model": asdict(config)},
            "tokenizer_hash": tokenizer.sha256,
        },
        checkpoint,
    )
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(asdict(fixture_task())) + "\n")
    output = tmp_path / "evaluation"
    scorecard = tmp_path / "scorecard.jsonl"
    assert (
        main(
            [
                "eval-choice",
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                str(tokenizer_path),
                "--tasks",
                str(tasks),
                "--output",
                str(output),
                "--scorecard",
                str(scorecard),
            ]
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text())
    assert report["task_count"] == 1 and report["metrics"]["accuracy"] in {0.0, 1.0}
    assert report["per_task"][0]["status"] == "scored"
    assert json.loads(scorecard.read_text())["task_hash"] == report["task_hash"]


def test_checkpoint_scoring_uses_the_cached_encoder_with_the_reference_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scratchcoder.engine.generation import score_answers
    from scratchcoder.eval.choice_command import checkpoint_choice_scorer
    from scratchcoder.tokenizer.bpe import FIRST_MERGE

    prompt, labels = "Question: which colour is the sky?\nAnswer:", [" A", " B", " C"]
    merged = ByteBPE.train([prompt * 3, " A B C"], FIRST_MERGE + 24)
    assert merged.merges
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=merged.vocab_size, width=16, layers=1, heads=2, context=64))
    model.eval()
    expected = score_answers(
        model, merged.encode(prompt), [merged.encode(label) for label in labels]
    )

    def reference(self: ByteBPE, text: str, *, allow_special: bool = False) -> list[int]:
        raise AssertionError("the reference encoder ran on the scoring path")

    monkeypatch.setattr(ByteBPE, "encode", reference)
    assert checkpoint_choice_scorer(model, merged)(prompt, labels) == expected
