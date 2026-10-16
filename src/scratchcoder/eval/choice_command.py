"""Run local checkpoint choice scoring with the project's own tokenizer and model."""

from pathlib import Path

from scratchcoder.data.shards import file_sha256
from scratchcoder.engine.generation import score_answers
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.eval.choices import evaluate_choices, load_choices
from scratchcoder.eval.harness import ContextLimitError, save_report


def evaluate_choice_checkpoint(
    *,
    checkpoint: Path,
    tokenizer_path: Path,
    tasks_path: Path,
    output: Path,
    scorecard: Path,
    seed: int = 0,
) -> None:
    model, tokenizer = load_checkpoint(checkpoint, tokenizer_path)

    def score(prompt: str, labels: list[str]) -> list[float]:
        prefix = tokenizer.encode(prompt)
        answers = [tokenizer.encode(label) for label in labels]
        if len(prefix) + max(map(len, answers)) > model.config.context:
            raise ContextLimitError("complete question/choices exceed context")
        return score_answers(model, prefix, answers)

    report = evaluate_choices(
        list(load_choices(tasks_path)),
        score,
        checkpoint_hash=file_sha256(checkpoint),
        tokenizer_hash=tokenizer.sha256,
        seed=seed,
    )
    save_report(report, output, scorecard=scorecard)
