"""Run local checkpoint choice scoring with the project's own tokenizer and model."""

from collections.abc import Callable
from pathlib import Path

from scratchcoder.data.shards import file_sha256
from scratchcoder.engine.generation import score_answers
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.eval.choices import evaluate_choices, load_choices
from scratchcoder.eval.harness import ContextLimitError, save_report
from scratchcoder.model.gpt import GPT
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.tokenizer.bulk import FastEncoder


def checkpoint_choice_scorer(
    model: GPT, tokenizer: ByteBPE
) -> Callable[[str, list[str]], list[float]]:
    """The existing fixed label protocol, shared by local and verified-suite evaluation.

    Text is encoded with the cached ``FastEncoder``, which gives ``tokenizer.encode``'s IDs
    exactly (``tests/test_tokenizer.py``); the reference encoder applies every merge to every
    chunk and takes about a third of a second per ARC question at 65,536 tokens.
    """
    encoder = FastEncoder(tokenizer)

    def score(prompt: str, labels: list[str]) -> list[float]:
        prefix = encoder.encode(prompt)
        answers = [encoder.encode(label) for label in labels]
        if len(prefix) + max(map(len, answers)) > model.config.context:
            raise ContextLimitError("complete question/choices exceed context")
        return score_answers(model, prefix, answers)

    return score


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

    report = evaluate_choices(
        list(load_choices(tasks_path)),
        checkpoint_choice_scorer(model, tokenizer),
        checkpoint_hash=file_sha256(checkpoint),
        tokenizer_hash=tokenizer.sha256,
        seed=seed,
    )
    save_report(report, output, scorecard=scorecard)
