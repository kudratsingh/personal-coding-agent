"""Run local checkpoint choice scoring with the project's own tokenizer and model."""

from pathlib import Path

import torch

from scratchcoder.common.config import ModelConfig
from scratchcoder.data.shards import file_sha256
from scratchcoder.engine.generation import score_answers
from scratchcoder.eval.choices import evaluate_choices, load_choices
from scratchcoder.eval.harness import ContextLimitError, save_report
from scratchcoder.model.gpt import GPT
from scratchcoder.tokenizer.bpe import ByteBPE


def evaluate_choice_checkpoint(
    *,
    checkpoint: Path,
    tokenizer_path: Path,
    tasks_path: Path,
    output: Path,
    scorecard: Path,
    seed: int = 0,
) -> None:
    tokenizer = ByteBPE.load(tokenizer_path)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["tokenizer_hash"] != tokenizer.sha256:
        raise ValueError("checkpoint and tokenizer identities differ")
    model = GPT(ModelConfig(**saved["config"]["model"]))
    model.load_state_dict(saved["model"])
    model.eval()
    torch.set_num_threads(1)

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
