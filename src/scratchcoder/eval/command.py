"""Evaluate an explicit local checkpoint and normalized held-out task file."""

import re
from collections.abc import Callable
from pathlib import Path

import torch

from scratchcoder.data.shards import file_sha256
from scratchcoder.engine.generation import generate
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.eval.harness import ContextLimitError, EvaluationConfig, evaluate, save_report
from scratchcoder.eval.tasks import Task, load_tasks
from scratchcoder.model.gpt import GPT
from scratchcoder.tokenizer.bpe import ByteBPE


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    tokenizer_path: Path,
    tasks_path: Path,
    output: Path,
    scorecard: Path,
    config: EvaluationConfig,
) -> None:
    model, tokenizer = load_checkpoint(checkpoint, tokenizer_path)

    tasks = list(load_tasks(tasks_path))
    report = evaluate(
        tasks,
        checkpoint_generator(model, tokenizer, config),
        config=config,
        checkpoint_hash=file_sha256(checkpoint),
        tokenizer_hash=tokenizer.sha256,
    )
    save_report(report, output, scorecard=scorecard)


def checkpoint_generator(
    model: GPT, tokenizer: ByteBPE, config: EvaluationConfig
) -> Callable[[Task, int], str]:
    def candidate(task: Task, seed: int) -> str:
        prompt = tokenizer.encode(task.public_prompt())
        remaining = model.config.context - len(prompt)
        if remaining < 1:
            raise ContextLimitError("prompt exceeds configured context")
        generated = generate(
            model,
            torch.tensor([prompt]),
            max_new_tokens=min(config.max_new_tokens, remaining),
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            seed=seed,
        )
        text = tokenizer.decode(generated[0, len(prompt) :].tolist(), skip_special=True)
        fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
        code = fence.group(1) if fence else text.strip("\r\n")
        if (
            task.entry_point
            and f"def {task.entry_point}" in task.prompt
            and f"def {task.entry_point}" not in code
        ):
            code = task.prompt + code
        return code

    return candidate
