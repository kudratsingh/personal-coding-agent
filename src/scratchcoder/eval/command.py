"""Evaluate an explicit local checkpoint and normalized held-out task file.

This is the legacy ``scratchcoder eval`` command. Its two prompt modes encode plain text,
so they fit base checkpoints only (``scratchcoder.eval.interface``). Chat-v1 checkpoints
are evaluated with ``eval-generate`` and ``eval-execute``.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from scratchcoder.data.shards import file_sha256
from scratchcoder.engine.generation import SAMPLER_VERSION, generate
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.eval.harness import ContextLimitError, EvaluationConfig, evaluate, save_report
from scratchcoder.eval.interface import read_checkpoint_interface, resolve_legacy_mode
from scratchcoder.eval.prompt import base_prompt, prompt_protocol, truncate_completion
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
    prompt_mode: str | None,
    allow_interface_mismatch: bool = False,
) -> None:
    """Evaluate ``checkpoint`` on ``tasks_path`` under a legacy prompt mode.

    ``prompt_mode`` is ``base``, ``instruction`` or ``None``; ``None`` takes the mode from
    the checkpoint's interface (``resolve_legacy_mode``), and a requested mode must fit it
    unless ``allow_interface_mismatch``. The resolved mode replaces ``config.prompt_mode``.
    Both checks run before the weights are loaded.
    """
    interface = read_checkpoint_interface(checkpoint)
    mode = resolve_legacy_mode(interface, prompt_mode, allow_mismatch=allow_interface_mismatch)
    config = replace(config, prompt_mode=mode)
    tasks = list(load_tasks(tasks_path))
    model, tokenizer = load_checkpoint(checkpoint, tokenizer_path)
    report = evaluate(
        tasks,
        checkpoint_generator(model, tokenizer, config),
        config=config,
        checkpoint_hash=file_sha256(checkpoint),
        tokenizer_hash=tokenizer.sha256,
    )
    report.update(prompt_protocol(legacy_record_mode(mode, tasks)))
    report.update(sampler_record(config))
    report["checkpoint_interface"] = {
        **interface.record(),
        "prompt_mode_source": "requested" if prompt_mode is not None else "checkpoint interface",
        "interface_mismatch_allowed": allow_interface_mismatch,
    }
    save_report(report, output, scorecard=scorecard)


def legacy_record_mode(mode: str, tasks: Sequence[Task]) -> str:
    """The ``prompt_protocol`` mode that names what the legacy command ran.

    Base mode on benchmark tasks (split ``eval``) is ``base-completion-rstrip-v1``. On any
    other task file it is the same procedure on pool tasks, recorded as
    ``legacy-pool-rstrip-v1`` so it is never mistaken for that benchmark protocol or for
    ``base-pool-completion-v1``.
    """
    if mode == "base" and any(task.split != "eval" for task in tasks):
        return "legacy-pool"
    return mode


def sampler_record(config: EvaluationConfig) -> dict[str, Any]:
    """Name the seeded sampler of a sampled run; greedy decoding uses no sampler."""
    return {"sampler_version": SAMPLER_VERSION} if config.temperature > 0 else {}


def checkpoint_generator(
    model: GPT, tokenizer: ByteBPE, config: EvaluationConfig
) -> Callable[[Task, int], str]:
    def candidate(task: Task, seed: int) -> str:
        prefix = base_prompt(task.prompt) if config.prompt_mode == "base" else task.public_prompt()
        prompt = tokenizer.encode(prefix)
        remaining = model.config.context - len(prompt)
        if remaining < 1:
            raise ContextLimitError("prompt exceeds configured context")
        generated = generate(
            model,
            torch.tensor([prompt], device=next(model.parameters()).device),
            max_new_tokens=min(config.max_new_tokens, remaining),
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            seed=seed,
        )
        text = tokenizer.decode(generated[0, len(prompt) :].tolist(), skip_special=True)
        if config.prompt_mode == "base":
            return prefix + truncate_completion(text)
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
