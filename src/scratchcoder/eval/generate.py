"""Generate a completions file: checkpoint + suite + protocol + sampling settings.

Generation is the GPU half of evaluation. It builds each task's prompt under the protocol,
runs ``scratchcoder.engine.generation.generate_many`` in length-sorted batches, applies
the protocol's extraction, and writes one hashed completions file. Nothing is executed.
Each attempt's seed is ``harness.attempt_seed(seed, task_id, sample)``, so a sampled
attempt draws the same random numbers whatever batch it lands in.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from scratchcoder.common.records import git_state, utc_now
from scratchcoder.data.shards import file_sha256
from scratchcoder.engine.generation import PRECISIONS, SAMPLER_VERSION, generate_many
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.eval.completions import (
    CompletionRow,
    CompletionsHeader,
    Sampling,
    prompt_sha256,
    task_ids_sha256,
    write_completions,
)
from scratchcoder.eval.protocols import load_suite_tasks, protocol_for
from scratchcoder.tokenizer.bulk import FastEncoder
from scratchcoder.train.trainer import training_device

DEVICES = "cpu, cuda, cuda:N or mps"


def generation_device(value: str) -> torch.device:
    """Resolve ``cpu``, ``cuda``, ``cuda:N`` or ``mps`` with no fallback to another device."""
    if value == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable; select cpu explicitly to use CPU")
        return torch.device("mps")
    try:
        return training_device(value)
    except ValueError as error:
        raise ValueError(f"generation device must be {DEVICES}: {error}") from error


def _synchronized(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    return time.perf_counter()


def generate_completions(
    *,
    checkpoint: Path,
    tokenizer_path: Path,
    suite: str,
    split: str,
    tasks_path: Path,
    protocol: str,
    output: Path,
    sampling: Sampling | None = None,
    device: str = "cpu",
    precision: str = "float32",
    batch_size: int = 64,
    generate: Callable[..., Any] = generate_many,
) -> tuple[CompletionsHeader, str]:
    """Write ``output`` and return its header and SHA-256.

    An attempt whose prompt leaves no room to generate is recorded as ``context_limit``
    with no tokens and no program; it is never shortened. MPS runs greedy decoding only.
    """
    sampling = sampling or Sampling()
    if precision not in PRECISIONS:
        raise ValueError("precision must be float32 or bfloat16")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if output.exists():
        raise FileExistsError(f"{output} already exists; completions files are immutable")
    tasks, exclusions = load_suite_tasks(suite, split, tasks_path)
    definition = protocol_for(protocol, suite)
    resolved = generation_device(device)
    if resolved.type == "mps" and not sampling.greedy:
        raise ValueError(
            "sampled decoding is not supported on MPS: the seeded sampler works in float64, "
            "which MPS lacks; use cpu or cuda for sampling, or greedy decoding on mps"
        )
    model, tokenizer = load_checkpoint(
        checkpoint, tokenizer_path, device="cpu" if resolved.type == "mps" else device
    )
    if resolved.type == "mps":
        model.to(resolved)
    context = model.config.context
    # The cached encoder gives the reference encoder's IDs exactly, in milliseconds a prompt
    # rather than seconds at 65,536 tokens (2,000 pool prompts would take most of an hour).
    encoder = FastEncoder(tokenizer)
    prompts = {task.id: definition.prompt_ids(encoder, task) for task in tasks}
    attempts = [
        (task, sample, seed)
        for task in tasks
        for sample, seed in enumerate(sampling.seeds(task.id))
    ]
    runnable = [
        index
        for index, (task, _, _) in enumerate(attempts)
        if definition.budget(len(prompts[task.id]), context) >= 1
    ]
    started = _synchronized(resolved)
    generated = []
    if runnable:
        generated = generate(
            model,
            [prompts[attempts[index][0].id] for index in runnable],
            batch_size=batch_size,
            max_new_tokens=definition.cap,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            seeds=None if sampling.greedy else [attempts[index][2] for index in runnable],
            stops=[definition.stop_ids] * len(runnable) if definition.stop_ids else None,
            precision=precision,
        )
    seconds = _synchronized(resolved) - started
    by_attempt = dict(zip(runnable, generated, strict=True))
    rows = []
    for index, (task, sample, seed) in enumerate(attempts):
        prompt = prompts[task.id]
        budget = definition.budget(len(prompt), context)
        completion = by_attempt.get(index)
        if completion is None:
            tokens: tuple[int, ...] = ()
            text, program, reason = "", None, "context_limit"
        else:
            if completion.budget != budget:
                raise RuntimeError("generation budget differs from the protocol budget")
            tokens = tuple(completion.tokens)
            text = tokenizer.decode(tokens, skip_special=True)
            program = definition.candidate(task, tokens=tokens, completion=text, decoder=tokenizer)
            reason = completion.finish_reason
        rows.append(
            CompletionRow(
                task_id=task.id,
                sample=sample,
                seed=seed,
                prompt_tokens=len(prompt),
                prompt_sha256=prompt_sha256(prompt),
                tokens=tokens,
                completion=text,
                program=program,
                stop_reason=reason,
                budget=budget,
            )
        )
    commit, dirty = git_state()
    header = CompletionsHeader(
        suite=suite,
        split=split,
        protocol=definition.name,
        protocol_sha256=definition.sha256(),
        protocol_record=definition.record(),
        checkpoint_sha256=file_sha256(checkpoint),
        tokenizer_sha256=tokenizer.sha256,
        tokenizer_vocab_size=tokenizer.vocab_size,
        model_context=context,
        task_file_sha256=file_sha256(tasks_path),
        task_count=len(tasks),
        task_ids_sha256=task_ids_sha256(task.id for task in tasks),
        sampling=sampling,
        max_new_tokens=definition.cap,
        precision=precision,
        device=str(resolved),
        rows=len(rows),
        created_at=utc_now(),
        excluded_tasks=tuple(exclusions),
        sampler_version=None if sampling.greedy else SAMPLER_VERSION,
        device_name=torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else "",
        batch_size=batch_size,
        generation_seconds=seconds,
        git_commit=commit,
        dirty=dirty,
    )
    return header, write_completions(output, header, rows)
