"""Seeded chat-v1 rollouts: the prompts, the per-row seeds and the generated attempts.

Every sampled row gets its own seed from ``scratchcoder.eval.harness.attempt_seed``, which
is ``(base + H(task) + index) mod (2^63 - 1)`` with ``H`` the first 64 bits of the task
ID's SHA-256. Three families of seeds use well-separated bases, so for one task they never
meet:

- dev checks: base ``DEV_SEED`` (0), index = sample (0 to 7);
- RL screening: base ``SCREENING_SEED`` (2^40), index = sample (0 to 7);
- RL rollouts: base ``ROLLOUT_SEED`` (2^41), index = ``(update - 1) * G + sample``.

For a fixed task, rollout indices differ for every (update, sample) pair, so its seeds are
distinct across the whole run by construction. Two different tasks collide only if their
64-bit hashes differ by less than the run's index range (under 1,000), a chance below
10^-10 for the rehearsal; the RL loop still checks every seed it uses and stops with an
error on a repeat. See ``docs/walkthroughs/rl-update.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from scratchcoder.data.chat import CODE_STOP_IDS, TextDecoder, TextEncoder, extract_program
from scratchcoder.data.chat import task_prompt as chat_task_prompt
from scratchcoder.engine.generation import generate_many
from scratchcoder.eval.harness import attempt_seed
from scratchcoder.eval.tasks import Task
from scratchcoder.model.gpt import GPT
from scratchcoder.rl.rewards import Attempt

DEV_SEED = 0
SCREENING_SEED = 2**40
ROLLOUT_SEED = 2**41
ROLLOUT_TOKENS = 256
"""RL rollouts and screening generate at most min(256, context - prompt) new tokens."""
DEV_TOKENS = 512
"""Dev checks generate at most min(512, context - prompt) new tokens."""


def screening_seed(task_id: str, sample: int, *, base: int = SCREENING_SEED) -> int:
    return attempt_seed(base, task_id, sample)


def rollout_seed(
    update: int, task_id: str, sample: int, *, group_size: int, base: int = ROLLOUT_SEED
) -> int:
    """The seed of sample ``sample`` of ``task_id`` in RL update ``update`` (1-based)."""
    if type(update) is not int or update < 1:
        raise ValueError("RL updates are numbered from 1")
    if type(sample) is not int or not 0 <= sample < group_size:
        raise ValueError("sample must lie in [0, group_size)")
    return attempt_seed(base, task_id, (update - 1) * group_size + sample)


@dataclass(frozen=True)
class Rollout:
    """One generated completion: the attempt sent for reward, plus what training needs.

    ``tokens`` are the generated IDs, the stop token included when one was produced.
    ``finish`` is ``stop`` or ``length`` (from the sampler), or ``context`` when the prompt
    left no room for any new token and nothing was generated.
    """

    attempt: Attempt
    prompt: tuple[int, ...]
    tokens: tuple[int, ...]
    finish: str
    seed: int


def prompts_for(encoder: TextEncoder, tasks: Sequence[Task]) -> list[list[int]]:
    """chat-v1 generation prompts: the task prompt, a blank line and the public test."""
    return [chat_task_prompt(encoder, task) for task in tasks]


def generate_rollouts(
    model: GPT,
    tasks: Sequence[Task],
    prompts: Sequence[Sequence[int]],
    decoder: TextDecoder,
    *,
    samples: int,
    seed_of: Callable[[Task, int], int],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
    precision: str,
) -> list[Rollout]:
    """``samples`` completions of each task, task-major, through ``generate_many``.

    Row ``(task, sample)`` samples with ``seed_of(task, sample)`` and stops at
    ``<|python_end|>`` or ``<|assistant_end|>``. Its budget is
    ``min(max_new_tokens, context - len(prompt))``. A prompt that fills the context gets
    no generation: its rollouts are empty, carry no program and score 0. The program is
    the first closed python block (``extract_program``).
    """
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    if len(prompts) != len(tasks):
        raise ValueError("one prompt per task")
    context = model.config.context
    rows: list[tuple[int, int]] = []
    for position in range(len(tasks)):
        if len(prompts[position]) < context:
            rows.extend((position, sample) for sample in range(samples))
    seeds = {
        (position, sample): seed_of(tasks[position], sample)
        for position in range(len(tasks))
        for sample in range(samples)
    }
    completions = (
        generate_many(
            model,
            [prompts[position] for position, _ in rows],
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seeds=[seeds[row] for row in rows] if temperature > 0 else None,
            stops=[CODE_STOP_IDS] * len(rows),
            precision=precision,
        )
        if rows
        else []
    )
    generated = dict(zip(rows, completions, strict=True))
    rollouts: list[Rollout] = []
    for position, task in enumerate(tasks):
        prompt = tuple(prompts[position])
        for sample in range(samples):
            completion = generated.get((position, sample))
            tokens = () if completion is None else tuple(completion.tokens)
            code = None if completion is None else extract_program(tokens, decoder)
            rollouts.append(
                Rollout(
                    Attempt(task.id, sample, code, len(tokens)),
                    prompt,
                    tokens,
                    "context" if completion is None else completion.finish_reason,
                    seeds[(position, sample)],
                )
            )
    return rollouts


def screening_rollouts(
    model: GPT,
    tasks: Sequence[Task],
    encoder: TextEncoder,
    decoder: TextDecoder,
    *,
    batch_size: int,
    precision: str = "float32",
    samples: int = 8,
    seed: int = SCREENING_SEED,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = ROLLOUT_TOKENS,
) -> list[Rollout]:
    """RL screening generation: 8 samples per candidate from the SFT checkpoint.

    Defaults are the screening protocol: temperature 1.0, top-p 1.0, at most
    min(256, context - prompt) new tokens, the chat-v1 prompt with the public test, and
    the seeds ``attempt_seed(SCREENING_SEED, task, sample)``. Send
    ``[rollout.attempt for rollout in ...]`` for rewards, then apply
    ``scratchcoder.rl.screening.screen``.
    """
    return generate_rollouts(
        model,
        tasks,
        prompts_for(encoder, tasks),
        decoder,
        samples=samples,
        seed_of=lambda task, sample: screening_seed(task.id, sample, base=seed),
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        precision=precision,
    )
