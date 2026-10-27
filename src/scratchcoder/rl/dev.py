"""Dev checks during RL and the more-SFT control, and the checkpoint selection rule.

A dev check scores the current policy on a fixed 200-task subset of the pool's dev split:

- **Subset:** the 200 dev tasks with the smallest ``sha256("rl-dev-subset-v1|" + task ID)``,
  in that order (``dev_subset``).
- **Generation:** the chat-v1 prompt with the public test; one greedy completion and
  ``n = 8`` samples at temperature 0.8 and top-p 0.95, sample ``i`` seeded
  ``attempt_seed(0, task ID, i)``; at most min(512, context - prompt) new tokens; stops at
  ``<|python_end|>`` or ``<|assistant_end|>``; through ``generate_many``
  (``dev_generations``).
- **Execution:** the first closed python block of each completion goes to an injected
  executor, which returns pass or fail for each ``(task, code)`` pair. A completion without
  a program, or a prompt that fills the context, fails without running.
- **Metrics:** greedy pass@1, the mean pass rate of the 8 samples, and pass@8, each over all
  subset tasks.
- **Constant baseline:** beside them, the subset's mean ``degenerate_baseline`` reward (the
  best constant answer's fraction of hidden tests) and, in the units of the pass rates
  above, the constant pass rate: the share of subset tasks on which the best constant answer
  passes every hidden test. The two pass rates and the pass@1 compare like with like; the
  mean reward compares with a model's mean reward (the per-update monitoring). A dev task
  the reward channel cannot parse counts as 0 in both (the baseline is a lower bound there),
  and the number of such tasks is recorded.

``select_check`` picks the check with the highest greedy pass@1; ties go to the higher
8-sample mean, then to the later check. See ``docs/rl.md``.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch

from scratchcoder.data.chat import CODE_STOP_IDS, TextDecoder, TextEncoder, extract_program
from scratchcoder.engine.generation import generate_many
from scratchcoder.eval.harness import attempt_seed
from scratchcoder.eval.metrics import pass_at_k
from scratchcoder.eval.tasks import Task
from scratchcoder.model.gpt import GPT
from scratchcoder.rl.rollout import DEV_SEED, DEV_TOKENS, prompts_for
from scratchcoder.rl.screening import baseline_of
from scratchcoder.rl.withheld import refuse_withheld

DEV_SUBSET_SALT = "rl-dev-subset-v1"
DEV_SUBSET_SIZE = 200
DEV_SAMPLES = 8
DEV_TEMPERATURE = 0.8
DEV_TOP_P = 0.95

DevExecutor = Callable[[Sequence[tuple[Task, str]]], Sequence[bool]]
"""Runs each ``(task, program)`` against the task's hidden tests; one pass flag per pair."""


def dev_subset(tasks: Sequence[Task], n: int = DEV_SUBSET_SIZE) -> list[Task]:
    """The fixed dev subset: the ``n`` dev tasks with the smallest
    ``sha256(f"rl-dev-subset-v1|{task.id}")``, in that order."""
    if type(n) is not int or n < 1:
        raise ValueError("the subset size must be a positive integer")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("dev task IDs must be unique")
    if any(task.split != "dev" for task in tasks):
        raise ValueError("the RL dev subset is drawn from the dev split only")
    if n > len(tasks):
        raise ValueError(f"{n} dev tasks requested from {len(tasks)}")
    return sorted(
        tasks, key=lambda task: hashlib.sha256(f"{DEV_SUBSET_SALT}|{task.id}".encode()).digest()
    )[:n]


@dataclass(frozen=True)
class ConstantBaseline:
    """The best constant answer over a task set (unparseable tasks counted as 0).

    ``mean`` is its mean reward (the fraction of hidden tests it passes, averaged over
    tasks); ``pass_rate`` is the share of tasks on which it passes every hidden test, the
    unit of pass@1."""

    mean: float
    unparseable: int
    tasks: int
    pass_rate: float = 0.0
    solved: int = 0


def constant_baseline(tasks: Sequence[Task]) -> ConstantBaseline:
    """Mean ``degenerate_baseline`` reward and the share of tasks it fully passes; a task
    the channel cannot parse counts as 0 and as not passed."""
    if not tasks:
        raise ValueError("a constant baseline needs at least one task")
    values: list[float] = []
    unparseable = solved = 0
    for task in tasks:
        baseline = baseline_of(task)
        if baseline.reward is None:
            unparseable += 1
            values.append(0.0)
        else:
            values.append(baseline.reward)
            solved += baseline.reward >= 1.0
    return ConstantBaseline(
        math.fsum(values) / len(values), unparseable, len(tasks), solved / len(tasks), solved
    )


@dataclass(frozen=True)
class DevTaskResult:
    task_id: str
    greedy_passed: bool
    sample_passes: int


@dataclass(frozen=True)
class DevCheck:
    """One dev check. ``check`` is the RL update (0 before the first) or the control's
    optimizer step it was taken after. Rates are over all ``tasks``. Seconds are split into
    generation (GPU) and waiting for execution (the Mac)."""

    check: int
    greedy_pass_at_1: float
    mean_pass_n8: float
    pass_at_8: float
    constant_baseline: float
    tasks: int
    generation_seconds: float
    wait_seconds: float
    samples: int = DEV_SAMPLES
    context_limited: int = 0
    no_program: int = 0
    baseline_unparseable: int = 0
    constant_pass_rate: float | None = None
    """The share of dev tasks on which the best constant answer passes every hidden test:
    the baseline in the units of the pass rates above (``None`` when not recorded).
    ``constant_baseline`` is the mean fraction of hidden tests instead."""
    per_task: tuple[DevTaskResult, ...] = ()

    def __post_init__(self) -> None:
        if type(self.check) is not int or self.check < 0:
            raise ValueError("a dev check index is a nonnegative integer")
        if type(self.tasks) is not int or self.tasks < 1:
            raise ValueError("a dev check covers at least one task")
        for name in ("greedy_pass_at_1", "mean_pass_n8", "pass_at_8", "constant_baseline"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        rate = self.constant_pass_rate
        if rate is not None and (not isinstance(rate, (int, float)) or not 0 <= rate <= 1):
            raise ValueError("constant_pass_rate must lie in [0, 1] or be None")
        for name in ("generation_seconds", "wait_seconds"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def to_json(self, *, per_task: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["per_task"] = [asdict(row) for row in self.per_task] if per_task else []
        return value


class DevEvaluator(Protocol):
    """Scores a policy on the fixed dev subset; ``check`` labels the result."""

    def evaluate(self, model: GPT, check: int) -> DevCheck: ...


@dataclass(frozen=True)
class DevGeneration:
    """One task's dev programs: greedy, then the ``n`` samples (``None`` = no program)."""

    task_id: str
    greedy: str | None
    samples: tuple[str | None, ...]
    context_limited: bool


def _synchronized(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def dev_generations(
    model: GPT,
    tasks: Sequence[Task],
    encoder: TextEncoder,
    decoder: TextDecoder,
    *,
    batch_size: int,
    precision: str = "float32",
    samples: int = DEV_SAMPLES,
    temperature: float = DEV_TEMPERATURE,
    top_p: float = DEV_TOP_P,
    seed: int = DEV_SEED,
    max_new_tokens: int = DEV_TOKENS,
) -> list[DevGeneration]:
    """The generation half of a dev check: greedy plus ``samples`` seeded samples per task.

    A prompt that fills the model's context gets no generation and no programs; it fails
    every attempt and stays in the denominator, as a context limit does in evaluation.
    """
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    prompts = prompts_for(encoder, tasks)
    fits = [index for index, prompt in enumerate(prompts) if len(prompt) < model.config.context]
    greedy: dict[int, str | None] = {}
    sampled: dict[tuple[int, int], str | None] = {}
    if fits:
        stops = [CODE_STOP_IDS] * len(fits)
        completions = generate_many(
            model,
            [prompts[index] for index in fits],
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            stops=stops,
            precision=precision,
        )
        for index, completion in zip(fits, completions, strict=True):
            greedy[index] = extract_program(completion.tokens, decoder)
        rows = [(index, sample) for index in fits for sample in range(samples)]
        completions = generate_many(
            model,
            [prompts[index] for index, _ in rows],
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seeds=[attempt_seed(seed, tasks[index].id, sample) for index, sample in rows],
            stops=[CODE_STOP_IDS] * len(rows),
            precision=precision,
        )
        for row, completion in zip(rows, completions, strict=True):
            sampled[row] = extract_program(completion.tokens, decoder)
    return [
        DevGeneration(
            task.id,
            greedy.get(index),
            tuple(sampled.get((index, sample)) for sample in range(samples)),
            index not in greedy,
        )
        for index, task in enumerate(tasks)
    ]


def score_generations(
    tasks: Sequence[Task],
    generations: Sequence[DevGeneration],
    execute: DevExecutor,
) -> tuple[list[DevTaskResult], int]:
    """Execute every program once, in one batch; return per-task results and how many
    attempts had no program (context-limited tasks included)."""
    if [task.id for task in tasks] != [row.task_id for row in generations]:
        raise ValueError("generations must align with the tasks")
    pairs: list[tuple[Task, str]] = []
    slots: list[tuple[int, int]] = []  # (task position, -1 for greedy or sample index)
    missing = 0
    for position, (task, row) in enumerate(zip(tasks, generations, strict=True)):
        for slot, code in [(-1, row.greedy), *enumerate(row.samples)]:
            if code is None:
                missing += 1
            else:
                pairs.append((task, code))
                slots.append((position, slot))
    passed = list(execute(pairs)) if pairs else []
    if len(passed) != len(pairs) or any(type(flag) is not bool for flag in passed):
        raise ValueError("the dev executor must return one boolean per program")
    greedy = [False] * len(tasks)
    sample_passes = [0] * len(tasks)
    for (position, slot), flag in zip(slots, passed, strict=True):
        if slot < 0:
            greedy[position] = flag
        else:
            sample_passes[position] += flag
    results = [
        DevTaskResult(task.id, greedy[position], sample_passes[position])
        for position, task in enumerate(tasks)
    ]
    return results, missing


def dev_metrics(results: Sequence[DevTaskResult], samples: int) -> dict[str, float]:
    """Greedy pass@1, the n-sample mean pass rate and pass@n, each over all tasks.

    The mean is total passes over ``n x tasks``, so equal pass counts give equal floats.
    """
    if not results:
        raise ValueError("dev metrics need at least one task")
    tasks = len(results)
    return {
        "greedy_pass_at_1": sum(row.greedy_passed for row in results) / tasks,
        "mean_pass_n8": sum(row.sample_passes for row in results) / (samples * tasks),
        "pass_at_8": math.fsum(pass_at_k(samples, row.sample_passes, samples) for row in results)
        / tasks,
    }


class PoolDevEvaluator:
    """A ``DevEvaluator`` over fixed dev tasks: ``dev_generations`` on this machine, then
    the injected ``execute`` for pass or fail (a local runner, or the Mac bridge)."""

    def __init__(
        self,
        tasks: Sequence[Task],
        encoder: TextEncoder,
        decoder: TextDecoder,
        execute: DevExecutor,
        *,
        batch_size: int = 256,
        precision: str | None = None,
        samples: int = DEV_SAMPLES,
        baseline: ConstantBaseline | None = None,
    ) -> None:
        """``precision=None`` follows the model's device: bfloat16 on CUDA, float32 otherwise.

        ``baseline`` is the constant baseline the Mac computed from the real hidden tests
        (``scratchcoder.rl.facts.constant_baseline_from_facts``), for a Pod that holds none;
        it must describe exactly these tasks. Without it the baseline is computed from the
        tasks' hidden tests, and tasks that withhold them are refused."""
        if not tasks:
            raise ValueError("a dev evaluator needs tasks")
        self.tasks = list(tasks)
        self.encoder, self.decoder, self.execute = encoder, decoder, execute
        self.batch_size, self.precision, self.samples = batch_size, precision, samples
        if baseline is None:
            refuse_withheld(tuple(self.tasks), "the dev check's constant baseline")
            baseline = constant_baseline(self.tasks)
        elif baseline.tasks != len(self.tasks):
            raise ValueError(f"the baseline covers {baseline.tasks} tasks, not {len(self.tasks)}")
        self.baseline = baseline

    def evaluate(self, model: GPT, check: int) -> DevCheck:
        device = next(model.parameters()).device
        started = _synchronized(device)
        generations = dev_generations(
            model,
            self.tasks,
            self.encoder,
            self.decoder,
            batch_size=self.batch_size,
            precision=self.precision or ("bfloat16" if device.type == "cuda" else "float32"),
            samples=self.samples,
        )
        generated = _synchronized(device)
        results, missing = score_generations(self.tasks, generations, self.execute)
        waited = time.perf_counter()
        metrics = dev_metrics(results, self.samples)
        return DevCheck(
            check=check,
            greedy_pass_at_1=metrics["greedy_pass_at_1"],
            mean_pass_n8=metrics["mean_pass_n8"],
            pass_at_8=metrics["pass_at_8"],
            constant_baseline=self.baseline.mean,
            tasks=len(self.tasks),
            generation_seconds=generated - started,
            wait_seconds=waited - generated,
            samples=self.samples,
            context_limited=sum(row.context_limited for row in generations),
            no_program=missing,
            baseline_unparseable=self.baseline.unparseable,
            constant_pass_rate=self.baseline.pass_rate,
            per_task=tuple(results),
        )


def selection_key(check: DevCheck) -> tuple[float, float, int]:
    return (check.greedy_pass_at_1, check.mean_pass_n8, check.check)


def select_check(checks: Sequence[DevCheck]) -> DevCheck:
    """Best greedy pass@1; ties to the higher 8-sample mean, then to the later check."""
    if not checks:
        raise ValueError("selection needs at least one check")
    if len({check.check for check in checks}) != len(checks):
        raise ValueError("checks must have distinct indices")
    return max(checks, key=selection_key)


def dev_check_from_json(value: Mapping[str, Any]) -> DevCheck:
    """Rebuild a ``DevCheck`` written by ``to_json`` (for the control's shared start)."""
    fields = dict(value)
    fields["per_task"] = tuple(DevTaskResult(**row) for row in fields.get("per_task", []))
    return DevCheck(**fields)
