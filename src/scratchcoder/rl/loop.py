"""The RL run: on-policy group-advantage policy-gradient updates, dev checks and stops.

Each update, numbered from 1:

1. takes the next 32 tasks of the schedule (``task_schedule``);
2. generates G = 8 attempts per task from the current policy (chat-v1 prompt with the
   public test, temperature 1.0, top-p 1.0, at most min(256, context - prompt) new tokens,
   seeds from ``scratchcoder.rl.rollout.rollout_seed``);
3. sends the attempts to the reward source and waits;
4. forms group-mean advantages, skips all-equal groups, and takes one optimizer step on
   the token-normalized policy-gradient loss (``scratchcoder.rl.policy``), with log
   probabilities from the trainer's own forward pass; the batch is then discarded;
5. logs one ``rl_update`` event and checks the stops.

Dev checks run before the first update (check 0) and after the updates listed in
``RLConfig.dev_checks``; each saves a weights-only checkpoint (and calls ``on_check``, so
the caller can upload it at once). The run stops on a non-finite reward, loss or gradient
norm, on collapse, on dev patience before the last update, on the leakage rule, when the
reward source times out or cannot resolve a host failure, when ``should_stop`` asks, or
after the last update. A stop whose weights changed since the last check takes one more
check, except a reward-channel stop (the check needs that side), a ``should_stop`` stop
unless ``check_at_external_stop`` (a cost, wall or idle stop should not spend more), and
either kind when its flag is off; the check taken at a leakage stop is recorded but never
selected. Whatever the stop, unchecked weights are saved, and so are the latest weights if
anything unexpected raises, the stop-time check included. Selection then picks the best
selectable check (``scratchcoder.rl.dev.select_check``). See ``docs/rl.md`` and
``docs/walkthroughs/rl-update.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from scratchcoder.common.logging import EventLogger
from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.chat import BOS, TextDecoder, TextEncoder
from scratchcoder.data.chat import task_prompt as chat_task_prompt
from scratchcoder.data.sft import pass_key
from scratchcoder.eval.tasks import Task
from scratchcoder.model.gpt import GPT
from scratchcoder.optim.adamw import AdamW, clip_gradients, learning_rate
from scratchcoder.rl.checkpoint import save_weights
from scratchcoder.rl.dev import DevCheck, DevEvaluator, select_check
from scratchcoder.rl.policy import group_advantages, policy_gradient_backward, policy_sequences
from scratchcoder.rl.rewards import (
    RewardError,
    RewardHostFailure,
    RewardSource,
    RewardTimeout,
    check_rewards,
)
from scratchcoder.rl.rollout import ROLLOUT_SEED, ROLLOUT_TOKENS, generate_rollouts, rollout_seed
from scratchcoder.rl.screening import Baseline, require_train_tasks
from scratchcoder.sandbox.reward import AuditTally
from scratchcoder.train.trainer import synchronized_time

COMPLETED = "completed"
NON_FINITE = "non_finite"
COLLAPSE = "collapse"
DEV_PATIENCE = "dev_patience"
LEAKAGE = "leakage"
REWARD_WAIT = "reward_wait"
REWARD_HOST_FAILURE = "reward_host_failure"
EXTERNAL = "external"
STOP_REASONS = (
    COMPLETED,
    NON_FINITE,
    COLLAPSE,
    DEV_PATIENCE,
    LEAKAGE,
    REWARD_WAIT,
    REWARD_HOST_FAILURE,
    EXTERNAL,
)
CHANNEL_STOPS = frozenset({REWARD_WAIT, REWARD_HOST_FAILURE})
"""Stops caused by the execution side; no dev check follows them (it needs that side)."""

ShouldStop = Callable[[], str | bool | None]
"""Checked between updates. ``None`` or ``False`` continues; ``True`` or a nonempty string
stops with reason ``external`` (the string, such as ``cost``, ``wall`` or ``idle``, becomes
the stop detail)."""


def _positive_int(value: object, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class RLConfig:
    """Every RL setting; the defaults are the approved rehearsal protocol."""

    updates: int = 100
    tasks_per_update: int = 32
    group_size: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = ROLLOUT_TOKENS
    learning_rate: float = 3e-6
    warmup_updates: int = 5
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    micro_batch: int = 16
    """Sequences per forward/backward pass; gradients accumulate to one full-batch step."""
    generation_batch: int = 256
    optimizer_foreach: bool = False
    shuffle_seed: int = 0
    rollout_seed: int = ROLLOUT_SEED
    dev_checks: tuple[int, ...] = (0, 12, 25, 50, 75, 100)
    patience: int = 3
    """Stop when greedy pass@1 is not above the best so far for this many checks in a row."""
    collapse_distinct: float = 2.0
    collapse_updates: int = 5
    """Stop when the mean distinct completions per group stays below ``collapse_distinct``
    for this many consecutive updates."""
    leakage_rate: float = 0.01
    leakage_min_rewarded: int = 100
    """Stop when more than ``leakage_rate`` of rewarded attempts (reward > 0), counted from
    the first update, carry static-audit flags, once at least this many exist."""
    check_at_stop: bool = True
    """Take one more dev check at a stop the run decides itself (completion, collapse, patience,
    leakage, non-finite) when its weights changed since the last check."""
    check_at_external_stop: bool = False
    """The same at a ``should_stop`` stop. Off by default: a cost, wall or idle stop should not
    start about 1,800 generations and a Mac round trip. The weights are saved either way."""

    def __post_init__(self) -> None:
        for name in (
            "updates",
            "tasks_per_update",
            "max_new_tokens",
            "micro_batch",
            "generation_batch",
            "patience",
            "collapse_updates",
        ):
            _positive_int(getattr(self, name), name)
        _positive_int(self.group_size, "group_size", 2)
        for name in ("warmup_updates", "shuffle_seed", "rollout_seed", "leakage_min_rewarded"):
            _positive_int(getattr(self, name), name, 0)
        if self.warmup_updates > self.updates:
            raise ValueError("warmup_updates must not exceed updates")
        if _number(self.temperature, "temperature") <= 0:
            raise ValueError("temperature must be positive")
        if _number(self.top_p, "top_p") != 1.0:
            raise ValueError(
                "top_p must be 1.0: the loss scores tokens under the full softmax, so "
                "nucleus filtering would make the update off-policy"
            )
        for name in ("learning_rate", "adam_epsilon", "gradient_clip", "collapse_distinct"):
            if _number(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if _number(self.weight_decay, "weight_decay") < 0:
            raise ValueError("weight_decay must be nonnegative")
        if not all(0 <= _number(beta, "beta") < 1 for beta in (self.beta1, self.beta2)):
            raise ValueError("Adam betas must lie in [0, 1)")
        if not 0 <= _number(self.leakage_rate, "leakage_rate") <= 1:
            raise ValueError("leakage_rate must lie in [0, 1]")
        for name in ("optimizer_foreach", "check_at_stop", "check_at_external_stop"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        checks = self.dev_checks
        if (
            not isinstance(checks, tuple)
            or any(type(value) is not int for value in checks)
            or list(checks) != sorted(set(checks))
            or any(not 0 <= value <= self.updates for value in checks)
        ):
            raise ValueError("dev_checks must be increasing update numbers in [0, updates]")

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["dev_checks"] = list(self.dev_checks)
        return value


def task_schedule(
    task_ids: Sequence[str], *, updates: int, per_update: int, seed: int
) -> list[tuple[int, tuple[str, ...]]]:
    """``(pass, task IDs)`` for each update, numbered from 1 in list order.

    Pass ``p`` orders the tasks by ``pass_key(seed, p, ID)`` (the first eight bytes of
    ``sha256(f"{seed}|{p}|{ID}")`` as an integer, ID breaking ties), the order the SFT loader
    uses. Updates take consecutive blocks of ``per_update`` from it, so an update never
    repeats a task. A final block shorter than ``per_update`` is not used, and the next
    pass reshuffles.
    """
    ids = list(task_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("task IDs must be unique")
    _positive_int(updates, "updates")
    _positive_int(per_update, "per_update")
    if len(ids) < per_update:
        raise ValueError(f"{len(ids)} tasks cannot fill an update of {per_update}")
    blocks: list[tuple[int, tuple[str, ...]]] = []
    epoch = 0
    while len(blocks) < updates:
        order = sorted(ids, key=lambda identifier: pass_key(seed, epoch, identifier))
        for start in range(0, len(order) - per_update + 1, per_update):
            blocks.append((epoch, tuple(order[start : start + per_update])))
            if len(blocks) == updates:
                break
        epoch += 1
    return blocks


@dataclass(frozen=True)
class CheckRecord:
    """A dev check, the optimizer steps behind it and its weights-only checkpoint."""

    dev: DevCheck
    optimizer_steps: int
    checkpoint: str
    sha256: str
    label: str = ""
    selectable: bool = True
    """``False`` for the check taken at a leakage stop: its score is recorded, but weights
    trained into a reward exploit are not offered as the run's result."""

    def to_json(self) -> dict[str, Any]:
        return {
            "dev": self.dev.to_json(),
            "optimizer_steps": self.optimizer_steps,
            "checkpoint": self.checkpoint,
            "sha256": self.sha256,
            "label": self.label,
            "selectable": self.selectable,
        }


@dataclass
class RLResult:
    """What an RL run did, why it stopped, and which check it selects."""

    stop_reason: str
    stop_detail: str
    updates: int
    optimizer_steps: int
    skipped_updates: list[int]
    checks: list[CheckRecord]
    selected: CheckRecord | None
    seconds: dict[str, float]
    precision: str
    device: str
    tasks: int
    schedule_sha256: str
    seeds_used: int
    config: RLConfig
    monitoring: list[dict[str, Any]] = field(default_factory=list)
    stop_checkpoint: dict[str, str] | None = None
    attention: str = "reference"
    """The model's training attention kernel; the control's ``attention`` should match."""

    @property
    def busy_seconds(self) -> float:
        """GPU-busy time: rollout generation plus training steps (the control's budget)."""
        return self.seconds["rollout_generation"] + self.seconds["training_step"]

    def to_json(self) -> dict[str, Any]:
        commit, dirty = git_state()
        return {
            "format": "rl-result-v1",
            "created_at": utc_now(),
            "git_commit": commit,
            "dirty": dirty,
            "stop_reason": self.stop_reason,
            "stop_detail": self.stop_detail,
            "updates": self.updates,
            "optimizer_steps": self.optimizer_steps,
            "skipped_updates": self.skipped_updates,
            "checks": [check.to_json() for check in self.checks],
            "selected": None if self.selected is None else self.selected.to_json(),
            "seconds": dict(self.seconds),
            "busy_seconds": self.busy_seconds,
            "precision": self.precision,
            "attention": self.attention,
            "device": self.device,
            "tasks": self.tasks,
            "schedule_sha256": self.schedule_sha256,
            "seeds_used": self.seeds_used,
            "config": self.config.to_json(),
            "stop_checkpoint": self.stop_checkpoint,
            "monitoring": self.monitoring,
        }


def _rollout_seeds(update: int, group: int, base: int) -> Callable[[Task, int], int]:
    def seed_of(task: Task, sample: int) -> int:
        return rollout_seed(update, task.id, sample, group_size=group, base=base)

    return seed_of


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def stop_request(value: object) -> str | None:
    """The stop detail a ``ShouldStop`` value asks for, or ``None`` to continue."""
    if value is None or value is False:
        return None
    if value is True:
        return "requested"
    if isinstance(value, str) and value:
        return value
    raise ValueError("should_stop must return None, a boolean or a nonempty string")


def run_rl(
    model: GPT,
    tasks: Sequence[Task],
    *,
    encoder: TextEncoder,
    decoder: TextDecoder,
    rewards: RewardSource,
    evaluator: DevEvaluator,
    output: Path,
    run_id: str,
    tokenizer_hash: str,
    config: RLConfig | None = None,
    should_stop: ShouldStop | None = None,
    on_check: Callable[[CheckRecord], None] | None = None,
    facts: Mapping[str, Baseline] | None = None,
) -> RLResult:
    """Run RL on ``model`` (the SFT parent, already on its device) over the kept tasks.

    Writes ``events.jsonl`` (one ``rl_update`` event per update, plus dev checks and the
    stop), ``checkpoints/update-NNNN.pt`` (weights at each check, and at a stop whose
    weights no check covers) and ``result.json`` under ``output``, which must not already
    hold a result. Precision follows the device: bfloat16 autocast on CUDA, float32 on the CPU.

    ``tasks`` must be train-split tasks the reward channel can parse (``require_train_tasks``).
    ``facts`` are the Mac's facts about their hidden tests, for a Pod that holds none
    (``scratchcoder.rl.facts``); without them the tasks must carry their real hidden tests.
    ``on_check`` is called with each dev check's record once its checkpoint file exists, so
    a caller can upload it at once; an exception from it propagates like any other error,
    after the latest weights are saved. An unexpected error in the loop or in the stop-time
    check saves the latest weights, logs ``rl_failed`` and re-raises; no result file is
    written then.
    """
    config = config or RLConfig()
    if (output / "result.json").exists():
        raise ValueError(f"{output} already holds an RL result")
    by_id = {task.id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("RL task IDs must be unique")
    task_baselines = require_train_tasks(tasks, baselines=facts)
    device = next(model.parameters()).device
    precision = "bfloat16" if device.type == "cuda" else "float32"
    schedule = task_schedule(
        sorted(by_id),
        updates=config.updates,
        per_update=config.tasks_per_update,
        seed=config.shuffle_seed,
    )
    schedule_sha256 = hashlib.sha256(
        json.dumps([list(ids) for _, ids in schedule]).encode()
    ).hexdigest()
    prompts = {task.id: chat_task_prompt(encoder, task) for task in tasks}
    baselines = {identifier: found.reward or 0.0 for identifier, found in task_baselines.items()}
    named = list(model.named_parameters())
    optimizer = AdamW(
        [parameter for _, parameter in named],
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        decay_mask=[
            parameter.ndim >= 2 and name != "embedding.weight" for name, parameter in named
        ],
        foreach=config.optimizer_foreach,
    )
    group = config.group_size
    tally = AuditTally()
    seen_seeds: set[int] = set()
    seconds = dict.fromkeys(
        (
            "rollout_generation",
            "training_step",
            "reward_wait",
            "dev_generation",
            "dev_wait",
            "checkpoint",
        ),
        0.0,
    )
    checks: list[CheckRecord] = []
    monitoring: list[dict[str, Any]] = []
    skipped_updates: list[int] = []
    stop_reason: str | None = None
    stop_detail = ""
    stop_checkpoint: dict[str, str] | None = None
    completed = steps = 0
    best: float | None = None
    since_best = collapse_run = 0
    checkpoints = output / "checkpoints"

    def save(update: int, *, checked: bool) -> tuple[str, str]:
        path = checkpoints / f"update-{update:04d}.pt"
        started = time.perf_counter()
        digest = save_weights(
            path,
            model,
            tokenizer_hash=tokenizer_hash,
            run_id=run_id,
            step=update,
            metadata={"kind": "rl", "update": update, "optimizer_steps": steps, "checked": checked},
        )
        seconds["checkpoint"] += time.perf_counter() - started
        return str(path), digest

    output.mkdir(parents=True, exist_ok=True)
    with EventLogger(output / "events.jsonl", run_id) as logger:

        def record_failure(error: Exception) -> None:
            """Log an unexpected error and keep the policy's latest weights before it propagates."""
            logger.event(
                "rl_failed",
                error_type=type(error).__name__,
                detail=str(error)[:500],
                updates=completed,
                optimizer_steps=steps,
            )
            if not checks or checks[-1].optimizer_steps < steps:
                save(completed, checked=False)

        def dev_check(update: int, *, selectable: bool = True) -> CheckRecord:
            result = evaluator.evaluate(model, update)
            if result.check != update:
                raise ValueError(f"the dev evaluator labelled check {update} as {result.check}")
            seconds["dev_generation"] += result.generation_seconds
            seconds["dev_wait"] += result.wait_seconds
            path, digest = save(update, checked=True)
            record = CheckRecord(result, steps, path, digest, selectable=selectable)
            checks.append(record)
            logger.event(
                "rl_dev_check",
                update=update,
                optimizer_steps=steps,
                checkpoint=path,
                sha256=digest,
                selectable=selectable,
                **result.to_json(per_task=False),
            )
            if on_check is not None:
                on_check(record)
            return record

        try:
            logger.event(
                "rl_started",
                tasks=len(tasks),
                device=str(device),
                precision=precision,
                schedule_sha256=schedule_sha256,
                config=config.to_json(),
            )
            try:
                if 0 in config.dev_checks:
                    best = dev_check(0).dev.greedy_pass_at_1
            except RewardTimeout as error:
                stop_reason, stop_detail = REWARD_WAIT, f"dev check 0: {error}"
            except RewardHostFailure as error:
                stop_reason, stop_detail = REWARD_HOST_FAILURE, f"dev check 0: {error}"
            while stop_reason is None and completed < config.updates:
                if should_stop is not None:
                    requested = stop_request(should_stop())
                    if requested is not None:
                        stop_reason, stop_detail = EXTERNAL, requested
                        break
                update = completed + 1
                epoch, ids = schedule[update - 1]
                batch = [by_id[identifier] for identifier in ids]
                started = synchronized_time(device)
                rollouts = generate_rollouts(
                    model,
                    batch,
                    [prompts[task.id] for task in batch],
                    decoder,
                    samples=group,
                    seed_of=_rollout_seeds(update, group, config.rollout_seed),
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_new_tokens=config.max_new_tokens,
                    batch_size=config.generation_batch,
                    precision=precision,
                )
                generation_seconds = synchronized_time(device) - started
                seconds["rollout_generation"] += generation_seconds
                for rollout in rollouts:
                    if rollout.seed in seen_seeds:
                        raise RuntimeError(
                            f"rollout seed {rollout.seed} repeated in update {update}"
                        )
                    seen_seeds.add(rollout.seed)
                attempts = [rollout.attempt for rollout in rollouts]
                waited = time.perf_counter()
                try:
                    replies = list(rewards.rewards(update, attempts))
                except RewardError as error:
                    wait_seconds = time.perf_counter() - waited
                    seconds["reward_wait"] += wait_seconds
                    stop_reason = (
                        REWARD_WAIT if isinstance(error, RewardTimeout) else REWARD_HOST_FAILURE
                    )
                    stop_detail = f"update {update}: {error}"
                    logger.event(
                        "rl_reward_failed",
                        update=update,
                        reason=stop_reason,
                        detail=str(error),
                        seconds={
                            "rollout_generation": generation_seconds,
                            "reward_wait": wait_seconds,
                        },
                    )
                    break
                wait_seconds = time.perf_counter() - waited
                seconds["reward_wait"] += wait_seconds
                check_rewards(attempts, replies)
                values = [float(reply.reward) for reply in replies]
                nonfinite_reward = not all(math.isfinite(value) for value in values)
                groups = (
                    []
                    if nonfinite_reward
                    else [
                        group_advantages(task.id, values[index * group : (index + 1) * group])
                        for index, task in enumerate(batch)
                    ]
                )
                for reply in replies:
                    rewarded = math.isfinite(reply.reward) and reply.reward > 0
                    tally.observe(reply.audit_flags, rewarded=rewarded)
                loss: float | None = None
                norm: float | None = None
                rate: float | None = None
                denominator = 0
                stepped = False
                started = synchronized_time(device)
                if nonfinite_reward:
                    stop_reason, stop_detail = NON_FINITE, f"update {update}: reward"
                else:
                    sequences, denominator = policy_sequences(
                        [rollout.prompt for rollout in rollouts],
                        [rollout.tokens for rollout in rollouts],
                        [advantage for item in groups for advantage in item.advantages],
                        [item.skipped for item in groups for _ in range(group)],
                    )
                    if sequences:
                        optimizer.zero_grad()
                        stage = "loss"
                        try:
                            loss = policy_gradient_backward(
                                model,
                                sequences,
                                denominator=denominator,
                                micro_batch=config.micro_batch,
                                device=device,
                                precision=precision,
                                pad_id=BOS,
                                temperature=config.temperature,
                            )
                            stage = "gradient_norm"
                            norm = clip_gradients(model.parameters(), config.gradient_clip)
                        except FloatingPointError:
                            optimizer.zero_grad()
                            stop_reason, stop_detail = NON_FINITE, f"update {update}: {stage}"
                            loss = norm = None
                        else:
                            rate = learning_rate(
                                steps,
                                total_steps=config.updates,
                                warmup_steps=config.warmup_updates,
                                peak=config.learning_rate,
                                minimum_ratio=1.0,
                            )
                            optimizer.lr = rate
                            optimizer.step()
                            optimizer.zero_grad()
                            steps += 1
                            stepped = True
                    if not stepped and stop_reason is None:
                        skipped_updates.append(update)
                training_seconds = synchronized_time(device) - started
                seconds["training_step"] += training_seconds
                completed = update
                skipped = sum(item.skipped for item in groups)
                distinct = _mean(
                    [
                        float(
                            len(
                                {
                                    rollout.tokens
                                    for rollout in rollouts[k * group : (k + 1) * group]
                                }
                            )
                        )
                        for k in range(len(batch))
                    ]
                )
                update_rewarded = [reply for reply in replies if reply.reward > 0]
                update_rewarded_flagged = sum(bool(reply.audit_flags) for reply in update_rewarded)
                tokens = [rollout.attempt.completion_tokens for rollout in rollouts]
                row: dict[str, Any] = {
                    "update": update,
                    "pass": epoch,
                    "task_ids": list(ids),
                    "groups": len(batch),
                    "attempts": len(attempts),
                    "mean_reward": None if nonfinite_reward else _mean(values),
                    "baseline_mean": _mean([baselines[task.id] for task in batch]),
                    "skipped_groups": skipped,
                    "skipped_share": skipped / len(batch),
                    "loss_tokens": denominator,
                    "mean_attempt_length": _mean([float(count) for count in tokens]),
                    "stopped_share": sum(r.finish == "stop" for r in rollouts) / len(rollouts),
                    "no_program": sum(attempt.code is None for attempt in attempts),
                    "timeouts": sum(reply.unfinished > 0 for reply in replies),
                    "statuses": dict(sorted(Counter(reply.status for reply in replies).items())),
                    "distinct_per_group": distinct,
                    "audit_flagged": sum(bool(reply.audit_flags) for reply in replies),
                    "rewarded": len(update_rewarded),
                    "rewarded_flagged": update_rewarded_flagged,
                    "rewarded_flagged_rate": (
                        update_rewarded_flagged / len(update_rewarded) if update_rewarded else 0.0
                    ),
                    "cumulative_rewarded": tally.rewarded,
                    "cumulative_rewarded_flagged_rate": tally.rewarded_rate,
                    "loss": loss,
                    "gradient_norm": norm,
                    "learning_rate": rate,
                    "optimizer_step": stepped,
                    "optimizer_steps": steps,
                    "seconds": {
                        "rollout_generation": generation_seconds,
                        "training_step": training_seconds,
                        "reward_wait": wait_seconds,
                    },
                }
                monitoring.append(row)
                logger.event("rl_update", **row)
                if stop_reason is not None:
                    break
                collapse_run = collapse_run + 1 if distinct < config.collapse_distinct else 0
                if collapse_run >= config.collapse_updates:
                    stop_reason = COLLAPSE
                    stop_detail = (
                        f"{collapse_run} updates below {config.collapse_distinct} distinct"
                    )
                    break
                if (
                    tally.rewarded >= config.leakage_min_rewarded
                    and tally.rewarded_rate > config.leakage_rate
                ):
                    stop_reason = LEAKAGE
                    stop_detail = (
                        f"{tally.rewarded_flagged} of {tally.rewarded} rewarded attempts flagged"
                    )
                    break
                if update in config.dev_checks:
                    try:
                        greedy = dev_check(update).dev.greedy_pass_at_1
                    except RewardTimeout as error:
                        stop_reason, stop_detail = REWARD_WAIT, f"dev check {update}: {error}"
                        break
                    except RewardHostFailure as error:
                        stop_reason, stop_detail = (
                            REWARD_HOST_FAILURE,
                            f"dev check {update}: {error}",
                        )
                        break
                    if best is None or greedy > best:
                        best, since_best = greedy, 0
                    else:
                        since_best += 1
                        if since_best >= config.patience:
                            note = f"{since_best} checks not above {best}"
                            if update < config.updates:
                                stop_reason, stop_detail = DEV_PATIENCE, note
                                break
                            # No update is left to cut short: the run completed.
                            stop_detail = f"patience reached at the final update: {note}"
        except Exception as error:
            record_failure(error)
            raise
        if stop_reason is None:
            stop_reason = COMPLETED
        unchecked = not checks or checks[-1].optimizer_steps < steps
        wanted = config.check_at_external_stop if stop_reason == EXTERNAL else config.check_at_stop
        if stop_reason not in CHANNEL_STOPS and wanted and unchecked:
            try:
                dev_check(completed, selectable=stop_reason != LEAKAGE)
                unchecked = False
            except RewardError as error:
                stop_detail = f"{stop_detail}; stop check failed: {error}"
            except Exception as error:
                record_failure(error)
                raise
        if unchecked:
            path, digest = save(completed, checked=False)
            stop_checkpoint = {"checkpoint": path, "sha256": digest}
        candidates = [check for check in checks if check.selectable]
        selected_dev = select_check([check.dev for check in candidates]) if candidates else None
        selected = next((check for check in candidates if check.dev is selected_dev), None)
        result = RLResult(
            stop_reason=stop_reason,
            stop_detail=stop_detail,
            updates=completed,
            optimizer_steps=steps,
            skipped_updates=skipped_updates,
            checks=checks,
            selected=selected,
            seconds=seconds,
            precision=precision,
            device=str(device),
            tasks=len(tasks),
            schedule_sha256=schedule_sha256,
            seeds_used=len(seen_seeds),
            config=config,
            monitoring=monitoring,
            stop_checkpoint=stop_checkpoint,
            attention="fused" if model.fused_attention else "reference",
        )
        logger.event(
            "rl_stopped",
            reason=stop_reason,
            detail=stop_detail,
            updates=completed,
            optimizer_steps=steps,
            selected=None if selected is None else selected.dev.check,
            seconds=dict(seconds),
        )
    atomic_json(output / "result.json", result.to_json())
    return result
