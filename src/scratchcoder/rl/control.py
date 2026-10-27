"""The more-SFT control: the RL run's compute spent on more supervised fine-tuning instead.

The control starts from the same SFT parent as RL and trains on the reference solutions of
the kept RL task list, each rendered in chat-v1 (``render_code_task``), with the project's
SFT path unchanged: ``SFTLoader`` rows and the ``Trainer`` with assistant-only loss divided
by each update's assistant-token count (``loss_normalization: update_targets``). Updates
are 16 rows x 2 accumulated microbatches; the optimizer is a fresh AdamW (betas 0.9/0.95,
weight decay 0.1, clip 1.0) at a constant 6e-5 after a 10-update linear warmup (the
trainer's cosine schedule with ``min_lr_ratio = 1`` is constant after warmup).

**Budget.** The control's GPU-busy time is the sum of the trainer's synchronized
``step_seconds``. It is matched to the RL run's rollout-generation plus training-step
seconds (``RLResult.busy_seconds``). Before each update the control stops if
``busy + mean step / 2 >= budget``: taking one more step of the mean length would then
overshoot by more than stopping undershoots. Dev checks, checkpoints and execution waits
are outside ``step_seconds`` and recorded separately, as in RL; neither arm compiles, so
compile time is 0 in both. The result reports the actual mismatch and whether it lies
within plus or minus 5%.

**Checks.** The start is RL's update-0 check (the same weights), reused, not repeated.
Further checks follow after 1, 2, 4 and 8 passes over the rows (after the first update
whose rows complete that pass, ``ceil(p x rows / rows per update)``) and at the budget end.
Selection is RL's rule. See ``docs/walkthroughs/more-sft-control.md``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.common.logging import EventLogger
from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.chat import TextEncoder, render_code_task
from scratchcoder.data.sft import SFTLoader, SFTRow
from scratchcoder.eval.tasks import Task
from scratchcoder.rl.checkpoint import read_weights, save_weights
from scratchcoder.rl.dev import DevEvaluator, select_check
from scratchcoder.rl.loop import CheckRecord, ShouldStop, stop_request
from scratchcoder.rl.rewards import RewardError, RewardTimeout
from scratchcoder.train.trainer import Trainer, TrainingStop

BUDGET = "budget"
PASSES_EXHAUSTED = "passes_exhausted"
EXTERNAL = "external"
NON_FINITE = "non_finite"
REWARD_WAIT = "reward_wait"
REWARD_HOST_FAILURE = "reward_host_failure"
CHANNEL_STOPS = frozenset({REWARD_WAIT, REWARD_HOST_FAILURE})


@dataclass(frozen=True)
class ControlConfig:
    """The control's settings; the defaults are the approved rehearsal protocol."""

    batch_size: int = 16
    accumulation_steps: int = 2
    learning_rate: float = 6e-5
    warmup_updates: int = 10
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    seed: int = 0
    """Row order seed (``SFTLoader``: pass ``p`` sorts by ``sha256(f"{seed}|{p}|{id}")``)."""
    pass_checks: tuple[int, ...] = (1, 2, 4, 8)
    max_passes: int = 400
    """Passes the loader plans; the budget normally ends the control long before."""
    tolerance: float = 0.05
    sequence_length: int | None = None
    """Row length; ``None`` uses the model's context."""
    optimizer_foreach: bool = False

    def __post_init__(self) -> None:
        for name in ("batch_size", "accumulation_steps", "max_passes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_updates", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 < self.tolerance < 1:
            raise ValueError("tolerance must lie in (0, 1)")
        checks = self.pass_checks
        if (
            not isinstance(checks, tuple)
            or any(type(value) is not int or value < 1 for value in checks)
            or list(checks) != sorted(set(checks))
        ):
            raise ValueError("pass_checks must be increasing positive pass counts")
        if self.sequence_length is not None and (
            type(self.sequence_length) is not int or self.sequence_length < 1
        ):
            raise ValueError("sequence_length must be a positive integer or None")

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["pass_checks"] = list(self.pass_checks)
        return value


def control_rows(encoder: TextEncoder, tasks: Sequence[Task]) -> list[SFTRow]:
    """One chat-v1 row per kept RL task: its prompt and public test, then its reference."""
    rows = []
    for task in tasks:
        if task.solution is None:
            raise ValueError(f"RL task {task.id} has no reference solution for the control")
        rows.append(SFTRow.of(task.id, task.source, render_code_task(encoder, task)))
    return rows


def pass_boundaries(rows: int, rows_per_update: int, passes: Sequence[int]) -> dict[int, list[int]]:
    """The update after which each pass count is complete: ``ceil(p x rows / per update)``."""
    boundaries: dict[int, list[int]] = {}
    for count in passes:
        boundaries.setdefault(-(-count * rows // rows_per_update), []).append(count)
    return boundaries


@dataclass
class ControlResult:
    stop_reason: str
    stop_detail: str
    budget_seconds: float
    busy_seconds: float
    updates: int
    rows_per_pass: int
    rows_per_update: int
    dropped: dict[str, int]
    checks: list[CheckRecord]
    selected: CheckRecord
    seconds: dict[str, float]
    precision: str
    device: str
    config: ControlConfig
    tolerance: float
    stop_checkpoint: dict[str, str] | None = None
    step_seconds: list[float] = field(default_factory=list)

    @property
    def mismatch_seconds(self) -> float:
        return self.busy_seconds - self.budget_seconds

    @property
    def mismatch_fraction(self) -> float | None:
        return self.mismatch_seconds / self.budget_seconds if self.budget_seconds > 0 else None

    @property
    def within_tolerance(self) -> bool:
        fraction = self.mismatch_fraction
        return fraction is not None and abs(fraction) <= self.tolerance

    @property
    def passes(self) -> float:
        return self.updates * self.rows_per_update / self.rows_per_pass

    def to_json(self) -> dict[str, Any]:
        commit, dirty = git_state()
        return {
            "format": "more-sft-control-v1",
            "created_at": utc_now(),
            "git_commit": commit,
            "dirty": dirty,
            "stop_reason": self.stop_reason,
            "stop_detail": self.stop_detail,
            "budget_seconds": self.budget_seconds,
            "busy_seconds": self.busy_seconds,
            "mismatch_seconds": self.mismatch_seconds,
            "mismatch_fraction": self.mismatch_fraction,
            "within_tolerance": self.within_tolerance,
            "tolerance": self.tolerance,
            "updates": self.updates,
            "passes": self.passes,
            "rows_per_pass": self.rows_per_pass,
            "rows_per_update": self.rows_per_update,
            "dropped": self.dropped,
            "checks": [check.to_json() for check in self.checks],
            "selected": self.selected.to_json(),
            "seconds": dict(self.seconds),
            "precision": self.precision,
            "device": self.device,
            "config": self.config.to_json(),
            "stop_checkpoint": self.stop_checkpoint,
            "step_seconds": self.step_seconds,
        }


def run_more_sft(
    *,
    parent: Path,
    tasks: Sequence[Task],
    encoder: TextEncoder,
    tokenizer_hash: str,
    budget_seconds: float,
    start: CheckRecord,
    evaluator: DevEvaluator,
    output: Path,
    run_id: str,
    device: str = "cpu",
    config: ControlConfig | None = None,
    should_stop: ShouldStop | None = None,
) -> ControlResult:
    """Train the control from ``parent`` for ``budget_seconds`` of GPU-busy time.

    ``start`` is RL's update-0 check of the same parent weights. The trainer writes its run
    record and events under ``output / run_id``; the control adds
    ``control-events.jsonl``, ``checkpoints/step-NNNNN.pt`` and ``control-result.json``
    there. ``should_stop`` is read once per update, as in RL.
    """
    config = config or ControlConfig()
    if isinstance(budget_seconds, bool) or not math.isfinite(budget_seconds) or budget_seconds < 0:
        raise ValueError("the budget must be finite nonnegative seconds")
    if start.dev.check != 0:
        raise ValueError("the control's start is RL's update-0 check")
    run_dir = output / run_id
    if (run_dir / "control-result.json").exists():
        raise ValueError(f"{run_dir} already holds a control result")
    state = read_weights(parent)
    if state["tokenizer_hash"] != tokenizer_hash:
        raise ValueError("parent checkpoint tokenizer differs")
    model_config = ModelConfig(**state["config"]["model"])
    del state
    rows = control_rows(encoder, tasks)
    length = config.sequence_length or model_config.context
    loader = SFTLoader(
        rows,
        sequence_length=length,
        batch_size=config.batch_size,
        accumulation_steps=config.accumulation_steps,
        seed=config.seed,
        passes=config.max_passes,
    )
    kept = len(rows) - sum(loader.dropped.values())
    precision = "bfloat16" if device.startswith("cuda") else "float32"
    run = RunConfig(
        model=model_config,
        train=TrainConfig(
            seed=0,
            steps=loader.planned_updates,
            batch_size=config.batch_size,
            sequence_length=length,
            learning_rate=config.learning_rate,
            min_lr_ratio=1.0,
            warmup_steps=min(config.warmup_updates, loader.planned_updates),
            weight_decay=config.weight_decay,
            gradient_clip=config.gradient_clip,
            accumulation_steps=config.accumulation_steps,
            beta1=config.beta1,
            beta2=config.beta2,
            adam_epsilon=config.adam_epsilon,
            precision=precision,
            optimizer_foreach=config.optimizer_foreach,
            schedule="cosine",
            loss_normalization="update_targets",
        ),
        data_mix={"rl-references": 1.0},
        parent_checkpoint=str(parent),
    )
    trainer = Trainer(
        run,
        loader,
        run_id=run_id,
        run_root=output,
        data_hash=loader.signature,
        tokenizer_hash=tokenizer_hash,
        device=device,
    )
    boundaries = pass_boundaries(kept, loader.rows_per_update, config.pass_checks)
    seconds = dict.fromkeys(("training_step", "dev_generation", "dev_wait", "checkpoint"), 0.0)
    seconds["compile"] = 0.0
    step_seconds: list[float] = []
    checks: list[CheckRecord] = []
    stop: dict[str, str] = {}
    external: dict[str, Any] = {"step": -1, "value": None}

    with EventLogger(run_dir / "control-events.jsonl", run_id) as logger:

        def save(step: int, *, checked: bool) -> tuple[str, str]:
            path = run_dir / "checkpoints" / f"step-{step:05d}.pt"
            started = time.perf_counter()
            digest = save_weights(
                path,
                trainer.model,
                tokenizer_hash=tokenizer_hash,
                run_id=run_id,
                step=step,
                metadata={"kind": "more-sft", "step": step, "checked": checked},
            )
            seconds["checkpoint"] += time.perf_counter() - started
            return str(path), digest

        def dev_check(step: int, label: str) -> None:
            result = evaluator.evaluate(trainer.model, step)
            if result.check != step:
                raise ValueError(f"the dev evaluator labelled check {step} as {result.check}")
            seconds["dev_generation"] += result.generation_seconds
            seconds["dev_wait"] += result.wait_seconds
            path, digest = save(step, checked=True)
            checks.append(CheckRecord(result, step, path, digest, label))
            logger.event(
                "control_dev_check",
                step=step,
                label=label,
                busy_seconds=math.fsum(step_seconds),
                checkpoint=path,
                sha256=digest,
                **result.to_json(per_task=False),
            )

        def on_step(current: Trainer, metrics: dict[str, float]) -> None:
            step_seconds.append(metrics["step_seconds"])
            if current.step in boundaries:
                label = "passes-" + "-".join(str(count) for count in boundaries[current.step])
                try:
                    dev_check(current.step, label)
                except RewardError as error:
                    reason = (
                        REWARD_WAIT if isinstance(error, RewardTimeout) else REWARD_HOST_FAILURE
                    )
                    stop.update(reason=reason, detail=f"dev check {current.step}: {error}")
                    raise TrainingStop(reason) from error

        def budget_stop() -> bool:
            if trainer.step != external["step"]:
                external.update(
                    step=trainer.step,
                    value=None if should_stop is None else stop_request(should_stop()),
                )
            if external["value"] is not None:
                stop.setdefault("reason", EXTERNAL)
                stop.setdefault("detail", str(external["value"]))
                return True
            busy = math.fsum(step_seconds)
            mean = busy / len(step_seconds) if step_seconds else 0.0
            if budget_seconds <= 0 or (step_seconds and busy + mean / 2 >= budget_seconds):
                stop.setdefault("reason", BUDGET)
                stop.setdefault("detail", f"{busy:.3f} s busy against {budget_seconds:.3f} s")
                return True
            return False

        logger.event(
            "control_started",
            budget_seconds=budget_seconds,
            rows=len(rows),
            kept_rows=kept,
            dropped=loader.dropped,
            rows_per_update=loader.rows_per_update,
            planned_updates=loader.planned_updates,
            boundaries={str(step): counts for step, counts in boundaries.items()},
            config=config.to_json(),
        )
        try:
            trainer.train(should_stop=budget_stop, on_step=on_step)
        except TrainingStop:
            pass
        except FloatingPointError as error:
            stop.update(reason=NON_FINITE, detail=f"step {trainer.step + 1}: {error}")
        except Exception as error:
            # Keep the latest weights before the error propagates.
            logger.event(
                "control_failed",
                error_type=type(error).__name__,
                detail=str(error)[:500],
                updates=trainer.step,
            )
            if trainer.step > (checks[-1].optimizer_steps if checks else 0):
                save(trainer.step, checked=False)
            raise
        reason = stop.get("reason", PASSES_EXHAUSTED)
        detail = stop.get("detail", f"loader exhausted after {trainer.step} updates")
        last = checks[-1].optimizer_steps if checks else 0
        stop_checkpoint: dict[str, str] | None = None
        if reason not in CHANNEL_STOPS and trainer.step > last:
            try:
                dev_check(trainer.step, "end")
                last = trainer.step
            except RewardError as error:
                detail = f"{detail}; end check failed: {error}"
        if trainer.step > last:
            path, digest = save(trainer.step, checked=False)
            stop_checkpoint = {"checkpoint": path, "sha256": digest}
        trainer.finish(success=reason != NON_FINITE)
        seconds["training_step"] = math.fsum(step_seconds)
        candidates = [start, *checks]
        best = select_check([check.dev for check in candidates])
        selected = next(check for check in candidates if check.dev is best)
        result = ControlResult(
            stop_reason=reason,
            stop_detail=detail,
            budget_seconds=float(budget_seconds),
            busy_seconds=seconds["training_step"],
            updates=trainer.step,
            rows_per_pass=kept,
            rows_per_update=loader.rows_per_update,
            dropped=dict(loader.dropped),
            checks=candidates,
            selected=selected,
            seconds=seconds,
            precision=precision,
            device=str(trainer.device),
            config=config,
            tolerance=config.tolerance,
            stop_checkpoint=stop_checkpoint,
            step_seconds=step_seconds,
        )
        logger.event(
            "control_stopped",
            reason=reason,
            detail=detail,
            updates=trainer.step,
            busy_seconds=result.busy_seconds,
            budget_seconds=result.budget_seconds,
            mismatch_seconds=result.mismatch_seconds,
            within_tolerance=result.within_tolerance,
            selected=selected.dev.check,
        )
    atomic_json(run_dir / "control-result.json", result.to_json())
    return result
