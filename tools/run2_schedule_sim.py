"""Deterministic full-schedule simulator of post-training run 2's worker; never provisions.

Run 2's stop rules interact: the cost stop at 120% of the gate's estimate, the finalization
reserve, one full Mac wait kept free before every update and exchange (``wait_margin``), the
projected-finish rule before every stage, the arm-stop rules and the same-weights rule. This
tool asks what they do to a whole run at given rates, without a GPU, a Mac or R2.

It runs the **real** worker's decision code: ``Run2Worker.execute`` and ``run_stage``, the
real ``run_rl`` and ``run_more_sft`` stage bodies, ``should_stop``, ``margin_reason``,
``before_wait``/``after_wait``, ``observe_wait``, ``RunLimits.stop_reason``, ``stopped``,
``tick``, the projected-finish rule with ``drop_control_for_rl``, ``measure_rl``,
``stop_kind``, ``stop_arm``, ``choose_endpoint``, ``skip_reason`` and ``after_stages``, on the
approved run-2 template filled with the gate's values by ``posttrain_approved.fill_gate``.
The gate's estimate comes from the real projection (``pilot_feasibility.posttrain2_report``)
and the gate's launch rule from ``posttrain_approved.gate_launch_fit``.

What is fake (``SimulatedRun2Worker``, ``patched_worker``):

- **the clock**: a counter (``SimClock``) that moves only when simulated work takes time;
- **the GPU**: RL's inner loop (``simulated_rl``) and the control's (``simulated_control``)
  replace ``run_rl`` and ``run_more_sft`` of ``scratchcoder.rl``; they make the same calls
  in the same order (``should_stop`` once per update, a reward request per RL update, a dev
  check at the same updates, the same end checks and stop rules; the conformance tests run
  both against the real loops), but each step only advances the clock by its seconds at the
  scenario's rates. ARC and the evaluation files advance it the same way, ticking per batch;
- **the Mac**: each wait lasts the scenario's seconds; the client's sleep is the worker's
  ``wait_sleep``, so the real wait-time limits apply; the Mac's lease is always fresh;
- **R2**: an upload is recorded and takes no foreground time (the worker's uploader is
  asynchronous); the finalization is not timed, only the time left for it is reported.

Simulated model behavior: every dev check scores a little higher than the one before, so
no dev-patience stop happens and each arm selects its latest scored check; an arm whose
selected check has no optimizer step holds the SFT weights. Collapse, leakage, non-finite
values, retries, slow or lost bridges are not simulated. Generation is priced at the token
caps, as the projections price it. Every output is a **forecast at the given rates**
(by default multiples of the planning rates, which come from earlier measured runs), not a
measurement of run 2. See ``docs/walkthroughs/run2-schedule-simulator.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scratchcoder.rl import control as control_module
from scratchcoder.rl import loop as loop_module
from scratchcoder.rl.control import ControlConfig, ControlResult, pass_boundaries
from scratchcoder.rl.dev import ConstantBaseline, DevCheck, select_check
from scratchcoder.rl.loop import CheckRecord, RLConfig, RLResult, ShouldStop, stop_request
from scratchcoder.rl.rewards import RewardError, RewardHostFailure, RewardTimeout
from scratchcoder.rl.rollout import DEV_TOKENS
from tools import posttrain_approved, posttrain_ops
from tools import posttrain_run2 as run2_module
from tools.posttrain_run import (
    CONTROL_REFERENCE,
    RL_REFERENCE,
    SFT_RESULTS,
    Rates,
    RunLimits,
    Specification,
    StagePlan,
    dev_check_generation_seconds,
)
from tools.posttrain_run2 import Run2Worker

FORMAT = "run2-schedule-simulation-v1"
FORECAST_LABEL = (
    "forecast at the given rates (multiples of the planning rates unless a projection report "
    "supplies them), driven through the real run-2 worker's decision code with a fake clock, "
    "GPU, Mac and R2; not a measurement of run 2"
)
ASSUMPTIONS = (
    "every RL update takes an optimizer step; no collapse, leakage, non-finite or "
    "dev-patience stop (dev scores rise with every check, so each arm selects its latest "
    "scored check)",
    "an arm whose selected check has no optimizer step holds the SFT weights",
    "the Mac answers every request in its nominal time and its lease stays fresh; no "
    "retries, slow waits or lost bridge",
    "generation is priced at the token caps, as the projections price it",
    "uploads take no foreground time; the finalization is not timed (the time left for it "
    "is reported)",
    "the control's rows are the kept tasks less any dropped for length: 'tasks' is that count",
)
MAC_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)
GPU_GRID = (0.8, 0.9, 1.0, 1.1, 1.2)
SETUP_GRID = (900.0, 1500.0)
TASKS_GRID = (161, 320, 1000)
DEFAULT_CEILING_USD = 10.0
"""The run-2 cost limit is the lower of the ceiling and 120% of the estimate; the default
ceiling is high enough that the 120% rule binds in every grid case. Pass the real ceiling to
check it."""
WAIT_SLICE_SECONDS = 5.0
"""The fake Mac wait advances the clock through the worker's ``wait_sleep`` in slices this
long, ``HEARTBEAT_REFRESH_SECONDS``: the real client observes a wait no more often."""
TICK_SLICES = 8
"""A generation advances the clock in this many slices, each followed by a tick."""
SIM_SHA256 = "0" * 64
START_UNIX = 1_800_000_000.0


# The fake clock and hardware --------------------------------------------------------------


class SimClock:
    """Unix time that moves only when ``advance`` is called."""

    def __init__(self, start: float = START_UNIX) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("the clock moves forward by finite seconds")
        self.now += seconds


@dataclass(frozen=True)
class Timing:
    """Seconds of each simulated piece of work at one scenario's rates."""

    rollout_per_update: float
    training_per_update: float
    reward_wait: float
    dev_generation: float
    dev_wait: float
    save: float
    load: float
    control_step: float
    arc_per_checkpoint: float
    generation_tokens_per_second: float

    @classmethod
    def of(cls, spec: Specification, rates: Rates) -> Timing:
        rl = spec.rl
        attempts = int(rl["tasks_per_update"]) * int(rl["group_size"])
        control = spec.control_config(attention=str(rl["attention"]))
        rows = control.batch_size * control.accumulation_steps
        length = control.sequence_length or spec.model.context
        assert rates.rollout_tokens_per_second is not None
        assert rates.reward_wait_seconds_per_update is not None
        assert rates.dev_wait_seconds_per_check is not None
        return cls(
            rollout_per_update=attempts
            * int(rl["max_new_tokens"])
            / rates.rollout_tokens_per_second,
            training_per_update=attempts * spec.model.context / rates.training_slots_per_second,
            reward_wait=float(rates.reward_wait_seconds_per_update),
            dev_generation=dev_check_generation_seconds(spec, rates),
            dev_wait=float(rates.dev_wait_seconds_per_check),
            save=rates.weights_save_seconds,
            load=rates.checkpoint_load_seconds,
            control_step=rows * length / rates.training_slots_per_second,
            arc_per_checkpoint=rates.arc_seconds_per_checkpoint,
            generation_tokens_per_second=rates.generation_tokens_per_second,
        )


def scaled_rates(spec: Specification, *, mac: float, gpu: float) -> Rates:
    """The planning rates with the Mac's time times ``mac`` and the GPU's times ``gpu``.

    GPU time is generation (rollouts, dev checks, evaluation), training and ARC; the Mac's is
    the reward and dev-check waits. Checkpoint loads and saves are not scaled."""
    for name, value in (("mac", mac), ("gpu", gpu)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"the {name} time multiple must be positive")
    planning = Rates.planning(spec)
    assert planning.rollout_tokens_per_second and planning.dev_generation_tokens_per_second
    return replace(
        planning,
        generation_tokens_per_second=planning.generation_tokens_per_second / gpu,
        rollout_tokens_per_second=planning.rollout_tokens_per_second / gpu,
        dev_generation_tokens_per_second=planning.dev_generation_tokens_per_second / gpu,
        training_slots_per_second=planning.training_slots_per_second / gpu,
        arc_seconds_per_checkpoint=planning.arc_seconds_per_checkpoint * gpu,
        reward_wait_seconds_per_update=float(planning.reward_wait_seconds_per_update or 0) * mac,
        dev_wait_seconds_per_check=float(planning.dev_wait_seconds_per_check or 0) * mac,
    )


def rates_from_projection(spec: Specification, rows: Mapping[str, Any]) -> Rates:
    """The rates a projection report's scenario recorded (``Rates.recorded``)."""
    planning = Rates.planning(spec)
    fields: dict[str, Any] = {
        name: float(rows[name])
        for name in (
            "generation_tokens_per_second",
            "training_slots_per_second",
            "arc_seconds_per_checkpoint",
            "checkpoint_load_seconds",
            "weights_save_seconds",
            "rollout_tokens_per_second",
            "dev_generation_tokens_per_second",
            "reward_wait_seconds_per_update",
            "dev_wait_seconds_per_check",
        )
    }
    return replace(planning, **fields)


def measured_input(spec: Specification, rates: Rates) -> dict[str, Any]:
    """``rates`` in the form ``posttrain2_report`` takes run 1's measured rates."""
    batches = {
        str(spec.generation_batch): rates.generation_tokens_per_second,
        str(int(spec.rl["generation_batch"])): rates.rollout_tokens_per_second,
    }
    if str(int(spec.rl["dev_batch_size"])) != str(int(spec.rl["generation_batch"])):
        batches[str(int(spec.rl["dev_batch_size"]))] = rates.dev_generation_tokens_per_second
    return {
        "generation_tokens_per_second": batches,
        "training_slots_per_second": rates.training_slots_per_second,
        "arc_seconds_per_checkpoint": rates.arc_seconds_per_checkpoint,
        "reward_wait_seconds_per_update": rates.reward_wait_seconds_per_update,
        "dev_wait_seconds_per_check": rates.dev_wait_seconds_per_check,
    }


def projected_estimate(rates: Rates, *, ceiling_usd: float) -> float:
    """The gate's estimate at ``rates``: the real run-2 projection's total
    (``pilot_feasibility.posttrain2_report``, its "run 1's measured rates" scenario) on the
    approved template."""
    from tools.pilot_feasibility import posttrain2_report

    template = Specification.from_dict(
        posttrain_approved.run2_specification(), allow_gate_nulls=True
    )
    report = posttrain2_report(
        posttrain_approved.SPECIFICATION_PATH_2,
        ceiling_usd=ceiling_usd,
        measured=measured_input(template, rates),
    )
    return float(report["gate_estimate_usd"])


def gate_values(*, ceiling_usd: float, estimate_usd: float) -> dict[str, Any]:
    """The gate's values for a simulated launch: the ceiling and the estimate are the
    scenario's; every pin is a placeholder (nothing is read)."""
    pin = {"size_bytes": 1, "sha256": SIM_SHA256}
    return {
        "ceiling_usd": ceiling_usd,
        "gate_estimate_usd": estimate_usd,
        "sft-weights": {"key": "posttrain-runs/1b/simulated/sft-weights/checkpoint.pt", **pin},
        "rl-tasks": dict(pin),
        "rl-dev": dict(pin),
        "rl-facts": dict(pin),
        "pool-test": dict(pin),
        "sft-results": {
            name: {"key": f"posttrain-runs/1b/simulated/{name}", **pin} for name in SFT_RESULTS
        },
        "sft-gate": {
            "key": "posttrain-runs/1b/simulated/format-gate/gate.json",
            "size_bytes": 1,
            "sha256": SIM_SHA256,
            "checkpoint_sha256": SIM_SHA256,
            "passed": True,
        },
    }


# The inner loops: the same calls in the same order as scratchcoder.rl --------------------


def checkpoint_name(arm: str, steps: int) -> str:
    return f"sim/{arm}/step-{steps:05d}.pt"


def simulated_rl(
    *,
    tasks: Sequence[Any],
    rewards: Any,
    evaluator: Any,
    config: RLConfig,
    should_stop: ShouldStop | None,
    on_check: Callable[[CheckRecord], None] | None,
    gpu: Callable[[float], None],
    now: Callable[[], float],
    timing: Timing | None,
    model: Any = None,
    attention: str = "reference",
) -> RLResult:
    """``scratchcoder.rl.loop.run_rl``'s control flow, without a model: dev check 0, then per
    update ``should_stop``, rollouts, one reward request, a training step and, at
    ``config.dev_checks``, a dev check; the same stops (reward wait, host failure, external,
    dev patience, completed) and the same end check and stop weights. ``gpu`` spends the
    simulated seconds (``timing``; ``None`` spends none)."""
    names = ("rollout_generation", "training_step", "reward_wait")
    seconds = dict.fromkeys((*names, "dev_generation", "dev_wait", "checkpoint"), 0.0)
    checks: list[CheckRecord] = []
    steps = completed = 0
    stop_reason: str | None = None
    stop_detail = ""
    best: float | None = None
    since_best = 0

    def save(update: int) -> str:
        spent = 0.0 if timing is None else timing.save
        gpu(spent)
        seconds["checkpoint"] += spent
        return checkpoint_name("rl", steps)

    def dev_check(update: int, *, selectable: bool = True) -> CheckRecord:
        result = evaluator.evaluate(model, update)
        if result.check != update:
            raise ValueError(f"the dev evaluator labelled check {update} as {result.check}")
        seconds["dev_generation"] += result.generation_seconds
        seconds["dev_wait"] += result.wait_seconds
        path = save(update)
        record = CheckRecord(result, steps, path, SIM_SHA256, selectable=selectable)
        checks.append(record)
        if on_check is not None:
            on_check(record)
        return record

    try:
        if 0 in config.dev_checks:
            best = dev_check(0).dev.greedy_pass_at_1
    except RewardTimeout as error:
        stop_reason, stop_detail = loop_module.REWARD_WAIT, f"dev check 0: {error}"
    except RewardHostFailure as error:
        stop_reason, stop_detail = loop_module.REWARD_HOST_FAILURE, f"dev check 0: {error}"
    while stop_reason is None and completed < config.updates:
        if should_stop is not None:
            requested = stop_request(should_stop())
            if requested is not None:
                stop_reason, stop_detail = loop_module.EXTERNAL, requested
                break
        update = completed + 1
        spent = 0.0 if timing is None else timing.rollout_per_update
        gpu(spent)
        seconds["rollout_generation"] += spent
        waited = now()
        try:
            list(rewards.rewards(update, []))
        except RewardError as error:
            seconds["reward_wait"] += now() - waited
            stop_reason = (
                loop_module.REWARD_WAIT
                if isinstance(error, RewardTimeout)
                else loop_module.REWARD_HOST_FAILURE
            )
            stop_detail = f"update {update}: {error}"
            break
        seconds["reward_wait"] += now() - waited
        spent = 0.0 if timing is None else timing.training_per_update
        gpu(spent)
        seconds["training_step"] += spent
        steps += 1
        completed = update
        if update in config.dev_checks:
            try:
                greedy = dev_check(update).dev.greedy_pass_at_1
            except RewardTimeout as error:
                stop_reason, stop_detail = loop_module.REWARD_WAIT, f"dev check {update}: {error}"
                break
            except RewardHostFailure as error:
                stop_reason = loop_module.REWARD_HOST_FAILURE
                stop_detail = f"dev check {update}: {error}"
                break
            if best is None or greedy > best:
                best, since_best = greedy, 0
            else:
                since_best += 1
                if since_best >= config.patience:
                    note = f"{since_best} checks not above {best}"
                    if update < config.updates:
                        stop_reason, stop_detail = loop_module.DEV_PATIENCE, note
                        break
                    stop_detail = f"patience reached at the final update: {note}"
    if stop_reason is None:
        stop_reason = loop_module.COMPLETED
    unchecked = not checks or checks[-1].optimizer_steps < steps
    external = stop_reason == loop_module.EXTERNAL
    wanted = config.check_at_external_stop if external else config.check_at_stop
    if stop_reason not in loop_module.CHANNEL_STOPS and wanted and unchecked:
        try:
            dev_check(completed, selectable=stop_reason != loop_module.LEAKAGE)
            unchecked = False
        except RewardError as error:
            stop_detail = f"{stop_detail}; stop check failed: {error}"
    stop_checkpoint = None
    if unchecked:
        stop_checkpoint = {"checkpoint": save(completed), "sha256": SIM_SHA256}
    candidates = [check for check in checks if check.selectable]
    chosen = select_check([check.dev for check in candidates]) if candidates else None
    selected = next((check for check in candidates if check.dev is chosen), None)
    return RLResult(
        stop_reason=stop_reason,
        stop_detail=stop_detail,
        updates=completed,
        optimizer_steps=steps,
        skipped_updates=[],
        checks=checks,
        selected=selected,
        seconds=seconds,
        precision="bfloat16",
        device="simulated",
        tasks=len(tasks),
        schedule_sha256=SIM_SHA256,
        seeds_used=completed * config.tasks_per_update * config.group_size,
        config=config,
        stop_checkpoint=stop_checkpoint,
        attention=attention,
    )


def control_planned_updates(rows: int, config: ControlConfig) -> int:
    """The control loader's planned updates: ``max_passes`` passes over ``rows`` rows."""
    per_update = config.batch_size * config.accumulation_steps
    return -(-rows * config.max_passes // per_update)


def simulated_control(
    *,
    rows: int,
    budget_seconds: float,
    start: CheckRecord,
    evaluator: Any,
    config: ControlConfig,
    should_stop: ShouldStop | None,
    on_check: Callable[[CheckRecord], None] | None,
    step_seconds: Callable[[int], float],
    gpu: Callable[[float], None],
    save_seconds: float = 0.0,
) -> ControlResult:
    """``scratchcoder.rl.control.run_more_sft``'s control flow over ``rows`` kept rows,
    without a model: before each update the budget rule (``should_stop`` once per update,
    then ``busy + mean step / 2 >= budget``), a step of ``step_seconds(update)``, a dev check
    after each pass boundary (``pass_boundaries``); at most ``max_passes`` passes; the same
    end check and stop weights."""
    per_update = config.batch_size * config.accumulation_steps
    planned = control_planned_updates(rows, config)
    boundaries = pass_boundaries(rows, per_update, config.pass_checks)
    seconds = dict.fromkeys(("training_step", "dev_generation", "dev_wait", "checkpoint"), 0.0)
    seconds["compile"] = 0.0
    steps: list[float] = []
    checks: list[CheckRecord] = []
    stop: dict[str, str] = {}
    step = 0

    def save(at: int) -> str:
        gpu(save_seconds)
        seconds["checkpoint"] += save_seconds
        return checkpoint_name("more-sft", at)

    def dev_check(at: int, label: str) -> None:
        result = evaluator.evaluate(None, at)
        if result.check != at:
            raise ValueError(f"the dev evaluator labelled check {at} as {result.check}")
        seconds["dev_generation"] += result.generation_seconds
        seconds["dev_wait"] += result.wait_seconds
        record = CheckRecord(result, at, save(at), SIM_SHA256, label)
        checks.append(record)
        if on_check is not None:
            on_check(record)

    while step < planned:
        requested = None if should_stop is None else stop_request(should_stop())
        if requested is not None:
            stop.setdefault("reason", control_module.EXTERNAL)
            stop.setdefault("detail", str(requested))
            break
        busy = math.fsum(steps)
        mean = busy / len(steps) if steps else 0.0
        if budget_seconds <= 0 or (steps and busy + mean / 2 >= budget_seconds):
            stop.setdefault("reason", control_module.BUDGET)
            stop.setdefault("detail", f"{busy:.3f} s busy against {budget_seconds:.3f} s")
            break
        spent = step_seconds(step + 1)
        gpu(spent)
        step += 1
        steps.append(spent)
        if step in boundaries:
            label = "passes-" + "-".join(str(count) for count in boundaries[step])
            try:
                dev_check(step, label)
            except RewardError as error:
                reason = (
                    control_module.REWARD_WAIT
                    if isinstance(error, RewardTimeout)
                    else control_module.REWARD_HOST_FAILURE
                )
                stop.update(reason=reason, detail=f"dev check {step}: {error}")
                break
    reason = stop.get("reason", control_module.PASSES_EXHAUSTED)
    detail = stop.get("detail", f"loader exhausted after {step} updates")
    last = checks[-1].optimizer_steps if checks else 0
    wanted = (
        config.check_at_external_stop if reason == control_module.EXTERNAL else config.check_at_stop
    )
    if reason not in control_module.CHANNEL_STOPS and wanted and step > last:
        try:
            dev_check(step, "end")
            last = step
        except RewardError as error:
            detail = f"{detail}; end check failed: {error}"
    stop_checkpoint = None
    if step > last:
        stop_checkpoint = {"checkpoint": save(step), "sha256": SIM_SHA256}
    seconds["training_step"] = math.fsum(steps)
    candidates = [start, *checks]
    best = select_check([check.dev for check in candidates])
    selected = next(check for check in candidates if check.dev is best)
    return ControlResult(
        stop_reason=reason,
        stop_detail=detail,
        budget_seconds=float(budget_seconds),
        busy_seconds=seconds["training_step"],
        updates=step,
        rows_per_pass=rows,
        rows_per_update=per_update,
        dropped={},
        checks=candidates,
        selected=selected,
        seconds=seconds,
        precision="bfloat16",
        device="simulated",
        config=config,
        tolerance=config.tolerance,
        stop_checkpoint=stop_checkpoint,
        step_seconds=steps,
    )


# The fake Mac and the worker around the real decision code --------------------------------


class FakeBridgeClient:
    """The worker's bridge client: ``heartbeat`` describes the wait in progress."""

    def __init__(self) -> None:
        self.requests_sent = 0
        self.wait: dict[str, Any] | None = None

    def heartbeat(self) -> dict[str, Any] | None:
        return None if self.wait is None else dict(self.wait)

    def close(self) -> None:
        self.wait = None


class FakeRewardSource:
    """``BridgeRewardSource``'s stand-in: each reward request waits the Mac's seconds."""

    def __init__(self, client: FakeBridgeClient, **_: Any) -> None:
        self.client = client
        self.worker: SimulatedRun2Worker | None = None

    def rewards(self, update: int, attempts: Sequence[Any]) -> list[Any]:
        assert self.worker is not None
        self.worker.mac_wait("reward", update, self.worker.timing.reward_wait)
        return []


class SimulatedDevEvaluator:
    """``BridgeDevEvaluator`` with ``WorkerDevExecutor``'s bookkeeping: generation on the
    fake GPU (``dev_tick`` per batch), then the real ``before_wait``, the fake Mac's wait and
    the real ``after_wait``. Scores rise with every check (see the module docstring)."""

    def __init__(self, worker: SimulatedRun2Worker) -> None:
        self.worker = worker

    def evaluate(self, model: Any, check: int) -> DevCheck:
        worker = self.worker
        programs = int(worker.plan.rl["dev_tasks"]) * (int(worker.plan.rl["dev_samples"]) + 1)
        generated = worker.generate(worker.timing.dev_generation, programs * DEV_TOKENS, dev=True)
        worker.before_wait("dev", check)
        outcome = "failed"
        started = worker.clock()
        try:
            worker.mac_wait("dev", check, worker.timing.dev_wait)
            outcome = "scored"
        finally:
            worker.after_wait("dev", check, outcome)
        score = 0.1 + check * 1e-6
        return DevCheck(
            check=check,
            greedy_pass_at_1=score,
            mean_pass_n8=score,
            pass_at_8=score,
            constant_baseline=0.0,
            tasks=int(worker.plan.rl["dev_tasks"]),
            generation_seconds=generated,
            wait_seconds=worker.clock() - started,
        )


class SimulatedRun2Worker(Run2Worker):
    """``Run2Worker`` with its inputs, files, GPU, Mac and R2 replaced by fakes; every
    decision method is inherited unchanged. ``__init__`` sets what the real constructors set,
    without reading or writing a file."""

    def __init__(
        self,
        spec: Specification,
        *,
        clock: SimClock,
        rates: Rates,
        tasks: int,
        pod_created: float,
    ) -> None:
        timeouts = spec.bridge_timeouts
        assert timeouts is not None
        self.sim_clock, self.timing, self.sim_tasks = clock, Timing.of(spec, rates), tasks
        self.actual_rates = rates
        self.events: list[dict[str, Any]] = []
        self.finalization: dict[str, Any] | None = None
        # Run2Worker's attributes.
        self.timeouts = timeouts
        self.bridge_root = Path("simulated-bridge")
        self.bridge_clock, self.bridge_sleep = clock, clock.advance
        self.fake_client = FakeBridgeClient()
        self.bridge_client = self.fake_client  # type: ignore[assignment]
        self.awaiting = None
        self.progress_unix = None
        self.lease_renewed_unix = None
        self.last_wait_check = 0.0
        self.rl_result = None
        self.control_result = None
        self.produced = {}
        self.uploaded = set()
        self.ending = None
        self.arm_stop = None
        self.not_on_pod = {}
        self.stopped_stages = {}
        self.generation_floor_basis = "planning"
        # PosttrainWorker's attributes.
        self.plan, self.root = spec, Path("simulated-run")
        self.inputs = {name: Path("simulated-inputs") / name for name in spec.inputs}
        self.prefix = "posttrain-runs/2/simulated/"
        self.sleep, self.clock = clock.advance, clock
        self.handover = None
        self.pod_id, self.pod_created = "simulated", pod_created
        # As ``posttrain_run.run``: the worker's wall is the Pod's lifetime, the first of the
        # wall, the in-Pod timer and the worker's and watchdog's cost stops.
        self.limits = RunLimits.of(spec, pod_seconds=posttrain_ops.lifetime_seconds(spec.ceiling))
        self.pod_end = pod_created + self.limits.pod_seconds
        self.reserve_unix = self.pod_end - spec.finalization_reserve_seconds
        self.device = SimpleNamespace(type="simulated")  # type: ignore[assignment]
        self.tokenizer = SimpleNamespace(sha256=SIM_SHA256)  # type: ignore[assignment]
        self.encoder = None  # type: ignore[assignment]
        self.secrets = ()
        self.last_activity = clock()
        self.last_heartbeat = 0.0
        self.stop_reason = None
        self.uploading = False
        self.phase = "initializing"
        self.stage_name = spec.stages[0].name
        self.progress = 0
        self.generated_tokens = 0
        self.labels = []
        self.milestones = []
        self.label_files = {}
        self.artifacts = []
        self.r2_final = None
        self.verification_errors = {}
        self.rates = Rates.planning(spec)
        self.sft_weights = None
        self.probe_rates = {}
        self.current = None
        self.observed = {"cap_tokens": 0.0, "generation_seconds": 0.0}
        self.stage_by_name = {
            stage.name: {
                "name": stage.name,
                "kind": stage.kind,
                "status": "pending",
                "outputs": [],
                "seconds": None,
            }
            for stage in spec.stages
        }
        self.report = {
            "run": spec.run,
            "status": "running",
            "stages": list(self.stage_by_name.values()),
            "artifacts": self.artifacts,
            "projections": [],
            "limits": {
                "pod_seconds": self.limits.pod_seconds,
                "cost_stop_usd": self.limits.cost_stop_usd,
                "pod_end_unix": self.pod_end,
                "finalization_reserve_unix": self.reserve_unix,
            },
            "timings": {"pod_created_unix": pod_created, "worker_started_unix": clock()},
            "rl": None,
            "more_sft": None,
            "arm_stop": None,
            "selection": {"endpoints": {"rule": run2_module.SAME_WEIGHTS_RULE}},
            "bridge": {
                "timeouts": asdict(timeouts),
                "waits": [],
                "lease_last_renewed_unix": None,
                "lost": None,
                "margin_stop": None,
                "requests_sent": 0,
            },
        }

    # Fakes of files, publication and R2 ------------------------------------------------------

    def publish(self) -> str:
        return ""

    def heartbeat(self) -> None:
        self.last_heartbeat = self.clock()

    def release(self) -> None:
        pass

    def check_planned_tasks(self) -> None:
        """The real check needs the task files; the floor is reported by the simulation."""

    def bridge_inputs(
        self, stage: StagePlan
    ) -> tuple[list[Any], list[Any], dict[str, Any], ConstantBaseline]:
        tasks = [SimpleNamespace(id=f"sim-{index}") for index in range(self.sim_tasks)]
        dev = [
            SimpleNamespace(id=f"dev-{index}") for index in range(int(self.plan.rl["dev_tasks"]))
        ]
        return tasks, dev, {}, ConstantBaseline(0.0, 0, len(dev))

    def input_sha256(self, name: str) -> str:
        return SIM_SHA256

    def checkpoint_path(self, reference: str) -> Path:
        if reference in (RL_REFERENCE, CONTROL_REFERENCE):
            return super().checkpoint_path(reference)
        return self.inputs[reference]

    def lease_renewed(self) -> float | None:
        return self.clock()  # The fake Mac renews its lease throughout.

    def dev_evaluator(self, dev: Sequence[Any], baseline: ConstantBaseline) -> Any:
        return SimulatedDevEvaluator(self)

    def export_label(self, label: str, files: Sequence[tuple[str, Path]]) -> list[dict[str, Any]]:
        """The fake R2: record the label at once; the upload takes no foreground time."""
        outputs = [
            {"label": label, "name": name, "sha256": SIM_SHA256, "bytes": 0} for name, _ in files
        ]
        outputs.append({"label": label, "name": "record.json", "sha256": SIM_SHA256, "bytes": 0})
        self.labels.append(label)
        self.artifacts.extend(outputs)
        if self.current is not None:
            self.stage_by_name[self.current.name]["outputs"].extend(outputs)
        self.event("upload", label=label)
        return outputs

    def finalize(self) -> None:
        now = self.clock()
        self.finalization = {
            "began_at_pod_seconds": now - self.pod_created,
            "seconds_left_before_wall": self.pod_end - now,
            "reserve_seconds": self.plan.finalization_reserve_seconds,
        }

    # The fake GPU and Mac ----------------------------------------------------------------

    def event(self, kind: str, **fields: Any) -> None:
        self.events.append({"t": self.clock() - self.pod_created, "kind": kind, **fields})

    def gpu(self, seconds: float) -> None:
        self.sim_clock.advance(seconds)

    def generate(self, seconds: float, tokens: int, *, dev: bool = False) -> float:
        """Generation on the fake GPU in slices, each followed by the worker's tick (a dev
        check's ``dev_tick``, which enforces nothing; an evaluation's ``tick``, which stops
        at the limits and the reserve)."""
        started = self.clock()
        for _ in range(TICK_SLICES):
            self.sim_clock.advance(seconds / TICK_SLICES)
            if dev:
                self.dev_tick(tokens // TICK_SLICES)
            else:
                self.tick(tokens // TICK_SLICES)
        return self.clock() - started

    def mac_wait(self, kind: str, index: int, seconds: float) -> None:
        """One request answered after ``seconds``; the worker observes the wait through its
        real ``wait_sleep`` (the wall and cost stops, a lost bridge)."""
        client = self.fake_client
        client.requests_sent += 1
        started = self.clock()
        client.wait = {
            "request_id": f"{kind}-{index}-{client.requests_sent}",
            "kind": kind,
            "index": index,
            "count": 0,
            "started_unix": started,
            "age_seconds": 0.0,
        }
        try:
            left = seconds
            while left > 1e-9:
                step = min(WAIT_SLICE_SECONDS, left)
                assert client.wait is not None
                client.wait["age_seconds"] = self.clock() - started
                self.wait_sleep(step)
                left -= step
        finally:
            client.wait = None

    # Stages that need a model: their timing, with the real tick and rate updates ---------

    def run_arc(self, stage: StagePlan) -> list[dict[str, Any]]:
        evaluated = self.evaluated_stage(stage)
        left_out = {
            ref: self.not_on_pod[ref]
            for ref in stage.settings["checkpoints"]
            if ref in self.not_on_pod
        }
        if left_out:
            self.stage_by_name[stage.name]["checkpoints_not_evaluated"] = left_out
        outputs, seconds = [], []
        for reference in evaluated.settings["checkpoints"]:
            self.checkpoint_path(reference)
            self.sim_clock.advance(self.timing.load)
            started = self.clock()
            self.generate(self.timing.arc_per_checkpoint, 0)
            self.tick()
            outputs += self.export_label(f"{stage.name}-{reference}", [("arc.json", Path("x"))])
            seconds.append(self.clock() - started)
        self.rates = replace(
            self.rates,
            arc_seconds_per_checkpoint=max(seconds)
            if seconds
            else self.rates.arc_seconds_per_checkpoint,
            basis={**self.rates.basis, "arc": "measured"},
        )
        return outputs

    def run_eval(self, stage: StagePlan) -> list[dict[str, Any]]:
        self.checkpoint_path(str(stage.settings["checkpoint"]))
        outputs = []
        for entry in stage.files:
            self.sim_clock.advance(self.timing.load)
            tokens = entry.planned_tokens
            seconds = self.generate(tokens / self.timing.generation_tokens_per_second, tokens)
            self.last_activity = self.clock()
            outputs += self.export_label(
                f"{stage.name}-{entry.name}", [("completions.jsonl", Path("x"))]
            )
            self.observe_generation(entry, seconds)
        return outputs


@contextlib.contextmanager
def patched_worker(worker: SimulatedRun2Worker) -> Iterator[None]:
    """Swap ``tools.posttrain_run2``'s model-bound names for the fakes while the simulated
    worker runs, so the real ``Run2Worker.run_rl`` and ``run_more_sft`` bodies execute."""
    timing = worker.timing

    def load_weights(*_: Any, **__: Any) -> Any:
        worker.gpu(timing.load)
        return SimpleNamespace(set_fused_attention=lambda on: None, fused_attention=True)

    def rl_loop(model: Any, tasks: Sequence[Any], **options: Any) -> RLResult:
        rewards = options["rewards"]
        rewards.source.worker = worker
        return simulated_rl(
            tasks=tasks,
            rewards=rewards,
            evaluator=options["evaluator"],
            config=options["config"],
            should_stop=options["should_stop"],
            on_check=options["on_check"],
            gpu=worker.gpu,
            now=worker.clock,
            timing=timing,
            model=model,
            attention=str(worker.plan.rl["attention"]),
        )

    def control(**options: Any) -> ControlResult:
        worker.gpu(timing.load)
        return simulated_control(
            rows=len(options["tasks"]),
            budget_seconds=float(options["budget_seconds"]),
            start=options["start"],
            evaluator=options["evaluator"],
            config=options["config"],
            should_stop=options["should_stop"],
            on_check=options["on_check"],
            step_seconds=lambda update: timing.control_step,
            gpu=worker.gpu,
            save_seconds=timing.save,
        )

    def compare(path: Path, sft: Path) -> Any:
        same = str(path).endswith("step-00000.pt")
        return SimpleNamespace(same=same, tensors=0, first_difference=None if same else "sim")

    swaps = {
        "run_rl_loop": rl_loop,
        "run_more_sft": control,
        "load_weights": load_weights,
        "BridgeRewardSource": FakeRewardSource,
        "compare_to_loaded": compare,
        "file_sha256": lambda path: SIM_SHA256,
    }
    saved = {name: getattr(run2_module, name) for name in swaps}
    try:
        for name, value in swaps.items():
            setattr(run2_module, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(run2_module, name, value)


# One scenario -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One simulated launch. ``mac`` and ``gpu`` multiply the planning time (unless
    ``rates`` is given); ``estimate_usd`` defaults to the real projection at those rates."""

    mac: float = 1.0
    gpu: float = 1.0
    setup_seconds: float = 900.0
    tasks: int = 320
    ceiling_usd: float = DEFAULT_CEILING_USD
    estimate_usd: float | None = None
    rates: Rates | None = field(default=None, compare=False)
    label: str = ""
    stop_fraction: float | None = None
    """A hypothetical cost-stop fraction of the estimate in place of the code's 120%: the
    gate is given ``estimate x fraction / 1.2``, so its 120% rule yields that stop. ``None``
    keeps the code's rule. For the owner's decision only; the worker is unchanged."""


def _arm_phrase(arm: str, updates: int, margin: Mapping[str, Any] | None) -> str:
    name = "RL" if arm == RL_REFERENCE else "the control"
    if margin is None:
        return name
    kind = margin.get("kind")
    if kind == "next update":
        return f"{name}, before update {updates + 1}"
    if kind == "dev":
        return f"{name}, at its dev check after update {margin.get('index')}, unscored"
    return f"{name}, at the reward wait of update {margin.get('index')}"


def simulate(scenario: Scenario) -> dict[str, Any]:
    """Run one scenario through the gate and the worker; return its forecast record."""
    template = Specification.from_dict(
        posttrain_approved.run2_specification(), allow_gate_nulls=True
    )
    rates = scenario.rates or scaled_rates(template, mac=scenario.mac, gpu=scenario.gpu)
    estimate = (
        projected_estimate(rates, ceiling_usd=scenario.ceiling_usd)
        if scenario.estimate_usd is None
        else scenario.estimate_usd
    )
    written = estimate
    if scenario.stop_fraction is not None:
        written = estimate * scenario.stop_fraction / posttrain_ops.ESTIMATE_STOP_FRACTION
    floor = int(template.rl["min_tasks"])
    record: dict[str, Any] = {
        "label": scenario.label,
        "mac_time_multiple": scenario.mac if scenario.rates is None else None,
        "gpu_time_multiple": scenario.gpu if scenario.rates is None else None,
        "setup_seconds": scenario.setup_seconds,
        "tasks": scenario.tasks,
        "floor": floor,
        "admitted_by_floor": scenario.tasks >= floor,
        "estimate_usd": estimate,
        "cost_limit_usd": posttrain_ops.estimate_limit_usd(scenario.ceiling_usd, written),
        "ceiling_usd": scenario.ceiling_usd,
        "stop_fraction": posttrain_ops.ESTIMATE_STOP_FRACTION
        if scenario.stop_fraction is None
        else scenario.stop_fraction,
        "hypothetical_stop": scenario.stop_fraction is not None,
        "forecast": FORECAST_LABEL,
    }
    try:
        # The gate's own refusals: fill_gate refuses limits that leave less than the planned
        # setup once the launch rule's need at the planning rates is held.
        raw = posttrain_approved.run2_specification(
            gate_values(ceiling_usd=scenario.ceiling_usd, estimate_usd=written)
        )
        gate = posttrain_approved.gate_launch_fit(raw)
    except ValueError as error:
        record.update(
            outcome="refused at the gate",
            first_stop="refused at the gate",
            gate={"fits": False, "refusal": str(error)},
        )
        return record
    record["gate"] = gate
    if not gate["fits"]:
        record.update(outcome="refused at the gate", first_stop="refused at the gate")
        return record
    spec = Specification.from_dict(raw)
    clock = SimClock()
    worker = SimulatedRun2Worker(
        spec,
        clock=clock,
        rates=rates,
        tasks=scenario.tasks,
        pod_created=clock() - scenario.setup_seconds,
    )
    with patched_worker(worker):
        worker.execute()
    record.update(summarize(worker, spec))
    return record


def summarize(worker: SimulatedRun2Worker, spec: Specification) -> dict[str, Any]:
    report = worker.report
    rl, control = worker.rl_result, worker.control_result
    margin = report["bridge"]["margin_stop"]
    statuses = {stage["name"]: stage["status"] for stage in report["stages"]}
    rl_updates = int(spec.rl["updates"])
    arm_stop = report.get("arm_stop")
    ended_unix = worker.clock()
    elapsed = ended_unix - worker.pod_created
    summary: dict[str, Any] = {
        "outcome": report["status"],
        "stop_reason": report.get("stop_reason"),
        "stages": statuses,
        "rl": None
        if rl is None
        else {
            "updates": rl.updates,
            "declared_updates": rl_updates,
            "reached_budget": rl.updates == rl_updates,
            "stop_reason": rl.stop_reason,
            "stop_detail": rl.stop_detail,
            "busy_seconds": rl.busy_seconds,
            "checks": [check.dev.check for check in rl.checks],
        },
        "control": None,
        "arm_stop": None
        if arm_stop is None
        else {key: arm_stop.get(key) for key in ("arm", "kind", "reason", "arm_stop_reason")},
        "margin_stop": margin,
        "endpoints": {
            arm: {key: entry.get(key) for key in ("selected", "evaluation", "independent")}
            for arm, entry in report["selection"]["endpoints"].items()
            if isinstance(entry, dict)
        },
        "pod_elapsed_seconds": elapsed,
        "spend_before_finalization_usd": elapsed / 3600 * worker.limits.quoted_total_hourly,
        "cost_stop_usd": worker.limits.cost_stop_usd,
        "finalization": worker.finalization,
    }
    if control is not None:
        per_step = worker.timing.control_step
        summary["control"] = {
            "updates": control.updates,
            "budget_seconds": control.budget_seconds,
            "busy_seconds": control.busy_seconds,
            "budget_updates": round(control.budget_seconds / per_step) if per_step else None,
            "planned_updates": control_planned_updates(control.rows_per_pass, control.config),
            "busy_fraction": control.busy_seconds / control.budget_seconds
            if control.budget_seconds
            else None,
            "within_tolerance": control.within_tolerance,
            "tolerance": control.tolerance,
            "passes": control.passes,
            "stop_reason": control.stop_reason,
            "stop_detail": control.stop_detail,
            "checks": [check.dev.check for check in control.checks],
        }
    summary["first_stop"] = first_stop(summary, margin)
    return summary


def first_stop(summary: Mapping[str, Any], margin: Mapping[str, Any] | None) -> str:
    """Where and why the run first stopped short, in the docs' words."""
    rl, control = summary.get("rl"), summary.get("control")
    arm = summary.get("arm_stop")
    if arm is not None:
        updates = (
            (rl or {}).get("updates", 0)
            if arm["arm"] == RL_REFERENCE
            else ((control or {}).get("updates", 0))
        )
        phrase = _arm_phrase(str(arm["arm"]), int(updates), margin)
        return f"{phrase} ({arm['reason']})"
    if summary["outcome"] == "completed":
        if control is not None and not control["within_tolerance"]:
            return f"none; the control is outside its tolerance ({control['stop_reason']})"
        return "none"
    return f"the run, {summary.get('stop_reason')}"


def control_spends_budget(record: Mapping[str, Any]) -> bool:
    """Whether both arms ran to their declared budgets: RL's updates, the control within its
    tolerance, no arm stopped short and the run completed."""
    rl, control = record.get("rl"), record.get("control")
    return bool(
        record.get("outcome") == "completed"
        and record.get("arm_stop") is None
        and rl is not None
        and rl["reached_budget"]
        and control is not None
        and control["within_tolerance"]
    )


def minimum_stop_fraction(
    scenario: Scenario, *, highest: float = 3.0, step: float = 0.01
) -> dict[str, Any]:
    """The lowest cost-stop fraction of the estimate, in ``step`` increments from the code's
    120%, at which both arms reach their budgets (``control_spends_budget``) with the fixed
    wait margin unchanged. A hypothetical for the owner's decision, not a rule in force."""
    base = replace(scenario, stop_fraction=None)
    estimate = simulate(base)["estimate_usd"]
    fraction = posttrain_ops.ESTIMATE_STOP_FRACTION
    while fraction <= highest + 1e-9:
        record = simulate(replace(base, estimate_usd=estimate, stop_fraction=round(fraction, 4)))
        if control_spends_budget(record):
            return {
                "stop_fraction": round(fraction, 4),
                "cost_limit_usd": record["cost_limit_usd"],
                "estimate_usd": estimate,
                "spend_before_finalization_usd": record["spend_before_finalization_usd"],
                "hypothetical": True,
            }
        fraction += step
    return {"stop_fraction": None, "estimate_usd": estimate, "highest": highest}


# The grid and the CLI ---------------------------------------------------------------------


def grid(
    *,
    macs: Sequence[float] = MAC_GRID,
    gpus: Sequence[float] = GPU_GRID,
    setups: Sequence[float] = SETUP_GRID,
    tasks: Sequence[int] = TASKS_GRID,
    ceiling_usd: float = DEFAULT_CEILING_USD,
) -> list[dict[str, Any]]:
    """Every combination; the estimate of each (mac, gpu) is projected once."""
    template = Specification.from_dict(
        posttrain_approved.run2_specification(), allow_gate_nulls=True
    )
    rows = []
    for mac, gpu in itertools.product(macs, gpus):
        rates = scaled_rates(template, mac=mac, gpu=gpu)
        estimate = projected_estimate(rates, ceiling_usd=ceiling_usd)
        for setup, count in itertools.product(setups, tasks):
            rows.append(
                simulate(
                    Scenario(
                        mac=mac,
                        gpu=gpu,
                        setup_seconds=setup,
                        tasks=count,
                        ceiling_usd=ceiling_usd,
                        estimate_usd=estimate,
                        rates=rates,
                        label=f"mac {mac:g}x, gpu {gpu:g}x",
                    )
                )
                | {"mac_time_multiple": mac, "gpu_time_multiple": gpu}
            )
    return rows


def projection_scenarios(
    path: Path, *, setups: Sequence[float], tasks: Sequence[int], ceiling_usd: float
) -> list[dict[str, Any]]:
    """Each scenario of a run-2 projection report (``pilot_feasibility posttrain --output``)
    at its recorded rates, with the report's own gate estimate where it has one."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("kind") != "posttrain-feasibility" or report.get("run") != "2":
        raise ValueError(f"{path} is not a run-2 projection report")
    template = Specification.from_dict(
        posttrain_approved.run2_specification(), allow_gate_nulls=True
    )
    rows = []
    for name, scenario in report["scenarios"].items():
        rates = rates_from_projection(template, scenario["rates"])
        estimate = float(scenario["total_usd"])
        for setup, count in itertools.product(setups, tasks):
            rows.append(
                simulate(
                    Scenario(
                        setup_seconds=setup,
                        tasks=count,
                        ceiling_usd=ceiling_usd,
                        estimate_usd=estimate,
                        rates=rates,
                        label=f"{path.name}: {name}",
                    )
                )
            )
    return rows


def document(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "forecast": FORECAST_LABEL,
        "assumptions": list(ASSUMPTIONS),
        "scenarios": rows,
    }


def table(rows: Sequence[Mapping[str, Any]]) -> str:
    """A Markdown table of the forecasts."""
    lines = [
        "| Scenario | Setup | Tasks | Estimate, stop | RL updates | Control busy / budget "
        "| Within 5% | First stop |",
        "| --- | ---: | ---: | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        rl, control = row.get("rl"), row.get("control")
        fraction = (
            "not run"
            if control is None
            else f"{control['busy_seconds']:.0f} / {control['budget_seconds']:.0f} s "
            f"({100 * (control['busy_fraction'] or 0):.0f}%)"
        )
        tasks = f"{row['tasks']}" + ("" if row["admitted_by_floor"] else " (below floor)")
        lines.append(
            f"| {row['label']} | {row['setup_seconds']:.0f} s | {tasks} "
            f"| ${row['estimate_usd']:.2f}, ${row['cost_limit_usd']:.2f} "
            f"| {'-' if rl is None else rl['updates']} | {fraction} "
            f"| {'-' if control is None else control['within_tolerance']} "
            f"| {row['first_stop']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mac", type=float, nargs="*", default=list(MAC_GRID))
    parser.add_argument("--gpu", type=float, nargs="*", default=list(GPU_GRID))
    parser.add_argument("--setup", type=float, nargs="*", default=list(SETUP_GRID))
    parser.add_argument("--tasks", type=int, nargs="*", default=list(TASKS_GRID))
    parser.add_argument("--ceiling-usd", type=float, default=DEFAULT_CEILING_USD)
    parser.add_argument(
        "--projection",
        type=Path,
        action="append",
        default=[],
        help="also simulate each scenario of this run-2 projection report at its rates",
    )
    parser.add_argument("--output", type=Path, help="write the JSON record here")
    args = parser.parse_args(argv)
    rows = grid(
        macs=args.mac,
        gpus=args.gpu,
        setups=args.setup,
        tasks=args.tasks,
        ceiling_usd=args.ceiling_usd,
    )
    for path in args.projection:
        rows += projection_scenarios(
            path, setups=args.setup, tasks=args.tasks, ceiling_usd=args.ceiling_usd
        )
    if args.output is not None:
        args.output.write_text(
            json.dumps(document(rows), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(f"Run 2 schedule: {FORECAST_LABEL}.\n")
    print(table(rows), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
