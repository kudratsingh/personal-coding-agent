"""Run 2's Pod worker: RL with the Mac in the loop, the more-SFT control, ARC and evaluation.

``Run2Worker`` is run 1's ``PosttrainWorker`` (the same receipt, limits, projected-finish rule,
finalization reserve, R2 prefix guard, uploads and finalization) with two more stage kinds and
the reward bridge:

- ``rl``: ``scratchcoder.rl.loop.run_rl`` from run 1's selected SFT weights on the withheld
  kept tasks, with the Mac's facts in place of their hidden tests; rewards through
  ``BridgeRewardSource`` and dev checks through ``BridgeDevEvaluator`` (one ``BridgeClient``
  for the run). ``on_check`` uploads each dev check's weights at once as its own weights-only
  label (``rl-check-<update>``).
- ``more-sft``: ``scratchcoder.rl.control.run_more_sft`` from the same SFT weights on the kept
  tasks' references, its GPU-busy budget RL's measured busy seconds, RL's attention kernel,
  its checks' weights uploaded at once (``more-sft-check-<step>``), its dev checks through
  the bridge.
- ``arc`` and ``eval`` stages read ``rl`` and ``more-sft``: each arm's selected checkpoint.

How an arm's stop is handled (the owner's run 2 gate rulings, ``ARM_STOP_RULE``): a wall or
cost stop ends the run, as in run 1. A stop of the Mac's side (``reward_wait`` on either side,
``reward_host_failure``, ``bridge_lost``) or a margin stop (``wait_margin``) ends only that
arm's training, after its latest weights are uploaded: the arm's endpoint is selected from
the dev checks already scored, the stages that need no Mac wait continue under the
projected-finish rule, the reserve and every limit, and the run then finalizes as
``stopped`` with that arm's reason. A channel stop in either arm and a margin stop in RL
leave the control unevaluated (it needs the Mac for its dev checks); a margin stop in the
control evaluates both arms as the projected-finish rule admits them (when both do not fit
but RL's alone does, the control is left unevaluated and RL's is evaluated). With no RL dev
check scored there is nothing to select and the run ends at once. Any other stop is a result
and the run goes on.

Same weights (``SAME_WEIGHTS_RULE``): when an arm's selected checkpoint holds the SFT weights
the run loaded, compared tensor by tensor and bit for bit, the arm is not evaluated again on
the Pod (no ARC entry, no ``eval-<arm>`` files). ``report.json``'s
``selection.endpoints.<arm>`` records the comparison and that run 1's SFT results, pinned in
the specification's ``sft_results``, stand for it: not an independent measurement.

Liveness while the Pod waits on the Mac (``RUN2_LIVENESS``): the client's sleep is the
worker's ``wait_sleep``, so a wait heartbeats ``awaiting_rewards`` with the client's
``heartbeat()`` fields, stamps ``progress_unix`` from a fresh Mac lease (``server.lease``,
renewed by the Mac's polls and its progress ticker, dated by the Pod's clock) for the in-Pod
idle rule, enforces the wall and cost stops, and gives the wait up as ``bridge_lost`` when the
lease has stayed stale for ``lost_bridge_seconds``. Before every update and every exchange the
worker keeps one full wait's time and money free (``wait_margin``). At the finalization the
bridge directories are uploaded as one label, so a late reply is kept. See
``docs/walkthroughs/posttrain-run.md``.
"""

from __future__ import annotations

import io
import json
import math
import os
import stat
import tarfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from scratchcoder.common.records import utc_now
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.tasks import Task, load_tasks
from scratchcoder.rl.bridge import (
    BridgeClient,
    BridgeDevEvaluator,
    BridgeDevExecutor,
    BridgeRewardSource,
    BridgeSchemaError,
    read_bounded,
)
from scratchcoder.rl.bridge_mac import LEASE_FILE, LEASE_FORMAT
from scratchcoder.rl.checkpoint import compare_to_loaded, load_weights
from scratchcoder.rl.control import ControlResult, run_more_sft
from scratchcoder.rl.dev import ConstantBaseline, dev_subset
from scratchcoder.rl.facts import constant_baseline_from_facts, load_facts
from scratchcoder.rl.loop import CHANNEL_STOPS, EXTERNAL, CheckRecord, RLResult
from scratchcoder.rl.loop import run_rl as run_rl_loop
from scratchcoder.rl.rewards import Attempt, AttemptReward, RewardTimeout
from scratchcoder.rl.rollout import DEV_TOKENS
from scratchcoder.rl.screening import Baseline
from scratchcoder.train.pilot import PilotStop
from tools import posttrain_ops
from tools.posttrain_run import (
    BRIDGE_LOST_STOP,
    CONTROL_REFERENCE,
    PROJECTED_STOP,
    RESERVE_STOP,
    RL_REFERENCE,
    SFT_ARC,
    SFT_RESULT_FILES,
    WAIT_MARGIN_STOP,
    EvalFile,
    PosttrainWorker,
    Specification,
    StagePlan,
)
from tools.rehearsal_run import HEARTBEAT_REFRESH_SECONDS

RUN2_LIVENESS = (
    "while the Pod waits on the Mac the heartbeat reads awaiting_rewards with the request it "
    "waits on; the Mac watchdog then allows the GPU to idle for that request kind's Pod "
    "timeout plus a slack, never more; the in-Pod idle rule counts a fresh Mac lease "
    "(server.lease, renewed within its 60 s) as progress and a stale one as nothing; a wait "
    "whose lease stays stale for lost_bridge_seconds ends as bridge_lost; before every update "
    "and every exchange the worker stops (wait_margin) unless one full wait fits before the "
    "cost stop, the wall and the finalization reserve"
)
ARM_STOPS = frozenset({*CHANNEL_STOPS, EXTERNAL})
"""The stops of either arm that are not a result: the execution side failed or was lost
(``reward_wait``, ``reward_host_failure``; ``bridge_lost`` and ``wait_margin`` reach the arm as
one of these) or ``should_stop`` asked (``external``: a limit or the margin). Any other RL stop
(completed, collapse, dev patience, leakage, non-finite) is a result: the control runs. The
control's budget, passes-exhausted or non-finite stop is a result too."""
CHANNEL = "channel"
MARGIN = "margin"
LIMIT = "limit"
ARM_STOP_RULE = (
    "a wall or cost stop ends the run; a stop of the Mac's side (reward_wait, "
    "reward_host_failure, bridge_lost) or a wait_margin stop ends only that arm's training: "
    "its latest weights are uploaded, its endpoint is selected from the dev checks already "
    "scored, and the stages that need no Mac wait run under the projected-finish rule, the "
    "finalization reserve and every limit; the control is skipped after any such stop in RL, "
    "and not evaluated after a stop of the Mac's side in the control; after a margin stop in "
    "the control each arm is evaluated as the projected-finish rule admits it, RL's first; "
    "with no RL dev check scored the run ends"
)
SELECTION_RULE = (
    "the selectable check with the highest dev-subset greedy pass@1; ties to the higher "
    "8-sample mean pass rate, then to the later check"
)
SAME_WEIGHTS_RULE = (
    "an arm whose selected checkpoint holds the SFT weights this run loaded, every tensor "
    "compared bit for bit (not a file's SHA-256, which differs), is not evaluated again: run "
    "1's SFT results, pinned in sft_results, stand for it and are not an independent "
    "measurement"
)
POD = "pod"
REUSES_SFT = "reuses-run-1-sft"
NOT_EVALUATED = "not-evaluated"
BRIDGE_FILES = ("server.lease", "server.lease.lock", "client.lock")
BRIDGE_BOXES = ("outbox", "inbox", "done", "withdrawn", "rejected")
MAX_BRIDGE_TAR_BYTES = 512 * 1024 * 1024
"""The bridge label's bound: about 110 requests and replies are tens of MB."""


class WaitMarginStop(RewardTimeout):
    """No time or money is left for one full wait: raised before an exchange starts, so
    ``run_rl`` (or the control) stops cleanly as for a reward wait, with the latest weights
    saved; the worker reports ``wait_margin``."""


class WaitLimitStop(RewardTimeout):
    """The wall or cost stop was reached during a wait: the wait ends cleanly as a reward
    wait and the worker reports the limit (the finalization then does not wait for uploads,
    as for any wall or cost stop)."""


class BridgeLost(RewardTimeout):
    """The Mac's lease stayed stale for ``lost_bridge_seconds`` during a wait: no live Mac
    server is scoring. Ends the wait cleanly as a reward wait; the worker reports
    ``bridge_lost``."""


class WorkerRewards:
    """The RL loop's reward source: the bridge's, with the worker's bookkeeping around each
    wait (the GPU stamp and the margin check before, the end of ``awaiting_rewards`` after)."""

    def __init__(self, worker: Run2Worker, source: BridgeRewardSource) -> None:
        self.worker, self.source = worker, source

    def rewards(self, update: int, attempts: Sequence[Attempt]) -> list[AttemptReward]:
        self.worker.before_wait("reward", update)
        outcome = "failed"
        try:
            rewards = self.source.rewards(update, attempts)
            outcome = "scored"
            return rewards
        finally:
            self.worker.after_wait("reward", update, outcome)


class WorkerDevExecutor(BridgeDevExecutor):
    """The bridge's dev executor, with the worker's bookkeeping around each wait."""

    def __init__(
        self, worker: Run2Worker, client: BridgeClient, *, task_file_sha256: str, timeout: float
    ) -> None:
        super().__init__(client, task_file_sha256=task_file_sha256, timeout=timeout)
        self.worker = worker

    def __call__(self, pairs: Sequence[tuple[Task, str]]) -> list[bool]:
        if not pairs:
            return []
        index = self.calls if self._label is None else self._label
        self.worker.before_wait("dev", index)
        outcome = "failed"
        try:
            passed = super().__call__(pairs)
            outcome = "scored"
            return passed
        finally:
            self.worker.after_wait("dev", index, outcome)


def check_summary(record: CheckRecord, label: str) -> dict[str, Any]:
    """A dev check as the report lists it: its scores, its weights and their label."""
    dev = record.dev
    return {
        "check": dev.check,
        "label": label,
        "optimizer_steps": record.optimizer_steps,
        "selectable": record.selectable,
        "greedy_pass_at_1": dev.greedy_pass_at_1,
        "mean_pass_n8": dev.mean_pass_n8,
        "pass_at_8": dev.pass_at_8,
        "constant_baseline": dev.constant_baseline,
        "constant_pass_rate": dev.constant_pass_rate,
        "tasks": dev.tasks,
        "generation_seconds": dev.generation_seconds,
        "wait_seconds": dev.wait_seconds,
        "sha256": record.sha256,
    }


class Run2Worker(PosttrainWorker):
    """Post-training run 2 on one machine (see the module docstring)."""

    def __init__(
        self,
        spec: Specification,
        root: Path,
        *,
        bridge_root: Path | None = None,
        bridge_clock: Callable[[], float] = time.monotonic,
        bridge_sleep: Callable[[float], None] = time.sleep,
        **options: Any,
    ) -> None:
        if spec.run != "2":
            raise ValueError("Run2Worker runs post-training run 2 only")
        timeouts = spec.bridge_timeouts
        assert timeouts is not None  # A run-2 specification always has its bridge.
        self.timeouts = timeouts
        self.bridge_root = Path(spec.bridge["root"]) if bridge_root is None else bridge_root
        self.bridge_clock, self.bridge_sleep = bridge_clock, bridge_sleep
        self.bridge_client: BridgeClient | None = None
        self.awaiting: dict[str, Any] | None = None
        self.progress_unix: float | None = None
        self.lease_renewed_unix: float | None = None
        self.last_wait_check = 0.0
        self.rl_result: RLResult | None = None
        self.control_result: ControlResult | None = None
        self.produced: dict[str, Path] = {}
        self.uploaded: set[Path] = set()
        self.ending: str | None = None
        """Why the run ends early, if it does: the stop reason the worker reports."""
        self.arm_stop: dict[str, Any] | None = None
        """The stop that ended an arm's training but not the run (``ARM_STOP_RULE``)."""
        self.not_on_pod: dict[str, str] = {}
        """Arms that are not evaluated on this Pod, each with the reason."""
        self.stopped_stages: dict[str, str] = {}
        """Stages whose arm stopped early although the run went on, with the reason."""
        self.generation_floor_basis = "planning"
        super().__init__(spec, root, **options)
        self.report.update(
            rl=None,
            more_sft=None,
            arm_stop=None,
            selection={"endpoints": {"rule": SAME_WEIGHTS_RULE}},
            bridge={
                "root": str(self.bridge_root),
                "timeouts": asdict(timeouts),
                "liveness": RUN2_LIVENESS,
                "waits": [],
                "lease_last_renewed_unix": None,
                "lost": None,
                "margin_stop": None,
                "requests_sent": 0,
            },
        )
        self.publish()

    # The heartbeat while waiting ------------------------------------------------------------

    def heartbeat_fields(self, value: dict[str, Any]) -> None:
        if self.awaiting is not None and value["status"] == "running":
            value["status"] = posttrain_ops.AWAITING_REWARDS_STATUS
            value["awaiting"] = dict(self.awaiting)
        if self.progress_unix is not None:
            value["progress_unix"] = self.progress_unix

    def lease_renewed(self) -> float | None:
        """When the Mac last renewed its lease, if that is within the lease (Pod clock).

        The lease is ``server.lease`` in the bridge root, written by the Mac's remote program
        with the Pod's own time, so no clock skew enters. A missing, unreadable, malformed,
        future-dated or lapsed lease is ``None``: it proves nothing about a live Mac."""
        try:
            raw = read_bounded(self.bridge_root / LEASE_FILE, 4096)
            value = json.loads(raw) if raw is not None else None
        except (BridgeSchemaError, OSError, ValueError):
            return None
        if not isinstance(value, dict) or value.get("format") != LEASE_FORMAT:
            return None
        renewed, seconds = value.get("renewed"), value.get("seconds")
        if any(
            isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item)
            for item in (renewed, seconds)
        ):
            return None
        assert isinstance(renewed, int | float) and isinstance(seconds, int | float)
        age = self.clock() - float(renewed)
        if not 0 <= age < min(float(seconds), self.timeouts.lease_seconds):
            return None
        return float(renewed)

    def wait_sleep(self, seconds: float) -> None:
        """The bridge client's sleep: between looks at the inbox, observe the wait."""
        now = self.clock()
        if self.awaiting is None or now - self.last_wait_check >= HEARTBEAT_REFRESH_SECONDS:
            self.observe_wait(now)
        self.bridge_sleep(seconds)

    def observe_wait(self, now: float) -> None:
        """Heartbeat ``awaiting_rewards``; a fresh lease is progress; limits and a lost
        bridge end the wait."""
        self.last_wait_check = now
        wait = None if self.bridge_client is None else self.bridge_client.heartbeat()
        if wait is None:
            return
        self.awaiting = {
            key: wait[key]
            for key in ("request_id", "kind", "index", "count", "started_unix", "age_seconds")
        }
        renewed = self.lease_renewed()
        if renewed is not None:
            self.lease_renewed_unix = max(self.lease_renewed_unix or 0.0, renewed)
            self.progress_unix = max(self.progress_unix or 0.0, renewed)
            self.report["bridge"]["lease_last_renewed_unix"] = self.lease_renewed_unix
        reason = self.limits.stop_reason(pod_elapsed=max(0.0, now - self.pod_created))
        if reason is not None:
            # The margin keeps this from happening; should it, the wait ends as a reward wait
            # (RL and the control both stop cleanly on it, the control's trainer would read a
            # TrainingStop as its loader running out) and the run ends with the limit.
            self.stop_reason = self.ending = reason
            raise WaitLimitStop(f"the {reason} stop was reached while waiting on the Mac")
        silent_since = max(float(wait["started_unix"]), self.lease_renewed_unix or 0.0)
        if now - silent_since >= self.timeouts.lost_bridge_seconds:
            self.report["bridge"]["lost"] = {
                "request_id": wait["request_id"],
                "kind": wait["kind"],
                "index": wait["index"],
                "lease_last_renewed_unix": self.lease_renewed_unix,
                "silent_seconds": now - silent_since,
                "lost_bridge_seconds": self.timeouts.lost_bridge_seconds,
                "at_unix": now,
            }
            self.ending = BRIDGE_LOST_STOP
            raise BridgeLost(
                f"no Mac lease renewal for {now - silent_since:.0f} s while waiting on "
                f"{wait['request_id']}: the bridge is lost"
            )
        self.heartbeat()

    def margin_reason(self, wait_seconds: float) -> str | None:
        """The first stop one more wait of ``wait_seconds`` would reach, or ``None``."""
        now = self.clock()
        reason = self.limits.stop_reason(
            pod_elapsed=max(0.0, now - self.pod_created) + wait_seconds
        )
        if reason is None and now + wait_seconds >= self.reserve_unix:
            reason = RESERVE_STOP
        return reason

    def before_wait(self, kind: str, index: int) -> None:
        """The GPU work before a request just ended; refuse a wait that cannot fit."""
        now = self.clock()
        self.last_activity = now
        timeout = self.timeouts.timeout(kind)
        reason = self.margin_reason(timeout)
        if reason is not None:
            self.report["bridge"]["margin_stop"] = {
                "kind": kind,
                "index": index,
                "wait_seconds": timeout,
                "limit": reason,
                "at_unix": now,
            }
            self.ending = WAIT_MARGIN_STOP
            raise WaitMarginStop(
                f"a {kind} wait of up to {timeout:.0f} s would pass the {reason} stop"
            )
        self.heartbeat()

    def after_wait(self, kind: str, index: int, outcome: str) -> None:
        """The wait ended (answered or not): the next step is GPU work again."""
        now = self.clock()
        wait = self.awaiting
        self.awaiting = None
        self.last_activity = now
        client = self.bridge_client
        self.report["bridge"]["requests_sent"] = 0 if client is None else client.requests_sent
        self.report["bridge"]["waits"].append(
            {
                "kind": kind,
                "index": index,
                "outcome": outcome,
                "request_id": None if wait is None else wait["request_id"],
                "ended_unix": now,
                "seconds": None if wait is None else now - float(wait["started_unix"]),
            }
        )
        self.heartbeat()

    def should_stop(self) -> str | None:
        """``should_stop`` of RL and the control, read between updates: stamp the GPU work,
        stop at a limit, and keep one full wait free (``wait_margin``)."""
        now = self.clock()
        self.last_activity = now
        self.progress += 1
        reason = self.limits.stop_reason(pod_elapsed=max(0.0, now - self.pod_created))
        if reason is not None:
            self.stop_reason = reason
            self.ending = reason
            return reason
        margin = self.margin_reason(self.timeouts.wait_margin_seconds)
        if margin is not None:
            self.report["bridge"]["margin_stop"] = {
                "kind": "next update",
                "wait_seconds": self.timeouts.wait_margin_seconds,
                "limit": margin,
                "at_unix": now,
            }
            self.ending = WAIT_MARGIN_STOP
            return f"{WAIT_MARGIN_STOP}: one full wait would pass the {margin} stop"
        if now - self.last_heartbeat >= HEARTBEAT_REFRESH_SECONDS:
            self.heartbeat()
        return None

    # Inputs -----------------------------------------------------------------------------

    def client(self) -> BridgeClient:
        """The run's one bridge client, created at the first bridge stage."""
        if self.bridge_client is None:
            self.bridge_client = BridgeClient(
                self.bridge_root,
                poll_seconds=float(self.plan.bridge["pod_poll_seconds"]),
                clock=self.bridge_clock,
                wall=self.clock,
                sleep=self.wait_sleep,
            )
        return self.bridge_client

    def bridge_inputs(
        self, stage: StagePlan
    ) -> tuple[list[Task], list[Task], dict[str, Baseline], ConstantBaseline]:
        """The withheld RL tasks, the withheld dev subset, the RL tasks' facts and the dev
        subset's constant baseline from the facts file (checked against its pinned SHA-256)."""
        settings = stage.settings
        tasks = list(load_tasks(self.inputs[settings["tasks"]]))
        dev = list(load_tasks(self.inputs[settings["dev"]]))
        facts = load_facts(
            self.inputs[settings["facts"]].read_bytes(),
            sha256=str(self.plan.inputs[settings["facts"]]["sha256"]),
        )
        wanted = {task.id for task in tasks} | {task.id for task in dev}
        if set(facts) != wanted:
            raise ValueError(
                "the facts file does not cover exactly the RL tasks and the dev subset"
            )
        rl_facts = {task.id: facts[task.id] for task in tasks}
        baseline = constant_baseline_from_facts(facts, [task.id for task in dev])
        return tasks, dev, rl_facts, baseline

    def check_planned_tasks(self) -> None:
        """Run 1's checks, and run 2's task files: at least ``min_tasks`` kept tasks, the
        dev subset of exactly ``dev_tasks`` tasks in subset order, the facts covering both."""
        super().check_planned_tasks()
        rl = self.plan.rl
        for stage in self.plan.stages:
            if stage.kind != "rl":
                continue
            tasks, dev, _, _ = self.bridge_inputs(stage)
            if len(tasks) < int(rl["min_tasks"]):
                raise ValueError(f"{len(tasks)} RL tasks, fewer than the {rl['min_tasks']} kept")
            if len(tasks) < int(rl["tasks_per_update"]):
                raise ValueError("the RL tasks cannot fill one update")
            if len(dev) != int(rl["dev_tasks"]):
                raise ValueError(f"the dev subset holds {len(dev)} tasks, not {rl['dev_tasks']}")
            if [task.id for task in dev_subset(dev, len(dev))] != [task.id for task in dev]:
                raise ValueError("the dev-subset file is not in the subset's hash order")

    def checkpoint_path(self, reference: str) -> Path:
        if reference in (RL_REFERENCE, CONTROL_REFERENCE):
            if reference not in self.produced:
                raise RuntimeError(f"the {reference} weights were read before they were selected")
            return self.produced[reference]
        return super().checkpoint_path(reference)

    def dev_evaluator(self, dev: Sequence[Task], baseline: ConstantBaseline) -> BridgeDevEvaluator:
        rl = self.plan.rl
        executor = WorkerDevExecutor(
            self,
            self.client(),
            task_file_sha256=self.input_sha256(self.plan.stages[0].settings["dev"]),
            timeout=self.timeouts.dev_timeout_seconds,
        )
        return BridgeDevEvaluator(
            dev,
            self.encoder,
            self.tokenizer,
            executor,
            batch_size=int(rl["dev_batch_size"]),
            samples=int(rl["dev_samples"]),
            baseline=baseline,
            tick=self.dev_tick,
        )

    def dev_tick(self, tokens: int) -> None:
        """A dev check's generation batch ended: GPU work. Stamp it and keep the heartbeat
        live, so a check's generation (1,800 programs at batch 256, about 92 s at the planning
        speed) is never read as a hung worker by the idle rules or the heartbeat's staleness. No
        limit is enforced here: ``should_stop`` kept a full wait free before the update, and
        the check's own wait is refused by ``before_wait`` if it cannot fit."""
        now = self.clock()
        self.last_activity = now
        self.progress += 1
        if now - self.last_heartbeat >= HEARTBEAT_REFRESH_SECONDS:
            self.heartbeat()

    def input_sha256(self, name: str) -> str:
        return file_sha256(self.inputs[name])

    # Uploads ----------------------------------------------------------------------------

    def upload_weights(self, label: str, path: Path) -> list[dict[str, Any]]:
        """Export one weights-only label (``checkpoint.pt`` and ``record.json``) now."""
        self.uploaded.add(path.resolve())
        return self.export_label(label, [("checkpoint.pt", path)])

    def salvage(
        self,
        directory: Path,
        label: str,
        outputs: list[dict[str, Any]],
        *,
        stage: str,
        events: tuple[str, Path],
    ) -> None:
        """After a failure: upload the newest weights the run saved that are not in R2 yet
        (``run_rl`` and the control save the latest weights before an error propagates),
        then the arm's event log as the stage's label, which would otherwise be lost with
        the Pod. Neither may mask the error that ended the stage."""
        try:
            saved = sorted(path for path in directory.glob("*.pt") if path.is_file())
            fresh = [path for path in saved if path.resolve() not in self.uploaded]
            if fresh:
                outputs += self.upload_weights(label, fresh[-1])
        except Exception as error:
            self.report.setdefault("salvage_errors", []).append(
                f"{label}: {type(error).__name__}: {str(error)[:200]}"
            )
        try:
            if events[1].exists():
                outputs += self.export_label(stage, [events])
        except Exception as error:
            self.report.setdefault("salvage_errors", []).append(
                f"{stage}: {type(error).__name__}: {str(error)[:200]}"
            )

    # The RL stage -----------------------------------------------------------------------

    def run_rl(self, stage: StagePlan) -> list[dict[str, Any]]:
        tasks, dev, facts, baseline = self.bridge_inputs(stage)
        parent_name = stage.settings["parent"]
        parent = self.checkpoint_path(parent_name)
        self.phase = f"{stage.name}-load"
        model = load_weights(
            parent,
            tokenizer_hash=self.tokenizer.sha256,
            expected_sha256=str(self.plan.inputs[parent_name]["sha256"]),
            device=self.device,
        )
        model.set_fused_attention(self.plan.rl["attention"] == "fused")
        self.last_activity = self.clock()
        self.phase = stage.name
        config = self.plan.rl_config()
        output = self.root / stage.name
        record: dict[str, Any] = {
            "parent": parent_name,
            "parent_sha256": file_sha256(parent),
            "tasks": len(tasks),
            "task_file_sha256": self.input_sha256(stage.settings["tasks"]),
            "dev_tasks": len(dev),
            "dev_file_sha256": self.input_sha256(stage.settings["dev"]),
            "facts_sha256": self.input_sha256(stage.settings["facts"]),
            "dev_constant_baseline": asdict(baseline),
            "config": config.to_json(),
            "attention": self.plan.rl["attention"],
            "checks": [],
            "status": "running",
        }
        self.report["rl"] = record
        self.publish()
        outputs: list[dict[str, Any]] = []

        def on_check(check: CheckRecord) -> None:
            label = f"rl-check-{check.dev.check:04d}"
            outputs.extend(self.upload_weights(label, Path(check.checkpoint)))
            record["checks"].append(check_summary(check, label))
            self.publish()

        source = BridgeRewardSource(
            self.client(),
            task_file_sha256=record["task_file_sha256"],
            timeout=self.timeouts.reward_timeout_seconds,
        )
        try:
            result = run_rl_loop(
                model,
                tasks,
                encoder=self.encoder,
                decoder=self.tokenizer,
                rewards=WorkerRewards(self, source),
                evaluator=self.dev_evaluator(dev, baseline),
                output=output,
                run_id="posttrain-2-rl",
                tokenizer_hash=self.tokenizer.sha256,
                config=config,
                should_stop=self.should_stop,
                on_check=on_check,
                facts=facts,
            )
        except BaseException:
            record["status"] = "failed"
            self.salvage(
                output / "checkpoints",
                "rl-latest-weights",
                outputs,
                stage=stage.name,
                events=("events.jsonl", output / "events.jsonl"),
            )
            raise
        finally:
            del model
            self.release()
        self.last_activity = self.clock()
        self.rl_result = result
        record.update(self.rl_record(result))
        if result.stop_checkpoint is not None:
            path = Path(result.stop_checkpoint["checkpoint"])
            outputs += self.upload_weights("rl-stop-weights", path)
            record["stop_weights_label"] = "rl-stop-weights"
        labels = {row["check"]: row["label"] for row in record["checks"]}
        self.report["selection"]["rl"] = self.selection(result.checks, result.selected, labels)
        outputs += self.export_label(
            stage.name,
            [("result.json", output / "result.json"), ("events.jsonl", output / "events.jsonl")],
        )
        self.measure_rl(result)
        kind = self.stop_kind(result.stop_reason)
        if kind == LIMIT or result.selected is None:
            # A wall or cost stop ends the run; so does an RL that scored no check to select.
            raise PilotStop(self.end_reason(result.stop_reason, result.stop_detail))
        if kind is not None:
            self.stop_arm(stage, RL_REFERENCE, kind, result.stop_reason, result.stop_detail)
            self.not_on_pod[CONTROL_REFERENCE] = (
                f"the control did not run: RL ended with {self.arm_stop_reason()}, and the "
                "control's dev checks need the Mac"
            )
        self.choose_endpoint(RL_REFERENCE, result.selected, labels)
        return outputs

    def rl_record(self, result: RLResult) -> dict[str, Any]:
        seconds = dict(result.seconds)
        return {
            "status": "completed",
            "stop_reason": result.stop_reason,
            "stop_detail": result.stop_detail,
            "updates": result.updates,
            "optimizer_steps": result.optimizer_steps,
            "skipped_updates": result.skipped_updates,
            "seconds": seconds,
            "busy_seconds": result.busy_seconds,
            "busy_rule": (
                "GPU-busy seconds are rollout generation plus training steps; compile (none: "
                "RL does not compile), dev-check generation, the Mac waits and checkpoint saves "
                "are excluded and recorded separately"
            ),
            "compile_seconds": 0.0,
            "excluded_seconds": {
                key: seconds[key]
                for key in ("reward_wait", "dev_generation", "dev_wait", "checkpoint")
            },
            "precision": result.precision,
            "attention": result.attention,
            "schedule_sha256": result.schedule_sha256,
            "seeds_used": result.seeds_used,
            "stop_checkpoint_sha256": (
                None if result.stop_checkpoint is None else result.stop_checkpoint["sha256"]
            ),
        }

    def measure_rl(self, result: RLResult) -> None:
        """Re-base the later projections on RL's measured rates (priced at the token caps)."""
        rl = self.plan.rl
        seconds = result.seconds
        attempts = result.updates * int(rl["tasks_per_update"]) * int(rl["group_size"])
        programs = int(rl["dev_tasks"]) * (int(rl["dev_samples"]) + 1)
        checks = len(result.checks)
        changes: dict[str, Any] = {"rl_busy_seconds": result.busy_seconds}
        if attempts and seconds["rollout_generation"] > 0:
            changes["rollout_tokens_per_second"] = (
                attempts * int(rl["max_new_tokens"]) / seconds["rollout_generation"]
            )
        if attempts and seconds["training_step"] > 0:
            changes["training_slots_per_second"] = (
                attempts * self.plan.model.context / seconds["training_step"]
            )
        if result.updates:
            changes["reward_wait_seconds_per_update"] = seconds["reward_wait"] / result.updates
        basis = {name: "measured" for name in ("rollout", "dev", "mac")}
        if checks and seconds["dev_generation"] > 0:
            rate = checks * programs * DEV_TOKENS / seconds["dev_generation"]
            changes["dev_generation_tokens_per_second"] = rate
            current = self.rates.generation_tokens_per_second
            if int(rl["dev_batch_size"]) == self.plan.generation_batch and rate < current:
                # Dev checks generated at the evaluation batch and the same 512-token cap
                # measure the evaluation's speed: the evaluation stages are then projected at
                # the lower of the planning speed and that rate (run 2 has no probe of its
                # own), so, as in run 1, a measurement only makes the projected-finish rule
                # stricter. Run 2's dev checks generate at batch 256 and its evaluation at 64,
                # so there the evaluation stays at its batch's planning speed until its own
                # files are observed.
                changes["generation_tokens_per_second"] = rate
                basis["generation"] = "measured-dev-checks"
        if checks:
            changes["dev_wait_seconds_per_check"] = seconds["dev_wait"] / checks
        self.rates = replace(self.rates, **changes, basis={**self.rates.basis, **basis})

    def observe_generation(self, entry: EvalFile, seconds: float) -> None:
        """Run 1's rule, the lower of a floor and the evaluation files' rate so far, with
        run 2's floor: there is no probe, so the floor is the rate the evaluation was
        projected at when it began (the planning speed, or RL's dev checks where slower).
        The projection never loosens past it."""
        batch = self.plan.generation_batch
        if batch not in self.probe_rates:
            self.probe_rates[batch] = self.rates.generation_tokens_per_second
            self.generation_floor_basis = self.rates.basis.get("generation", "planning")
        super().observe_generation(entry, seconds)
        if self.rates.generation_tokens_per_second == self.probe_rates[batch]:
            basis = {**self.rates.basis, "generation": self.generation_floor_basis}
            self.rates = replace(self.rates, basis=basis)
        if "generation_observed" in self.report:
            self.report["generation_observed"]["floor_basis"] = self.generation_floor_basis

    def end_reason(self, stop: str, detail: str) -> str:
        """The run's stop reason when an arm's stop ends it: ``bridge_lost``, ``wait_margin``
        or a limit where the worker raised it, else the arm's own stop (``reward_wait`` for a
        timeout on either side, ``reward_host_failure``)."""
        if self.ending is not None:
            return self.ending
        if stop == EXTERNAL:
            return str(self.stop_reason or detail or EXTERNAL)
        return stop

    def selection(
        self,
        checks: Sequence[CheckRecord],
        selected: CheckRecord | None,
        labels: Mapping[int, str],
    ) -> dict[str, Any]:
        """Which check an arm selected, by what: the rule and every check's scores."""
        rows = [check_summary(check, labels.get(check.dev.check, "")) for check in checks]
        chosen = None
        if selected is not None:
            chosen = check_summary(selected, labels.get(selected.dev.check, ""))
        return {"rule": SELECTION_RULE, "selected": chosen, "checks": rows}

    # Arm stops, endpoints and the stages that follow ------------------------------------

    def stop_kind(self, stop: str) -> str | None:
        """How an arm's stop is handled (``ARM_STOP_RULE``): ``limit`` for a wall or cost stop
        (it ends the run), ``margin`` for ``wait_margin``, ``channel`` for a stop of the Mac's
        side (a reward-wait timeout on either side, a host failure, a lost bridge), ``None``
        for a result."""
        if self.stop_reason is not None:
            return LIMIT  # should_stop or a wait reached the wall or the cost stop.
        if self.ending == WAIT_MARGIN_STOP:
            return MARGIN
        if self.ending == BRIDGE_LOST_STOP or stop in CHANNEL_STOPS:
            return CHANNEL
        if stop in ARM_STOPS or self.ending is not None:
            return LIMIT  # Defensive: an external stop with no limit recorded ends the run.
        return None

    def arm_stop_reason(self) -> str:
        assert self.arm_stop is not None
        return str(self.arm_stop["reason"])

    def stop_arm(self, stage: StagePlan, arm: str, kind: str, stop: str, detail: str) -> None:
        """Record that ``arm``'s training ended early but the run goes on."""
        reason = self.end_reason(stop, detail)
        self.arm_stop = {
            "arm": arm,
            "kind": kind,
            "reason": reason,
            "arm_stop_reason": stop,
            "detail": detail,
            "at_unix": self.clock(),
            "rule": ARM_STOP_RULE,
        }
        self.report["arm_stop"] = self.arm_stop
        self.stopped_stages[stage.name] = reason
        self.ending = None  # The run goes on: the stages that need no Mac wait follow.
        self.publish()

    def after_stages(self) -> None:
        """An arm that stopped early leaves the run ``stopped`` with its reason, after the
        stages that could still run."""
        if self.arm_stop is not None:
            raise PilotStop(self.arm_stop_reason())

    def choose_endpoint(
        self, arm: str, selected: CheckRecord | None, labels: Mapping[int, str]
    ) -> None:
        """Record ``arm``'s endpoint: whether its selected weights are the SFT weights the run
        loaded (bit for bit), and so whether it is evaluated here or reuses run 1's SFT
        results. An arm that is not evaluated at all is not compared."""
        endpoints = self.report["selection"]["endpoints"]
        assert selected is not None
        label = labels.get(selected.dev.check, "")
        entry: dict[str, Any] = {
            "selected": label,
            "checkpoint_sha256": selected.sha256,
            "same_weights_as_sft": None,
            "comparison": None,
            "evaluation": POD,
            "independent": True,
            "reason": None,
            "reuses": None,
        }
        if arm in self.not_on_pod:
            entry.update(evaluation=NOT_EVALUATED, independent=None, reason=self.not_on_pod[arm])
        else:
            started = self.clock()
            self.phase = f"{arm}-same-weights"
            sft = self.inputs["sft-weights"]
            found = compare_to_loaded(Path(selected.checkpoint), sft)
            self.last_activity = self.clock()
            entry["same_weights_as_sft"] = found.same
            entry["comparison"] = {
                "method": (
                    "every tensor's name, dtype, shape and bytes against the SFT weights as "
                    "the run loaded them, on the CPU"
                ),
                "sft_weights_sha256": file_sha256(sft),
                "tensors": found.tensors,
                "first_difference": found.first_difference,
                "seconds": self.clock() - started,
            }
            if found.same:
                reason = (
                    f"{label} holds the SFT weights bit for bit: run 1's SFT results stand for "
                    "it, not an independent measurement"
                )
                self.not_on_pod[arm] = reason
                entry.update(
                    evaluation=REUSES_SFT,
                    independent=False,
                    reason=reason,
                    reuses={"endpoint": "sft", "results": self.reused_results(arm)},
                )
            else:
                self.produced[arm] = Path(selected.checkpoint)
        endpoints[arm] = entry
        self.publish()

    def reused_results(self, arm: str) -> dict[str, dict[str, Any]]:
        """Each artifact ``arm`` would have produced here, mapped to run 1's pinned artifact
        that stands for it (``Specification.sft_results``)."""
        pins = self.plan.sft_results
        mapped = {f"arc-post-{arm}/arc.json": {"run_1": SFT_ARC, **pins[SFT_ARC]}}
        for name in SFT_RESULT_FILES:
            source = f"eval-sft-{name}/completions.jsonl"
            mapped[f"eval-{arm}-{name}/completions.jsonl"] = {"run_1": source, **pins[source]}
        return mapped

    def skip_reason(self, stage: StagePlan) -> str | None:
        """Why ``stage`` does not run on this Pod, or ``None``: the control after RL's early
        stop; an evaluation stage of an arm not evaluated here; an ARC stage none of whose
        checkpoints is. (Asked only of stages that have not run.)"""
        if stage.kind == "more-sft":
            return self.not_on_pod.get(CONTROL_REFERENCE)
        if stage.kind == "eval":
            return self.not_on_pod.get(str(stage.settings["checkpoint"]))
        if stage.kind == "arc":
            references = list(stage.settings["checkpoints"])
            skipped = [self.not_on_pod.get(reference) for reference in references]
            if all(skipped):
                return "; ".join(
                    f"{ref}: {why}" for ref, why in zip(references, skipped, strict=True)
                )
        return None

    def evaluated_stage(self, stage: StagePlan) -> StagePlan:
        """``stage`` as it runs here: an ARC stage keeps only the checkpoints evaluated here."""
        if stage.kind != "arc":
            return stage
        kept = [ref for ref in stage.settings["checkpoints"] if ref not in self.not_on_pod]
        return replace(stage, settings={**stage.settings, "checkpoints": kept})

    def remaining_stages(self, stage: StagePlan) -> list[StagePlan]:
        """The projected-finish rule prices only what will run: no skipped stage, and no ARC
        entry of an arm not evaluated here."""
        return [
            self.evaluated_stage(later)
            for later in super().remaining_stages(stage)
            if self.skip_reason(later) is None
        ]

    def projected_finish(self, stage: StagePlan) -> None:
        """The projected-finish rule, admitting each arm: after a margin stop in the control
        both arms are evaluable, and when the evaluation of both does not fit but RL's alone
        might, the control is left unevaluated (``not-evaluated``, the reason recorded) and
        RL's evaluation is projected again on its own. Otherwise as in run 1."""
        try:
            super().projected_finish(stage)
        except PilotStop:
            if not self.drop_control_for_rl(stage):
                raise
            self.stop_reason = None
            super().projected_finish(stage)  # RL alone; raises if that does not fit either.

    def drop_control_for_rl(self, stage: StagePlan) -> bool:
        """Leave the control unevaluated so RL's evaluation may run alone; ``True`` when that
        changed what remains (only after the control's margin stop, while both arms were
        still to be evaluated here)."""
        stop = self.arm_stop
        if (
            stop is None
            or (stop["arm"], stop["kind"]) != (CONTROL_REFERENCE, MARGIN)
            or CONTROL_REFERENCE in self.not_on_pod
            or RL_REFERENCE in self.not_on_pod
            or stage.kind not in {"arc", "eval"}
        ):
            return False
        before = [later.name for later in self.remaining_stages(stage)]
        reason = (
            "not evaluated: after the control's margin stop the projected-finish rule did "
            "not admit both arms' evaluation; RL's alone was projected again"
        )
        self.not_on_pod[CONTROL_REFERENCE] = reason
        entry = self.report["selection"]["endpoints"].get(CONTROL_REFERENCE)
        if isinstance(entry, dict):
            entry.update(evaluation=NOT_EVALUATED, independent=None, reason=reason)
        self.report["arm_stop"]["control_not_admitted"] = {
            "at_stage": stage.name,
            "remaining_before": before,
            "remaining_after": [later.name for later in self.remaining_stages(stage)],
        }
        self.publish()
        return True

    def run_stage(self, stage: StagePlan) -> None:
        """Skip a stage that does not run here (recording why); mark an arm that stopped
        early as stopped, though the run goes on."""
        reason = self.skip_reason(stage)
        record = self.stage_by_name[stage.name]
        if reason is not None:
            record.update(status="skipped", skip_reason=reason)
            self.publish()
            return
        super().run_stage(stage)
        if stage.name in self.stopped_stages:
            record.update(status="stopped", stop_reason=self.stopped_stages[stage.name])
            self.publish()

    def run_arc(self, stage: StagePlan) -> list[dict[str, Any]]:
        evaluated = self.evaluated_stage(stage)
        left_out = {
            ref: self.not_on_pod[ref]
            for ref in stage.settings["checkpoints"]
            if ref in self.not_on_pod
        }
        if left_out:
            self.stage_by_name[stage.name]["checkpoints_not_evaluated"] = left_out
        return super().run_arc(evaluated)

    # The control stage ------------------------------------------------------------------

    def run_more_sft(self, stage: StagePlan) -> list[dict[str, Any]]:
        rl = self.rl_result
        if rl is None:
            raise RuntimeError("the control runs only after RL measured its busy seconds")
        start = next((check for check in rl.checks if check.dev.check == 0), None)
        if start is None:
            raise RuntimeError("the control's start is RL's update-0 check, which RL did not take")
        tasks, dev, facts, baseline = self.bridge_inputs(stage)
        parent = self.checkpoint_path(stage.settings["parent"])
        config = self.plan.control_config(attention=rl.attention)
        output = self.root / stage.name
        record: dict[str, Any] = {
            "parent": stage.settings["parent"],
            "parent_sha256": file_sha256(parent),
            "budget_seconds": rl.busy_seconds,
            "budget_rule": (
                "GPU-busy seconds equal to RL's rollout generation plus training steps, within "
                "the tolerance; dev checks, checkpoint saves and the Mac waits are excluded in "
                "both arms"
            ),
            "start_check": check_summary(start, f"rl-check-{start.dev.check:04d}"),
            "config": config.to_json(),
            "checks": [],
            "status": "running",
        }
        self.report["more_sft"] = record
        self.phase = stage.name
        self.publish()
        outputs: list[dict[str, Any]] = []

        def on_check(check: CheckRecord) -> None:
            label = f"more-sft-check-{check.dev.check:05d}"
            outputs.extend(self.upload_weights(label, Path(check.checkpoint)))
            record["checks"].append(check_summary(check, label))
            self.publish()

        run_dir = output / "more-sft"
        try:
            result = run_more_sft(
                parent=parent,
                tasks=tasks,
                encoder=self.encoder,
                tokenizer_hash=self.tokenizer.sha256,
                budget_seconds=rl.busy_seconds,
                start=start,
                evaluator=self.dev_evaluator(dev, baseline),
                output=output,
                run_id="more-sft",
                device=str(self.device),
                config=config,
                should_stop=self.should_stop,
                on_check=on_check,
                facts=facts,
            )
        except BaseException:
            record["status"] = "failed"
            self.salvage(
                run_dir / "checkpoints",
                "more-sft-latest-weights",
                outputs,
                stage=stage.name,
                events=("control-events.jsonl", run_dir / "control-events.jsonl"),
            )
            raise
        finally:
            self.release()
        self.last_activity = self.clock()
        self.control_result = result
        record.update(
            status="completed",
            stop_reason=result.stop_reason,
            stop_detail=result.stop_detail,
            busy_seconds=result.busy_seconds,
            mismatch_seconds=result.mismatch_seconds,
            mismatch_fraction=result.mismatch_fraction,
            within_tolerance=result.within_tolerance,
            tolerance=result.tolerance,
            updates=result.updates,
            passes=result.passes,
            rows_per_pass=result.rows_per_pass,
            rows_per_update=result.rows_per_update,
            dropped=result.dropped,
            seconds=dict(result.seconds),
            precision=result.precision,
            attention=config.attention,
            optimizer_foreach=config.optimizer_foreach,
        )
        if result.stop_checkpoint is not None:
            path = Path(result.stop_checkpoint["checkpoint"])
            outputs += self.upload_weights("more-sft-stop-weights", path)
            record["stop_weights_label"] = "more-sft-stop-weights"
        labels = {row["check"]: row["label"] for row in record["checks"]}
        labels[0] = record["start_check"]["label"]
        self.report["selection"]["more-sft"] = self.selection(
            result.checks, result.selected, labels
        )
        outputs += self.export_label(
            stage.name,
            [
                ("control-result.json", run_dir / "control-result.json"),
                ("control-events.jsonl", run_dir / "control-events.jsonl"),
            ],
        )
        kind = self.stop_kind(result.stop_reason)
        if kind == LIMIT:
            raise PilotStop(self.end_reason(result.stop_reason, result.stop_detail))
        if kind is not None:
            self.stop_arm(stage, CONTROL_REFERENCE, kind, result.stop_reason, result.stop_detail)
            if kind == CHANNEL:
                self.not_on_pod[CONTROL_REFERENCE] = (
                    f"not evaluated: the control ended with {self.arm_stop_reason()}, a stop of "
                    "the Mac's side"
                )
        self.choose_endpoint(CONTROL_REFERENCE, result.selected, labels)
        return outputs

    # Finalization -----------------------------------------------------------------------

    def before_finalize(self) -> None:
        """Upload the bridge directories as one label (``bridge``), so a late reply is kept."""
        try:
            if self.bridge_client is not None:
                self.bridge_client.close()
            if not self.bridge_root.is_dir():
                return
            path = self.root / "artifacts" / "bridge" / "bridge.tar"
            path.parent.mkdir(parents=True, exist_ok=True)
            archived = self.archive_bridge(path)
            self.current = None
            self.export_label("bridge", [("bridge.tar", path)])
            self.report["bridge"]["archived_at"] = utc_now()
            self.report["bridge"]["archive"] = archived
        except Exception as error:  # The finalization must run whatever happens here.
            self.report["bridge"]["archive_error"] = f"{type(error).__name__}: {str(error)[:200]}"
            self.publish()

    def archive_bridge(self, path: Path) -> dict[str, Any]:
        """Write the bridge directories' regular files into the tar at ``path``.

        The Mac may still push a late reply, and renews ``server.lease`` by atomic rename,
        while this runs: each file is read once through one descriptor opened without
        following a link, so a file replaced meanwhile is archived whole (the old or the new
        bytes, never a torn mix) and one removed after the listing is left out and named,
        instead of losing the whole label. A link, a FIFO or any other non-regular file is
        never followed or read."""
        members = [self.bridge_root / name for name in BRIDGE_FILES]
        for box in BRIDGE_BOXES:
            directory = self.bridge_root / box
            if directory.is_dir() and not directory.is_symlink():
                members += sorted(directory.iterdir())
        total, files, vanished = 0, 0, []
        with tarfile.open(path, "w") as archive:
            for member in members:
                name = str(member.relative_to(self.bridge_root))
                try:
                    snapshot = regular_snapshot(member, MAX_BRIDGE_TAR_BYTES - total)
                except FileNotFoundError:
                    if name not in BRIDGE_FILES:  # A missing lock or lease is no news.
                        vanished.append(name)
                    continue
                if snapshot is None:
                    continue
                data, mtime = snapshot
                total += len(data)
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(data), int(mtime), 0o600
                archive.addfile(info, io.BytesIO(data))
                files += 1
        return {"files": files, "bytes": total, "vanished": vanished}


def regular_snapshot(path: Path, limit: int) -> tuple[bytes, float] | None:
    """The bytes and modification time of the regular file at ``path``, read through one
    descriptor opened without following a link; ``None`` for a link or any other non-regular
    file. A file removed before it is opened raises ``FileNotFoundError``; one longer than
    ``limit`` bytes raises ``ValueError`` (the bridge label's bound)."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError:  # A link (ELOOP under O_NOFOLLOW) or an unreadable entry.
        return None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            return None
        chunks, size = [], 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise ValueError("the bridge directories exceed the label's bound")
            chunks.append(chunk)
        return b"".join(chunks), info.st_mtime
    finally:
        os.close(descriptor)
