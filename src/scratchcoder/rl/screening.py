"""RL task screening: which candidates the RL run may train on.

Screening samples 8 completions of every candidate from the SFT checkpoint
(``scratchcoder.rl.rollout.screening_rollouts``), scores them through a reward source and
then applies the keep rule. A candidate is kept when all of these hold, checked in this
order (the first that fails is its drop reason):

1. ``not_parseable``: every hidden test is in a form the reward channel can score
   (``scratchcoder.sandbox.reward.is_parseable``);
2. ``constant_baseline``: the best constant answer (``degenerate_baseline``: a fixed
   return value or printed line, a value copied from the public tests, or echoing the
   first argument or the input) earns less than the threshold, 0.5 by default. Because an
   advantage is a reward minus its group's mean, a task that pays a constant answer well
   pays the policy for answering without solving;
3. ``equal_rewards``: the 8 rewards are not all equal. An all-equal group has zero
   advantage everywhere and contributes no gradient;
4. ``over_cap``: it is among the first ``cap`` (1,000) eligible tasks in ID-hash order.

ID-hash order sorts by ``sha256("rl-candidate-v1|" + task ID)``, the salt the candidate
holdout used, so the order can be recomputed from the IDs alone. If fewer than ``floor``
(320, enough for 10 updates of 32 without reuse) are kept, the result is a floor result
for the owner: it carries no task list. See ``docs/rl.md``.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from scratchcoder.eval.tasks import Task
from scratchcoder.rl.rewards import Attempt, AttemptReward, RewardSource, check_rewards
from scratchcoder.rl.withheld import refuse_withheld
from scratchcoder.sandbox.reward import degenerate_baseline, is_parseable

DEFAULT_THRESHOLD = 0.5
"""Drop a candidate whose best constant answer earns at least this reward (owner ruling)."""
MAX_KEPT = 1000
FLOOR = 320
SAMPLES = 8
ORDER_SALT = "rl-candidate-v1"
REASONS = ("not_parseable", "constant_baseline", "equal_rewards", "over_cap")
KEPT = "kept"
FLOOR_OUTCOME = "floor"


def hash_order(task_id: str, *, salt: str = ORDER_SALT) -> bytes:
    """The ID-hash sort key: ``sha256(f"{salt}|{task_id}")``."""
    return hashlib.sha256(f"{salt}|{task_id}".encode()).digest()


@dataclass(frozen=True)
class Baseline:
    """A task's best constant answer, or ``None`` values when it is not parseable."""

    parseable: bool
    reward: float | None
    answer: str | None


def baseline_of(task: Task) -> Baseline:
    if not is_parseable(task):
        return Baseline(False, None, None)
    best = degenerate_baseline(task)
    assert best is not None  # parseable tasks always have a baseline
    return Baseline(True, best.reward, best.answer)


def require_train_tasks(
    tasks: Sequence[Task],
    *,
    references: bool = False,
    baselines: Mapping[str, Baseline] | None = None,
) -> dict[str, Baseline]:
    """Refuse a task list RL or the control must not train on; return each task's baseline.

    Every task must be a train-split task (RL executes its hidden tests and the control
    trains on its reference, so a dev, test or holdout task would leak it), and every hidden
    test must be parseable by the reward channel (an unparseable task scores 0 for any
    program, so its groups would be skipped silently). With ``references`` every task must
    also carry a reference solution (the control's targets). Raises ``ValueError`` naming
    the first offender of each kind.

    ``baselines`` are the facts the Mac computed from the real hidden tests
    (``scratchcoder.rl.facts``), for a Pod that holds none: they must cover exactly these
    tasks, and replace the tests as the source of "parseable" and of the baseline reward.
    Without them a task whose hidden tests are withheld is refused.
    """
    problems: list[str] = []
    if baselines is None:
        refuse_withheld(tuple(tasks), "RL")
        found = {task.id: baseline_of(task) for task in tasks}
    else:
        if set(baselines) != {task.id for task in tasks}:
            raise ValueError("the baselines must cover exactly the RL tasks")
        found = dict(baselines)
    for label, bad in (
        ("not train-split tasks", [task for task in tasks if task.split != "train"]),
        ("unparseable by the reward channel", [t for t in tasks if not found[t.id].parseable]),
        (
            "without a reference solution",
            [task for task in tasks if references and task.solution is None],
        ),
    ):
        if bad:
            problems.append(f"{len(bad)} {label}, e.g. {bad[0].id}")
    if problems:
        raise ValueError("RL tasks must be parseable train tasks: " + "; ".join(problems))
    return found


@dataclass(frozen=True)
class ScreenedTask:
    """One candidate's screening evidence and its drop reason (``None`` when kept)."""

    id: str
    source: str
    parseable: bool
    baseline: float | None
    baseline_answer: str | None
    rewards: tuple[float, ...]
    statuses: tuple[str, ...]
    audit_flagged: int
    reason: str | None


@dataclass(frozen=True)
class ScreeningResult:
    """The keep rule's outcome. ``kept`` is the task list in ID-hash order, or ``None``
    when fewer than ``floor`` tasks were kept (``outcome == "floor"``); ``eligible`` always
    lists every task that passed rules 1 to 3."""

    outcome: str
    kept: tuple[str, ...] | None
    eligible: tuple[str, ...]
    tasks: tuple[ScreenedTask, ...]
    threshold: float
    cap: int
    floor: int
    samples: int

    def counts(self) -> dict[str, Any]:
        """Counts by reason, by source, and by source and reason."""
        reasons: Counter[str] = Counter()
        by_source: dict[str, Counter[str]] = {}
        for task in self.tasks:
            label = task.reason or KEPT
            reasons[label] += 1
            by_source.setdefault(task.source, Counter())[label] += 1
        parseable = sum(task.parseable for task in self.tasks)
        at_threshold = sum(
            task.baseline is not None and task.baseline >= self.threshold for task in self.tasks
        )
        return {
            "candidates": len(self.tasks),
            "parseable": parseable,
            "baseline_at_or_above_threshold": at_threshold,
            "eligible_before_equal_rewards": parseable - at_threshold,
            "eligible": len(self.eligible),
            "kept": 0 if self.kept is None else len(self.kept),
            "by_reason": {label: reasons[label] for label in (KEPT, *REASONS)},
            "by_source": {
                source: {label: counter[label] for label in (KEPT, *REASONS)}
                for source, counter in sorted(by_source.items())
            },
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "format": "rl-screening-v1",
            "outcome": self.outcome,
            "rule": {
                "threshold": self.threshold,
                "cap": self.cap,
                "floor": self.floor,
                "samples": self.samples,
                "order": f"sha256('{ORDER_SALT}|' + task id)",
                "reasons": list(REASONS),
            },
            "counts": self.counts(),
            "kept": None if self.kept is None else list(self.kept),
            "eligible": list(self.eligible),
            "tasks": [asdict(task) for task in self.tasks],
        }


def screen(
    tasks: Sequence[Task],
    attempts: Sequence[Attempt],
    rewards: Sequence[AttemptReward],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    cap: int = MAX_KEPT,
    floor: int = FLOOR,
    samples: int = SAMPLES,
) -> ScreeningResult:
    """Apply the keep rule to scored screening attempts (aligned ``attempts``/``rewards``).

    Every candidate needs exactly ``samples`` attempts with sample indices 0 to
    ``samples - 1``. Unparseable candidates are dropped whatever their rewards.
    """
    if not 0 <= threshold <= 1 or isinstance(threshold, bool):
        raise ValueError("threshold must lie in [0, 1]")
    for name, value in (("cap", cap), ("floor", floor), ("samples", samples)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("candidate IDs must be unique")
    check_rewards(attempts, rewards)
    by_task: dict[str, dict[int, AttemptReward]] = {task.id: {} for task in tasks}
    for attempt, reward in zip(attempts, rewards, strict=True):
        if attempt.task_id not in by_task:
            raise ValueError(f"attempt for a task that is not a candidate: {attempt.task_id}")
        if attempt.sample in by_task[attempt.task_id] or not 0 <= attempt.sample < samples:
            raise ValueError(f"duplicate or out-of-range sample for {attempt.task_id}")
        by_task[attempt.task_id][attempt.sample] = reward
    screened: list[ScreenedTask] = []
    eligible: list[str] = []
    for task in sorted(tasks, key=lambda item: hash_order(item.id)):
        group = by_task[task.id]
        if len(group) != samples:
            raise ValueError(f"{task.id} has {len(group)} of {samples} screening attempts")
        ordered = [group[index] for index in range(samples)]
        values = tuple(float(item.reward) for item in ordered)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{task.id} has a non-finite screening reward")
        baseline = baseline_of(task)
        reason: str | None = None
        if not baseline.parseable:
            reason = "not_parseable"
        elif baseline.reward is not None and baseline.reward >= threshold:
            reason = "constant_baseline"
        elif len(set(values)) == 1:
            reason = "equal_rewards"
        elif len(eligible) >= cap:
            reason = "over_cap"
        if reason is None or reason == "over_cap":
            eligible.append(task.id)
        screened.append(
            ScreenedTask(
                task.id,
                task.source,
                baseline.parseable,
                baseline.reward,
                baseline.answer,
                values,
                tuple(item.status for item in ordered),
                sum(bool(item.audit_flags) for item in ordered),
                reason,
            )
        )
    kept = tuple(item.id for item in screened if item.reason is None)
    return ScreeningResult(
        FLOOR_OUTCOME if len(kept) < floor else KEPT,
        None if len(kept) < floor else kept,
        tuple(eligible),
        tuple(screened),
        float(threshold),
        cap,
        floor,
        samples,
    )


def score_screening(
    source: RewardSource, attempts: Sequence[Attempt], *, chunk: int = 256
) -> list[AttemptReward]:
    """Score screening attempts in batches of ``chunk``; batch ``k`` is sent as update ``k``.

    The update number only tags the request, so a remote source can reject a stale reply.
    """
    if type(chunk) is not int or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    rewards: list[AttemptReward] = []
    for number, start in enumerate(range(0, len(attempts), chunk)):
        batch = attempts[start : start + chunk]
        reply = list(source.rewards(number, batch))
        check_rewards(batch, reply)
        rewards.extend(reply)
    return rewards


def prescreen(tasks: Sequence[Task], *, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """The rules that need no generation (1 and 2), counted by source; runs no code."""
    by_source: dict[str, Counter[str]] = {}
    for task in tasks:
        baseline = baseline_of(task)
        counter = by_source.setdefault(task.source, Counter())
        counter["candidates"] += 1
        if not baseline.parseable:
            counter["not_parseable"] += 1
            continue
        counter["parseable"] += 1
        assert baseline.reward is not None
        if baseline.reward >= threshold:
            counter["constant_baseline"] += 1
        else:
            counter["eligible_before_equal_rewards"] += 1
    labels = (
        "candidates",
        "not_parseable",
        "parseable",
        "constant_baseline",
        "eligible_before_equal_rewards",
    )
    totals = {label: sum(counter[label] for counter in by_source.values()) for label in labels}
    return {
        "threshold": threshold,
        **totals,
        "by_source": {
            source: {label: counter[label] for label in labels}
            for source, counter in sorted(by_source.items())
        },
    }
