"""Shared fixtures for the RL reward bridge tests: tasks, fake clocks and fake scorers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scratchcoder.eval.tasks import Task
from scratchcoder.rl.bridge import BridgeClient
from scratchcoder.rl.bridge_mac import BridgeServer, DirectoryTransport, PodTransport
from scratchcoder.rl.fakes import FakeRewardSource
from scratchcoder.rl.rewards import Attempt, AttemptReward

RL_SHA = "a" * 64
DEV_SHA = "b" * 64
SESSION = "0123456789ab"


def train_task(identifier: str) -> Task:
    return Task(
        id=identifier,
        prompt="Add one.",
        public_tests="assert f(11) == 12",
        hidden_tests="assert f(12) == 13\nassert f(15) == 16",
        source="synthetic",
        license="fixture",
        written_by="human",
        split="train",
        solution="def f(x):\n    return x + 1",
    )


def dev_task(identifier: str) -> Task:
    return Task(
        id=identifier,
        prompt="Compute f.",
        public_tests="assert f(3) == 11",
        hidden_tests="assert f(6) == 17",
        source="synthetic",
        license="fixture",
        written_by="human",
        split="dev",
    )


TRAIN = [train_task(f"t{index}") for index in range(6)]
DEV_TASKS = [dev_task(f"d{index}") for index in range(4)]


def attempts(task_ids: Sequence[str] = ("t0", "t1"), group: int = 4) -> list[Attempt]:
    """``group`` attempts per task; sample 0 has no program."""
    return [
        Attempt(task_id, sample, None if sample == 0 else "x" * sample, sample + 1)
        for task_id in task_ids
        for sample in range(group)
    ]


class FakeClock:
    """A monotonic clock that only moves when ``sleep`` is called; each sleep then runs the
    hooks, which play the Mac's part between the Pod's looks at its inbox."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []
        self.hooks: list[Callable[[], object]] = []

    def clock(self) -> float:
        return self.now

    def wall(self) -> float:
        return 1_800_000_000.0 + self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds
        for hook in list(self.hooks):
            hook()


@dataclass
class CountingRewards:
    """``FakeRewardSource`` that counts the calls it scored."""

    inner: FakeRewardSource = field(default_factory=FakeRewardSource)
    calls: list[tuple[int, int]] = field(default_factory=list)
    error: Exception | None = None

    def rewards(self, update: int, items: Sequence[Attempt]) -> list[AttemptReward]:
        self.calls.append((update, len(items)))
        if self.error is not None:
            raise self.error
        return self.inner.rewards(update, items)


@dataclass
class CountingDev:
    """A Mac-side dev executor: a program passes when its length is even."""

    calls: list[list[tuple[str, str]]] = field(default_factory=list)
    error: Exception | None = None

    def __call__(self, pairs: Sequence[tuple[Task, str]]) -> list[bool]:
        self.calls.append([(task.id, code) for task, code in pairs])
        if self.error is not None:
            raise self.error
        return [len(code) % 2 == 0 for _, code in pairs]


def pod_client(root: Path, clock: FakeClock | None = None, **options: Any) -> BridgeClient:
    if clock is None:
        return BridgeClient(root, session=SESSION, **options)
    return BridgeClient(
        root,
        session=SESSION,
        clock=clock.clock,
        wall=clock.wall,
        sleep=clock.sleep,
        **options,
    )


def mac_server(
    pod_root: Path,
    state: Path,
    *,
    transport: PodTransport | None = None,
    rewards: Any = None,
    dev: Any = None,
    sleeps: list[float] | None = None,
    **options: Any,
) -> BridgeServer:
    recorded = sleeps if sleeps is not None else []
    return BridgeServer(
        transport or DirectoryTransport(pod_root),
        reward_tasks=TRAIN,
        dev_tasks=DEV_TASKS,
        pod_task_sha256={"reward": RL_SHA, "dev": DEV_SHA},
        state=state,
        reward_source=rewards if rewards is not None else CountingRewards(),
        dev_execute=dev if dev is not None else CountingDev(),
        sleep=recorded.append,
        **options,
    )
