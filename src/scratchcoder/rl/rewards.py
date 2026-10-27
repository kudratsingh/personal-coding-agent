"""Attempts, their rewards, and the reward sources the RL loop and screening read from.

An ``Attempt`` is what crosses from the GPU side to the machine that runs code: the task
ID, the sample index and the program extracted from the completion with chat-v1 rules
(``scratchcoder.data.chat.extract_program``), or ``None`` when the completion has no
closed python block. Only the program travels, never the completion text.

A ``RewardSource`` scores a list of attempts and returns one ``AttemptReward`` per
attempt, in the same order. Two rules hold for every implementation:

- an attempt whose ``code`` is ``None`` scores 0 with status ``no_program`` and is never
  executed;
- a host failure (the reward channel's ``host_timeout``, ``unavailable`` or ``host_error``,
  which include its host-stall outcome) is never a reward. The source retries it and, if it
  cannot resolve it, raises ``RewardHostFailure``. It never returns 0 for a host failure.

A source may also raise ``RewardTimeout`` when one ``rewards`` call takes longer than its
caller allows; the RL loop then saves the policy's weights and stops with the reason
``reward_wait``. See ``docs/walkthroughs/rl-update.md``.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.reward import HOST_FAILURES, NOT_PARSEABLE, RewardResult, score_attempt

NO_PROGRAM = "no_program"
"""Status of an attempt whose completion held no closed python block: reward 0, not run."""
DEFAULT_WORKERS = 8
DEFAULT_HOST_RETRIES = 3


class RewardError(RuntimeError):
    """A reward could not be produced; the RL loop stops cleanly instead of guessing one."""


class RewardTimeout(RewardError):
    """One ``rewards`` call took longer than the caller's timeout."""


class RewardHostFailure(RewardError):
    """A host failure persisted through every retry; it is never scored."""


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


@dataclass(frozen=True)
class Attempt:
    """One sampled completion of a task, as the reward side sees it.

    ``sample`` is the index within the task's group (0 to G - 1). ``completion_tokens``
    counts the generated tokens, the stop token included when one was produced.
    """

    task_id: str
    sample: int
    code: str | None
    completion_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("an attempt needs a nonempty task ID")
        if type(self.sample) is not int or self.sample < 0:
            raise ValueError("an attempt's sample index must be a nonnegative integer")
        if self.code is not None and not isinstance(self.code, str):
            raise ValueError("an attempt's code is text or None")
        if type(self.completion_tokens) is not int or self.completion_tokens < 0:
            raise ValueError("completion_tokens must be a nonnegative integer")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Attempt:
        if not isinstance(value, Mapping) or set(value) != ATTEMPT_KEYS:
            raise ValueError(f"an attempt has exactly the keys {sorted(ATTEMPT_KEYS)}")
        return cls(value["task_id"], value["sample"], value["code"], value["completion_tokens"])


ATTEMPT_KEYS = frozenset({"task_id", "sample", "code", "completion_tokens"})


@dataclass(frozen=True)
class AttemptReward:
    """The reward of one attempt.

    ``reward`` is the fraction of the task file's hidden tests that passed, in [0, 1].
    ``status`` is the reward channel's status string (or ``no_program``); it is never a
    host-failure status. ``audit_flags`` are the channel's static-audit flags for the code.
    ``seconds`` is the wall time spent scoring this attempt, retries included.
    ``unfinished`` counts hidden checks or cases that produced no result in time, so an
    attempt with ``unfinished > 0`` is counted as a timeout in the RL monitoring.

    A non-finite ``reward`` is representable on purpose: the RL loop stops on it (reason
    ``non_finite``) instead of crashing, so a corrupted reward never reaches the gradient.
    A finite reward outside [0, 1] is rejected here.
    """

    reward: float
    status: str
    audit_flags: tuple[str, ...] = ()
    seconds: float = 0.0
    unfinished: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)):
            raise ValueError("a reward must be a number")
        if math.isfinite(self.reward) and not 0.0 <= self.reward <= 1.0:
            raise ValueError("a finite reward must lie in [0, 1]")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("a reward needs a status")
        if self.status in HOST_FAILURES:
            raise ValueError("a host failure is never a reward")
        if not isinstance(self.audit_flags, tuple) or any(
            not isinstance(flag, str) or not flag for flag in self.audit_flags
        ):
            raise ValueError("audit flags must be a tuple of nonempty strings")
        _finite_nonnegative(self.seconds, "seconds")
        if type(self.unfinished) is not int or self.unfinished < 0:
            raise ValueError("unfinished must be a nonnegative integer")

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["audit_flags"] = list(self.audit_flags)
        return value

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> AttemptReward:
        if not isinstance(value, Mapping) or set(value) != REWARD_KEYS:
            raise ValueError(f"a reward has exactly the keys {sorted(REWARD_KEYS)}")
        flags = value["audit_flags"]
        if not isinstance(flags, list):
            raise ValueError("audit_flags must be a list")
        return cls(
            value["reward"], value["status"], tuple(flags), value["seconds"], value["unfinished"]
        )


REWARD_KEYS = frozenset({"reward", "status", "audit_flags", "seconds", "unfinished"})


class RewardSource(Protocol):
    """Scores attempts; returns one reward per attempt, in the same order.

    ``update`` names the batch (the RL update number, or a screening batch number) so a
    remote implementation can tag its request and reject a stale reply. Implementations
    score a ``None``-code attempt as 0 with status ``no_program`` without running it,
    retry host failures and raise ``RewardHostFailure`` when they persist, and may raise
    ``RewardTimeout``.
    """

    def rewards(self, update: int, attempts: Sequence[Attempt]) -> Sequence[AttemptReward]: ...


def reward_from_result(result: RewardResult, *, seconds: float) -> AttemptReward:
    """The ``AttemptReward`` of a scored reward-channel result (never a host failure).

    An unparseable task scores 0 with status ``not_parseable``: screening drops such tasks
    by their hidden tests, and RL never sees them.
    """
    if result.is_host_failure:
        raise RewardHostFailure(f"host failure {result.status}: {result.detail}")
    if result.reward is None:
        if result.status != NOT_PARSEABLE:
            raise RewardHostFailure(f"result {result.status} carries no reward")
        return AttemptReward(0.0, NOT_PARSEABLE, result.audit_flags, seconds)
    return AttemptReward(
        float(result.reward), result.status, result.audit_flags, seconds, result.unfinished
    )


def check_rewards(attempts: Sequence[Attempt], rewards: Sequence[AttemptReward]) -> None:
    """Reject a reply that breaks the ``RewardSource`` contract (a bug, not a stop)."""
    if len(rewards) != len(attempts):
        raise ValueError(f"{len(rewards)} rewards returned for {len(attempts)} attempts")
    for attempt, reward in zip(attempts, rewards, strict=True):
        if not isinstance(reward, AttemptReward):
            raise ValueError("a reward source must return AttemptReward values")
        if attempt.code is None and (reward.reward != 0.0 or reward.status != NO_PROGRAM):
            raise ValueError("an attempt without a program must score 0 with no_program")


class LocalRewardSource:
    """Scores attempts on this machine through ``scratchcoder.sandbox.reward``.

    Each attempt runs ``score_attempt(task, code)`` with the channel's default budget
    (``budget=None``: 8 s for a function task, 2 s plus 0.25 s per case up to 60 s for an
    I/O task) in a pool of ``workers`` threads. A host failure is retried up to
    ``host_retries`` times after the first try (so 3 retries are 4 tries; the channel's
    ``score_with_host_retries(attempts=3)`` counts tries, 3 in all), waiting
    ``backoff_seconds``, then twice that, and so on, never more than
    ``max_backoff_seconds``; if it persists the call raises ``RewardHostFailure``.
    ``timeout`` bounds one ``rewards`` call in seconds
    (``None`` for no bound): when it passes, attempts not yet started are cancelled and
    ``RewardTimeout`` is raised. Attempts already running finish inside the channel's own
    deadline and remove their containers. There is no default timeout: the caller derives
    it from the budgets, the channel's 15 s host slack per try, the tries and the worker
    count (``docs/rl.md`` gives the worst case: 256 attempts at the 26.75 s budget on 8
    workers wait about 856 s before slack).

    ``score``, ``sleep`` and ``clock`` are injectable for tests.
    """

    def __init__(
        self,
        tasks: Iterable[Task],
        *,
        timeout: float | None,
        workers: int = DEFAULT_WORKERS,
        host_retries: int = DEFAULT_HOST_RETRIES,
        backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 8.0,
        score: Callable[[Task, str], RewardResult] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.tasks: dict[str, Task] = {}
        for task in tasks:
            if task.id in self.tasks:
                raise ValueError(f"duplicate task ID: {task.id}")
            self.tasks[task.id] = task
        if timeout is not None and _finite_nonnegative(timeout, "timeout") <= 0:
            raise ValueError("timeout must be positive seconds or None")
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        if type(host_retries) is not int or host_retries < 0:
            raise ValueError("host_retries must be a nonnegative integer")
        _finite_nonnegative(backoff_seconds, "backoff_seconds")
        if _finite_nonnegative(max_backoff_seconds, "max_backoff_seconds") < backoff_seconds:
            raise ValueError("max_backoff_seconds must be at least backoff_seconds")
        self.timeout = None if timeout is None else float(timeout)
        self.workers = workers
        self.host_retries = host_retries
        self.backoff_seconds = float(backoff_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self._score = score or (lambda task, code: score_attempt(task, code))
        self._sleep = sleep
        self._clock = clock

    def backoff(self, retry: int) -> float:
        """The wait before retry ``retry`` (1-based): doubling, capped."""
        return min(self.max_backoff_seconds, self.backoff_seconds * 2.0 ** (retry - 1))

    def _score_one(
        self, task: Task, code: str, deadline: float | None, cancelled: threading.Event
    ) -> AttemptReward:
        started = self._clock()
        retry = 0
        while True:
            result = self._score(task, code)
            if not result.is_host_failure and not (
                result.reward is None and result.status != NOT_PARSEABLE
            ):
                return reward_from_result(result, seconds=max(0.0, self._clock() - started))
            retry += 1
            if retry > self.host_retries:
                raise RewardHostFailure(
                    f"task {task.id}: host failure {result.status} persisted through "
                    f"{self.host_retries} retries ({result.detail})"
                )
            delay = self.backoff(retry)
            if cancelled.is_set() or (deadline is not None and self._clock() + delay > deadline):
                raise RewardTimeout(f"task {task.id}: no time left to retry {result.status}")
            self._sleep(delay)

    def rewards(self, update: int, attempts: Sequence[Attempt]) -> list[AttemptReward]:
        if type(update) is not int or update < 0:
            raise ValueError("update must be a nonnegative integer")
        unknown = sorted({attempt.task_id for attempt in attempts} - set(self.tasks))
        if unknown:
            raise ValueError(f"attempts for unknown tasks: {unknown[:3]}")
        results: list[AttemptReward | None] = [
            AttemptReward(0.0, NO_PROGRAM) if attempt.code is None else None for attempt in attempts
        ]
        pending = [index for index, result in enumerate(results) if result is None]
        if not pending:
            return [result for result in results if result is not None]
        deadline = None if self.timeout is None else self._clock() + self.timeout
        cancelled = threading.Event()
        executor = ThreadPoolExecutor(max_workers=min(self.workers, len(pending)))
        futures: dict[Future[AttemptReward], int] = {}
        try:
            for index in pending:
                attempt = attempts[index]
                assert attempt.code is not None
                future = executor.submit(
                    self._score_one, self.tasks[attempt.task_id], attempt.code, deadline, cancelled
                )
                futures[future] = index
            done, not_done = wait(
                futures,
                timeout=None if deadline is None else max(0.0, deadline - self._clock()),
                return_when=FIRST_EXCEPTION,
            )
            failed = [future for future in done if future.exception() is not None]
            if failed:
                error = failed[0].exception()
                assert error is not None
                raise error
            if not_done:
                raise RewardTimeout(
                    f"update {update}: {len(done)} of {len(futures)} attempts scored "
                    f"within {self.timeout} s"
                )
            for future, index in futures.items():
                results[index] = future.result()
        finally:
            cancelled.set()
            executor.shutdown(wait=False, cancel_futures=True)
        scored = [result for result in results if result is not None]
        assert len(scored) == len(attempts)
        return scored
