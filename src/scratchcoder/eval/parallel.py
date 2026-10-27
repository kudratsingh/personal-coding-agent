"""Run sandbox calls in parallel, retrying host failures and never scoring them.

Each call is one candidate program with its tests, executed through an ``execute``
function such as :func:`scratchcoder.eval.harness.isolated_execution`. A result is a
verdict on the candidate only if its status is one of the runner's candidate statuses
and the host did not stall (:func:`scratchcoder.eval.harness.is_host_failure`). Anything
else is retried up to ``retries`` times with :func:`~scratchcoder.eval.harness.host_backoff`;
a call still failing then is *unresolved*: reported as such, never as a pass or a fail.
"""

import time
from collections.abc import Callable, Hashable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass

from scratchcoder.eval.harness import HOST_RETRIES, host_backoff, is_host_failure
from scratchcoder.sandbox.runner import Result

Execute = Callable[[str, str, str], Result]
OVERSIZED = "oversized"
"""The runner refused the input before starting a container (the candidate with its tests
is larger than the sandbox input limit). Attributed to the candidate: scored as failed."""


class HostUnavailableError(RuntimeError):
    """Too many consecutive calls stayed unresolved; the sandbox host looks down."""


@dataclass(frozen=True)
class Call:
    key: Hashable
    code: str
    tests: str
    mode: str


@dataclass(frozen=True)
class Outcome:
    """The end of one call. ``resolved`` is false only for a host failure that outlasted the
    retries; then ``status`` is the last host status and ``passed`` is false but means
    nothing. ``elapsed_seconds`` adds up every try; ``host_failures`` counts the failed tries."""

    key: Hashable
    resolved: bool
    status: str
    passed: bool
    stdout: str
    elapsed_seconds: float
    host_failures: int


def resolve_call(
    call: Call,
    execute: Execute,
    *,
    retries: int = HOST_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> Outcome:
    failures, elapsed = 0, 0.0
    while True:
        try:
            result = execute(call.code, call.tests, call.mode)
        except ValueError:
            return Outcome(call.key, True, OVERSIZED, False, "", elapsed, failures)
        elapsed += result.elapsed_seconds
        if not is_host_failure(result):
            return Outcome(
                call.key, True, result.status, result.passed, result.stdout, elapsed, failures
            )
        failures += 1
        if failures > retries:
            status = "host_stalled" if result.host_stalled else result.status
            return Outcome(call.key, False, status, False, "", elapsed, failures)
        sleep(host_backoff(failures))


def run_calls(
    calls: Iterable[Call],
    execute: Execute,
    *,
    on_outcome: Callable[[Outcome], None],
    workers: int = 8,
    retries: int = HOST_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
    max_consecutive_unresolved: int | None = None,
) -> int:
    """Execute ``calls`` on ``workers`` threads and hand each outcome to ``on_outcome`` on
    the calling thread, as it completes. Returns the number of outcomes delivered.

    At most ``2 * workers`` calls are in flight. If ``max_consecutive_unresolved`` outcomes
    in a row are unresolved, no further call starts; the calls already running finish and
    are delivered, then :class:`HostUnavailableError` is raised.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    pending: Iterator[Call] = iter(calls)
    delivered, streak, stopped = 0, 0, False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running: set[Future[Outcome]] = set()

        def refill() -> None:
            while not stopped and len(running) < 2 * workers:
                call = next(pending, None)
                if call is None:
                    return
                running.add(pool.submit(resolve_call, call, execute, retries=retries, sleep=sleep))

        refill()
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                running.discard(future)
                outcome = future.result()
                on_outcome(outcome)
                delivered += 1
                streak = 0 if outcome.resolved else streak + 1
                if max_consecutive_unresolved is not None and streak >= max_consecutive_unresolved:
                    stopped = True
            refill()
    if stopped:
        raise HostUnavailableError(
            f"{streak} consecutive calls stayed unresolved after {retries} retries each; "
            "stopped starting new calls"
        )
    return delivered
