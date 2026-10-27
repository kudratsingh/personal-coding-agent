"""Tests for RL attempts, rewards and the local reward source.

The unit tests inject a fake ``score`` in place of the reward channel, so they need no
Docker and run no candidate code. The tests at the bottom are gated behind
``SCRATCHCODER_SANDBOX_TESTS=1`` and score reference and wrong solutions in the real pinned
container through ``LocalRewardSource``; with ``SCRATCHCODER_POOL_TRAIN`` naming the pool's
train file they also use a few real train tasks.
"""

import ast
import json
import math
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from scratchcoder.eval.tasks import Task, load_tasks
from scratchcoder.rl.rewards import (
    NO_PROGRAM,
    Attempt,
    AttemptReward,
    LocalRewardSource,
    RewardHostFailure,
    RewardTimeout,
    check_rewards,
    reward_from_result,
)
from scratchcoder.sandbox.reward import (
    COMPLETED,
    HOST_ERROR,
    HOST_TIMEOUT,
    NOT_PARSEABLE,
    UNAVAILABLE,
    FunctionChecks,
    RewardResult,
    is_parseable,
    parse_task,
)


def task(identifier: str, hidden: str = "assert f(1) == 2", mode: str = "function") -> Task:
    return Task(
        id=identifier,
        prompt="Add one.",
        public_tests="assert f(3) == 4" if mode == "function" else hidden,
        hidden_tests=hidden,
        source="synthetic",
        license="fixture",
        written_by="human",
        split="train",
        mode=mode,
        solution="def f(x):\n    return x + 1",
    )


def scored(reward: float = 1.0, *, status: str = COMPLETED, **fields: object) -> RewardResult:
    return RewardResult(status, reward, 1, 1, **fields)  # type: ignore[arg-type]


def host(status: str = HOST_TIMEOUT) -> RewardResult:
    return RewardResult(status, None, 0, 1, detail="stalled")


class Recorder:
    """A fake ``score_attempt`` that plays scripted results per (task, code)."""

    def __init__(self, plays: dict[tuple[str, str], list[RewardResult]] | None = None) -> None:
        self.plays = plays or {}
        self.calls: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def __call__(self, item: Task, code: str) -> RewardResult:
        with self.lock:
            self.calls.append((item.id, code))
            queue = self.plays.get((item.id, code))
            if queue:
                return queue.pop(0) if len(queue) > 1 else queue[0]
        return scored(len(code) % 2 / 1.0)


def source(score: Callable[[Task, str], RewardResult], **options: object) -> LocalRewardSource:
    sleeps: list[float] = []
    defaults: dict[str, object] = {"timeout": None, "sleep": sleeps.append}
    defaults.update(options)
    return LocalRewardSource([task("a"), task("b")], score=score, **defaults)  # type: ignore[arg-type]


def test_attempt_and_reward_validation() -> None:
    assert Attempt("t", 0, None, 0).code is None
    for fields in (("", 0, None, 0), ("t", -1, None, 0), ("t", 0, 3, 0), ("t", 0, "x", -1)):
        with pytest.raises(ValueError):
            Attempt(*fields)
    for reward, status in ((1.5, COMPLETED), (-0.1, COMPLETED), (0.5, HOST_TIMEOUT), (0.5, "")):
        with pytest.raises(ValueError):
            AttemptReward(reward, status)
    with pytest.raises(ValueError):
        AttemptReward(True, COMPLETED)
    with pytest.raises(ValueError):
        AttemptReward(0.5, COMPLETED, ("",))
    # A non-finite reward is representable so the loop can stop on it.
    assert math.isnan(AttemptReward(math.nan, COMPLETED).reward)


def test_reward_from_result_maps_statuses_and_never_scores_a_host_failure() -> None:
    reward = reward_from_result(
        RewardResult(COMPLETED, 0.5, 1, 2, ("inspect",), unfinished=1), seconds=2.0
    )
    assert reward == AttemptReward(0.5, COMPLETED, ("inspect",), 2.0, 1)
    assert reward_from_result(RewardResult(NOT_PARSEABLE, None), seconds=0.0).reward == 0.0
    for status in (HOST_TIMEOUT, UNAVAILABLE, HOST_ERROR):
        with pytest.raises(RewardHostFailure):
            reward_from_result(host(status), seconds=0.0)
    with pytest.raises(RewardHostFailure):
        reward_from_result(RewardResult("odd", None), seconds=0.0)


def test_no_program_scores_zero_without_running() -> None:
    recorder = Recorder()
    replies = source(recorder).rewards(1, [Attempt("a", 0, None, 5), Attempt("b", 1, "x", 3)])
    assert replies[0] == AttemptReward(0.0, NO_PROGRAM)
    assert recorder.calls == [("b", "x")]
    assert source(recorder).rewards(2, [Attempt("a", 0, None, 0)]) == [
        AttemptReward(0.0, NO_PROGRAM)
    ]


def test_rewards_keep_attempt_order_under_the_thread_pool() -> None:
    def slow(item: Task, code: str) -> RewardResult:
        time.sleep(0.002 * (int(code) % 5))
        return scored(int(code) / 100)

    attempts = [Attempt("a" if index % 2 else "b", index, str(index), 1) for index in range(40)]
    replies = source(slow, workers=8).rewards(3, attempts)
    assert [reply.reward for reply in replies] == [index / 100 for index in range(40)]


def test_host_failures_are_retried_with_bounded_backoff() -> None:
    sleeps: list[float] = []
    recorder = Recorder({("a", "x"): [host(), host(UNAVAILABLE), host(HOST_ERROR), scored(0.5)]})
    reward_source = source(
        recorder, sleep=sleeps.append, backoff_seconds=1.0, max_backoff_seconds=3.0
    )
    replies = reward_source.rewards(1, [Attempt("a", 0, "x", 1)])
    assert replies[0].reward == 0.5 and replies[0].status == COMPLETED
    assert sleeps == [1.0, 2.0, 3.0]  # doubling, capped at max_backoff_seconds
    assert len(recorder.calls) == 4


def test_a_persistent_host_failure_raises_instead_of_scoring_zero() -> None:
    recorder = Recorder({("a", "x"): [host()]})
    with pytest.raises(RewardHostFailure):
        source(recorder, host_retries=3).rewards(1, [Attempt("a", 0, "x", 1)])
    assert len(recorder.calls) == 4  # the first try and three retries
    recorder = Recorder({("b", "y"): [host()]})
    with pytest.raises(RewardHostFailure):
        source(recorder, host_retries=0).rewards(1, [Attempt("b", 0, "y", 1)])
    assert len(recorder.calls) == 1


def test_a_result_without_reward_is_a_host_failure_unless_unparseable() -> None:
    recorder = Recorder({("a", "x"): [RewardResult(NOT_PARSEABLE, None)]})
    assert source(recorder).rewards(1, [Attempt("a", 0, "x", 1)])[0].status == NOT_PARSEABLE
    recorder = Recorder({("a", "x"): [RewardResult("odd", None)]})
    with pytest.raises(RewardHostFailure):
        source(recorder, host_retries=1).rewards(1, [Attempt("a", 0, "x", 1)])


def test_a_slow_call_raises_reward_timeout() -> None:
    release = threading.Event()

    def blocked(item: Task, code: str) -> RewardResult:
        release.wait(5)
        return scored()

    started = time.monotonic()
    attempts = [Attempt("a", index, "x", 1) for index in range(4)]
    with pytest.raises(RewardTimeout):
        source(blocked, timeout=0.2, workers=2).rewards(9, attempts)
    assert time.monotonic() - started < 2
    release.set()


def test_no_retry_starts_past_the_deadline() -> None:
    now = [0.0]
    recorder = Recorder({("a", "x"): [host()]})
    reward_source = source(
        recorder, timeout=1.5, clock=lambda: now[0], backoff_seconds=2.0, max_backoff_seconds=2.0
    )
    with pytest.raises(RewardTimeout):
        reward_source.rewards(1, [Attempt("a", 0, "x", 1)])
    assert len(recorder.calls) == 1


def test_local_source_rejects_bad_inputs() -> None:
    recorder = Recorder()
    with pytest.raises(ValueError):
        source(recorder).rewards(1, [Attempt("zzz", 0, "x", 1)])
    with pytest.raises(ValueError):
        source(recorder).rewards(-1, [])
    for options in ({"timeout": 0.0}, {"workers": 0}, {"host_retries": -1}):
        with pytest.raises(ValueError):
            source(recorder, **options)
    with pytest.raises(ValueError):
        LocalRewardSource([task("a"), task("a")], timeout=None)
    with pytest.raises(TypeError):
        LocalRewardSource([task("a")])  # type: ignore[call-arg]  # timeout has no default


def test_check_rewards_enforces_the_contract() -> None:
    attempts = [Attempt("a", 0, None, 0), Attempt("a", 1, "x", 1)]
    check_rewards(attempts, [AttemptReward(0.0, NO_PROGRAM), AttemptReward(1.0, COMPLETED)])
    with pytest.raises(ValueError):
        check_rewards(attempts, [AttemptReward(0.0, NO_PROGRAM)])
    with pytest.raises(ValueError):
        check_rewards(attempts, [AttemptReward(0.5, COMPLETED), AttemptReward(1.0, COMPLETED)])


# --------------------------------------------------------------------------------------
# Docker-backed tests (opt-in; disposable worker only).
# --------------------------------------------------------------------------------------

integration = pytest.mark.skipif(
    os.environ.get("SCRATCHCODER_SANDBOX_TESTS") != "1",
    reason="requires explicitly enabled disposable Docker worker",
)


def io_task(identifier: str, cases: list[tuple[str, str]], solution: str) -> Task:
    rows = json.dumps([{"input": given, "output": wanted} for given, wanted in cases])
    return Task(
        id=identifier,
        prompt="Echo the sum.",
        public_tests=json.dumps([{"input": cases[0][0], "output": cases[0][1]}]),
        hidden_tests=rows,
        source="synthetic",
        license="fixture",
        written_by="human",
        split="train",
        mode="io",
        solution=solution,
    )


RL_SOURCES = ("nvidia/OpenCodeInstruct", "BAAI/TACO", "google-research-datasets/mbpp")
AUTHORED = [
    task("auth/fn", "assert f(1) == 2\nassert f(5) == 6\nassert f(-3) == -2"),
    io_task(
        "auth/io",
        [("1 2\n", "3\n"), ("10 5\n", "15\n"), ("-4 4\n", "0\n")],
        "a, b = map(int, input().split())\nprint(a + b)",
    ),
]


def assert_reference_and_wrong(tasks: list[Task], wrong: dict[str, str]) -> None:
    reward_source = LocalRewardSource(tasks, timeout=600.0, workers=4)
    attempts: list[Attempt] = []
    for item in tasks:
        assert item.solution is not None
        attempts.append(Attempt(item.id, 0, item.solution, 1))
        attempts.append(Attempt(item.id, 1, wrong[item.id], 1))
        attempts.append(Attempt(item.id, 2, None, 0))
    replies = reward_source.rewards(1, attempts)
    for index, item in enumerate(tasks):
        reference, bad, missing = replies[3 * index : 3 * index + 3]
        assert reference.status == COMPLETED and reference.reward == 1.0, (item.id, reference)
        assert bad.status not in {NO_PROGRAM} and bad.reward < 1.0, (item.id, bad)
        assert missing == AttemptReward(0.0, NO_PROGRAM)
        assert reference.seconds > 0


@integration
def test_live_local_source_scores_reference_and_wrong_solutions() -> None:
    wrong = {"auth/fn": "def f(x):\n    return x - 1", "auth/io": "print(input())"}
    assert_reference_and_wrong(AUTHORED, wrong)


def called_name(item: Task) -> str:
    """The name the first hidden assertion calls, such as ``f`` in ``f(1) == 2``."""
    parsed = parse_task(item)
    assert isinstance(parsed, FunctionChecks)
    node = ast.parse(parsed.checks[0].call, mode="eval").body
    while isinstance(node, (ast.Call, ast.Attribute, ast.Subscript)):
        node = node.func if isinstance(node, ast.Call) else node.value
    assert isinstance(node, ast.Name)
    return node.id


def wrong_solution(item: Task) -> str:
    if item.mode == "io":
        return "print('no')"
    return f"def {called_name(item)}(*args, **kwargs):\n    return 'definitely not the answer'"


@integration
@pytest.mark.skipif(
    not os.environ.get("SCRATCHCODER_POOL_TRAIN"),
    reason="set SCRATCHCODER_POOL_TRAIN to the pool's train.jsonl to use real train tasks",
)
def test_live_local_source_on_real_train_tasks() -> None:
    """The first parseable train task of each RL candidate source, in file order."""
    chosen: dict[str, Task] = {}
    for item in load_tasks(Path(os.environ["SCRATCHCODER_POOL_TRAIN"])):
        if item.source not in RL_SOURCES or item.source in chosen or item.solution is None:
            continue
        if item.split == "train" and is_parseable(item):
            chosen[item.source] = item
        if len(chosen) == len(RL_SOURCES):
            break
    tasks = list(chosen.values())
    assert len(tasks) == len(RL_SOURCES)
    assert_reference_and_wrong(tasks, {item.id: wrong_solution(item) for item in tasks})
