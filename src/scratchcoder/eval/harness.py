"""One recorded evaluation path, with explicit infrastructure failure semantics."""

import hashlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.eval.metrics import mean_pass_rate, paired_bootstrap, pass_at_k
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.budget import io_seconds
from scratchcoder.sandbox.runner import Limits, Result, run

SCORED_STATUSES = frozenset({"passed", "failed", "timeout", "output_limit", "resource_limit"})
"""Runner statuses that are a verdict on the candidate. Anything else is a host failure."""
HOST_RETRIES = 3
"""How many times a host failure is retried before the attempt is given up as unresolved."""
HOST_BACKOFF_SECONDS = 2.0
HOST_BACKOFF_MAX_SECONDS = 30.0


class ContextLimitError(ValueError):
    """The configured model cannot fit a complete task prompt."""


@dataclass(frozen=True)
class EvaluationConfig:
    samples: int = 1
    seed: int = 0
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    bootstrap_samples: int = 2000
    prompt_mode: str = "base"

    def __post_init__(self) -> None:
        if self.prompt_mode not in {"base", "instruction"}:
            raise ValueError("prompt_mode must be base or instruction")
        if self.samples < 1 or self.max_new_tokens < 1 or not 0 <= self.seed < 2**63:
            raise ValueError("invalid evaluation counts/seed")
        if not 0 <= self.temperature < float("inf") or not 0 < self.top_p <= 1:
            raise ValueError("invalid sampling settings")
        if self.top_k is not None and self.top_k < 1 or self.bootstrap_samples < 2:
            raise ValueError("invalid top-k/bootstrap settings")
        if self.temperature == 0 and self.samples != 1:
            raise ValueError(
                "greedy decoding has one deterministic attempt; use sampling for repeats"
            )


def execution_limits(tests: str, mode: str) -> Limits:
    """The evaluation runner's limits for one call: the pool verification's deadline rule.

    A function call gets the default 2 s. An I/O call runs every case in sequence under one
    deadline of 2 s plus 0.25 s per case, at most 60 s
    (:func:`scratchcoder.sandbox.budget.io_seconds`), which is the deadline the pool's
    reference solutions were verified under. Memory and the other limits are the defaults.
    """
    if mode != "io":
        return Limits()
    cases = json.loads(tests)
    count = len(cases) if isinstance(cases, list) else 0
    return Limits(timeout_seconds=io_seconds(count))


def isolated_execution(code: str, tests: str, mode: str) -> Result:
    """Run a candidate with its tests in the sandbox under :func:`execution_limits`."""
    return run(code, tests, execution_limits(tests, mode), mode=mode)


def is_host_failure(result: Result) -> bool:
    """A result that says nothing about the candidate: the host stalled, Docker was
    unavailable or the runner failed. Such a result is retried and never scored."""
    return result.host_stalled or result.status not in SCORED_STATUSES


def host_backoff(failures: int) -> float:
    """The wait before retry number ``failures`` (1, 2, 3, ...): 2 s doubling, at most 30 s."""
    if type(failures) is not int or failures < 1:
        raise ValueError("failures must be a positive integer")
    return float(min(HOST_BACKOFF_MAX_SECONDS, HOST_BACKOFF_SECONDS * 2.0 ** (failures - 1)))


def attempt_seed(seed: int, task_id: str, index: int) -> int:
    """Sampling seed of attempt ``index`` on a task: one per row in batched generation."""
    return (seed + int(hashlib.sha256(task_id.encode()).hexdigest()[:16], 16) + index) % (2**63 - 1)


def evaluate(
    tasks: Sequence[Task],
    generate: Callable[[Task, int], str],
    *,
    config: EvaluationConfig,
    checkpoint_hash: str,
    tokenizer_hash: str,
    execute: Callable[[str, str, str], Result] = isolated_execution,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Generate and score every attempt, in task-ID order.

    A result with ``host_stalled`` set is retried up to ``HOST_RETRIES`` times with
    :func:`host_backoff` and never scored; if it is still stalled, or if isolation is
    unavailable or fails, evaluation aborts rather than report a score.
    """
    if not tasks or len({task.id for task in tasks}) != len(tasks):
        raise ValueError("evaluation needs nonempty unique tasks")
    if any(task.split == "train" for task in tasks):
        raise ValueError("reported evaluation cannot use training tasks")
    if any(task.split == "test" for task in tasks):
        raise ValueError(
            "the sealed pool test split is scored only by eval-execute under an unseal "
            "record, which logs the read; this path would score it unlogged"
        )
    if not checkpoint_hash or not tokenizer_hash:
        raise ValueError("checkpoint and tokenizer identities are required")
    commit, dirty = git_state()
    task_hash = hashlib.sha256(
        json.dumps(
            [asdict(task) for task in sorted(tasks, key=lambda task: task.id)], sort_keys=True
        ).encode()
    ).hexdigest()
    rows: list[dict[str, Any]] = []
    attempts: list[list[bool]] = []
    source_scores: dict[str, list[float]] = defaultdict(list)
    author_scores: dict[str, list[float]] = defaultdict(list)
    for task in sorted(tasks, key=lambda task: task.id):
        outcomes = []
        successes = []
        for index in range(config.samples):
            seed = attempt_seed(config.seed, task.id, index)
            try:
                code = generate(task, seed)
                result = execute(code, task.hidden_tests, task.mode)
                stalls = 0
                while result.host_stalled and stalls < HOST_RETRIES:
                    stalls += 1
                    sleep(host_backoff(stalls))
                    result = execute(code, task.hidden_tests, task.mode)
                code_hash = hashlib.sha256(code.encode()).hexdigest()
            except ContextLimitError:
                result = Result("context_limit")
                code_hash = None
            if result.status != "context_limit" and is_host_failure(result):
                raise RuntimeError("evaluation aborted: isolation is unavailable or failed")
            successes.append(result.passed)
            outcomes.append({"seed": seed, "status": result.status, "code_hash": code_hash})
        attempts.append(successes)
        score = sum(successes) / len(successes)
        source_scores[task.source].append(score)
        author_scores[task.written_by].append(score)
        rows.append(
            {
                "id": task.id,
                "source": task.source,
                "written_by": task.written_by,
                "mean_pass_rate": score,
                "attempts": outcomes,
            }
        )
    per_task = [sum(row) / len(row) for row in attempts]
    ks = sorted({1, config.samples})
    metrics: dict[str, Any] = {
        "mean_pass_rate": mean_pass_rate(attempts),
        "pass_at_k": {
            str(k): sum(pass_at_k(config.samples, sum(row), k) for row in attempts) / len(tasks)
            for k in ks
        },
        "confidence_interval": asdict(
            paired_bootstrap(
                [0.0] * len(tasks), per_task, samples=config.bootstrap_samples, seed=config.seed
            )
        ),
        "by_source": {name: sum(values) / len(values) for name, values in source_scores.items()},
        "by_authorship": {
            name: sum(values) / len(values) for name, values in author_scores.items()
        },
    }
    if config.temperature == 0 and config.samples == 1:
        metrics["greedy_pass_at_1"] = metrics["mean_pass_rate"]
    return {
        "schema_version": 1,
        "status": "completed",
        "created_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "checkpoint_hash": checkpoint_hash,
        "tokenizer_hash": tokenizer_hash,
        "task_hash": task_hash,
        "task_count": len(tasks),
        "configuration": asdict(config),
        "metrics": metrics,
        "per_task": rows,
    }


def save_report(report: dict[str, Any], directory: Path, *, scorecard: Path) -> None:
    """Publish a complete report before appending one compact local scorecard row."""
    if report["status"] != "completed" or directory.exists():
        raise ValueError("only new completed reports can be published")
    directory.mkdir(parents=True)
    atomic_json(directory / "report.json", report)
    row = {key: value for key, value in report.items() if key != "per_task"}
    row["report_path"] = str(directory / "report.json")
    scorecard.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(scorecard, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(descriptor, (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
