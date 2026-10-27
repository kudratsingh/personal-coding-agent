"""The facts about hidden tests that the Pod needs, computed on the Mac.

The RL loop, the dev evaluator and the control read each task's hidden tests for two
things: whether the reward channel can parse them (an unparseable task scores 0 for any
program, so RL refuses it) and the best constant answer's reward (the monitoring baseline
and the dev check's constant baseline). A Pod that holds no hidden tests cannot compute
either. The Mac computes both from the real tests and ships them as a facts file; the Pod
checks the file's SHA-256 against a value the lead configures, and hands the facts to
``run_rl``, ``run_more_sft`` and ``PoolDevEvaluator`` in place of the tasks' tests.

A fact is one task's ``Baseline(parseable, reward, None)``. The constant answer itself is
left out: it can be a hidden test's expected value, and the Pod never needs it. See
``docs/rl.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from scratchcoder.eval.tasks import Task
from scratchcoder.rl.bridge import canonical_json, require_sha256
from scratchcoder.rl.dev import ConstantBaseline
from scratchcoder.rl.screening import Baseline, baseline_of
from scratchcoder.rl.withheld import is_withheld

FACTS_FORMAT = "rl-task-facts-v1"
MAX_FACTS_BYTES = 8 * 1024 * 1024
FACT_KEYS = frozenset({"parseable", "reward"})


def task_facts(tasks: Sequence[Task]) -> dict[str, Baseline]:
    """Each task's facts, from its real hidden tests (run on the Mac)."""
    facts: dict[str, Baseline] = {}
    for task in tasks:
        if is_withheld(task):
            raise ValueError(f"task {task.id} withholds its hidden tests: facts need the real ones")
        if task.id in facts:
            raise ValueError(f"duplicate task ID: {task.id}")
        found = baseline_of(task)
        facts[task.id] = Baseline(found.parseable, found.reward, None)
    return facts


def facts_bytes(facts: Mapping[str, Baseline]) -> bytes:
    """The facts file: canonical JSON, so its SHA-256 is a function of the facts alone."""
    rows: dict[str, Any] = {}
    for identifier, fact in sorted(facts.items()):
        _check_fact(identifier, fact)
        rows[identifier] = {"parseable": fact.parseable, "reward": fact.reward}
    return canonical_json({"format": FACTS_FORMAT, "tasks": rows})


def facts_sha256(facts: Mapping[str, Baseline]) -> str:
    return hashlib.sha256(facts_bytes(facts)).hexdigest()


def _check_fact(identifier: object, fact: object) -> None:
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("a fact needs a nonempty task ID")
    if not isinstance(fact, Baseline) or fact.answer is not None:
        raise ValueError(f"the fact for {identifier} must be a Baseline without its answer")
    if type(fact.parseable) is not bool:
        raise ValueError(f"the fact for {identifier} needs a boolean parseable flag")
    reward = fact.reward
    if fact.parseable:
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
            or not 0.0 <= reward <= 1.0
        ):
            raise ValueError(f"the reward for parseable {identifier} must lie in [0, 1]")
    elif reward is not None:
        raise ValueError(f"unparseable {identifier} has no reward")


def load_facts(raw: bytes, *, sha256: str) -> dict[str, Baseline]:
    """The facts in ``raw``, after checking its SHA-256 against the configured ``sha256``
    and the exact schema; raises ``ValueError`` otherwise."""
    expected = require_sha256(sha256, "the facts file SHA-256")
    if len(raw) > MAX_FACTS_BYTES:
        raise ValueError(f"the facts file is over {MAX_FACTS_BYTES} bytes")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("the facts file's SHA-256 is not the configured one")
    try:
        value = json.loads(raw)
    except ValueError as error:
        raise ValueError("the facts file is not JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "tasks"}
        or value["format"] != FACTS_FORMAT
        or not isinstance(value["tasks"], dict)
        or not value["tasks"]
    ):
        raise ValueError(f"the facts file is not a nonempty {FACTS_FORMAT} record")
    facts: dict[str, Baseline] = {}
    for identifier, row in value["tasks"].items():
        if not isinstance(row, dict) or set(row) != FACT_KEYS:
            raise ValueError(f"the fact for {identifier} has exactly the keys {sorted(FACT_KEYS)}")
        fact = Baseline(row["parseable"], row["reward"], None)
        _check_fact(identifier, fact)
        facts[identifier] = fact
    if facts_bytes(facts) != raw:
        raise ValueError("the facts file is not in canonical form")
    return facts


def constant_baseline_from_facts(
    facts: Mapping[str, Baseline], task_ids: Sequence[str]
) -> ConstantBaseline:
    """``constant_baseline(tasks)`` for tasks with these IDs, from their facts: the same sums
    in the same order, so the result equals the one computed from the real tests."""
    if not task_ids:
        raise ValueError("a constant baseline needs at least one task")
    missing = [identifier for identifier in task_ids if identifier not in facts]
    if missing:
        raise ValueError(f"no facts for {len(missing)} tasks, e.g. {missing[0]}")
    values: list[float] = []
    unparseable = solved = 0
    for identifier in task_ids:
        reward = facts[identifier].reward
        if reward is None:
            unparseable += 1
            values.append(0.0)
        else:
            values.append(reward)
            solved += reward >= 1.0
    count = len(task_ids)
    return ConstantBaseline(math.fsum(values) / count, unparseable, count, solved / count, solved)
