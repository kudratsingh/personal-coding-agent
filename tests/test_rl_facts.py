"""Tests for tasks the Pod may hold (no hidden tests) and the Mac's facts about them."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import pytest

from scratchcoder.eval.tasks import Task
from scratchcoder.rl.dev import ConstantBaseline, PoolDevEvaluator, constant_baseline
from scratchcoder.rl.facts import (
    FACTS_FORMAT,
    constant_baseline_from_facts,
    facts_bytes,
    facts_sha256,
    load_facts,
    task_facts,
)
from scratchcoder.rl.screening import Baseline, baseline_of, require_train_tasks
from scratchcoder.rl.withheld import (
    WITHHELD_FUNCTION_TESTS,
    WITHHELD_IO_TESTS,
    is_withheld,
    refuse_withheld,
    withheld_task,
)
from scratchcoder.tokenizer.bpe import ByteBPE

SECRET = "assert f(6) == 987654321"


def task(identifier: str, hidden: str, split: str = "train", **fields: Any) -> Task:
    values: dict[str, Any] = {
        "id": identifier,
        "prompt": "Compute f.",
        "public_tests": "assert f(3) == 11",
        "hidden_tests": hidden,
        "source": "synthetic",
        "license": "fixture",
        "written_by": "human",
        "split": split,
        "solution": "def f(x):\n    return 0",
    }
    return Task(**{**values, **fields})


def train_tasks() -> list[Task]:
    return [
        task("a", "assert f(1) == 0\nassert f(2) == 0"),  # a constant answer earns 1.0
        task("b", "assert f(6) == 17\nassert f(2) == 0"),  # 0.5
        task(
            "c",
            json.dumps([{"input": "1\n", "output": "7\n"}, {"input": "2\n", "output": "7\n"}]),
            mode="io",
            public_tests=json.dumps([{"input": "0\n", "output": "7\n"}]),
        ),
        task("d", SECRET),  # 0
    ]


def dev_tasks() -> list[Task]:
    return [
        task("e", "assert f(1) == 0\nassert f(2) == 0", "dev", solution=None),
        task("f", "y = f(1)\nassert y == 2", "dev", solution=None),  # unparseable
        task("g", SECRET, "dev", solution=None),
    ]


def test_a_withheld_task_is_a_valid_task_without_its_hidden_tests() -> None:
    for original in (*train_tasks(), *dev_tasks()):
        held = withheld_task(original)
        assert is_withheld(held) and not is_withheld(original)
        assert held.hidden_tests in (WITHHELD_FUNCTION_TESTS, WITHHELD_IO_TESTS)
        assert held.mode == original.mode and held.id == original.id
        assert held.prompt == original.prompt and held.public_tests == original.public_tests
        assert original.hidden_tests not in json.dumps(asdict(held))
        # A reference solution stays for a train task (the control's targets), never for dev.
        assert held.solution == (original.solution if original.split == "train" else None)
    assert withheld_task(train_tasks()[2]).hidden_tests == WITHHELD_IO_TESTS
    refuse_withheld(tuple(train_tasks()), "x")
    with pytest.raises(ValueError, match="withhold them"):
        refuse_withheld(tuple(withheld_task(item) for item in train_tasks()), "x")


def test_facts_come_from_the_real_tests_and_leave_the_answer_out() -> None:
    facts = task_facts(train_tasks())
    assert facts["a"] == Baseline(True, 1.0, None)
    assert facts["b"].reward == 0.5 and facts["d"].reward == 0.0
    assert all(fact.answer is None for fact in facts.values())
    assert task_facts(dev_tasks())["f"] == Baseline(False, None, None)
    raw = facts_bytes(facts)
    assert SECRET.encode() not in raw  # nothing of a hidden test, not even the answer
    assert json.loads(raw)["format"] == FACTS_FORMAT
    assert facts_bytes(dict(reversed(list(facts.items())))) == raw  # order never matters
    assert load_facts(raw, sha256=facts_sha256(facts)) == facts
    with pytest.raises(ValueError, match="withholds"):
        task_facts([withheld_task(train_tasks()[0])])
    with pytest.raises(ValueError, match="duplicate"):
        task_facts([train_tasks()[0], train_tasks()[0]])


def test_load_facts_refuses_anything_but_the_configured_canonical_file() -> None:
    facts = task_facts(train_tasks())
    raw = facts_bytes(facts)
    digest = facts_sha256(facts)

    def reseal(value: Any) -> tuple[bytes, str]:
        data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return data, hashlib.sha256(data).hexdigest()

    with pytest.raises(ValueError, match="not the configured one"):
        load_facts(raw, sha256="0" * 64)
    with pytest.raises(ValueError, match="not the configured one"):
        load_facts(raw + b" ", sha256=digest)
    with pytest.raises(ValueError, match="64 lowercase hex"):
        load_facts(raw, sha256="nope")
    value = json.loads(raw)
    changes: list[tuple[Callable[[dict[str, Any]], object], str]] = [
        (lambda v: v.update(format="other"), "nonempty"),
        (lambda v: v.update(extra=1), "nonempty"),
        (lambda v: v.update(tasks={}), "nonempty"),
        (lambda v: v["tasks"]["a"].update(reward=2.0), "must lie in"),
        (lambda v: v["tasks"]["a"].update(reward=None), "must lie in"),
        (lambda v: v["tasks"]["a"].update(parseable=False), "no reward"),
        (lambda v: v["tasks"]["a"].update(parseable=1), "boolean"),
        (lambda v: v["tasks"]["a"].update(answer="x"), "exactly the keys"),
        (lambda v: v["tasks"]["a"].pop("reward"), "exactly the keys"),
        (lambda v: v["tasks"].update({"": {"parseable": False, "reward": None}}), "task ID"),
    ]
    for change, message in changes:
        changed = json.loads(raw)
        change(changed)
        data, sealed = reseal(changed)
        with pytest.raises(ValueError, match=message):
            load_facts(data, sha256=sealed)
    spaced = json.dumps(value, sort_keys=True, indent=1).encode()  # valid, but not canonical
    with pytest.raises(ValueError, match="canonical"):
        load_facts(spaced, sha256=hashlib.sha256(spaced).hexdigest())
    with pytest.raises(ValueError, match="not JSON"):
        load_facts(b"{", sha256=hashlib.sha256(b"{").hexdigest())


def test_the_baseline_from_facts_equals_the_one_from_the_real_tests() -> None:
    for tasks in (train_tasks(), dev_tasks(), dev_tasks()[:1]):
        facts = task_facts(tasks)
        ids = [item.id for item in tasks]
        assert constant_baseline_from_facts(facts, ids) == constant_baseline(tasks)
    baseline = constant_baseline_from_facts(task_facts(dev_tasks()), ["e", "f", "g"])
    assert baseline == ConstantBaseline(1 / 3, 1, 3, 1 / 3, 1)
    with pytest.raises(ValueError, match="no facts"):
        constant_baseline_from_facts(task_facts(dev_tasks()), ["e", "missing"])
    with pytest.raises(ValueError, match="at least one"):
        constant_baseline_from_facts({}, [])
    assert not math.isnan(baseline.mean)


def test_require_train_tasks_uses_facts_and_refuses_withheld_tasks_without_them() -> None:
    real = train_tasks()
    held = [withheld_task(item) for item in real]
    facts = task_facts(real)
    assert require_train_tasks(held, references=True, baselines=facts) == facts
    expected = {item.id: baseline_of(item) for item in real}
    assert {i: (b.parseable, b.reward) for i, b in facts.items()} == {
        i: (b.parseable, b.reward) for i, b in expected.items()
    }
    with pytest.raises(ValueError, match="withhold them"):
        require_train_tasks(held)
    with pytest.raises(ValueError, match="cover exactly"):
        require_train_tasks(held, baselines={k: v for k, v in facts.items() if k != "a"})
    with pytest.raises(ValueError, match="cover exactly"):
        require_train_tasks(held[:3], baselines=facts)
    unparseable = {**facts, "a": Baseline(False, None, None)}
    with pytest.raises(ValueError, match="1 unparseable by the reward channel, e.g. a"):
        require_train_tasks(held, baselines=unparseable)
    wrong_split = [*held[:3], withheld_task(task("z", SECRET, "dev"))]
    with pytest.raises(ValueError, match="1 not train-split"):
        require_train_tasks(
            wrong_split, baselines={**{k: facts[k] for k in "abc"}, "z": facts["d"]}
        )
    # Without facts, real tasks work exactly as before.
    assert {i: b.reward for i, b in require_train_tasks(real).items()} == {
        i: b.reward for i, b in facts.items()
    }


def test_the_dev_evaluator_takes_the_precomputed_baseline_or_refuses_withheld_tasks() -> None:
    tokenizer = ByteBPE()
    real = dev_tasks()
    held = [withheld_task(item) for item in real]
    baseline = constant_baseline_from_facts(task_facts(real), [item.id for item in real])

    def never(pairs: Any) -> list[bool]:
        raise AssertionError("no execution in this test")

    local = PoolDevEvaluator(real, tokenizer, tokenizer, never)
    precomputed = PoolDevEvaluator(held, tokenizer, tokenizer, never, baseline=baseline)
    assert precomputed.baseline == local.baseline
    # The wrong-baseline failure the review found: placeholders give mean 0 and every task
    # unparseable. It is now an error, not a record.
    with pytest.raises(ValueError, match="withhold them"):
        PoolDevEvaluator(held, tokenizer, tokenizer, never)
    with pytest.raises(ValueError, match="covers 3 tasks, not 2"):
        PoolDevEvaluator(held[:2], tokenizer, tokenizer, never, baseline=baseline)
