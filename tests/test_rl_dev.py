"""Tests for the RL dev subset, dev checks, the constant baseline and checkpoint selection."""

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import pytest
import torch

from scratchcoder.common.config import ModelConfig
from scratchcoder.data.chat import CODE_STOP_IDS, PYTHON_END, PYTHON_START, task_prompt
from scratchcoder.engine.generation import Completion, generate_many
from scratchcoder.eval.harness import attempt_seed
from scratchcoder.eval.tasks import Task
from scratchcoder.model.gpt import GPT
from scratchcoder.rl import dev
from scratchcoder.rl.dev import (
    DevCheck,
    DevGeneration,
    DevTaskResult,
    PoolDevEvaluator,
    constant_baseline,
    dev_check_from_json,
    dev_generations,
    dev_metrics,
    dev_subset,
    score_generations,
    select_check,
)
from scratchcoder.tokenizer.bpe import ByteBPE

TOKENIZER = ByteBPE()


def dev_task(identifier: str, hidden: str = "assert f(6) == 17", split: str = "dev") -> Task:
    return Task(
        id=identifier,
        prompt="Compute f.",
        public_tests="assert f(3) == 11",
        hidden_tests=hidden,
        source="synthetic",
        license="fixture",
        written_by="human",
        split=split,
    )


def check(index: int, greedy: float, mean: float) -> DevCheck:
    return DevCheck(index, greedy, mean, min(1.0, 8 * mean), 0.1, 200, 1.0, 2.0)


def test_dev_subset_is_the_smallest_id_hashes_in_order() -> None:
    tasks = [dev_task(f"d{index}") for index in range(30)]
    subset = dev_subset(tasks, n=10)
    key = [hashlib.sha256(f"rl-dev-subset-v1|{task.id}".encode()).digest() for task in tasks]
    expected = [task for _, task in sorted(zip(key, tasks, strict=True), key=lambda pair: pair[0])]
    assert subset == expected[:10]
    assert dev_subset(list(reversed(tasks)), n=10) == subset  # input order does not matter
    with pytest.raises(ValueError):
        dev_subset(tasks, n=31)
    with pytest.raises(ValueError):
        dev_subset([*tasks, dev_task("train", split="train")], n=5)
    with pytest.raises(ValueError):
        dev_subset([tasks[0], tasks[0]], n=1)


def test_constant_baseline_counts_unparseable_tasks_as_zero() -> None:
    tasks = [
        dev_task("a", "assert f(1) == 0\nassert f(2) == 0"),  # constant 0 earns 1.0
        dev_task("b", "assert f(6) == 17\nassert f(2) == 0"),  # 0.5
        dev_task("c", "y = f(1)\nassert y == 2"),  # unparseable: 0
        dev_task("d", "assert f(6) == 17"),  # 0
    ]
    baseline = constant_baseline(tasks)
    assert baseline.mean == pytest.approx(1.5 / 4)
    assert baseline.unparseable == 1 and baseline.tasks == 4
    # The pass rate is in pass@1's units: only task a is fully passed by a constant answer,
    # far below the mean reward of 0.375 that the half-credit task b also feeds.
    assert baseline.solved == 1 and baseline.pass_rate == pytest.approx(0.25)


def test_dev_metrics_are_over_all_tasks() -> None:
    results = [
        DevTaskResult("a", True, 8),
        DevTaskResult("b", False, 1),
        DevTaskResult("c", False, 0),
        DevTaskResult("d", True, 3),
    ]
    metrics = dev_metrics(results, 8)
    assert metrics["greedy_pass_at_1"] == 0.5
    assert metrics["mean_pass_n8"] == 12 / 32
    assert metrics["pass_at_8"] == 0.75


def test_score_generations_runs_only_programs_in_one_batch() -> None:
    tasks = [dev_task("a"), dev_task("b")]
    generations = [
        DevGeneration("a", "good", ("good", None, "bad"), False),
        DevGeneration("b", None, (None, None, None), True),
    ]
    batches: list[list[tuple[str, str]]] = []

    def execute(pairs: Sequence[tuple[Task, str]]) -> list[bool]:
        batches.append([(task.id, code) for task, code in pairs])
        return [code == "good" for _, code in pairs]

    results, missing = score_generations(tasks, generations, execute)
    assert batches == [[("a", "good"), ("a", "good"), ("a", "bad")]]
    assert results == [DevTaskResult("a", True, 1), DevTaskResult("b", False, 0)]
    assert missing == 5
    with pytest.raises(ValueError):
        score_generations(tasks, generations, lambda pairs: [True])
    with pytest.raises(ValueError):
        score_generations(tasks[:1], generations, execute)


def test_dev_generations_follow_the_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """Greedy, then 8 samples at T = 0.8 and top-p 0.95 with seeds attempt_seed(0, id, i),
    at most 512 new tokens, stopping at the code stops; a prompt filling the context is skipped."""
    calls: list[dict[str, Any]] = []

    def fake(model: GPT, prompts: Sequence[Sequence[int]], **options: Any) -> list[Completion]:
        calls.append({"prompts": [list(p) for p in prompts], **options})
        body = [PYTHON_START, ord("x"), PYTHON_END]
        return [Completion(body, "stop", 512) for _ in prompts]

    monkeypatch.setattr(dev, "generate_many", fake)
    model = GPT(ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=64))
    short, long = dev_task("short"), dev_task("long")
    long = Task(**{**long.__dict__, "prompt": "x" * 80})
    rows = dev_generations(model, [short, long], TOKENIZER, TOKENIZER, batch_size=7)
    greedy, sampled = calls
    assert greedy["prompts"] == [task_prompt(TOKENIZER, short)]
    assert greedy.get("temperature", 0.0) == 0.0 and greedy["max_new_tokens"] == 512
    assert sampled["temperature"] == 0.8 and sampled["top_p"] == 0.95
    assert sampled["seeds"] == [attempt_seed(0, "short", index) for index in range(8)]
    assert all(set(stops) == set(CODE_STOP_IDS) for stops in sampled["stops"])
    assert greedy["batch_size"] == sampled["batch_size"] == 7
    assert rows[0] == DevGeneration("short", "x", ("x",) * 8, False)
    assert rows[1] == DevGeneration("long", None, (None,) * 8, True)


def test_pool_dev_evaluator_runs_end_to_end_on_a_tiny_model() -> None:
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=64))
    tasks = [dev_task("a"), dev_task("b", "assert f(1) == 0"), dev_task("c", "q = 1")]
    executed: list[int] = []

    def execute(pairs: Sequence[tuple[Task, str]]) -> list[bool]:
        executed.append(len(pairs))
        return [True] * len(pairs)

    evaluator = PoolDevEvaluator(tasks, TOKENIZER, TOKENIZER, execute, batch_size=4)
    result = evaluator.evaluate(model, 12)
    assert result.check == 12 and result.tasks == 3 and len(result.per_task) == 3
    assert result.constant_baseline == pytest.approx(1 / 3) and result.baseline_unparseable == 1
    assert result.constant_pass_rate == pytest.approx(1 / 3)  # task b only; c is unparseable
    assert result.generation_seconds >= 0 and result.wait_seconds >= 0
    programs = 3 * 9 - result.no_program
    assert sum(executed) == programs
    assert result.mean_pass_n8 * 8 * 3 + result.greedy_pass_at_1 * 3 == pytest.approx(programs)
    again = evaluator.evaluate(model, 13)
    assert again.per_task == result.per_task  # seeded: the same policy gives the same check


def test_a_ticking_dev_check_generates_exactly_what_an_unticked_one_does() -> None:
    """``tick`` after every batch keeps a worker's heartbeat live through a long check; the
    batches are ``generate_many``'s, so every row gets the same completion."""
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=128))
    tasks = [
        Task(**{**dev_task(f"t{index}").__dict__, "prompt": "Compute f." + " now" * index})
        for index in range(5)
    ]
    plain = dev_generations(model, tasks, TOKENIZER, TOKENIZER, batch_size=3)
    ticks: list[int] = []
    ticked = dev_generations(model, tasks, TOKENIZER, TOKENIZER, batch_size=3, tick=ticks.append)
    assert ticked == plain
    assert len(ticks) == math.ceil(5 / 3) + math.ceil(5 * 8 / 3) and all(n >= 0 for n in ticks)
    prompts = [task_prompt(TOKENIZER, task) for task in tasks]
    options: dict[str, Any] = {"max_new_tokens": 6, "temperature": 0.8, "top_p": 0.95}
    seeds = list(range(10, 15))
    expected = generate_many(model, prompts, batch_size=2, seeds=seeds, **options)
    calls: list[int] = []
    got = dev.ticking_generations(
        model, prompts, tick=calls.append, batch_size=2, seeds=seeds, **options
    )
    assert [row.tokens for row in got] == [row.tokens for row in expected]
    assert len(calls) == 3 and sum(calls) == sum(len(row.tokens) for row in got)
    with pytest.raises(ValueError, match="nonempty"):
        dev.ticking_generations(model, [], tick=calls.append, batch_size=2, **options)
    with pytest.raises(ValueError, match="one entry per prompt"):
        dev.ticking_generations(
            model, prompts, tick=calls.append, batch_size=2, seeds=[1], **options
        )
    executed: list[int] = []

    def execute(pairs: Sequence[tuple[Task, str]]) -> list[bool]:
        executed.append(len(pairs))
        return [len(code) % 2 == 0 for _, code in pairs]

    beats: list[int] = []
    evaluator = PoolDevEvaluator(tasks, TOKENIZER, TOKENIZER, execute, batch_size=4)
    ticking = PoolDevEvaluator(
        tasks, TOKENIZER, TOKENIZER, execute, batch_size=4, tick=beats.append
    )
    assert ticking.evaluate(model, 1).per_task == evaluator.evaluate(model, 1).per_task
    assert len(beats) == math.ceil(5 / 4) + math.ceil(40 / 4)


def test_selection_prefers_greedy_then_the_mean_then_the_later_check() -> None:
    checks = [check(0, 0.10, 0.20), check(12, 0.15, 0.10), check(25, 0.15, 0.12)]
    assert select_check(checks).check == 25  # tie on greedy: higher mean
    checks = [check(0, 0.10, 0.20), check(12, 0.15, 0.12), check(25, 0.15, 0.12)]
    assert select_check(checks).check == 25  # full tie: later check
    assert select_check(list(reversed(checks))).check == 25
    checks = [check(0, 0.30, 0.0), check(12, 0.15, 0.9)]
    assert select_check(checks).check == 0  # greedy pass@1 first
    with pytest.raises(ValueError):
        select_check([])
    with pytest.raises(ValueError):
        select_check([check(1, 0.1, 0.1), check(1, 0.2, 0.1)])


def test_equal_pass_counts_give_equal_means_for_tie_breaking() -> None:
    """The mean is total passes over n x tasks, so the task order cannot change it."""
    counts = [(7 * index) % 9 for index in range(200)]
    counts = [min(count, 8) for count in counts]
    first = dev_metrics([DevTaskResult(str(i), False, c) for i, c in enumerate(counts)], 8)
    shuffled = counts[1::2] + counts[::2]
    second = dev_metrics([DevTaskResult(str(i), False, c) for i, c in enumerate(shuffled)], 8)
    assert first["mean_pass_n8"] == second["mean_pass_n8"] == sum(counts) / 1600


def test_dev_check_validation_and_json_round_trip() -> None:
    value = DevCheck(3, 0.5, 0.25, 0.75, 0.1, 4, 1.0, 2.0, per_task=(DevTaskResult("a", True, 2),))
    assert dev_check_from_json(value.to_json()) == value
    assert value.to_json(per_task=False)["per_task"] == []
    assert value.constant_pass_rate is None  # not recorded unless the evaluator supplies it
    rated = DevCheck(3, 0.5, 0.25, 0.75, 0.1, 4, 1.0, 2.0, constant_pass_rate=0.05)
    assert dev_check_from_json(rated.to_json()) == rated and rated.to_json()["constant_pass_rate"]
    with pytest.raises(ValueError):
        DevCheck(3, 0.5, 0.25, 0.75, 0.1, 4, 1.0, 2.0, constant_pass_rate=1.5)
    for fields in (
        (-1, 0.5, 0.5, 0.5, 0.1, 4, 0.0, 0.0),
        (1, 1.5, 0.5, 0.5, 0.1, 4, 0.0, 0.0),
        (1, 0.5, 0.5, 0.5, 0.1, 0, 0.0, 0.0),
        (1, 0.5, 0.5, 0.5, 0.1, 4, math.nan, 0.0),
    ):
        with pytest.raises(ValueError):
            DevCheck(*fields)
