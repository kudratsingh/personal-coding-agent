"""Tests for the RL loop: schedule, seeds, rollout protocol, every stop, and its records.

Most tests replace rollout generation with scripted completions, so the stops and records
are exercised on exact inputs; the policy-gradient step still runs on a tiny real model.
"""

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from scratchcoder.common.config import ModelConfig
from scratchcoder.data.chat import CODE_STOP_IDS, PYTHON_END, PYTHON_START, task_prompt
from scratchcoder.data.sft import pass_key
from scratchcoder.engine.generation import Completion
from scratchcoder.eval.harness import attempt_seed
from scratchcoder.eval.tasks import Task
from scratchcoder.model.gpt import GPT
from scratchcoder.optim.adamw import clip_gradients
from scratchcoder.rl import loop, rollout
from scratchcoder.rl.fakes import FakeRewardSource, ScriptedDevEvaluator
from scratchcoder.rl.loop import RLConfig, RLResult, run_rl, task_schedule
from scratchcoder.rl.policy import policy_gradient_backward
from scratchcoder.rl.rewards import Attempt, AttemptReward, RewardHostFailure
from scratchcoder.rl.rollout import (
    ROLLOUT_SEED,
    SCREENING_SEED,
    Rollout,
    rollout_seed,
    screening_rollouts,
    screening_seed,
)
from scratchcoder.tokenizer.bpe import ByteBPE

TOKENIZER = ByteBPE()


def task(identifier: str) -> Task:
    return Task(
        id=identifier,
        prompt="Add one.",
        public_tests="assert f(1) == 2",
        hidden_tests="assert f(2) == 3\nassert f(5) == 6",
        source="synthetic",
        license="fixture",
        written_by="human",
        split="train",
        solution="def f(x):\n    return x + 1",
    )


TASKS = [task(f"t{index}") for index in range(6)]


def tiny_model() -> GPT:
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=64))


def config(**overrides: Any) -> RLConfig:
    options: dict[str, Any] = {
        "updates": 3,
        "tasks_per_update": 2,
        "group_size": 4,
        "max_new_tokens": 8,
        "micro_batch": 3,
        "generation_batch": 16,
        "warmup_updates": 2,
        "learning_rate": 1e-3,
        "dev_checks": (0, 2),
    }
    options.update(overrides)
    return RLConfig(**options)


def scripted(
    distinct: bool = True,
) -> Callable[..., list[Rollout]]:
    """A generate_rollouts stand-in: sample s writes a program of s + 1 letters."""

    def fake(
        model: GPT,
        tasks: Sequence[Task],
        prompts: Sequence[Sequence[int]],
        decoder: Any,
        *,
        samples: int,
        seed_of: Callable[[Task, int], int],
        **options: Any,
    ) -> list[Rollout]:
        rows = []
        for item, prompt in zip(tasks, prompts, strict=True):
            for sample in range(samples):
                width = sample + 1 if distinct else 1
                body = (PYTHON_START, *([ord("a")] * width), PYTHON_END)
                rows.append(
                    Rollout(
                        Attempt(item.id, sample, "a" * width, len(body)),
                        tuple(prompt),
                        body,
                        "stop",
                        seed_of(item, sample),
                    )
                )
        return rows

    return fake


def run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rewards: Any = None,
    evaluator: Any = None,
    rollouts: Callable[..., list[Rollout]] | None = None,
    should_stop: Any = None,
    **overrides: Any,
) -> RLResult:
    monkeypatch.setattr(loop, "generate_rollouts", rollouts or scripted())
    return run_rl(
        tiny_model(),
        TASKS,
        encoder=TOKENIZER,
        decoder=TOKENIZER,
        rewards=rewards or FakeRewardSource(),
        evaluator=evaluator or ScriptedDevEvaluator(),
        output=tmp_path / "rl",
        run_id="rl-test",
        tokenizer_hash="authored",
        config=config(**overrides),
        should_stop=should_stop,
    )


def events(tmp_path: Path, name: str) -> list[dict[str, Any]]:
    lines = (tmp_path / "rl" / "events.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    return [row["fields"] for row in rows if row["event"] == name]


# ------------------------------------------------------------------------- schedule


def test_schedule_takes_disjoint_blocks_and_reshuffles_each_pass() -> None:
    ids = [f"t{index}" for index in range(7)]
    blocks = task_schedule(ids, updates=7, per_update=3, seed=0)
    order = [sorted(ids, key=lambda item: pass_key(0, epoch, item)) for epoch in range(4)]
    # 7 tasks give two blocks of 3 per pass; the 7th task of each pass is not used.
    assert blocks[:2] == [(0, tuple(order[0][:3])), (0, tuple(order[0][3:6]))]
    assert blocks[2:4] == [(1, tuple(order[1][:3])), (1, tuple(order[1][3:6]))]
    assert [epoch for epoch, _ in blocks] == [0, 0, 1, 1, 2, 2, 3]
    assert all(len(set(block)) == 3 for _, block in blocks)
    assert task_schedule(list(reversed(ids)), updates=7, per_update=3, seed=0) == blocks
    assert task_schedule(ids, updates=7, per_update=3, seed=1) != blocks
    with pytest.raises(ValueError):
        task_schedule(ids[:2], updates=1, per_update=3, seed=0)
    with pytest.raises(ValueError):
        task_schedule(["a", "a", "b"], updates=1, per_update=1, seed=0)


def test_the_rehearsal_schedule_never_repeats_a_task_within_an_update_or_pass() -> None:
    ids = [f"task-{index}" for index in range(771)]
    blocks = task_schedule(ids, updates=100, per_update=32, seed=0)
    assert len(blocks) == 100 and all(len(set(block)) == 32 for _, block in blocks)
    for epoch in {epoch for epoch, _ in blocks}:
        seen = [item for number, block in blocks if number == epoch for item in block]
        assert len(seen) == len(set(seen))
    assert blocks[23][0] == 0 and blocks[24][0] == 1  # 771 // 32 = 24 updates per pass


# ---------------------------------------------------------------------------- seeds


def test_rollout_seeds_are_distinct_for_every_update_task_and_sample() -> None:
    ids = [f"task-{index}" for index in range(771)]
    blocks = task_schedule(ids, updates=100, per_update=32, seed=0)
    seeds = [
        rollout_seed(update, item, sample, group_size=8)
        for update, (_, block) in enumerate(blocks, 1)
        for item in block
        for sample in range(8)
    ]
    assert len(seeds) == 25_600 and len(set(seeds)) == 25_600
    screening = {screening_seed(item, sample) for item in ids for sample in range(8)}
    dev = {attempt_seed(0, item, sample) for item in ids for sample in range(8)}
    assert not screening & set(seeds) and not dev & set(seeds) and not dev & screening


def test_rollout_seed_formula() -> None:
    assert rollout_seed(1, "x", 0, group_size=8) == attempt_seed(ROLLOUT_SEED, "x", 0)
    assert rollout_seed(3, "x", 5, group_size=8) == attempt_seed(ROLLOUT_SEED, "x", 21)
    assert screening_seed("x", 7) == attempt_seed(SCREENING_SEED, "x", 7)
    for update, sample in ((0, 0), (1, 8), (1, -1)):
        with pytest.raises(ValueError):
            rollout_seed(update, "x", sample, group_size=8)


def test_a_repeated_seed_stops_the_run_with_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop, "_rollout_seeds", lambda update, group, base: lambda t, s: 7)
    with pytest.raises(RuntimeError, match="repeated"):
        run(tmp_path, monkeypatch)


# ------------------------------------------------------------------ rollout protocol


def recording_generate(calls: list[dict[str, Any]]) -> Callable[..., list[Completion]]:
    def fake(model: GPT, prompts: Sequence[Sequence[int]], **options: Any) -> list[Completion]:
        calls.append({"prompts": [list(p) for p in prompts], **options})
        return [Completion([PYTHON_START, ord("y"), PYTHON_END], "stop", 1) for _ in prompts]

    return fake


def test_rl_rollouts_follow_the_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(rollout, "generate_many", recording_generate(calls))
    result = run_rl(
        tiny_model(),
        TASKS,
        encoder=TOKENIZER,
        decoder=TOKENIZER,
        rewards=FakeRewardSource(),
        evaluator=ScriptedDevEvaluator(),
        output=tmp_path / "rl",
        run_id="rl-test",
        tokenizer_hash="authored",
        config=config(updates=2, max_new_tokens=256, dev_checks=()),
    )
    assert result.updates == 2 and len(calls) == 2
    first = calls[0]
    _, block = task_schedule([t.id for t in TASKS], updates=1, per_update=2, seed=0)[0]
    by_id = {item.id: item for item in TASKS}
    assert first["prompts"] == [task_prompt(TOKENIZER, by_id[i]) for i in block for _ in range(4)]
    assert first["temperature"] == 1.0 and first["top_p"] == 1.0
    assert first["max_new_tokens"] == 256 and first["precision"] == "float32"
    assert first["seeds"] == [
        rollout_seed(1, item, sample, group_size=4) for item in block for sample in range(4)
    ]
    assert all(set(stops) == set(CODE_STOP_IDS) for stops in first["stops"])
    assert "log_probs" not in first  # the loss never uses the sampler's log probabilities


def test_screening_rollouts_follow_the_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(rollout, "generate_many", recording_generate(calls))
    rows = screening_rollouts(tiny_model(), TASKS[:2], TOKENIZER, TOKENIZER, batch_size=5)
    (call,) = calls
    assert call["temperature"] == 1.0 and call["top_p"] == 1.0
    assert call["max_new_tokens"] == 256 and call["batch_size"] == 5
    assert call["seeds"] == [screening_seed(t.id, s) for t in TASKS[:2] for s in range(8)]
    assert [(row.attempt.task_id, row.attempt.sample) for row in rows] == [
        (t.id, s) for t in TASKS[:2] for s in range(8)
    ]
    assert all(row.attempt.code == "y" and row.attempt.completion_tokens == 3 for row in rows)


def test_a_prompt_that_fills_the_context_gets_no_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(rollout, "generate_many", recording_generate(calls))
    long = Task(**{**TASKS[0].__dict__, "id": "long", "prompt": "x" * 70})
    rows = screening_rollouts(tiny_model(), [long, TASKS[1]], TOKENIZER, TOKENIZER, batch_size=4)
    assert len(calls[0]["prompts"]) == 8  # only the short task was generated
    assert all(row.attempt.code is None and row.finish == "context" for row in rows[:8])
    assert all(row.attempt.completion_tokens == 0 for row in rows[:8])


# ------------------------------------------------------------------------ records


def test_a_completed_run_records_updates_checks_and_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = ScriptedDevEvaluator(script=lambda check: (0.1 + 0.01 * check, 0.0))
    result = run(tmp_path, monkeypatch, evaluator=evaluator)
    assert result.stop_reason == "completed" and result.updates == 3
    assert result.optimizer_steps == 3 and result.skipped_updates == []
    # Checks 0 and 2 as configured, then 3 at the end because the weights changed since 2.
    assert evaluator.calls == [0, 2, 3]
    assert [check.dev.check for check in result.checks] == [0, 2, 3]
    assert [check.optimizer_steps for check in result.checks] == [0, 2, 3]
    assert result.selected is not None and result.selected.dev.check == 3
    assert result.seeds_used == 3 * 2 * 4
    for check in result.checks:
        assert Path(check.checkpoint).exists()
    saved = json.loads((tmp_path / "rl" / "result.json").read_text())
    assert saved["stop_reason"] == "completed" and saved["selected"]["dev"]["check"] == 3
    assert saved["busy_seconds"] == pytest.approx(
        saved["seconds"]["rollout_generation"] + saved["seconds"]["training_step"]
    )
    rows = events(tmp_path, "rl_update")
    assert [row["update"] for row in rows] == [1, 2, 3]
    expected_keys = {
        "mean_reward",
        "baseline_mean",
        "skipped_share",
        "mean_attempt_length",
        "timeouts",
        "distinct_per_group",
        "audit_flagged",
        "loss",
        "gradient_norm",
        "learning_rate",
        "seconds",
    }
    assert expected_keys <= set(rows[0])
    assert set(rows[0]["seconds"]) == {"rollout_generation", "training_step", "reward_wait"}
    assert rows[0]["distinct_per_group"] == 4.0 and rows[0]["mean_attempt_length"] == 4.5
    # Fake rewards: program length mod 4 over 3, lengths 1..4 -> (1/3, 2/3, 1, 0).
    assert rows[0]["mean_reward"] == pytest.approx(0.5)
    assert rows[0]["baseline_mean"] == 0.0
    # The loss denominator counts every attempt token of the non-skipped groups.
    assert rows[0]["loss_tokens"] == 2 * (3 + 4 + 5 + 6)
    with pytest.raises(ValueError, match="already holds"):
        run(tmp_path, monkeypatch)


def test_learning_rate_warms_up_linearly_then_stays_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run(tmp_path, monkeypatch, updates=6, warmup_updates=3, dev_checks=())
    rates = [row["learning_rate"] for row in events(tmp_path, "rl_update")]
    assert rates == pytest.approx([1e-3 / 3, 2e-3 / 3, 1e-3, 1e-3, 1e-3, 1e-3])


def test_an_update_whose_groups_are_all_skipped_takes_no_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constant = FakeRewardSource(reward_of=lambda attempt: 0.5)
    model_before = tiny_model().state_dict()
    result = run(tmp_path, monkeypatch, rewards=constant, dev_checks=(0,))
    assert result.stop_reason == "completed"
    assert result.optimizer_steps == 0 and result.skipped_updates == [1, 2, 3]
    rows = events(tmp_path, "rl_update")
    assert all(row["skipped_share"] == 1.0 and row["loss"] is None for row in rows)
    assert all(row["optimizer_step"] is False for row in rows)
    # Nothing changed, so no further check follows check 0, and the weights are untouched.
    assert [check.dev.check for check in result.checks] == [0]
    saved = torch.load(result.checks[0].checkpoint, weights_only=True)["model"]
    assert all(torch.equal(saved[name], value) for name, value in model_before.items())


# --------------------------------------------------------------------------- stops


def test_a_nonfinite_reward_stops_before_any_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reward_of(attempt: Attempt) -> float:
        return math.nan if attempt.sample == 2 else 0.5

    source = FakeRewardSource(reward_of=reward_of)
    result = run(tmp_path, monkeypatch, rewards=source, dev_checks=(0,))
    assert result.stop_reason == "non_finite" and "reward" in result.stop_detail
    assert result.updates == 1 and result.optimizer_steps == 0
    assert events(tmp_path, "rl_update")[0]["mean_reward"] is None


@pytest.mark.parametrize("stage", ["loss", "gradient_norm"])
def test_a_nonfinite_loss_or_gradient_norm_stops_without_stepping(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}
    real = policy_gradient_backward if stage == "loss" else clip_gradients

    def broken(*args: Any, **kwargs: Any) -> float:
        calls["count"] += 1
        if calls["count"] == 2:
            raise FloatingPointError("nonfinite")
        return float(real(*args, **kwargs))

    monkeypatch.setattr(
        loop, "policy_gradient_backward" if stage == "loss" else "clip_gradients", broken
    )
    result = run(tmp_path, monkeypatch, dev_checks=(0,))
    assert result.stop_reason == "non_finite" and result.stop_detail == f"update 2: {stage}"
    assert result.updates == 2 and result.optimizer_steps == 1
    # The stop takes one more check: the weights changed since check 0.
    assert [check.dev.check for check in result.checks] == [0, 2]
    assert [check.optimizer_steps for check in result.checks] == [0, 1]


def test_collapse_stops_after_five_updates_below_two_distinct_completions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = run(tmp_path, monkeypatch, rollouts=scripted(distinct=False), updates=8, dev_checks=())
    assert result.stop_reason == "collapse" and result.updates == 5
    assert all(row["distinct_per_group"] == 1.0 for row in events(tmp_path, "rl_update"))


def test_dev_patience_stops_after_three_checks_not_above_the_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = {0: 0.2, 1: 0.25, 2: 0.25, 3: 0.1, 4: 0.24, 5: 0.9}
    evaluator = ScriptedDevEvaluator(script=lambda check: (scores[check], 0.0))
    result = run(
        tmp_path, monkeypatch, evaluator=evaluator, updates=6, dev_checks=(0, 1, 2, 3, 4, 5)
    )
    assert result.stop_reason == "dev_patience" and result.updates == 4
    assert evaluator.calls == [0, 1, 2, 3, 4]  # no extra check: the stop follows a check
    # Checks 1 and 2 tie on both keys, so the later one is selected.
    assert result.selected is not None and result.selected.dev.check == 2


def test_leakage_stops_once_enough_rewarded_attempts_are_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def flags_of(attempt: Attempt) -> tuple[str, ...]:
        return ("inspect",) if attempt.task_id == "t0" else ()

    source = FakeRewardSource(flags_of=flags_of)
    result = run(
        tmp_path,
        monkeypatch,
        rewards=source,
        updates=10,
        dev_checks=(),
        leakage_min_rewarded=12,
        leakage_rate=0.01,
    )
    assert result.stop_reason == "leakage"
    rows = events(tmp_path, "rl_update")
    assert rows[-1]["cumulative_rewarded"] >= 12
    assert rows[-1]["cumulative_rewarded_flagged_rate"] > 0.01
    assert all(row["cumulative_rewarded"] < 12 for row in rows[:-1])
    assert all("rewarded_flagged_rate" in row for row in rows)


def test_leakage_needs_the_minimum_count_of_rewarded_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = FakeRewardSource(flags_of=lambda attempt: ("inspect",))
    result = run(tmp_path, monkeypatch, rewards=source, dev_checks=(), leakage_min_rewarded=100)
    assert result.stop_reason == "completed"  # 18 rewarded attempts, all flagged, under 100


def test_a_reward_timeout_saves_weights_and_returns_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = FakeRewardSource(timeout_at=3)
    evaluator = ScriptedDevEvaluator()
    result = run(tmp_path, monkeypatch, rewards=source, evaluator=evaluator, updates=4)
    assert result.stop_reason == "reward_wait" and result.updates == 2
    assert result.optimizer_steps == 2
    assert evaluator.calls == [0, 2]  # no dev check after a reward-channel stop
    assert result.stop_checkpoint is None  # weights unchanged since check 2
    failed = events(tmp_path, "rl_reward_failed")
    assert failed[0]["update"] == 3 and failed[0]["reason"] == "reward_wait"


def test_a_reward_timeout_with_unchecked_weights_saves_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = run(tmp_path, monkeypatch, rewards=FakeRewardSource(timeout_at=2), dev_checks=(0,))
    assert result.stop_reason == "reward_wait" and result.optimizer_steps == 1
    assert result.stop_checkpoint is not None
    saved = torch.load(result.stop_checkpoint["checkpoint"], weights_only=True)
    assert saved["metadata"] == {"kind": "rl", "update": 1, "optimizer_steps": 1, "checked": False}


def test_an_unresolved_host_failure_stops_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Failing:
        def rewards(self, update: int, attempts: Sequence[Attempt]) -> list[AttemptReward]:
            raise RewardHostFailure("docker unavailable")

    result = run(tmp_path, monkeypatch, rewards=Failing())
    assert result.stop_reason == "reward_host_failure" and result.updates == 0


def test_a_dev_check_timeout_is_a_reward_wait_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = ScriptedDevEvaluator(timeout_at=2)
    result = run(tmp_path, monkeypatch, evaluator=evaluator)
    assert result.stop_reason == "reward_wait" and "dev check 2" in result.stop_detail
    assert result.stop_checkpoint is not None and result.updates == 2


def test_should_stop_is_checked_between_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = iter([None, False, "cost"])
    evaluator = ScriptedDevEvaluator()
    result = run(
        tmp_path,
        monkeypatch,
        evaluator=evaluator,
        should_stop=lambda: next(answers),
        updates=10,
        dev_checks=(0,),
    )
    assert result.stop_reason == "external" and result.stop_detail == "cost"
    assert result.updates == 2 and evaluator.calls == [0, 2]  # a check at the stop
    no_check = run(
        tmp_path / "again",
        monkeypatch,
        should_stop=lambda: True,
        check_at_stop=False,
    )
    assert no_check.stop_detail == "requested" and no_check.updates == 0


def test_rl_config_validation() -> None:
    for overrides in (
        {"top_p": 0.95},
        {"group_size": 1},
        {"temperature": 0.0},
        {"dev_checks": (0, 5)},
        {"dev_checks": (2, 1)},
        {"warmup_updates": 4},
        {"leakage_rate": 1.5},
        {"check_at_stop": 1},
    ):
        with pytest.raises(ValueError):
            config(**overrides)
    defaults = RLConfig()
    assert (defaults.updates, defaults.tasks_per_update, defaults.group_size) == (100, 32, 8)
    assert defaults.learning_rate == 3e-6 and defaults.warmup_updates == 5
    assert defaults.dev_checks == (0, 12, 25, 50, 75, 100)
    assert (defaults.beta1, defaults.beta2, defaults.weight_decay) == (0.9, 0.95, 0.1)
    assert defaults.gradient_clip == 1.0 and defaults.max_new_tokens == 256


def test_an_unexpected_error_saves_the_latest_weights_then_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken:
        def rewards(self, update: int, attempts: Sequence[Attempt]) -> list[AttemptReward]:
            if update == 2:
                raise KeyError("bridge reply lost a field")
            return FakeRewardSource().rewards(update, attempts)

    with pytest.raises(KeyError):
        run(tmp_path, monkeypatch, rewards=Broken(), dev_checks=(0,))
    saved = torch.load(tmp_path / "rl" / "checkpoints" / "update-0001.pt", weights_only=True)
    assert saved["metadata"]["checked"] is False and saved["metadata"]["optimizer_steps"] == 1
    (failure,) = events(tmp_path, "rl_failed")
    assert failure["error_type"] == "KeyError" and failure["updates"] == 1
    assert not (tmp_path / "rl" / "result.json").exists()
