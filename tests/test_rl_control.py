"""Tests for the more-SFT control and a CPU end-to-end run of RL followed by the control.

The budget tests replace the trainer's synchronized clock with a scripted one, so each
update's ``step_seconds`` is exact and the stop rule can be checked by hand.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from scratchcoder.common.config import ModelConfig
from scratchcoder.data.chat import render_code_task
from scratchcoder.data.onepass import IGNORE
from scratchcoder.eval.tasks import Task
from scratchcoder.model.gpt import GPT
from scratchcoder.optim.adamw import AdamW, cross_entropy_sum
from scratchcoder.rl.checkpoint import load_weights, save_weights
from scratchcoder.rl.control import (
    ControlConfig,
    ControlResult,
    control_rows,
    pass_boundaries,
    run_more_sft,
)
from scratchcoder.rl.dev import DevCheck
from scratchcoder.rl.fakes import FakeRewardSource, ScriptedDevEvaluator
from scratchcoder.rl.loop import CheckRecord, RLConfig, run_rl
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.train import trainer as trainer_module

TOKENIZER = ByteBPE()
MODEL = ModelConfig(vocab_size=272, width=32, layers=2, heads=2, context=96)


def task(index: int) -> Task:
    return Task(
        id=f"t{index}",
        prompt=f"Add {index}.",
        public_tests=f"assert f(1) == {1 + index}",
        hidden_tests=f"assert f(2) == {2 + index}\nassert f(5) == {5 + index}",
        source="synthetic",
        license="fixture",
        written_by="human",
        split="train",
        solution=f"def f(x):\n    return x + {index}",
    )


TASKS = [task(index) for index in range(6)]


def start_check(path: Path) -> CheckRecord:
    dev = DevCheck(0, 0.1, 0.05, 0.2, 0.1, 200, 0.0, 0.0)
    return CheckRecord(dev, 0, str(path), "0" * 64)


@pytest.fixture
def parent(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    path = tmp_path / "parent.pt"
    save_weights(path, GPT(MODEL), tokenizer_hash=TOKENIZER.sha256, run_id="sft", step=0)
    return path


def scripted_clock(steps: list[float]) -> Iterator[float]:
    """Successive readings: each update reads the clock at its start and end."""
    now = 0.0
    for seconds in steps:
        yield now
        now += seconds
        yield now
        now += 100.0  # time between updates (checks, logging) is never counted


def use_clock(monkeypatch: pytest.MonkeyPatch, steps: list[float]) -> None:
    readings = scripted_clock(steps)
    monkeypatch.setattr(trainer_module, "synchronized_time", lambda device: next(readings))


def control(
    tmp_path: Path, parent: Path, budget: float, **kwargs: Any
) -> tuple[ControlResult, ScriptedDevEvaluator]:
    evaluator = kwargs.pop("evaluator", None) or ScriptedDevEvaluator(
        script=lambda check: (0.1 + 0.001 * check, 0.0)
    )
    options: dict[str, Any] = {"batch_size": 2, "accumulation_steps": 2, "warmup_updates": 3}
    options.update(kwargs.pop("config", {}))
    result = run_more_sft(
        parent=parent,
        tasks=TASKS,
        encoder=TOKENIZER,
        tokenizer_hash=TOKENIZER.sha256,
        budget_seconds=budget,
        start=start_check(parent),
        evaluator=evaluator,
        output=tmp_path / "control",
        run_id="more-sft",
        config=ControlConfig(**options),
        **kwargs,
    )
    return result, evaluator


def test_control_rows_are_the_rendered_references() -> None:
    rows = control_rows(TOKENIZER, TASKS[:2])
    assert [row.ids for row in rows] == [render_code_task(TOKENIZER, t).ids for t in TASKS[:2]]
    assert [row.mask for row in rows] == [render_code_task(TOKENIZER, t).mask for t in TASKS[:2]]
    assert [(row.id, row.source) for row in rows] == [("t0", "synthetic"), ("t1", "synthetic")]
    with pytest.raises(ValueError):
        control_rows(TOKENIZER, [Task(**{**TASKS[0].__dict__, "solution": None})])


def test_pass_boundaries() -> None:
    assert pass_boundaries(6, 4, (1, 2, 4, 8)) == {2: [1], 3: [2], 6: [4], 12: [8]}
    assert pass_boundaries(3, 4, (1, 2)) == {1: [1], 2: [2]}
    assert pass_boundaries(2, 4, (1, 2)) == {1: [1, 2]}  # one update reads two passes
    assert pass_boundaries(1, 4, (1, 2, 4)) == {1: [1, 2, 4]}
    assert pass_boundaries(771, 32, (1, 2, 4, 8)) == {25: [1], 49: [2], 97: [4], 193: [8]}


def test_the_budget_stops_at_the_nearest_update(
    tmp_path: Path, parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Steps of 0.5 s against a 5 s budget: after 9 updates 4.5 + 0.25 < 5 continues; after
    10, 5.0 + 0.25 >= 5 stops. Six rows at four per update complete passes 1, 2 and 4
    after updates 2, 3 and 6; the end check follows update 10."""
    use_clock(monkeypatch, [0.5] * 50)
    result, evaluator = control(tmp_path, parent, 5.0)
    assert result.stop_reason == "budget" and result.updates == 10
    assert result.busy_seconds == 5.0 and result.mismatch_seconds == 0.0
    assert result.within_tolerance and result.mismatch_fraction == 0.0
    assert evaluator.calls == [2, 3, 6, 10]  # the start check is reused, not repeated
    assert [check.label for check in result.checks] == [
        "",
        "passes-1",
        "passes-2",
        "passes-4",
        "end",
    ]
    assert result.selected.dev.check == 10
    assert result.passes == pytest.approx(10 * 4 / 6)
    assert result.seconds["compile"] == 0.0 and result.seconds["training_step"] == 5.0
    saved = json.loads((tmp_path / "control" / "more-sft" / "control-result.json").read_text())
    assert saved["within_tolerance"] and saved["selected"]["label"] == "end"
    for check in result.checks[1:]:
        assert Path(check.checkpoint).exists()


def test_the_control_reuses_the_sft_trainer_path(
    tmp_path: Path, parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_clock(monkeypatch, [1.0] * 20)
    result, _ = control(tmp_path, parent, 12.0, config={"warmup_updates": 4})
    run = tmp_path / "control" / "more-sft"
    record = json.loads((run / "record.json").read_text())
    assert record["config"]["train"]["loss_normalization"] == "update_targets"
    assert record["config"]["train"]["batch_size"] == 2
    assert record["config"]["train"]["accumulation_steps"] == 2
    assert record["parent_run_id"] == "sft" and record["status"] == "completed"
    lines = [json.loads(line) for line in (run / "events-rank0.jsonl").read_text().splitlines()]
    metrics = [row["fields"]["metrics"] for row in lines if row["event"] == "metrics"]
    rates = [row["learning_rate"] for row in metrics]
    assert rates == pytest.approx([1.5e-5, 3e-5, 4.5e-5] + [6e-5] * (result.updates - 3))
    # Every update trains on assistant targets only: the logged token count is the
    # update's assistant-target count, never its capacity of 2 x 2 x 96 slots.
    assert all(row["tokens"] < 2 * 2 * 96 for row in metrics)


def test_an_uneven_step_reports_the_mismatch(
    tmp_path: Path, parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Steps of 1 s against 2.6 s: after 2, 2 + 0.5 < 2.6 continues; after 3 it stops at
    3 s, 0.4 s (15.4%) over, outside the 5% tolerance, and the result says so."""
    use_clock(monkeypatch, [1.0] * 10)
    result, _ = control(tmp_path, parent, 2.6)
    assert result.updates == 3 and result.busy_seconds == 3.0
    assert result.mismatch_seconds == pytest.approx(0.4)
    assert result.mismatch_fraction == pytest.approx(0.4 / 2.6)
    assert not result.within_tolerance


def test_a_zero_budget_trains_nothing_and_selects_the_start(
    tmp_path: Path, parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_clock(monkeypatch, [])
    result, evaluator = control(tmp_path, parent, 0.0)
    assert result.updates == 0 and result.stop_reason == "budget"
    assert evaluator.calls == [] and result.selected.label == ""
    assert result.selected.dev.check == 0


def test_an_external_stop_and_the_passes_running_out(
    tmp_path: Path, parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_clock(monkeypatch, [0.5] * 50)
    answers = iter([None, None, None, "wall"])
    result, evaluator = control(tmp_path / "a", parent, 100.0, should_stop=lambda: next(answers))
    assert result.stop_reason == "external" and result.stop_detail == "wall"
    assert result.updates == 3 and evaluator.calls == [2, 3]  # passes 1 and 2; no end check
    use_clock(monkeypatch, [0.5] * 50)
    result, evaluator = control(tmp_path / "b", parent, 100.0, config={"max_passes": 2})
    assert result.stop_reason == "passes_exhausted" and result.updates == 3
    assert evaluator.calls == [2, 3]


def test_a_dev_wait_in_the_control_saves_its_weights(
    tmp_path: Path, parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_clock(monkeypatch, [0.5] * 50)
    evaluator = ScriptedDevEvaluator(timeout_at=3)
    result, _ = control(tmp_path, parent, 100.0, evaluator=evaluator)
    assert result.stop_reason == "reward_wait" and result.updates == 3
    assert result.stop_checkpoint is not None
    assert Path(result.stop_checkpoint["checkpoint"]).name == "step-00003.pt"


def test_control_config_validation_and_defaults(parent: Path, tmp_path: Path) -> None:
    defaults = ControlConfig()
    assert (defaults.batch_size, defaults.accumulation_steps) == (16, 2)
    assert defaults.learning_rate == 6e-5 and defaults.warmup_updates == 10
    assert defaults.pass_checks == (1, 2, 4, 8) and defaults.tolerance == 0.05
    for options in ({"pass_checks": (2, 1)}, {"tolerance": 0.0}, {"batch_size": 0}):
        with pytest.raises(ValueError):
            ControlConfig(**options)
    wrong_start = CheckRecord(DevCheck(5, 0.1, 0.1, 0.1, 0.1, 1, 0.0, 0.0), 5, "x", "0" * 64)
    with pytest.raises(ValueError):
        run_more_sft(
            parent=parent,
            tasks=TASKS,
            encoder=TOKENIZER,
            tokenizer_hash=TOKENIZER.sha256,
            budget_seconds=1.0,
            start=wrong_start,
            evaluator=ScriptedDevEvaluator(),
            output=tmp_path,
            run_id="x",
        )


# ---------------------------------------------------------------------- end to end


def warmed_policy(steps: int = 120) -> GPT:
    """A tiny model briefly fitted to the tasks' chat-v1 rows, so its samples hold programs."""
    torch.manual_seed(0)
    model = GPT(MODEL)
    rows = [render_code_task(TOKENIZER, item) for item in TASKS]
    width = max(len(row.ids) for row in rows) - 1
    inputs = torch.tensor([[*row.ids[:-1], *[256] * (width - len(row.ids) + 1)] for row in rows])
    targets = torch.tensor(
        [
            [
                token if keep else IGNORE
                for token, keep in zip(row.ids[1:], row.mask[1:], strict=True)
            ]
            + [IGNORE] * (width - len(row.ids) + 1)
            for row in rows
        ]
    )
    optimizer = AdamW(list(model.parameters()), learning_rate=3e-3, weight_decay=0.0)
    for _ in range(steps):
        optimizer.zero_grad()
        total, count = cross_entropy_sum(model(inputs), targets, ignore_index=IGNORE)
        (total / count).backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    return model


def test_cpu_end_to_end_rl_then_the_more_sft_control(tmp_path: Path) -> None:
    """Three RL updates on a tiny policy with fake rewards and dev checks, then the control
    on the RL run's measured busy time. Real clocks; only robust properties are asserted."""
    policy = warmed_policy()
    parent = tmp_path / "sft-parent.pt"
    save_weights(parent, policy, tokenizer_hash=TOKENIZER.sha256, run_id="sft", step=0)
    evaluator = ScriptedDevEvaluator(script=lambda check: (0.05 * min(check, 2), 0.01 * check))
    rl = run_rl(
        load_weights(parent),
        TASKS,
        encoder=TOKENIZER,
        decoder=TOKENIZER,
        rewards=FakeRewardSource(),
        evaluator=evaluator,
        output=tmp_path / "rl",
        run_id="rl",
        tokenizer_hash=TOKENIZER.sha256,
        config=RLConfig(
            updates=3,
            tasks_per_update=4,
            group_size=4,
            max_new_tokens=40,
            micro_batch=5,
            generation_batch=32,
            warmup_updates=1,
            learning_rate=1e-3,
            dev_checks=(0, 2, 3),
        ),
    )
    assert rl.stop_reason == "completed" and rl.updates == 3
    assert rl.optimizer_steps >= 1  # programs appear, so some groups have unequal rewards
    assert [check.dev.check for check in rl.checks] == [0, 2, 3]
    assert rl.selected is not None and rl.selected.dev.check == 3
    first = torch.load(rl.checks[0].checkpoint, weights_only=True)["model"]
    last = torch.load(rl.checks[-1].checkpoint, weights_only=True)["model"]
    assert any(not torch.equal(first[name], last[name]) for name in first)
    assert rl.seconds["rollout_generation"] > 0 and rl.seconds["training_step"] > 0
    monitoring = rl.monitoring
    assert len(monitoring) == 3 and sum(row["no_program"] for row in monitoring) < 3 * 16
    budget = rl.busy_seconds
    result = run_more_sft(
        parent=parent,
        tasks=TASKS,
        encoder=TOKENIZER,
        tokenizer_hash=TOKENIZER.sha256,
        budget_seconds=budget,
        start=rl.checks[0],
        evaluator=evaluator,
        output=tmp_path / "control",
        run_id="more-sft",
        config=ControlConfig(
            batch_size=2, accumulation_steps=2, warmup_updates=2, pass_checks=(1, 2)
        ),
    )
    assert result.budget_seconds == budget and result.stop_reason in {"budget", "passes_exhausted"}
    assert result.checks[0] is rl.checks[0]
    if result.stop_reason == "budget" and result.step_seconds:
        # The stop rule keeps the mismatch within one update's time.
        assert abs(result.mismatch_seconds) <= max(result.step_seconds)
    assert result.selected in result.checks
    record = json.loads((tmp_path / "control" / "more-sft" / "control-result.json").read_text())
    assert record["budget_seconds"] == budget and "mismatch_fraction" in record
