"""The run-2 schedule simulator: it reproduces the documented truncation, its inner loops make
the real loops' calls in the real order, and it drives the real worker's decision code.

The conformance tests run ``scratchcoder.rl``'s real ``run_rl`` and ``run_more_sft`` on a tiny
CPU model next to the simulator's ``simulated_rl`` and ``simulated_control`` with the same
scripted stops, and compare every reward request, dev check, stop and selection. Nothing
touches a network and no generated code runs.
"""

import json
import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from scratchcoder.common.config import ModelConfig
from scratchcoder.data.chat import PYTHON_END, PYTHON_START
from scratchcoder.eval.tasks import Task
from scratchcoder.model.gpt import GPT
from scratchcoder.rl import loop
from scratchcoder.rl.checkpoint import save_weights
from scratchcoder.rl.control import ControlConfig, run_more_sft
from scratchcoder.rl.dev import DevCheck
from scratchcoder.rl.fakes import FakeRewardSource, ScriptedDevEvaluator
from scratchcoder.rl.loop import CheckRecord, RLConfig, run_rl
from scratchcoder.rl.rewards import Attempt
from scratchcoder.rl.rollout import Rollout
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.train import trainer as trainer_module
from tools import posttrain_approved, posttrain_run2, run2_schedule_sim
from tools.posttrain_run import (
    CONTROL_REFERENCE,
    WAIT_MARGIN_STOP,
    PosttrainWorker,
    Rates,
    Specification,
)
from tools.posttrain_run2 import Run2Worker
from tools.run2_schedule_sim import (
    Scenario,
    SimClock,
    SimulatedRun2Worker,
    minimum_stop_fraction,
    patched_worker,
    scaled_rates,
    simulate,
    simulated_control,
    simulated_rl,
)

TOKENIZER = ByteBPE()


def task(index: int) -> Task:
    return Task(
        id=f"t{index}",
        prompt=f"Add {index}.",
        public_tests=f"assert f(1) == {1 + index}",
        hidden_tests=f"assert f(2) == {2 + index}",
        source="synthetic",
        license="fixture",
        written_by="human",
        split="train",
        solution=f"def f(x):\n    return x + {index}",
    )


TASKS = [task(index) for index in range(6)]


# The margin-stop table (docs/posttrain-runs.md, "An arm's stop") --------------------------

# (Mac time, GPU time) -> (estimate, stop, first stop after a 900 s setup for 320 and 1,000
# kept tasks, the same after a 1,500 s setup), each stop in the simulator's words, at the
# current specification (post-2-v2: planning generation at 7,991 and 9,049 tok/s).
CURRENT = {
    (1.0, 1.0): (
        "3.10",
        "3.72",
        ("the control, before update 11", "the control, before update 33"),
        (
            "RL, at its dev check after update 100, unscored",
            "RL, at its dev check after update 100, unscored",
        ),
    ),
    (1.1, 1.0): (
        "3.24",
        "3.88",
        ("the control, before update 11", "the control, before update 33"),
        ("the control, before update 1", "the control, before update 1"),
    ),
    (1.5, 1.0): (
        "3.78",
        "4.53",
        ("the control, before update 41", "the control, before update 126"),
        ("the control, before update 21", "the control, before update 64"),
    ),
    (2.0, 1.0): (
        "4.45",
        "5.34",
        ("the control, before update 727", "the control, before update 727"),
        ("the control, before update 81", "the control, before update 251"),
    ),
    (0.9, 1.0): (
        "2.97",
        "3.56",
        ("the control, before update 1", "the control, before update 1"),
        ("RL, before update 95", "RL, before update 95"),
    ),
    (0.75, 1.0): (
        "2.77",
        "3.31",
        ("RL, before update 95", "RL, before update 95"),
        ("RL, before update 82", "RL, before update 82"),
    ),
    (0.7, 1.0): (
        "2.70",
        "3.23",
        ("RL, before update 91", "RL, before update 91"),
        None,  # the run stops before RL: test_a_late_low_estimate_run_stops_before_rl
    ),
    (1.0, 0.9): (
        "2.96",
        "3.55",
        ("RL, before update 96", "RL, before update 96"),
        ("RL, before update 86", "RL, before update 86"),
    ),
    (1.0, 0.8): (
        "2.83",
        "3.39",
        ("RL, before update 83", "RL, before update 83"),
        ("RL, before update 76", "RL, before update 76"),
    ),
}

# The same at the superseded first template (planning generation at 7,731 and 8,733 tok/s):
# the table docs/posttrain-runs.md keeps as the historical scenario.
HISTORICAL = {
    (1.0, 1.0): (
        "3.15",
        "3.77",
        ("the control, before update 11", "the control, before update 33"),
        ("the control, before update 1", "the control, before update 1"),
    ),
    (1.1, 1.0): (
        "3.28",
        "3.93",
        ("the control, before update 21", "the control, before update 64"),
        (
            "the control, at its dev check after update 10, unscored",
            "the control, at its dev check after update 32, unscored",
        ),
    ),
    (1.5, 1.0): (
        "3.82",
        "4.58",
        ("the control, before update 81", "the control, before update 251"),
        ("the control, before update 41", "the control, before update 126"),
    ),
    (2.0, 1.0): (
        "4.50",
        "5.39",
        ("the control, before update 1138", "the control, before update 1138"),
        ("the control, before update 98", "the control, before update 251"),
    ),
    (0.9, 1.0): (
        "3.01",
        "3.61",
        (
            "the control, at its dev check after update 10, unscored",
            "the control, at its dev check after update 32, unscored",
        ),
        ("RL, before update 99", "RL, before update 99"),
    ),
    (0.75, 1.0): (
        "2.81",
        "3.36",
        ("RL, before update 100", "RL, before update 100"),
        ("RL, before update 87", "RL, before update 87"),
    ),
    (0.7, 1.0): (
        "2.74",
        "3.28",
        ("RL, before update 96", "RL, before update 96"),
        ("RL, before update 83", "RL, before update 83"),
    ),
    (1.0, 0.9): (
        "3.00",
        "3.60",
        (
            "RL, at its dev check after update 100, unscored",
            "RL, at its dev check after update 100, unscored",
        ),
        ("RL, before update 91", "RL, before update 91"),
    ),
    (1.0, 0.8): (
        "2.86",
        "3.43",
        ("RL, before update 86", "RL, before update 86"),
        ("RL, before update 76", "RL, before update 76"),
    ),
}

# (Mac time, GPU time, setup) -> the share of the control's budget spent, in whole percent,
# for 320 and 1,000 kept tasks; None for "under 2%".
CURRENT_SHARE: dict[tuple[float, float, float], tuple[int, int] | None] = {
    (1.0, 1.0, 900.0): None,
    (1.1, 1.0, 900.0): None,
    (1.5, 1.0, 900.0): (2, 6),
    (2.0, 1.0, 900.0): (35, 35),
    (1.5, 1.0, 1500.0): (1, 3),
    (2.0, 1.0, 1500.0): (4, 12),
}
HISTORICAL_SHARE: dict[tuple[float, float, float], tuple[int, int] | None] = {
    (1.0, 1.0, 900.0): None,
    (1.1, 1.0, 900.0): (1, 3),
    (1.5, 1.0, 900.0): (4, 12),
    (2.0, 1.0, 900.0): (54, 54),
    (1.5, 1.0, 1500.0): (2, 6),
    (2.0, 1.0, 1500.0): (5, 12),
}
Documented = dict[tuple[float, float], tuple[str, str, Any, Any]]
Shares = dict[tuple[float, float, float], tuple[int, int] | None]
TABLES: dict[str, tuple[Documented, Shares]] = {
    run2_schedule_sim.CURRENT_TEMPLATE: (CURRENT, CURRENT_SHARE),
    run2_schedule_sim.HISTORICAL_TEMPLATE: (HISTORICAL, HISTORICAL_SHARE),
}


def check_truncation(template: str, mac: float, gpu: float) -> None:
    documented, documented_share = TABLES[template]
    estimate, stop, short, long = documented[(mac, gpu)]
    for setup, expected in ((900.0, short), (1500.0, long)):
        if expected is None:
            continue
        shares: list[float] = []
        for tasks, where in zip((320, 1000), expected, strict=True):
            row = simulate(
                Scenario(mac=mac, gpu=gpu, setup_seconds=setup, tasks=tasks, template=template)
            )
            assert row["specification"]["name"] == template
            assert f"{row['estimate_usd']:.2f}" == estimate
            assert f"{row['cost_limit_usd']:.2f}" == stop
            assert row["first_stop"] == f"{where} ({WAIT_MARGIN_STOP})"
            assert row["outcome"] == "stopped" and row["stop_reason"] == WAIT_MARGIN_STOP
            assert row["margin_stop"]["wait_seconds"] == 11_746.5
            if row["control"] is not None:
                shares.append(100 * row["control"]["busy_fraction"])
        if (mac, gpu, setup) in documented_share:
            share = documented_share[(mac, gpu, setup)]
            if share is None:
                assert len(shares) == 2 and all(value < 2 for value in shares)
            else:
                assert tuple(round(value) for value in shares) == share


@pytest.mark.parametrize(("mac", "gpu"), sorted(CURRENT))
def test_the_current_truncation_is_reproduced(mac: float, gpu: float) -> None:
    """Every row of the docs' table at the current specification (the default template): the
    estimate projected at the rates, the 120% stop, the first margin stop for 320 and 1,000
    kept tasks after a 900 s and a 1,500 s setup, and the share of the control's budget spent
    where the table gives one."""
    check_truncation(run2_schedule_sim.CURRENT_TEMPLATE, mac, gpu)


@pytest.mark.parametrize(("mac", "gpu"), sorted(HISTORICAL))
def test_the_historical_truncation_is_reproduced(mac: float, gpu: float) -> None:
    """The historical scenario: the same rows at the superseded first template, the figures
    the earlier analysis gave ($3.15 and $3.77 at the planning rates)."""
    check_truncation(run2_schedule_sim.HISTORICAL_TEMPLATE, mac, gpu)


@pytest.mark.parametrize(
    ("template", "low", "high"),
    [
        (run2_schedule_sim.CURRENT_TEMPLATE, 2.56, 2.56),
        (run2_schedule_sim.HISTORICAL_TEMPLATE, 2.59, 2.61),
    ],
)
@pytest.mark.parametrize(("mac", "gpu"), [(0.6, 1.0), (0.8, 0.8)])
def test_the_documented_gate_refusals_are_reproduced(
    template: str, low: float, high: float, mac: float, gpu: float
) -> None:
    """The table's last row: refused at the gate for 320 and 1,000 kept tasks after either
    setup, at an estimate of $2.56 (current) or $2.59 to $2.61 (historical)."""
    for setup in (900.0, 1500.0):
        for tasks in (320, 1000):
            row = simulate(
                Scenario(mac=mac, gpu=gpu, setup_seconds=setup, tasks=tasks, template=template)
            )
            assert low <= round(row["estimate_usd"], 2) <= high
            assert row["outcome"] == "refused at the gate" and "rl" not in row


def test_a_late_low_estimate_run_stops_before_rl() -> None:
    """At the current specification, a Mac at 0.7 of the planning time gives so low an
    estimate that a 1,500 s setup leaves less than the projected-finish rule needs: the run
    stops at the RL stage's start, nothing trained, and finalizes."""
    for tasks in (320, 1000):
        row = simulate(Scenario(mac=0.7, setup_seconds=1500.0, tasks=tasks))
        assert row["outcome"] == "stopped"
        assert row["stop_reason"] == "projected_finish_after_deadline"
        assert row["rl"] is None and row["control"] is None and row["arm_stop"] is None
        assert row["stages"]["rl"] == "stopped"
        assert row["pod_elapsed_seconds"] == 1500.0


DOCS = Path(__file__).resolve().parents[1] / "docs" / "posttrain-runs.md"


def test_the_docs_tables_are_the_tool_output() -> None:
    """docs/posttrain-runs.md's margin-stop table is ``margin_table`` at the current
    specification, byte for byte, and the historical table it keeps is the same rendering at
    the superseded template, labelled as the historical scenario after the current one."""
    text = DOCS.read_text(encoding="utf-8")
    current = run2_schedule_sim.margin_table(
        run2_schedule_sim.margin_rows(run2_schedule_sim.CURRENT_TEMPLATE)
    )
    historical = run2_schedule_sim.margin_table(
        run2_schedule_sim.margin_rows(run2_schedule_sim.HISTORICAL_TEMPLATE)
    )
    assert "| 1.0, 1.0 | $3.10, $3.72 |" in current
    assert "| 1.0, 1.0 | $3.15, $3.77 |" in historical
    assert text.count(current) == 1 and text.count(historical) == 1
    at, before = text.index(current), text.index(historical)
    assert at < before
    label = text.rfind("**Historical scenario", 0, before)
    assert at < label < before
    assert "superseded" in text[label:before] and "never launched" in text[label:before]


def test_the_template_defaults_to_the_current_specification() -> None:
    """The simulator, its projection and its CLI default to post-2-v2; the superseded template
    is reachable only by its historical name."""
    chosen = run2_schedule_sim.template_named(run2_schedule_sim.CURRENT_TEMPLATE)
    assert chosen.path == posttrain_approved.SPECIFICATION_PATH_2
    assert chosen.path.name == "rehearsal-a40-s-post-2-v2.json" and not chosen.historical
    assert Scenario().template == run2_schedule_sim.CURRENT_TEMPLATE
    old = run2_schedule_sim.template_named(run2_schedule_sim.HISTORICAL_TEMPLATE)
    assert old.historical and old.path in posttrain_approved.SUPERSEDED_RUN2_TEMPLATES
    assert old.record()["sha256"] == posttrain_approved.SUPERSEDED_RUN2_TEMPLATES[old.path]
    assert "HISTORICAL" in old.record()["description"]
    row = simulate(Scenario(tasks=320))
    assert row["specification"]["path"] == "configs/rehearsal-a40-s-post-2-v2.json"
    assert not row["historical"]
    assert simulate(Scenario(tasks=320, template=old.name))["historical"]
    with pytest.raises(ValueError, match="no run-2 template"):
        run2_schedule_sim.template_named("post-2")


def test_a_template_with_other_bytes_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The current template must equal ``run2_specification()`` and the historical one its
    pinned SHA-256; a projection report of other bytes is refused too."""
    from dataclasses import replace

    templates = dict(run2_schedule_sim.TEMPLATES)
    current = templates[run2_schedule_sim.CURRENT_TEMPLATE]
    changed = tmp_path / current.path.name
    changed.write_text(current.path.read_text().replace("7991.0", "7992.0"))
    monkeypatch.setitem(run2_schedule_sim.TEMPLATES, current.name, replace(current, path=changed))
    with pytest.raises(ValueError, match="differs from"):
        simulate(Scenario(tasks=320))
    old = templates[run2_schedule_sim.HISTORICAL_TEMPLATE]
    monkeypatch.setitem(run2_schedule_sim.TEMPLATES, old.name, replace(old, path=changed))
    with pytest.raises(ValueError, match="not the pinned superseded"):
        simulate(Scenario(tasks=320, template=old.name))
    monkeypatch.setitem(run2_schedule_sim.TEMPLATES, current.name, current)
    report = tmp_path / "projection.json"
    report.write_text(
        json.dumps(
            {
                "kind": "posttrain-feasibility",
                "run": "2",
                "specification_sha256": "0" * 64,
                "scenarios": {},
            }
        )
    )
    with pytest.raises(ValueError, match="priced another specification"):
        run2_schedule_sim.projection_scenarios(
            report, setups=(900.0,), tasks=(320,), ceiling_usd=10.0
        )


def test_the_planning_rates_cut_the_control_to_under_two_percent_of_its_budget() -> None:
    """The documented case in numbers, at the current specification: RL runs its 100
    updates; the control, about 2,055 updates of budget, stops after its pass-1 check,
    unmatched; it reuses no SFT result because it trained; RL's endpoint is evaluated and the
    run ends stopped, finalized in full within the reserve."""
    row = simulate(Scenario(tasks=320))
    assert row["rl"]["reached_budget"] and row["rl"]["updates"] == 100
    control = row["control"]
    assert control["budget_updates"] == 2055 and control["updates"] == 10
    assert control["busy_fraction"] < 0.02 and not control["within_tolerance"]
    assert control["stop_reason"] == "external" and control["checks"] == [0, 10]
    margin = row["margin_stop"]
    assert (margin["kind"], margin["limit"], margin["wait_seconds"]) == (
        "next update",
        "finalization_reserve",
        11_746.5,
    )
    assert row["arm_stop"]["arm"] == "more-sft" and row["arm_stop"]["kind"] == "margin"
    assert row["stages"] == {
        "rl": "completed",
        "more-sft": "stopped",
        "arc-post": "completed",
        "eval-rl": "completed",
        "eval-more-sft": "completed",
    }
    assert row["endpoints"]["more-sft"]["selected"] == "more-sft-check-00010"
    assert row["finalization"]["seconds_left_before_wall"] >= 900.0
    assert row["forecast"].startswith("forecast at the given rates")
    # After a 1,500 s setup RL's last dev check is refused its Mac wait: RL selects its
    # update-75 check, the control is skipped and only RL's endpoint is evaluated.
    late = simulate(Scenario(tasks=320, setup_seconds=1500.0))
    assert late["rl"]["updates"] == 100 and late["control"] is None
    assert late["arm_stop"]["arm"] == "rl" and late["margin_stop"]["kind"] == "dev"
    assert late["stages"]["more-sft"] == "skipped"
    assert late["endpoints"]["rl"]["selected"] == "rl-check-0075"


def test_the_historical_planning_rates_left_the_control_its_start() -> None:
    """The historical scenario's documented case: at the superseded template the control's
    budget was about 2,100 updates, and after a 1,500 s setup the control trained nothing and
    selected its start, RL's update-0 check, so the SFT results stood for it."""
    old = run2_schedule_sim.HISTORICAL_TEMPLATE
    row = simulate(Scenario(tasks=320, template=old))
    assert row["control"]["budget_updates"] == 2100 and row["control"]["updates"] == 10
    late = simulate(Scenario(tasks=320, setup_seconds=1500.0, template=old))
    assert late["control"]["updates"] == 0
    assert late["endpoints"]["more-sft"] == {
        "selected": "rl-check-0000",
        "evaluation": "reuses-run-1-sft",
        "independent": False,
    }


def test_a_low_estimate_is_refused_at_the_gate() -> None:
    row = simulate(Scenario(mac=0.6, tasks=320))
    assert row["outcome"] == "refused at the gate" and "rl" not in row
    assert "leaves" in row["gate"]["refusal"]


def test_a_pool_below_the_floor_is_flagged_and_the_passes_can_run_out() -> None:
    """161 kept tasks are below the 320 floor (the worker refuses them; the simulation is a
    hypothetical). At a rate where the control is not cut short, its 400-pass loader plans
    2,013 updates, fewer than the 2,055 its budget buys: it stops at passes_exhausted."""
    row = simulate(Scenario(mac=2.0, gpu=1.2, tasks=161))
    assert row["floor"] == 320 and not row["admitted_by_floor"]
    control = row["control"]
    assert control["stop_reason"] == "passes_exhausted" and control["updates"] == 2013
    assert control["planned_updates"] == 2013 and control["budget_updates"] == 2055
    assert 0.97 < control["busy_fraction"] < 0.98 and control["within_tolerance"]


def test_a_higher_stop_fraction_lets_the_control_spend_its_budget() -> None:
    """A hypothetical, never a rule in force: the search returns the lowest stop fraction at
    which both arms reach their budgets with the fixed margin; one step less does not."""
    found = minimum_stop_fraction(Scenario(tasks=320))
    assert found["hypothetical"] and found["stop_fraction"] > 1.2
    record = simulate(Scenario(tasks=320, stop_fraction=found["stop_fraction"]))
    assert record["hypothetical_stop"] and record["stop_fraction"] == found["stop_fraction"]
    assert record["outcome"] == "completed" and record["control"]["within_tolerance"]
    below = simulate(Scenario(tasks=320, stop_fraction=round(found["stop_fraction"] - 0.01, 4)))
    assert below["outcome"] == "stopped"
    assert not simulate(Scenario(tasks=320))["hypothetical_stop"]


def test_the_simulated_worker_inherits_every_decision_method() -> None:
    """The decision code is the real worker's: none of it is overridden."""
    for name in (
        "execute",
        "run_stage",
        "run_rl",
        "run_more_sft",
        "should_stop",
        "margin_reason",
        "before_wait",
        "after_wait",
        "wait_sleep",
        "observe_wait",
        "stopped",
        "tick",
        "dev_tick",
        "projected_finish",
        "drop_control_for_rl",
        "remaining_stages",
        "skip_reason",
        "measure_rl",
        "stop_kind",
        "stop_arm",
        "choose_endpoint",
        "after_stages",
        "observe_generation",
        "account",
    ):
        assert name not in vars(SimulatedRun2Worker), name
        assert getattr(SimulatedRun2Worker, name) is getattr(Run2Worker, name)


def sim_worker() -> SimulatedRun2Worker:
    clock = SimClock()
    spec = Specification.from_dict(
        posttrain_approved.run2_specification(
            run2_schedule_sim.gate_values(ceiling_usd=10.0, estimate_usd=3.15)
        )
    )
    return SimulatedRun2Worker(
        spec, clock=clock, rates=Rates.planning(spec), tasks=320, pod_created=clock()
    )


SWAPPED = (
    "run_rl_loop",
    "run_more_sft",
    "load_weights",
    "BridgeRewardSource",
    "compare_to_loaded",
    "file_sha256",
)


def test_the_patch_is_undone_after_the_run() -> None:
    """Every name ``patched_worker`` swaps is swapped during the run and restored after it."""
    saved = {name: getattr(posttrain_run2, name) for name in SWAPPED}
    with patched_worker(sim_worker()):
        assert all(getattr(posttrain_run2, name) is not saved[name] for name in SWAPPED)
    assert {name: getattr(posttrain_run2, name) for name in SWAPPED} == saved
    simulate(Scenario(tasks=320))
    assert {name: getattr(posttrain_run2, name) for name in SWAPPED} == saved


@pytest.mark.parametrize("left_out", [{}, {CONTROL_REFERENCE: "not evaluated: a reason"}])
def test_the_simulated_arc_keeps_the_real_bookkeeping(
    left_out: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drift guard for the one stage body the simulator repeats: the real
    ``Run2Worker.run_arc``, with the model-bound ARC body beneath it replaced by a recorder,
    and the simulated one evaluate the same checkpoints and record the same left out."""
    simulated, real = sim_worker(), sim_worker()
    stage = next(stage for stage in simulated.plan.stages if stage.kind == "arc")
    assert CONTROL_REFERENCE in stage.settings["checkpoints"]
    evaluated: list[list[str]] = []

    def recorder(self: Any, arc: Any) -> list[dict[str, Any]]:
        evaluated.append(list(arc.settings["checkpoints"]))
        return []

    monkeypatch.setattr(PosttrainWorker, "run_arc", recorder)
    for worker in (simulated, real):
        worker.not_on_pod.update(left_out)
        worker.produced.update({ref: Path(f"{ref}.pt") for ref in ("rl", CONTROL_REFERENCE)})
    simulated.run_arc(stage)
    Run2Worker.run_arc(real, stage)
    simulated_labels = [label.removeprefix(f"{stage.name}-") for label in simulated.labels]
    assert evaluated == [simulated_labels]
    record, expected = (
        simulated.stage_by_name[stage.name],
        real.stage_by_name[stage.name],
    )
    assert record.get("checkpoints_not_evaluated") == expected.get("checkpoints_not_evaluated")
    assert ("checkpoints_not_evaluated" in expected) == bool(left_out)


def test_a_limit_reached_during_a_wait_is_the_real_wait_limit_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the margin disabled (as the worker's own test does) and a Mac whose first dev
    check outlasts the Pod, the fake wait is observed by the real ``observe_wait``, which
    ends it at the wall: what the margin exists to prevent."""
    from dataclasses import replace

    from tools import posttrain_approved

    monkeypatch.setattr(Run2Worker, "margin_reason", lambda self, wait_seconds: None)
    spec = Specification.from_dict(posttrain_approved.run2_specification(), allow_gate_nulls=True)
    rates = replace(Rates.planning(spec), dev_wait_seconds_per_check=50_000.0)
    row = simulate(Scenario(tasks=320, rates=rates, estimate_usd=3.145))
    assert row["outcome"] == "stopped" and row["stop_reason"] == "pod_wall_limit"
    assert row["rl"]["stop_reason"] == "reward_wait" and row["rl"]["updates"] == 0
    assert row["control"] is None and row["margin_stop"] is None
    assert row["pod_elapsed_seconds"] == pytest.approx(22_440, abs=10)


def test_the_rates_scale_gpu_and_mac_time_only() -> None:
    from tools import posttrain_approved

    spec = Specification.from_dict(posttrain_approved.run2_specification(), allow_gate_nulls=True)
    planning = Rates.planning(spec)
    rates = scaled_rates(spec, mac=2.0, gpu=0.5)
    assert rates.reward_wait_seconds_per_update == 2 * (
        planning.reward_wait_seconds_per_update or 0
    )
    assert rates.training_slots_per_second == 2 * planning.training_slots_per_second
    assert rates.arc_seconds_per_checkpoint == 0.5 * planning.arc_seconds_per_checkpoint
    assert rates.checkpoint_load_seconds == planning.checkpoint_load_seconds
    with pytest.raises(ValueError):
        scaled_rates(spec, mac=0.0, gpu=1.0)


def test_the_simulated_worker_sets_every_attribute_the_real_one_does(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A drift guard: the real worker's constructor sets no attribute the simulated one
    lacks, except the file and R2 plumbing the fakes replace."""
    from tools import posttrain_approved
    from tools.posttrain_smoke2 import fixtures2, make_run2_worker
    from tools.rehearsal_run import MemoryBucket

    torch.set_num_threads(1)
    found = fixtures2(tmp_path_factory.mktemp("sim-fixtures"))
    real = make_run2_worker(found, tmp_path_factory.mktemp("sim") / "run", bucket=MemoryBucket())
    spec = Specification.from_dict(
        posttrain_approved.run2_specification(
            run2_schedule_sim.gate_values(ceiling_usd=10.0, estimate_usd=3.15)
        )
    )
    clock = SimClock()
    fake = SimulatedRun2Worker(
        spec, clock=clock, rates=Rates.planning(spec), tasks=320, pod_created=clock()
    )
    plumbing = {"store", "check_store_for", "uploader", "tokenizer_path"}
    assert set(vars(real)) - set(vars(fake)) == plumbing
    with patched_worker(fake):
        swapped: Any = vars(posttrain_run2)["BridgeRewardSource"]
        assert swapped is run2_schedule_sim.FakeRewardSource


def test_the_command_writes_the_forecast(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "sim.json"
    code = run2_schedule_sim.main(
        ["--mac", "1", "--gpu", "1", "--setup", "900", "--tasks", "320", "--output", str(output)]
    )
    assert code == 0
    saved = json.loads(output.read_text())
    assert saved["format"] == "run2-schedule-simulation-v1"
    assert saved["forecast"].startswith("forecast at the given rates")
    assert [row["first_stop"] for row in saved["scenarios"]] == [
        "the control, before update 11 (wait_margin)"
    ]
    assert saved["specification"]["name"] == "post-2-v2" and not saved["historical"]
    out = capsys.readouterr().out
    assert "forecast at the given rates" in out and "HISTORICAL" not in out
    assert "configs/rehearsal-a40-s-post-2-v2.json" in out


def test_the_command_labels_the_historical_scenario(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The superseded template runs only when named, and every output of it says it is the
    historical scenario."""
    output = tmp_path / "sim.json"
    code = run2_schedule_sim.main(
        [
            "--template",
            "post-2-superseded",
            "--mac",
            "1",
            "--gpu",
            "1",
            "--setup",
            "900",
            "--tasks",
            "320",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    saved = json.loads(output.read_text())
    assert saved["historical"] and saved["specification"]["name"] == "post-2-superseded"
    assert all(row["historical"] for row in saved["scenarios"])
    assert saved["scenarios"][0]["estimate_usd"] == pytest.approx(3.145, abs=0.0005)
    out = capsys.readouterr().out
    assert out.startswith("HISTORICAL scenario: run 2's superseded first template")


def test_the_command_prints_the_margin_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert run2_schedule_sim.main(["--margin-table"]) == 0
    out = capsys.readouterr().out
    assert out.endswith(
        run2_schedule_sim.margin_table(
            run2_schedule_sim.margin_rows(run2_schedule_sim.CURRENT_TEMPLATE)
        )
    )


# Conformance: the simulator's loops make the real loops' calls -----------------------------


def scripted_rollouts(
    model: GPT,
    tasks: Sequence[Task],
    prompts: Sequence[Sequence[int]],
    decoder: Any,
    *,
    samples: int,
    seed_of: Any,
    **options: Any,
) -> list[Rollout]:
    """``generate_rollouts`` without a model: sample s writes a program of s + 1 letters."""
    rows = []
    for item, prompt in zip(tasks, prompts, strict=True):
        for sample in range(samples):
            body = (PYTHON_START, *([ord("a")] * (sample + 1)), PYTHON_END)
            attempt = Attempt(item.id, sample, "a" * (sample + 1), len(body))
            rows.append(Rollout(attempt, tuple(prompt), body, "stop", seed_of(item, sample)))
    return rows


def rl_config(**overrides: Any) -> RLConfig:
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


def stops(answers: Sequence[Any]) -> Any:
    script = iter(answers)
    return lambda: next(script, None)


RL_CASES: dict[str, dict[str, Any]] = {
    "completed with an end check": {},
    "external stop": {"answers": [None, None, "cost"], "updates": 10, "dev_checks": (0,)},
    "external stop, checked": {
        "answers": [None, None, "cost"],
        "updates": 10,
        "dev_checks": (0,),
        "check_at_external_stop": True,
    },
    "reward timeout": {"reward_timeout": 2},
    "dev timeout": {"dev_timeout": 2},
    "dev timeout at check 0": {"dev_timeout": 0},
    "dev patience": {
        "falling": True,
        "updates": 6,
        "dev_checks": (0, 1, 2, 3, 4),
    },
}


def summary(result: Any) -> dict[str, Any]:
    return {
        "stop_reason": result.stop_reason,
        "updates": result.updates,
        "optimizer_steps": result.optimizer_steps,
        "checks": [(check.dev.check, check.optimizer_steps) for check in result.checks],
        "selected": None if result.selected is None else result.selected.dev.check,
        "stop_weights": result.stop_checkpoint is not None,
    }


@pytest.mark.parametrize("case", sorted(RL_CASES))
def test_the_simulated_rl_loop_makes_the_real_loops_calls(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = dict(RL_CASES[case])
    answers = options.pop("answers", [])
    reward_timeout = options.pop("reward_timeout", None)
    dev_timeout = options.pop("dev_timeout", None)
    falling = options.pop("falling", False)
    config = rl_config(**options)

    def parts() -> tuple[FakeRewardSource, ScriptedDevEvaluator, list[str], Any]:
        calls: list[str] = []

        def should_stop() -> Any:
            calls.append("should_stop")
            return answer()

        answer = stops(answers)
        rewards = FakeRewardSource(timeout_at=reward_timeout)
        evaluator = ScriptedDevEvaluator(
            script=(lambda check: (0.5 - 0.1 * check, 0.0)) if falling else (lambda c: (0.0, 0.0)),
            timeout_at=dev_timeout,
        )
        return rewards, evaluator, calls, should_stop

    monkeypatch.setattr(loop, "generate_rollouts", scripted_rollouts)
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=64))
    real_rewards, real_dev, real_calls, real_stop = parts()
    real = run_rl(
        model,
        TASKS,
        encoder=TOKENIZER,
        decoder=TOKENIZER,
        rewards=real_rewards,
        evaluator=real_dev,
        output=tmp_path / "rl",
        run_id="rl-conformance",
        tokenizer_hash="authored",
        config=config,
        should_stop=real_stop,
    )
    sim_rewards, sim_dev, sim_calls, sim_stop = parts()
    simulated = simulated_rl(
        tasks=TASKS,
        rewards=sim_rewards,
        evaluator=sim_dev,
        config=config,
        should_stop=sim_stop,
        on_check=None,
        gpu=lambda seconds: None,
        now=lambda: 0.0,
        timing=None,
    )
    assert summary(simulated) == summary(real)
    assert [update for update, _ in sim_rewards.calls] == [
        update for update, _ in real_rewards.calls
    ]
    assert sim_dev.calls == real_dev.calls
    assert sim_calls == real_calls


@pytest.fixture
def parent(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    path = tmp_path / "parent.pt"
    model = GPT(ModelConfig(vocab_size=272, width=32, layers=2, heads=2, context=96))
    save_weights(path, model, tokenizer_hash=TOKENIZER.sha256, run_id="sft", step=0)
    return path


def scripted_clock(steps: Sequence[float]) -> Iterator[float]:
    now = 0.0
    for seconds in steps:
        yield now
        now += seconds
        yield now
        now += 100.0


CONTROL_CASES: dict[str, dict[str, Any]] = {
    "budget": {"budget": 5.0, "steps": [0.5] * 50},
    "uneven budget": {"budget": 2.6, "steps": [1.0] * 10},
    "uneven steps": {"budget": 4.0, "steps": [0.3, 0.9, 0.4, 1.1, 0.2, 0.8] * 5},
    "zero budget": {"budget": 0.0, "steps": []},
    "external stop": {"budget": 100.0, "steps": [0.5] * 50, "answers": [None, None, None, "wall"]},
    "passes exhausted": {"budget": 100.0, "steps": [0.5] * 50, "max_passes": 2},
    "dev timeout": {"budget": 100.0, "steps": [0.5] * 50, "dev_timeout": 3},
}


def control_summary(result: Any) -> dict[str, Any]:
    return {
        "stop_reason": result.stop_reason,
        "updates": result.updates,
        "busy_seconds": round(result.busy_seconds, 9),
        "within_tolerance": result.within_tolerance,
        "checks": [(check.dev.check, check.label) for check in result.checks],
        "selected": result.selected.dev.check,
        "stop_weights": result.stop_checkpoint is not None,
    }


@pytest.mark.parametrize("case", sorted(CONTROL_CASES))
def test_the_simulated_control_makes_the_real_controls_calls(
    case: str, tmp_path: Path, parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = dict(CONTROL_CASES[case])
    steps: list[float] = options["steps"]
    config = ControlConfig(
        batch_size=2,
        accumulation_steps=2,
        warmup_updates=3,
        max_passes=options.get("max_passes", 400),
    )
    start = CheckRecord(DevCheck(0, 0.1, 0.05, 0.2, 0.1, 200, 0.0, 0.0), 0, str(parent), "0" * 64)
    score = lambda check: (0.1 + 0.001 * check, 0.0)  # noqa: E731

    real_calls: list[str] = []
    answer = stops(options.get("answers", []))

    def real_stop() -> Any:
        real_calls.append("should_stop")
        return answer()

    readings = scripted_clock(steps)
    monkeypatch.setattr(trainer_module, "synchronized_time", lambda device: next(readings))
    real_dev = ScriptedDevEvaluator(script=score, timeout_at=options.get("dev_timeout"))
    real = run_more_sft(
        parent=parent,
        tasks=TASKS,
        encoder=TOKENIZER,
        tokenizer_hash=TOKENIZER.sha256,
        budget_seconds=options["budget"],
        start=start,
        evaluator=real_dev,
        output=tmp_path / "control",
        run_id="more-sft",
        config=config,
        should_stop=real_stop,
    )
    sim_calls: list[str] = []
    answer = stops(options.get("answers", []))

    def sim_stop() -> Any:
        sim_calls.append("should_stop")
        return answer()

    sim_dev = ScriptedDevEvaluator(script=score, timeout_at=options.get("dev_timeout"))
    simulated = simulated_control(
        rows=real.rows_per_pass,
        budget_seconds=options["budget"],
        start=start,
        evaluator=sim_dev,
        config=config,
        should_stop=sim_stop,
        on_check=None,
        step_seconds=lambda update: steps[update - 1],
        gpu=lambda seconds: None,
    )
    assert real.rows_per_pass == len(TASKS)
    assert control_summary(simulated) == control_summary(real)
    assert sim_dev.calls == real_dev.calls
    assert sim_calls == real_calls
    assert math.isclose(simulated.passes, real.passes)
