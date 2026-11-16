"""Run 1c (agent SFT from run 1b's SFT weights): its approved specification, the bind that
fills its placeholders, the checker's refusals, the parser's rules, the worker's final-weights
selection and update-0 dev loss, the stop at 120% of the estimate, the environment record and
a CPU smoke of every stage on tiny fixtures (a two-layer model; nothing here touches a network
or a real checkpoint).

Runs 1, 1b and 2 are unchanged: their own tests pass untouched, and the tests at the end of
this file pin that the new keys and stops are run 1c's only.

The approved specification pins the frozen training rows (``RUN1C_TRAIN_ROWS``). These tests
bind authored rows, so every test here runs with that pin unset (``unpinned_rows``); the pin
itself and the tracked file are tested in ``test_run1c_train_rows.py``.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from scratchcoder.data.sft import SFTRow, read_rows, write_rows
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.train.validation import assistant_loss, token_weighted_loss
from tools import pilot_supervisor, posttrain_approved, posttrain_ops
from tools.pilot_feasibility import posttrain_estimate_usd, posttrain_report
from tools.posttrain_readiness import APPROVED_CHECKS, MAC_COPIES
from tools.posttrain_run import (
    FORMAT_GATE_RULE,
    FORMAT_GATE_RULE_FINAL,
    FORMAT_GATE_STOP,
    KEPT_FINAL,
    PosttrainWorker,
    Rates,
    RunLimits,
    Specification,
    check_policy,
    environment_record,
    measure_runtime,
    run_extra,
    stage_seconds,
)
from tools.posttrain_smoke import (
    SMOKE_IMAGE,
    SMOKE_POD_ID_1C,
    Fixtures,
    failing_gate_1c,
    fixtures_1c,
    make_worker_1c,
    patched_benchmarks,
    smoke_1c,
)
from tools.rehearsal_binding import RUNS as REHEARSAL_RUNS
from tools.rehearsal_run import MemoryBucket, quarter_updates
from tools.runpod_watchdog import Policy

ROOT = Path(__file__).resolve().parents[1]
SPEC_1C = ROOT / "configs" / "rehearsal-a40-s-post-1c.json"
SPEC_1B = ROOT / "configs" / "rehearsal-a40-s-post-1b.json"


CODE_SOURCES = sorted(posttrain_approved.RUN1C_ROW_SOURCES)


def row(index: int, length: int, *, source: str | None = None) -> SFTRow:
    """An authored SFT row of ``length`` token IDs, the second half assistant targets, its
    source one of the code-only labels run 1c accepts unless ``source`` names another."""
    half = length // 2
    return SFTRow(
        f"fixture/{index}",
        CODE_SOURCES[index % len(CODE_SOURCES)] if source is None else source,
        tuple(300 + (index + offset) % 500 for offset in range(length)),
        (False,) * half + (True,) * (length - half),
    )


def rows_file(path: Path, count: int, *, longer: int = 0, source: str | None = None) -> Path:
    """``count`` authored rows of 40 tokens, then ``longer`` rows of 1,026 (over 1,024 pairs);
    with ``source``, the last row carries that source label."""
    total = count + longer
    rows = [row(index, 40 if index < count else 1_026) for index in range(total)]
    if source is not None:
        rows[-1] = row(total - 1, len(rows[-1].ids), source=source)
    write_rows(path, rows)
    return path


UNPINNED = {key: None for key in posttrain_approved.RUN1C_TRAIN_ROWS}


@pytest.fixture(scope="module", autouse=True)
def unpinned_rows() -> Iterator[None]:
    """The approved specification with the training rows' pin unset, so authored rows bind."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(posttrain_approved, "RUN1C_TRAIN_ROWS", UNPINNED)
        yield


def bind(rows: Path) -> dict[str, Any]:
    """The bind of ``rows`` with their own SHA-256 as the approved one."""
    return posttrain_approved.bind_run1c(rows, expected_sha256=file_sha256(rows))


@pytest.fixture(scope="module")
def bound(unpinned_rows: None, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run 1c's specification bound to 100 authored rows (4 updates of 32)."""
    return bind(rows_file(tmp_path_factory.mktemp("rows") / "train.jsonl", 100))


def fresh(bound: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(bound)


# The approved specification -----------------------------------------------------------------


def test_the_unpinned_specification_has_every_bind_value_as_a_placeholder() -> None:
    raw = posttrain_approved.run1c_specification()
    for field in ("train_rows", "planned_updates"):
        assert raw["sft"][field] is None
    assert raw["inputs"]["sft-train"]["size_bytes"] is None
    assert raw["inputs"]["sft-train"]["sha256"] is None
    assert raw["limits"]["gate_estimate_usd"] is None and raw["limits"]["cost_limit_usd"] is None
    with pytest.raises(ValueError, match="placeholders"):
        posttrain_approved.check_run1c(raw)
    with pytest.raises(ValueError, match="placeholders"):
        Specification.from_dict(raw)


def test_run_1c_is_the_prepared_protocol() -> None:
    raw = posttrain_approved.run1c_specification()
    run1b = posttrain_approved.run1b_specification()
    assert raw["run"] == "1c" and set(raw) == set(run1b) | {"runtime"}
    assert raw["runtime"] == {
        "expected": {
            "python": "3.12.3",
            "torch": "2.9.1+cu128",
            "torch_cuda": "12.8",
            "device_name": "NVIDIA A40",
            "device_count": 1,
        },
        "image": posttrain_approved.RUN1C_IMAGE,
    }
    assert [(stage["name"], stage["kind"]) for stage in raw["stages"]] == [
        ("probe", "probe"),
        ("sft", "sft"),
        ("format-gate", "format-gate"),
        ("eval-sft", "eval"),
    ]
    probe, sft, gate, evaluation = raw["stages"]
    assert probe == {**run1b["stages"][0], "checkpoint": "sft-weights"}
    assert sft == {**run1b["stages"][1], "parent": "sft-weights"}
    assert gate == run1b["stages"][2]
    run1b_files = {entry["name"]: entry for entry in run1b["stages"][4]["files"]}
    assert [entry["name"] for entry in evaluation["files"]] == [
        "pool-dev-greedy",
        "humaneval-greedy",
        "humaneval-plus-greedy",
        "mbpp-plus-greedy",
    ]
    for entry in evaluation["files"]:
        assert entry == run1b_files[entry["name"]] and entry["samples"] == 1
    assert evaluation["checkpoint"] == "sft"
    # Run 1b's optimizer, schedule, batch, sequence and dev settings; one pass; no redraw.
    same = {key: value for key, value in run1b["sft"].items() if not key.startswith("special_")}
    for key in ("train_rows", "passes", "reference_slots_per_second"):
        same.pop(key)
    assert {key: raw["sft"][key] for key in same} == same
    assert raw["sft"]["passes"] == 1
    assert not [key for key in raw["sft"] if key.startswith("special_rows_")]
    assert raw["sft"]["kept_weights"] == KEPT_FINAL and raw["sft"]["parent_dev_loss"] is True
    # Inputs: the parent, the tokenizer, run 1b's dev sets, pool dev and the benchmarks.
    assert set(raw["inputs"]) == {
        "tokenizer",
        "sft-weights",
        "sft-train",
        "sft-code-dev",
        "sft-language-dev",
        "pool-dev",
        "humaneval",
        "humaneval-plus",
        "mbpp-plus",
    }
    assert raw["inputs"]["sft-weights"] == {
        "kind": "checkpoint",
        "key": "posttrain-runs/1b/23wqovsc9gm8h4/sft-weights/checkpoint.pt",
        "size_bytes": 541_164_677,
        "sha256": "f9064cd0e277bf7c2158b0560879494e2b0f3480c89587f45d47c96f0a59c059",
    }
    for name in set(raw["inputs"]) - {"sft-weights", "sft-train"}:
        assert raw["inputs"][name] == run1b["inputs"][name], name
    # Run 1b's limits plus the estimate stop; its screening section (unused); its model.
    assert {k: v for k, v in raw["limits"].items() if k in run1b["limits"]} == run1b["limits"]
    assert set(raw["limits"]) - set(run1b["limits"]) == {"gate_estimate_usd", "cost_limit_usd"}
    for section in ("model", "generation", "screening", "r2"):
        assert raw[section] == run1b[section], section


def test_run_1c_floors_are_the_standing_rule_over_the_four_measured_hosts() -> None:
    hosts = posttrain_approved.MEASURED_HOST_PROBES[: posttrain_approved.RUN1C_FLOOR_HOSTS]
    assert [name for name, _ in hosts] == [
        "run 1 first launch",
        "run 1",
        "run 1b first launch",
        "run 1b",
    ]
    floors = posttrain_approved.derive_generation_floors(hosts)
    assert floors == posttrain_approved.RUN1C_FLOORS == {"128": 7_991.0, "256": 9_049.0}
    sft_hosts = posttrain_approved.MEASURED_SFT_HOSTS[:2]
    assert [name for name, _ in sft_hosts] == ["run 1", "run 1b"]
    reference = sum(rate for _, rate in sft_hosts) / 2
    assert reference == posttrain_approved.RUN1C_SFT_REFERENCE
    assert math.isclose(0.9 * reference, 53_227.25, abs_tol=0.01)
    raw = posttrain_approved.run1c_specification()
    assert raw["probe"]["minimum_tokens_per_second"] == floors
    assert raw["planning"]["generation_tokens_per_second"] == floors
    assert raw["sft"]["reference_slots_per_second"] == reference
    assert raw["sft"]["minimum_speed_fraction"] == 0.9


def test_the_readiness_receipt_the_supervisor_and_the_ops_know_run_1c() -> None:
    assert APPROVED_CHECKS["1c"] is posttrain_approved.check_run1c
    assert MAC_COPIES["1c"] == ("sft-weights",)
    assert posttrain_ops.RUNS == ("1", "1b", "2", "1c")
    assert posttrain_ops.run_prefix("1c", "pod1") == "posttrain-runs/1c/pod1/"
    assert not set(REHEARSAL_RUNS) & set(posttrain_ops.RUNS)
    (run,) = [action for action in pilot_supervisor.build_parser()._actions if action.dest == "run"]
    assert run.choices is not None and "1c" in run.choices


# The bind ------------------------------------------------------------------------------------


def test_the_bind_fills_every_placeholder_from_the_rows(
    bound: dict[str, Any], tmp_path: Path
) -> None:
    assert bound["sft"]["train_rows"] == 100 and bound["sft"]["planned_updates"] == 4
    spec = Specification.from_dict(bound)
    assert spec.sft_updates == 4
    posttrain_approved.check_run1c(bound)
    # The estimate is posttrain_report's 1x row of this very specification.
    path = tmp_path / "bound.json"
    path.write_bytes(posttrain_approved.specification_bytes(bound))
    planned = posttrain_report(path)["scenarios"]["1x generation"]
    estimate = bound["limits"]["gate_estimate_usd"]
    assert estimate == planned["total_usd"] == posttrain_estimate_usd(spec)
    limit = min(1.5, math.floor(round(1.2 * estimate * 100, 6)) / 100)
    assert bound["limits"]["cost_limit_usd"] == limit
    derived = spec.ceiling
    assert isinstance(derived, posttrain_ops.EstimateLimits)
    assert derived.limit_usd == limit and derived.ceiling_usd == 1.5
    assert derived.worker_cost_stop_usd == limit
    assert derived.watchdog_cost_stop_usd == round(limit - 0.01, 2)
    assert derived.wall_seconds == math.floor(limit * 3600 / 0.6 + 1e-6)
    assert derived.self_destruct_seconds == derived.wall_seconds - 180
    assert posttrain_approved.gate_launch_fit(bound)["fits"]


def test_the_bind_refuses_rows_the_worker_would_drop(tmp_path: Path) -> None:
    rows = rows_file(tmp_path / "long.jsonl", 3, longer=1)
    with pytest.raises(ValueError, match="drop"):
        bind(rows)


def test_the_bind_refuses_rows_other_than_the_approved_file(tmp_path: Path) -> None:
    rows = rows_file(tmp_path / "train.jsonl", 8)
    other = rows_file(tmp_path / "other.jsonl", 9)
    with pytest.raises(ValueError, match="not the approved training rows'"):
        posttrain_approved.bind_run1c(rows, expected_sha256=file_sha256(other))
    for malformed in ("", "AB" * 32, "ab" * 31):
        with pytest.raises(ValueError, match="64 lowercase hex digits"):
            posttrain_approved.bind_run1c(rows, expected_sha256=malformed)


@pytest.mark.parametrize("source", ["OpenAssistant/oasst2", "fixture", "TIGER-Lab/AceCode-87K"])
def test_the_bind_refuses_any_row_outside_the_code_only_sources(
    tmp_path: Path, source: str
) -> None:
    rows = rows_file(tmp_path / "mixed.jsonl", 8, source=source)
    with pytest.raises(ValueError, match="code-only rows"):
        bind(rows)


def test_the_code_only_sources_are_the_agent_rows_and_the_code_replay_sources() -> None:
    from tools.sft_data import CODE_SOURCES as REPLAY_SOURCES

    assert posttrain_approved.RUN1C_ROW_SOURCES == {
        "agent-sft1-edit",
        "agent-sft1-repair",
        *REPLAY_SOURCES,
    }


def test_the_bind_refuses_the_file_of_another_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = rows_file(tmp_path / "dev.jsonl", 8)
    approved = posttrain_approved.run1c_specification

    def code_dev_is_these_rows() -> dict[str, Any]:
        raw = approved()
        raw["inputs"]["sft-code-dev"]["sha256"] = file_sha256(rows)
        return raw

    monkeypatch.setattr(posttrain_approved, "run1c_specification", code_dev_is_these_rows)
    with pytest.raises(ValueError, match=r"\['sft-code-dev'\] input, not training rows"):
        bind(rows)


def test_the_bind_command_writes_a_new_launch_specification_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = rows_file(tmp_path / "train.jsonl", 64)
    output = tmp_path / "launch.json"
    argv: list[str] = ["run1c-bind", "--sft-train", str(rows), "--output", str(output)]
    with pytest.raises(SystemExit):  # The approved SHA-256 is required.
        posttrain_approved.main(argv)
    capsys.readouterr()
    wrong = [*argv, "--expected-sha256", "ab" * 32]
    assert posttrain_approved.main(wrong) == 1 and not output.exists()
    assert "not the approved training rows'" in capsys.readouterr().err
    argv += ["--expected-sha256", file_sha256(rows)]
    assert posttrain_approved.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())
    posttrain_approved.check_run1c(written)
    assert printed["sha256"] == file_sha256(output)
    assert printed["train_rows"] == 64 and printed["planned_updates"] == 2
    assert printed["cost_limit_usd"] == written["limits"]["cost_limit_usd"]
    assert printed["sft_train_sha256"] == file_sha256(rows)
    assert posttrain_approved.main(argv) == 1  # A launch specification is never rewritten.
    assert "already exists" in capsys.readouterr().err


# The checker's refusals -----------------------------------------------------------------------


def _pool_test(raw: dict[str, Any]) -> None:
    raw["inputs"]["pool-test"] = dict(posttrain_approved.INPUTS["pool-test"])


def _pool_test_file(raw: dict[str, Any]) -> None:
    entry = next(e for e in posttrain_approved.eval_files(False) if e["name"] == "pool-test-greedy")
    raw["stages"][3]["files"].append(entry)


def _screen(raw: dict[str, Any]) -> None:
    raw["stages"].append(posttrain_approved.run1_stages()[-1])


def _rl_input(raw: dict[str, Any]) -> None:
    raw["inputs"]["rl-candidates-list"] = dict(posttrain_approved.INPUTS["rl-candidates-list"])


def _candidates_kind(raw: dict[str, Any]) -> None:
    raw["inputs"]["extra-list"] = {**posttrain_approved.INPUTS["rl-candidates"]}


def _redraw(raw: dict[str, Any]) -> None:
    raw["sft"].update(
        special_rows_init="ordinary-mean-noise", special_rows_seed=0, special_rows_lr_scale=1.0
    )


def _row_rate(raw: dict[str, Any]) -> None:
    raw["sft"].update(special_rows_init="none", special_rows_seed=0, special_rows_lr_scale=10.0)


def _set(path: tuple[str, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def change(raw: dict[str, Any]) -> None:
        target: Any = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return change


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (_pool_test, "no pool test export"),
        (_pool_test_file, "no pool test completions"),
        (_screen, "no screen stage"),
        (_rl_input, "no RL or screening input"),
        (_candidates_kind, "no RL or screening input"),
        (_redraw, "no re-initialization"),
        (_row_rate, "plain learning rate"),
        (_set(("sft", "passes"), 2), "exactly one pass"),
        (_set(("limits", "ceiling_usd"), 1.51), r"at most \$1.50"),
        (_set(("limits", "ceiling_usd"), 3.0), r"at most \$1.50"),
        (_set(("sft", "train_rows"), None), "placeholders"),
        (_set(("inputs", "sft-train", "sha256"), None), "placeholders"),
        (_set(("limits", "gate_estimate_usd"), None), "placeholders"),
        (_set(("limits", "gate_estimate_usd"), 0.25), "not the projection"),
        (_set(("limits", "cost_limit_usd"), 1.5), "cost_limit_usd"),
        (_set(("sft", "planned_updates"), 5), "planned_updates"),
        (_set(("sft", "kept_weights"), "lowest-dev-loss"), "differs"),
        (_set(("sft", "parent_dev_loss"), False), "differs"),
        (_set(("sft", "learning_rate"), 1e-4), "differs"),
        (_set(("inputs", "sft-weights", "sha256"), "0" * 64), "differs"),
        (
            _set(("inputs", "sft-train", "key"), "posttrain-runs/other/x.jsonl"),
            r"inputs\.sft-train\.key",
        ),
        (_set(("inputs", "sft-train", "kind"), "sft-dev-rows"), r"inputs\.sft-train\.kind"),
        (_set(("probe", "minimum_tokens_per_second"), {"128": 7_731.0, "256": 8_733.0}), "differ"),
        (_set(("planning", "format_gate_seconds"), 1), "differs"),
        (_set(("run",), "1b"), "not a run-1c"),
        (lambda raw: raw["stages"].pop(2), "differs"),
        (_set(("runtime", "expected", "torch"), "2.10.0+cu128"), r"\['runtime'\]"),
        (_set(("runtime", "expected", "python"), "3.12.4"), r"\['runtime'\]"),
        (_set(("runtime", "expected", "torch_cuda"), "12.9"), r"\['runtime'\]"),
        (_set(("runtime", "expected", "device_name"), "NVIDIA L40S"), r"\['runtime'\]"),
        (_set(("runtime", "image", "index_digest"), "sha256:" + "0" * 64), r"\['runtime'\]"),
        (lambda raw: raw.pop("runtime"), "runtime"),
    ],
)
def test_check_run_1c_refuses(
    bound: dict[str, Any], change: Callable[[dict[str, Any]], None], match: str
) -> None:
    raw = fresh(bound)
    change(raw)
    with pytest.raises(ValueError, match=match):
        posttrain_approved.check_run1c(raw)


def test_a_bound_specification_from_other_rows_is_still_checked(bound: dict[str, Any]) -> None:
    raw = fresh(bound)
    # Another file of the same row count: the pin moves, nothing approved does.
    raw["inputs"]["sft-train"].update(size_bytes=123, sha256="ab" * 32)
    posttrain_approved.check_run1c(raw)


# The parser ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda raw: (raw["stages"].pop(2), raw["planning"].pop("format_gate_seconds")),
            "run 1c's SFT weights go through the format gate",
        ),
        (lambda raw: raw["sft"].update(planned_updates=3), "planned_updates"),
        (lambda raw: raw["sft"].update(kept_weights="best"), "kept_weights"),
        (lambda raw: raw["sft"].update(parent_dev_loss=1), "parent_dev_loss"),
        (lambda raw: raw["sft"].pop("planned_updates"), "sft must have exactly"),
        (lambda raw: raw["limits"].pop("cost_limit_usd"), "limits must have exactly"),
        (lambda raw: raw["limits"].update(cost_limit_usd=0.01), "differs from its derivation"),
        (lambda raw: raw.pop("runtime"), "the specification must have exactly"),
        (lambda raw: raw["runtime"].pop("image"), "runtime must have exactly"),
        (lambda raw: raw["runtime"]["expected"].pop("torch"), "runtime.expected must have"),
        (lambda raw: raw["runtime"]["expected"].update(torch=""), "nonempty string"),
        (lambda raw: raw["runtime"]["expected"].update(device_count=0), "device_count"),
        (
            lambda raw: raw["runtime"]["image"].update(reference=IMAGE + "@sha256:" + "a" * 64),
            "the image tag",
        ),
        (lambda raw: raw["runtime"]["image"].update(index_digest="sha256:ab"), "64 hex"),
    ],
)
def test_the_parser_holds_run_1c_to_its_rules(
    bound: dict[str, Any], change: Callable[[dict[str, Any]], Any], match: str
) -> None:
    raw = fresh(bound)
    change(raw)
    with pytest.raises(ValueError, match=match):
        Specification.from_dict(raw)


def test_the_planning_prices_the_update_0_dev_loss_and_one_save(bound: dict[str, Any]) -> None:
    spec = Specification.from_dict(bound)
    rates = Rates.planning(spec)
    quarters = len(quarter_updates(spec.sft_updates)) + 1
    expected = (
        rates.checkpoint_load_seconds
        + rates.compile_seconds
        + spec.sft_updates * spec.slots_per_update / rates.training_slots_per_second
        + (quarters + 1) * int(spec.sft["planned_dev_tokens"]) / rates.forward_tokens_per_second
        + 1 * rates.weights_save_seconds
    )
    assert stage_seconds(spec, spec.stage("sft"), rates) == expected


# The stop at 120% of the estimate -------------------------------------------------------------


def test_the_worker_and_the_watchdog_stop_at_the_estimate_limit(bound: dict[str, Any]) -> None:
    spec = Specification.from_dict(bound)
    derived = spec.ceiling
    limit = bound["limits"]["cost_limit_usd"]
    assert limit < 1.5  # 120% of the estimate binds below the ceiling.
    limits = RunLimits.of(spec)
    assert limits.cost_stop_usd == limit and limits.pod_seconds == derived.wall_seconds
    started = replace(limits, starting_cost_usd=limit - 0.0001)
    assert started.stop_reason(pod_elapsed=1.0) == "cost_limit"
    # The ceiling alone (run 1b's limits) would not have stopped there.
    assert replace(started, cost_stop_usd=1.5).stop_reason(pod_elapsed=1.0) is None
    policy = posttrain_ops.watchdog_policy(derived, origin=1_800_000_000, idle_baseline=1.0)
    assert policy["max_cost_usd"] == round(limit - 0.01, 2)
    check_policy(Policy(**policy), spec)
    # The ceiling's own limits are weaker and refused; so is any other stricter wall.
    ceiling_only = posttrain_ops.ceiling_limits(1.5, 0.6)
    weaker = posttrain_ops.watchdog_policy(ceiling_only, origin=1_800_000_000, idle_baseline=1.0)
    with pytest.raises(ValueError, match="weaker"):
        check_policy(Policy(**weaker), spec)
    stricter = {**policy, "max_wall_seconds": policy["max_wall_seconds"] - 60}
    with pytest.raises(ValueError, match="run 1c's cost limit"):
        check_policy(Policy(**stricter), spec)


# The worker on the smoke fixtures --------------------------------------------------------------


@pytest.fixture(scope="module")
def found(tmp_path_factory: pytest.TempPathFactory) -> Fixtures:
    torch.set_num_threads(1)
    return fixtures_1c(tmp_path_factory.mktemp("posttrain-1c-inputs"))


@pytest.fixture(scope="module")
def smoked(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[dict[str, Any], Path]]:
    output = tmp_path_factory.mktemp("posttrain-1c-smoke") / "smoke"
    yield smoke_1c(output), output


def test_the_smoke_runs_every_stage_and_passes_the_real_gate(
    smoked: tuple[dict[str, Any], Path],
) -> None:
    summary, output = smoked
    assert summary["status"] == "completed" and summary["objects_outside_prefix"] == []
    assert summary["stages"] == dict.fromkeys(
        ("probe", "sft", "format-gate", "eval-sft"), "completed"
    )
    gate = summary["format_gate"]
    assert gate["passed"] is True and gate["checkpoint_sha256"] == summary["sft_selected"]["sha256"]
    assert gate["rule"] == FORMAT_GATE_RULE_FINAL != FORMAT_GATE_RULE
    report = json.loads((output / "run" / "report.json").read_text())
    assert report["r2"]["prefix"] == f"posttrain-runs/1c/{SMOKE_POD_ID_1C}/"
    labels = {row["label"] for row in report["artifacts"]}
    assert {"environment", "probe", "sft", "sft-weights", "format-gate"} <= labels
    assert {label for label in labels if label.startswith("eval-sft-")} == {
        "eval-sft-pool-dev-greedy",
        "eval-sft-humaneval-greedy",
        "eval-sft-humaneval-plus-greedy",
        "eval-sft-mbpp-plus-greedy",
    }
    # The limits are the 120% stop's: the lower of $1.50 and 120% of the smoke's estimate.
    limits = summary["limits"]
    expected = posttrain_ops.estimate_limit_usd(1.5, limits["gate_estimate_usd"])
    assert limits["limit_usd"] == limits["worker_cost_stop_usd"] == expected


def test_the_smoke_keeps_the_final_weights_and_measures_the_parent_first(
    smoked: tuple[dict[str, Any], Path], found: Fixtures
) -> None:
    summary, output = smoked
    report = json.loads((output / "run" / "report.json").read_text())
    sft = report["sft"]
    updates = sft["planned_updates"]
    assert sft["kept_weights"] == KEPT_FINAL
    assert summary["sft_quarters"] == [update for _, update in quarter_updates(updates)] + [updates]
    assert sft["selected"]["update"] == updates and sft["selected"]["quarter"] == 4
    assert [entry.get("saved", False) for entry in sft["quarters"]] == [False, False, False, True]
    assert not list((output / "run" / "artifacts" / "sft").glob("quarter-[123].pt"))
    parent = sft["parent_dev_loss"]
    assert parent["update"] == 0 and set(parent["by_set"]) == {"sft-code-dev", "sft-language-dev"}
    final = sft["quarters"][-1]
    change = sft["dev_loss_change"]
    assert change["dev_loss"] == final["dev_loss"] - parent["dev_loss"]
    for name, value in change["by_set"].items():
        assert value == final["by_set"][name]["loss"] - parent["by_set"][name]["loss"]
    assert sft["special_rows"] is None and sft["special_rows_lr_scale"] == 1.0
    assert sft["parent_sha256"] == file_sha256(found.inputs["sft-weights"])


def test_the_update_0_dev_loss_is_the_parents_own(
    smoked: tuple[dict[str, Any], Path], found: Fixtures
) -> None:
    """A reference: the parent loaded as evaluation loads it, the same dev rows, the same loss."""
    _, output = smoked
    parent = json.loads((output / "run" / "report.json").read_text())["sft"]["parent_dev_loss"]
    model, _ = load_checkpoint(found.inputs["sft-weights"], found.inputs["tokenizer"])
    results = {
        name: assistant_loss(model, list(read_rows(found.inputs[name])), batch_size=4)
        for name in ("sft-code-dev", "sft-language-dev")
    }
    assert math.isclose(parent["dev_loss"], token_weighted_loss(results.values()), rel_tol=1e-6)


def test_the_parent_is_a_weights_only_file_and_the_environment_is_uploaded_first(
    smoked: tuple[dict[str, Any], Path], found: Fixtures
) -> None:
    state = torch.load(found.inputs["sft-weights"], map_location="cpu", weights_only=True)
    assert {"format", "step", "metadata", "config", "tokenizer_hash", "model", "run_id"} == set(
        state
    )
    _, output = smoked
    report = json.loads((output / "run" / "report.json").read_text())
    environment = report["environment"]
    assert environment["image"]["reference"] == "runpod/pytorch:smoke-tag"
    assert environment["image"]["digest"] is None
    first = report["artifacts"][0]
    assert (first["label"], first["name"]) == ("environment", "environment.json")
    # The probe's label record, uploaded before SFT starts, carries the environment too.
    probe = json.loads((output / "run" / "exports" / "probe" / "record.json").read_text())
    assert probe["environment"]["packages_sha256"] == environment["packages_sha256"]


def _scripted(losses: list[float]) -> Callable[..., dict[str, Any]]:
    calls = iter(losses)

    def fake(self: PosttrainWorker, stage: Any, trainer: Any, dev: Any) -> dict[str, Any]:
        loss = next(calls)
        sets = {name: {"loss": loss, "rows": 1.0, "tokens": 1.0} for name in dev}
        return {"update": trainer.step, "dev_loss": loss, "by_set": sets, "seconds": 0.0}

    return fake


@pytest.mark.parametrize(("kept", "quarter"), [("final", 4), ("lowest-dev-loss", 1)])
def test_final_keeps_the_last_update_even_when_an_earlier_quarter_was_lower(
    found: Fixtures, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kept: str, quarter: int
) -> None:
    monkeypatch.setattr(PosttrainWorker, "sft_dev_loss", _scripted([5.0, 1.0, 2.0, 3.0, 4.0]))
    worker = make_worker_1c(
        found,
        tmp_path / "run",
        bucket=MemoryBucket(),
        changes=lambda raw: raw["sft"].update(kept_weights=kept),
    )
    with patched_benchmarks():
        report = worker.execute()
    selected = report["sft"]["selected"]
    assert selected["quarter"] == quarter and selected["dev_loss"] == [4.0, 1.0][quarter == 1]
    assert report["sft"]["parent_dev_loss"]["dev_loss"] == 5.0
    assert [entry["dev_loss"] for entry in report["sft"]["quarters"]] == [1.0, 2.0, 3.0, 4.0]


def test_a_failing_gate_stops_run_1c_before_its_evaluation(tmp_path: Path) -> None:
    summary = smoke_1c(tmp_path / "smoke", changes=failing_gate_1c)
    assert summary["status"] == "stopped" and summary["stop_reason"] == FORMAT_GATE_STOP
    assert summary["stages"] == {
        "probe": "completed",
        "sft": "completed",
        "format-gate": "stopped",
        "eval-sft": "skipped",
    }
    assert summary["objects_outside_prefix"] == []


def test_the_worker_stops_at_the_estimate_limit_before_any_stage(
    found: Fixtures, tmp_path: Path
) -> None:
    bucket = MemoryBucket()
    probe = make_worker_1c(found, tmp_path / "probe", bucket=MemoryBucket())
    derived = probe.plan.ceiling
    assert isinstance(derived, posttrain_ops.EstimateLimits)
    limits = replace(RunLimits.of(probe.plan), starting_cost_usd=derived.limit_usd)
    worker = make_worker_1c(found, tmp_path / "run", bucket=bucket, limits=limits)
    with patched_benchmarks():
        report = worker.execute()
    assert report["status"] == "stopped" and report["stop_reason"] == "cost_limit"
    assert [stage["status"] for stage in report["stages"]] == [
        "stopped",
        "skipped",
        "skipped",
        "skipped",
    ]


# The environment record ------------------------------------------------------------------------


def test_the_environment_record_lists_every_package_and_a_pinned_digest() -> None:
    tagged = environment_record(image="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404")
    assert tagged["image"]["digest"] is None
    assert any(name.startswith("torch==") for name in tagged["packages"])
    assert tagged["packages"] == sorted(set(tagged["packages"]))
    assert tagged["torch"]["version"] == torch.__version__
    pinned = environment_record(image="runpod/pytorch@sha256:" + "ab" * 32)
    assert pinned["image"]["digest"] == "sha256:" + "ab" * 32
    assert pinned["packages_sha256"] == tagged["packages_sha256"]
    assert environment_record(image=None)["image"]["reference"] is None


def test_the_pod_run_adds_the_environment_of_the_receipts_image_for_run_1c(
    bound: dict[str, Any],
) -> None:
    spec = Specification.from_dict(bound)
    base = {"code_revision": {"commit": "1" * 40}, "launch_fit": {"fits": True}}
    extra = run_extra(spec, {"pod_image": IMAGE}, base)
    added = {"environment", "runtime"}
    assert {key: value for key, value in extra.items() if key not in added} == base
    assert extra["runtime"] == measure_runtime()  # Measured here; checked by the worker.
    assert extra["environment"]["kind"] == "posttrain-environment"
    assert extra["environment"]["image"]["reference"] == IMAGE
    assert "environment" not in base  # The caller's fields are copied, never changed.


@pytest.mark.parametrize("receipt", [{}, {"pod_image": ""}, {"pod_image": None}, {"pod_image": 7}])
def test_the_pod_run_refuses_a_run_1c_receipt_without_the_image(
    bound: dict[str, Any], receipt: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="names no Pod image"):
        run_extra(Specification.from_dict(bound), receipt, {})


def test_the_pod_runs_extra_is_unchanged_for_runs_1_1b_and_2() -> None:
    run2 = json.loads((ROOT / "configs" / "rehearsal-a40-s-post-2-v2.json").read_text())
    specs = [
        Specification.load(ROOT / "configs" / "rehearsal-a40-s-post-1.json"),
        Specification.load(SPEC_1B),
        Specification.from_dict(run2, allow_gate_nulls=True),
    ]
    base = {"code_revision": {"commit": "1" * 40}}
    for spec in specs:
        for receipt in ({}, {"pod_image": IMAGE}):
            assert run_extra(spec, receipt, base) == base, spec.run


def test_the_smoke_worker_builds_its_extra_as_the_pod_run_does(
    found: Fixtures, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import posttrain_run

    calls: list[tuple[str, dict[str, Any]]] = []

    def spy(spec: Specification, receipt: Any, base: Any, **options: Any) -> dict[str, Any]:
        calls.append((spec.run, dict(receipt)))
        return run_extra(spec, receipt, base, **options)

    monkeypatch.setattr(posttrain_run, "run_extra", spy)
    worker = make_worker_1c(found, tmp_path / "run", bucket=MemoryBucket())
    assert calls == [("1c", {"pod_image": SMOKE_IMAGE})]
    assert worker.report["environment"]["image"]["reference"] == SMOKE_IMAGE


# Runs 1, 1b and 2 are unchanged -----------------------------------------------------------------


def test_the_new_sft_keys_and_the_estimate_stop_are_run_1cs_only() -> None:
    run1b = posttrain_approved.run1b_specification()
    assert posttrain_ops.limit_keys(run1b) == posttrain_ops.LIMIT_KEYS
    assert posttrain_ops.limit_keys({"run": "1"}) == posttrain_ops.LIMIT_KEYS
    assert posttrain_ops.limit_keys({"run": "2"}) == posttrain_ops.RUN2_LIMIT_KEYS
    for key, value in (("kept_weights", "final"), ("parent_dev_loss", True)):
        raw = copy.deepcopy(run1b)
        raw["sft"][key] = value
        with pytest.raises(ValueError, match="sft must have exactly"):
            Specification.from_dict(raw)
    spec = Specification.load(SPEC_1B)
    assert not isinstance(spec.ceiling, posttrain_ops.EstimateLimits)
    rates = Rates.planning(spec)
    quarters = len(quarter_updates(spec.sft_updates)) + 1
    assert stage_seconds(spec, spec.stage("sft"), rates) == (
        rates.checkpoint_load_seconds
        + rates.compile_seconds
        + spec.sft_updates * spec.slots_per_update / rates.training_slots_per_second
        + quarters * int(spec.sft["planned_dev_tokens"]) / rates.forward_tokens_per_second
        + quarters * rates.weights_save_seconds
    )
    estimate = posttrain_estimate_usd(spec)
    assert estimate == posttrain_report(SPEC_1B)["scenarios"]["1x generation"]["total_usd"]


# The readiness receipt ---------------------------------------------------------------------------

IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
FILES_1C = {
    "sft-train": "train.jsonl",
    "sft-code-dev": "code_dev.jsonl",
    "sft-language-dev": "language_dev.jsonl",
    "pool-dev": "dev-export.jsonl",
    "humaneval": "HumanEval.jsonl.gz",
    "humaneval-plus": "HumanEvalPlus-OriginFmt.jsonl.gz",
    "mbpp-plus": "MbppPlus-OriginFmt.jsonl.gz",
}


def receipt_evidence(
    tmp_path: Path, bound: dict[str, Any], **changes: Callable[[Any], None]
) -> dict[str, Any]:
    """Every file the receipt tool reads for run 1c, authored as the Pod and runbook write them."""
    from test_posttrain_readiness import NOW, POD, REVISION, fetch_rows

    from scratchcoder.common.records import atomic_json
    from tools.posttrain_binding import _serialized, binding_sha256, staged_key
    from tools.posttrain_readiness import DEFAULT_HEARTBEAT

    created = NOW - 300
    spec_path = tmp_path / "rehearsal-a40-s-post-1c-launch.json"
    spec_path.write_bytes(posttrain_approved.specification_bytes(bound))
    inputs: dict[str, Any] = {}
    for name, pin in bound["inputs"].items():
        if pin["key"] is None:
            file = FILES_1C[name]
            entry = {"key": staged_key(pin["sha256"], file), "path": f"{name}/{file}"}
            entry["origin"] = "staged"
        else:
            file = pin["key"].rsplit("/", 1)[-1]
            entry = {"key": pin["key"], "path": f"{name}/{file}", "origin": "existing"}
        inputs[name] = {**entry, "kind": pin["kind"], "size_bytes": pin["size_bytes"]}
        inputs[name].update(sha256=pin["sha256"], source=None)
    binding: dict[str, Any] = {
        "schema_version": 1,
        "kind": "posttrain-input-binding",
        "run": "1c",
        "created_at": "2026-11-16T00:00:00Z",
        "binder": {"git_commit": "1" * 40, "dirty": False},
        "specification": {"path": str(spec_path), "sha256": "a" * 64},
        "inputs": inputs,
    }
    derived = Specification.from_dict(bound).ceiling
    origin = int(created) - 20
    prefix = posttrain_ops.run_prefix("1c", POD)
    values: dict[str, Any] = {
        "binding.json": _serialized(binding),
        "identity": {
            "id": POD,
            "name": "posttrain-a40-s-1c",
            "gpuId": "NVIDIA A40",
            "gpuCount": 1,
            "dataCenterId": "test",
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(created)),
        },
        "policy": posttrain_ops.watchdog_policy(derived, origin=origin, idle_baseline=NOW),
        "self-destruct.json": {
            "pod_id": POD,
            "pid": 4242,
            "state": "armed",
            "armed_at_unix": created + 60,
            "checked_at_unix": NOW - 2,
            "deadline_unix": derived.deadline_unix(origin),
            "dry_run": False,
            "idle_seconds": 900.0,
            "activity_files": [DEFAULT_HEARTBEAT, "/workspace/posttrain/setup-progress.json"],
        },
        "confirmations": {
            **dict.fromkeys(posttrain_ops.OWNER_GATES, True),
            "termination_guarantee": "redundant_best_effort",
            "owner_accepts_best_effort_termination": True,
        },
        "preflight": {
            "status": "completed",
            "hardware_precision_and_resume_checked": True,
            "device": "cuda:0",
            "hardware": {
                "device": "cuda:0",
                "device_type": "cuda",
                "name": "NVIDIA A40",
                "gpu_count": 1,
                "torch_version": "2.9.1+cu128",
                "torch_cuda_version": "12.8",
            },
            "fused_attention": {"status": "passed"},
            "compile_equivalence": {"status": "passed"},
        },
        "data-fetch": fetch_rows(binding, sorted(binding["inputs"])),
        "data-verification": {
            "binding_sha256": binding_sha256(binding),
            "run": "1c",
            "inputs": {name: entry["sha256"] for name, entry in binding["inputs"].items()},
        },
        "mac-copy": fetch_rows(binding, ["sft-weights"]),
        "round-trip": {
            "kind": "r2-round-trip",
            "status": "passed",
            "run": "1c",
            "pod_id": POD,
            "prefix": prefix,
            "key": prefix + "preflight/round-trip-x.bin",
            "bytes": posttrain_ops.ROUND_TRIP_PROBE_BYTES,
            "steps": {
                "put_seconds": 20.0,
                "head_seconds": 0.1,
                "get_seconds": 10.0,
                "delete_seconds": 0.1,
            },
            "upload_bytes_per_second": posttrain_ops.ROUND_TRIP_PROBE_BYTES / 20.0,
            "download_bytes_per_second": posttrain_ops.ROUND_TRIP_PROBE_BYTES / 10.0,
        },
        "pod-connection.json": {"pod_id": POD, "cost_usd_per_hour": 0.59, "image": IMAGE},
        "revision.json": REVISION,
        "launch-check.json": {
            "kind": "posttrain-launch-check",
            "launch_commit": REVISION["git_commit"],
            "revision_commit": REVISION["git_commit"],
            "archive_commit": REVISION["git_commit"],
            "archive_sha256": REVISION["archive_sha256"],
            "files": 3,
            "tree_clean": True,
        },
        "runtime-check.json": {  # The bootstrap's record; read next to the identity.
            "kind": "posttrain-runtime-check",
            "expected": dict(posttrain_approved.RUN1C_RUNTIME),
            "measured": {**posttrain_approved.RUN1C_RUNTIME, "cuda_available": True},
            "mismatched": [],
            "status": "passed",
        },
    }
    for name, change in changes.items():
        change(values[name])
    files = {}
    for name, value in values.items():
        path = tmp_path / name
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            atomic_json(path, value)
        files[name] = path
    return {
        "specification": spec_path,
        "binding": files["binding.json"],
        "binding_sha256": binding_sha256(binding),
        "preflight": files["preflight"],
        "identity": files["identity"],
        "policy": files["policy"],
        "self_destruct_receipt": files["self-destruct.json"],
        "confirmations": files["confirmations"],
        "data_fetch": files["data-fetch"],
        "data_verification": files["data-verification"],
        "round_trip": files["round-trip"],
        "mac_copy": files["mac-copy"],
        "revision": files["revision.json"],
        "connection": files["pod-connection.json"],
        "billing_origin_unix": float(origin),
        "launch_commit": REVISION["git_commit"],
        "launch_check": files["launch-check.json"],
    }


def test_the_run_1c_receipt_is_what_the_worker_accepts_and_names_the_image(
    tmp_path: Path, bound: dict[str, Any]
) -> None:
    from test_posttrain_readiness import NOW, POD

    from tools.posttrain_readiness import DEFAULT_HEARTBEAT, build_receipt
    from tools.posttrain_run import check_readiness

    arguments = receipt_evidence(tmp_path, bound)
    receipt = build_receipt(**arguments, now=NOW)
    assert receipt["run"] == "1c" and receipt["r2_run_prefix"] == f"posttrain-runs/1c/{POD}/"
    assert receipt["pod_image"] == IMAGE
    assert receipt["limits"]["limit_usd"] == bound["limits"]["cost_limit_usd"]
    assert receipt["limits"]["ceiling_usd"] == 1.5 and receipt["time_fit"]["fits_at_planning"]
    check_readiness(
        json.loads(json.dumps(receipt)),
        now=NOW,
        spec=Specification.load(arguments["specification"]),
        specification_sha256=receipt["specification_sha256"],
        binding_sha256=arguments["binding_sha256"],
        preflight_sha256=receipt["preflight_sha256"],
        heartbeat=DEFAULT_HEARTBEAT,
    )


@pytest.mark.parametrize(
    ("name", "change", "match"),
    [
        ("pod-connection.json", lambda value: value.pop("image"), "names no container image"),
        (
            "policy",
            lambda value: value.update(max_wall_seconds=9_000, max_cost_usd=1.49),
            "weaker",
        ),
        ("mac-copy", lambda value: value.update(inputs=[]), r"cover exactly \['sft-weights'\]"),
    ],
)
def test_the_run_1c_receipt_refuses(
    tmp_path: Path,
    bound: dict[str, Any],
    name: str,
    change: Callable[[Any], None],
    match: str,
) -> None:
    from test_posttrain_readiness import NOW

    from tools.posttrain_readiness import build_receipt

    with pytest.raises(ValueError, match=f"(?i){match}"):
        build_receipt(**receipt_evidence(tmp_path, bound, **{name: change}), now=NOW)


def test_the_receipt_refuses_the_tracked_template_with_its_placeholders(
    tmp_path: Path, bound: dict[str, Any]
) -> None:
    from test_posttrain_readiness import NOW

    from tools.posttrain_readiness import build_receipt

    arguments = receipt_evidence(tmp_path, bound)
    template = tmp_path / "template.json"
    template.write_bytes(SPEC_1C.read_bytes())
    with pytest.raises(ValueError, match="placeholders"):
        build_receipt(**{**arguments, "specification": template}, now=NOW)


@pytest.mark.parametrize(
    "image",
    [
        IMAGE,
        IMAGE + "@" + posttrain_approved.RUN1C_IMAGE["index_digest"],
        IMAGE + "@" + posttrain_approved.RUN1C_IMAGE["linux_amd64_digest"],
    ],
)
def test_the_receipt_accepts_the_tag_or_the_tag_pinned_to_a_recorded_digest(
    tmp_path: Path, bound: dict[str, Any], image: str
) -> None:
    from test_posttrain_readiness import NOW

    from tools.posttrain_readiness import build_receipt

    connection = {"pod-connection.json": lambda value: value.update(image=image)}
    receipt = build_receipt(**receipt_evidence(tmp_path, bound, **connection), now=NOW)
    assert receipt["pod_image"] == image


# The launch commit -------------------------------------------------------------------------------


def _receipt_with(
    tmp_path: Path, bound: dict[str, Any], *, arguments: dict[str, Any], **changes: Any
) -> dict[str, Any]:
    from test_posttrain_readiness import NOW

    from tools.posttrain_readiness import build_receipt

    evidence = {**receipt_evidence(tmp_path, bound, **changes), **arguments}
    return build_receipt(**evidence, now=NOW)


def test_the_receipt_records_the_launch_commit_and_the_pods_launch_check(
    tmp_path: Path, bound: dict[str, Any]
) -> None:
    from test_posttrain_readiness import REVISION

    receipt = _receipt_with(tmp_path, bound, arguments={})
    assert receipt["launch_commit"] == REVISION["git_commit"] == receipt["revision"]["git_commit"]
    assert receipt["launch_check"]["tree_clean"] is True
    assert receipt["evidence_sha256"]["launch_check"] == file_sha256(tmp_path / "launch-check.json")


@pytest.mark.parametrize(
    ("arguments", "changes", "match"),
    [
        ({"launch_commit": None}, {}, "needs --launch-commit"),
        ({"launch_check": None}, {}, "needs --launch-commit"),
        ({"launch_commit": "ab" * 6}, {}, "40 lower-case hex digits"),
        ({"launch_commit": "AB" * 20}, {}, "40 lower-case hex digits"),
        ({"launch_commit": "cd" * 20}, {}, "not run 1c's launch commit"),
        (
            {},
            {"launch-check.json": lambda value: value.update(tree_clean=False)},
            "not exactly the launch commit's tree",
        ),
        (
            {},
            {"launch-check.json": lambda value: value.update(archive_commit="cd" * 20)},
            "archive_commit is not the launch commit",
        ),
        (
            {},
            {"launch-check.json": lambda value: value.update(archive_sha256="0" * 64)},
            "another archive",
        ),
        ({}, {"launch-check.json": lambda value: value.pop("files")}, "exactly the keys"),
        (
            {},
            {"pod-connection.json": lambda value: value.update(image="runpod/pytorch:other")},
            "not run 1c's image",
        ),
    ],
)
def test_the_receipt_refuses_another_launch_commit_or_tree(
    tmp_path: Path,
    bound: dict[str, Any],
    arguments: dict[str, Any],
    changes: dict[str, Callable[[Any], None]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _receipt_with(tmp_path, bound, arguments=arguments, **changes)


def test_the_worker_refuses_a_receipt_without_the_launch_commit(
    tmp_path: Path, bound: dict[str, Any]
) -> None:
    from test_posttrain_readiness import NOW

    from tools.posttrain_readiness import DEFAULT_HEARTBEAT
    from tools.posttrain_run import check_readiness

    receipt = _receipt_with(tmp_path, bound, arguments={})
    spec = Specification.from_dict(bound)
    identity = {
        "specification_sha256": receipt["specification_sha256"],
        "binding_sha256": receipt["binding_sha256"],
        "preflight_sha256": receipt["preflight_sha256"],
    }
    changes: list[Callable[[dict[str, Any]], Any]] = [
        lambda value: value.pop("launch_commit"),
        lambda value: value.update(launch_commit=value["launch_commit"][:12]),
        lambda value: value["launch_check"].update(launch_commit="cd" * 20),
        lambda value: value["revision"].update(git_commit="cd" * 20),
    ]
    for change in changes:
        edited = json.loads(json.dumps(receipt))
        change(edited)
        with pytest.raises(ValueError, match="launch commit|launch_commit"):
            check_readiness(edited, now=NOW, spec=spec, heartbeat=DEFAULT_HEARTBEAT, **identity)


def test_the_other_runs_receipts_take_no_launch_commit(tmp_path: Path) -> None:
    from test_posttrain_readiness import NOW, evidence

    from tools.posttrain_readiness import build_receipt

    with pytest.raises(ValueError, match="run 1c's"):
        build_receipt(**evidence(tmp_path), launch_commit="ab" * 20, now=NOW)
    with pytest.raises(ValueError, match="run 1c's"):
        build_receipt(**evidence(tmp_path), runtime_check=tmp_path / "runtime.json", now=NOW)


# The runtime before launch (the bootstrap's record and the preflight's, at the receipt) ----------


def test_the_receipt_records_the_runtime_the_pod_measured_before_launch(
    tmp_path: Path, bound: dict[str, Any]
) -> None:
    receipt = _receipt_with(tmp_path, bound, arguments={})
    record = receipt["runtime_preflight"]
    assert record["status"] == "passed" and record["mismatched"] == []
    assert record["expected"] == posttrain_approved.RUN1C_RUNTIME
    assert record["bootstrap"]["torch"] == record["preflight"]["torch"] == "2.9.1+cu128"
    assert receipt["evidence_sha256"]["runtime_check"] == file_sha256(
        tmp_path / "runtime-check.json"
    )
    # An explicit path is read in place of the one next to the identity.
    elsewhere = tmp_path / "records" / "runtime.json"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes((tmp_path / "runtime-check.json").read_bytes())
    (tmp_path / "runtime-check.json").unlink()
    receipt = _receipt_with(tmp_path, bound, arguments={"runtime_check": elsewhere})
    assert receipt["evidence_sha256"]["runtime_check"] == file_sha256(elsewhere)


def _measured(**change: Any) -> Callable[[dict[str, Any]], None]:
    return lambda value: value["measured"].update(change)


def _hardware(**change: Any) -> Callable[[dict[str, Any]], None]:
    return lambda value: value["hardware"].update(change)


@pytest.mark.parametrize(
    ("name", "change", "field"),
    [
        ("runtime-check.json", _measured(torch="2.9.0+cu128"), "bootstrap.torch"),
        ("runtime-check.json", _measured(python="3.12.4"), "bootstrap.python"),
        ("runtime-check.json", _measured(torch_cuda="12.6"), "bootstrap.torch_cuda"),
        ("runtime-check.json", _measured(device_count=2), "bootstrap.device_count"),
        ("runtime-check.json", _measured(cuda_available=False), "bootstrap.cuda_available"),
        ("runtime-check.json", lambda value: value.pop("measured"), "bootstrap.torch"),
        ("preflight", _hardware(torch_version="2.9.0+cu128"), "preflight.torch"),
        ("preflight", _hardware(torch_cuda_version="12.6"), "preflight.torch_cuda"),
        ("preflight", _hardware(name="NVIDIA RTX A6000"), "preflight.device_name"),
        ("preflight", _hardware(gpu_count=2), "preflight.device_count"),
        ("preflight", _hardware(device_type="cpu"), "preflight.cuda_available"),
    ],
)
def test_the_receipt_refuses_a_runtime_other_than_the_specifications(
    tmp_path: Path,
    bound: dict[str, Any],
    name: str,
    change: Callable[[dict[str, Any]], None],
    field: str,
) -> None:
    # The recorded status is not trusted: the receipt recomputes the comparison.
    with pytest.raises(ValueError, match=f"runtime is not run 1c's.*'{field}'"):
        _receipt_with(tmp_path, bound, arguments={}, **{name: change})


def test_the_receipt_refuses_a_missing_or_foreign_runtime_record(
    tmp_path: Path, bound: dict[str, Any]
) -> None:
    foreign = {"runtime-check.json": lambda value: value.update(kind="posttrain-launch-check")}
    with pytest.raises(ValueError, match="not the bootstrap's runtime check"):
        _receipt_with(tmp_path, bound, arguments={}, **foreign)
    with pytest.raises(OSError):
        _receipt_with(tmp_path, bound, arguments={"runtime_check": tmp_path / "absent.json"})


def test_the_worker_refuses_a_receipt_without_a_matching_runtime(
    tmp_path: Path, bound: dict[str, Any]
) -> None:
    from test_posttrain_readiness import NOW

    from tools.posttrain_readiness import DEFAULT_HEARTBEAT
    from tools.posttrain_run import check_readiness

    receipt = _receipt_with(tmp_path, bound, arguments={})
    spec = Specification.from_dict(bound)
    identity = {
        "specification_sha256": receipt["specification_sha256"],
        "binding_sha256": receipt["binding_sha256"],
        "preflight_sha256": receipt["preflight_sha256"],
    }
    check_readiness(
        json.loads(json.dumps(receipt)), now=NOW, spec=spec, heartbeat=DEFAULT_HEARTBEAT, **identity
    )
    changes: list[tuple[Callable[[dict[str, Any]], Any], str]] = [
        (lambda value: value.pop("runtime_preflight"), "no runtime measured before launch"),
        (
            lambda value: value["runtime_preflight"]["bootstrap"].update(torch="2.9.0+cu128"),
            "bootstrap.torch",
        ),
        (
            lambda value: value["runtime_preflight"]["preflight"].update(device_name="other"),
            "preflight.device_name",
        ),
        (
            lambda value: value["runtime_preflight"]["expected"].update(torch="2.9.0+cu128"),
            "expected another runtime",
        ),
        (lambda value: value["runtime_preflight"].update(status="failed"), "not run 1c's"),
    ]
    for change, match in changes:
        edited = json.loads(json.dumps(receipt))
        change(edited)
        with pytest.raises(ValueError, match=match):
            check_readiness(edited, now=NOW, spec=spec, heartbeat=DEFAULT_HEARTBEAT, **identity)


# The runtime check ----------------------------------------------------------------------------


def test_a_matching_runtime_passes_and_is_recorded_before_the_probe(
    smoked: tuple[dict[str, Any], Path],
) -> None:
    summary, output = smoked
    assert summary["runtime_check"] == "passed"
    report = json.loads((output / "run" / "report.json").read_text())
    check = report["runtime_check"]
    assert check["expected"] == posttrain_approved.RUN1C_RUNTIME
    assert check["mismatched"] == [] and report["runtime"]["cuda_available"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python", "3.12.4"),
        ("torch", "2.9.0+cu128"),
        ("torch_cuda", "12.6"),
        ("device_name", "NVIDIA RTX A6000"),
        ("device_count", 2),
        ("cuda_available", False),
    ],
)
def test_each_runtime_mismatch_stops_the_run_before_the_probe(
    found: Fixtures, tmp_path: Path, field: str, value: Any
) -> None:
    from tools.posttrain_run import RUNTIME_STOP
    from tools.posttrain_smoke import smoke_runtime_1c

    bucket = MemoryBucket()
    worker = make_worker_1c(
        found,
        tmp_path / "run",
        bucket=bucket,
        runtime=lambda: {**smoke_runtime_1c(), field: value},
    )
    with patched_benchmarks():
        report = worker.execute()
    assert report["status"] == "stopped" and report["stop_reason"] == RUNTIME_STOP
    assert [stage["status"] for stage in report["stages"]] == ["skipped"] * 4
    assert report["probe"] is None and report["generated_tokens"] == 0
    assert report["runtime_check"]["status"] == "failed"
    assert report["runtime_check"]["mismatched"] == [field]
    labels = {row["label"] for row in report["artifacts"]}
    assert labels == {"environment"}  # The environment record, then nothing paid for.
    prefix = report["r2"]["prefix"]
    assert prefix + "report.json" in bucket.objects  # Finalized like any other early stop.


def test_this_hosts_own_runtime_is_refused_without_cuda() -> None:
    from tools.posttrain_run import runtime_mismatches

    measured = measure_runtime()
    differ = runtime_mismatches(posttrain_approved.RUN1C_RUNTIME, measured)
    if not torch.cuda.is_available():
        assert "cuda_available" in differ and "device_name" in differ
    assert runtime_mismatches(posttrain_approved.RUN1C_RUNTIME, None)[0] == "cuda_available"


def test_runs_1_1b_and_2_pin_no_runtime_and_never_check_one() -> None:
    for path in ("rehearsal-a40-s-post-1.json", "rehearsal-a40-s-post-1b.json"):
        raw = json.loads((ROOT / "configs" / path).read_text())
        assert "runtime" not in raw
        raw["runtime"] = posttrain_approved.run1c_specification()["runtime"]
        with pytest.raises(ValueError, match="the specification must have exactly"):
            Specification.from_dict(raw)
    run2 = json.loads((ROOT / "configs" / "rehearsal-a40-s-post-2-v2.json").read_text())
    assert "runtime" not in run2


# The SFT speed check ----------------------------------------------------------------------------


@pytest.mark.parametrize(("updates", "fires_at"), [(120, 50), (20, 20)])
def test_the_speed_check_fires_at_update_50_or_at_the_last_of_fewer(
    found: Fixtures, tmp_path: Path, updates: int, fires_at: int
) -> None:
    """The rule is min(50, updates): 50 or more updates check at update 50 over updates 3 to 50
    (the two warmup updates excluded); fewer than 50 check at the last update."""
    from scratchcoder.train.pilot import PilotStop

    original = list(read_rows(found.inputs["sft-train"]))
    per_update = 4  # The smoke's 2 rows x 2 accumulation steps.
    rows = tmp_path / "rows" / "train.jsonl"
    write_rows(
        rows,
        [
            SFTRow(f"speed/{index}", original[index % len(original)].source, row.ids, row.mask)
            for index in range(updates * per_update)
            for row in [original[index % len(original)]]
        ],
    )
    fixtures = Fixtures(
        found.root, found.tokenizer, {**found.inputs, "sft-train": rows}, found.full_tasks
    )
    window = {"speed_check_updates": 50, "speed_check_warmup_updates": 2}
    worker = make_worker_1c(
        fixtures,
        tmp_path / "run",
        bucket=MemoryBucket(),
        changes=lambda raw: raw["sft"].update(window),
    )
    assert worker.plan.sft_updates == updates
    seen: list[dict[str, Any]] = []
    check = worker.sft_speed_check

    def observed(record: dict[str, Any], timing: Any, *, warmup: int) -> None:
        try:
            check(record, timing, warmup=warmup)
        finally:
            seen.append({"at": timing[-1][0], "timed": len(timing), **record["speed_check"]})
            raise PilotStop("observed")

    worker.sft_speed_check = observed  # type: ignore[method-assign]
    with pytest.raises(PilotStop, match="observed"):
        worker.run_sft(worker.plan.stage("sft"))
    (result,) = seen
    assert result["at"] == result["timed"] == fires_at
    assert result["after_updates"] == 50
    assert result["excluded_warmup_updates"] == 2 and result["counted_updates"] == fires_at - 2


def test_the_receipt_command_reads_the_launch_check_next_to_the_identity(
    tmp_path: Path, bound: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_posttrain_readiness import NOW

    from tools import posttrain_readiness

    arguments = receipt_evidence(tmp_path, bound)
    argv = [
        "--" + name.replace("_", "-") + "=" + str(value)
        for name, value in arguments.items()
        if name not in ("launch_commit", "launch_check", "billing_origin_unix")
    ]
    argv += [f"--billing-origin-unix={arguments['billing_origin_unix']}"]
    argv += [f"--output={tmp_path / 'receipt.json'}"]
    monkeypatch.setattr(time, "time", lambda: NOW)
    assert posttrain_readiness.main(argv) == 1  # Run 1c without its launch commit.
    commit = arguments["launch_commit"]
    assert posttrain_readiness.main([*argv, f"--launch-commit={commit}"]) == 0
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["launch_commit"] == commit
    assert receipt["launch_check"] == json.loads((tmp_path / "launch-check.json").read_text())
    assert receipt["runtime_preflight"]["status"] == "passed"  # runtime-check.json, alongside.


def test_each_stop_reason_carries_its_own_docstring() -> None:
    from tools import posttrain_run

    body = ast.parse(Path(posttrain_run.__file__).read_text(encoding="utf-8")).body
    documented = {}
    for node, following in zip(body, body[1:], strict=False):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            text = None
            if isinstance(following, ast.Expr) and isinstance(following.value, ast.Constant):
                text = following.value.value
            documented[node.targets[0].id] = text if isinstance(text, str) else None
    assert str(documented["FORMAT_GATE_STOP"]).startswith("Run 1b: the selected SFT weights")
    assert str(documented["RUNTIME_STOP"]).startswith("Run 1c: the Pod's Python")
