"""Offline tests for the rehearsal worker: tiny authored shards, a fake clock, an in-memory R2."""

import copy
import functools
import hashlib
import json
import math
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from scratchcoder.common.config import ModelConfig, RunConfig
from scratchcoder.data.onepass import (
    combine_digests,
    document_sha256,
    multiset_digest,
    order_digest,
    pass_order,
)
from scratchcoder.data.shards import TokenShard, file_sha256
from scratchcoder.model.gpt import expected_parameters
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.train.trainer import Trainer
from scratchcoder.train.validation import ValidationWindow
from tools import rehearsal_binding, rehearsal_ops, rehearsal_run
from tools.pilot_preflight import run_preflight
from tools.pod_self_destruct import ABSOLUTE_DEADLINE_SECONDS
from tools.r2_store import Credentials
from tools.rehearsal_binding import ORDER_SEED, RUNS, SUBSET_RULE, VALIDATION_STREAMS
from tools.rehearsal_ops import EVIDENCE_GATES, OWNER_GATES, PULL_RESERVE_SECONDS
from tools.rehearsal_run import (
    DEFAULT_REVISION,
    FINAL_WRITE_SECONDS,
    FINALIZE_MARGIN_SECONDS,
    INITIALIZATION_RUN_ID,
    LAST_CALL_SECONDS,
    LAUNCH_RULE,
    MAX_IDLE_SECONDS,
    MAX_REPORT_BYTES,
    REHEARSAL_GATES,
    ROLLING_SKIP_SECONDS,
    SPEED_CHECK_RULE,
    Launch,
    MemoryBucket,
    Specification,
    Worker,
    author_streams,
    authored_run_data,
    binding_targets,
    check_preflight,
    check_readiness,
    check_revision,
    environment_secrets,
    length_sorted,
    load_run_data,
    main,
    planned_seconds,
    quarter_updates,
    smoke_specification,
    verify_initialization,
    write_initialization,
)
from tools.runpod_watchdog import MODE_WALL_SECONDS

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SPECS = {"A": CONFIGS / "rehearsal-a40-s-a.json", "B": CONFIGS / "rehearsal-a40-s-b.json"}
SECRET = "r2pod-secret-Zq81LmT0vXk3Jw"
KEY_ID = "r2pod-key-id-Hc5Nd2Rr"
CREDENTIALS = Credentials("https://acct.r2.cloudflarestorage.com", KEY_ID, SECRET, "rehearsal")
# Real S build (docs/rehearsal-data.md): validation tokens minus one BOS per document.
FULL_VALIDATION_TARGETS = (6_948_391 - 7_341) + (5_334_710 - 4_848)


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@dataclass
class Authored:
    root: Path
    tokenizer: ByteBPE
    streams: dict[str, list[tuple[str, TokenShard]]]
    initialization: Path
    initialization_sha256: str


def authored(root: Path) -> Authored:
    tokenizer = ByteBPE()
    streams = author_streams(root / "data", tokenizer)
    specs = [smoke_specification(run, streams) for run in RUNS]
    path = root / "initialization.pt"
    return Authored(
        root, tokenizer, streams, path, write_initialization(specs, tokenizer.sha256, path)
    )


@pytest.fixture
def data(tmp_path: Path) -> Authored:
    return authored(tmp_path)


def tiny_spec(data: Authored, run: str, **limits: Any) -> Specification:
    """The smoke specification with fake-clock-friendly cadences.

    Every 3 global updates, A's stage 1 (8 updates) ends between two validations, so
    stage 2's validations fall at other stage-local steps than global ones.
    """
    spec = smoke_specification(run, data.streams)
    spec = replace(
        spec,
        checkpoint_seconds=3.0,
        validation=replace(spec.validation, interval_updates=3),
    )
    return spec.with_limits(**{"speed_check_seconds": 5, **limits})


def make_worker(
    data: Authored,
    run: str,
    *,
    spec: Specification | None = None,
    clock: Clock | None = None,
    bucket: MemoryBucket | None = None,
    name: str | None = None,
    secrets: tuple[str, ...] = (),
    upload_retry_seconds: float = 15.0,
    extra: dict[str, Any] | None = None,
) -> Worker:
    spec = spec or tiny_spec(data, run)
    return Worker(
        spec,
        data.root / (name or f"run-{run}"),
        authored_run_data(spec, data.streams, data.tokenizer),
        initialization=data.initialization,
        initialization_sha256=data.initialization_sha256,
        store=(bucket or MemoryBucket()).store(CREDENTIALS),
        pod_id=f"pod-{run.lower()}",
        pod_created=1000.0,
        clock=clock or Clock(),
        secrets=secrets,
        upload_retry_seconds=upload_retry_seconds,
        extra=extra,
    )


Hook = Callable[[Trainer, dict[str, float]], dict[str, float] | None]


def stepping(worker: Worker, clock: Clock, seconds: float = 1.0, hook: Hook | None = None) -> None:
    """Advance the fake clock before each recorded update; ``hook`` may change its metrics."""
    original = worker.on_step

    def advance(trainer: Trainer, metrics: dict[str, float]) -> None:
        clock.now += seconds
        if hook is not None:
            metrics = hook(trainer, metrics) or metrics
        original(trainer, metrics)

    worker.on_step = advance  # type: ignore[method-assign]


def heartbeat(worker: Worker) -> dict[str, Any]:
    value: dict[str, Any] = json.loads((worker.root / "heartbeat.json").read_text())
    return value


def rows(worker: Worker) -> list[dict[str, Any]]:
    lines = (worker.root / "updates.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


# The checked-in specifications --------------------------------------------------------


APPROVED_TRAIN = {
    "seed": 0,
    "batch_size": 16,
    "sequence_length": 1024,
    "accumulation_steps": 8,
    "min_lr_ratio": 0.1,
    "weight_decay": 0.1,
    "gradient_clip": 1.0,
    "beta1": 0.9,
    "beta2": 0.95,
    "adam_epsilon": 1e-08,
    "precision": "bfloat16",
    "attention": "fused",
    "optimizer_foreach": True,
    "z_loss": 0.0,
    "schedule": "cosine",
    "compile": True,
}


def test_specifications_encode_the_approved_runs() -> None:
    a, b = Specification.load(SPECS["A"]), Specification.load(SPECS["B"])
    for spec in (a, b):
        model = spec.model
        assert (model.vocab_size, model.width, model.layers, model.heads, model.context) == (
            65_536,
            768,
            12,
            12,
            1024,
        )
        assert model.qk_norm is False and expected_parameters(model) == 135_285_504
        assert spec.to_dict()["train"] == APPROVED_TRAIN
        assert spec.tokens_per_update == 131_072 == rehearsal_binding.PACKING["tokens_per_update"]
        assert spec.warmup_fraction == 0.02
        assert spec.validation.interval_updates == 500
        assert spec.validation.subset_target_tokens == SUBSET_RULE["target_tokens_per_domain"]
        assert tuple(stream for _, stream in spec.validation.streams) == VALIDATION_STREAMS
        assert spec.checkpoint_seconds == 1800 and spec.checkpoint_save_seconds == 15
        assert spec.rolling_skip_seconds == ROLLING_SKIP_SECONDS == 900
        assert spec.max_transfer_seconds == rehearsal_ops.MAX_TRANSFER_SECONDS == 900
        assert (
            spec.expected_tokens_per_second == 56_479 == rehearsal_ops.REFERENCE_TOKENS_PER_SECOND
        )
        assert spec.quoted_gpu_hourly == 0.59 and spec.limits.quoted_total_hourly == 0.6
        limits = spec.limits
        assert limits.pod_seconds == 45_000 == MODE_WALL_SECONDS["rehearsal"]
        assert limits.pod_seconds == ABSOLUTE_DEADLINE_SECONDS
        assert limits.finalization_reserve_seconds == 900
        assert (limits.approved_cost_cap, limits.estimated_total_cost) == (10.0, 6.2917)
        # The hard stop is the $10 cap; the speed check stops below 90% (owner, 2026-10-25).
        assert limits.cost_multiplier == 1.6 and limits.cost_stop == 10.0
        assert (limits.minimum_speed_fraction, limits.speed_check_seconds) == (0.9, 900)
        assert limits.idle_seconds == 600 and limits.rising_steps == 25  # As in the pilot.
        # 45,000 s at the quoted rate stays under the cost stop, so the wall binds first.
        assert limits.pod_seconds / 3600 * limits.quoted_total_hourly < limits.cost_stop
    stage1, stage2 = a.stages
    assert ((stage1.name, stage1.streams), (stage2.name, stage2.streams)) == RUNS["A"]
    assert (stage1.learning_rate, stage2.learning_rate) == (0.0006, 0.0003)
    (mixed,) = b.stages
    assert (mixed.name, mixed.streams) == RUNS["B"][0] and mixed.learning_rate == 0.0006
    # The binding's counts at the S packing: 1e9 tokens per A stage, 2e9 for B.
    packed = rehearsal_binding._packed
    assert stage1.updates == stage2.updates == packed(10**9)["planned_updates"] == 7_630
    assert mixed.updates == packed(2 * 10**9)["planned_updates"] == 15_259
    assert stage1.warmup_updates == stage2.warmup_updates == round(0.02 * 7_630) == 153
    assert mixed.warmup_updates == round(0.02 * 15_259) == 305
    assert (a.total_updates, b.total_updates) == (15_260, 15_259)
    # Apart from the run and its stages the two specifications are identical.
    left, right = a.to_dict(), b.to_dict()
    for value in (left, right):
        del value["run"], value["stages"], value["train"]
    assert left == right


def test_specification_files_are_plain_json_without_private_material() -> None:
    for path in SPECS.values():
        text = path.read_text()
        assert "/Users/" not in text and "sha256" not in text and "R2_" not in text
        assert json.loads(text)["kind"] == "rehearsal-run-specification"


def test_launch_plan_is_the_whole_run_at_the_reference_speed() -> None:
    plans = {
        run: planned_seconds(
            Specification.load(path),
            stage_targets=[10**9, 10**9] if run == "A" else [2 * 10**9],
            full_validation_targets=FULL_VALIDATION_TARGETS,
        )
        for run, path in SPECS.items()
    }
    stages = {"A": 2, "B": 1}
    for run, plan in plans.items():
        assert plan["tokens_per_second"] == 56_479  # The reference speed, not the floor.
        assert plan["training_seconds"] == pytest.approx(2 * 10**9 / 56_479)  # 35,411 s
        assert plan["interval_validations"] == 31  # Global updates 0, 500, ..., 15,000.
        # Both full streams at each quarter and end of every stage: 8 passes for A, 4 for B.
        assert plan["full_validations"] == 4 * stages[run]
        tokens = 31 * 2 * 1_000_000 + 4 * stages[run] * FULL_VALIDATION_TARGETS
        assert plan["validation_tokens"] == tokens
        assert plan["validation_seconds"] == pytest.approx(tokens / 100_000)  # 1,602 / 1,111 s
        assert plan["startup_seconds"] == 300 * stages[run]  # Once per stage.
        # One save per 1,800 s of training and validation, plus one per stage end, at 15 s.
        saves = math.floor((plan["training_seconds"] + plan["validation_seconds"]) / 1_800)
        assert plan["checkpoint_saves"] == saves + stages[run]
        assert plan["checkpoint_seconds"] == 15 * plan["checkpoint_saves"]
        assert plan["finalization_reserve_seconds"] == 900
        parts = ("training", "validation", "startup", "checkpoint", "finalization_reserve")
        assert plan["total_seconds"] == pytest.approx(sum(plan[f"{p}_seconds"] for p in parts))
    assert plans["A"]["checkpoint_saves"] == 22 and plans["B"]["checkpoint_saves"] == 21
    assert plans["A"]["total_seconds"] == pytest.approx(38_843, abs=1)
    assert plans["B"]["total_seconds"] == pytest.approx(38_037, abs=1)
    # Pod setup before the worker starts may take up to this long before the armed
    # deadline 180 s inside the wall: about 100 and 113 minutes.
    assert 44_820 - plans["A"]["total_seconds"] == pytest.approx(5_977, abs=1)
    assert 44_820 - plans["B"]["total_seconds"] == pytest.approx(6_783, abs=1)
    # At the 90% floor the same run needs about 3,970 s more, and both still fit.
    floors = {
        run: planned_seconds(
            Specification.load(path),
            stage_targets=[10**9, 10**9] if run == "A" else [2 * 10**9],
            full_validation_targets=FULL_VALIDATION_TARGETS,
            tokens_per_second=0.9 * 56_479,
        )
        for run, path in SPECS.items()
    }
    assert floors["A"]["training_seconds"] == pytest.approx(39_346, abs=1)
    assert floors["A"]["total_seconds"] == pytest.approx(42_808, abs=1)
    assert floors["B"]["total_seconds"] == pytest.approx(42_002, abs=1)
    assert floors["B"]["total_seconds"] < floors["A"]["total_seconds"] < 44_820


def test_quarter_updates_are_rounded_quarters_strictly_inside_the_stage() -> None:
    assert quarter_updates(7_630) == ((1, 1_908), (2, 3_815), (3, 5_722))  # Run A's stages.
    assert quarter_updates(15_259) == ((1, 3_815), (2, 7_630), (3, 11_444))  # Run B's.
    assert quarter_updates(8) == ((1, 2), (2, 4), (3, 6))
    assert quarter_updates(7) == ((1, 2), (2, 4), (3, 5))  # round(1.75), round(3.5), ...
    assert quarter_updates(2) == ((2, 1),)  # round(0.5) = 0 and round(1.5) = 2 fall outside.
    assert quarter_updates(5) == ((1, 1), (2, 2), (3, 4))


def raw_spec() -> dict[str, Any]:
    value: dict[str, Any] = json.loads(SPECS["A"].read_text())
    return value


MUTATIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    "extra key": lambda v: v.update(notes="x"),
    "missing key": lambda v: v.pop("r2"),
    "train steps": lambda v: v["train"].update(steps=10),
    "train missing": lambda v: v["train"].pop("compile"),
    "kind": lambda v: v.update(kind="pilot"),
    "run": lambda v: v.update(run="C"),
    "run type": lambda v: v.update(run=["A"]),
    "stage streams": lambda v: v["stages"][1].update(streams=["stage2-web", "stage2-python"]),
    "stage name": lambda v: v["stages"][0].update(name="first"),
    "one stage": lambda v: v["stages"].pop(),
    "warmup": lambda v: v["stages"][0].update(warmup_updates=152),
    "updates type": lambda v: v["stages"][0].update(updates="7630"),
    "learning rate": lambda v: v["stages"][1].update(learning_rate=0),
    "z-loss": lambda v: v["train"].update(z_loss=0.0001),
    "interval": lambda v: v["validation"].update(interval_updates=0),
    "validation streams": lambda v: v["validation"].update(streams={"web": "validation-web"}),
    "autocast": lambda v: v["validation"].update(autocast_bf16=1),
    "drain": lambda v: v["r2"].update(drain_seconds=480),  # The drain uses the window now.
    "rolling skip": lambda v: v["r2"].update(rolling_skip_seconds=-1),
    "rolling skip missing": lambda v: v["r2"].pop("rolling_skip_seconds"),
    "transfer margin": lambda v: v["r2"].update(max_transfer_seconds=120),  # No drain left.
    "transfer window": lambda v: v["r2"].update(max_transfer_seconds=1_801),
    "transfer missing": lambda v: v["r2"].pop("max_transfer_seconds"),
    "save seconds": lambda v: v.update(checkpoint_save_seconds=0),
    "keep rolling": lambda v: v["r2"].update(keep_rolling=0),
    "price": lambda v: v.update(quoted_gpu_hourly=0.61),
    "speed warmup": lambda v: v.update(speed_check_warmup_updates=8000),
    "limits key": lambda v: v["limits"].pop("idle_seconds"),
    "no reserve": lambda v: v["limits"].update(finalization_reserve_seconds=0),
    "speed fraction": lambda v: v["limits"].update(minimum_speed_fraction=1.5),
    "context": lambda v: v["model"].update(context=512),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_specification_schema_is_strict(name: str) -> None:
    value = raw_spec()
    Specification.from_dict(copy.deepcopy(value))
    MUTATIONS[name](value)
    with pytest.raises(ValueError):
        Specification.from_dict(value)


def test_limits_keep_the_reserve_rule_when_changed() -> None:
    spec = Specification.load(SPECS["A"])
    shorter = spec.with_limits(pod_seconds=44_000)
    assert shorter.limits.pod_seconds == 44_000
    assert shorter.limits.training_seconds == 44_000 - 900
    assert shorter.limits.training_deadline(pod_created=0, training_started=500) == 43_100
    with pytest.raises(ValueError):
        replace(spec, limits=replace(spec.limits, training_seconds=100))


# Readiness and preflight gates ---------------------------------------------------------


HASHES = {
    "specification_sha256": "1" * 64,
    "binding_sha256": "2" * 64,
    "preflight_sha256": "3" * 64,
    "initialization_sha256": "4" * 64,
}


def bound_counts() -> dict[str, Any]:
    """The S build's counts in the shape ``check_readiness`` reads from a checked binding."""
    stages = {
        "A": [
            ("stage1", ["stage1-web"], 7_630, 10**9),
            ("stage2", ["stage2-python", "stage2-web"], 7_630, 10**9),
        ],
        "B": [("mixed", ["stage1-web", "stage2-web", "stage2-python"], 15_259, 2 * 10**9)],
    }
    return {
        "runs": {
            run: {
                "stages": [
                    {"name": name, "streams": streams, "planned_updates": updates, "targets": t}
                    for name, streams, updates, t in rows
                ]
            }
            for run, rows in stages.items()
        },
        "streams": {
            "validation-web": {"tokens": 6_948_391, "documents": 7_341},
            "validation-python": {"tokens": 5_334_710, "documents": 4_848},
        },
    }


BOUND = bound_counts()


def receipt(run: str = "A", **changes: Any) -> dict[str, Any]:
    """The receipt the receipt builder writes, in the fields the worker reads."""
    value: dict[str, Any] = {
        "kind": "rehearsal-readiness",
        "run": run,
        **HASHES,
        **{gate: True for gate in REHEARSAL_GATES},
        "termination_guarantee": "redundant_best_effort",
        "owner_accepts_best_effort_termination": True,
        "termination_evidence": "self-destruct receipt and watchdog policy",
        "pod_identity": {
            "id": "pod123",
            "name": "rehearsal-a",
            "gpuId": "NVIDIA A40",
            "gpuCount": 1,
            "dataCenterId": "CA-MTL-1",
            "createdAt": "1970-01-01T00:16:40Z",
        },
        "pod_created_at_unix": 1000,
        "watchdog_policy": {
            "max_wall_seconds": 45_000,
            "max_cost_usd": 7.54,
            "max_idle_seconds": 600,
            "quoted_total_hourly": 0.6,
            "max_transfer_seconds": 900,
            "billing_started_at_unix": 1000,
            "mode": "rehearsal",
        },
        "self_destruct_deadline_unix": 1000 + 45_000 - 180,
        "r2_run_prefix": f"rehearsal-runs/{run}/pod123/",
    }
    value.update(changes)
    return value


def ready(value: dict[str, Any], *, now: float, spec: Specification, **hashes: str) -> Launch:
    return check_readiness(value, now=now, spec=spec, binding=BOUND, **{**HASHES, **hashes})


def test_readiness_requires_every_rehearsal_gate_and_the_whole_run_to_fit() -> None:
    spec = Specification.load(SPECS["A"])
    need = planned_seconds(
        spec, stage_targets=[10**9, 10**9], full_validation_targets=FULL_VALIDATION_TARGETS
    )["total_seconds"]
    launch = ready(receipt(), now=1001, spec=spec)
    assert (launch.origin, launch.pod_id, launch.end) == (1000, "pod123", 1000 + 44_820)
    # Every deadline plans against the armed in-Pod deadline, 180 s inside the wall.
    assert launch.spec.limits.pod_seconds == 44_820
    assert launch.fit["reference_seconds_needed"] == pytest.approx(need)
    assert launch.fit["fits_at_reference"] and launch.fit["rule"] == LAUNCH_RULE
    assert "ssh_transfer_tested" not in REHEARSAL_GATES  # R2 replaces the SSH transfer path.
    assert set(REHEARSAL_GATES) == {
        *OWNER_GATES,
        *EVIDENCE_GATES,
        "pod_self_destruct_armed",
    }  # The receipt builder's gate names.
    for gate in REHEARSAL_GATES:
        unresolved = {**receipt(), gate: None}
        with pytest.raises(ValueError, match=gate):
            ready(unresolved, now=1001, spec=spec)
    # The launch rule: the lifetime left must hold the whole run at the reference speed.
    latest = 1000 + 44_820 - need
    assert ready(receipt(), now=latest, spec=spec).fit["remaining_seconds"] == pytest.approx(need)
    with pytest.raises(ValueError, match="cannot hold the run at the reference speed"):
        ready(receipt(), now=latest + 1, spec=spec)
    with pytest.raises(ValueError, match="owner acceptance"):
        ready(receipt(owner_accepts_best_effort_termination=False), now=1001, spec=spec)
    policy = receipt()["watchdog_policy"]
    with pytest.raises(ValueError, match="weaker"):
        ready(receipt(watchdog_policy={**policy, "max_cost_usd": 10.5}), now=1001, spec=spec)
    with pytest.raises(ValueError, match="pilot ceiling"):  # A 45,000 s policy must say rehearsal.
        ready(receipt(watchdog_policy={**policy, "mode": "pilot"}), now=1001, spec=spec)
    for window in (0, 600, 1_800):
        with pytest.raises(ValueError, match="uploading after GPU work"):
            changed = {**policy, "max_transfer_seconds": window}
            ready(receipt(watchdog_policy=changed), now=1001, spec=spec)
    with pytest.raises(ValueError, match="R2 prefix"):
        ready(receipt(r2_run_prefix="rehearsal-runs/A/other/"), now=1001, spec=spec)


def test_readiness_binds_kind_run_and_every_hash() -> None:
    spec = Specification.load(SPECS["A"])
    for change, message in (
        ({"kind": "pilot-readiness"}, "not a rehearsal-readiness receipt for run A"),
        ({"run": "B"}, "not a rehearsal-readiness receipt for run A"),
        ({"specification_sha256": "0" * 64}, "specification differs"),
        ({"binding_sha256": "0" * 64}, "binding identity differs"),
        ({"preflight_sha256": "0" * 64}, "preflight report differs"),
        ({"initialization_sha256": "0" * 64}, "initialization identity differs"),
    ):
        with pytest.raises(ValueError, match=message):
            ready(receipt(**change), now=1001, spec=spec)
    without = {key: value for key, value in receipt().items() if key != "initialization_sha256"}
    with pytest.raises(ValueError, match="initialization identity"):
        ready(without, now=1001, spec=spec)  # No longer optional.
    assert ready(receipt("B"), now=1001, spec=Specification.load(SPECS["B"])).pod_id == "pod123"


def test_lifetime_ends_at_the_first_guard() -> None:
    spec = Specification.load(SPECS["A"])
    policy = receipt()["watchdog_policy"]
    # A shorter watchdog wall, or a higher watchdog rate whose $7.54 stop comes first,
    # ends the Pod before the armed deadline; the worker plans against that moment.
    for change, end in (
        ({"max_wall_seconds": 44_000}, 1000 + 44_000),
        ({"quoted_total_hourly": 0.61}, 1000 + 7.54 / 0.61 * 3600),
    ):
        launch = ready(receipt(watchdog_policy={**policy, **change}), now=1001, spec=spec)
        assert launch.end == pytest.approx(end)
        assert launch.spec.limits.pod_seconds == pytest.approx(end - 1000)


def test_the_binding_must_plan_the_specifications_stages() -> None:
    spec = Specification.load(SPECS["A"])
    other = bound_counts()
    other["runs"]["A"]["stages"][0]["planned_updates"] = 7_631
    with pytest.raises(ValueError, match="planned updates differ"):
        binding_targets(spec, other)
    targets, full = binding_targets(spec, BOUND)
    assert targets == [10**9, 10**9] and full == FULL_VALIDATION_TARGETS


@pytest.mark.parametrize(
    "changes", [{"pod_seconds": 45_001}, {"idle_seconds": 601}, {"approved_cost_cap": 10.5}]
)
def test_readiness_refuses_a_specification_above_the_rehearsal_ceilings(
    changes: dict[str, float],
) -> None:
    spec = Specification.load(SPECS["A"]).with_limits(**changes)
    with pytest.raises(ValueError, match="ceiling"):
        ready(receipt(), now=1001, spec=spec)


def test_readiness_refuses_rolling_saves_too_close_to_a_stage_end() -> None:
    spec = replace(Specification.load(SPECS["A"]), rolling_skip_seconds=899.0)
    with pytest.raises(ValueError, match="rolling saves must stop 900 s before a stage ends"):
        ready(receipt(), now=1001, spec=spec)


REVISION: dict[str, Any] = {
    "git_commit": "c0ffee" + "0" * 34,
    "clean": True,
    "archive_sha256": "d" * 64,
}


def test_the_shipped_revision_must_equal_the_receipts() -> None:
    assert check_revision(receipt(revision=REVISION), dict(REVISION)) == REVISION
    others: list[Any] = [
        {**REVISION, "git_commit": "1" * 40},
        {**REVISION, "archive_sha256": "e" * 64},
        {key: value for key, value in REVISION.items() if key != "clean"},
        None,
        [],
    ]
    for shipped in others:
        with pytest.raises(ValueError, match="differs from the readiness receipt's"):
            check_revision(receipt(revision=REVISION), shipped)
    for recorded in (
        None,
        {**REVISION, "clean": False},  # A dirty tree never trains.
        {**REVISION, "git_commit": "C0FFEE" + "0" * 34},
        {**REVISION, "git_commit": "0" * 39},
        {**REVISION, "archive_sha256": "d" * 63},
        {**REVISION, "note": "x"},
    ):
        value = receipt(revision=recorded) if recorded is not None else receipt()
        with pytest.raises(ValueError, match="revision of a clean tree"):
            check_revision(value, recorded)


def test_preflight_gate_requires_the_compile_and_fused_attention_checks() -> None:
    spec = Specification.load(SPECS["B"])
    hardware = {"device": "cuda:0", "name": "NVIDIA A40", "uuid": "GPU-1"}
    passed = {
        "status": "completed",
        "hardware_precision_and_resume_checked": True,
        "device": "cuda:0",
        "hardware": hardware,
        "fused_attention": {"status": "passed"},
        "compile_equivalence": {"status": "passed"},
    }
    check_preflight(passed, hardware=hardware, spec=spec)
    without = {key: value for key, value in passed.items() if key != "compile_equivalence"}
    with pytest.raises(ValueError, match="compile equivalence"):
        check_preflight(without, hardware=hardware, spec=spec)
    with pytest.raises(ValueError, match="fused attention"):
        check_preflight(
            {**passed, "fused_attention": {"status": "failed"}}, hardware=hardware, spec=spec
        )
    for bad in ({"device": "cpu"}, {"status": "failed"}, {"hardware": {"name": "other"}}):
        with pytest.raises(ValueError, match="matching A40"):
            check_preflight({**passed, **bad}, hardware=hardware, spec=spec)


@functools.lru_cache(maxsize=1)
def _toolchain_available() -> bool:
    """Whether torch.compile can build even a trivial CPU function on this machine."""
    try:
        compiled = torch.compile(lambda x: x * 2 + 1)
        return bool(torch.equal(compiled(torch.ones(4)), torch.full((4,), 3.0)))
    except Exception:
        return False


def test_compile_equivalence_check_is_opt_in_and_passes_on_cpu(tmp_path: Path) -> None:
    pilot = run_preflight(tmp_path / "pilot")
    assert "compile_equivalence" not in pilot  # The pilot's preflight is unchanged.
    if not _toolchain_available():
        pytest.skip("torch.compile cannot build even a trivial CPU function here")
    report = run_preflight(tmp_path / "rehearsal", compile_check=True)
    check = report["compile_equivalence"]
    assert report["status"] == "completed" and check["status"] == "passed"
    assert set(check["precisions"]) == {"float32", "bfloat16"}
    for result in check["precisions"].values():
        assert result["loss_absolute_error"] <= result["loss_atol"] + result["loss_rtol"]
        assert result["max_gradient_relative_norm_error"] <= result["gradient_relative_norm_max"]
    assert check["precisions"]["float32"]["gradient_relative_norm_max"] == 1e-3


# The shared initialization -------------------------------------------------------------


def test_initialization_is_the_seed_zero_model_in_parent_format(data: Authored) -> None:
    state = torch.load(data.initialization, map_location="cpu", weights_only=True)
    spec = smoke_specification("A", data.streams)
    assert set(state) == {"config", "tokenizer_hash", "model", "run_id"}
    assert state["config"] == {"model": asdict(spec.model)}
    assert state["run_id"] == INITIALIZATION_RUN_ID
    assert state["tokenizer_hash"] == data.tokenizer.sha256
    assert file_sha256(data.initialization) == data.initialization_sha256
    # Deterministic bytes, so the Mac-side hash can be recomputed.
    again = data.root / "again.pt"
    specs = [smoke_specification(run, data.streams) for run in RUNS]
    assert write_initialization(specs, data.tokenizer.sha256, again) == data.initialization_sha256
    with pytest.raises(FileExistsError):
        write_initialization(specs, data.tokenizer.sha256, again)
    # A parent-loading trainer starts from exactly the weights a seed-0 trainer builds.
    train = spec.stage_train(0)
    fresh = Trainer(
        RunConfig(model=spec.model, train=train),
        [],  # type: ignore[arg-type]  # Never read: no update runs.
        run_id="fresh",
        run_root=data.root / "trainers",
        data_hash="x",
        tokenizer_hash=data.tokenizer.sha256,
    )
    parented = Trainer(
        RunConfig(model=spec.model, train=train, parent_checkpoint=str(data.initialization)),
        [],  # type: ignore[arg-type]
        run_id="parented",
        run_root=data.root / "trainers",
        data_hash="x",
        tokenizer_hash=data.tokenizer.sha256,
    )
    for name, tensor in fresh.model.state_dict().items():
        assert torch.equal(tensor, parented.model.state_dict()[name])
    assert parented.optimizer.steps == [0] * len(parented.optimizer.steps)


def test_initialization_writer_and_verifier_refuse_mismatches(data: Authored) -> None:
    spec = smoke_specification("A", data.streams)
    other = replace(spec, train=replace(spec.train, seed=1))
    with pytest.raises(ValueError, match="share one model and seed"):
        write_initialization([spec, other], data.tokenizer.sha256, data.root / "x.pt")
    with pytest.raises(ValueError, match="SHA-256"):
        write_initialization([spec], "not-a-digest", data.root / "x.pt")
    verify_initialization(
        data.initialization,
        expected_sha256=data.initialization_sha256,
        model=spec.model,
        tokenizer_identity=data.tokenizer.sha256,
    )
    for kwargs, message in (
        ({"expected_sha256": "0" * 64}, "SHA-256"),
        ({"tokenizer_identity": "1" * 64}, "tokenizer"),
        ({"model": replace(spec.model, width=32, heads=2)}, "model"),
    ):
        arguments: dict[str, Any] = {
            "expected_sha256": data.initialization_sha256,
            "model": spec.model,
            "tokenizer_identity": data.tokenizer.sha256,
            **kwargs,
        }
        with pytest.raises(ValueError, match=message):
            verify_initialization(data.initialization, **arguments)


# Both runs on authored data ------------------------------------------------------------


@dataclass
class Completed:
    data: Authored
    bucket: MemoryBucket
    workers: dict[str, Worker]
    reports: dict[str, dict[str, Any]]
    beats: dict[str, list[dict[str, Any]]]


@pytest.fixture(scope="module")
def completed(tmp_path_factory: pytest.TempPathFactory) -> Completed:
    """Runs A and B on one bucket, a fake clock advancing 1 s per update, every heartbeat kept."""
    data = authored(tmp_path_factory.mktemp("rehearsal"))
    bucket = MemoryBucket()
    workers, reports, beats = {}, {}, {}
    for run in RUNS:
        clock = Clock()
        worker = make_worker(data, run, clock=clock, bucket=bucket)
        stepping(worker, clock)
        seen: list[dict[str, Any]] = []
        original = worker.heartbeat

        def capture(worker: Worker = worker, original: Any = original, seen: Any = seen) -> None:
            original()
            beat = heartbeat(worker)
            beat["bucket_holds_report"] = worker.prefix + "report.json" in bucket.objects
            seen.append(beat)

        worker.heartbeat = capture  # type: ignore[method-assign]
        workers[run], reports[run], beats[run] = worker, worker.execute(), seen
    return Completed(data, bucket, workers, reports, beats)


def test_both_runs_complete_with_the_bound_tickets(completed: Completed) -> None:
    a, b = completed.reports["A"], completed.reports["B"]
    assert a["status"] == b["status"] == "completed"
    for report in (a, b):
        for stage in report["stages"]:
            ticket, expected = stage["ticket"], stage["expected_ticket"]
            assert stage["expected_ticket_match"] is True and ticket["complete"] is True
            assert ticket["multiset_sha256"] == expected["multiset_sha256"]
            assert ticket["order_sha256"] == expected["order_sha256"]
            assert ticket["targets"] == expected["targets"] == stage["tokens"]
        assert report["run_ticket"]["match"] is True
    stages = [stage["ticket"]["multiset_sha256"] for stage in a["stages"]]
    # A's two stages read exactly B's multiset, and both equal the bound value.
    assert combine_digests(*stages) == b["stages"][0]["ticket"]["multiset_sha256"]
    assert a["run_ticket"]["multiset_sha256"] == b["run_ticket"]["multiset_sha256"]
    assert a["run_ticket"]["expected_multiset_sha256"] == a["run_ticket"]["multiset_sha256"]
    assert a["run_ticket"]["targets"] == b["run_ticket"]["targets"]
    assert a["initial_weights_sha256"] == b["initial_weights_sha256"]
    assert a["validation_identity"] == b["validation_identity"]
    # One shared initialization and one fixed subset: identical step-0 validations.
    assert a["stages"][0]["validations"][0] == {
        **b["stages"][0]["validations"][0],
        "seconds": a["stages"][0]["validations"][0]["seconds"],
    }


def test_every_target_is_read_once_and_the_last_update_is_masked(completed: Completed) -> None:
    for run, worker in completed.workers.items():
        report = completed.reports[run]
        updates = rows(worker)
        assert [row["update"] for row in updates] == list(range(1, worker.spec.total_updates + 1))
        for stage in report["stages"]:
            mine = [row for row in updates if row["stage"] == stage["name"]]
            assert [row["stage_update"] for row in mine] == list(
                range(1, stage["planned_updates"] + 1)
            )
            assert stage["updates"] == stage["planned_updates"]
            full = worker.spec.tokens_per_update
            assert all(row["tokens"] == full for row in mine[:-1])
            last = stage["planned_targets"] - full * (stage["planned_updates"] - 1)
            assert mine[-1]["tokens"] == last and 0 < last <= full
        assert report["training_tokens_processed"] == worker.total_targets


def test_stage_two_starts_from_the_verified_stage_one_weights_with_a_fresh_optimizer(
    completed: Completed,
) -> None:
    worker, report = completed.workers["A"], completed.reports["A"]
    stage1, stage2 = report["stages"]
    assert stage1["parent_checkpoint_sha256"] == report["initial_weights_sha256"]
    assert stage2["parent_checkpoint_sha256"] == stage1["final_checkpoint_sha256"]
    records = {
        name: json.loads((worker.root / "runs" / f"rehearsal-A-{name}" / "record.json").read_text())
        for name in ("stage1", "stage2")
    }
    assert records["stage1"]["parent_run_id"] == INITIALIZATION_RUN_ID
    assert records["stage2"]["parent_run_id"] == "rehearsal-A-stage1"
    assert records["stage2"]["config"]["parent_checkpoint"] == str(
        worker.root / "checkpoints" / "stage1" / "final.pt"
    )
    for record, stage in zip(records.values(), report["stages"], strict=True):
        train = record["config"]["train"]
        assert (train["learning_rate"], train["warmup_steps"], train["steps"]) == (
            stage["learning_rate"],
            stage["warmup_updates"],
            stage["planned_updates"],
        )
        final = torch.load(
            worker.root / "exports" / f"{stage['name']}-final" / "checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        # A fresh AdamW per stage: its counters hold this stage's updates only.
        assert final["optimizer"]["steps"] == [stage["planned_updates"]] * len(
            final["optimizer"]["steps"]
        )
        assert final["step"] == stage["planned_updates"]
    updates = rows(worker)
    for stage in report["stages"]:
        mine = [row for row in updates if row["stage"] == stage["name"]]
        peak, warmup = stage["learning_rate"], stage["warmup_updates"]
        assert mine[0]["learning_rate"] == pytest.approx(peak / warmup)  # A fresh warmup.
        assert mine[warmup - 1]["learning_rate"] == pytest.approx(peak)
        assert mine[-1]["learning_rate"] == pytest.approx(peak * 0.1)  # min_lr_ratio.


def test_validation_runs_on_global_updates_and_full_streams_at_each_stage_end(
    completed: Completed,
) -> None:
    for run, report in completed.reports.items():
        worker = completed.workers[run]
        interval = worker.spec.validation.interval_updates
        start = 0
        for index, stage in enumerate(report["stages"]):
            end = start + stage["planned_updates"]
            first = start if index == 0 else start + 1
            expected = [u for u in range(first, end + 1) if u % interval == 0]
            assert [row["global_update"] for row in stage["validations"]] == expected
            for row in stage["validations"]:
                for label in ("web", "python"):
                    assert row[label]["tokens"] == worker.spec.validation.subset_target_tokens
            full = stage["full_validation"]
            assert full["global_update"] == end
            # Both full streams at each quarter of the stage too, then at its end.
            quarters = quarter_updates(stage["planned_updates"])
            assert stage["planned_quarter_updates"] == [update for _, update in quarters]
            assert [
                (row["quarter"], row["stage_update"], row["global_update"])
                for row in stage["quarter_validations"]
            ] == [(quarter, update, start + update) for quarter, update in quarters]
            for row in [*stage["quarter_validations"], full]:
                for label, shard in worker.data.validation:
                    upper = sum(length - 1 for _, length in shard.offsets)
                    assert row[label]["tokens"] == upper  # Every authored token is one byte.
            start = end
    # A's stage-2 cadence continues A's global count rather than restarting.
    a = completed.reports["A"]["stages"]
    assert a[1]["validations"][0]["global_update"] > a[0]["planned_updates"]
    assert [row["global_update"] for row in a[0]["quarter_validations"]] == [2, 4, 6]
    assert [row["global_update"] for row in a[1]["quarter_validations"]] == [10, 12, 13]
    b = completed.reports["B"]["stages"][0]
    assert [row["global_update"] for row in b["quarter_validations"]] == [4, 8, 11]


def test_checkpoints_reach_r2_as_rolling_labels_and_milestones(completed: Completed) -> None:
    bucket = completed.bucket
    for run, worker in completed.workers.items():
        report = completed.reports[run]
        labels = [label for stage in report["stages"] for label in stage["r2_labels"]]
        rolling = [label for label in labels if not label.endswith("-final")]
        assert rolling and all(
            label.split("-step")[0] in {stage["name"] for stage in report["stages"]}
            for label in rolling
        )
        assert worker.milestones == [f"{stage['name']}-final" for stage in report["stages"]]
        held = {
            key.removeprefix(worker.prefix).split("/")[0]
            for key in bucket.objects
            if key.startswith(worker.prefix) and key != worker.prefix + "report.json"
        }
        assert set(worker.milestones) <= held
        assert len(held - set(worker.milestones)) <= worker.spec.keep_rolling  # Retention.
        for label in held:
            files = {
                key.rsplit("/", 1)[1]
                for key in bucket.objects
                if key.startswith(f"{worker.prefix}{label}/")
            }
            names = {"checkpoint.pt", "checkpoint.pt.sha256", "record.json", "updates.jsonl"}
            assert files == (names | {"ticket.json"} if label.endswith("-final") else names)
            checkpoint = bucket.objects[f"{worker.prefix}{label}/checkpoint.pt"]
            digest = bucket.objects[f"{worker.prefix}{label}/checkpoint.pt.sha256"].decode()
            assert digest == f"{file_sha256_bytes(checkpoint)}  checkpoint.pt\n"
            record = json.loads(bucket.objects[f"{worker.prefix}{label}/record.json"])
            assert record["status"] == "running"  # A label is never the run's completion.
        for stage in report["stages"]:
            label = f"{stage['name']}-final"
            ticket = json.loads(bucket.objects[f"{worker.prefix}{label}/ticket.json"])
            assert ticket["ticket"] == stage["ticket"] and ticket["expected_ticket_match"] is True
            checkpoint = bucket.objects[f"{worker.prefix}{label}/checkpoint.pt"]
            assert file_sha256_bytes(checkpoint) == stage["final_checkpoint_sha256"]
            log = bucket.objects[f"{worker.prefix}{label}/updates.jsonl"].decode().splitlines()
            assert json.loads(log[-1])["update"] == stage_end(report, stage["name"])  # Cumulative.
        # Locally only the milestones and the newest rolling label remain.
        exports = {path.name for path in (worker.root / "exports").iterdir()}
        assert exports - set(worker.milestones) <= {rolling[-1]}


def file_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stage_end(report: dict[str, Any], name: str) -> int:
    total = 0
    for stage in report["stages"]:
        total += int(stage["planned_updates"])
        if stage["name"] == name:
            return total
    raise AssertionError(name)


def test_heartbeat_runs_across_the_stage_boundary_and_ends_after_r2_verification(
    completed: Completed,
) -> None:
    for run, beats in completed.beats.items():
        worker, report = completed.workers[run], completed.reports[run]
        statuses = [beat["status"] for beat in beats]
        first = statuses.index("uploading")
        # Training reads running; the final drain to R2 reads uploading; then the proof.
        assert set(statuses[:first]) == {"running"} and statuses[-1] == "awaiting_transfer"
        assert set(statuses[first:-1]) == {"uploading"}
        uploading = beats[first:-1]
        assert {beat["phase"] for beat in uploading} == {"finalizing"}
        assert all("r2_final" not in beat for beat in uploading)
        # No GPU work while uploading: the activity stamp never moves, only updated_at.
        assert len({beat["last_gpu_activity_unix"] for beat in beats[first:]}) == 1
        final = beats[-1]
        assert final["bucket_holds_report"] is True
        assert "r2_final" not in beats[-2]
        r2 = final["r2_final"]
        assert r2["report_key"] == f"rehearsal-runs/{run}/pod-{run.lower()}/report.json"
        assert r2["checkpoint_key"] == f"{worker.prefix}{worker.milestones[-1]}/checkpoint.pt"
        assert r2["checkpoint_sha256"] == report["stages"][-1]["final_checkpoint_sha256"]
        stored = completed.bucket.objects[r2["report_key"]]
        assert stored == (worker.root / "report.json").read_bytes()
        assert file_sha256_bytes(stored) == r2["report_sha256"]
        assert json.loads(stored)["status"] == "completed"
        assert all(type(beat["step"]) is int for beat in beats)
        # Every heartbeat names its Pod, so a Mac-side reader can ignore another Pod's.
        assert {beat["pod_id"] for beat in beats} == {f"pod-{run.lower()}"}
        assert {"schema_version", "phase", "last_gpu_activity_unix", "updated_at_unix"} <= set(
            final
        )
        assert "failure" not in final
        record = json.loads((worker.root / "finalization.json").read_text())
        assert record["unverified_milestones"] == [] and record["status"] == "completed"
        assert record["settle_rounds"][0]["drained"] is True
        assert record["drain_budget_seconds"] == pytest.approx(
            worker.spec.max_transfer_seconds - PULL_RESERVE_SECONDS - FINALIZE_MARGIN_SECONDS,
            abs=1,
        )  # The whole drain: the last GPU work was the stage-end validation.
        assert record["handover_due_unix"] == pytest.approx(
            record["transfer_window_end_unix"] - PULL_RESERVE_SECONDS
        )
    a = completed.beats["A"]
    stages = [beat["stage"] for beat in a]
    boundary = stages.index("stage2")
    assert a[boundary - 1]["phase"].startswith("stage1") and a[boundary]["status"] == "running"
    assert {beat["status"] for beat in a if beat["phase"] == "stage1-full-validation"} == {
        "running"
    }


def test_reports_and_heartbeats_stay_within_their_bounds(completed: Completed) -> None:
    for worker in completed.workers.values():
        assert (worker.root / "report.json").stat().st_size < MAX_REPORT_BYTES
        assert (worker.root / "heartbeat.json").stat().st_size < 65_536
        report = completed.reports[worker.spec.run]
        assert "loss_curves" not in report  # Per-update rows live in updates.jsonl.
        assert report["speed_check"]["status"] == "passed"
        assert report["cost"]["quoted_accrued_cost_usd"] < report["cost"]["cost_stop_usd"]


def test_report_at_the_real_run_scale_stays_far_below_one_megabyte(data: Authored) -> None:
    worker = make_worker(data, "A")
    for index, stage in enumerate(worker.stage_records):
        result = {"loss": 1.0, "bits_per_byte": 1.0, "tokens": 1e6, "bytes": 4e6}
        stage["validations"] = [
            {"global_update": 500 * k, "web": result, "python": result, "seconds": 30.0}
            for k in range(16 * index, 16 * index + 16)
        ]
        stage["ticket"] = {"multiset_sha256": "0" * 64, "per_stream": {"a": {"tokens": 1}}}
    # About 25 rolling saves in 45,000 s at the 1,800 s cadence, plus the milestones.
    row = {
        "label": "stage1-step12345",
        "stage": "stage1",
        "global_update": 12_345,
        "stage_update": 4_715,
        "milestone": False,
        "path": "checkpoints/stage1/last.pt",
        "save_seconds": 9.5,
        "size_bytes": 1_623_507_391,
        "sha256": "0" * 64,
        "training_tokens_processed": 1_618_083_840,
    }
    worker.checkpoints.extend(copy.deepcopy(row) for _ in range(30))
    worker.report["r2"]["uploader"] = {"verified": [f"stage1-step{i}" for i in range(30)]}
    worker.publish()
    assert (worker.root / "report.json").stat().st_size < 100_000
    before = (worker.root / "report.json").read_bytes()
    worker.report["padding"] = "x" * MAX_REPORT_BYTES
    with pytest.raises(ValueError, match="over 1,000,000"):
        worker.publish()
    assert (worker.root / "report.json").read_bytes() == before
    worker.uploader.close()


# Stops ---------------------------------------------------------------------------------


def finished(worker: Worker) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads((worker.root / "report.json").read_text())
    return report, heartbeat(worker)


def test_speed_check_stops_a_run_below_the_throughput_floor(data: Authored) -> None:
    clock = Clock()
    spec = replace(tiny_spec(data, "B"), expected_tokens_per_second=1e12)
    worker = make_worker(data, "B", spec=spec, clock=clock)
    stepping(worker, clock)
    result = worker.execute()
    assert result["status"] == "stopped" and result["stop_reason"] == "throughput_below_limit"
    check = result["speed_check"]
    assert check["status"] == "failed" and check["reason"] == "throughput_below_limit"
    assert check["measured_tokens_per_second"] < check["floor_tokens_per_second"]
    assert check["rule"] == SPEED_CHECK_RULE
    report, beat = finished(worker)
    assert beat["status"] == "stopped" and report["stages"][0]["status"] == "running"
    assert not worker.milestones
    # No stage-final milestone closes anything: no r2_final, so the supervisor's
    # diagnostics path (not its verification) applies.
    assert "r2_final" not in beat


def test_speed_check_counts_planned_saves_before_the_first_save(data: Authored) -> None:
    clock = Clock()
    spec = replace(tiny_spec(data, "A"), checkpoint_seconds=1e6, checkpoint_save_seconds=7.0)
    worker = make_worker(data, "A", spec=spec, clock=clock)
    stepping(worker, clock)
    result = worker.execute()
    check = result["speed_check"]
    assert check["status"] == "passed"
    assert check["checkpoint_save_seconds_basis"] == "planned checkpoint_save_seconds"
    # Stage 1's final save and stage 2's: two planned saves of 7 s each.
    assert check["projected_seconds"]["checkpoint_seconds"] == 14.0


def test_final_drain_ends_before_the_handover_that_leaves_the_ssh_pull_reserve(
    data: Authored,
) -> None:
    clock = Clock()
    spec = tiny_spec(data, "B", finalization_reserve_seconds=900)
    worker = make_worker(data, "B", spec=spec, clock=clock)
    # 900 s window: a 360 s drain, 120 s of checks, then 420 s for the Mac's SSH pull.
    assert worker.handover_due() == pytest.approx(worker.last_activity + 480)
    assert 480 < MAX_IDLE_SECONDS  # Handed over before the watchdog's idle limit.
    for idle, budget in ((0, 360), (300, 60), (360, 0), (2_000, 0)):
        clock.now = worker.last_activity + idle
        assert worker.drain_budget() == pytest.approx(budget)
    clock.now = worker.pod_end - 300  # Never past the Pod end, less a minute.
    worker.last_activity = clock.now
    assert worker.handover_due() == pytest.approx(worker.pod_end - 60)
    assert worker.drain_budget() == pytest.approx(120)
    worker.uploader.close()


class TimedStore:
    """A check store whose calls take time on the fake clock and may fail.

    ``seconds`` maps the attempt's socket timeout to how long the call takes, and
    ``put_seconds`` does for a PUT; ``fails`` says whether a call on that key raises a
    timeout. Every call is recorded with its key, start and timeout.
    """

    def __init__(
        self,
        inner: Any,
        clock: Clock,
        calls: list[tuple[str, str, float, float]],
        timeout: float,
        seconds: Callable[[float], float],
        fails: Callable[[str, str], bool],
        put_seconds: Callable[[float], float],
    ) -> None:
        self.inner, self.clock, self.calls, self.timeout = inner, clock, calls, timeout
        self.seconds, self.fails, self.put_seconds = seconds, fails, put_seconds

    def _call(self, method: str, key: str) -> None:
        self.calls.append((method, key, self.clock.now, self.timeout))
        self.clock.now += (self.put_seconds if method == "PUT" else self.seconds)(self.timeout)
        if self.fails(method, key):
            raise TimeoutError("no answer")

    def stored_sha256(self, key: str) -> str | None:
        self._call("HEAD", key)
        return self.inner.stored_sha256(key)  # type: ignore[no-any-return]

    def put_known(self, key: str, path: Path, sha256: str, size: int) -> int:
        self._call("PUT", key)
        return self.inner.put_known(key, path, sha256, size)  # type: ignore[no-any-return]


def timed_finalization(
    data: Authored,
    *,
    start: float,
    seconds: Callable[[float], float],
    fails: Callable[[str, str], bool] = lambda method, key: False,
    put_seconds: Callable[[float], float] | None = None,
) -> tuple[Worker, dict[str, Any], list[tuple[str, str, float, float]], dict[str, Any]]:
    """Run A with the real 900 s window, finalization starting ``start`` s after GPU work.

    Closing the uploader takes its whole timeout on the fake clock, as when an upload in
    flight keeps its thread busy, so the final heartbeat's time includes that wait.
    """
    clock = Clock()
    bucket = MemoryBucket()
    worker = make_worker(data, "A", clock=clock, bucket=bucket)
    assert worker.spec.max_transfer_seconds == 900
    calls: list[tuple[str, str, float, float]] = []
    inner = bucket.store(CREDENTIALS)  # The uploader keeps its own store.
    worker.check_store_for = lambda timeout: TimedStore(  # type: ignore[assignment]
        inner, clock, calls, timeout, seconds, fails, put_seconds or seconds
    )
    worker.sleep = lambda pause: setattr(clock, "now", clock.now + pause)
    close = worker.uploader.close

    def slow_close(timeout: float | None = None) -> bool:
        assert timeout is not None  # Finalization never waits for the uploader unbounded.
        clock.now += timeout
        return close(timeout)

    worker.uploader.close = slow_close  # type: ignore[method-assign]
    stepping(worker, clock)
    original = worker.finalize

    def late() -> None:
        assert worker.uploader.drain(30)  # Every upload done: only the checks remain.
        clock.now = worker.last_activity + start
        original()

    worker.finalize = late  # type: ignore[method-assign]
    result = worker.execute()
    record = json.loads((worker.root / "finalization.json").read_text())
    return worker, result, calls, record


def test_checks_after_the_drain_end_in_time_even_at_the_worst_start(data: Authored) -> None:
    # R2 never answers, each attempt takes four socket timeouts (four addresses), and the
    # first check starts just before the last-call time: 434 s after the last GPU work.
    worker, result, calls, record = timed_finalization(
        data, start=434.0, seconds=lambda timeout: 4 * timeout, fails=lambda m, k: True
    )
    handover = record["handover_due_unix"]
    assert handover == pytest.approx(worker.last_activity + 480)
    assert calls and all(start < handover - LAST_CALL_SECONDS for _, _, start, _ in calls)
    assert all(
        start + 4 * timeout <= handover - FINAL_WRITE_SECONDS + 1e-9
        for _, _, start, timeout in calls
    )
    assert "/stage2-final/" in calls[0][1]  # The newest milestone first.
    assert result["status"] == "failed" and result["failure"] == "r2_milestones_unverified"
    assert {row["label"] for row in record["unverified_milestones"]} == {
        "stage1-final",
        "stage2-final",
    }
    assert record["error"].startswith("HandoverDue") and "report.json" in record["error"]
    # The final heartbeat lands by the handover, before the watchdog's 600 s idle limit,
    # even after uploader.close(timeout=5) waited its whole timeout on the fake clock.
    final = heartbeat(worker)
    last_check_end = max(start + 4 * timeout for _, _, start, timeout in calls)
    assert record["uploader_closed"] is True
    assert last_check_end + 5 <= final["updated_at_unix"] <= handover
    assert final["status"] == "failed"
    assert handover - worker.last_activity < MAX_IDLE_SECONDS


def test_a_slow_healthy_r2_checks_milestones_then_the_report_and_completes(
    data: Authored,
) -> None:
    # 4 s per request from the end of a full drain: milestones, then the report's PUT and
    # its HEAD, then rolling labels only while time remains.
    worker, result, calls, record = timed_finalization(data, start=359.0, seconds=lambda _: 4.0)
    assert result["status"] == "completed" and heartbeat(worker)["r2_final"] is not None
    order = [key.split("/")[3] for _, key, _, _ in calls]
    report = order.index("report.json")
    assert set(order[:report]) == {"stage2-final", "stage1-final"}
    assert order[report : report + 2] == ["report.json", "report.json"]  # PUT, then HEAD.
    assert all("-step" in label for label in order[report + 2 :])
    assert record["labels_verified"]["stage2-final"] is True
    assert heartbeat(worker)["updated_at_unix"] <= record["handover_due_unix"]


def test_transient_failures_after_the_drain_are_retried_while_time_lasts(data: Authored) -> None:
    failures = {"PUT": 3, "HEAD": 2}

    def fails(method: str, key: str) -> bool:
        if key.endswith("report.json") and failures[method]:
            failures[method] -= 1
            return True
        return False

    worker, result, calls, record = timed_finalization(
        data, start=0.0, seconds=lambda _: 1.0, fails=fails
    )
    assert result["status"] == "completed" and "error" not in record
    puts = [call for call in calls if call[0] == "PUT"]
    assert len(puts) == 4  # Three timeouts, then the upload; more than one bounded store's 2.
    assert record["report_sha256"] == heartbeat(worker)["r2_final"]["report_sha256"]


@pytest.mark.parametrize("stalls", [False, True])
def test_rolling_labels_are_recorded_when_the_report_upload_fails(
    data: Authored, stalls: bool
) -> None:
    """Every milestone verifies, then every report.json PUT times out: the run fails
    ``r2_finalization``. The rolling labels are still checked by HEAD while time lasts;
    when PUTs that stall on four addresses use the time up, they are named unchecked."""
    worker, result, calls, record = timed_finalization(
        data,
        start=0.0,
        seconds=lambda _: 1.0,
        fails=lambda method, key: method == "PUT",
        put_seconds=(lambda timeout: 4 * timeout) if stalls else None,
    )
    assert result["status"] == "failed" and result["failure"] == "r2_finalization"
    assert record["unverified_milestones"] == [] and "report_sha256" not in record
    rolling = [label for label in worker.label_files if label not in worker.milestones]
    kept = [label for label in rolling if label in worker.kept_labels()]
    assert kept and len(kept) < len(rolling)  # The uploader deleted the older ones.
    verified = record["labels_verified"]
    assert verified["stage1-final"] is True and verified["stage2-final"] is True
    assert {label: verified[label] for label in rolling if label not in kept} == {
        label: False for label in rolling if label not in kept
    }
    rolling_heads = [
        index for index, (method, key, _, _) in enumerate(calls) if "-step" in key.split("/")[3]
    ]
    last_put = max(index for index, call in enumerate(calls) if call[0] == "PUT")
    if stalls:
        # The fifth PUT ended after the last call: no rolling HEAD could start.
        assert record["error"].startswith("HandoverDue") and "report.json" in record["error"]
        assert record["labels_unchecked"] == kept and not rolling_heads
        assert not set(kept) & set(verified)
        assert all(record["verification_errors"][label].startswith("HandoverDue") for label in kept)
    else:
        # Every attempt failed with time to spare.
        assert record["error"] == "TimeoutError: no answer"
        assert sum(call[0] == "PUT" for call in calls) == rehearsal_run.CHECK_ATTEMPTS
        assert record["labels_unchecked"] == []
        assert all(verified[label] is True for label in kept)
        assert rolling_heads and min(rolling_heads) > last_put  # Checked after the failure.
    assert record["last_verified_label"] == "stage2-final"
    final = heartbeat(worker)
    assert final["status"] == "failed" and final["updated_at_unix"] <= record["handover_due_unix"]


def test_speed_check_stops_a_run_projected_to_pass_the_deadline(data: Authored) -> None:
    clock = Clock()
    spec = replace(tiny_spec(data, "A", pod_seconds=400), expected_tokens_per_second=0.1)
    worker = make_worker(data, "A", spec=spec, clock=clock)

    def slow(trainer: Trainer, metrics: dict[str, float]) -> dict[str, float]:
        return {**metrics, "step_seconds": 100.0}  # Measured throughput 0.32 tok/s.

    stepping(worker, clock, hook=slow)
    result = worker.execute()
    assert result["status"] == "stopped"
    assert result["stop_reason"] == "projected_finish_after_deadline"
    check = result["speed_check"]
    assert check["measured_tokens_per_second"] >= check["floor_tokens_per_second"]
    assert check["projected_finish_unix"] > check["pod_end_unix"] == 1400
    parts = check["projected_seconds"]
    assert parts["finalization_reserve_seconds"] == 120
    assert parts["training_seconds"] == pytest.approx(check["remaining_targets"] / 0.32)
    assert parts["recompile_seconds"] == 0  # Every update looked alike: nothing to recompile.
    assert check["remaining_validation_tokens"] > 0


def test_speed_check_excludes_the_compiling_first_updates(data: Authored) -> None:
    clock = Clock()
    spec = replace(tiny_spec(data, "A"), expected_tokens_per_second=10.0)
    worker = make_worker(data, "A", spec=spec, clock=clock)

    def compiling(trainer: Trainer, metrics: dict[str, float]) -> dict[str, float]:
        if trainer.step <= 2:
            return {**metrics, "step_seconds": 1000.0}
        return metrics

    stepping(worker, clock, hook=compiling)
    result = worker.execute()
    assert result["status"] == "completed"
    check = result["speed_check"]
    assert check["status"] == "passed" and check["measured_tokens_per_second"] > 8.5
    # Stage 2 will compile again; the estimate is the first updates' excess time.
    assert check["projected_seconds"]["recompile_seconds"] > 1900


@pytest.mark.parametrize(("jump", "reason"), [(1300.0, "cost_limit"), (4000.0, "pod_wall_limit")])
def test_wall_and_cost_stops_end_the_run_without_waiting_for_uploads(
    data: Authored, jump: float, reason: str
) -> None:
    clock = Clock()
    spec = tiny_spec(data, "A", quoted_total_hourly=3.6)
    spec = replace(spec, quoted_gpu_hourly=3.0)
    worker = make_worker(data, "A", spec=spec, clock=clock)

    def leap(trainer: Trainer, metrics: dict[str, float]) -> None:
        if trainer.step == 4:  # Not a validation update: the limit is seen by the update.
            clock.now += jump

    stepping(worker, clock, hook=leap)
    result = worker.execute()
    assert result["status"] == "stopped" and result["stop_reason"] == reason
    report, beat = finished(worker)
    assert beat["status"] == "stopped" and "r2_final" not in beat
    assert not worker.milestones
    final = json.loads((worker.root / "finalization.json").read_text())
    assert final["drain_budget_seconds"] == 0 and "report_sha256" not in final


def test_idle_and_rising_loss_stop_the_run(data: Authored) -> None:
    clock = Clock()
    idle = make_worker(data, "B", clock=clock, name="idle")
    original = idle.on_step

    def stall(trainer: Trainer, metrics: dict[str, float]) -> None:
        original(trainer, metrics)
        if trainer.step == 4:
            clock.now += 700  # No GPU work for longer than the idle limit.

    idle.on_step = stall  # type: ignore[method-assign]
    assert idle.execute()["stop_reason"] == "idle_gpu_limit"
    assert heartbeat(idle)["status"] == "stopped"
    clock = Clock()
    rising = make_worker(
        data, "B", spec=tiny_spec(data, "B", rising_steps=5), clock=clock, name="rising"
    )
    stepping(rising, clock, hook=lambda t, m: {**m, "loss": 10.0 + t.step})
    result = rising.execute()
    assert result["status"] == "stopped" and result["stop_reason"] == "loss_rising"
    assert result["stages"][0]["updates"] == 6  # Five consecutive increases after update 1.


def test_nonfinite_loss_fails_the_run(data: Authored) -> None:
    clock = Clock()
    worker = make_worker(data, "B", clock=clock)
    stepping(worker, clock, hook=lambda t, m: {**m, "loss": math.nan} if t.step == 3 else m)
    with pytest.raises(FloatingPointError):
        worker.execute()
    report, beat = finished(worker)
    assert report["status"] == "failed" and report["error_type"] == "FloatingPointError"
    assert beat["status"] == "failed"


def test_finalization_reserve_ends_a_stage_early_with_a_verified_milestone(data: Authored) -> None:
    clock = Clock()
    spec = tiny_spec(data, "A", pod_seconds=300)  # Training deadline 1180 = 1300 - 120.
    bucket = MemoryBucket()
    worker = make_worker(data, "A", spec=spec, clock=clock, bucket=bucket)
    stepping(worker, clock, seconds=20.0)
    result = worker.execute()
    assert result["status"] == "time_limited"
    assert result["training_stop_reason"] == "finalization_reserve"
    stage1, stage2 = result["stages"]
    assert stage1["status"] == "completed" and stage1["expected_ticket_match"] is True
    assert stage2["status"] == "time_limited" and 0 < stage2["updates"] < stage2["planned_updates"]
    assert stage2["ticket"]["complete"] is False and stage2["expected_ticket_match"] is False
    assert stage2["full_validation"] == {"status": "skipped", "reason": "finalization_reserve"}
    assert result["run_ticket"]["match"] is False
    beat = heartbeat(worker)
    assert beat["status"] == "awaiting_transfer"
    assert beat["r2_final"]["checkpoint_key"].endswith("/stage2-final/checkpoint.pt")
    assert f"{worker.prefix}stage2-final/ticket.json" in bucket.objects


def refuse_put(part: str) -> Callable[[str, str], tuple[int, bytes] | None]:
    def fault(method: str, key: str) -> tuple[int, bytes] | None:
        return (403, b"AccessDenied") if method == "PUT" and part in key else None

    return fault


def short_window(data: Authored, run: str = "A") -> Specification:
    """A one-second drain before the finalize margin and pull reserve: a failing drain ends
    fast."""
    return replace(
        tiny_spec(data, run),
        max_transfer_seconds=FINALIZE_MARGIN_SECONDS + PULL_RESERVE_SECONDS + 1,
    )


@pytest.mark.parametrize("broken", ["stage2-final", "stage1-final", "report"])
def test_unverified_r2_copies_never_read_awaiting_transfer(data: Authored, broken: str) -> None:
    clock = Clock()
    bucket = MemoryBucket()
    bucket.fault = refuse_put("report.json" if broken == "report" else f"{broken}/")
    worker = make_worker(
        data,
        "A",
        spec=short_window(data),
        clock=clock,
        bucket=bucket,
        name=broken,
        upload_retry_seconds=0.05,
    )
    stepping(worker, clock)
    result = worker.execute()
    assert result["status"] == "failed" and result["training_status"] == "completed"
    beat, record = heartbeat(worker), json.loads((worker.root / "finalization.json").read_text())
    assert beat["status"] == "failed" and "r2_final" not in beat
    assert beat["pod_id"] == "pod-a" and beat["failure"] == result["failure"]
    if broken == "report":
        assert result["failure"] == "r2_finalization" and record["unverified_milestones"] == []
        return
    # The milestone was retried for the rest of the run and resubmitted at finalization.
    assert result["failure"] == "r2_milestones_unverified" == record["failure"]
    uploader = result["r2"]["uploader"]
    assert uploader["failure_count"] >= 2
    assert {item["label"] for item in uploader["failures"]} == {broken}
    assert broken in record["settle_rounds"][0]["resubmitted"]
    assert record["drained"] is False  # The retrying milestone kept the drain waiting.
    assert result["r2"]["milestones_verified"] == {
        "stage1-final": broken != "stage1-final",
        "stage2-final": broken != "stage2-final",
    }
    # The local export the supervisor pulls over SSH: its path and every file's SHA-256.
    (listed,) = record["unverified_milestones"]
    assert listed == result["r2"]["unverified_milestones"][0]
    assert listed["label"] == broken and listed["export"] == f"exports/{broken}"
    directory = Path(listed["export_path"])
    assert directory == worker.root / "exports" / broken and directory.is_dir()
    stage = result["stages"][0 if broken == "stage1-final" else 1]
    assert listed["checkpoint_sha256"] == stage["final_checkpoint_sha256"]
    names = {"checkpoint.pt", "checkpoint.pt.sha256", "record.json", "updates.jsonl", "ticket.json"}
    assert {row["name"] for row in listed["files"]} == names
    for row in listed["files"]:
        path = Path(row["path"])
        assert path.parent == directory and file_sha256(path) == row["sha256"]
        assert path.stat().st_size == row["size_bytes"]
    assert listed["checkpoint_size_bytes"] == (directory / "checkpoint.pt").stat().st_size
    # report.json still reached R2, saying failed; nothing names the missing label as kept.
    stored = json.loads(bucket.objects[f"{worker.prefix}report.json"])
    assert stored["failure"] == "r2_milestones_unverified"
    assert not any(key.startswith(f"{worker.prefix}{broken}/") for key in bucket.objects)


def test_a_stopped_run_with_an_unverified_milestone_fails_for_the_ssh_pull(
    data: Authored,
) -> None:
    clock = Clock()
    bucket = MemoryBucket()
    bucket.fault = refuse_put("stage1-final/")
    spec = short_window(data).with_limits(rising_steps=5)
    worker = make_worker(
        data, "A", spec=spec, clock=clock, bucket=bucket, upload_retry_seconds=0.05
    )
    stepping(
        worker,
        clock,
        hook=lambda t, m: {**m, "loss": 10.0 + t.step} if worker.stage_name == "stage2" else m,
    )
    result = worker.execute()
    assert result["stop_reason"] == "loss_rising" and result["training_status"] == "stopped"
    assert result["status"] == "failed" and result["failure"] == "r2_milestones_unverified"
    (listed,) = json.loads((worker.root / "finalization.json").read_text())["unverified_milestones"]
    assert listed["label"] == "stage1-final" and "r2_final" not in heartbeat(worker)


def test_a_milestone_that_fails_in_training_is_retried_and_verifies(data: Authored) -> None:
    clock = Clock()
    bucket = MemoryBucket()
    refusals = [0]

    def flaky(method: str, key: str) -> tuple[int, bytes] | None:
        if method == "PUT" and "stage1-final/" in key and refusals[0] < 3:
            refusals[0] += 1
            return 403, b"AccessDenied"  # Not transient: each upload attempt fails at once.
        return None

    bucket.fault = flaky
    worker = make_worker(data, "A", clock=clock, bucket=bucket, upload_retry_seconds=0.01)
    stepping(worker, clock)
    result = worker.execute()
    assert result["status"] == "completed" and refusals[0] == 3
    uploader = result["r2"]["uploader"]
    assert uploader["failure_count"] == 3 and uploader["retrying"] == {}
    assert result["r2"]["milestones_verified"] == {"stage1-final": True, "stage2-final": True}
    assert heartbeat(worker)["status"] == "awaiting_transfer"


def test_finalization_resubmits_a_milestone_waiting_out_its_backoff(data: Authored) -> None:
    clock = Clock()
    bucket = MemoryBucket()
    finalizing = threading.Event()
    bucket.fault = lambda method, key: (
        (403, b"AccessDenied")
        if method == "PUT" and "stage2-final/" in key and not finalizing.is_set()
        else None
    )
    # A 15 s backoff, far longer than the test: only the resubmission sends it again.
    worker = make_worker(data, "A", clock=clock, bucket=bucket, upload_retry_seconds=15.0)
    stepping(worker, clock)
    original = worker.settle

    def settle(budget: float, receipt: dict[str, Any]) -> dict[str, bool]:
        deadline = time.monotonic() + 30
        while worker.uploader.status()["retrying"] != {"stage2-final": 1}:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        finalizing.set()  # R2 recovers while the milestone waits out its backoff.
        return original(budget, receipt)

    worker.settle = settle  # type: ignore[method-assign]
    started = time.monotonic()
    result = worker.execute()
    assert time.monotonic() - started < 15  # It never waited for the backoff.
    assert result["status"] == "completed"
    record = json.loads((worker.root / "finalization.json").read_text())
    assert record["settle_rounds"][0]["resubmitted"] == ["stage2-final"]
    assert record["unverified_milestones"] == [] and heartbeat(worker)["status"] == (
        "awaiting_transfer"
    )


def test_a_head_check_that_keeps_failing_counts_as_unverified(data: Authored) -> None:
    clock = Clock()
    bucket = MemoryBucket()
    broken = threading.Event()
    bucket.fault = lambda method, key: (
        (403, b"") if method == "HEAD" and "stage2-final/" in key and broken.is_set() else None
    )
    worker = make_worker(
        data, "A", spec=short_window(data), clock=clock, bucket=bucket, upload_retry_seconds=0.05
    )
    stepping(worker, clock)
    original = worker.settle

    def settle(budget: float, receipt: dict[str, Any]) -> dict[str, bool]:
        assert worker.uploader.drain(30)  # stage2-final verified by the uploader first.
        broken.set()  # Then R2 stops answering HEAD for it.
        return original(budget, receipt)

    worker.settle = settle  # type: ignore[method-assign]
    result = worker.execute()
    assert result["status"] == "failed" and result["failure"] == "r2_milestones_unverified"
    record = json.loads((worker.root / "finalization.json").read_text())
    assert "HTTP 403" in record["verification_errors"]["stage2-final"]
    assert [row["label"] for row in record["unverified_milestones"]] == ["stage2-final"]
    assert record["settle_rounds"][0]["resubmitted"] == []  # The uploader had verified it.
    assert record["settle_rounds"][1]["resubmitted"] == ["stage2-final"]


def test_rolling_saves_stop_near_the_stage_end(data: Authored) -> None:
    clock = Clock()
    # 100 s per update; the stage's last 4.5 updates fall inside the skip window.
    spec = replace(tiny_spec(data, "B"), rolling_skip_seconds=450.0, expected_tokens_per_second=0.1)
    worker = make_worker(data, "B", spec=spec, clock=clock)
    stepping(worker, clock, hook=lambda t, m: {**m, "step_seconds": 100.0})
    result = worker.execute()
    assert result["status"] == "completed"
    (stage,) = result["stages"]
    updates = stage["planned_updates"]
    rolling = [int(label.split("-step")[1]) for label in stage["r2_labels"] if "-step" in label]
    assert rolling and all((updates - update) * 100 > 450 for update in rolling)
    skipped = stage["rolling_saves_skipped_from_update"]
    assert skipped is not None and (updates - skipped) * 100 <= 450
    # Without the rule a save would have started there: the cadence had passed.
    assert skipped - rolling[-1] >= spec.checkpoint_seconds
    # Before any update the reference speed counts (32 targets at 0.1 tok/s: 320 s each),
    # and the training deadline, where a time-limited stage ends, counts too.
    worker.active = rehearsal_run._Active(0, worker.data.stages[0], 0, stage, worker.root)
    worker.training_deadline = clock.now + 10_000
    assert worker.near_stage_end(updates - 1) and not worker.near_stage_end(updates - 2)
    assert not worker.near_stage_end(1)
    worker.training_deadline = clock.now + 400
    assert worker.near_stage_end(1)


def test_a_tampered_stage_one_checkpoint_is_refused_as_the_stage_two_parent(
    data: Authored, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    worker = make_worker(data, "A", clock=clock)
    stepping(worker, clock)
    original = worker.save

    def corrupt(trainer: Trainer, path: Path, label: str, **kwargs: Any) -> str:
        digest = original(trainer, path, label, **kwargs)
        if label == "stage1-final":
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 0xFF
            path.write_bytes(bytes(raw))
        return digest

    monkeypatch.setattr(worker, "save", corrupt)
    with pytest.raises(ValueError, match="parent checkpoint bytes differ"):
        worker.execute()
    report, beat = finished(worker)
    assert report["status"] == "failed" and beat["status"] == "failed"
    assert not (worker.root / "runs" / "rehearsal-A-stage2").exists()


def test_a_ticket_that_differs_from_the_binding_stops_before_the_next_stage(
    data: Authored,
) -> None:
    spec = tiny_spec(data, "A")
    bound = authored_run_data(spec, data.streams, data.tokenizer)
    first = bound.stages[0]
    common: dict[str, Any] = {
        "initialization": data.initialization,
        "initialization_sha256": data.initialization_sha256,
        "store": MemoryBucket().store(CREDENTIALS),
        "pod_id": "pod-a",
        "pod_created": 1000.0,
    }
    # The loader reads every document, but the bound multiset names other ones.
    wrong = replace(first, expected={**first.expected, "multiset_sha256": "0" * 64})
    clock = Clock()
    worker = Worker(
        spec,
        data.root / "mismatch",
        replace(bound, stages=(wrong, *bound.stages[1:])),
        clock=clock,
        **common,
    )
    stepping(worker, clock)
    result = worker.execute()
    assert result["status"] == "stopped" and result["stop_reason"] == "ticket_mismatch"
    stage1, stage2 = result["stages"]
    assert stage1["ticket"]["complete"] is True and stage1["expected_ticket_match"] is False
    assert stage1["final_checkpoint_sha256"] and stage2["status"] == "pending"
    assert not (worker.root / "runs" / "rehearsal-A-stage2").exists()
    # A loader whose plan differs from the binding is refused before any update.
    other = replace(first, expected={**first.expected, "targets": first.expected["targets"] + 1})
    clock = Clock()
    worker = Worker(
        spec,
        data.root / "plan",
        replace(bound, stages=(other, *bound.stages[1:])),
        clock=clock,
        **common,
    )
    with pytest.raises(ValueError, match="loader's plan differs"):
        worker.execute()
    assert rows(worker) == []


def test_no_secret_value_reaches_any_record(
    data: Authored, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("R2_POD_SECRET_ACCESS_KEY", SECRET)
    monkeypatch.setenv("R2_POD_ACCESS_KEY_ID", KEY_ID)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_pod-scoped-key-9f8e7d6c")
    clock = Clock()
    bucket = MemoryBucket()
    leak = f"SignatureDoesNotMatch AWSAccessKeyId={KEY_ID} secret={SECRET}".encode()
    # One rolling upload fails with an answer that echoes the credentials.
    bucket.fault = lambda method, key: (403, leak) if method == "PUT" and "-step" in key else None
    worker = make_worker(
        data,
        "A",
        clock=clock,
        bucket=bucket,
        secrets=(KEY_ID, SECRET, *environment_secrets()),
    )
    stepping(worker, clock)
    result = worker.execute()
    assert result["status"] == "completed"
    assert result["r2"]["uploader"]["failures"]  # The echo was recorded, redacted.
    assert "[REDACTED]" in (worker.root / "report.json").read_text()
    values = [SECRET.encode(), KEY_ID.encode(), b"rpa_pod-scoped-key-9f8e7d6c"]
    for path in worker.root.rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            assert not any(value in content for value in values), path
    for key, content in bucket.objects.items():
        assert not any(value in content for value in values), key


# Glue between the binding and the worker -----------------------------------------------


SMOKE_PACKING = {
    "sequence_length": 8,
    "microbatch": 2,
    "accumulation": 2,
    "windows_per_update": 4,
    "tokens_per_update": 32,
}
SPLITS = {
    "stage1-web": ("web", "stage1"),
    "stage2-web": ("web", "stage2"),
    "stage2-python": ("python", "train"),
    "validation-web": ("web", "validation"),
    "validation-python": ("python", "validation"),
}


def write_binding(data: Authored) -> tuple[Path, str]:
    """A binding over the authored shards that ``check_binding`` accepts at the tiny packing."""
    root, prefix = data.root / "data", "rehearsal-s-test/"
    tokenizer_path = root / "tokenizer" / "tokenizer.json"
    data.tokenizer.save(tokenizer_path)
    streams: dict[str, Any] = {}
    for name, (domain, split) in SPLITS.items():
        shards: list[dict[str, Any]] = []
        for sha256, shard in data.streams[name]:
            file = shard.path.relative_to(root).as_posix()
            manifest = shard.path.with_name(shard.path.name + ".json")
            shards.append(
                {
                    "file": file,
                    "sha256": sha256,
                    "manifest_sha256": file_sha256(manifest),
                    "size_bytes": shard.path.stat().st_size,
                    "manifest_size_bytes": manifest.stat().st_size,
                    "documents": len(shard),
                    "tokens": sum(length for _, length in shard.offsets),
                    "key": prefix + file,
                    "manifest_key": prefix + file + ".json",
                }
            )
        tokens = sum(shard["tokens"] for shard in shards)
        streams[name] = {
            "domain": domain,
            "split": split,
            "quota_tokens": None if split == "validation" else tokens,
            "tokens": tokens,
            "documents": sum(shard["documents"] for shard in shards),
            "multiset_sha256": multiset_digest(
                document_sha256(np.asarray(shard[i], dtype=np.uint16))
                for _, shard in data.streams[name]
                for i in range(len(shard))
            ),
            "shards": shards,
        }
    runs: dict[str, Any] = {}
    for run, stages in RUNS.items():
        rows = []
        for stage, names in stages:
            held = [shard for name in names for shard in streams[name]["shards"]]
            order = pass_order(
                [(shard["sha256"], shard["documents"]) for shard in held], ORDER_SEED
            )
            rows.append(
                {
                    "name": stage,
                    "streams": list(names),
                    "shards": [shard["file"] for shard in held],
                    "order_seed": ORDER_SEED,
                    "order_sha256": order_digest(order),
                    "documents": sum(shard["documents"] for shard in held),
                    **rehearsal_binding._packed(sum(shard["tokens"] for shard in held)),
                    "multiset_sha256": combine_digests(
                        *(streams[name]["multiset_sha256"] for name in names)
                    ),
                }
            )
        runs[run] = {
            "stages": rows,
            "documents": sum(row["documents"] for row in rows),
            "tokens": sum(row["tokens"] for row in rows),
            "multiset_sha256": combine_digests(*(row["multiset_sha256"] for row in rows)),
        }
    path = data.root / "binding.json"
    binding = {
        "schema_version": 1,
        "kind": rehearsal_binding.KIND,
        "created_at": "2026-10-25T00:00:00+00:00",
        "written_to": str(path),
        "binder": {"git_commit": "f" * 40, "dirty": False},
        "build": {
            "name": "authored",
            **{
                key: "a" * 64
                for key in (
                    "spec_sha256",
                    "report_sha256",
                    "verification_sha256",
                    "tokens_identity",
                    "build_identity",
                    "holdout_sha256",
                )
            },
            "tokens_identity": rehearsal_binding.tokens_identity(
                data.tokenizer.sha256, [*SPLITS], streams
            ),
            "git_commit": "f" * 40,
            "admission": {"approved": True},
            "documents": {"file": "documents.jsonl", "sha256": "b" * 64},
            "exclusions": {"file": "exclusions.jsonl", "sha256": "c" * 64},
        },
        "r2_prefix": prefix,
        "packing": SMOKE_PACKING,
        "tokenizer": {
            "path": "tokenizer/tokenizer.json",
            "source": str(tokenizer_path),
            "key": prefix + "tokenizer/tokenizer.json",
            "sha256": file_sha256(tokenizer_path),
            "size_bytes": tokenizer_path.stat().st_size,
            "identity": data.tokenizer.sha256,
            "vocab_size": data.tokenizer.vocab_size,
            "bos_id": 256,
        },
        "stream_order": list(SPLITS),
        "streams": streams,
        "runs": runs,
        "same_multiset": {run: runs[run]["multiset_sha256"] for run in RUNS},
        "validation": {"streams": list(VALIDATION_STREAMS), "subset_rule": SUBSET_RULE},
    }
    rehearsal_binding.check_binding(binding)
    path.write_bytes(rehearsal_binding._serialized(binding))
    return path, file_sha256(path)


@pytest.fixture
def tiny_binding(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The binding rules at the authored data's vocabulary and packing."""
    monkeypatch.setattr(rehearsal_binding, "VOCAB_SIZE", 272)
    monkeypatch.setattr(rehearsal_binding, "SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(rehearsal_binding, "PACKING", SMOKE_PACKING)
    yield


def test_load_run_data_pairs_a_verified_binding_with_the_specification(
    data: Authored, tiny_binding: None, capsys: pytest.CaptureFixture[str]
) -> None:
    path, sha256 = write_binding(data)
    for run in RUNS:
        spec = tiny_spec(data, run)
        loaded = load_run_data(path, data.root / "data", expected_sha256=sha256, spec=spec)
        authored_data = authored_run_data(spec, data.streams, data.tokenizer)
        assert loaded.multiset_sha256 == authored_data.multiset_sha256
        for left, right in zip(loaded.stages, authored_data.stages, strict=True):
            assert left.expected == right.expected  # The binding's ticket is the shards' own.
            assert np.array_equal(left.order, right.order)
            assert [(n, s) for n, _, s in left.shards] == [(n, s) for n, _, s in right.shards]
        assert [label for label, _ in loaded.validation] == ["web", "python"]
    # The loaded data trains end to end.
    clock = Clock()
    spec = tiny_spec(data, "B")
    worker = Worker(
        spec,
        data.root / "bound-B",
        load_run_data(path, data.root / "data", expected_sha256=sha256, spec=spec),
        initialization=data.initialization,
        initialization_sha256=data.initialization_sha256,
        store=MemoryBucket().store(CREDENTIALS),
        pod_id="pod-b",
        pod_created=1000.0,
        clock=clock,
    )
    stepping(worker, clock)
    assert worker.execute()["status"] == "completed"
    # A specification that plans other updates than the binding is refused.
    spec = smoke_specification("B", data.streams)
    (stage,) = spec.stages
    longer = replace(
        stage, updates=stage.updates + 5, warmup_updates=round(0.2 * (stage.updates + 5))
    )
    other = replace(
        spec,
        stages=(longer,),
        train=replace(spec.train, steps=longer.updates, warmup_steps=longer.warmup_updates),
    )
    with pytest.raises(ValueError, match="plans"):
        load_run_data(path, data.root / "data", expected_sha256=sha256, spec=other)
    # The Mac-side writer takes the tokenizer identity from the pinned binding.
    files = []
    for run in RUNS:
        file = data.root / f"spec-{run}.json"
        file.write_text(json.dumps(smoke_specification(run, data.streams).to_dict()))
        assert Specification.load(file) == smoke_specification(run, data.streams)
        files += ["--specification", str(file)]
    output = data.root / "cli-init.pt"
    command = ["write-initialization", *files, "--binding", str(path), "--output", str(output)]
    assert main([*command, "--binding-sha256", sha256]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["sha256"] == file_sha256(output) == data.initialization_sha256
    assert printed["tokenizer_identity"] == data.tokenizer.sha256 and printed["seed"] == 0
    assert printed["parameters"] == expected_parameters(spec.model)
    with pytest.raises(ValueError, match="SHA-256"):
        main([*command[:-1], str(data.root / "other.pt"), "--binding-sha256", "0" * 64])


def test_run_command_refuses_mismatched_receipts_before_creating_output(tmp_path: Path) -> None:
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}")
    readiness = tmp_path / "receipt.json"
    output = tmp_path / "out"
    arguments = [
        "run",
        "--specification",
        str(SPECS["A"]),
        "--binding",
        str(tmp_path / "binding.json"),
        "--binding-sha256",
        "a" * 64,
        "--root",
        str(tmp_path),
        "--initialization",
        str(tmp_path / "init.pt"),
        "--initialization-sha256",
        "b" * 64,
        "--readiness",
        str(readiness),
        "--preflight",
        str(preflight),
        "--output",
        str(output),
    ]
    good = {
        "kind": "rehearsal-readiness",
        "run": "A",
        "specification_sha256": file_sha256(SPECS["A"]),
        "binding_sha256": "a" * 64,
        "preflight_sha256": file_sha256(preflight),
        "initialization_sha256": "b" * 64,
    }
    for change, message in (
        ({"kind": "readiness"}, "not a rehearsal-readiness receipt"),
        ({"run": "B"}, "not a rehearsal-readiness receipt for run A"),
        ({"specification_sha256": "0" * 64}, "specification differs"),
        ({"binding_sha256": "0" * 64}, "binding identity"),
        ({"preflight_sha256": "0" * 64}, "preflight report"),
        ({"initialization_sha256": "0" * 64}, "initialization identity"),
    ):
        readiness.write_text(json.dumps({**good, **change}))
        with pytest.raises(ValueError, match=message):
            main(arguments)
        assert not output.exists()


def test_run_command_requires_the_receipts_code_revision_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other gate is stubbed; the revision decides, and nothing starts without it."""
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}")
    readiness, revision = tmp_path / "receipt.json", tmp_path / "revision.json"
    output = tmp_path / "out"
    arguments = ["run", "--specification", str(SPECS["A"]), "--binding", str(tmp_path / "b")]
    arguments += ["--binding-sha256", "a" * 64, "--root", str(tmp_path)]
    arguments += ["--initialization", str(tmp_path / "init.pt")]
    arguments += ["--initialization-sha256", "b" * 64, "--readiness", str(readiness)]
    arguments += ["--preflight", str(preflight), "--output", str(output)]
    identity = {
        "specification_sha256": file_sha256(SPECS["A"]),
        "binding_sha256": "a" * 64,
        "preflight_sha256": file_sha256(preflight),
        "initialization_sha256": "b" * 64,
    }
    spec = Specification.load(SPECS["A"])
    launch = Launch(1000.0, 1000.0 + 44_820, "pod123", spec, {"rule": LAUNCH_RULE})
    started: dict[str, Any] = {}

    class Started:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            started.update(kwargs)

        def execute(self) -> dict[str, Any]:
            return {"run": "A", "status": "completed"}

    stubs: list[tuple[str, Any]] = [
        ("load_binding", lambda *_, **__: BOUND),
        ("check_readiness", lambda *_, **__: launch),
        ("training_device", lambda _: torch.device("cpu")),
        ("hardware_description", lambda _: {"name": "NVIDIA A40"}),
        ("check_preflight", lambda *_, **__: None),
        ("load_run_data", lambda *_, **__: SimpleNamespace(tokenizer=SimpleNamespace(sha256="t"))),
        ("verify_initialization", lambda *_, **__: None),
        ("Store", lambda credentials: None),
        ("Worker", Started),
    ]
    for name, stub in stubs:
        monkeypatch.setattr(rehearsal_run, name, stub)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    for variable, setting in (
        ("R2_ENDPOINT", "https://acct.r2.cloudflarestorage.com"),
        ("R2_BUCKET", "rehearsal"),
        ("R2_POD_ACCESS_KEY_ID", KEY_ID),
        ("R2_POD_SECRET_ACCESS_KEY", SECRET),
    ):
        monkeypatch.setenv(variable, setting)
    readiness.write_text(json.dumps({**receipt(**identity), "revision": REVISION}))
    with pytest.raises(ValueError, match="cannot read the shipped code revision"):
        main(arguments)  # The default /workspace/rehearsal/revision.json is not here.
    arguments += ["--revision", str(revision)]
    for shipped, recorded, message in (
        ({**REVISION, "git_commit": "1" * 40}, REVISION, "differs from the readiness"),
        (REVISION, {**REVISION, "clean": False}, "revision of a clean tree"),
        (REVISION, None, "revision of a clean tree"),
    ):
        revision.write_text(json.dumps(shipped))
        value = receipt(**identity)
        if recorded is not None:
            value["revision"] = recorded
        readiness.write_text(json.dumps(value))
        with pytest.raises(ValueError, match=message):
            main(arguments)
        assert not started and not output.exists()
    revision.write_text(json.dumps(REVISION))
    readiness.write_text(json.dumps({**receipt(**identity), "revision": REVISION}))
    assert main(arguments) == 0
    assert started["extra"]["code_revision"] == REVISION  # report.json's code_revision.
    assert DEFAULT_REVISION == Path("/workspace/rehearsal/revision.json")
    # The checks after the final drain get single-attempt stores with the timeout asked for;
    # the worker retries them itself while time lasts.
    bounded = started["check_store_for"](7.5)
    assert bounded.attempts == 1 and bounded.transport.keywords == {"timeout": 7.5}


def test_the_report_names_the_shipped_commit_not_the_pods_missing_git(data: Authored) -> None:
    worker = make_worker(data, "B", extra={"code_revision": REVISION})
    assert worker.report["git_commit"] == REVISION["git_commit"]
    assert worker.report["dirty"] is False
    worker.uploader.close()


def test_cli_smoke_runs_both_runs_offline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["smoke", "--output", str(tmp_path / "smoke")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == {"A": "completed", "B": "completed"}
    assert summary["same_multiset"] is True and summary["tickets_match"] is True
    for run in RUNS:
        beat = json.loads((tmp_path / "smoke" / f"run-{run}" / "heartbeat.json").read_text())
        assert beat["status"] == "awaiting_transfer"


def test_length_sorted_batches_keep_every_window_once() -> None:
    windows = [ValidationWindow(tuple(range(1, 3 + n % 5)), 1) for n in range(11)]
    result = list(length_sorted(windows, 4))
    assert sorted(map(len_tokens, result)) == sorted(map(len_tokens, windows))
    for start in range(0, 11, 4):
        chunk = [len(window.tokens) for window in result[start : start + 4]]
        assert chunk == sorted(chunk)


def len_tokens(window: ValidationWindow) -> int:
    return len(window.tokens)


def test_worker_refuses_mismatched_inputs(data: Authored) -> None:
    spec = tiny_spec(data, "A")
    run_data = authored_run_data(spec, data.streams, data.tokenizer)
    common: dict[str, Any] = {
        "initialization": data.initialization,
        "initialization_sha256": data.initialization_sha256,
        "store": MemoryBucket().store(CREDENTIALS),
        "pod_created": 1000.0,
    }
    with pytest.raises(ValueError, match="stages differ"):
        Worker(tiny_spec(data, "B"), data.root / "x", run_data, pod_id="p", **common)
    with pytest.raises(ValueError, match="Pod ID"):
        Worker(spec, data.root / "x", run_data, pod_id="../p", **common)
    (data.root / "exists").mkdir()
    with pytest.raises(FileExistsError):
        Worker(spec, data.root / "exists", run_data, pod_id="p", **common)
    bigger = replace(
        spec, model=ModelConfig(vocab_size=300, width=16, layers=1, heads=2, context=8)
    )
    with pytest.raises(ValueError, match="vocabulary"):
        Worker(bigger, data.root / "y", run_data, pod_id="p", **common)
