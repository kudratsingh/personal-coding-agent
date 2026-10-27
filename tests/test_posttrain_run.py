"""The post-training worker for run 1: stages, artifacts, stops and launch gates, on CPU.

Every run uses the smoke's fixtures (a tiny model, authored tasks exported for the Pod,
fixture benchmarks and ARC questions) and an in-memory bucket. Nothing touches a network,
and no generated code runs.
"""

import copy
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from scratchcoder.engine.generation import generate_many
from scratchcoder.eval.completions import read_completions, validate_against_tasks
from scratchcoder.eval.export import WITHHELD
from scratchcoder.model.gpt import GPT
from scratchcoder.rl.attempts import read_attempts_file
from scratchcoder.rl.command import check_attempts_header, read_candidates
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.train.pilot import PilotStop
from tools import posttrain_approved, posttrain_ops
from tools.posttrain_run import (
    PROBE_STOP,
    PROJECTED_STOP,
    RESERVE_STOP,
    SPEED_STOP,
    PosttrainWorker,
    Rates,
    RunLimits,
    Specification,
    check_readiness,
    launch_fit,
    planned_seconds,
    ticking_generate,
)
from tools.posttrain_smoke import (
    SMOKE_POD_ID,
    Fixtures,
    fixtures,
    make_worker,
    patched_benchmarks,
    smoke_model,
)
from tools.rehearsal_run import MemoryBucket

SPEC = posttrain_approved.SPECIFICATION_PATH
STAGES = ["probe", "arc-base", "eval-base", "sft", "arc-sft", "eval-sft", "screen"]


@pytest.fixture(scope="module")
def found(tmp_path_factory: pytest.TempPathFactory) -> Fixtures:
    torch.set_num_threads(1)
    return fixtures(tmp_path_factory.mktemp("posttrain-fixtures"))


@dataclass
class Ran:
    report: dict[str, Any]
    root: Path
    bucket: MemoryBucket
    heartbeat: dict[str, Any]


def reference_encoder(self: ByteBPE, text: str, *, allow_special: bool = False) -> list[int]:
    raise AssertionError("the reference BPE encoder ran on the worker's hot path")


def execute(
    found: Fixtures,
    root: Path,
    *,
    changes: Callable[[dict[str, Any]], None] | None = None,
    **options: Any,
) -> Ran:
    """Run a worker; the reference encoder is disabled throughout: at 65,536 tokens it takes
    about a second a prompt, so every prompt path must use the cached ``FastEncoder``."""
    bucket = MemoryBucket()
    worker = make_worker(found, root, bucket=bucket, changes=changes, **options)
    with patched_benchmarks(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(ByteBPE, "encode", reference_encoder)
        report = worker.execute()
    heartbeat = json.loads((root / "heartbeat.json").read_text())
    return Ran(report, root, bucket, heartbeat)


@pytest.fixture(scope="module")
def completed(found: Fixtures, tmp_path_factory: pytest.TempPathFactory) -> Ran:
    return execute(found, tmp_path_factory.mktemp("posttrain-run") / "run")


def statuses(report: dict[str, Any]) -> dict[str, str]:
    return {stage["name"]: stage["status"] for stage in report["stages"]}


# The specification of record ----------------------------------------------------------------


def test_the_specification_file_is_the_approved_run_one() -> None:
    raw = json.loads(SPEC.read_text())
    assert raw == posttrain_approved.run1_specification()
    spec = Specification.from_dict(raw)
    posttrain_approved.check_run1(spec.raw)
    assert [stage.name for stage in spec.stages] == STAGES
    assert spec.sft_updates == 106 and spec.slots_per_update == 32_768
    assert spec.inputs["b-mixed-final"]["sha256"].startswith("56339e3f")
    derived = spec.ceiling
    assert (derived.worker_cost_stop_usd, derived.watchdog_cost_stop_usd) == (3.0, 2.99)
    assert (derived.wall_seconds, derived.self_destruct_seconds) == (18_000, 17_820)
    text = SPEC.read_text()
    assert "private/" not in text and "D-3" not in text


@pytest.mark.parametrize(
    "path, value",
    [
        (("limits", "ceiling_usd"), 3.5),
        (("probe", "minimum_tokens_per_second", "64"), 4_000.0),
        (("sft", "learning_rate"), 1e-4),
        (("screening", "samples"), 4),
        (("stages", 2, "files", 0, "protocol"), "chat-v1"),
        (("inputs", "b-mixed-final", "sha256"), "0" * 64),
        (("planning", "generation_tokens_per_second", "64"), 9_000.0),
    ],
)
def test_the_receipt_check_refuses_any_unapproved_value(path: tuple[Any, ...], value: Any) -> None:
    raw = posttrain_approved.run1_specification()
    target: Any = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="differs from the approved run 1"):
        posttrain_approved.check_run1(raw)


def test_the_pool_test_pin_may_be_filled_only_at_the_go() -> None:
    raw = posttrain_approved.run1_specification()
    raw["inputs"]["pool-test"].update(size_bytes=10, sha256="a" * 64)
    posttrain_approved.check_run1(raw)
    raw["inputs"]["pool-dev"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="inputs.pool-dev.sha256"):
        posttrain_approved.check_run1(raw)


def test_the_launch_plan_fits_the_ceiling_with_the_setup_allowance() -> None:
    spec = Specification.load(SPEC)
    plan = planned_seconds(spec, Rates.planning(spec))
    # Base and SFT evaluation price 18.6M and 18.8M tokens at cap at 5,000 tok/s.
    assert plan["stages"]["eval-base"] == pytest.approx(7 * 20 + 18_612_480 / 5_000)
    assert plan["stages"]["eval-sft"] == pytest.approx(7 * 20 + 18_792_960 / 5_000)
    assert plan["stages"]["screen"] == pytest.approx(20 + 1_000 * 8 * 256 / 5_000)
    lifetime = spec.ceiling.self_destruct_seconds
    fit = launch_fit(spec, remaining_seconds=lifetime - spec.limits["startup_seconds"])
    assert fit["fits_at_planning"] and fit["planning_seconds_needed"] < 10_100
    assert not launch_fit(spec, remaining_seconds=9_000)["fits_at_planning"]


# A completed run ------------------------------------------------------------------------------


def test_every_stage_completes_and_every_artifact_reaches_r2(completed: Ran) -> None:
    report = completed.report
    assert report["status"] == "completed" and statuses(report) == dict.fromkeys(
        STAGES, "completed"
    )
    prefix = posttrain_ops.run_prefix("1", SMOKE_POD_ID)
    objects = completed.bucket.objects
    assert all(key.startswith(prefix) for key in objects)  # The prefix guard held.
    for artifact in report["artifacts"]:
        assert completed.bucket.digests[artifact["key"]] == artifact["sha256"]
        assert len(objects[artifact["key"]]) == artifact["bytes"]
    assert prefix + "report.json" in objects
    labels = {artifact["label"] for artifact in report["artifacts"]}
    assert {"probe", "sft", "sft-weights", "screen", "arc-sft-sft"} <= labels
    assert sum(label.startswith("eval-base-") for label in labels) == 7
    assert sum(label.startswith("eval-sft-") for label in labels) == 7
    for stage in report["stages"]:
        assert stage["outputs"] and stage["seconds"] is not None
        assert all(len(output["sha256"]) == 64 for output in stage["outputs"])
    final = completed.heartbeat
    assert final["status"] == "awaiting_transfer"
    assert final["r2_final"]["artifact_count"] == len(report["artifacts"])


def test_the_r2_report_passes_the_macs_final_verification(completed: Ran, tmp_path: Path) -> None:
    r2_final = completed.heartbeat["r2_final"]
    result = posttrain_ops.verify_final(
        completed.bucket.store(),
        r2_final,
        run="1",
        pod_id=SMOKE_POD_ID,
        destination=tmp_path / "verify",
    )
    assert result["status"] == "verified" and result["report_status"] == "completed"
    assert len(result["artifacts"]) == r2_final["artifact_count"]
    tampered = MemoryBucket()
    tampered.objects = dict(completed.bucket.objects)
    tampered.digests = dict(completed.bucket.digests)
    key = completed.report["artifacts"][3]["key"]
    tampered.digests[key] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        posttrain_ops.verify_final(
            tampered.store(), r2_final, run="1", pod_id=SMOKE_POD_ID, destination=tmp_path / "t"
        )


def test_no_hidden_test_reaches_the_pod_and_the_mac_binds_every_output(
    completed: Ran, found: Fixtures
) -> None:
    hidden = []
    for name in ("dev", "test", "train"):
        for line in found.full_tasks[name].read_text().splitlines():
            tests = json.loads(line)["hidden_tests"]
            hidden += [tests, json.dumps(tests)[1:-1]]  # As written, and JSON-escaped.
    # Everything the Pod holds: its inputs and everything the worker wrote.
    pod_files = [*found.inputs.values(), *(p for p in completed.root.rglob("*") if p.is_file())]
    texts = [path.read_bytes().decode(errors="replace") for path in pod_files]
    for tests in hidden:
        assert all(tests not in text for text in texts)
    assert WITHHELD in found.inputs["pool-test"].read_text()  # Placeholders only.
    assert all(
        tests.encode() not in data for data in completed.bucket.objects.values() for tests in hidden
    )
    # The Mac binds the Pod's completions to its full files by re-deriving the exports.
    for name in ("pool-dev-greedy", "pool-test-n8"):
        path = completed.root / "exports" / f"eval-sft-{name}" / "completions.jsonl"
        split = name.split("-")[1]
        tasks = validate_against_tasks(read_completions(path), found.full_tasks[split])
        assert len(tasks) == 3
    attempts_path = completed.root / "exports" / "screen" / "attempts.jsonl"
    header, attempts = read_attempts_file(attempts_path)
    assert header is not None and header.attempts == len(attempts) == 4 * 8
    candidates = found.inputs["rl-candidates-list"]
    tasks = read_candidates(found.full_tasks["train"], candidates)
    identity = check_attempts_header(
        header, attempts, tasks, tasks_path=found.full_tasks["train"], candidates_path=candidates
    )
    assert identity == "export"


def test_the_mac_fetches_every_artifact_pinned_from_the_verified_report(
    completed: Ran, tmp_path: Path
) -> None:
    report = completed.root / "report.json"
    store = completed.bucket.store()
    fetched = posttrain_ops.fetch_artifacts(
        store, report, tmp_path / "artifacts", receipt=tmp_path / "fetch.json", labels=["eval-"]
    )
    assert len(fetched["files"]) == 28  # 14 completions files and their records.
    for row in fetched["files"]:
        assert Path(row["path"]).read_bytes() == completed.bucket.objects[row["key"]]
    again = posttrain_ops.fetch_artifacts(
        store, report, tmp_path / "artifacts", receipt=tmp_path / "fetch2.json", labels=["eval-"]
    )
    assert {row["method"] for row in again["files"]} == {"present"}


def test_the_benchmark_releases_are_the_pinned_ones(tmp_path: Path) -> None:
    from scratchcoder.data import evalplus, humaneval
    from tools.posttrain_binding import benchmark_releases

    fixtures_by_size: dict[int, bytes] = {}

    def fake_download(manifest: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fixtures_by_size.get(manifest.size_bytes, b"not the release"))

    with pytest.raises(ValueError, match="not the pinned humaneval release"):
        benchmark_releases(tmp_path / "bench", download=fake_download)
    assert humaneval.DATA_BYTES == 44_877 and evalplus.release("mbpp-plus").task_count == 378


def test_the_smoke_tokenizer_merges_so_the_encoders_are_exercised(found: Fixtures) -> None:
    from scratchcoder.data import chat
    from scratchcoder.eval.tasks import load_tasks
    from scratchcoder.tokenizer.bulk import FastEncoder

    assert len(found.tokenizer.merges) > 0
    task = next(iter(load_tasks(found.inputs["pool-dev"])))
    fast, reference = FastEncoder(found.tokenizer), found.tokenizer
    assert chat.task_prompt(fast, task) == chat.task_prompt(reference, task)
    text = chat.task_user_text(task)
    assert len(reference.encode(text)) < len(text.encode())  # Real merges applied.


def test_sft_keeps_the_lowest_quarter_dev_loss_as_weights_only(completed: Ran) -> None:
    sft = completed.report["sft"]
    quarters = sft["quarters"]
    assert [entry["update"] for entry in quarters] == [1, 2, 3]
    lowest = min(quarters, key=lambda entry: entry["dev_loss"])
    assert sft["selected"]["update"] == lowest["update"]
    assert sft["speed_check"]["status"] == "passed"
    weights = completed.root / "exports" / "sft-weights" / "checkpoint.pt"
    state = torch.load(weights, map_location="cpu", weights_only=True)
    assert state["format"] == "weights-v1" and "optimizer" not in state
    assert state["step"] == lowest["update"]
    screen = completed.root / "exports" / "screen" / "attempts.jsonl"
    header, _ = read_attempts_file(screen)
    assert header is not None and header.checkpoint_sha256 == sft["selected"]["sha256"]


def test_records_stay_within_their_bounds(completed: Ran) -> None:
    assert (completed.root / "heartbeat.json").stat().st_size < 65_536
    assert (completed.root / "report.json").stat().st_size < 1_000_000
    assert completed.report["projections"] and all(
        row["fits"] for row in completed.report["projections"]
    )
    # The projected-finish rule runs at every stage's start, generation or not.
    assert [row["stage"] for row in completed.report["projections"]] == STAGES
    assert completed.report["projections"][1]["basis"]["generation"] == "measured"
    observed = completed.report["generation_observed"]
    assert observed["projection_tokens_per_second"] == min(
        observed["probe_tokens_per_second"], observed["cap_tokens_per_second"]
    )


# Stops ----------------------------------------------------------------------------------------


def test_a_probe_below_the_planning_speed_stops_the_run_after_uploading(
    found: Fixtures, tmp_path: Path
) -> None:
    def slow(raw: dict[str, Any]) -> None:
        raw["probe"]["minimum_tokens_per_second"] = {"4": 1e12}

    ran = execute(found, tmp_path / "run", changes=slow)
    assert (ran.report["status"], ran.report["stop_reason"]) == ("stopped", PROBE_STOP)
    assert statuses(ran.report) == {"probe": "stopped", **dict.fromkeys(STAGES[1:], "skipped")}
    assert ran.report["probe"]["passed"] is False
    assert ran.report["probe"]["measured_tokens_per_second"]["4"] > 0
    probe = ran.report["stages"][0]  # A stopped stage still names its seconds and outputs.
    assert probe["seconds"] is not None and {row["name"] for row in probe["outputs"]} == {
        "generation-benchmark.json",
        "record.json",
    }
    prefix = posttrain_ops.run_prefix("1", SMOKE_POD_ID)
    assert prefix + "probe/generation-benchmark.json" in ran.bucket.objects
    assert prefix + "report.json" in ran.bucket.objects
    assert ran.heartbeat["status"] == "stopped" and "r2_final" in ran.heartbeat


def test_sft_below_the_speed_floor_stops_the_run(found: Fixtures, tmp_path: Path) -> None:
    def slow(raw: dict[str, Any]) -> None:
        raw["sft"]["reference_slots_per_second"] = 1e12

    ran = execute(found, tmp_path / "run", changes=slow)
    assert (ran.report["status"], ran.report["stop_reason"]) == ("stopped", SPEED_STOP)
    assert statuses(ran.report)["sft"] == "stopped"
    assert statuses(ran.report)["eval-base"] == "completed"
    assert ran.report["sft"]["speed_check"]["status"] == "failed"
    prefix = posttrain_ops.run_prefix("1", SMOKE_POD_ID)
    assert prefix + "eval-base-pool-dev-greedy/completions.jsonl" in ran.bucket.objects


def test_a_stage_that_cannot_finish_never_starts(
    found: Fixtures, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def huge(raw: dict[str, Any]) -> None:
        # At the planning rate the whole run fits, so the probe starts; at the probe's
        # measured CPU rate the evaluation of the SFT weights would take days.
        raw["planning"]["generation_tokens_per_second"] = {"4": 1e12}
        for entry in raw["stages"][5]["files"]:  # eval-sft
            entry["planned_tasks"] = 10**6

    # Only this test prices files at more tasks than they hold; the count check would
    # otherwise refuse the specification before any stage.
    monkeypatch.setattr(PosttrainWorker, "check_planned_tasks", lambda self: None)
    ran = execute(found, tmp_path / "run", changes=huge)
    assert (ran.report["status"], ran.report["stop_reason"]) == ("stopped", PROJECTED_STOP)
    # With the measured rate the work remaining from arc-base on cannot finish, so arc-base
    # never starts, though arc-base itself would fit.
    assert statuses(ran.report)["probe"] == "completed"
    assert statuses(ran.report)["arc-base"] == "stopped"
    projection = ran.report["projections"][-1]
    assert projection["stage"] == "arc-base" and projection["fits"] is False
    assert projection["projected_finish_unix"] > projection["pod_end_unix"]
    assert projection["basis"]["generation"] == "measured"
    assert not any(key.split("/")[3].startswith(("arc-", "eval-")) for key in ran.bucket.objects)


class ShiftedClock:
    """Real time plus an offset a test moves, so a stage can be made to run late."""

    def __init__(self) -> None:
        self.offset = 0.0

    def __call__(self) -> float:
        return time.time() + self.offset


def test_a_stage_still_running_at_the_reserve_stops_and_finalizes_in_full(
    found: Fixtures, tmp_path: Path
) -> None:
    """The worker's in-stage stop is the finalization reserve, not the in-Pod deadline: the
    stop leaves the drain, report.json and r2_final their time."""
    clock = ShiftedClock()
    bucket = MemoryBucket()
    spec = Specification.load(SPEC)
    worker = make_worker(
        found,
        tmp_path / "run",
        bucket=bucket,
        clock=clock,
        pod_created=clock(),
        limits=RunLimits.of(spec, pod_seconds=spec.ceiling.self_destruct_seconds),
    )
    assert worker.reserve_unix == worker.pod_end - 900
    original = worker.run_eval

    def late(stage: Any) -> Any:
        # The stage was allowed to start; now generation runs slower than projected.
        clock.offset = worker.reserve_unix - time.time() + 1
        return original(stage)

    worker.run_eval = late  # type: ignore[method-assign]
    with patched_benchmarks():
        report = worker.execute()
    assert (report["status"], report["stop_reason"]) == ("stopped", RESERVE_STOP)
    assert statuses(report)["eval-base"] == "stopped"
    assert statuses(report)["sft"] == "skipped"
    finalization = json.loads((tmp_path / "run" / "finalization.json").read_text())
    assert finalization["drain_budget_seconds"] > 0  # Not the wall or cost stop's hurry.
    assert finalization["r2_final"] is not None  # Every label made so far verified.
    heartbeat = json.loads((tmp_path / "run" / "heartbeat.json").read_text())
    assert heartbeat["r2_final"]["artifact_count"] == len(report["artifacts"])
    assert posttrain_ops.run_prefix("1", SMOKE_POD_ID) + "report.json" in bucket.objects
    with pytest.raises(PilotStop, match=RESERVE_STOP):
        worker.tick()
    worker.stop_reason = None
    assert worker.stopped() and worker.stop_reason == RESERVE_STOP


def test_a_late_stage_that_is_not_generation_never_starts(found: Fixtures, tmp_path: Path) -> None:
    """The projected-finish rule runs before every stage: SFT after a slow eval-base."""
    clock = ShiftedClock()
    worker = make_worker(found, tmp_path / "run", bucket=MemoryBucket(), clock=clock)
    original = worker.run_eval

    def slow(stage: Any) -> Any:
        outputs = original(stage)
        if stage.name == "eval-base":
            # The stage ran long: it ends, its GPU work just done, a second before the reserve.
            clock.offset = worker.reserve_unix - time.time() - 1
            worker.last_activity = clock()
        return outputs

    worker.run_eval = slow  # type: ignore[method-assign]
    with patched_benchmarks():
        report = worker.execute()
    assert (report["status"], report["stop_reason"]) == ("stopped", PROJECTED_STOP)
    assert statuses(report)["eval-base"] == "completed"
    assert statuses(report)["sft"] == "stopped"
    assert report["projections"][-1]["stage"] == "sft"
    assert report["projections"][-1]["fits"] is False


def test_slow_evaluation_files_re_base_the_projection(found: Fixtures, tmp_path: Path) -> None:
    worker = make_worker(found, tmp_path / "run", bucket=MemoryBucket())
    worker.probe_rates = {4: 1_000.0}
    entry = worker.plan.stage("eval-base").files[0]
    worker.observe_generation(entry, entry.planned_tokens / 10.0)  # 10 cap tokens a second.
    assert worker.rates.generation_tokens_per_second == pytest.approx(10.0)
    assert worker.rates.basis["generation"] == "observed-evaluation"
    worker.observe_generation(entry, entry.planned_tokens / 100_000.0)  # A fast file.
    # The rate over both files (about 20 a second) stays below the probe's 1,000.
    assert worker.rates.generation_tokens_per_second == pytest.approx(
        2 * entry.planned_tokens / (entry.planned_tokens / 10.0 + entry.planned_tokens / 1e5)
    )
    worker.uploader.close(timeout=5)


def test_a_task_file_of_another_size_than_planned_is_refused_before_any_stage(
    found: Fixtures, tmp_path: Path
) -> None:
    def misplanned(raw: dict[str, Any]) -> None:
        raw["stages"][2]["files"][0]["planned_tasks"] = 2_000

    worker = make_worker(found, tmp_path / "run", bucket=MemoryBucket(), changes=misplanned)
    with patched_benchmarks(), pytest.raises(ValueError, match="plans 2000"):
        worker.execute()
    assert statuses(worker.report) == dict.fromkeys(STAGES, "skipped")
    assert worker.report["status"] == "failed"


def test_a_stopped_stage_records_its_seconds_and_outputs(found: Fixtures, tmp_path: Path) -> None:
    def slow(raw: dict[str, Any]) -> None:
        raw["sft"]["reference_slots_per_second"] = 1e12

    ran = execute(found, tmp_path / "run", changes=slow)
    stopped = next(stage for stage in ran.report["stages"] if stage["name"] == "sft")
    assert stopped["status"] == "stopped" and stopped["seconds"] is not None
    probe = next(stage for stage in ran.report["stages"] if stage["name"] == "probe")
    assert [output["name"] for output in probe["outputs"]] == [
        "generation-benchmark.json",
        "record.json",
    ]


class SteppingClock:
    """A clock that advances on every reading, so a limit falls at a fixed point."""

    def __init__(self, step: float) -> None:
        self.now, self.step = time.time(), step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


@pytest.mark.parametrize("limit", ["cost_limit", "pod_wall_limit"])
def test_wall_and_cost_stops_end_the_run_without_waiting_for_uploads(
    found: Fixtures, tmp_path: Path, limit: str
) -> None:
    clock = SteppingClock(1.0)
    spec = Specification.load(SPEC)
    limits = RunLimits.of(spec)
    if limit == "cost_limit":
        limits = replace(limits, quoted_total_hourly=3.0 / 100 * 3600)  # $3.00 after 100 s.
    else:
        limits = replace(limits, pod_seconds=100.0)

    def fast(raw: dict[str, Any]) -> None:
        # Projections priced near zero (and no finalization reserve) leave only the limit.
        planning = raw["planning"]
        planning["generation_tokens_per_second"] = {"4": 1e12}
        for name in ("training_slots_per_second", "forward_tokens_per_second"):
            planning[name] = 1e12
        for name in (
            "probe_seconds",
            "arc_seconds_per_checkpoint",
            "checkpoint_load_seconds",
            "compile_seconds",
            "weights_save_seconds",
        ):
            planning[name] = 1e-6
        raw["limits"]["finalization_reserve_seconds"] = 1e-6

    ran = execute(
        found, tmp_path / "run", changes=fast, limits=limits, clock=clock, pod_created=clock()
    )
    assert (ran.report["status"], ran.report.get("stop_reason")) == ("stopped", limit)
    finalization = json.loads((ran.root / "finalization.json").read_text())
    assert finalization["drain_budget_seconds"] == 0.0 and finalization["r2_final"] is None
    assert "stopped" in statuses(ran.report).values()


def test_the_prefix_guard_refuses_writes_and_deletes_outside_the_run(
    found: Fixtures, tmp_path: Path
) -> None:
    bucket = MemoryBucket()
    bucket.objects["rehearsal-runs/B/x/mixed-final/checkpoint.pt"] = b"parent"
    bucket.digests["rehearsal-runs/B/x/mixed-final/checkpoint.pt"] = "e" * 64
    worker = make_worker(found, tmp_path / "run", bucket=bucket)
    probe = tmp_path / "probe.bin"
    probe.write_bytes(b"x")
    assert isinstance(worker.store, posttrain_ops.GuardedStore)  # The worker's only store.
    digest = "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"  # sha256(b"x")
    for key in (
        "rehearsal-runs/B/x/mixed-final/checkpoint.pt",
        "posttrain-runs/1/other-pod/report.json",
        posttrain_ops.run_prefix("1", SMOKE_POD_ID) + "../escape",
    ):
        with pytest.raises(posttrain_ops.PrefixViolation):
            worker.store.put_known(key, probe, digest, 1)
        with pytest.raises(posttrain_ops.PrefixViolation):
            worker.store.delete(key)
    inside = posttrain_ops.run_prefix("1", SMOKE_POD_ID) + "probe/x.bin"
    worker.store.put_known(inside, probe, digest, 1)
    assert bucket.objects["rehearsal-runs/B/x/mixed-final/checkpoint.pt"] == b"parent"
    assert worker.store.head("rehearsal-runs/B/x/mixed-final/checkpoint.pt") is not None
    assert len(worker.store.refused) == 6
    worker.uploader.close(timeout=5)


# Launch gates ---------------------------------------------------------------------------------


IDENTITY = {
    "id": "podpost01",
    "name": "posttrain-a40-s-1",
    "gpuId": "NVIDIA A40",
    "gpuCount": 1,
    "dataCenterId": "test",
    "createdAt": "2026-11-01T00:00:00+00:00",
}
CREATED = 1_793_491_200.0  # 2026-11-01T00:00:00Z


def receipt(**changes: Any) -> dict[str, Any]:
    spec = Specification.load(SPEC)
    derived = spec.ceiling
    value: dict[str, Any] = {
        "kind": posttrain_ops.READINESS_KIND,
        "run": "1",
        "specification_sha256": "a" * 64,
        "binding_sha256": "b" * 64,
        "preflight_sha256": "c" * 64,
        **dict.fromkeys(posttrain_ops.READINESS_GATES, True),
        "termination_guarantee": "redundant_best_effort",
        "owner_accepts_best_effort_termination": True,
        "pod_identity": IDENTITY,
        "watchdog_policy": posttrain_ops.watchdog_policy(
            derived, origin=CREATED - 5, idle_baseline=CREATED
        ),
        "self_destruct_deadline_unix": derived.deadline_unix(CREATED - 5),
        "self_destruct_idle": {
            "idle_seconds": 900,
            "activity_files": ["/workspace/posttrain-run/heartbeat.json"],
        },
        "r2_run_prefix": posttrain_ops.run_prefix("1", "podpost01"),
    }
    value.update(changes)
    return value


def ready(value: dict[str, Any], *, now: float) -> Any:
    return check_readiness(
        value,
        now=now,
        spec=Specification.load(SPEC),
        specification_sha256="a" * 64,
        binding_sha256="b" * 64,
        preflight_sha256="c" * 64,
        heartbeat="/workspace/posttrain-run/heartbeat.json",
    )


def test_readiness_accepts_the_receipt_and_applies_the_launch_rule() -> None:
    launch = ready(receipt(), now=CREATED + 1_200)
    assert launch.pod_id == "podpost01" and launch.origin == CREATED - 5
    assert launch.end == CREATED - 5 + 17_820  # The armed timer is the first guard.
    assert launch.fit["fits_at_planning"]
    with pytest.raises(ValueError, match="cannot hold"):
        ready(receipt(), now=CREATED + 9_000)  # Setup ran too long for the whole run.


def test_a_mac_clock_ahead_of_the_provider_still_arms_within_the_wall() -> None:
    """The runbook arms the timer at the Mac's create-request time plus 17,820 s; a provider
    creation time 30 s earlier makes the conservative origin earlier, still inside the wall."""
    spec = Specification.load(SPEC)
    mac_origin = CREATED + 30
    value = receipt(
        watchdog_policy=posttrain_ops.watchdog_policy(
            spec.ceiling, origin=mac_origin, idle_baseline=CREATED
        ),
        self_destruct_deadline_unix=spec.ceiling.deadline_unix(mac_origin),
    )
    launch = ready(value, now=CREATED + 600)
    assert launch.origin == CREATED and launch.end == CREATED + 30 + 17_820


@pytest.mark.parametrize(
    "change, message",
    [
        ({"r2_round_trip_tested": False}, "unresolved"),
        ({"mac_parent_copies_verified": None}, "unresolved"),
        ({"run": "2"}, "readiness receipt for run 1"),
        ({"binding_sha256": "d" * 64}, "binding differs"),
        ({"self_destruct_deadline_unix": CREATED - 5 + 18_100}, "ceiling-derived lifetime"),
        ({"self_destruct_idle": {"idle_seconds": 1_800, "activity_files": []}}, "idle limit"),
        (
            {"self_destruct_idle": {"idle_seconds": 900, "activity_files": ["/elsewhere"]}},
            "does not watch",
        ),
        ({"r2_run_prefix": "posttrain-runs/1/other/"}, "prefix"),
    ],
)
def test_readiness_refuses_an_unresolved_or_weaker_receipt(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ready(receipt(**change), now=CREATED + 600)


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_wall_seconds", 18_001),
        ("max_cost_usd", 3.0),
        ("mode", "rehearsal"),
        ("max_transfer_seconds", 600),
    ],
)
def test_readiness_refuses_a_policy_above_the_ceiling_derived_limits(
    field: str, value: Any
) -> None:
    value_receipt = receipt()
    policy = copy.deepcopy(value_receipt["watchdog_policy"])
    policy[field] = value
    value_receipt["watchdog_policy"] = policy
    with pytest.raises(ValueError):
        ready(value_receipt, now=CREATED + 600)


# Generation with a heartbeat -------------------------------------------------------------------


def tiny_model() -> GPT:
    torch.manual_seed(0)
    return GPT(smoke_model(ByteBPE())).eval()


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_ticking_generation_is_generate_many_batch_for_batch(temperature: float) -> None:
    model = tiny_model()
    generator = torch.Generator().manual_seed(1)
    prompts = [
        torch.randint(0, 200, (length,), generator=generator).tolist() for length in (5, 9, 3, 9, 7)
    ]
    seeds = [10, 11, 12, 13, 14] if temperature else None
    options: dict[str, Any] = {
        "batch_size": 2,
        "max_new_tokens": 6,
        "temperature": temperature,
        "seeds": seeds,
        "stops": [{1, 2}] * 5,
    }
    ticks: list[int] = []
    expected = generate_many(model, prompts, **options)
    found = ticking_generate(ticks.append)(model, prompts, **options)
    assert found == expected
    assert len(ticks) == 3 and sum(ticks) == sum(len(item.tokens) for item in found)
