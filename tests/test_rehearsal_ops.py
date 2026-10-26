"""Rehearsal R2 layout, final-state verification, the Pod key file, pinned fetch, round trip."""

import hashlib
import io
import json
import shlex
from pathlib import Path
from typing import Any

import pytest

from tools import rehearsal_ops
from tools.rehearsal_ops import (
    check_r2_final,
    fetch_object,
    pod_environment,
    round_trip,
    run_prefix,
    run_stages,
    run_summary,
    verify_final,
)

POD = "podfixture01"
ENVIRONMENT = {
    "R2_ENDPOINT": "https://account.r2.cloudflarestorage.com",
    "R2_BUCKET": "scratch-artifacts",
    "R2_ACCESS_KEY_ID": "mac-key-id",
    "R2_SECRET_ACCESS_KEY": "mac-secret-value",
    "R2_POD_ACCESS_KEY_ID": "pod-key-id",
    "R2_POD_SECRET_ACCESS_KEY": "pod secret 'quoted' $value",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Bucket:
    """An in-memory R2 with the Store methods the operations use."""

    def __init__(self) -> None:
        self.data: dict[str, tuple[bytes, str]] = {}
        self.fail_get = False
        self.keep_after_delete = False
        self.deleted: list[str] = []

    def put(self, key: str, data: bytes, *, meta: str | None = None) -> None:
        self.data[key] = (data, meta or sha(data))

    def put_known(self, key: str, path: Path, sha256: str, size: int) -> int:
        data = path.read_bytes()
        assert len(data) == size and sha(data) == sha256
        self.put(key, data)
        return 1

    def head(self, key: str) -> dict[str, str] | None:
        if key not in self.data:
            return None
        data, meta = self.data[key]
        return {"x-amz-meta-sha256": meta, "content-length": str(len(data))}

    def download(self, key: str, destination: Path, *, sha256: str, size: int) -> dict[str, Any]:
        if self.fail_get:
            raise RuntimeError("R2 GET answered HTTP 500")
        data, _ = self.data[key]
        if len(data) != size or sha(data) != sha256:
            raise ValueError(f"{key}: size or SHA-256 differs from the pinned value")
        destination.write_bytes(data)
        return {"key": key, "bytes": size, "sha256": sha256, "attempts": 1, "seconds": 0.0}

    def delete(self, key: str) -> None:
        self.deleted.append(key)
        if not self.keep_after_delete:
            del self.data[key]


def final_state(bucket: Bucket, run: str, status: str = "completed") -> dict[str, str]:
    prefix = run_prefix(run, POD)
    stages = []
    for name in run_stages(run):
        checkpoint = f"{run} {name} weights".encode()
        bucket.put(f"{prefix}{name}-final/checkpoint.pt", checkpoint)
        for other in ("checkpoint.pt.sha256", "record.json", "updates.jsonl", "ticket.json"):
            bucket.put(f"{prefix}{name}-final/{other}", other.encode())
        stages.append({"name": name, "final_checkpoint_sha256": sha(checkpoint), "updates": 7630})
    report = json.dumps({"status": status, "stages": stages}).encode()
    bucket.put(prefix + "report.json", report)
    last = f"{prefix}{run_stages(run)[-1]}-final/checkpoint.pt"
    return {
        "report_key": prefix + "report.json",
        "report_sha256": sha(report),
        "checkpoint_key": last,
        "checkpoint_sha256": bucket.data[last][1],
    }


def test_run_layout_and_limits_follow_d35() -> None:
    assert run_stages("A") == ("stage1", "stage2") and run_stages("B") == ("mixed",)
    assert run_prefix("B", POD) == f"rehearsal-runs/B/{POD}/"
    for bad in ("../x", "a/b", "", "pod id"):
        with pytest.raises(ValueError):
            run_prefix("A", bad)
    with pytest.raises(ValueError):
        run_prefix("C", POD)
    assert rehearsal_ops.COST_STOP_USD == pytest.approx(7.55004)
    assert rehearsal_ops.WATCHDOG_COST_STOP_USD < rehearsal_ops.COST_STOP_USD
    assert rehearsal_ops.TOKENS_PER_UPDATE == 131_072
    assert rehearsal_ops.POD_LIFETIME_SECONDS * rehearsal_ops.QUOTED_TOTAL_HOURLY / 3600 == 7.5


@pytest.mark.parametrize(
    "change",
    [
        {"report_key": f"rehearsal-runs/B/{POD}/report.json"},
        {"checkpoint_key": f"rehearsal-runs/A/{POD}/stage2-step15000/checkpoint.pt"},
        {"checkpoint_key": "rehearsal-runs/A/other/stage2-final/checkpoint.pt"},
        {"report_sha256": "ABC"},
        {"extra": "field"},
    ],
)
def test_r2_final_must_name_this_runs_own_milestone(change: dict[str, str]) -> None:
    final = {
        "report_key": f"rehearsal-runs/A/{POD}/report.json",
        "report_sha256": "1" * 64,
        "checkpoint_key": f"rehearsal-runs/A/{POD}/stage2-final/checkpoint.pt",
        "checkpoint_sha256": "2" * 64,
    }
    assert check_r2_final(final, run="A", pod_id=POD)[1] == "stage2"
    with pytest.raises(ValueError):
        check_r2_final({**final, **change}, run="A", pod_id=POD)


def test_verify_final_checks_every_milestone_and_keeps_the_pinned_report(
    tmp_path: Path,
) -> None:
    bucket = Bucket()
    final = final_state(bucket, "A")
    result = verify_final(bucket, final, run="A", pod_id=POD, destination=tmp_path)
    assert result["status"] == "verified" and result["report_status"] == "completed"
    assert [row["key"].split("/")[-2] for row in result["stage_milestones"]] == [
        "stage1-final",
        "stage2-final",
    ]
    assert len(result["final_label_files"]) == 5
    assert (tmp_path / "report.json").read_bytes() == bucket.data[final["report_key"]][0]


@pytest.mark.parametrize("problem", ["meta", "missing", "too-big", "status", "stage-order"])
def test_verify_final_refuses_anything_short_of_the_run_closing(
    tmp_path: Path, problem: str
) -> None:
    bucket = Bucket()
    final = final_state(bucket, "A")
    prefix = run_prefix("A", POD)
    if problem == "meta":
        bucket.put(prefix + "stage1-final/checkpoint.pt", b"other weights")
    elif problem == "missing":
        del bucket.data[prefix + "stage2-final/updates.jsonl"]
    elif problem == "too-big":
        bucket.put(prefix + "report.json", b"x" * 1_000_001)
        final["report_sha256"] = sha(b"x" * 1_000_001)
    elif problem == "status":
        final = final_state(bucket, "A", status="running")
    else:
        report = json.loads(bucket.data[prefix + "report.json"][0])
        report["stages"].reverse()
        data = json.dumps(report).encode()
        bucket.put(prefix + "report.json", data)
        final["report_sha256"] = sha(data)
    with pytest.raises(ValueError):
        verify_final(bucket, final, run="A", pod_id=POD, destination=tmp_path)


def test_summary_is_small_and_omits_loss_curves() -> None:
    report = {
        "status": "completed",
        "stages": [{"name": "mixed", "updates": 15259, "validations": [{"global_update": 500}]}],
        "updates": [{"loss": 1.0}] * 10_000,
        "cost": {"usd": 7.2},
    }
    summary = run_summary(report, run="B", pod_id=POD, verification=None, source="test")
    assert summary["stages"][0]["validations"] == 1
    assert summary["stages"][0]["last_validation"] == {"global_update": 500}
    assert "updates" not in summary and len(json.dumps(summary)) < 2_000


def test_pod_environment_holds_only_the_pod_key_and_sources_back_exactly() -> None:
    text = pod_environment(ENVIRONMENT)
    assert "mac-secret-value" not in text and "mac-key-id" not in text
    names = [line.split("=", 1)[0] for line in text.splitlines()]
    assert names == list(rehearsal_ops.POD_ENVIRONMENT)
    values = {
        line.split("=", 1)[0]: shlex.split(line.split("=", 1)[1])[0] for line in text.splitlines()
    }
    assert values["R2_POD_SECRET_ACCESS_KEY"] == ENVIRONMENT["R2_POD_SECRET_ACCESS_KEY"]


@pytest.mark.parametrize(
    "change",
    [
        {"R2_POD_ACCESS_KEY_ID": "mac-key-id"},
        {"R2_POD_SECRET_ACCESS_KEY": "two\nlines"},
        {"R2_POD_SECRET_ACCESS_KEY": ""},
        {"R2_ENDPOINT": "https://example.com"},
    ],
)
def test_pod_environment_refuses_the_mac_key_and_malformed_values(change: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        pod_environment({**ENVIRONMENT, **change})


def test_pod_env_never_writes_credentials_to_a_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name, value in ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    terminal = Terminal()
    monkeypatch.setattr("sys.stdout", terminal)
    with pytest.raises(SystemExit) as error:
        rehearsal_ops.main(["pod-env"])
    assert error.value.code == 2 and terminal.getvalue() == ""
    pipe = io.StringIO()
    monkeypatch.setattr("sys.stdout", pipe)
    assert rehearsal_ops.main(["pod-env"]) == 0
    assert pipe.getvalue() == pod_environment(ENVIRONMENT)


def test_fetch_object_is_pinned_and_keeps_only_identical_bytes(tmp_path: Path) -> None:
    bucket = Bucket()
    data = b"initial weights"
    bucket.put("rehearsal-init/x/rehearsal-init.pt", data)
    output, receipt = tmp_path / "initialization.pt", tmp_path / "receipt.json"
    record = fetch_object(
        bucket,
        key="rehearsal-init/x/rehearsal-init.pt",
        sha256=sha(data),
        size=len(data),
        output=output,
        receipt=receipt,
    )
    assert record["method"] == "downloaded" and output.read_bytes() == data
    again = fetch_object(
        bucket,
        key="rehearsal-init/x/rehearsal-init.pt",
        sha256=sha(data),
        size=len(data),
        output=output,
        receipt=receipt,
    )
    assert again["method"] == "present"
    output.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="other bytes"):
        fetch_object(
            bucket,
            key="rehearsal-init/x/rehearsal-init.pt",
            sha256=sha(data),
            size=len(data),
            output=output,
            receipt=receipt,
        )
    with pytest.raises(ValueError):
        fetch_object(bucket, key="k", sha256="bad", size=1, output=output, receipt=receipt)


def test_round_trip_passes_and_leaves_nothing_behind(tmp_path: Path) -> None:
    bucket = Bucket()
    record = round_trip(bucket, run="A", pod_id=POD, receipt=tmp_path / "trip.json", size=1024)
    assert record["status"] == "passed" and not bucket.data
    assert record["key"].startswith(f"rehearsal-runs/A/{POD}/preflight/round-trip-")
    assert set(record["steps"]) == {"put_seconds", "head_seconds", "get_seconds", "delete_seconds"}
    assert json.loads((tmp_path / "trip.json").read_text()) == record


@pytest.mark.parametrize("failure", ["get", "delete"])
def test_round_trip_failure_is_recorded_and_the_probe_removed(tmp_path: Path, failure: str) -> None:
    bucket = Bucket()
    if failure == "get":
        bucket.fail_get = True
    else:
        bucket.keep_after_delete = True
    with pytest.raises((RuntimeError, ValueError)):
        round_trip(bucket, run="B", pod_id=POD, receipt=tmp_path / "trip.json", size=64)
    saved = json.loads((tmp_path / "trip.json").read_text())
    assert saved["status"] == "failed" and saved["error"]
    assert bucket.deleted, "the probe is deleted after a failure"
    if failure == "get":
        assert not bucket.data
