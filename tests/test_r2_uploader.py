"""Offline tests for the background checkpoint uploader and its retried R2 requests."""

import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

import pytest

from tools import r2_store
from tools.r2_store import CheckpointUploader, Credentials, Store, TransientError

Fault = int | Exception | None
File = tuple[str, Path, str]


class FakeBucket:
    """An in-memory R2 bucket that records every request and can fail or stall on cue.

    ``faults[method]`` is consumed one entry per request: an int is answered as that
    HTTP status, an exception is raised and None is served normally. Clearing ``hold``
    stalls every PUT until it is set again; ``putting`` is set once a PUT has started.
    Each stored PUT advances the fake clock by 2.5 seconds.
    """

    def __init__(self, sizes: Mapping[str, int] | None = None) -> None:
        self.sizes = dict(sizes or {})
        self.digests: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []
        self.signed: list[str] = []
        self.faults: dict[str, list[Fault]] = {}
        self.hold, self.putting = threading.Event(), threading.Event()
        self.hold.set()
        self.corrupt = False
        self.now = 0.0
        self.lock = threading.Lock()

    def clock(self) -> float:
        return self.now

    def keys(self) -> set[str]:
        return set(self.sizes)

    def requested(self, method: str) -> list[str]:
        return [key for name, key in self.calls if name == method]

    def __call__(self, request: Request) -> tuple[int, Mapping[str, str], bytes]:
        method = request.get_method()
        path = urlsplit(request.full_url).path
        key = path.split("/", 2)[2] if path.count("/") >= 2 else ""
        headers = {name.lower(): value for name, value in request.header_items()}
        assert "authorization" in headers and "secret" not in json.dumps(headers)
        with self.lock:
            self.calls.append((method, key))
            queued = self.faults.get(method, [])
            fault = queued.pop(0) if queued else None
        if method == "PUT":
            self.putting.set()
            assert self.hold.wait(10)
        if isinstance(fault, Exception):
            raise fault
        if fault is not None:
            return fault, {}, b"injected"
        if method == "GET":
            contents = "".join(
                f"<Contents><Key>{name}</Key><Size>{size}</Size></Contents>"
                for name, size in self.sizes.items()
            )
            body = (
                '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                f"{contents}<IsTruncated>false</IsTruncated></ListBucketResult>"
            )
            return 200, {}, body.encode()
        if method == "HEAD":
            if key not in self.sizes:
                return 404, {}, b""
            return 200, {"X-Amz-Meta-Sha256": self.digests.get(key, "")}, b""
        if method == "DELETE":
            self.sizes.pop(key, None)
            self.digests.pop(key, None)
            return 204, {}, b""
        assert method == "PUT"
        payload: Any = request.data
        data = payload.read()
        assert isinstance(data, bytes) and int(headers["content-length"]) == len(data)
        self.signed.append(headers["x-amz-content-sha256"])
        if hashlib.sha256(data).hexdigest() != headers["x-amz-content-sha256"]:
            return 400, {}, b"XAmzContentSHA256Mismatch"
        self.sizes[key] = len(data)
        self.digests[key] = "0" * 64 if self.corrupt else headers["x-amz-meta-sha256"]
        self.now += 2.5
        return 200, {}, b""


CREDENTIALS = Credentials("https://acct.r2.cloudflarestorage.com", "id", "secret", "bucket-a")


def checkpoint(root: Path, label: str, *names: str) -> list[File]:
    """Files for one label, each with distinct bytes and its known digest."""
    files = []
    for name in names or ("model.pt",):
        path = root / "local" / label / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = f"{label}/{name}".encode()
        path.write_bytes(data)
        files.append((name, path, hashlib.sha256(data).hexdigest()))
    return files


Make = Callable[..., tuple[CheckpointUploader, FakeBucket, list[float]]]


@pytest.fixture
def make(tmp_path: Path) -> Iterator[Make]:
    made: list[tuple[CheckpointUploader, FakeBucket]] = []

    def build(fake: FakeBucket | None = None, **options: Any) -> Any:
        fake = fake or FakeBucket()
        sleeps: list[float] = []
        store = Store(CREDENTIALS, transport=fake, sleep=sleeps.append)
        settings = {"prefix": "runs/a/", "cap_usd": 5.0, "receipt_dir": tmp_path / "receipts"}
        uploader = CheckpointUploader(store, clock=fake.clock, **{**settings, **options})
        made.append((uploader, fake))
        return uploader, fake, sleeps

    yield build
    for uploader, fake in made:
        fake.hold.set()
        assert uploader.close(timeout=5)


def receipt(tmp_path: Path, label: str) -> dict[str, Any]:
    value: dict[str, Any] = json.loads((tmp_path / "receipts" / f"{label}.json").read_text())
    return value


def test_submit_never_waits_and_the_newest_rolling_submission_wins(
    tmp_path: Path, make: Make
) -> None:
    uploader, fake, _ = make()
    later = [("best-1", True), ("step-2", False), ("best-2", True), ("step-3", False)]
    files = {label: checkpoint(tmp_path, label) for label, _ in later}
    fake.hold.clear()
    uploader.submit("step-1", checkpoint(tmp_path, "step-1"), milestone=False)
    assert fake.putting.wait(5)
    started = time.monotonic()
    for label, milestone in later:
        uploader.submit(label, files[label], milestone=milestone)
    assert time.monotonic() - started < 1.0  # the first PUT is still held
    status = uploader.status()
    assert status["in_flight"] == "step-1" and status["last_verified"] is None
    assert status["pending"] == ["best-1", "best-2", "step-3"]
    assert status["superseded"] == ["step-2"]
    assert not uploader.drain(0.05)
    fake.hold.set()
    assert uploader.drain(5)
    status = uploader.status()
    assert status["verified"] == ["step-1", "best-1", "best-2", "step-3"]
    assert status["last_verified"] == "step-3" and status["in_flight"] is None
    assert status["pending"] == [] and status["failures"] == [] and status["refused"] == []
    assert fake.requested("PUT") == [
        f"runs/a/{label}/model.pt" for label in ("step-1", "best-1", "best-2", "step-3")
    ]
    assert not (tmp_path / "receipts" / "step-2.json").exists()


def test_retention_keeps_the_newest_rolling_labels_and_every_milestone(
    tmp_path: Path, make: Make
) -> None:
    uploader, fake, _ = make(keep_rolling=2)
    names = ("model.pt", "optimizer/state.pt")
    for label, milestone in [
        ("step-1", False),
        ("best-1", True),
        ("step-2", False),
        ("step-3", False),
        ("step-4", False),
    ]:
        uploader.submit(label, checkpoint(tmp_path, label, *names), milestone=milestone)
        assert uploader.drain(5)
    status = uploader.status()
    assert status["deleted"] == ["step-1", "step-2"] and status["rolling"] == ["step-3", "step-4"]
    assert fake.requested("DELETE") == [
        f"runs/a/{label}/{n}" for label in ("step-1", "step-2") for n in names
    ]
    assert fake.keys() == {
        f"runs/a/{label}/{n}" for label in ("best-1", "step-3", "step-4") for n in names
    }
    for label in ("step-1", "step-2"):
        assert "deleted_at" in receipt(tmp_path, label)
    for label in ("best-1", "step-3", "step-4"):
        assert "deleted_at" not in receipt(tmp_path, label)


def test_a_label_verified_as_a_milestone_is_never_deleted_by_retention(
    tmp_path: Path, make: Make
) -> None:
    uploader, fake, _ = make(keep_rolling=1)
    for label, milestone in [("step-1", False), ("step-1", True), ("step-2", False)]:
        uploader.submit(label, checkpoint(tmp_path, label), milestone=milestone)
        assert uploader.drain(5)
    status = uploader.status()
    assert status["deleted"] == [] and status["rolling"] == ["step-2"]
    assert fake.keys() == {"runs/a/step-1/model.pt", "runs/a/step-2/model.pt"}
    # Re-sent as rolling, the milestone label is neither listed nor a reason to delete.
    for label in ("step-1", "step-3"):
        uploader.submit(label, checkpoint(tmp_path, label), milestone=False)
        assert uploader.drain(5)
    status = uploader.status()
    assert status["deleted"] == ["step-2"] and status["rolling"] == ["step-3"]
    assert fake.keys() == {"runs/a/step-1/model.pt", "runs/a/step-3/model.pt"}


def test_the_cap_counts_this_upload_and_refuses_it_without_raising(
    tmp_path: Path, make: Make
) -> None:
    # 10 GB is exactly the free tier: any further byte prices above a $0 cap.
    uploader, fake, _ = make(FakeBucket({"other/held.bin": 10 * 10**9}), cap_usd=0.0)
    files = checkpoint(tmp_path, "step-1")
    uploader.submit("step-1", files, milestone=False)
    assert uploader.drain(5)
    status = uploader.status()
    size = files[0][1].stat().st_size
    assert status["refused"] == [
        {
            "label": "step-1",
            "bytes": size,
            "bucket_bytes_after": 10 * 10**9 + size,
            "monthly_storage_usd_after": pytest.approx(size / 1e9 * 0.015),
            "cap_usd_per_month": 0.0,
        }
    ]
    assert status["failures"] == [] and status["last_verified"] is None
    assert fake.requested("GET") == [""] and fake.requested("PUT") == []
    assert not (tmp_path / "receipts" / "step-1.json").exists()
    roomy, _, _ = make(fake, cap_usd=0.01)
    roomy.submit("step-1", files, milestone=False)
    assert roomy.drain(5) and roomy.status()["last_verified"] == "step-1"


def test_transient_failures_retry_with_backoff_then_succeed(tmp_path: Path, make: Make) -> None:
    uploader, fake, sleeps = make()
    fake.faults = {
        "GET": [502],
        "PUT": [500, TimeoutError("timed out"), URLError(ConnectionRefusedError())],
        "HEAD": [503],
    }
    uploader.submit("step-1", checkpoint(tmp_path, "step-1"), milestone=False)
    assert uploader.drain(5)
    assert uploader.status()["failures"] == []
    assert sleeps == [2.0, 2.0, 4.0, 8.0, 2.0]
    assert len(fake.requested("PUT")) == 4 and len(fake.requested("HEAD")) == 2
    assert receipt(tmp_path, "step-1")["files"][0]["attempts"] == 4


def test_client_errors_and_exhausted_retries_are_recorded_and_cleaned_up(
    tmp_path: Path, make: Make
) -> None:
    uploader, fake, sleeps = make(keep_rolling=2)
    fake.faults = {"PUT": [None, 403]}  # the model is stored, the optimizer refused
    uploader.submit("step-1", checkpoint(tmp_path, "step-1", "model.pt", "opt.pt"), milestone=False)
    assert uploader.drain(5)
    assert len(fake.requested("PUT")) == 2 and sleeps == []
    fake.faults = {"PUT": [500, 500, 500, 500]}
    uploader.submit("step-2", checkpoint(tmp_path, "step-2"), milestone=False)
    assert uploader.drain(5)
    assert len(fake.requested("PUT")) == 6 and sleeps == [2.0, 4.0, 8.0]
    failures = uploader.status()["failures"]
    assert [(item["label"], item["stage"]) for item in failures] == [
        ("step-1", "upload"),
        ("step-2", "upload"),
    ]
    assert "HTTP 403" in failures[0]["error"] and "TransientError" in failures[1]["error"]
    assert not (tmp_path / "receipts" / "step-1.json").exists()
    # The next verified rolling label deletes what the failed one managed to store.
    uploader.submit("step-3", checkpoint(tmp_path, "step-3"), milestone=False)
    assert uploader.drain(5)
    status = uploader.status()
    assert status["last_verified"] == "step-3" and status["deleted"] == ["step-1"]
    assert fake.requested("DELETE") == ["runs/a/step-1/model.pt"]
    assert fake.keys() == {"runs/a/step-3/model.pt"}


def test_the_known_digest_is_signed_as_is_and_r2_rejects_other_bytes(
    tmp_path: Path, make: Make
) -> None:
    uploader, fake, sleeps = make()
    (key, path, _), *_ = checkpoint(tmp_path, "step-1")
    wrong = hashlib.sha256(b"not these bytes").hexdigest()
    uploader.submit("step-1", [(key, path, wrong)], milestone=False)
    assert uploader.drain(5)
    assert fake.signed == [wrong] and sleeps == []
    assert "XAmzContentSHA256Mismatch" in uploader.status()["failures"][0]["error"]


def test_worker_errors_are_captured_and_later_work_continues(tmp_path: Path, make: Make) -> None:
    uploader, fake, _ = make()
    missing = [("model.pt", tmp_path / "absent.pt", "0" * 64)]
    uploader.submit("step-1", missing, milestone=False)
    assert uploader.drain(5)
    fake.corrupt = True
    uploader.submit("step-2", checkpoint(tmp_path, "step-2"), milestone=False)
    assert uploader.drain(5)
    fake.corrupt = False
    fake.faults = {"GET": [ValueError("unexpected transport bug")]}
    uploader.submit("step-3", checkpoint(tmp_path, "step-3"), milestone=False)
    assert uploader.drain(5)
    uploader.submit("step-4", checkpoint(tmp_path, "step-4"), milestone=False)
    assert uploader.drain(5)
    status = uploader.status()
    errors = [item["error"] for item in status["failures"]]
    assert errors[0].startswith("FileNotFoundError")
    assert errors[1] == "RuntimeError: runs/a/step-2/model.pt: stored digest differs after upload"
    assert errors[2] == "ValueError: unexpected transport bug"
    assert status["failure_count"] == 3 and status["retrying"] == {}  # Rolling: never retried.
    # The corrupt label's stored key is deleted once a later rolling label verifies.
    assert status["verified"] == ["step-4"] and status["deleted"] == ["step-2"]


def wait_for(condition: Callable[[], bool], seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


def test_a_failed_milestone_is_retried_with_backoff_until_it_verifies(
    tmp_path: Path, make: Make
) -> None:
    uploader, fake, _ = make(retry_seconds=0.01, retry_max_seconds=0.02)
    fake.faults = {"PUT": [403, 403, 403]}  # Not transient: each upload attempt fails at once.
    uploader.submit("best-1", checkpoint(tmp_path, "best-1"), milestone=True)
    assert uploader.drain(5)
    status = uploader.status()
    assert status["verified"] == ["best-1"] and status["retrying"] == {}
    assert [item["label"] for item in status["failures"]] == ["best-1"] * 3
    assert status["failure_count"] == 3 and len(fake.requested("PUT")) == 4
    assert receipt(tmp_path, "best-1")["milestone"] is True


def test_a_resubmitted_milestone_is_due_at_once_and_drain_waits_for_its_retry(
    tmp_path: Path, make: Make
) -> None:
    uploader, fake, _ = make(retry_seconds=60.0, retry_max_seconds=60.0)
    files = checkpoint(tmp_path, "best-1")
    fake.faults = {"PUT": [403]}
    uploader.submit("best-1", files, milestone=True)
    wait_for(lambda: uploader.status()["retrying"] == {"best-1": 1})
    # Waiting to retry is queued work: drain does not report it done.
    assert not uploader.drain(0.05) and uploader.status()["pending"] == ["best-1"]
    uploader.submit("best-1", files, milestone=True)  # Due now; nothing is queued twice.
    assert uploader.drain(5)
    status = uploader.status()
    assert status["verified"] == ["best-1"] and status["pending"] == []
    assert fake.requested("PUT") == ["runs/a/best-1/model.pt"] * 2


def test_a_refused_milestone_is_retried_once_the_cap_allows_it(tmp_path: Path, make: Make) -> None:
    fake = FakeBucket({"other/held.bin": 10 * 10**9})
    uploader, _, _ = make(fake, cap_usd=0.0, retry_seconds=0.01, retry_max_seconds=0.02)
    uploader.submit("best-1", checkpoint(tmp_path, "best-1"), milestone=True)
    wait_for(lambda: len(uploader.status()["refused"]) >= 2)
    assert not uploader.status()["verified"] and fake.requested("PUT") == []
    with fake.lock:
        del fake.sizes["other/held.bin"]  # Space freed: the next retry fits the cap.
    assert uploader.drain(5) and uploader.status()["verified"] == ["best-1"]


def test_a_milestone_drops_waiting_rolling_labels_but_not_the_one_in_flight(
    tmp_path: Path, make: Make
) -> None:
    uploader, fake, _ = make()
    files = {label: checkpoint(tmp_path, label) for label in ("step-2", "best-1", "step-3")}
    fake.hold.clear()
    uploader.submit("step-1", checkpoint(tmp_path, "step-1"), milestone=False)
    assert fake.putting.wait(5)
    uploader.submit("step-2", files["step-2"], milestone=False)
    uploader.submit("best-1", files["best-1"], milestone=True)
    uploader.submit("best-1", files["best-1"], milestone=True)  # Already waiting: kept once.
    status = uploader.status()
    assert status["in_flight"] == "step-1" and status["pending"] == ["best-1"]
    assert status["superseded"] == ["step-2"]
    uploader.submit("step-3", files["step-3"], milestone=False)  # Waits behind the milestone.
    fake.hold.set()
    assert uploader.drain(5)
    assert uploader.status()["verified"] == ["step-1", "best-1", "step-3"]


def test_resubmitting_an_older_milestone_keeps_newer_rolling_labels(
    tmp_path: Path, make: Make
) -> None:
    uploader, fake, _ = make(retry_seconds=60.0, retry_max_seconds=60.0)
    fake.faults = {"PUT": [403]}
    uploader.submit("stage1-final", checkpoint(tmp_path, "stage1-final"), milestone=True)
    wait_for(lambda: bool(uploader.status()["retrying"]))  # Waiting out its backoff.
    fake.hold.clear()
    fake.putting.clear()  # stage1-final's refused PUT already set it.
    uploader.submit("stage2-step8", checkpoint(tmp_path, "stage2-step8"), milestone=False)
    assert fake.putting.wait(5)
    wait_for(lambda: uploader.status()["in_flight"] == "stage2-step8")
    uploader.submit("stage2-step9", checkpoint(tmp_path, "stage2-step9"), milestone=False)
    # Finalization resubmits the older milestone: due at once, the newer weights kept.
    uploader.submit("stage1-final", checkpoint(tmp_path, "stage1-final"), milestone=True)
    status = uploader.status()
    assert status["in_flight"] == "stage2-step8"
    assert status["pending"] == ["stage1-final", "stage2-step9"]
    assert status["superseded"] == []
    fake.hold.set()
    assert uploader.drain(5)
    assert uploader.status()["verified"] == ["stage2-step8", "stage1-final", "stage2-step9"]


def test_close_cancels_a_milestone_waiting_to_retry(tmp_path: Path, make: Make) -> None:
    uploader, fake, _ = make(retry_seconds=60.0, retry_max_seconds=60.0)
    fake.faults = {"PUT": [403]}
    uploader.submit("best-1", checkpoint(tmp_path, "best-1"), milestone=True)
    wait_for(lambda: bool(uploader.status()["retrying"]))
    assert uploader.close(timeout=5)  # The worker never waits out the backoff.
    status = uploader.status()
    assert status["cancelled"] == ["best-1"] and status["pending"] == []
    assert len(fake.requested("PUT")) == 1


def test_recorded_failures_are_bounded(
    tmp_path: Path, make: Make, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r2_store, "MAX_RECORDED_FAILURES", 3)
    uploader, fake, _ = make(retry_seconds=0.001, retry_max_seconds=0.001)
    fake.faults = {"PUT": [403] * 5}
    uploader.submit("best-1", checkpoint(tmp_path, "best-1"), milestone=True)
    assert uploader.drain(5)
    status = uploader.status()
    assert len(status["failures"]) == 3 and status["failure_count"] == 5


def test_receipt_records_keys_bytes_digests_attempts_and_seconds(
    tmp_path: Path, make: Make
) -> None:
    uploader, fake, _ = make()
    files = checkpoint(tmp_path, "best-1", "model.pt", "optimizer.pt")
    uploader.submit("best-1", files, milestone=True)
    assert uploader.drain(5)
    record = receipt(tmp_path, "best-1")
    keys = ["runs/a/best-1/model.pt", "runs/a/best-1/optimizer.pt"]
    sizes = [path.stat().st_size for _, path, _ in files]
    assert record["label"] == "best-1" and record["milestone"] is True
    assert record["bucket"] == "bucket-a" and record["keys"] == keys
    assert record["files"] == [
        {
            "key": key,
            "path": str(path),
            "bytes": size,
            "sha256": digest,
            "attempts": 1,
            "seconds": 2.5,
        }
        for key, (_, path, digest), size in zip(keys, files, sizes, strict=True)
    ]
    assert record["bytes"] == sum(sizes) and record["attempts"] == 2 and record["seconds"] == 5.0
    assert record["bucket_bytes_after"] == sum(sizes) and record["cap_usd_per_month"] == 5.0
    assert fake.signed == [digest for _, _, digest in files]


def test_close_cancels_unstarted_work_and_refuses_new_submissions(
    tmp_path: Path, make: Make
) -> None:
    uploader, fake, _ = make()
    fake.hold.clear()
    uploader.submit("step-1", checkpoint(tmp_path, "step-1"), milestone=False)
    assert fake.putting.wait(5)
    uploader.submit("best-1", checkpoint(tmp_path, "best-1"), milestone=True)
    assert not uploader.close(timeout=0.05)
    assert uploader.status()["cancelled"] == ["best-1"] and uploader.status()["closed"]
    with pytest.raises(RuntimeError, match="closed"):
        uploader.submit("step-2", checkpoint(tmp_path, "step-2"), milestone=False)
    fake.hold.set()
    assert uploader.close(timeout=5) and uploader.drain(0)
    assert uploader.status()["verified"] == ["step-1"]
    assert fake.requested("PUT") == ["runs/a/step-1/model.pt"]


def test_concurrent_submissions_lose_nothing(tmp_path: Path, make: Make) -> None:
    uploader, fake, _ = make(keep_rolling=2)
    labels = [f"step-{thread}-{index}" for thread in range(4) for index in range(25)]
    files = {label: checkpoint(tmp_path, label) for label in labels}

    def submit_all(thread: int) -> None:
        for index in range(25):
            label = f"step-{thread}-{index}"
            uploader.submit(label, files[label], milestone=False)

    threads = [threading.Thread(target=submit_all, args=(thread,)) for thread in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert uploader.drain(10)
    status = uploader.status()
    verified, superseded = set(status["verified"]), set(status["superseded"])
    assert verified | superseded == set(labels) and not verified & superseded
    assert len(status["verified"]) == len(verified) and status["failures"] == []
    assert len(status["rolling"]) == min(2, len(verified))
    assert fake.keys() == {f"runs/a/{label}/model.pt" for label in status["rolling"]}


def test_malformed_submissions_raise_on_the_caller_and_send_nothing(
    tmp_path: Path, make: Make
) -> None:
    uploader, fake, _ = make()
    (key, path, digest), *_ = checkpoint(tmp_path, "step-1")
    for label, files in [
        ("a/b", [(key, path, digest)]),
        ("", [(key, path, digest)]),
        ("..", [(key, path, digest)]),
        ("step-1", []),
        ("step-1", [(key, path, digest), (key, path, digest)]),
        ("step-1", [("/model.pt", path, digest)]),
        ("step-1", [("x/../model.pt", path, digest)]),
        ("step-1", [(key, path, digest.upper())]),
    ]:
        with pytest.raises(ValueError):
            uploader.submit(label, files, milestone=False)
    store = Store(CREDENTIALS, transport=fake)
    for prefix, keep in [("runs/a", 2), ("/runs/", 2), ("", 0)]:
        with pytest.raises(ValueError):
            CheckpointUploader(
                store, prefix=prefix, cap_usd=1.0, receipt_dir=tmp_path, keep_rolling=keep
            )
    for retry, longest in [(0.0, 1.0), (2.0, 1.0)]:
        with pytest.raises(ValueError, match="retry_seconds"):
            CheckpointUploader(
                store,
                prefix="runs/",
                cap_usd=1.0,
                receipt_dir=tmp_path,
                retry_seconds=retry,
                retry_max_seconds=longest,
            )
    assert uploader.drain(5) and fake.calls == []


def test_head_and_put_known_keep_stored_sha256_and_put_semantics(tmp_path: Path) -> None:
    fake = FakeBucket()
    sleeps: list[float] = []
    store = Store(CREDENTIALS, transport=fake, sleep=sleeps.append)
    (_, path, digest), *_ = checkpoint(tmp_path, "step-1")
    assert store.head("k") is None and store.stored_sha256("k") is None
    assert store.put_known("k", path, digest, path.stat().st_size) == 1
    assert store.head("k") == {"x-amz-meta-sha256": digest} and store.stored_sha256("k") == digest
    fake.faults = {"HEAD": [503, 403]}
    with pytest.raises(TransientError, match="HTTP 503"):
        store.head("k")
    with pytest.raises(RuntimeError, match="HTTP 403") as refused:
        store.stored_sha256("k")
    assert not isinstance(refused.value, TransientError)
    # put keeps its single attempt; put_known retries only transient answers.
    fake.faults = {"PUT": [500, HTTPError("https://r2", 404, "absent", Message(), None)]}
    with pytest.raises(TransientError):
        store.put("k", path, digest)
    with pytest.raises(HTTPError):
        store.put_known("k", path, digest, path.stat().st_size)
    assert len(fake.requested("PUT")) == 3 and sleeps == []
    calls = len(fake.calls)
    with pytest.raises(ValueError, match="declared"):
        store.put_known("k", path, digest, path.stat().st_size + 1)
    assert len(fake.calls) == calls
