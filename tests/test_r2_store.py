"""Offline tests for the R2 client: signing, the cost cap and verified uploads."""

import datetime as dt
import hashlib
import http.client
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request

import pytest

from tools import r2_store
from tools.r2_store import Credentials, Store, monthly_storage_usd, signature_headers, upload


def test_signature_matches_the_aws_documented_get_object_example() -> None:
    credentials = Credentials(
        "https://examplebucket.s3.amazonaws.com",
        "AKIAIOSFODNN7EXAMPLE",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "examplebucket",
    )
    headers = signature_headers(
        credentials,
        "GET",
        "/test.txt",
        {},
        {"Range": "bytes=0-9"},
        hashlib.sha256(b"").hexdigest(),
        dt.datetime(2013, 5, 24, tzinfo=dt.UTC),
        region="us-east-1",
    )
    assert headers["Authorization"].endswith(
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
    )


def test_credentials_require_an_https_r2_endpoint_and_never_repr_secrets() -> None:
    environ = {
        "R2_ENDPOINT": "https://acct.r2.cloudflarestorage.com",
        "R2_ACCESS_KEY_ID": "id",
        "R2_SECRET_ACCESS_KEY": "secret",
        "R2_BUCKET": "scratch-coder-artifacts",
    }
    assert Credentials.from_environment(environ).bucket == "scratch-coder-artifacts"
    for key, value in [
        ("R2_ENDPOINT", "http://acct.r2.cloudflarestorage.com"),
        ("R2_ENDPOINT", "https://example.com"),
        ("R2_BUCKET", "Bad_Bucket"),
        ("R2_SECRET_ACCESS_KEY", ""),
    ]:
        with pytest.raises(ValueError):
            Credentials.from_environment({**environ, key: value})


def test_storage_is_priced_after_the_free_tier() -> None:
    assert monthly_storage_usd(5 * 10**9) == 0
    assert monthly_storage_usd(110 * 10**9) == pytest.approx(1.5)


class FakeR2:
    """An in-memory bucket that checks each PUT body against its signed digest."""

    def __init__(self, objects: Mapping[str, bytes] | None = None) -> None:
        self.objects = {
            key: (data, hashlib.sha256(data).hexdigest()) for key, data in (objects or {}).items()
        }
        self.puts: list[str] = []
        self.corrupt = False

    def __call__(self, request: Request) -> tuple[int, Mapping[str, str], bytes]:
        path = urlsplit(request.full_url).path
        key = path.split("/", 2)[2] if path.count("/") >= 2 else ""
        headers = {name.lower(): value for name, value in request.header_items()}
        assert "authorization" in headers and "secret" not in json.dumps(headers)
        if request.get_method() == "GET":
            contents = "".join(
                f"<Contents><Key>{k}</Key><Size>{len(v[0])}</Size></Contents>"
                for k, v in self.objects.items()
            )
            body = (
                '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                f"{contents}<IsTruncated>false</IsTruncated></ListBucketResult>"
            )
            return 200, {}, body.encode()
        if request.get_method() == "HEAD":
            if key not in self.objects:
                return 404, {}, b""
            return 200, {"x-amz-meta-sha256": self.objects[key][1]}, b""
        assert request.get_method() == "PUT"
        payload: Any = request.data
        data = payload.read() if hasattr(payload, "read") else payload
        assert isinstance(data, bytes)
        if hashlib.sha256(data).hexdigest() != headers["x-amz-content-sha256"]:
            return 400, {}, b"XAmzContentSHA256Mismatch"
        self.puts.append(key)
        stored = headers["x-amz-meta-sha256"]
        self.objects[key] = (data, "0" * 64 if self.corrupt else stored)
        return 200, {}, b""


def store(fake: FakeR2) -> Store:
    credentials = Credentials("https://acct.r2.cloudflarestorage.com", "id", "secret", "bucket-a")
    return Store(credentials, transport=fake)


def test_upload_skips_identical_objects_and_refuses_to_exceed_the_cap(tmp_path: Path) -> None:
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    fake = FakeR2({"run/a.bin": b"alpha"})
    record = upload(
        store(fake),
        [("run/a.bin", first), ("run/b.bin", second)],
        cap_usd=5.0,
        receipt=tmp_path / "receipt.json",
        progress=lambda _: None,
    )
    assert [row["method"] for row in record["files"]] == ["present", "uploaded"]
    assert fake.puts == ["run/b.bin"]
    assert json.loads((tmp_path / "receipt.json").read_text())["bucket_bytes_after"] == 9
    # 20 GB already held prices at $0.15; a $0.10 cap refuses any further upload.
    full = FakeR2({"big": b""})
    monkey_objects = [{"key": "big", "bytes": 20 * 10**9}]
    capped = store(full)
    capped.objects = lambda: monkey_objects  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="above the"):
        upload(capped, [("run/b.bin", second)], cap_usd=0.10, receipt=tmp_path / "r2.json")
    assert full.puts == []


def test_a_digest_that_does_not_read_back_stops_the_upload(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"alpha")
    fake = FakeR2()
    fake.corrupt = True
    with pytest.raises(RuntimeError, match="digest differs"):
        upload(store(fake), [("a.bin", path)], cap_usd=5.0, receipt=tmp_path / "r.json")
    assert r2_store.PRICES_CHECKED_ON == "2026-10-25"


class Reply(io.BytesIO):
    status = 200


def test_download_publishes_only_bytes_matching_the_pinned_digest(tmp_path: Path) -> None:
    data = b"shard bytes"
    calls: list[str] = []

    def opener(request: Request) -> Reply:
        calls.append(request.get_method())
        assert "Authorization" in dict(request.header_items())
        return Reply(data)

    credentials = Credentials("https://acct.r2.cloudflarestorage.com", "id", "secret", "bucket-a")
    store = Store(credentials, opener=opener, sleep=lambda _: None)
    target = tmp_path / "out" / "a.bin"
    receipt = store.download("k/a.bin", target, sha256=hashlib.sha256(data).hexdigest(), size=11)
    assert target.read_bytes() == data and receipt["bytes"] == 11 and calls == ["GET"]
    with pytest.raises(ValueError, match="differs"):
        store.download("k/a.bin", tmp_path / "bad.bin", sha256="0" * 64, size=11)
    assert not (tmp_path / "bad.bin").exists() and not (tmp_path / "bad.bin.part").exists()
    assert calls == ["GET", "GET"]  # A full-length body with another digest: no retry.


class Failing(Reply):
    """A 200 answer whose body read raises ``error``."""

    def __init__(self, error: Exception) -> None:
        super().__init__(b"")
        self.error = error

    def read(self, size: int | None = -1) -> bytes:
        raise self.error


def downloading(answers: list[bytes | Exception]) -> tuple[Store, list[float]]:
    """A store whose GETs answer ``answers`` in turn: bytes as the body, or a failing read."""
    sleeps: list[float] = []

    def opener(request: Request) -> Reply:
        answer = answers.pop(0)
        return Failing(answer) if isinstance(answer, Exception) else Reply(answer)

    credentials = Credentials("https://acct.r2.cloudflarestorage.com", "id", "secret", "bucket-a")
    return Store(credentials, opener=opener, sleep=sleeps.append), sleeps


def test_download_retries_a_truncated_body_within_its_attempts(tmp_path: Path) -> None:
    data = b"shard bytes"
    digest = hashlib.sha256(data).hexdigest()
    # A clean early close reads as a short body; IncompleteRead is an HTTPException.
    answers: list[bytes | Exception] = [data[:4], http.client.IncompleteRead(data[:6], 5), data]
    store, sleeps = downloading(answers)
    receipt = store.download("k/a.bin", tmp_path / "a.bin", sha256=digest, size=len(data))
    assert receipt["attempts"] == 3 and sleeps == [2.0, 4.0]
    assert (tmp_path / "a.bin").read_bytes() == data and not answers
    store, sleeps = downloading([data[:4]] * 4)
    with pytest.raises(r2_store.TransientError, match="ended after 4 of 11 bytes"):
        store.download("k/b.bin", tmp_path / "b.bin", sha256=digest, size=len(data))
    assert sleeps == [2.0, 4.0, 8.0]  # The whole budget, then the last error.
    assert not (tmp_path / "b.bin").exists() and not (tmp_path / "b.bin.part").exists()


def test_download_refuses_a_longer_body_at_once(tmp_path: Path) -> None:
    data = b"shard bytes"
    answers: list[bytes | Exception] = [data + b"!", data]
    store, sleeps = downloading(answers)
    with pytest.raises(ValueError, match="more bytes than the pinned size"):
        store.download(
            "k/a.bin", tmp_path / "a.bin", sha256=hashlib.sha256(data).hexdigest(), size=11
        )
    assert sleeps == [] and len(answers) == 1
    assert not (tmp_path / "a.bin").exists() and not (tmp_path / "a.bin.part").exists()
