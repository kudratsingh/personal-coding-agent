"""Offline tests for the R2 client: signing, the cost cap and verified uploads."""

import datetime as dt
import hashlib
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
