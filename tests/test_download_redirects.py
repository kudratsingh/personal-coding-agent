"""Actual urllib redirect processing over an authored, entirely offline transport."""

import hashlib
from dataclasses import replace
from http.client import HTTPMessage
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import HTTPHandler, HTTPSHandler, ProxyHandler, Request, build_opener
from urllib.response import addinfourl

import pytest

from scratchcoder.data import manifest
from scratchcoder.data.manifest import HTTPSOnlyRedirectHandler, ManifestEntry, download


@pytest.mark.parametrize(
    "url",
    [
        "http://example.invalid/data",
        "ftp://example.invalid/data",
        "file:///data",
        "data:text/plain,data",
        "https:///no-host",
        "https://user:password@example.invalid/a",
        "https://example.invalid:99999/a",
        "https://example.invalid:invalid/a",
        "https://example.invalid/has space",
        "https://[invalid/a",
    ],
)
def test_redirect_targets_require_the_same_url_policy(url: str) -> None:
    response = BytesIO()
    with pytest.raises(ValueError, match="HTTPS"):
        HTTPSOnlyRedirectHandler().redirect_request(
            Request("https://example.invalid/start"), response, 302, "Found", HTTPMessage(), url
        )
    assert response.closed


def offline_transport(
    monkeypatch: pytest.MonkeyPatch, redirects: dict[str, str], body: bytes = b"fixture"
) -> list[str]:
    visited: list[str] = []

    class OfflineResponse(addinfourl):
        msg: str

    class Transport(HTTPSHandler):
        def https_open(self, req: Request) -> Any:
            visited.append(req.full_url)
            headers = HTTPMessage()
            if req.full_url in redirects:
                headers["Location"] = redirects[req.full_url]
                result = OfflineResponse(BytesIO(), headers, req.full_url, code=302)
                result.msg = "Found"
            else:
                result = OfflineResponse(BytesIO(body), headers, req.full_url, code=200)
                result.msg = "OK"
            return result

    class ForbiddenHTTP(HTTPHandler):
        def http_open(self, req: Request) -> Any:
            raise AssertionError("an insecure target was opened")

    def opener(*handlers: Any) -> Any:
        return build_opener(ProxyHandler({}), Transport(), ForbiddenHTTP(), *handlers)

    monkeypatch.setattr(manifest, "build_opener", opener)
    return visited


def entry() -> ManifestEntry:
    return ManifestEntry(
        source="authored-transport-fixture",
        revision="a" * 40,
        url="https://example.invalid/start",
        license="MIT",
        attribution="Authored fixture",
        sha256=hashlib.sha256(b"fixture").hexdigest(),
        size_bytes=7,
        license_verified=True,
    )


def test_relative_and_cross_host_https_redirects_stream_verified_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    visited = offline_transport(
        monkeypatch,
        {
            "https://example.invalid/start": "/next",
            "https://example.invalid/next": "https://cdn.example.invalid/data?signature=fixture",
        },
    )
    target = download(entry(), tmp_path / "data")
    assert target.read_bytes() == b"fixture"
    assert visited == [
        "https://example.invalid/start",
        "https://example.invalid/next",
        "https://cdn.example.invalid/data?signature=fixture",
    ]


@pytest.mark.parametrize(
    "target", ["http://example.invalid/data", "https://user:password@example.invalid/data"]
)
def test_later_bad_hop_preserves_destination_and_never_opens_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    visited = offline_transport(
        monkeypatch,
        {
            "https://example.invalid/start": "https://cdn.example.invalid/next",
            "https://cdn.example.invalid/next": target,
        },
    )
    path = tmp_path / "data"
    path.write_bytes(b"previous")
    with pytest.raises(ValueError, match="HTTPS"):
        download(entry(), path)
    assert path.read_bytes() == b"previous"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["data"]
    assert visited == ["https://example.invalid/start", "https://cdn.example.invalid/next"]


def test_standard_redirect_loop_limit_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    visited = offline_transport(monkeypatch, {"https://example.invalid/start": "/start"})
    with pytest.raises(HTTPError):
        download(entry(), tmp_path / "data")
    assert 1 < len(visited) <= 10
    assert not list(tmp_path.iterdir())


def test_network_reader_keeps_one_megabyte_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"x" * (2 * 1024 * 1024 + 3)
    offline_transport(monkeypatch, {}, body)
    chunks = list(manifest._network_chunks(entry().url))
    assert list(map(len, chunks)) == [1024 * 1024, 1024 * 1024, 3]
    assert b"".join(chunks) == body


def test_hash_checks_still_apply_after_https_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline_transport(monkeypatch, {entry().url: "/data"})
    with pytest.raises(ValueError, match="hash"):
        download(replace(entry(), sha256="0" * 64), tmp_path / "data")
    assert not list(tmp_path.iterdir())
