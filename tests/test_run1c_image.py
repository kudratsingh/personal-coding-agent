"""Run 1c's image check (``tools.posttrain_image``): before the Pod is created, the tag must
still name the specification's pinned index digest and that index's linux/amd64 manifest.

A fake registry answers every request; nothing touches the network. A Mac-side test (the Mac
makes the check before the create request).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tools import posttrain_approved, posttrain_image

TAG = posttrain_approved.RUN1C_IMAGE["reference"]
NOW = 1_800_000_000.0
RECORDED_INDEX = Path(__file__).parent / "fixtures" / "run1c-image-index.json"
"""The tag's manifest index exactly as Docker Hub served it when run 1c's image was pinned
(public registry metadata: media types, digests, sizes and platforms)."""


def index(amd64: str, *, extra: list[dict[str, Any]] | None = None) -> bytes:
    manifests = [
        {
            "mediaType": posttrain_image.OCI_MANIFEST,
            "digest": amd64,
            "size": 1,
            "platform": {"architecture": "amd64", "os": "linux"},
        },
        {
            "mediaType": posttrain_image.OCI_MANIFEST,
            "digest": "sha256:" + "9" * 64,
            "size": 1,
            "platform": {"architecture": "unknown", "os": "unknown"},
        },
        *(extra or []),
    ]
    value = {"schemaVersion": 2, "mediaType": posttrain_image.OCI_INDEX, "manifests": manifests}
    return json.dumps(value).encode()


class Registry:
    """Answers the token request and the manifest read; records what was asked."""

    def __init__(
        self,
        body: bytes,
        *,
        stated: str | None = None,
        status: int = 200,
        token: str = "t0ken",
        content_type: str = posttrain_image.OCI_INDEX,
        states_digest: bool = True,
    ) -> None:
        self.body = body
        self.stated = "sha256:" + hashlib.sha256(body).hexdigest() if stated is None else stated
        self.status = status
        self.token = token
        self.content_type = content_type
        self.states_digest = states_digest
        self.requests: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: Mapping[str, str]) -> tuple[int, dict[str, str], bytes]:
        self.requests.append((url, dict(headers)))
        if url.startswith(posttrain_image.AUTH_URL):
            return 200, {}, json.dumps({"token": self.token}).encode()
        answer = {"content-type": self.content_type}
        if self.states_digest:
            answer["docker-content-digest"] = self.stated
        return self.status, answer, self.body


def pinned(amd64: str | None = None) -> tuple[dict[str, Any], Registry]:
    """A run-1c-shaped image pin and a registry whose tag names exactly it."""
    amd64 = amd64 or posttrain_approved.RUN1C_IMAGE["linux_amd64_digest"]
    body = index(amd64)
    image = {
        **posttrain_approved.RUN1C_IMAGE,
        "index_digest": "sha256:" + hashlib.sha256(body).hexdigest(),
        "linux_amd64_digest": amd64,
    }
    return image, Registry(body)


def test_the_tag_naming_the_pinned_digests_passes_and_the_token_is_not_recorded() -> None:
    image, registry = pinned()
    record = posttrain_image.check_image(image, fetch=registry, now=NOW)
    assert record["status"] == "passed" and record["mismatched"] == []
    assert record["resolved"]["index_digest"] == image["index_digest"]
    assert record["resolved"]["linux_amd64_digest"] == image["linux_amd64_digest"]
    assert set(record) == posttrain_image.IMAGE_CHECK_KEYS
    assert "t0ken" not in json.dumps(record)
    (token_url, _), (manifest_url, headers) = registry.requests
    assert "scope=repository:runpod/pytorch:pull" in token_url
    assert manifest_url.endswith("/runpod/pytorch/manifests/" + TAG.split(":", 1)[1])
    assert headers["Authorization"] == "Bearer t0ken"
    assert posttrain_image.OCI_INDEX in headers["Accept"]
    origin = NOW + 1
    accepted = posttrain_image.check_image_record(record, image, billing_origin_unix=origin)
    assert accepted == record


def test_the_recorded_index_is_the_pinned_one_and_passes_the_check() -> None:
    body = RECORDED_INDEX.read_bytes()
    pin = posttrain_approved.RUN1C_IMAGE
    assert "sha256:" + hashlib.sha256(body).hexdigest() == pin["index_digest"]
    assert "token" not in body.decode().lower()
    # The content type and digest headers the registry sent with it.
    recorded = Registry(
        body, stated=pin["index_digest"], content_type="application/vnd.oci.image.index.v1+json"
    )
    record = posttrain_image.check_image(pin, fetch=recorded, now=NOW)
    assert record["status"] == "passed" and record["mismatched"] == []
    assert record["resolved"] == {
        "media_type": posttrain_image.OCI_INDEX,
        "index_digest": pin["index_digest"],
        "linux_amd64_digest": pin["linux_amd64_digest"],
    }
    posttrain_image.check_image_record(record, pin, billing_origin_unix=NOW + 1)


@pytest.mark.parametrize("token", ["t0ken\r\nX-Injected: 1", "t0ken secret", "t0ken\x00", "tök"])
def test_a_token_that_is_not_header_safe_is_refused_without_being_echoed(token: str) -> None:
    image, _ = pinned()
    registry = Registry(index(image["linux_amd64_digest"]), token=token)
    with pytest.raises(OSError, match="the registry token is malformed") as raised:
        posttrain_image.check_image(image, fetch=registry, now=NOW)
    assert "t0ken" not in str(raised.value) and "tök" not in str(raised.value)
    assert len(registry.requests) == 1  # The manifest was never requested with it.


def test_a_jwt_shaped_token_is_accepted() -> None:
    image, _ = pinned()
    jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIifQ.c2ln-_+/="
    registry = Registry(index(image["linux_amd64_digest"]), token=jwt)
    assert posttrain_image.check_image(image, fetch=registry, now=NOW)["status"] == "passed"
    assert registry.requests[1][1]["Authorization"] == "Bearer " + jwt


def test_the_digest_is_the_hash_of_what_was_received_when_no_digest_header_is_sent() -> None:
    image, _ = pinned()
    unstated = Registry(index(image["linux_amd64_digest"]), states_digest=False)
    record = posttrain_image.check_image(image, fetch=unstated, now=NOW)
    assert record["status"] == "passed"
    assert record["resolved"]["index_digest"] == image["index_digest"]


def test_a_moved_tag_fails_and_names_what_moved() -> None:
    image, _ = pinned()
    moved = Registry(index("sha256:" + "4" * 64))
    record = posttrain_image.check_image(image, fetch=moved, now=NOW)
    assert record["status"] == "failed"
    assert record["mismatched"] == ["index_digest", "linux_amd64_digest"]
    with pytest.raises(ValueError, match="pinned digests"):
        posttrain_image.check_image_record(record, image, billing_origin_unix=NOW + 1)


def test_an_index_with_no_single_linux_amd64_manifest_fails() -> None:
    image, _ = pinned()
    twice = index(
        image["linux_amd64_digest"],
        extra=[
            {"digest": "sha256:" + "5" * 64, "platform": {"os": "linux", "architecture": "amd64"}}
        ],
    )
    record = posttrain_image.check_image(image, fetch=Registry(twice), now=NOW)
    assert record["resolved"]["linux_amd64_digest"] is None
    assert "linux_amd64_digest" in record["mismatched"] and record["status"] == "failed"


@pytest.mark.parametrize(
    ("registry", "error", "match"),
    [
        (Registry(index("sha256:" + "4" * 64), stated="sha256:" + "6" * 64), ValueError, "hashes"),
        (Registry(b"", status=401), OSError, "HTTP 401"),
        (Registry(b"[]"), ValueError, "not a JSON object"),
    ],
)
def test_an_unreadable_or_inconsistent_registry_answer_is_an_error(
    registry: Registry, error: type[Exception], match: str
) -> None:
    image, _ = pinned()
    with pytest.raises(error, match=match):
        posttrain_image.check_image(image, fetch=registry, now=NOW)


@pytest.mark.parametrize(
    "reference",
    [TAG + "@sha256:" + "a" * 64, "docker.io/" + TAG, "pytorch:latest", "runpod/pytorch"],
)
def test_only_a_docker_hub_tag_reference_is_checked(reference: str) -> None:
    with pytest.raises(ValueError, match="namespace/name:tag"):
        posttrain_image.split_reference(reference)


def test_the_window_is_the_seconds_before_the_create_request() -> None:
    image, registry = pinned()
    record = posttrain_image.check_image(image, fetch=registry, now=NOW)
    window = posttrain_image.IMAGE_CHECK_WINDOW_SECONDS
    for origin in (NOW, NOW + 0.5 - 1, NOW + window):
        posttrain_image.check_image_record(record, image, billing_origin_unix=origin)
    for origin in (NOW - 1.5, NOW + window + 1):
        with pytest.raises(ValueError, match="before the create request"):
            posttrain_image.check_image_record(record, image, billing_origin_unix=origin)


def test_the_command_writes_the_record_and_exits_by_its_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image, registry = pinned()
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"run": "1c", "runtime": {"image": image}}))
    output = tmp_path / "launch" / "image-check.json"
    monkeypatch.setattr(posttrain_image, "http_get", registry)
    argv = ["check", "--specification", str(spec), "--output", str(output)]
    assert posttrain_image.main(argv) == 0
    assert json.loads(output.read_text())["status"] == "passed"
    moved = Registry(index("sha256:" + "4" * 64))
    monkeypatch.setattr(posttrain_image, "http_get", moved)
    assert posttrain_image.main(argv) == 3  # The record is replaced by the failing one.
    assert json.loads(output.read_text())["status"] == "failed"
    spec.write_text(json.dumps({"run": "1b", "runtime": {"image": image}}))
    assert posttrain_image.main(argv) == 1


def test_the_specifications_pin_is_what_the_check_compares() -> None:
    expected = posttrain_image.pinned_image(posttrain_approved.RUN1C_IMAGE)
    assert expected["reference"] == TAG
    spec = posttrain_approved.run1c_specification()
    assert posttrain_image.pinned_image(spec["runtime"]["image"]) == expected
