"""Check, before run 1c's Pod is created, that its image tag still names the pinned digests.

``python -m tools.posttrain_image check --specification <launch spec> --output <file>`` (Mac,
free: one anonymous token request and one manifest read from the registry, no pull, no Pod)
reads the tag's manifest index, hashes the index it received, and compares that digest and the
index's linux/amd64 manifest with the specification's ``runtime.image``
(``posttrain_approved.RUN1C_IMAGE``). It writes ``image-check.json`` either way and exits 0
only when both are the pinned digests: 3 on a mismatch (the tag has moved), 1 when the
registry could not be read. The runbook creates the Pod only on exit 0, so a moved tag stops
the launch before anything bills.

The provider's Pod create body names the image by tag (its API documents tag references only),
so the pull itself is not pinned: the check bounds the window in which the tag could move to
the seconds between this read and the create request. The readiness receipt requires a passing
record made within ``IMAGE_CHECK_WINDOW_SECONDS`` before the create request's billing origin
(``check_image_record``, which the worker applies to the receipt too), and the runtime check
(bootstrap, receipt and worker) still compares what the Pod actually runs with
``runtime.expected``.

The registry token is used for the one manifest read and is never written or printed: a token
that is not header-safe text is refused before any header is built from it, without echoing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

from scratchcoder.common.records import atomic_json

IMAGE_CHECK_KIND = "posttrain-image-check"
IMAGE_CHECK_KEYS = frozenset(
    {"kind", "reference", "expected", "resolved", "checked_at_unix", "mismatched", "status"}
)
IMAGE_CHECK_WINDOW_SECONDS = 300.0
"""A passing check must be at most this long before the create request's billing origin (the
runbook runs it immediately before that request)."""
ORIGIN_ROUNDING_SECONDS = 1.0
"""The runbook's billing origin is ``date +%s``, whole seconds, taken just after the check."""
AUTH_URL = "https://auth.docker.io/token"
REGISTRY_URL = "https://registry-1.docker.io/v2"
OCI_INDEX = "application/vnd.oci.image.index.v1+json"
DOCKER_LIST = "application/vnd.docker.distribution.manifest.list.v2+json"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
ACCEPT = ", ".join((OCI_INDEX, DOCKER_LIST, OCI_MANIFEST, DOCKER_MANIFEST))
TIMEOUT_SECONDS = 30.0
MAX_RESPONSE_BYTES = 1 << 20
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
REFERENCE = re.compile(
    r"(?P<repository>[a-z0-9]+(?:[._-][a-z0-9]+)*/[a-z0-9]+(?:[._-][a-z0-9]+)*)"
    r":(?P<tag>[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})"
)
"""A Docker Hub ``namespace/name:tag`` reference: no registry host and no digest."""
TOKEN = re.compile(r"[A-Za-z0-9._~+/=-]+")
"""A bearer token that is safe in an HTTP header (the registry issues a JWT)."""

Fetch = Callable[[str, Mapping[str, str]], tuple[int, dict[str, str], bytes]]
"""``fetch(url, headers) -> (status, lower-cased headers, body)``; raises ``OSError`` when no
response was obtained."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def http_get(url: str, headers: Mapping[str, str]) -> tuple[int, dict[str, str], bytes]:
    """One GET with no redirect followed and a bounded body."""
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with build_opener(_NoRedirect).open(request, timeout=TIMEOUT_SECONDS) as response:
            items = response.headers.items()
            body = response.read(MAX_RESPONSE_BYTES + 1)
            return response.status, {k.lower(): v for k, v in items}, body
    except HTTPError as error:
        return error.code, {k.lower(): v for k, v in error.headers.items()}, b""
    except (URLError, TimeoutError) as error:
        raise OSError(f"the registry could not be reached: {error}") from error


def split_reference(reference: str) -> tuple[str, str]:
    """A Docker Hub ``namespace/name:tag`` reference's repository and tag."""
    match = REFERENCE.fullmatch(reference)
    if match is None:
        raise ValueError(f"{reference!r} is not a Docker Hub namespace/name:tag reference")
    return match["repository"], match["tag"]


def pinned_image(image: Any) -> dict[str, str]:
    """The specification's ``runtime.image``: its tag and its two digests."""
    if not isinstance(image, Mapping):
        raise ValueError("the specification pins no runtime.image")
    split_reference(str(image.get("reference")))
    for name in ("index_digest", "linux_amd64_digest"):
        if not isinstance(image.get(name), str) or not DIGEST.fullmatch(image[name]):
            raise ValueError(f"runtime.image.{name} is not a sha256 digest")
    return {name: str(image[name]) for name in ("reference", "index_digest", "linux_amd64_digest")}


def resolve(reference: str, *, fetch: Fetch | None = None) -> dict[str, Any]:
    """The digest of the manifest the tag names now, and its linux/amd64 manifest's digest
    (``fetch`` defaults to ``http_get``).

    The digest is the sha256 of the manifest bytes received, whatever the registry states; a
    ``Docker-Content-Digest`` header, when the registry sends one, must equal it. The linux/amd64
    digest is ``None`` unless the manifest is an index or list naming exactly one such manifest.
    """
    fetch = fetch or http_get
    repository, tag = split_reference(reference)
    scope = quote(f"repository:{repository}:pull", safe=":/")
    status, _, body = fetch(f"{AUTH_URL}?service=registry.docker.io&scope={scope}", {})
    if status != 200 or len(body) > MAX_RESPONSE_BYTES:
        raise OSError(f"the registry's token request answered HTTP {status}")
    answer = json.loads(body)
    token = (
        (answer.get("token") or answer.get("access_token")) if isinstance(answer, dict) else None
    )
    if not isinstance(token, str) or not token:
        raise OSError("the registry's token answer holds no token")
    if TOKEN.fullmatch(token) is None:
        raise OSError("the registry token is malformed")  # Never echoed: it is a credential.
    url = f"{REGISTRY_URL}/{repository}/manifests/{tag}"
    status, headers, body = fetch(url, {"Accept": ACCEPT, "Authorization": f"Bearer {token}"})
    if status != 200 or len(body) > MAX_RESPONSE_BYTES:
        raise OSError(f"the registry's manifest request answered HTTP {status}")
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    stated = headers.get("docker-content-digest")
    if stated is not None and stated != digest:
        raise ValueError(
            f"the registry states digest {stated}, but its manifest hashes to {digest}"
        )
    manifest = json.loads(body)
    if not isinstance(manifest, dict):
        raise ValueError("the registry's manifest is not a JSON object")
    content_type = headers.get("content-type", "").split(";", 1)[0].strip()
    media_type = manifest.get("mediaType") or content_type
    amd64: str | None = None
    if media_type in (OCI_INDEX, DOCKER_LIST):
        found = [
            entry.get("digest")
            for entry in manifest.get("manifests") or []
            if isinstance(entry, dict)
            and isinstance(entry.get("platform"), dict)
            and entry["platform"].get("os") == "linux"
            and entry["platform"].get("architecture") == "amd64"
        ]
        if len(found) == 1 and isinstance(found[0], str) and DIGEST.fullmatch(found[0]):
            amd64 = found[0]
    return {"media_type": media_type, "index_digest": digest, "linux_amd64_digest": amd64}


def check_image(
    image: Any, *, fetch: Fetch | None = None, now: float | None = None
) -> dict[str, Any]:
    """The image check's record: ``passed`` only when the tag names exactly the pinned index
    and that index's one linux/amd64 manifest is the pinned one."""
    expected = pinned_image(image)
    resolved = resolve(expected["reference"], fetch=fetch)
    mismatched = [
        name for name in ("index_digest", "linux_amd64_digest") if resolved[name] != expected[name]
    ]
    return {
        "kind": IMAGE_CHECK_KIND,
        "reference": expected["reference"],
        "expected": {name: expected[name] for name in ("index_digest", "linux_amd64_digest")},
        "resolved": resolved,
        "checked_at_unix": time.time() if now is None else now,
        "mismatched": mismatched,
        "status": "failed" if mismatched else "passed",
    }


def check_image_record(
    record: Any, image: Mapping[str, Any], *, billing_origin_unix: float
) -> dict[str, Any]:
    """A passing image check for exactly ``image``, made no more than
    ``IMAGE_CHECK_WINDOW_SECONDS`` before the create request's billing origin. Returns it."""
    expected = pinned_image(image)
    if not isinstance(record, Mapping) or set(record) != IMAGE_CHECK_KEYS:
        raise ValueError("run 1c's image check record is missing or malformed")
    if record["kind"] != IMAGE_CHECK_KIND or record["reference"] != expected["reference"]:
        raise ValueError(f"the image check is not for run 1c's tag {expected['reference']}")
    digests = {name: expected[name] for name in ("index_digest", "linux_amd64_digest")}
    if record["expected"] != digests:
        raise ValueError("the image check expected other digests than the specification's")
    resolved = record["resolved"]
    if (
        not isinstance(resolved, Mapping)
        or any(resolved.get(name) != value for name, value in digests.items())
        or record["mismatched"] != []
        or record["status"] != "passed"
    ):
        raise ValueError(
            f"the image tag did not name run 1c's pinned digests ({record['mismatched']})"
        )
    checked = record["checked_at_unix"]
    if (
        isinstance(checked, bool)
        or not isinstance(checked, int | float)
        or not math.isfinite(checked)
        or not (
            billing_origin_unix - IMAGE_CHECK_WINDOW_SECONDS
            <= checked
            <= billing_origin_unix + ORIGIN_ROUNDING_SECONDS
        )
    ):
        raise ValueError(
            "the image check was not made in the "
            f"{IMAGE_CHECK_WINDOW_SECONDS:.0f} s before the create request"
        )
    return dict(record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.posttrain_image", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="compare the tag's digests with the pinned ones")
    check.add_argument("--specification", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        spec = json.loads(args.specification.read_text(encoding="utf-8"))
        if not isinstance(spec, dict) or spec.get("run") != "1c":
            raise ValueError("the image check is run 1c's: the specification is another run's")
        runtime = spec.get("runtime")
        record = check_image(runtime.get("image") if isinstance(runtime, dict) else None)
    except (OSError, ValueError) as error:
        print(f"posttrain-image: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    atomic_json(args.output, record)
    print(json.dumps({"image_check": str(args.output), **record}, sort_keys=True))
    if record["status"] != "passed":
        print(
            f"posttrain-image: the tag no longer names the pinned {record['mismatched']}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
