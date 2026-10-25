"""Durable artifact storage in one Cloudflare R2 bucket, under a monthly cost cap.

R2 speaks the S3 API, signed with AWS Signature Version 4. This client does three things:

- ``status`` lists the bucket and prices what it holds.
- ``upload`` refuses any upload that would push the bucket's priced storage over the
  cap, then PUTs each file with its SHA-256 as the signed payload hash, so R2 itself
  rejects corrupted bytes. The digest is also stored as object metadata.
- A file already stored under the same key with the same digest is skipped.

Credentials come only from ``R2_ENDPOINT``, ``R2_ACCESS_KEY_ID``,
``R2_SECRET_ACCESS_KEY`` and ``R2_BUCKET``; they are never printed or written. Prices
are the published R2 Standard rates as checked on 2026-10-25.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from scratchcoder.common.records import atomic_json, utc_now
from scratchcoder.data.shards import file_sha256

STORAGE_USD_PER_GB_MONTH = 0.015
FREE_GB_MONTH = 10.0
PRICES_CHECKED_ON = "2026-10-25"
REGION = "auto"
SERVICE = "s3"
S3_NAMESPACE = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def monthly_storage_usd(total_bytes: int) -> float:
    """Priced R2 Standard storage for a month at this size, after the free tier."""
    return max(0.0, total_bytes / 1e9 - FREE_GB_MONTH) * STORAGE_USD_PER_GB_MONTH


@dataclass(frozen=True)
class Credentials:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] = os.environ) -> Credentials:
        values = [
            environ.get(name, "")
            for name in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
        ]
        if not all(values):
            raise ValueError("R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and R2_BUCKET")
        endpoint = urlsplit(values[0])
        if (
            endpoint.scheme != "https"
            or not (endpoint.hostname or "").endswith(".r2.cloudflarestorage.com")
            or endpoint.path not in ("", "/")
            or endpoint.username is not None
        ):
            raise ValueError("R2_ENDPOINT must be https://<account>.r2.cloudflarestorage.com")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", values[3]):
            raise ValueError("invalid bucket name")
        return cls(f"https://{endpoint.hostname}", values[1], values[2], values[3])


def signature_headers(
    credentials: Credentials,
    method: str,
    path: str,
    query: Mapping[str, str],
    headers: Mapping[str, str],
    payload_sha256: str,
    now: dt.datetime,
    region: str = REGION,
) -> dict[str, str]:
    """Signature Version 4 headers for one request (path already URI-encoded)."""
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    day = stamp[:8]
    host = urlsplit(credentials.endpoint).hostname or ""
    signed = {
        **{key.lower(): value.strip() for key, value in headers.items()},
        "host": host,
        "x-amz-content-sha256": payload_sha256,
        "x-amz-date": stamp,
    }
    names = sorted(signed)
    canonical_query = "&".join(
        f"{quote(key, safe='-_.~')}={quote(value, safe='-_.~')}"
        for key, value in sorted(query.items())
    )
    canonical = "\n".join(
        [
            method,
            path,
            canonical_query,
            "".join(f"{name}:{signed[name]}\n" for name in names),
            ";".join(names),
            payload_sha256,
        ]
    )
    scope = f"{day}/{region}/{SERVICE}/aws4_request"
    to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", stamp, scope, hashlib.sha256(canonical.encode()).hexdigest()]
    )
    key = ("AWS4" + credentials.secret_access_key).encode()
    for part in (day, region, SERVICE, "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    authorization = (
        f"AWS4-HMAC-SHA256 Credential={credentials.access_key_id}/{scope}, "
        f"SignedHeaders={';'.join(names)}, Signature={signature}"
    )
    return {
        **{name: signed[name] for name in names if name != "host"},
        "Authorization": authorization,
    }


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_: Any, **__: Any) -> None:
        return None


Transport = Callable[[Request], tuple[int, Mapping[str, str], bytes]]


def _urllib_transport(request: Request) -> tuple[int, Mapping[str, str], bytes]:
    opener = build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=300) as reply:
            return reply.status, dict(reply.headers), reply.read()
    except HTTPError as error:
        body = error.read()
        error.close()
        return error.code, dict(error.headers or {}), body


class Store:
    def __init__(
        self,
        credentials: Credentials,
        *,
        transport: Transport = _urllib_transport,
        clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    ) -> None:
        self.credentials, self.transport, self.clock = credentials, transport, clock

    def _request(
        self,
        method: str,
        key: str = "",
        *,
        query: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: IO[bytes] | bytes | None = None,
        payload_sha256: str = hashlib.sha256(b"").hexdigest(),
    ) -> tuple[int, Mapping[str, str], bytes]:
        path = "/" + quote(self.credentials.bucket) + ("/" + quote(key, safe="/") if key else "")
        query = dict(query or {})
        signed = signature_headers(
            self.credentials, method, path, query, dict(headers or {}), payload_sha256, self.clock()
        )
        url = self.credentials.endpoint + path
        if query:
            url += "?" + "&".join(
                f"{quote(k, safe='-_.~')}={quote(v, safe='-_.~')}" for k, v in sorted(query.items())
            )
        request = Request(url, data=body, method=method, headers=signed)
        return self.transport(request)

    def objects(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        token: str | None = None
        while True:
            query = {"list-type": "2", **({"continuation-token": token} if token else {})}
            status, _, body = self._request("GET", query=query)
            if status != 200:
                raise RuntimeError(f"R2 list failed with HTTP {status}")
            root = ElementTree.fromstring(body)
            for item in root.findall(f"{S3_NAMESPACE}Contents"):
                found.append(
                    {
                        "key": item.findtext(f"{S3_NAMESPACE}Key"),
                        "bytes": int(item.findtext(f"{S3_NAMESPACE}Size") or 0),
                    }
                )
            if root.findtext(f"{S3_NAMESPACE}IsTruncated") != "true":
                return found
            token = root.findtext(f"{S3_NAMESPACE}NextContinuationToken")

    def stored_sha256(self, key: str) -> str | None:
        status, headers, _ = self._request("HEAD", key)
        if status == 404:
            return None
        if status != 200:
            raise RuntimeError(f"R2 HEAD failed with HTTP {status}")
        lowered = {name.lower(): value for name, value in headers.items()}
        return lowered.get("x-amz-meta-sha256")

    def put(self, key: str, path: Path, sha256: str) -> None:
        size = path.stat().st_size
        with path.open("rb") as stream:
            status, _, body = self._request(
                "PUT",
                key,
                headers={"content-length": str(size), "x-amz-meta-sha256": sha256},
                body=stream,
                payload_sha256=sha256,
            )
        if status != 200:
            raise RuntimeError(f"R2 PUT {key} failed with HTTP {status}: {body[:200]!r}")


def upload(
    store: Store,
    files: Iterable[tuple[str, Path]],
    *,
    cap_usd: float,
    receipt: Path,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Upload (key, path) pairs once each, refusing anything that breaks the cap."""
    planned = [(key, path, file_sha256(path), path.stat().st_size) for key, path in files]
    held = {item["key"]: item["bytes"] for item in store.objects()}
    after = {**held, **{key: size for key, _, _, size in planned}}
    projected = monthly_storage_usd(sum(after.values()))
    if projected > cap_usd:
        raise RuntimeError(
            f"storing {sum(after.values()):,} bytes would price at ${projected:.2f}/month, "
            f"above the ${cap_usd:.2f} cap; nothing was uploaded"
        )
    rows = []
    for key, path, sha256, size in planned:
        if store.stored_sha256(key) == sha256:
            method = "present"
        else:
            store.put(key, path, sha256)
            if store.stored_sha256(key) != sha256:
                raise RuntimeError(f"{key}: stored digest differs after upload")
            method = "uploaded"
        rows.append(
            {"key": key, "path": str(path), "bytes": size, "sha256": sha256, "method": method}
        )
        progress(f"{method} {key} ({size:,} bytes)")
    total = sum(item["bytes"] for item in store.objects())
    record = {
        "schema_version": 1,
        "created_at": utc_now(),
        "bucket": store.credentials.bucket,
        "files": rows,
        "bucket_bytes_after": total,
        "monthly_storage_usd_after": monthly_storage_usd(total),
        "cap_usd_per_month": cap_usd,
        "prices": {
            "storage_usd_per_gb_month": STORAGE_USD_PER_GB_MONTH,
            "free_gb_month": FREE_GB_MONTH,
            "checked_on": PRICES_CHECKED_ON,
        },
    }
    atomic_json(receipt, record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "upload"))
    parser.add_argument("--cap-usd", type=float, default=5.0)
    parser.add_argument("--prefix", default="", help="key prefix for uploaded files")
    parser.add_argument("--root", type=Path, help="keys are paths relative to this directory")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    store = Store(Credentials.from_environment())
    if args.command == "status":
        objects = store.objects()
        total = sum(item["bytes"] for item in objects)
        print(
            json.dumps(
                {
                    "objects": len(objects),
                    "bytes": total,
                    "monthly_storage_usd": round(monthly_storage_usd(total), 4),
                }
            )
        )
        return 0
    if args.receipt is None or args.root is None or not args.files:
        parser.error("upload needs --root, --receipt and files")
    pairs = [
        (args.prefix + path.resolve().relative_to(args.root.resolve()).as_posix(), path)
        for path in args.files
    ]
    record = upload(store, pairs, cap_usd=args.cap_usd, receipt=args.receipt)
    print(
        json.dumps(
            {key: record[key] for key in ("bucket_bytes_after", "monthly_storage_usd_after")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
