"""Durable artifact storage in one Cloudflare R2 bucket, under a monthly cost cap.

R2 speaks the S3 API, signed with AWS Signature Version 4. This client does four things:

- ``status`` lists the bucket and prices what it holds.
- ``upload`` refuses any upload that would push the bucket's priced storage over the
  cap, then PUTs each file with its SHA-256 as the signed payload hash, so R2 itself
  rejects corrupted bytes. The digest is also stored as object metadata.
- A file already stored under the same key with the same digest is skipped.
- ``CheckpointUploader`` does the same for training checkpoints on one background
  thread, keeps the newest rolling checkpoints plus every milestone, and records
  failures instead of raising them into the training loop. A milestone that fails is
  retried with backoff until the uploader closes.

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
import threading
import time
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from http.client import HTTPException
from pathlib import Path
from typing import IO, Any
from urllib.error import HTTPError, URLError
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
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MILESTONE_RETRY_SECONDS = 15.0
"""The first wait before a failed milestone is uploaded again; it doubles per failure."""
MILESTONE_RETRY_MAX_SECONDS = 60.0
"""The longest wait between two uploads of a failing milestone."""
MAX_RECORDED_FAILURES = 100
"""``status()`` keeps the newest failures and refusals; ``failure_count`` counts them all."""


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


class TransientError(RuntimeError):
    """An R2 answer worth retrying (HTTP 5xx); still a ``RuntimeError`` for older callers."""


def _status_error(status: int) -> type[RuntimeError]:
    return TransientError if status >= 500 else RuntimeError


def _transient(error: Exception) -> bool:
    """Retry 5xx answers, timeouts and connection failures; never a 4xx."""
    if isinstance(error, HTTPError):
        return error.code >= 500
    return isinstance(
        error, (TransientError, URLError, TimeoutError, ConnectionError, HTTPException)
    )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_: Any, **__: Any) -> None:
        return None


Transport = Callable[[Request], tuple[int, Mapping[str, str], bytes]]
Opener = Callable[[Request], Any]
"""Opens a request and returns a response with ``status``, ``headers`` and ``read(n)``."""


REQUEST_TIMEOUT_SECONDS = 300.0
"""The socket timeout of every request a default ``Store`` makes."""
BOUNDED_REQUEST_TIMEOUT_SECONDS = 20.0
BOUNDED_ATTEMPTS = 2
"""``bounded_store``'s defaults: one call gives up within about 2 x 20 s plus 2 s backoff."""


def _urllib_open(request: Request, *, timeout: float = REQUEST_TIMEOUT_SECONDS) -> Any:
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def _urllib_transport(
    request: Request, *, timeout: float = REQUEST_TIMEOUT_SECONDS
) -> tuple[int, Mapping[str, str], bytes]:
    opener = build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as reply:
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
        opener: Opener = _urllib_open,
        clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
        attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.credentials, self.transport, self.clock = credentials, transport, clock
        self.opener, self.attempts, self.sleep = opener, attempts, sleep

    def _signed(self, method: str, key: str, headers: Mapping[str, str] | None = None) -> Request:
        path = "/" + quote(self.credentials.bucket) + "/" + quote(key, safe="/")
        signed = signature_headers(
            self.credentials,
            method,
            path,
            {},
            dict(headers or {}),
            hashlib.sha256(b"").hexdigest(),
            self.clock(),
        )
        return Request(self.credentials.endpoint + path, method=method, headers=signed)

    def download(self, key: str, destination: Path, *, sha256: str, size: int) -> dict[str, Any]:
        """Stream one object to ``destination``, publishing it only if size and hash match.

        The digest is checked against the caller's pinned value, not the object's
        metadata, which any holder of a write key could set. Transient failures retry
        within the attempt budget, and so does a body that ends early: ``read`` returns
        no more bytes on a clean early close instead of raising ``IncompleteRead``. Only
        more bytes than the pinned size, or a full-length body with another SHA-256,
        raise at once.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        started = time.monotonic()
        for attempt in range(1, self.attempts + 1):
            partial.unlink(missing_ok=True)
            digest, count = hashlib.sha256(), 0
            try:
                with self.opener(self._signed("GET", key)) as reply, partial.open("xb") as out:
                    if getattr(reply, "status", 200) != 200:
                        raise RuntimeError(f"R2 GET {key} answered HTTP {reply.status}")
                    while block := reply.read(8 * 1024 * 1024):
                        count += len(block)
                        if count > size:
                            raise ValueError(f"{key}: more bytes than the pinned size")
                        digest.update(block)
                        out.write(block)
                    out.flush()
                    os.fsync(out.fileno())
                if count < size:
                    raise TransientError(f"{key}: the body ended after {count:,} of {size:,} bytes")
                if digest.hexdigest() != sha256:
                    raise ValueError(f"{key}: SHA-256 differs from the pinned value")
                partial.replace(destination)
                return {
                    "key": key,
                    "bytes": count,
                    "sha256": sha256,
                    "attempts": attempt,
                    "seconds": time.monotonic() - started,
                }
            except ValueError:
                partial.unlink(missing_ok=True)
                raise
            except (OSError, RuntimeError, HTTPException):  # HTTPError is an OSError.
                partial.unlink(missing_ok=True)
                if attempt == self.attempts:
                    raise
                self.sleep(min(30.0, 2.0**attempt))
        raise RuntimeError("unreachable")

    def retry[T](self, action: Callable[[], T]) -> tuple[T, int]:
        """Run ``action`` until it succeeds; return its value and the attempts it took.

        Only transient failures retry, with the same capped backoff as ``download``.
        Anything else, and the last transient failure, is raised.
        """
        for attempt in range(1, self.attempts + 1):
            try:
                return action(), attempt
            except Exception as error:
                if attempt == self.attempts or not _transient(error):
                    raise
            self.sleep(min(30.0, 2.0**attempt))
        raise RuntimeError("unreachable")

    def delete(self, key: str) -> None:
        status, _, _ = self._request("DELETE", key)
        if status not in (200, 204):
            raise _status_error(status)(f"R2 DELETE {key} failed with HTTP {status}")

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
                raise _status_error(status)(f"R2 list failed with HTTP {status}")
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

    def head(self, key: str) -> dict[str, str] | None:
        """One HEAD request: the object's headers with lower-case names, or None if absent."""
        status, headers, _ = self._request("HEAD", key)
        if status == 404:
            return None
        if status != 200:
            raise _status_error(status)(f"R2 HEAD failed with HTTP {status}")
        return {name.lower(): value for name, value in headers.items()}

    def stored_sha256(self, key: str) -> str | None:
        headers = self.head(key)
        return None if headers is None else headers.get("x-amz-meta-sha256")

    def _put(self, key: str, path: Path, sha256: str, size: int) -> None:
        with path.open("rb") as stream:
            status, _, body = self._request(
                "PUT",
                key,
                headers={"content-length": str(size), "x-amz-meta-sha256": sha256},
                body=stream,
                payload_sha256=sha256,
            )
        if status != 200:
            raise _status_error(status)(f"R2 PUT {key} failed with HTTP {status}: {body[:200]!r}")

    def put(self, key: str, path: Path, sha256: str) -> None:
        self._put(key, path, sha256, path.stat().st_size)

    def put_known(self, key: str, path: Path, sha256: str, size: int) -> int:
        """PUT a file whose digest the caller already holds; return the attempts it took.

        The file is not read to hash it again. ``sha256`` is signed as the payload hash,
        so R2 rejects any bytes that differ (a 4xx, never retried). HTTP 5xx answers,
        timeouts and connection errors retry with backoff, reopening the file each time.
        """
        if not SHA256_HEX.fullmatch(sha256):
            raise ValueError(f"{key}: sha256 must be 64 lower-case hex digits")
        if path.stat().st_size != size:
            raise ValueError(f"{key}: {path} is not the declared {size:,} bytes")
        _, attempts = self.retry(partial(self._put, key, path, sha256, size))
        return attempts


def bounded_store(
    credentials: Credentials,
    *,
    timeout: float = BOUNDED_REQUEST_TIMEOUT_SECONDS,
    attempts: int = BOUNDED_ATTEMPTS,
) -> Store:
    """A store for checks that must end in time: each socket operation of a request times
    out after ``timeout`` seconds, and a call makes at most ``attempts`` tries.

    The default ``Store`` waits up to 300 s per request and tries four times, about 20
    minutes against an R2 endpoint that accepts connections but never answers. That is
    right for bulk transfers and wrong for a check that a deadline depends on: the worker's
    checks after its final drain and the Mac supervisor's listing and HEAD checks.
    """
    if not 0 < timeout <= REQUEST_TIMEOUT_SECONDS or attempts < 1:
        raise ValueError("a bounded store needs a positive timeout and at least one attempt")
    return Store(
        credentials,
        transport=partial(_urllib_transport, timeout=timeout),
        opener=partial(_urllib_open, timeout=timeout),
        attempts=attempts,
    )


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


@dataclass(frozen=True)
class _Job:
    label: str
    files: tuple[tuple[str, Path, str], ...]
    milestone: bool
    due: float = field(default=0.0, compare=False)
    """``time.monotonic()`` before which a retried milestone waits; 0 means now."""
    failures: int = field(default=0, compare=False)
    """Failed uploads of this milestone so far, which set its backoff."""


@dataclass(frozen=True)
class _Rolling:
    """A rolling label's keys in the bucket; ``receipt`` is None if it never verified."""

    label: str
    keys: tuple[str, ...]
    receipt: dict[str, Any] | None


class CheckpointUploader:
    """Upload checkpoints to R2 on one background thread, so training never waits.

    ``submit`` only queues work. Any submission drops the rolling labels still waiting,
    which are recorded as superseded: a newer rolling label or a milestone holds newer
    weights, so a milestone never waits behind a rolling label that has not started.
    Milestones are never dropped. Each label is checked against the cost cap, uploaded,
    verified with HEAD and given a receipt. A milestone that fails or is refused is
    queued again with backoff (``retry_seconds``, doubling up to ``retry_max_seconds``)
    until the uploader closes; submitting it again makes it due at once. Due milestones
    go before rolling labels. After a rolling label verifies, older rolling labels beyond
    ``keep_rolling`` are deleted, as are the keys any older failed rolling label managed
    to store. Milestones are never deleted. Worker errors are recorded in ``status()``;
    they are never raised on the caller's thread.

    Submitted files must stay unchanged until their label leaves the queue. Retention
    only knows the labels this instance uploaded; earlier processes' are left alone.
    """

    def __init__(
        self,
        store: Store,
        *,
        prefix: str,
        cap_usd: float,
        receipt_dir: Path,
        keep_rolling: int = 2,
        clock: Callable[[], float] = time.monotonic,
        retry_seconds: float = MILESTONE_RETRY_SECONDS,
        retry_max_seconds: float = MILESTONE_RETRY_MAX_SECONDS,
    ) -> None:
        if prefix.startswith("/") or (prefix and not prefix.endswith("/")):
            raise ValueError("prefix must be empty or a relative path ending in '/'")
        if keep_rolling < 1:
            raise ValueError("keep_rolling must be at least 1")
        if not 0 < retry_seconds <= retry_max_seconds:
            raise ValueError("retry_seconds must be positive and at most retry_max_seconds")
        self.store, self.prefix, self.cap_usd = store, prefix, cap_usd
        self.receipt_dir, self.keep_rolling, self.clock = receipt_dir, keep_rolling, clock
        self.retry_seconds, self.retry_max_seconds = retry_seconds, retry_max_seconds
        # Everything below is shared with the caller's thread: touch it only under the lock.
        self._condition = threading.Condition()
        self._pending: list[_Job] = []
        self._in_flight: _Job | None = None
        self._closed = False
        self._last_verified: str | None = None
        self._verified: list[str] = []
        self._milestones: set[str] = set()
        self._rolling: list[_Rolling] = []
        self._superseded: list[str] = []
        self._cancelled: list[str] = []
        self._deleted: list[str] = []
        self._failures: list[dict[str, str]] = []
        self._failure_count = 0
        self._refused: list[dict[str, Any]] = []
        self._thread = threading.Thread(target=self._work, name="r2-uploader", daemon=True)
        self._thread.start()

    def submit(
        self, label: str, files: Sequence[tuple[str, Path, str]], *, milestone: bool
    ) -> None:
        """Queue (relative key, path, SHA-256) files under ``prefix + label + "/"``.

        The same label and files already waiting are replaced in place and due at once;
        the same label and files already uploading are left to finish (a milestone is
        retried if that upload fails). Malformed arguments raise here; nothing else
        does, and no I/O happens.
        """
        if not LABEL.fullmatch(label):
            raise ValueError(f"invalid label {label!r}")
        job = _Job(
            label, tuple((key, Path(path), digest) for key, path, digest in files), milestone
        )
        keys = [key for key, _, _ in job.files]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError(f"{label}: needs at least one file and distinct keys")
        for key, _, digest in job.files:
            if key.startswith("/") or any(part in ("", ".", "..") for part in key.split("/")):
                raise ValueError(f"{label}: invalid key {key!r}")
            if not SHA256_HEX.fullmatch(digest):
                raise ValueError(f"{label}: {key} sha256 must be 64 lower-case hex digits")
        with self._condition:
            if self._closed:
                raise RuntimeError("the uploader is closed")
            self._superseded += [
                older.label for older in self._pending if not older.milestone and older != job
            ]
            self._pending = [older for older in self._pending if older.milestone or older == job]
            if self._in_flight != job:
                waiting = [index for index, older in enumerate(self._pending) if older == job]
                if waiting:
                    self._pending[waiting[0]] = job  # Due now, its backoff restarted.
                else:
                    self._pending.append(job)
            self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        """A snapshot that is safe to log: labels, failures and refusals so far.

        ``retrying`` maps each milestone waiting to be uploaded again to its failures;
        ``failures`` and ``refused`` keep the newest ``MAX_RECORDED_FAILURES`` entries.
        """
        with self._condition:
            return {
                "last_verified": self._last_verified,
                "in_flight": None if self._in_flight is None else self._in_flight.label,
                "pending": [job.label for job in self._pending],
                "retrying": {job.label: job.failures for job in self._pending if job.failures},
                "verified": list(self._verified),
                "rolling": [item.label for item in self._rolling if item.receipt is not None],
                "superseded": list(self._superseded),
                "deleted": list(self._deleted),
                "cancelled": list(self._cancelled),
                "failures": [dict(item) for item in self._failures],
                "failure_count": self._failure_count,
                "refused": [dict(item) for item in self._refused],
                "closed": self._closed,
            }

    def drain(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for all queued work; False if it is still running.

        A milestone waiting to be retried is queued work: it keeps ``drain`` waiting.
        """
        with self._condition:
            return self._condition.wait_for(
                lambda: not self._pending and self._in_flight is None, timeout
            )

    def close(self, timeout: float | None = None) -> bool:
        """Refuse new work, cancel uploads not yet started and wait for the one in flight.

        Call ``drain`` first to keep queued work. True once the worker has exited.
        """
        with self._condition:
            if not self._closed:
                self._closed = True
                self._cancelled += [job.label for job in self._pending]
                self._pending.clear()
                self._condition.notify_all()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _next(self) -> _Job | None:
        """Under the lock: wait for the next due job, a due milestone first; None once closed."""
        while True:
            if not self._pending:
                if self._closed:
                    return None
                self._condition.wait()
                continue
            now = time.monotonic()
            due = [job for job in self._pending if job.due <= now]
            if not due:
                self._condition.wait(min(job.due for job in self._pending) - now)
                continue
            job = next((job for job in due if job.milestone), due[0])
            self._pending = [other for other in self._pending if other is not job]
            return job

    def _work(self) -> None:
        while True:
            with self._condition:
                job = self._next()
                if job is None:
                    return
                self._in_flight = job
            stored: list[str] = []
            verified = False
            try:
                verified = self._upload(job, stored)
            except Exception as error:  # recorded, never raised: the caller is training
                self._fail(job.label, "upload", error)
                if stored and not job.milestone:
                    self._remember(_Rolling(job.label, tuple(stored), None))
            finally:
                with self._condition:
                    self._in_flight = None
                    if job.milestone and not verified:
                        self._retry(job)
                    self._condition.notify_all()

    def _retry(self, job: _Job) -> None:
        """Under the lock: queue a failed or refused milestone again after its backoff."""
        if self._closed or any(other == job for other in self._pending):
            return  # Closing cancels it; a resubmission of the label already waits.
        failures = job.failures + 1
        delay = min(self.retry_max_seconds, self.retry_seconds * 2.0 ** (failures - 1))
        self._pending.append(replace(job, due=time.monotonic() + delay, failures=failures))

    def _keys(self, job: _Job) -> tuple[str, ...]:
        return tuple(self.prefix + job.label + "/" + key for key, _, _ in job.files)

    def _fail(self, label: str, stage: str, error: Exception) -> None:
        with self._condition:
            self._failure_count += 1
            self._failures.append(
                {"label": label, "stage": stage, "error": f"{type(error).__name__}: {error}"}
            )
            del self._failures[:-MAX_RECORDED_FAILURES]

    def _remember(self, item: _Rolling) -> bool:
        """List a rolling label for retention, unless its keys now hold a milestone."""
        with self._condition:
            self._rolling = [old for old in self._rolling if old.label != item.label]
            if item.label in self._milestones:
                return False
            self._rolling.append(item)
            return True

    def _upload(self, job: _Job, stored: list[str]) -> bool:
        """Upload and verify one label; False if the cost cap refused it."""
        planned = [
            (key, path, digest, path.stat().st_size)
            for key, (_, path, digest) in zip(self._keys(job), job.files, strict=True)
        ]
        held, _ = self.store.retry(self.store.objects)
        # A key already held is replaced; rolling labels due for deletion still count.
        after = {item["key"]: item["bytes"] for item in held}
        after.update({key: size for key, _, _, size in planned})
        bucket_bytes = sum(after.values())
        projected = monthly_storage_usd(bucket_bytes)
        if projected > self.cap_usd:
            with self._condition:
                self._refused.append(
                    {
                        "label": job.label,
                        "bytes": sum(size for _, _, _, size in planned),
                        "bucket_bytes_after": bucket_bytes,
                        "monthly_storage_usd_after": projected,
                        "cap_usd_per_month": self.cap_usd,
                    }
                )
                del self._refused[:-MAX_RECORDED_FAILURES]
            return False
        started = self.clock()
        rows: list[dict[str, Any]] = []
        for key, path, digest, size in planned:
            began = self.clock()
            attempts = self.store.put_known(key, path, digest, size)
            stored.append(key)
            found, _ = self.store.retry(partial(self.store.stored_sha256, key))
            if found != digest:
                raise RuntimeError(f"{key}: stored digest differs after upload")
            rows.append(
                {
                    "key": key,
                    "path": str(path),
                    "bytes": size,
                    "sha256": digest,
                    "attempts": attempts,
                    "seconds": self.clock() - began,
                }
            )
        record: dict[str, Any] = {
            "schema_version": 1,
            "created_at": utc_now(),
            "bucket": self.store.credentials.bucket,
            "label": job.label,
            "milestone": job.milestone,
            "keys": [row["key"] for row in rows],
            "files": rows,
            "bytes": sum(size for _, _, _, size in planned),
            "attempts": sum(row["attempts"] for row in rows),
            "seconds": self.clock() - started,
            "bucket_bytes_after": bucket_bytes,
            "monthly_storage_usd_after": projected,
            "cap_usd_per_month": self.cap_usd,
            "prices": {
                "storage_usd_per_gb_month": STORAGE_USD_PER_GB_MONTH,
                "free_gb_month": FREE_GB_MONTH,
                "checked_on": PRICES_CHECKED_ON,
            },
        }
        atomic_json(self.receipt_dir / f"{job.label}.json", record)
        with self._condition:
            self._last_verified = job.label
            self._verified.append(job.label)
            if job.milestone:
                self._milestones.add(job.label)
                self._rolling = [old for old in self._rolling if old.label != job.label]
        if not job.milestone and self._remember(_Rolling(job.label, tuple(stored), record)):
            self._retain()
        return True

    def _retain(self) -> None:
        """Delete failed rolling labels and verified ones beyond ``keep_rolling``, oldest first."""
        with self._condition:
            older = self._rolling[:-1]  # the last one was just verified and always stays
        kept = sum(item.receipt is not None for item in older) + 1
        for item in older:
            if item.receipt is not None:
                if kept <= self.keep_rolling:
                    continue
                kept -= 1
            try:
                for key in item.keys:
                    self.store.retry(partial(self.store.delete, key))
                if item.receipt is not None:
                    deleted = {**item.receipt, "deleted_at": utc_now()}
                    atomic_json(self.receipt_dir / f"{item.label}.json", deleted)
            except Exception as error:  # left listed, so the next verified label retries it
                self._fail(item.label, "retention", error)
                continue
            with self._condition:
                self._rolling.remove(item)
                self._deleted.append(item.label)


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
