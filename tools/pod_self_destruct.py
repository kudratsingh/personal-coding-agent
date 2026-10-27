"""In-pod deadline terminator: the Pod deletes itself at an absolute deadline.

This runs INSIDE the rented Pod, detached from any SSH session, using the
Pod-scoped ``RUNPOD_API_KEY`` that Runpod injects into every Pod. It is
independent of the Mac, the SSH link, the training worker and the Mac-side
watchdog, so it still fires after a Mac disconnect, a worker crash or a GPU
hang. It is NOT a provider-enforced lifetime cap: it cannot fire if the Pod's
host or the Runpod control plane is unavailable, or if the Pod-scoped key is
not permitted to terminate its own Pod (verify on the real Pod before relying
on it). Only the Python standard library is used so the timer works even when
the project environment failed to install.

With ``--idle-seconds`` (post-training runs) the timer also deletes the Pod after that long
with no activity, independent of the Mac. Activity is the latest of the timer's own arming
and, in each ``--activity`` JSON file (the worker's heartbeat, the bootstrap's setup
progress), the ``last_gpu_activity_unix`` and ``progress_unix`` values. A missing,
unreadable, oversized, non-regular or malformed file adds nothing (a FIFO is never opened
for a blocking read, and no content can raise out of the check), and a value more than
``CLOCK_SKEW_SECONDS`` in the future is ignored, so a broken writer can only bring the
deletion closer: the wall-clock and monotonic deadlines are checked before the idle rule on
every pass, whatever the activity files hold. ``progress_unix`` is the documented hook for
waits that keep the Pod busy without GPU work: run 2's reward waits stamp it when the Mac's
reward files arrive.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

API_BASE = "https://api.runpod.io/v2"
API_KEY_ENV = "RUNPOD_API_KEY"
POD_ID_ENV = "RUNPOD_POD_ID"
MAX_RESPONSE_BYTES = 1_048_576
MAX_DEADLINE_SECONDS = 5_400  # The pilot's default Pod lifetime ceiling.
ABSOLUTE_DEADLINE_SECONDS = 60_000  # The rehearsal runs' 16.7-hour limit (D-35); never more.
MIN_IDLE_SECONDS = 60.0
MAX_IDLE_SECONDS = 3_600.0
CLOCK_SKEW_SECONDS = 60.0
MAX_ACTIVITY_BYTES = 65_536
ACTIVITY_FIELDS = ("last_gpu_activity_unix", "progress_unix")
_SAFE_POD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_BACKOFF = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)


def _read_regular(path: str) -> bytes | None:
    """At most ``MAX_ACTIVITY_BYTES + 1`` bytes of a regular file, or None.

    The file is opened without blocking and must be a regular file once open, so a FIFO,
    a device or a socket at an activity path can never stall the timer's loop.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1  # The stream owns it now and closes it.
            return stream.read(MAX_ACTIVITY_BYTES + 1)
    except OSError:
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def activity_unix(path: str, *, now: float) -> float | None:
    """The latest activity stamp one file records, or None when it records none usable.

    Nothing a writer puts in the file can raise here: an unreadable, oversized, non-regular
    or malformed file, a deeply nested document, an integer too large for a float and a
    stamp from the future all read as no activity.
    """
    try:
        raw = _read_regular(path)
        if raw is None or len(raw) > MAX_ACTIVITY_BYTES:
            return None
        value = json.loads(raw)
        if not isinstance(value, dict):
            return None
        found = []
        for name in ACTIVITY_FIELDS:
            stamp = value.get(name)
            if isinstance(stamp, bool) or not isinstance(stamp, int | float):
                continue
            number = float(stamp)  # An integer past the float range raises OverflowError.
            if math.isfinite(number) and 0 < number <= now + CLOCK_SKEW_SECONDS:
                found.append(number)
        return max(found) if found else None
    except Exception:  # Any malformed content is no activity.
        return None


class Transport(Protocol):
    def request(
        self, method: str, url: str, headers: dict[str, str], timeout: float
    ) -> tuple[int, dict[str, str], bytes]: ...


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Request | None:
        return None


class TransportError(RuntimeError):
    """No authoritative HTTP response was obtained."""


class UrllibTransport:
    def request(
        self, method: str, url: str, headers: dict[str, str], timeout: float
    ) -> tuple[int, dict[str, str], bytes]:
        request = Request(url, headers=headers, method=method)
        opener = build_opener(_NoRedirectHandler)
        try:
            with opener.open(request, timeout=timeout) as response:
                return (
                    response.status,
                    {k.lower(): v for k, v in response.headers.items()},
                    response.read(MAX_RESPONSE_BYTES + 1),
                )
        except HTTPError as error:
            return (
                error.code,
                {k.lower(): v for k, v in error.headers.items()},
                error.read(MAX_RESPONSE_BYTES + 1),
            )
        except (URLError, TimeoutError, OSError) as error:
            raise TransportError("Runpod API transport failed") from error


def _atomic_write(path: str, value: dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".self-destruct.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _retry_after(headers: dict[str, str], default: float) -> float:
    raw = headers.get("retry-after")
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        return default
    return value if math.isfinite(value) and 0 < value <= 120 else default


class SelfDestruct:
    """Sleep until a deadline, then terminate the Pod and confirm its absence."""

    def __init__(
        self,
        *,
        pod_id: str,
        token: str,
        deadline_unix: float,
        receipt_path: str,
        transport: Transport | None = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        request_timeout: float = 10.0,
        poll_seconds: float = 5.0,
        stop_file: str | None = None,
        dry_run: bool = False,
        runpodctl: str | None = None,
        idle_seconds: float | None = None,
        activity: tuple[str, ...] = (),
    ) -> None:
        if not _SAFE_POD_ID.fullmatch(pod_id):
            raise ValueError("Pod ID contains unsafe characters")
        if not token and not dry_run:
            raise ValueError("an API token is required unless --dry-run is set")
        if not math.isfinite(deadline_unix) or deadline_unix <= clock():
            raise ValueError("deadline must be a finite future unix timestamp")
        if not 0 < poll_seconds <= 60:
            raise ValueError("poll seconds must be in (0, 60]")
        if idle_seconds is not None and (
            not math.isfinite(idle_seconds)
            or not MIN_IDLE_SECONDS <= idle_seconds <= MAX_IDLE_SECONDS
            or not activity
        ):
            raise ValueError(
                f"an idle limit must lie in [{MIN_IDLE_SECONDS:g}, {MAX_IDLE_SECONDS:g}] s "
                "and name at least one activity file"
            )
        if activity and idle_seconds is None:
            raise ValueError("activity files need an idle limit")
        self.idle_seconds, self.activity = idle_seconds, tuple(activity)
        self.pod_id, self._token, self.deadline = pod_id, token, deadline_unix
        self.receipt_path, self.transport = receipt_path, transport or UrllibTransport()
        self.clock, self.monotonic, self.sleep = clock, monotonic, sleep
        self.request_timeout, self.poll_seconds = request_timeout, poll_seconds
        self.stop_file, self.dry_run, self.runpodctl = stop_file, dry_run, runpodctl
        self.armed_at = clock()
        self.last_activity = self.armed_at
        # Fire on whichever clock reaches the deadline first: a wall-clock jump
        # must not postpone termination, and a monotonic clock cannot be set back.
        self.deadline_monotonic = monotonic() + (deadline_unix - self.armed_at)
        self.attempts: list[dict[str, Any]] = []
        self.state = "armed"
        self.reason = "waiting_for_deadline"
        self.record()

    def record(self) -> None:
        _atomic_write(
            self.receipt_path,
            {
                "schema_version": 1,
                "pod_id": self.pod_id,
                "pid": os.getpid(),
                "armed_at_unix": self.armed_at,
                "deadline_unix": self.deadline,
                "checked_at_unix": self.clock(),
                "state": self.state,
                "reason": self.reason,
                "dry_run": self.dry_run,
                "attempts": self.attempts[-50:],
                "guarantee": ("in-pod best-effort backstop; not a provider-enforced lifetime cap"),
                "idle_seconds": self.idle_seconds,
                "activity_files": list(self.activity),
                "last_activity_unix": self.last_activity,
            },
        )

    def observe_activity(self) -> float:
        """Advance ``last_activity`` to the latest usable stamp; it never moves back."""
        now = self.clock()
        for path in self.activity:
            stamp = activity_unix(path, now=now)
            if stamp is not None and stamp > self.last_activity:
                self.last_activity = stamp
        return self.last_activity

    def _request(self, method: str, path: str) -> tuple[int, dict[str, str], bytes]:
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        return self.transport.request(method, API_BASE + path, headers, self.request_timeout)

    def due(self) -> str | None:
        if self.stop_file and os.path.exists(self.stop_file):
            return "stop_file_present"
        if self.clock() >= self.deadline:
            return "wall_clock_deadline"
        if self.monotonic() >= self.deadline_monotonic:
            return "monotonic_deadline"
        if self.idle_seconds is not None:
            try:
                last = self.observe_activity()
            except Exception:  # A failed read is no new activity.
                last = self.last_activity
            if self.clock() - last >= self.idle_seconds:
                return "idle_limit"
        return None

    def wait(self) -> str:
        while (reason := self.due()) is None:
            remaining = min(
                self.deadline - self.clock(), self.deadline_monotonic - self.monotonic()
            )
            if self.idle_seconds is not None:
                remaining = min(remaining, self.last_activity + self.idle_seconds - self.clock())
            self.reason = f"waiting; {max(0.0, remaining):.0f}s remaining"
            try:
                self.record()
            except OSError:  # A full disk must not end the wait: the deadline still fires.
                pass
            self.sleep(max(0.05, min(self.poll_seconds, remaining)))
        return reason

    def _confirm_absent(self) -> bool:
        for attempt in range(60):
            if attempt:
                self.sleep(2.0)
            try:
                status, headers, body = self._request("GET", f"/pods/{self.pod_id}")
            except TransportError:
                self.attempts.append({"phase": "confirm", "result": "transport_error"})
                self.record()
                continue
            if status == 404:
                return True
            if status == 200:
                try:
                    value = json.loads(body[:MAX_RESPONSE_BYTES])
                except ValueError:
                    value = None
                if isinstance(value, dict) and value.get("status") == "TERMINATED":
                    return True
            if status == 429:
                self.sleep(_retry_after(headers, 5.0))
            self.attempts.append({"phase": "confirm", "result": f"http_{status}"})
            self.record()
        return False

    def _runpodctl_fallback(self) -> bool:
        binary = self.runpodctl or shutil.which("runpodctl")
        if not binary:
            self.attempts.append({"phase": "fallback", "result": "runpodctl_unavailable"})
            return False
        try:
            result = subprocess.run(
                [binary, "pod", "delete", self.pod_id],
                capture_output=True,
                timeout=60,
                check=False,
                env={**os.environ, API_KEY_ENV: self._token},
            )
        except (OSError, subprocess.SubprocessError) as error:
            self.attempts.append({"phase": "fallback", "result": type(error).__name__})
            return False
        self.attempts.append({"phase": "fallback", "result": f"runpodctl_exit_{result.returncode}"})
        return result.returncode == 0

    def terminate(self, trigger: str) -> str:
        """Delete until the API confirms absence; the Pod's death ends this loop."""
        self.state, self.reason = "terminating", trigger
        self.record()
        if self.dry_run:
            self.state, self.reason = "dry_run_complete", trigger + "_no_delete_requested"
            self.record()
            return self.state
        backoff = 0
        while True:
            try:
                status, headers, _ = self._request("DELETE", f"/pods/{self.pod_id}")
            except TransportError:
                self.attempts.append({"phase": "delete", "result": "transport_error"})
                status, headers = -1, {}
            else:
                self.attempts.append({"phase": "delete", "result": f"http_{status}"})
            self.record()
            if status in (204, 404) and self._confirm_absent():
                self.state, self.reason = "terminated", trigger + "_delete_confirmed_absent"
                self.record()
                return self.state
            if status in (401, 403) and self._runpodctl_fallback() and self._confirm_absent():
                self.state, self.reason = "terminated", trigger + "_runpodctl_confirmed_absent"
                self.record()
                return self.state
            delay = _retry_after(headers, _BACKOFF[min(backoff, len(_BACKOFF) - 1)])
            backoff += 1
            self.state, self.reason = "retrying", f"{trigger}_last_http_{status}"
            self.record()
            self.sleep(delay)
            if backoff > 200:  # ~100 minutes of retries; the Pod wall ceiling has long passed.
                self.state, self.reason = "unconfirmed", trigger + "_retries_exhausted"
                self.record()
                return self.state

    def run(self) -> str:
        return self.terminate(self.wait())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", default=os.environ.get(POD_ID_ENV, ""))
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--receipt", required=True, help="atomic JSON status/receipt path")
    parser.add_argument("--stop-file", help="terminate early when this file appears")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="never send a delete request")
    parser.add_argument(
        "--max-lifetime-seconds",
        type=float,
        default=MAX_DEADLINE_SECONDS,
        help="refuse deadlines further than this from now (pilot ceiling 5400)",
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        help="also delete the Pod after this long with no activity in any --activity file",
    )
    parser.add_argument(
        "--activity",
        action="append",
        default=[],
        help="a JSON file whose last_gpu_activity_unix or progress_unix counts as activity",
    )
    args = parser.parse_args(argv)
    token = os.environ.get(API_KEY_ENV, "")
    if not args.pod_id:
        parser.error(f"--pod-id or {POD_ID_ENV} is required")
    if not token and not args.dry_run:
        parser.error(f"{API_KEY_ENV} is not set in the environment")
    now = time.time()
    if not 0 < args.max_lifetime_seconds <= ABSOLUTE_DEADLINE_SECONDS:
        parser.error(f"--max-lifetime-seconds must be at most {ABSOLUTE_DEADLINE_SECONDS}")
    if not now < args.deadline_unix <= now + args.max_lifetime_seconds:
        parser.error("deadline must be in the future and within the lifetime ceiling")
    try:
        timer = SelfDestruct(
            pod_id=args.pod_id,
            token=token,
            deadline_unix=args.deadline_unix,
            receipt_path=args.receipt,
            request_timeout=args.request_timeout_seconds,
            poll_seconds=args.poll_seconds,
            stop_file=args.stop_file,
            dry_run=args.dry_run,
            idle_seconds=args.idle_seconds,
            activity=tuple(args.activity),
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        json.dumps({"armed": True, "pod_id": args.pod_id, "deadline_unix": args.deadline_unix}),
        flush=True,
    )
    state = timer.run()
    print(json.dumps({"state": state, "reason": timer.reason}), file=sys.stderr, flush=True)
    return 0 if state in {"terminated", "dry_run_complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
