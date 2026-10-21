"""Best-effort Runpod Pod watchdog using only the REST v2 read/delete surface.

This process is independent of the training worker, but it cannot guarantee a
provider-enforced deadline: a failed host, network, or Runpod API can prevent
termination. Configure any provider-side termination deadline separately.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeGuard
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

API_BASE = "https://api.runpod.io/v2"
API_KEY_ENV = "RUNPOD_API_KEY"
MAX_RESPONSE_BYTES = 1_048_576
MAX_POD_WALL_SECONDS = 5_400
MAX_GPU_IDLE_SECONDS = 600
_SAFE_POD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def _is_number(value: object) -> TypeGuard[int | float]:
    return type(value) in (int, float)


class Transport(Protocol):
    def request(
        self, method: str, url: str, headers: dict[str, str], timeout: float
    ) -> tuple[int, bytes]: ...


class UrllibTransport:
    """HTTP transport that refuses redirects, preventing bearer-token forwarding."""

    def request(
        self, method: str, url: str, headers: dict[str, str], timeout: float
    ) -> tuple[int, bytes]:
        request = Request(url, headers=headers, method=method)
        opener = build_opener(_NoRedirectHandler)
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.status, response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            return error.code, error.read(MAX_RESPONSE_BYTES + 1)
        except (URLError, TimeoutError, OSError) as error:
            raise ApiTransportError("Runpod API transport failed") from error


class ApiTransportError(RuntimeError):
    """Transport could not establish an authoritative API response."""


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Request | None:
        return None


@dataclass(frozen=True)
class PodIdentity:
    pod_id: str
    name: str
    gpu_id: str
    gpu_count: int
    data_center_id: str
    created_at: str

    @classmethod
    def from_dict(cls, value: object) -> PodIdentity:
        if not isinstance(value, dict):
            raise ValueError("identity must be a JSON object")
        required = ("id", "name", "gpuId", "gpuCount", "dataCenterId", "createdAt")
        if any(key not in value for key in required):
            raise ValueError("identity is missing a required field")
        strings = ("id", "name", "gpuId", "dataCenterId", "createdAt")
        if any(not isinstance(value[key], str) or not value[key] for key in strings):
            raise ValueError("identity string fields must be nonempty")
        if not _SAFE_POD_ID.fullmatch(value["id"]):
            raise ValueError("Pod ID contains unsafe path characters")
        gpu_count = value["gpuCount"]
        if type(gpu_count) is not int or gpu_count <= 0:
            raise ValueError("gpuCount must be a positive integer")
        parse_timestamp(value["createdAt"])
        return cls(
            pod_id=value["id"],
            name=value["name"],
            gpu_id=value["gpuId"],
            gpu_count=gpu_count,
            data_center_id=value["dataCenterId"],
            created_at=value["createdAt"],
        )

    def matches(self, pod: object) -> bool:
        if not isinstance(pod, dict):
            return False
        gpu = pod.get("gpu")
        return (
            pod.get("id") == self.pod_id
            and pod.get("name") == self.name
            and isinstance(gpu, dict)
            and gpu.get("id") == self.gpu_id
            and type(gpu.get("count")) is int
            and gpu.get("count") == self.gpu_count
            and pod.get("dataCenterId") == self.data_center_id
            and pod.get("createdAt") == self.created_at
        )


@dataclass(frozen=True)
class Policy:
    max_wall_seconds: float
    max_cost_usd: float
    max_idle_seconds: float
    quoted_total_hourly: float
    billing_started_at_unix: float | None = None
    request_timeout_seconds: float = 5.0
    delete_verify_attempts: int = 3
    delete_verify_delay_seconds: float = 1.0
    starting_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        for key in (
            "max_wall_seconds",
            "max_cost_usd",
            "max_idle_seconds",
            "quoted_total_hourly",
            "request_timeout_seconds",
        ):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be a finite positive number")
        if self.max_wall_seconds > MAX_POD_WALL_SECONDS:
            raise ValueError("max_wall_seconds cannot exceed the 5400-second pilot ceiling")
        if (
            not _is_number(self.starting_cost_usd)
            or not math.isfinite(self.starting_cost_usd)
            or not 0 <= self.starting_cost_usd < self.max_cost_usd
        ):
            raise ValueError("starting cost must be finite, nonnegative and below the stop cost")
        if self.max_idle_seconds > MAX_GPU_IDLE_SECONDS:
            raise ValueError("max_idle_seconds cannot exceed the 600-second pilot ceiling")
        if self.billing_started_at_unix is not None and (
            not _is_number(self.billing_started_at_unix)
            or not math.isfinite(self.billing_started_at_unix)
        ):
            raise ValueError("billing_started_at_unix must be finite when provided")
        if type(self.delete_verify_attempts) is not int or self.delete_verify_attempts < 1:
            raise ValueError("delete_verify_attempts must be a positive integer")
        if (
            type(self.delete_verify_delay_seconds) not in (int, float)
            or not math.isfinite(self.delete_verify_delay_seconds)
            or self.delete_verify_delay_seconds < 0
        ):
            raise ValueError("delete verification delay must be finite and nonnegative")


@dataclass(frozen=True)
class Outcome:
    state: str
    reason: str
    pod_id: str
    checked_at_unix: float
    detail: str | None = None


def parse_timestamp(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp is not ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    try:
        timestamp = parsed.astimezone(UTC).timestamp()
    except (OverflowError, OSError, ValueError) as error:
        raise ValueError("timestamp is outside the supported range") from error
    if not math.isfinite(timestamp):
        raise ValueError("timestamp is not finite")
    return timestamp


@dataclass(frozen=True)
class Heartbeat:
    last_gpu_activity_unix: float
    error: str | None
    status: str | None
    shutdown_receipts_verified: bool = False


def read_heartbeat(path: Path, *, now: float, max_idle_seconds: float) -> Heartbeat:
    """Return last GPU activity and a failure reason; absent means just-created."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return Heartbeat(0.0, None, None)
    except OSError:
        return Heartbeat(0.0, "heartbeat_unreadable", None)
    if len(raw) > 65_536:
        return Heartbeat(0.0, "heartbeat_malformed", None)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return Heartbeat(0.0, "heartbeat_malformed", None)
    if not isinstance(value, dict):
        return Heartbeat(0.0, "heartbeat_malformed", None)
    activity = value.get("last_gpu_activity_unix")
    updated = value.get("updated_at_unix")
    step = value.get("step")
    worker_status = value.get("status")
    allowed_statuses = {
        "running",
        "awaiting_transfer",
        "completed",
        "time_limited",
        "shutdown_ready",
        "stopped",
        "failed",
    }
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or not isinstance(value.get("phase"), str)
        or not isinstance(worker_status, str)
        or worker_status not in allowed_statuses
        or type(step) is not int
        or step < 0
        or not _is_number(activity)
        or not math.isfinite(activity)
        or activity < 0
        or not _is_number(updated)
        or not math.isfinite(updated)
        or updated < 0
    ):
        return Heartbeat(0.0, "heartbeat_malformed", None)
    if activity > updated:
        return Heartbeat(float(activity), "heartbeat_malformed", worker_status)
    if updated > now:
        return Heartbeat(float(activity), "heartbeat_malformed", worker_status)
    if now - updated >= max_idle_seconds:
        return Heartbeat(float(activity), "heartbeat_stale", worker_status)
    receipts_verified = (
        worker_status == "shutdown_ready"
        and value.get("mac_receipt_verified") is True
        and value.get("github_receipt_verified") is True
    )
    return Heartbeat(float(activity), None, worker_status, receipts_verified)


def _decode_object(body: bytes) -> dict[str, object] | None:
    if len(body) > MAX_RESPONSE_BYTES:
        return None
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


class RunpodClient:
    def __init__(self, token: str, transport: Transport) -> None:
        if not token:
            raise ValueError("API token must not be empty")
        self._token = token
        self._transport = transport

    def _request(self, method: str, url: str, timeout: float) -> tuple[int, bytes]:
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        return self._transport.request(method, url, headers, timeout)

    def get_pod(self, pod_id: str, timeout: float) -> tuple[int, bytes]:
        safe_id = _validated_pod_id(pod_id)
        return self._request("GET", f"{API_BASE}/pods/{safe_id}", timeout)

    def delete_pod(self, pod_id: str, timeout: float) -> tuple[int, bytes]:
        safe_id = _validated_pod_id(pod_id)
        return self._request("DELETE", f"{API_BASE}/pods/{safe_id}", timeout)


def _validated_pod_id(pod_id: str) -> str:
    if not isinstance(pod_id, str) or not _SAFE_POD_ID.fullmatch(pod_id):
        raise ValueError("Pod ID contains unsafe path characters")
    return pod_id


def atomic_receipt(path: Path, outcome: Outcome) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(asdict(outcome), stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _outcome(
    state: str, reason: str, identity: PodIdentity, now: float, detail: str | None = None
) -> Outcome:
    return Outcome(state, reason, identity.pod_id, now, detail)


def check_once(
    client: RunpodClient,
    identity: PodIdentity,
    policy: Policy,
    heartbeat_path: Path,
    *,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> Outcome:
    """Inspect once, terminating only after exact captured identity matches."""
    raw_now = now()
    clock_valid = _is_number(raw_now) and math.isfinite(raw_now)
    checked_at = float(raw_now) if clock_valid else 0.0
    heartbeat = read_heartbeat(
        heartbeat_path, now=checked_at, max_idle_seconds=policy.max_idle_seconds
    )
    try:
        status, body = client.get_pod(identity.pod_id, policy.request_timeout_seconds)
    except ApiTransportError:
        return _outcome("unknown", "pod_get_transport_error", identity, checked_at)
    if status == 404:
        return _outcome("absent", "pod_get_404", identity, checked_at)
    if status != 200:
        return _outcome("unknown", f"pod_get_http_{status}", identity, checked_at)
    pod = _decode_object(body)
    if pod is None:
        return _outcome("unknown", "pod_get_malformed_json", identity, checked_at)
    if not identity.matches(pod):
        return _outcome("identity_mismatch", "identity_mismatch_no_delete", identity, checked_at)

    trigger = ""
    if not clock_valid:
        trigger = "watchdog_clock_invalid"
    else:
        try:
            created_at = parse_timestamp(identity.created_at)
            billing_started = policy.billing_started_at_unix
            if billing_started is not None and billing_started > created_at:
                trigger = "billing_start_after_pod_creation"
                elapsed = 0.0
            else:
                elapsed = checked_at - min(
                    created_at,
                    billing_started if billing_started is not None else created_at,
                )
            if elapsed < 0:
                trigger = "invalid_clock_or_created_at"
            elif elapsed >= policy.max_wall_seconds:
                trigger = "wall_limit"
            elif trigger == "billing_start_after_pod_creation":
                pass
            else:
                raw_hourly_cost = pod.get("cost")
                if (
                    not _is_number(raw_hourly_cost)
                    or not math.isfinite(raw_hourly_cost)
                    or raw_hourly_cost < 0
                ):
                    trigger = "cost_unknown"
                elif (
                    policy.starting_cost_usd
                    + elapsed * max(float(raw_hourly_cost), policy.quoted_total_hourly) / 3600
                    >= policy.max_cost_usd
                ):
                    trigger = "cost_limit"
                elif heartbeat.error is not None:
                    trigger = heartbeat.error
                elif heartbeat.status == "shutdown_ready" and heartbeat.shutdown_receipts_verified:
                    trigger = "shutdown_ready_verified_receipts"
                elif heartbeat.last_gpu_activity_unix == 0.0:
                    # On first launch, no heartbeat yet: use Pod creation as idle baseline.
                    if checked_at - created_at >= policy.max_idle_seconds:
                        trigger = "idle_gpu_limit"
                    else:
                        trigger = ""
                elif (
                    not math.isfinite(heartbeat.last_gpu_activity_unix)
                    or heartbeat.last_gpu_activity_unix > checked_at
                ):
                    trigger = "heartbeat_malformed"
                elif checked_at - heartbeat.last_gpu_activity_unix >= policy.max_idle_seconds:
                    trigger = "idle_gpu_limit"
                else:
                    trigger = ""
        except (ValueError, OverflowError):
            trigger = "guard_calculation_error"

    if not trigger:
        return _outcome("watching", "within_limits", identity, checked_at)

    try:
        delete_status, _ = client.delete_pod(identity.pod_id, policy.request_timeout_seconds)
    except ApiTransportError:
        return _outcome("unknown", trigger + "_delete_transport_error", identity, checked_at)
    if delete_status != 204:
        return _outcome("unknown", trigger + f"_delete_http_{delete_status}", identity, checked_at)

    for attempt in range(policy.delete_verify_attempts):
        if attempt:
            sleep(policy.delete_verify_delay_seconds)
        try:
            verify_status, verify_body = client.get_pod(
                identity.pod_id, policy.request_timeout_seconds
            )
        except ApiTransportError:
            return _outcome(
                "unknown", trigger + "_delete_unconfirmed_transport_error", identity, checked_at
            )
        if verify_status == 404:
            return _outcome("terminated", trigger + "_delete_204_get_404", identity, checked_at)
        if verify_status != 200:
            return _outcome(
                "unknown",
                trigger + f"_delete_unconfirmed_http_{verify_status}",
                identity,
                checked_at,
            )
        remaining = _decode_object(verify_body)
        if remaining is None:
            return _outcome(
                "unknown", trigger + "_delete_unconfirmed_malformed_json", identity, checked_at
            )
        if not identity.matches(remaining):
            return _outcome(
                "identity_mismatch",
                trigger + "_post_delete_identity_mismatch",
                identity,
                checked_at,
            )
    return _outcome("unknown", trigger + "_delete_still_present", identity, checked_at)


def _load_identity(path: Path) -> PodIdentity:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read identity JSON") from error
    return PodIdentity.from_dict(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True, help="captured Pod identity JSON")
    parser.add_argument("--heartbeat", type=Path, required=True, help="worker heartbeat JSON")
    parser.add_argument(
        "--receipt", type=Path, required=True, help="atomic local watchdog receipt path"
    )
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--max-idle-seconds", type=float, default=600)
    parser.add_argument("--quoted-total-hourly", type=float, required=True)
    parser.add_argument("--starting-cost-usd", type=float, default=0.0)
    parser.add_argument("--billing-started-at-unix", type=float)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--delete-verify-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be finite and positive")
    token = os.environ.get(API_KEY_ENV, "")
    if not token:
        parser.error(f"set {API_KEY_ENV} in the local environment")
    try:
        identity = _load_identity(args.identity)
        policy = Policy(
            args.max_wall_seconds,
            args.max_cost_usd,
            args.max_idle_seconds,
            args.quoted_total_hourly,
            args.billing_started_at_unix,
            args.request_timeout_seconds,
            args.delete_verify_attempts,
            starting_cost_usd=args.starting_cost_usd,
        )
    except ValueError as error:
        parser.error(str(error))
    client = RunpodClient(token, UrllibTransport())
    while True:
        outcome = check_once(client, identity, policy, args.heartbeat)
        atomic_receipt(args.receipt, outcome)
        if outcome.state in {"terminated", "absent", "identity_mismatch"}:
            return 0 if outcome.state != "identity_mismatch" else 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
