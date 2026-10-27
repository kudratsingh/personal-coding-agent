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
import signal
import sys
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
# The owner-approved rehearsal runs (D-35) have a 60,000-second (16.7-hour) Pod limit, so the
# $10 cap at $0.60/h is the binding limit; the pilot keeps 5,400.
# The post-training runs keep the same absolute 60,000-second ceiling here; each run's own wall
# is derived from its ceiling (run 1: $3.00 at $0.60/h, 18,000 s) and enforced by its policy file,
# which the readiness receipt checks against that derivation.
MODE_WALL_SECONDS = {"pilot": MAX_POD_WALL_SECONDS, "rehearsal": 60_000, "posttrain": 60_000}
MAX_GPU_IDLE_SECONDS = 600
MAX_TRANSFER_SECONDS = 1_800
CLOCK_SKEW_SECONDS = 60.0  # Tolerated difference between Pod, provider and Mac clocks.
TRANSFER_STATUSES = frozenset(
    {"awaiting_transfer", "completed", "time_limited", "stopped", "failed", "uploading"}
)
MODE_STATUSES = {
    "pilot": frozenset(),
    "rehearsal": frozenset({"uploading"}),
    "posttrain": frozenset({"uploading", "awaiting_rewards"}),
}
"""Worker statuses beyond the pilot's that a mode's heartbeat may carry. A rehearsal worker
reads ``uploading`` while it drains its last checkpoints to R2 with the GPU idle. A
post-training worker may also read ``awaiting_rewards``, run 2's wait for the Mac's RL
rewards or dev verdicts. Without an extension in the policy (run 1's) it is an ordinary
in-progress status and the idle rule applies unchanged; a run-2 policy grants it a bounded
idle extension per request kind (``AWAITING_KINDS``, ``_awaiting_within_extension``)."""
AWAITING_STATUS = "awaiting_rewards"
AWAITING_KINDS = ("reward", "dev")
"""The bridge request kinds an ``awaiting_rewards`` heartbeat names in ``awaiting.kind``."""
MAX_AWAITING_EXTENSION_SECONDS = 60_000.0
"""No extension may exceed the post-training wall ceiling, whatever a policy file says."""
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
        try:
            same_creation = (
                abs(parse_timestamp(pod.get("createdAt")) - parse_timestamp(self.created_at)) < 1.0
            )  # Same instant; the API may render the timestamp differently.
        except ValueError:
            return False
        return (
            pod.get("id") == self.pod_id
            and pod.get("name") == self.name
            and isinstance(gpu, dict)
            and gpu.get("id") == self.gpu_id
            and type(gpu.get("count")) is int
            and gpu.get("count") == self.gpu_count
            and pod.get("dataCenterId") == self.data_center_id
            and same_creation
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
    max_transfer_seconds: float = 0.0
    # Before the worker's first heartbeat the idle clock starts at Pod creation, or
    # at this later moment (the watchdog's own start), so image pull and bootstrap
    # are not read as an idle GPU. Wall and cost limits are unaffected.
    idle_baseline_unix: float | None = None
    mode: str = "pilot"
    # Run 2: while the heartbeat reads ``awaiting_rewards`` for a reward request (or a dev
    # check), the GPU may sit idle up to this long instead of ``max_idle_seconds``: the Pod's
    # own timeout for that wait plus a stated slack (``posttrain_ops.bridge_timeouts``). Zero,
    # the default and run 1's value, is no extension.
    awaiting_reward_extension_seconds: float = 0.0
    awaiting_dev_extension_seconds: float = 0.0

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
        if self.mode not in MODE_WALL_SECONDS:
            raise ValueError(f"mode must be one of {', '.join(sorted(MODE_WALL_SECONDS))}")
        if self.max_wall_seconds > MODE_WALL_SECONDS[self.mode]:
            raise ValueError(
                f"max_wall_seconds cannot exceed the {self.mode} ceiling "
                f"of {MODE_WALL_SECONDS[self.mode]} seconds"
            )
        if (
            not _is_number(self.starting_cost_usd)
            or not math.isfinite(self.starting_cost_usd)
            or not 0 <= self.starting_cost_usd < self.max_cost_usd
        ):
            raise ValueError("starting cost must be finite, nonnegative and below the stop cost")
        if self.max_idle_seconds > MAX_GPU_IDLE_SECONDS:
            raise ValueError("max_idle_seconds cannot exceed the 600-second pilot ceiling")
        if (
            not _is_number(self.max_transfer_seconds)
            or not math.isfinite(self.max_transfer_seconds)
            or not 0 <= self.max_transfer_seconds <= MAX_TRANSFER_SECONDS
        ):
            raise ValueError("max_transfer_seconds must be between 0 and the 1800-second ceiling")
        if self.billing_started_at_unix is not None and (
            not _is_number(self.billing_started_at_unix)
            or not math.isfinite(self.billing_started_at_unix)
        ):
            raise ValueError("billing_started_at_unix must be finite when provided")
        if self.idle_baseline_unix is not None and (
            not _is_number(self.idle_baseline_unix)
            or not math.isfinite(self.idle_baseline_unix)
            or self.idle_baseline_unix <= 0
        ):
            raise ValueError("idle_baseline_unix must be a finite positive time when provided")
        for key in ("awaiting_reward_extension_seconds", "awaiting_dev_extension_seconds"):
            value = getattr(self, key)
            if (
                not _is_number(value)
                or not math.isfinite(value)
                or not 0 <= value <= MAX_AWAITING_EXTENSION_SECONDS
            ):
                raise ValueError(f"{key} must be finite, nonnegative and at most 60000 seconds")
            if value and self.mode != "posttrain":
                raise ValueError(f"{key} applies only to a posttrain policy")
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
    transfer_progress_unix: float | None = None
    ignored: str | None = None
    """Why a heartbeat on disk was treated as absent (another Pod's, or older than this Pod)."""
    awaiting_kind: str | None = None
    """For an ``awaiting_rewards`` heartbeat, the kind of request the Pod waits on (``reward``
    or ``dev``), when its ``awaiting`` object names one; anything else reads as ``None``."""


def foreign_heartbeat(
    value: dict[str, Any], *, pod_id: str | None, created_at_unix: float | None
) -> str | None:
    """Why a heartbeat cannot be this Pod's, or None.

    A heartbeat naming another ``pod_id``, or updated before this Pod was created, was
    written for an earlier Pod (a reused controller directory). It is never evidence about
    this Pod: the guards treat it as absent, never as fresh, never as a reason to delete.
    A heartbeat without ``pod_id`` (the pilot worker's) is judged by its time alone.
    """
    if pod_id is not None and "pod_id" in value and value["pod_id"] != pod_id:
        return "heartbeat_names_another_pod"
    updated = value.get("updated_at_unix")
    if (
        created_at_unix is not None
        and _is_number(updated)
        and math.isfinite(updated)
        and updated < created_at_unix
    ):
        return "heartbeat_predates_pod_creation"
    return None


def read_heartbeat(
    path: Path,
    *,
    now: float,
    max_idle_seconds: float,
    mode: str = "pilot",
    pod_id: str | None = None,
    created_at_unix: float | None = None,
) -> Heartbeat:
    """Return last GPU activity and a failure reason; absent means just-created.

    ``mode`` adds that mode's statuses (``MODE_STATUSES``); under a pilot policy an
    ``uploading`` heartbeat stays malformed, exactly as before. With ``pod_id`` and
    ``created_at_unix`` (the captured identity), a heartbeat that cannot be this Pod's
    (``foreign_heartbeat``) reads as absent, whatever else it says.
    """
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
    foreign = foreign_heartbeat(value, pod_id=pod_id, created_at_unix=created_at_unix)
    if foreign is not None:
        return Heartbeat(0.0, None, None, ignored=foreign)
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
        *MODE_STATUSES.get(mode, frozenset()),
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
    if updated > now + CLOCK_SKEW_SECONDS:
        return Heartbeat(float(activity), "heartbeat_malformed", worker_status)
    updated = min(updated, now)  # A slightly fast Pod clock is not evidence of anything.
    if now - updated >= max_idle_seconds:
        return Heartbeat(float(activity), "heartbeat_stale", worker_status)
    receipts_verified = (
        worker_status == "shutdown_ready"
        and value.get("mac_receipt_verified") is True
        and value.get("github_receipt_verified") is True
    )
    progress = value.get("transfer_progress_unix")
    if progress is not None and (
        not _is_number(progress) or not math.isfinite(progress) or not 0 <= progress <= now
    ):
        return Heartbeat(float(activity), "heartbeat_malformed", worker_status)
    awaiting = value.get("awaiting")
    kind = awaiting.get("kind") if isinstance(awaiting, dict) else None
    return Heartbeat(
        float(activity),
        None,
        worker_status,
        receipts_verified,
        None if progress is None else float(progress),
        awaiting_kind=kind if worker_status == AWAITING_STATUS and kind in AWAITING_KINDS else None,
    )


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

    def request_path(self, method: str, path: str, timeout: float) -> tuple[int, bytes]:
        """Send a request for an API-relative path such as ``/pods/{id}``."""
        if not isinstance(path, str) or not path.startswith("/") or ".." in path:
            raise ValueError("API path must be absolute within the API base")
        return self._request(method, API_BASE + path, timeout)

    def get_pod(self, pod_id: str, timeout: float) -> tuple[int, bytes]:
        safe_id = _validated_pod_id(pod_id)
        return self.request_path("GET", f"/pods/{safe_id}", timeout)

    def delete_pod(self, pod_id: str, timeout: float) -> tuple[int, bytes]:
        safe_id = _validated_pod_id(pod_id)
        return self.request_path("DELETE", f"/pods/{safe_id}", timeout)


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


def _first_heartbeat_trigger(policy: Policy, created_at: float, checked_at: float) -> str:
    """Idle decision before any heartbeat exists; a baseline can never be in the future."""
    baseline = created_at
    if policy.idle_baseline_unix is not None:
        if (
            policy.idle_baseline_unix > checked_at + CLOCK_SKEW_SECONDS
            or policy.idle_baseline_unix > created_at + policy.max_wall_seconds
        ):
            return "idle_baseline_invalid"
        baseline = max(created_at, min(policy.idle_baseline_unix, checked_at))
    return "idle_gpu_limit" if checked_at - baseline >= policy.max_idle_seconds else ""


def check_once(
    client: RunpodClient,
    identity: PodIdentity,
    policy: Policy,
    heartbeat_path: Path,
    *,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    observed: bool = False,
) -> Outcome:
    """Inspect once, terminating only after exact captured identity matches.

    ``observed`` states whether this identity has already been seen alive
    through the API. Until then a 404 is not evidence of termination: a wrong
    base URL, Pod ID or key scope would otherwise read as a successful
    shutdown while the Pod keeps billing.

    A heartbeat naming another Pod reads as absent. Under a rehearsal or post-training
    policy so does one updated before this Pod's ``createdAt``: a controller directory reused
    for a new Pod still holds the old Pod's mirrored heartbeat, which must never delete the
    new one.
    """
    raw_now = now()
    clock_valid = _is_number(raw_now) and math.isfinite(raw_now)
    checked_at = float(raw_now) if clock_valid else 0.0
    try:
        created_unix = parse_timestamp(identity.created_at) if policy.mode != "pilot" else None
    except ValueError:
        created_unix = None  # The guard below fails closed on the same timestamp.
    heartbeat = read_heartbeat(
        heartbeat_path,
        now=checked_at,
        max_idle_seconds=policy.max_idle_seconds,
        mode=policy.mode,
        pod_id=identity.pod_id,
        created_at_unix=created_unix,
    )
    try:
        status, body = client.get_pod(identity.pod_id, policy.request_timeout_seconds)
    except ApiTransportError:
        return _outcome("unknown", "pod_get_transport_error", identity, checked_at)
    if status == 404:
        if not observed:
            return _outcome("unknown", "pod_get_404_before_identity_observed", identity, checked_at)
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
            if billing_started is not None and billing_started > created_at + CLOCK_SKEW_SECONDS:
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
                    # On first launch, no heartbeat yet: the idle clock starts at Pod
                    # creation or at the later explicit idle baseline.
                    trigger = _first_heartbeat_trigger(policy, created_at, checked_at)
                elif (
                    not math.isfinite(heartbeat.last_gpu_activity_unix)
                    or heartbeat.last_gpu_activity_unix > checked_at + CLOCK_SKEW_SECONDS
                ):
                    trigger = "heartbeat_malformed"
                elif checked_at - heartbeat.last_gpu_activity_unix >= policy.max_idle_seconds:
                    if _transfer_in_progress(heartbeat, policy, checked_at):
                        trigger = ""
                    elif _awaiting_within_extension(heartbeat, policy, checked_at):
                        trigger = ""
                    else:
                        trigger = "idle_gpu_limit"
                else:
                    trigger = ""
        except (ValueError, OverflowError):
            trigger = "guard_calculation_error"

    def settled(state: str, reason: str) -> Outcome:
        # An ignored heartbeat is recorded so the operator sees why it was not read.
        return _outcome(state, reason, identity, checked_at, heartbeat.ignored)

    if not trigger:
        return settled("watching", "within_limits")

    try:
        delete_status, _ = client.delete_pod(identity.pod_id, policy.request_timeout_seconds)
    except ApiTransportError:
        return settled("unknown", trigger + "_delete_transport_error")
    if delete_status != 204:
        return settled("unknown", trigger + f"_delete_http_{delete_status}")

    for attempt in range(policy.delete_verify_attempts):
        if attempt:
            sleep(policy.delete_verify_delay_seconds)
        try:
            verify_status, verify_body = client.get_pod(
                identity.pod_id, policy.request_timeout_seconds
            )
        except ApiTransportError:
            return settled("unknown", trigger + "_delete_unconfirmed_transport_error")
        if verify_status == 404:
            return settled("terminated", trigger + "_delete_204_get_404")
        if verify_status != 200:
            return settled("unknown", trigger + f"_delete_unconfirmed_http_{verify_status}")
        remaining = _decode_object(verify_body)
        if remaining is None:
            return settled("unknown", trigger + "_delete_unconfirmed_malformed_json")
        if not identity.matches(remaining):
            return settled("identity_mismatch", trigger + "_post_delete_identity_mismatch")
    return settled("unknown", trigger + "_delete_still_present")


def _transfer_in_progress(heartbeat: Heartbeat, policy: Policy, now: float) -> bool:
    """Allow a bounded transfer window after GPU work when the Mac shows live progress.

    Progress stamps come from the Mac-side supervisor (staged bytes growing; in
    rehearsal mode, new objects under the run's R2 prefix seen with the Mac's own
    key), never from the worker, so a hung Pod cannot extend its own lease. The
    wall and cost limits still apply unchanged.
    """
    return (
        policy.max_transfer_seconds > 0
        and heartbeat.status in TRANSFER_STATUSES
        and heartbeat.transfer_progress_unix is not None
        and now - heartbeat.transfer_progress_unix < policy.max_idle_seconds
        and now - heartbeat.last_gpu_activity_unix < policy.max_transfer_seconds
    )


def awaiting_extension(policy: Policy, kind: str | None) -> float:
    """The policy's idle extension for a wait on a request of ``kind`` (0 for none)."""
    if policy.mode != "posttrain" or kind not in AWAITING_KINDS:
        return 0.0
    if kind == "reward":
        return float(policy.awaiting_reward_extension_seconds)
    return float(policy.awaiting_dev_extension_seconds)


def _awaiting_within_extension(heartbeat: Heartbeat, policy: Policy, now: float) -> bool:
    """Run 2: an ``awaiting_rewards`` heartbeat keeps an idle GPU for the policy's extension.

    The Pod waits on the Mac for one reward request or dev check at a time, and its own
    timeout ends that wait; the extension is that timeout plus a stated slack, measured from
    the last GPU work like the idle rule itself, and never more, so a worker that hangs
    while claiming to wait is still deleted. The wall and cost limits apply unchanged, and a
    stale or malformed heartbeat has already failed before this is asked.
    """
    extension = awaiting_extension(policy, heartbeat.awaiting_kind)
    return (
        heartbeat.status == AWAITING_STATUS
        and extension > 0
        and now - heartbeat.last_gpu_activity_unix < max(policy.max_idle_seconds, extension)
    )


POLICY_FILE_FIELDS = (
    "mode",
    "max_wall_seconds",
    "max_cost_usd",
    "max_idle_seconds",
    "quoted_total_hourly",
    "max_transfer_seconds",
    "billing_started_at_unix",
)
"""The limits a ``--policy`` file pins. ``idle_baseline_unix`` is left out: the runbook's
helper re-baselines it on every restart."""


def check_policy_file(policy: Policy, path: Path) -> None:
    """Refuse a watchdog whose flags give other limits than the policy file the readiness
    receipt checked, so a mistyped shell variable can never start a weaker watchdog."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read the policy file: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("the policy file must hold a JSON object")
    missing = [name for name in POLICY_FILE_FIELDS if name not in value]
    if missing:
        raise ValueError(f"the policy file lacks {', '.join(missing)}")
    differ = [name for name in POLICY_FILE_FIELDS if getattr(policy, name) != value[name]]
    for name in (
        "starting_cost_usd",
        "awaiting_reward_extension_seconds",
        "awaiting_dev_extension_seconds",
    ):
        if value.get(name, 0.0) != getattr(policy, name):
            differ.append(name)
    if differ:
        raise ValueError("the watchdog's flags differ from the policy file in " + ", ".join(differ))


def _load_identity(path: Path) -> PodIdentity:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read identity JSON") from error
    return PodIdentity.from_dict(value)


def build_parser() -> argparse.ArgumentParser:
    """The watchdog command line; the runbook's launch command is tested against it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True, help="captured Pod identity JSON")
    parser.add_argument("--heartbeat", type=Path, required=True, help="worker heartbeat JSON")
    parser.add_argument(
        "--receipt", type=Path, required=True, help="atomic local watchdog receipt path"
    )
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_WALL_SECONDS),
        default="pilot",
        help="wall ceiling: pilot 5400 s, rehearsal and posttrain 60000 s",
    )
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--max-idle-seconds", type=float, default=600)
    parser.add_argument("--quoted-total-hourly", type=float, required=True)
    parser.add_argument("--starting-cost-usd", type=float, default=0.0)
    parser.add_argument("--billing-started-at-unix", type=float)
    parser.add_argument(
        "--idle-baseline-unix",
        type=float,
        help="before the first heartbeat, start the idle clock here instead of at Pod creation",
    )
    parser.add_argument("--max-transfer-seconds", type=float, default=0.0)
    parser.add_argument(
        "--awaiting-reward-extension-seconds",
        type=float,
        default=0.0,
        help="posttrain run 2: idle limit while the Pod awaits a reward request",
    )
    parser.add_argument(
        "--awaiting-dev-extension-seconds",
        type=float,
        default=0.0,
        help="posttrain run 2: idle limit while the Pod awaits a dev check",
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--delete-verify-attempts", type=int, default=3)
    parser.add_argument(
        "--max-unobserved-seconds",
        type=float,
        default=900.0,
        help="exit with status 3 if the captured identity is never seen alive within this time",
    )
    parser.add_argument(
        "--pidfile",
        type=Path,
        help="write this process's PID here, so one Pod's watchdog can be restarted alone",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        help="refuse to start unless the flags give exactly this policy file's limits",
    )
    return parser


def policy_from_args(args: argparse.Namespace) -> Policy:
    return Policy(
        args.max_wall_seconds,
        args.max_cost_usd,
        args.max_idle_seconds,
        args.quoted_total_hourly,
        args.billing_started_at_unix,
        args.request_timeout_seconds,
        args.delete_verify_attempts,
        starting_cost_usd=args.starting_cost_usd,
        max_transfer_seconds=args.max_transfer_seconds,
        idle_baseline_unix=args.idle_baseline_unix,
        mode=args.mode,
        awaiting_reward_extension_seconds=args.awaiting_reward_extension_seconds,
        awaiting_dev_extension_seconds=args.awaiting_dev_extension_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be finite and positive")
    token = os.environ.get(API_KEY_ENV, "")
    if not token:
        parser.error(f"set {API_KEY_ENV} in the local environment")
    try:
        identity = _load_identity(args.identity)
        policy = policy_from_args(args)
        if args.policy is not None:
            check_policy_file(policy, args.policy)
    except ValueError as error:
        parser.error(str(error))
    client = RunpodClient(token, UrllibTransport())
    if args.pidfile is None:
        return _watch(client, identity, policy, args)
    write_pidfile(args.pidfile)
    # A SIGTERM from the runbook's per-run restart unwinds through ``finally`` below.
    previous = signal.signal(signal.SIGTERM, _exit_on_signal)
    try:
        return _watch(client, identity, policy, args)
    finally:
        signal.signal(signal.SIGTERM, previous)
        release_pidfile(args.pidfile)


def _exit_on_signal(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


def write_pidfile(path: Path) -> None:
    """Atomically record this process's PID; a restart reads it to stop only this one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(f"{os.getpid()}\n")
    os.replace(temp_name, path)


def release_pidfile(path: Path) -> None:
    """Remove the PID file only while it still names this process."""
    try:
        if path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def _watch(
    client: RunpodClient, identity: PodIdentity, policy: Policy, args: argparse.Namespace
) -> int:
    state_path = args.receipt.with_name(args.receipt.name + ".observed.json")
    observed = load_observed(state_path, identity)
    started = time.time()
    while True:
        outcome = check_once(client, identity, policy, args.heartbeat, observed=observed)
        atomic_receipt(args.receipt, outcome)
        if outcome.state in {"watching", "terminated"} and not observed:
            observed = True
            save_observed(state_path, identity, outcome.checked_at_unix)
        if outcome.state in {"terminated", "absent", "identity_mismatch"}:
            return 0 if outcome.state != "identity_mismatch" else 2
        if not observed and time.time() - started >= args.max_unobserved_seconds:
            print("watchdog never observed the captured Pod identity alive", file=sys.stderr)
            return 3
        delay = args.poll_seconds
        if outcome.reason.endswith("_429") or "_http_429" in outcome.reason:
            delay = max(delay, 30.0)  # Respect the API rate limit before polling again.
        time.sleep(delay)


def load_observed(path: Path, identity: PodIdentity) -> bool:
    """Read whether a previous watchdog process already saw this Pod alive."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and value.get("pod_id") == identity.pod_id
        and (value.get("observed") is True)
    )


def save_observed(path: Path, identity: PodIdentity, when: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"pod_id": identity.pod_id, "observed": True, "at_unix": when}, stream)
    os.replace(temp_name, path)


if __name__ == "__main__":
    raise SystemExit(main())
