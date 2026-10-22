"""Read-only Runpod REST v2 helpers: capture a Pod identity, connection details and billing.

The watchdog and supervisor require a captured identity in the exact shape of
``PodIdentity.from_dict``; assembling it by hand from the console invites a
typo that would make the watchdog refuse to ever delete the Pod. This tool
reads the live Pod object once and writes two secret-free files: the identity
(id, name, gpuId, gpuCount, dataCenterId, createdAt) and the connection
details (status, cost, direct SSH host/port, ports). The Pod's ``env`` block
is never written because it can contain injected secrets. Nothing here
creates, starts, stops or deletes a resource.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from scratchcoder.common.records import atomic_json
from tools.runpod_watchdog import (
    API_KEY_ENV,
    ApiTransportError,
    PodIdentity,
    RunpodClient,
    Transport,
    UrllibTransport,
    _validated_pod_id,
    parse_timestamp,
)

_RATE_LIMIT_DELAY = 30.0


def _get_json(
    client: RunpodClient,
    path: str,
    *,
    timeout: float,
    sleep: Callable[[float], None],
    attempts: int = 3,
) -> dict[str, Any]:
    for attempt in range(attempts):
        try:
            status, body = client.request_path("GET", path, timeout)
        except ApiTransportError as error:
            raise RuntimeError("Runpod API transport failed") from error
        if status == 429 and attempt + 1 < attempts:
            sleep(_RATE_LIMIT_DELAY)
            continue
        if status == 404:
            raise LookupError("resource not found (HTTP 404)")
        if status in (401, 403):
            raise PermissionError(f"API key rejected or lacks access (HTTP {status})")
        if status != 200:
            raise RuntimeError(f"unexpected API response HTTP {status}")
        try:
            value = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("API returned malformed JSON") from error
        if not isinstance(value, dict):
            raise ValueError("API returned a non-object body")
        return value
    raise RuntimeError("Runpod API rate limit persisted")


def identity_from_pod(pod: dict[str, Any]) -> dict[str, Any]:
    """Translate a v2 Pod object into the watchdog's captured-identity shape."""
    gpu = pod.get("gpu")
    if not isinstance(gpu, dict):
        raise ValueError("Pod object has no gpu block; CPU pods are not supported")
    candidate = {
        "id": pod.get("id"),
        "name": pod.get("name"),
        "gpuId": gpu.get("id"),
        "gpuCount": gpu.get("count"),
        "dataCenterId": pod.get("dataCenterId"),
        "createdAt": pod.get("createdAt"),
    }
    PodIdentity.from_dict(candidate)  # Validates every field exactly as the watchdog will.
    return candidate


def connection_from_pod(pod: dict[str, Any], *, captured_at: float) -> dict[str, Any]:
    """Secret-free operational details; the env block is deliberately omitted."""
    raw_ssh = pod.get("ssh")
    ssh: dict[str, Any] = raw_ssh if isinstance(raw_ssh, dict) else {}
    raw_direct = ssh.get("direct")
    direct: dict[str, Any] | None = raw_direct if isinstance(raw_direct, dict) else None
    raw_runtime = pod.get("runtime")
    runtime: dict[str, Any] | None = raw_runtime if isinstance(raw_runtime, dict) else None
    cost = pod.get("cost")
    cost_value: float | None = (
        float(cost)
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost)
        else None
    )
    return {
        "schema_version": 1,
        "captured_at_unix": captured_at,
        "pod_id": pod.get("id"),
        "status": pod.get("status"),
        "actions": pod.get("actions"),
        "cost_usd_per_hour": cost_value,
        "image": pod.get("image"),
        "disk_gb": pod.get("disk"),
        "cuda_version": pod.get("cudaVersion"),
        "created_at": pod.get("createdAt"),
        "created_at_unix": parse_timestamp(pod["createdAt"]) if pod.get("createdAt") else None,
        "started_at": pod.get("startedAt"),
        "ssh_direct": None
        if direct is None
        else {
            "host": direct.get("host"),
            "port": direct.get("port"),
            "username": direct.get("username"),
        },
        "ssh_proxy_only": direct is None,
        "runtime_ports": None if runtime is None else runtime.get("ports"),
        "mounts": pod.get("mounts"),
        "env_omitted": True,
    }


def capture(
    client: RunpodClient,
    pod_id: str,
    *,
    identity_path: Path,
    connection_path: Path,
    timeout: float = 10.0,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pod = _get_json(client, f"/pods/{_validated_pod_id(pod_id)}", timeout=timeout, sleep=sleep)
    identity = identity_from_pod(pod)
    connection = connection_from_pod(pod, captured_at=now())
    if identity["id"] != pod_id:
        raise ValueError("API returned a different Pod ID")
    atomic_json(identity_path, identity)
    atomic_json(connection_path, connection)
    return identity, connection


def billing(
    client: RunpodClient,
    pod_id: str,
    *,
    output: Path,
    last_buckets: int = 3,
    timeout: float = 10.0,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Fetch provider-side billing records for one Pod (hourly buckets)."""
    if type(last_buckets) is not int or not 1 <= last_buckets <= 200:
        raise ValueError("last_buckets must be an integer from 1 to 200")
    query = urlencode(
        {"podId": _validated_pod_id(pod_id), "bucketSize": "hour", "lastN": last_buckets}
    )
    value = _get_json(client, f"/billing/pods?{query}", timeout=timeout, sleep=sleep)
    records = value.get("records")
    metadata = value.get("metadata")
    if not isinstance(records, list) or not isinstance(metadata, dict):
        raise ValueError("billing response lacks records/metadata")
    totals = metadata.get("totals")
    report = {
        "schema_version": 1,
        "fetched_at_unix": now(),
        "pod_id": pod_id,
        "bucket_size": "hour",
        "records": records,
        "totals": totals if isinstance(totals, dict) else None,
        "note": (
            "provider billing runs on its own schedule; re-fetch after termination "
            "until the records stop changing"
        ),
    }
    atomic_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture", help="write identity and connection JSON for a Pod")
    cap.add_argument("--pod-id", required=True)
    cap.add_argument("--identity", type=Path, required=True)
    cap.add_argument("--connection", type=Path, required=True)
    bill = sub.add_parser("billing", help="write provider billing records for a Pod")
    bill.add_argument("--pod-id", required=True)
    bill.add_argument("--output", type=Path, required=True)
    bill.add_argument("--last-buckets", type=int, default=3)
    status = sub.add_parser("status", help="print status, cost and uptime without secrets")
    status.add_argument("--pod-id", required=True)
    parser.add_argument("--request-timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    token = os.environ.get(API_KEY_ENV, "")
    if not token:
        parser.error(f"set {API_KEY_ENV} in the local environment")
    transport: Transport = UrllibTransport()
    client = RunpodClient(token, transport)
    try:
        if args.command == "capture":
            identity, connection = capture(
                client,
                args.pod_id,
                identity_path=args.identity,
                connection_path=args.connection,
                timeout=args.request_timeout_seconds,
            )
            print(json.dumps({"identity": identity, "status": connection["status"]}, indent=2))
        elif args.command == "billing":
            report = billing(
                client,
                args.pod_id,
                output=args.output,
                last_buckets=args.last_buckets,
                timeout=args.request_timeout_seconds,
            )
            print(json.dumps({"totals": report["totals"], "records": len(report["records"])}))
        else:
            pod = _get_json(
                client,
                f"/pods/{_validated_pod_id(args.pod_id)}",
                timeout=args.request_timeout_seconds,
                sleep=time.sleep,
            )
            print(json.dumps(connection_from_pod(pod, captured_at=time.time()), indent=2))
    except (LookupError, PermissionError, RuntimeError, ValueError) as error:
        print(f"runpod-pod-identity: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
