"""Offline tests for the read-only Pod identity/billing capture tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.runpod_pod_identity import billing, capture, connection_from_pod, identity_from_pod
from tools.runpod_watchdog import API_BASE, PodIdentity, RunpodClient


def pod_object() -> dict[str, object]:
    return {
        "id": "7h9k2m4n6p",
        "name": "pilot-a40",
        "image": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
        "disk": 20,
        "env": {"RUNPOD_SECRET_example": "must-never-be-written"},
        "status": "RUNNING",
        "actions": ["stop", "restart", "terminate"],
        "mounts": {"persistent": None},
        "gpu": {"id": "NVIDIA A40", "count": 1, "vcpuCount": 9, "memory": 50},
        "dataCenterId": "US-KS-2",
        "cudaVersion": "12.8",
        "ssh": {
            "proxy": {"host": "ssh.runpod.io", "port": 22, "username": "7h9k2m4n6p-6441"},
            "direct": {"host": "195.26.233.3", "port": 34446, "username": "root"},
        },
        "cost": 0.59,
        "runtime": {"uptime": 120, "ports": [{"private": 22, "public": 34446, "type": "tcp"}]},
        "createdAt": "2026-10-22T19:00:00Z",
        "startedAt": "2026-10-22T19:01:00Z",
    }


class FakeTransport:
    def __init__(self, responses: list[tuple[int, bytes]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def request(
        self, method: str, url: str, headers: dict[str, str], timeout: float
    ) -> tuple[int, bytes]:
        self.calls.append((method, url))
        return self.responses.pop(0)


def test_capture_writes_watchdog_shaped_identity_and_secret_free_connection(
    tmp_path: Path,
) -> None:
    transport = FakeTransport([(200, json.dumps(pod_object()).encode())])
    client = RunpodClient("placeholder-key", transport)
    identity, connection = capture(
        client,
        "7h9k2m4n6p",
        identity_path=tmp_path / "pod-identity.json",
        connection_path=tmp_path / "pod-connection.json",
        now=lambda: 1_800_000_000.0,
    )
    assert transport.calls == [("GET", f"{API_BASE}/pods/7h9k2m4n6p")]
    assert identity == {
        "id": "7h9k2m4n6p",
        "name": "pilot-a40",
        "gpuId": "NVIDIA A40",
        "gpuCount": 1,
        "dataCenterId": "US-KS-2",
        "createdAt": "2026-10-22T19:00:00Z",
    }
    assert PodIdentity.from_dict(json.loads((tmp_path / "pod-identity.json").read_text()))
    saved = (tmp_path / "pod-connection.json").read_text()
    assert "must-never-be-written" not in saved and "placeholder-key" not in saved
    assert connection["ssh_direct"] == {"host": "195.26.233.3", "port": 34446, "username": "root"}
    assert connection["ssh_proxy_only"] is False
    assert connection["cost_usd_per_hour"] == 0.59
    assert connection["created_at_unix"] == 1_792_695_600.0
    assert connection["env_omitted"] is True


def test_identity_capture_rejects_cpu_pods_and_mismatched_ids(tmp_path: Path) -> None:
    cpu = pod_object()
    cpu.pop("gpu")
    with pytest.raises(ValueError, match="gpu"):
        identity_from_pod(cpu)
    transport = FakeTransport([(200, json.dumps(pod_object()).encode())])
    with pytest.raises(ValueError, match="different Pod ID"):
        capture(
            RunpodClient("k", transport),
            "other-id-000",
            identity_path=tmp_path / "i.json",
            connection_path=tmp_path / "c.json",
        )
    assert not (tmp_path / "i.json").exists()


def test_proxy_only_pods_are_flagged_for_the_transfer_requirement() -> None:
    proxied = pod_object()
    proxied["ssh"] = {"proxy": {"host": "ssh.runpod.io", "port": 22, "username": "x"}}
    proxied["runtime"] = None
    connection = connection_from_pod(proxied, captured_at=1.0)
    assert connection["ssh_proxy_only"] is True and connection["ssh_direct"] is None
    assert connection["runtime_ports"] is None


def test_api_errors_fail_closed_and_rate_limits_are_retried(tmp_path: Path) -> None:
    for status, error in ((404, LookupError), (403, PermissionError), (500, RuntimeError)):
        transport = FakeTransport([(status, b"{}")])
        with pytest.raises(error):
            capture(
                RunpodClient("k", transport),
                "7h9k2m4n6p",
                identity_path=tmp_path / "i.json",
                connection_path=tmp_path / "c.json",
            )
    slept: list[float] = []
    transport = FakeTransport([(429, b""), (200, json.dumps(pod_object()).encode())])
    capture(
        RunpodClient("k", transport),
        "7h9k2m4n6p",
        identity_path=tmp_path / "i.json",
        connection_path=tmp_path / "c.json",
        sleep=slept.append,
    )
    assert slept == [30.0] and len(transport.calls) == 2


def test_billing_report_records_provider_totals_without_secrets(tmp_path: Path) -> None:
    body = {
        "records": [
            {
                "startTime": "2026-10-22T19:00:00Z",
                "endTime": "2026-10-22T20:00:00Z",
                "podId": "7h9k2m4n6p",
                "totalAmount": 0.61,
                "gpuAmount": 0.59,
                "cpuAmount": 0,
                "diskAmount": 0.02,
            }
        ],
        "metadata": {"totals": {"totalAmount": 0.61, "gpuAmount": 0.59, "diskAmount": 0.02}},
    }
    transport = FakeTransport([(200, json.dumps(body).encode())])
    report = billing(
        RunpodClient("secret-k", transport),
        "7h9k2m4n6p",
        output=tmp_path / "billing.json",
        now=lambda: 5.0,
    )
    assert transport.calls[0][1].startswith(f"{API_BASE}/billing/pods?")
    assert "podId=7h9k2m4n6p" in transport.calls[0][1]
    assert "bucketSize=hour" in transport.calls[0][1] and "lastN=3" in transport.calls[0][1]
    assert report["totals"] == {"totalAmount": 0.61, "gpuAmount": 0.59, "diskAmount": 0.02}
    text = (tmp_path / "billing.json").read_text()
    assert "secret-k" not in text and json.loads(text)["records"][0]["totalAmount"] == 0.61
    with pytest.raises(ValueError):
        billing(
            RunpodClient("k", FakeTransport([])),
            "7h9k2m4n6p",
            output=tmp_path / "b",
            last_buckets=0,
        )
