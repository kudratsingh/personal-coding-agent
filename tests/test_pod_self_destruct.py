"""Offline tests for the in-pod self-destruct timer; no Runpod call is made."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from tools.pod_self_destruct import API_BASE, SelfDestruct, main


class Clock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.wall = start
        self.mono = 50.0

    def sleep(self, seconds: float) -> None:
        self.wall += seconds
        self.mono += seconds


class FakeTransport:
    def __init__(self, responses: list[tuple[int, dict[str, str], bytes] | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, str]] = []

    def request(
        self, method: str, url: str, headers: dict[str, str], timeout: float
    ) -> tuple[int, dict[str, str], bytes]:
        self.calls.append((method, url, headers["Authorization"]))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def timer(tmp_path: Path, transport: FakeTransport, clock: Clock, **kwargs: object) -> SelfDestruct:
    return SelfDestruct(
        pod_id="podabc123",
        token="secret-token-placeholder",
        deadline_unix=1_010.0,
        receipt_path=str(tmp_path / "receipts" / "self-destruct.json"),
        transport=transport,
        clock=lambda: clock.wall,
        monotonic=lambda: clock.mono,
        sleep=clock.sleep,
        poll_seconds=5,
        **kwargs,  # type: ignore[arg-type]
    )


def test_waits_for_deadline_then_deletes_and_confirms_absence(tmp_path: Path) -> None:
    clock = Clock()
    transport = FakeTransport([(204, {}, b""), (404, {}, b"")])
    unit = timer(tmp_path, transport, clock)
    receipt = json.loads(Path(unit.receipt_path).read_text())
    assert receipt["state"] == "armed" and receipt["pod_id"] == "podabc123"
    assert unit.run() == "terminated"
    assert clock.wall >= 1_010.0
    assert [call[:2] for call in transport.calls] == [
        ("DELETE", f"{API_BASE}/pods/podabc123"),
        ("GET", f"{API_BASE}/pods/podabc123"),
    ]
    text = Path(unit.receipt_path).read_text()
    final = json.loads(text)
    assert final["state"] == "terminated"
    assert final["reason"] == "wall_clock_deadline_delete_confirmed_absent"
    assert "secret-token-placeholder" not in text
    assert all(call[2] == "Bearer secret-token-placeholder" for call in transport.calls)


def test_monotonic_deadline_fires_when_wall_clock_is_set_back(tmp_path: Path) -> None:
    clock = Clock()
    transport = FakeTransport([(204, {}, b""), (404, {}, b"")])
    unit = timer(tmp_path, transport, clock)
    original_sleep = clock.sleep

    def sleep(seconds: float) -> None:
        original_sleep(seconds)
        clock.wall -= 100  # A wall-clock jump backwards must not postpone termination.

    unit.sleep = sleep
    assert unit.run() == "terminated"
    assert json.loads(Path(unit.receipt_path).read_text())["reason"].startswith(
        "monotonic_deadline"
    )


def test_stop_file_triggers_early_termination(tmp_path: Path) -> None:
    clock = Clock()
    transport = FakeTransport([(204, {}, b""), (404, {}, b"")])
    stop = tmp_path / "TERMINATE_NOW"
    unit = timer(tmp_path, transport, clock, stop_file=str(stop))
    stop.write_text("terminate")
    assert unit.run() == "terminated"
    assert clock.wall < 1_010.0
    assert json.loads(Path(unit.receipt_path).read_text())["reason"].startswith("stop_file_present")


def test_transport_errors_and_still_present_pod_are_retried_until_absent(tmp_path: Path) -> None:
    clock = Clock()
    from tools.pod_self_destruct import TransportError

    transport = FakeTransport(
        [
            TransportError("down"),
            (500, {}, b"error"),
            (204, {}, b""),
            (200, {}, json.dumps({"status": "RUNNING"}).encode()),
            (200, {}, json.dumps({"status": "TERMINATED"}).encode()),
        ]
    )
    unit = timer(tmp_path, transport, clock)
    assert unit.run() == "terminated"
    results = [row["result"] for row in unit.attempts]
    assert results[:3] == ["transport_error", "http_500", "http_204"]
    assert json.loads(Path(unit.receipt_path).read_text())["state"] == "terminated"


def test_forbidden_delete_uses_runpodctl_fallback_then_confirms(tmp_path: Path) -> None:
    clock = Clock()
    fake = tmp_path / "runpodctl"
    script = '[ "$1" = pod ] && [ "$2" = delete ] && [ -n "$RUNPOD_API_KEY" ] && exit 0'
    fake.write_text(f"#!/bin/sh\n{script}\nexit 9\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    transport = FakeTransport([(403, {}, b'{"detail":"forbidden"}'), (404, {}, b"")])
    unit = timer(tmp_path, transport, clock, runpodctl=str(fake))
    assert unit.run() == "terminated"
    assert json.loads(Path(unit.receipt_path).read_text())["reason"].endswith(
        "runpodctl_confirmed_absent"
    )
    assert any(row["result"] == "runpodctl_exit_0" for row in unit.attempts)


def test_retry_after_header_is_respected_on_confirmation(tmp_path: Path) -> None:
    clock = Clock()
    transport = FakeTransport([(204, {}, b""), (429, {"retry-after": "7"}, b""), (404, {}, b"")])
    unit = timer(tmp_path, transport, clock)
    before = clock.wall
    unit.wait()
    at_deadline = clock.wall
    assert unit.terminate("test") == "terminated"
    assert clock.wall - at_deadline >= 7 and at_deadline >= before


def test_dry_run_never_sends_a_delete(tmp_path: Path) -> None:
    clock = Clock()
    transport = FakeTransport([])
    unit = timer(tmp_path, transport, clock, dry_run=True)
    assert unit.run() == "dry_run_complete"
    assert transport.calls == []


@pytest.mark.parametrize("pod_id", ["../x", "", "a/b", "bad id"])
def test_unsafe_pod_ids_are_rejected_before_any_request(tmp_path: Path, pod_id: str) -> None:
    clock = Clock()
    with pytest.raises(ValueError):
        SelfDestruct(
            pod_id=pod_id,
            token="t",
            deadline_unix=1_010.0,
            receipt_path=str(tmp_path / "r.json"),
            transport=FakeTransport([]),
            clock=lambda: clock.wall,
            monotonic=lambda: clock.mono,
            sleep=clock.sleep,
        )


def test_main_refuses_missing_key_and_out_of_ceiling_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "podabc123")
    receipt = str(tmp_path / "r.json")
    with pytest.raises(SystemExit) as info:
        main(["--deadline-unix", "9999999999", "--receipt", receipt])
    assert info.value.code == 2
    monkeypatch.setenv("RUNPOD_API_KEY", "placeholder")
    import time

    with pytest.raises(SystemExit) as info:
        main(["--deadline-unix", str(time.time() + 6000), "--receipt", receipt])
    assert info.value.code == 2
    assert not os.path.exists(receipt)
