"""Each rehearsal Pod's watchdog records its own PID, so one can be restarted alone."""

import json
import os
import signal
from pathlib import Path
from typing import Any

import pytest

from tools import runpod_watchdog
from tools.runpod_watchdog import build_parser, release_pidfile, write_pidfile


def arguments(tmp_path: Path, *extra: str) -> list[str]:
    identity = tmp_path / "pod-identity.json"
    identity.write_text(
        json.dumps(
            {
                "id": "podfixture01",
                "name": "rehearsal-a40-s-a",
                "gpuId": "NVIDIA A40",
                "gpuCount": 1,
                "dataCenterId": "CA-MTL-1",
                "createdAt": "2027-01-15T08:00:00+00:00",
            }
        )
    )
    return [
        "--mode",
        "rehearsal",
        "--identity",
        str(identity),
        "--heartbeat",
        str(tmp_path / "supervisor" / "heartbeat.json"),
        "--receipt",
        str(tmp_path / "watchdog-receipt.json"),
        "--max-wall-seconds",
        "45000",
        "--max-cost-usd",
        "7.54",
        "--quoted-total-hourly",
        "0.60",
        *extra,
    ]


def test_pidfile_names_this_process_and_is_released_only_by_it(tmp_path: Path) -> None:
    path = tmp_path / "ctrl" / "watchdog.pid"
    write_pidfile(path)
    assert path.read_text() == f"{os.getpid()}\n"
    release_pidfile(path)
    assert not path.exists()
    path.write_text("1\n")  # Another run's watchdog, or a newer one, now owns the file.
    release_pidfile(path)
    assert path.read_text() == "1\n"
    release_pidfile(tmp_path / "missing.pid")


def test_main_holds_the_pidfile_while_watching_and_releases_it_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "placeholder")
    path = tmp_path / "watchdog.pid"
    seen: list[str] = []

    def watch(*args: Any) -> int:
        seen.append(path.read_text().strip())
        return 0

    monkeypatch.setattr(runpod_watchdog, "_watch", watch)
    before = signal.getsignal(signal.SIGTERM)
    assert runpod_watchdog.main(arguments(tmp_path, "--pidfile", str(path))) == 0
    assert seen == [str(os.getpid())] and not path.exists()
    assert signal.getsignal(signal.SIGTERM) == before  # The handler is restored.


def test_sigterm_from_a_per_run_restart_releases_the_pidfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "placeholder")
    path = tmp_path / "watchdog.pid"

    def watch(*args: Any) -> int:
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM must stop the watchdog")

    monkeypatch.setattr(runpod_watchdog, "_watch", watch)
    with pytest.raises(SystemExit) as error:
        runpod_watchdog.main(arguments(tmp_path, "--pidfile", str(path)))
    assert error.value.code == 128 + signal.SIGTERM and not path.exists()


def test_without_a_pidfile_the_watchdog_behaves_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "placeholder")
    calls: list[int] = []

    def watch(*args: Any) -> int:
        calls.append(1)
        return 0

    monkeypatch.setattr(runpod_watchdog, "_watch", watch)
    assert runpod_watchdog.main(arguments(tmp_path)) == 0
    assert calls == [1] and not list(tmp_path.glob("*.pid"))
    assert build_parser().parse_args(arguments(tmp_path)).pidfile is None
