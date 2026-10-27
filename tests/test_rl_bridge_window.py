"""Run 2's reward bridge repeats the Mac execution-window checks while it runs.

The owner's run 2 gate rulings: the bridge runs the window checks (no ``.env`` name, secret
or SSH agent variable in its own environment, no identity in any SSH agent, the pinned image,
Docker Desktop's file sharing) at its start and again before scoring a request once its last
pass is older than its interval. A failed or errored pass stops it before anything more is
scored, with a receipt; the Pod then sees a lost bridge. Offline: fake checks, a fake
``docker`` and a fake ``ssh-add`` stand in, and no container runs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from bridge_fixtures import (
    DEV_SHA,
    DEV_TASKS,
    RL_SHA,
    SESSION,
    TRAIN,
    CountingDev,
    CountingRewards,
    FakeClock,
    attempts,
)
from test_mac_window_check import FakeAgent, FakeDocker
from test_rl_bridge import lease_file, serve_in_thread, wait_until, write_request

from scratchcoder.cli import main
from scratchcoder.rl import bridge_command
from scratchcoder.rl.bridge import REWARD, BridgeRequest, RequestName
from scratchcoder.rl.bridge_command import window_pass
from scratchcoder.rl.bridge_mac import (
    WINDOW_CHECK_INTERVAL_SECONDS,
    BridgeServer,
    DirectoryTransport,
    WindowCheckFailed,
)

IMAGE_ID = "sha256:" + "d" * 64


def request(seq: int) -> BridgeRequest:
    return BridgeRequest.build(
        RequestName(SESSION, seq, REWARD, seq),
        attempts(),
        task_file_sha256=RL_SHA,
        created_unix=1_800_000_000.0,
    )


class Passes:
    """A window check returning the scripted outcomes in turn (the last repeats): a status,
    or an exception to raise. ``order`` records each pass beside the Mac's scoring calls."""

    def __init__(self, *outcomes: str | BaseException, order: list[str] | None = None) -> None:
        self.outcomes = list(outcomes)
        self.order = order if order is not None else []
        self.calls = 0

    def __call__(self) -> Mapping[str, Any]:
        self.calls += 1
        self.order.append("pass")
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        failed = outcome != "passed"
        return {
            "kind": "mac-execution-window-check",
            "status": outcome,
            "checks": {
                "environment": {"passed": True, "set": []},
                "file_sharing": {"passed": not failed, "shares": []},
            },
        }


class OrderedRewards(CountingRewards):
    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self.order = order

    def rewards(self, update: int, items: Any) -> Any:
        self.order.append("score")
        return super().rewards(update, items)


def server_for(
    pod: Path,
    state: Path,
    clock: FakeClock,
    *,
    window_check: Callable[[], Mapping[str, Any]],
    rewards: Any = None,
    window_check_seconds: float = WINDOW_CHECK_INTERVAL_SECONDS,
) -> BridgeServer:
    return BridgeServer(
        DirectoryTransport(pod, wall=clock.wall),
        reward_tasks=TRAIN,
        dev_tasks=DEV_TASKS,
        pod_task_sha256={"reward": RL_SHA, "dev": DEV_SHA},
        state=state,
        reward_source=rewards if rewards is not None else CountingRewards(),
        dev_execute=CountingDev(),
        window_check=window_check,
        window_check_seconds=window_check_seconds,
        clock=clock.clock,
        wall=clock.wall,
        sleep=clock.sleep,
    )


def receipts(state: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text()) for path in sorted((state / "window-checks").glob("*.json"))
    ]


def answered(pod: Path, seq: int) -> bool:
    return (pod / "inbox" / request(seq).name.file_name).exists()


def test_a_pass_runs_at_start_and_again_before_a_request_once_the_last_is_old(
    tmp_path: Path,
) -> None:
    pod, state, clock = tmp_path / "pod", tmp_path / "mac", FakeClock()
    order: list[str] = []
    passes = Passes("passed", order=order)
    rewards = OrderedRewards(order)
    write_request(pod, request(1))
    with server_for(pod, state, clock, window_check=passes, rewards=rewards) as server:
        server.serve(poll_seconds=0.0, max_polls=1, progress_seconds=None)
        assert answered(pod, 1) and order == ["pass", "score"]  # The start pass, then 1.
        clock.now += WINDOW_CHECK_INTERVAL_SECONDS - 1
        write_request(pod, request(2))
        server.poll_once()
        assert answered(pod, 2) and order == ["pass", "score", "score"]  # 899 s: no pass.
        clock.now += 1
        write_request(pod, request(3))
        server.poll_once()
        assert answered(pod, 3) and order == ["pass", "score", "score", "pass", "score"]
    rows = receipts(state)
    assert [row["bridge_pass"]["pass"] for row in rows] == [1, 2]
    assert rows[0]["bridge_pass"]["reason"] == "start"
    assert rows[1]["bridge_pass"]["reason"] == f"before scoring {request(3).request_id}"
    assert all(row["status"] == "passed" and row["bridge_pass"]["failed"] == [] for row in rows)
    progress = json.loads((state / "progress.json").read_text())
    assert progress["window_check"]["passes"] == 2
    assert progress["window_check"]["interval_seconds"] == 900.0
    events = [json.loads(line) for line in (state / "events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events].count("window_check") == 2


def test_a_failed_pass_at_start_scores_nothing_and_records_why(tmp_path: Path) -> None:
    pod, state, clock = tmp_path / "pod", tmp_path / "mac", FakeClock()
    rewards = CountingRewards()
    write_request(pod, request(1))
    with server_for(pod, state, clock, window_check=Passes("failed"), rewards=rewards) as server:
        with pytest.raises(WindowCheckFailed, match="file_sharing"):
            server.serve(poll_seconds=0.0, max_polls=3, progress_seconds=None)
    assert rewards.calls == [] and not answered(pod, 1)
    assert not list((state / "replies").iterdir())
    progress = json.loads((state / "progress.json").read_text())
    assert progress["state"] == "failed" and "file_sharing" in progress["last_error"]
    (row,) = receipts(state)
    assert row["status"] == "failed" and row["bridge_pass"]["failed"] == ["file_sharing"]


@pytest.mark.parametrize(
    "outcome, failed",
    [("failed", ["file_sharing"]), (OSError("docker did not start"), ["error"])],
)
def test_a_failed_or_errored_pass_before_a_request_leaves_it_unanswered(
    tmp_path: Path, outcome: str | BaseException, failed: list[str]
) -> None:
    """The start pass passes; 10 s later a request arrives with the interval (5 s here) gone,
    the pass before it fails or raises, and the server stops without scoring or answering it:
    its lease lapses, and the Pod gives the wait up as a lost bridge."""
    pod, state, clock = tmp_path / "pod", tmp_path / "mac", FakeClock()
    (pod / "outbox").mkdir(parents=True)
    (pod / "inbox").mkdir()
    clock.hooks.append(lambda: write_request(pod, request(1)))
    rewards = CountingRewards()
    passes = Passes("passed", outcome)
    with server_for(
        pod, state, clock, window_check=passes, rewards=rewards, window_check_seconds=5.0
    ) as server:
        with pytest.raises(WindowCheckFailed, match="nothing more is scored"):
            server.serve(poll_seconds=10.0, max_polls=5, progress_seconds=None)
    assert passes.calls == 2 and rewards.calls == [] and not answered(pod, 1)
    assert (pod / "outbox" / request(1).name.file_name).exists()  # Still waiting, unscored.
    rows = receipts(state)
    assert [row["bridge_pass"]["failed"] for row in rows] == [[], failed]
    if isinstance(outcome, BaseException):
        assert rows[1]["status"] == "error" and "docker did not start" in rows[1]["error"]
    progress = json.loads((state / "progress.json").read_text())
    assert progress["state"] == "failed" and progress["answered"] == 0


def test_the_lease_is_renewed_while_a_pass_runs(tmp_path: Path) -> None:
    """A pass may take minutes at its worst; the ticker keeps the lease fresh meanwhile, so a
    slow pass never reads as a dead Mac."""
    pod = tmp_path / "pod"
    (pod / "outbox").mkdir(parents=True)
    (pod / "inbox").mkdir()
    renewals: list[float] = []

    def slow_pass() -> Mapping[str, Any]:
        wait_until(lambda: (pod / "server.lease").exists(), "a lease during the pass")
        first = lease_file(pod)["renewed"]
        wait_until(lambda: lease_file(pod)["renewed"] > first, "a renewal during the pass")
        renewals.append(lease_file(pod)["renewed"])
        return {"status": "passed", "checks": {}}

    server = BridgeServer(
        DirectoryTransport(pod),
        reward_tasks=TRAIN,
        dev_tasks=DEV_TASKS,
        pod_task_sha256={"reward": RL_SHA, "dev": DEV_SHA},
        state=tmp_path / "mac",
        reward_source=CountingRewards(),
        dev_execute=CountingDev(),
        window_check=slow_pass,
    )
    errors: list[BaseException] = []
    thread, stop = serve_in_thread(server, errors)
    try:
        wait_until(lambda: bool(renewals), "the slow pass to end")
    finally:
        stop.set()
        thread.join(timeout=10)
        server.close()
    assert not errors and renewals


def test_the_bridges_pass_checks_its_own_environment_names_only(tmp_path: Path) -> None:
    """R5c's child-environment rule, enforced: the bridge runs with no ``.env`` name, secret
    or SSH agent variable, so the Pod's R2 token may stay in ``.env`` during the window. The
    receipt names a leaked variable, never its value."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("R2_POD_SECRET_ACCESS_KEY=never-printed\nMY_OWN_TOKEN=also-hidden\n")
    agent = FakeAgent()
    clean = {"PATH": "/usr/bin:/bin", "HOME": "/Users/owner"}

    def outcome(environ: dict[str, str], docker: FakeDocker | None = None) -> dict[str, Any]:
        check = window_pass(
            IMAGE_ID,
            dotenv=[dotenv],
            environ=environ,
            session_socket=lambda: "/session/Listeners",
            run=docker or FakeDocker(),
            run_agent=agent,
        )
        return dict(check())

    assert outcome(dict(clean))["status"] == "passed"
    assert agent.asked[-1:] == ["/session/Listeners"]  # The login session's agent, each pass.
    for name, value in (
        ("MY_OWN_TOKEN", "also-hidden"),
        ("R2_POD_SECRET_ACCESS_KEY", "never-printed"),
        ("SSH_AUTH_SOCK", "/tmp/agent"),
        ("RUNPOD_API_KEY", "k-value"),
    ):
        record = outcome({**clean, name: value})
        assert record["status"] == "failed", name
        assert record["checks"]["environment"]["set"] == [name]
        if name != "SSH_AUTH_SOCK":  # A socket path is recorded, as the window check does.
            assert value not in json.dumps(record)
    shared = outcome(dict(clean), FakeDocker(shared=("/Users",)))
    assert shared["checks"]["file_sharing"]["passed"] is False
    keyed = FakeAgent({"/session/Listeners": (0, "256 SHA256:k key (ED25519)\n")})
    agent_record = window_pass(
        IMAGE_ID,
        dotenv=None,
        environ=dict(clean),
        session_socket=lambda: "/session/Listeners",
        run=FakeDocker(),
        run_agent=keyed,
    )()
    assert agent_record["checks"]["ssh_agent"]["passed"] is False
    assert "SHA256:k" not in json.dumps(agent_record)
    tampered = outcome(dict(clean), FakeDocker(image_id="sha256:" + "e" * 64))
    assert tampered["checks"]["image"]["passed"] is False
    with pytest.raises(ValueError, match="image ID"):
        window_pass("latest", dotenv=None, environ={})


def test_the_command_runs_its_window_passes_on_its_own_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_rl_bridge_end_to_end import command_files

    files = command_files(tmp_path)
    pod = tmp_path / "pod"
    for box in ("outbox", "inbox"):
        (pod / box).mkdir(parents=True)
    built: list[dict[str, Any]] = []

    def fake_pass(image_id: str, **options: Any) -> Callable[[], Mapping[str, Any]]:
        built.append({"image_id": image_id, **options})
        return Passes("passed")

    monkeypatch.setattr(bridge_command, "window_pass", fake_pass)
    environ = {"PATH": "/usr/bin", "SSH_AUTH_SOCK": "/tmp/agent"}
    status = bridge_command.serve_command(
        host="unused",
        port=22,
        identity_file=tmp_path / "unused",
        known_hosts=tmp_path / "unused",
        remote_root="/unused",
        pod_reward_sha256=RL_SHA,
        pod_dev_sha256=DEV_SHA,
        state=tmp_path / "mac",
        reward_timeout=600,
        dev_timeout=300,
        poll_seconds=0.0,
        dotenv=None,
        image_id=IMAGE_ID,
        max_polls=1,
        environ=environ,
        transport=DirectoryTransport(pod),
        **files,
    )
    assert status == 0 and len(built) == 1
    assert built[0]["image_id"] == IMAGE_ID and built[0]["environ"] is environ
    assert "SSH_AUTH_SOCK" not in environ  # Cleared before the first pass reads it.
    (row,) = receipts(tmp_path / "mac")
    assert row["bridge_pass"]["reason"] == "start"


def test_the_cli_exits_4_when_a_window_pass_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from test_rl_bridge_end_to_end import cli_arguments

    def refuse(**options: Any) -> int:
        assert options["image_id"] == IMAGE_ID and options["window_check_seconds"] == 900.0
        raise WindowCheckFailed("the Mac execution-window check failed (ssh_agent)")

    monkeypatch.setattr(bridge_command, "serve_command", refuse)
    (tmp_path / "empty.env").write_text("")
    assert main([*cli_arguments(tmp_path), f"--dotenv={tmp_path / 'empty.env'}"]) == 4
    assert "ssh_agent" in capsys.readouterr().err
    arguments = [item for item in cli_arguments(tmp_path) if not item.startswith("--image-id")]
    with pytest.raises(SystemExit):  # The window passes need the pinned image's ID.
        main([*arguments, "--no-dotenv"])


def test_the_pods_timeouts_hold_the_checks_at_their_call_bounds() -> None:
    """The terms the Pod's timeouts hold for the Mac's checks before a request
    (``posttrain_ops.bridge_timeouts``) are the bridge's own call bounds: a pass's 11
    ``docker`` calls at 60 s, two ``ssh-add`` at 30 s and one ``launchctl`` at 10 s; the image
    check's 4 tries at 30 s and its 1, 2 and 4 s waits."""
    from scratchcoder.rl import bridge_mac
    from scratchcoder.sandbox import window
    from tools import posttrain_ops

    assert window.docker_calls() == 1 + 2 * len(window.SHARES + window.RESIDUAL_SHARES) == 11
    assert bridge_mac.WINDOW_DOCKER_TIMEOUT_SECONDS == 60.0
    assert (window.AGENT_TIMEOUT_SECONDS, window.LAUNCHCTL_TIMEOUT_SECONDS) == (30.0, 10.0)
    assert window.MAX_AGENT_SOCKETS == len(window.agent_sockets({"SSH_AUTH_SOCK": "/a"}, "/b"))
    assert bridge_mac.window_check_seconds_bound() == window.pass_seconds_bound(60.0) == 730.0
    assert (bridge_mac.IMAGE_CHECK_RETRIES, bridge_mac.IMAGE_CHECK_TIMEOUT_SECONDS) == (3, 30.0)
    assert bridge_mac.image_check_seconds_bound() == 127.0
    bounds = posttrain_ops.bridge_timeouts(attempts_per_request=256, dev_programs=1_800)
    assert (bounds.window_check_seconds, bounds.image_check_seconds) == (730.0, 127.0)


def test_the_interval_must_be_positive(tmp_path: Path) -> None:
    for bad in (0, -1.0, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="window_check_seconds"):
            server_for(
                tmp_path / "pod",
                tmp_path / f"mac-{time.monotonic_ns()}",
                FakeClock(),
                window_check=Passes("passed"),
                window_check_seconds=bad,
            )
