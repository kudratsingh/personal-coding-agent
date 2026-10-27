"""Tests for the bridge's SSH channel: options, child environment and the remote program.

The remote program runs here under a local ``python -I -c`` in place of ``ssh``, so the
exact code the Pod would run is exercised against a temporary bridge root.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from bridge_fixtures import RL_SHA, SESSION, attempts, mac_server

from scratchcoder.rl.bridge import REWARD, BridgeRequest, RequestName
from scratchcoder.rl.bridge_mac import (
    EXIT_REFUSED,
    REMOTE_SCRIPT,
    BridgeLeaseHeld,
    BridgeRemoteError,
    BridgeTransportError,
    DirectoryTransport,
    RequestMissing,
    RequestTooLarge,
    SSHTransport,
    dotenv_names,
    environment_refusals,
    ssh_child_environment,
    ssh_command,
)
from tools.pilot_transfer import _ssh_command, child_environment

NAME = RequestName(SESSION, 1, REWARD, 1)


def keys(tmp_path: Path) -> tuple[Path, Path]:
    identity, hosts = tmp_path / "id_ed25519", tmp_path / "known_hosts"
    identity.write_text("key")
    hosts.write_text("host")
    return identity, hosts


def test_the_ssh_command_uses_the_rehearsal_options(tmp_path: Path) -> None:
    identity, hosts = keys(tmp_path)
    ours = ssh_command("root@203.0.113.7", 22022, identity, hosts)
    theirs = _ssh_command("root@203.0.113.7", 22022, identity, hosts)
    assert ours[:-1] == theirs[:-1]  # every option and the host, in the same order
    assert ours[-1].startswith("python3 -I -c ") and "outbox" in ours[-1]
    assert "BatchMode=yes" in ours and "StrictHostKeyChecking=yes" in ours
    for host in ("-oProxyCommand=x", "a@b@c", "pod.ssh.runpod.io", "x;rm"):
        with pytest.raises(ValueError):
            ssh_command(host, 22, identity, hosts)
    for port in (0, 70000, True):
        with pytest.raises(ValueError):
            ssh_command("host", port, identity, hosts)
    link = tmp_path / "link"
    link.symlink_to(identity)
    with pytest.raises(ValueError):
        ssh_command("host", 22, link, hosts)


def test_no_secret_reaches_an_ssh_child(tmp_path: Path) -> None:
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/Users/me",
        "R2_POD_SECRET_ACCESS_KEY": "s1",
        "R2_ENDPOINT": "e",
        "RUNPOD_API_KEY": "s2",
        "SSH_AUTH_SOCK": "/tmp/agent",
        "GITHUB_TOKEN": "s3",
        "FROM_DOTENV": "s4",
    }
    assert ssh_child_environment(environ) == {"PATH": "/usr/bin", "HOME": "/Users/me"}
    # The rehearsal's own filter keeps everything but R2_ and the Runpod key; ours is a
    # strict subset of what it keeps.
    assert set(ssh_child_environment(environ)) <= set(child_environment())
    dotenv = tmp_path / ".env"
    dotenv.write_text("# comment\nexport FROM_DOTENV=s4\nR2_BUCKET=b\n\nnot a line\n")
    assert dotenv_names(dotenv) == {"FROM_DOTENV", "R2_BUCKET"}
    refused = environment_refusals(environ, dotenv)
    assert refused == ["FROM_DOTENV", "R2_ENDPOINT", "R2_POD_SECRET_ACCESS_KEY", "RUNPOD_API_KEY"]
    assert not {"s1", "s2", "s4"} & set(refused)  # names only, never values
    assert environment_refusals({"PATH": "/usr/bin"}, None) == []
    with pytest.raises(FileNotFoundError):  # a check with nothing to read must not pass
        environment_refusals({"PATH": "/usr/bin"}, tmp_path / "missing.env")
    with pytest.raises(ValueError, match="must not hold"):
        SSHTransport(["ssh"], "/workspace/bridge", environment={"SSH_AUTH_SOCK": "x"})


def test_the_transport_sends_inputs_on_stdin_only(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def run(command: list[str], **options: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append({"command": command, **options})
        listing = b'{"ready":true,"requests":{},"replies":[]}'
        return subprocess.CompletedProcess(command, 0, listing, b"")

    command = ["ssh", "-p", "22", "host", "python3 -I -c 'script'"]
    transport = SSHTransport(command, "/workspace/bridge", run=run, environment={"PATH": "/bin"})
    assert transport.listing().requests == {}
    (call,) = calls
    assert call["command"] == command  # nothing interpolated into the remote command
    assert json.loads(call["input"]) == {"root": "/workspace/bridge", "action": "list"}
    assert call["env"] == {"PATH": "/bin"} and call["timeout"] == 30.0
    for root in ("relative", "/a/../b", "/a/\x00"):
        with pytest.raises(ValueError):
            SSHTransport(command, root, run=run)


def results(*outcomes: tuple[int, bytes, bytes]) -> Any:
    queue = list(outcomes)

    def run(command: list[str], **options: Any) -> subprocess.CompletedProcess[bytes]:
        code, out, err = queue.pop(0)
        return subprocess.CompletedProcess(command, code, out, err)

    return run


def test_transport_failures_are_errors_never_listings() -> None:
    def make(*outcomes: tuple[int, bytes, bytes]) -> SSHTransport:
        return SSHTransport(["ssh"], "/r", run=results(*outcomes), environment={})

    with pytest.raises(BridgeTransportError, match="exit 255"):
        make((255, b"", b"Connection refused")).listing()
    for junk in (
        b"not json",
        b'{"requests": []}',
        b'{"ready":true,"requests":{"../x.json":1},"replies":[]}',
        b'{"ready":1,"requests":{},"replies":[]}',
    ):
        with pytest.raises(BridgeTransportError, match="malformed"):
            make((0, junk, b"")).listing()
    with pytest.raises(RequestTooLarge):
        make((3, b"", b"")).read_request(NAME.file_name, max_bytes=10)
    with pytest.raises(RequestMissing):
        make((4, b"", b"")).read_request(NAME.file_name, max_bytes=10)
    with pytest.raises(BridgeTransportError, match="too many bytes"):
        make((0, b"x" * 11, b"")).read_request(NAME.file_name, max_bytes=10)
    with pytest.raises(BridgeTransportError, match="did not confirm"):
        make((0, b'{"sha256": "0", "size": 1}', b"")).write_reply(NAME.file_name, b"reply")

    def timed_out(command: list[str], **options: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(command, 1)

    with pytest.raises(BridgeTransportError, match="timed out"):
        SSHTransport(["ssh"], "/r", run=timed_out, environment={}).listing()
    with pytest.raises(ValueError):
        make().read_request("../outbox.json", max_bytes=1)


@pytest.fixture
def remote(tmp_path: Path) -> tuple[Path, SSHTransport]:
    """A bridge root and a transport running the remote program in a local interpreter."""
    root = tmp_path / "bridge"
    for box in ("outbox", "inbox"):
        (root / box).mkdir(parents=True)
    command = [sys.executable, "-I", "-c", REMOTE_SCRIPT]
    environment = ssh_child_environment()
    return root, SSHTransport(command, str(root.resolve()), environment=environment)


def test_the_remote_program_lists_reads_and_writes(remote: tuple[Path, SSHTransport]) -> None:
    root, transport = remote
    request = BridgeRequest.build(NAME, attempts(), task_file_sha256=RL_SHA, created_unix=0)
    raw = request.to_bytes()
    (root / "outbox" / NAME.file_name).write_bytes(raw)
    (root / "outbox" / f".{NAME.file_name}.abc.tmp").write_bytes(b"partial")  # hidden
    (root / "outbox" / "notes.txt").write_text("ignored")
    other = RequestName(SESSION, 2, REWARD, 2).file_name
    (root / "outbox" / other).symlink_to(root / "outbox" / NAME.file_name)  # not regular
    (root / "inbox" / RequestName(SESSION, 3, REWARD, 3).file_name).write_text("{}")
    listing = transport.listing()
    assert dict(listing.requests) == {NAME.file_name: len(raw)}
    assert listing.replies == {RequestName(SESSION, 3, REWARD, 3).file_name}
    assert transport.read_request(NAME.file_name, max_bytes=len(raw)) == raw
    with pytest.raises(RequestTooLarge):
        transport.read_request(NAME.file_name, max_bytes=len(raw) - 1)
    with pytest.raises(RequestMissing):
        transport.read_request(RequestName(SESSION, 9, REWARD, 9).file_name, max_bytes=10)
    with pytest.raises(BridgeTransportError):  # the symlink is never followed
        transport.read_request(other, max_bytes=len(raw))
    transport.write_reply(NAME.file_name, b'{"reply": true}')
    assert (root / "inbox" / NAME.file_name).read_bytes() == b'{"reply": true}'
    assert not [path for path in (root / "inbox").iterdir() if path.name.startswith(".")]
    transport.write_reply(NAME.file_name, b"replaced")  # a re-push replaces atomically
    assert (root / "inbox" / NAME.file_name).read_bytes() == b"replaced"


def test_the_remote_program_refuses_unsafe_roots_and_names(tmp_path: Path) -> None:
    real = tmp_path / "real"
    for box in ("outbox", "inbox"):
        (real / box).mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real)
    command = [sys.executable, "-I", "-c", REMOTE_SCRIPT]
    through_link = SSHTransport(command, str(link), environment=ssh_child_environment())
    with pytest.raises(BridgeTransportError):  # no path component is followed as a symlink
        through_link.listing()
    direct = SSHTransport(command, str(real.resolve()), environment=ssh_child_environment())
    payload = json.dumps({"root": str(real.resolve()), "action": "write", "name": "../x.json"})
    result = subprocess.run(command, input=payload.encode(), capture_output=True, check=False)
    assert result.returncode != 0 and not (tmp_path / "x.json").exists()
    payload = json.dumps({"root": str(real.resolve()), "action": "delete"})
    result = subprocess.run(command, input=payload.encode(), capture_output=True, check=False)
    assert result.returncode != 0
    assert direct.listing().requests == {}
    assert os.listdir(real / "inbox") == []


def test_a_root_the_remote_program_would_refuse_is_refused_before_any_ssh_call() -> None:
    """A trailing or doubled slash passed the Mac's check but failed every remote call, which
    looked like an SSH outage for 30 polls."""
    for root in ("/x/", "//x", "/x//y", "/x/./y", "/x/../y", "/", "x", "", "/x\x00y"):
        with pytest.raises(ValueError, match="bridge root"):
            SSHTransport(["ssh"], root)
    for root in ("/x", "/workspace/rl-bridge", "/a/b.c/d-e_f"):
        assert SSHTransport(["ssh"], root, environment={}).root == root


def test_a_refusal_by_the_remote_program_is_raised_at_once_not_retried(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "bridge"
    for box in ("outbox", "inbox"):
        (root / box).mkdir(parents=True)
    command = [sys.executable, "-I", "-c", REMOTE_SCRIPT]
    transport = SSHTransport(command, str(root), environment=ssh_child_environment())
    transport.root = str(root) + "/"  # past the local check, as a remote-only rule would be
    with pytest.raises(BridgeRemoteError, match="unsafe component"):
        transport.listing()
    transport.root = str(root)
    payload = json.dumps({"root": str(root), "action": "read", "name": "../x", "max_bytes": 1})
    result = subprocess.run(command, input=payload.encode(), capture_output=True, check=False)
    assert result.returncode == EXIT_REFUSED  # a refused input is not an outage (exit 1)
    # The server does not retry it: one call, then the error ends the poll.
    calls: list[int] = []

    class Counting(SSHTransport):
        def _call(self, request: dict[str, Any], *, timeout: float, limit: int) -> bytes:
            calls.append(1)
            return super()._call(request, timeout=timeout, limit=limit)

    broken = Counting(command, str(root), environment=ssh_child_environment())
    broken.root = str(root) + "//"
    with mac_server(tmp_path / "pod", tmp_path / "mac", transport=broken) as server:
        with pytest.raises(BridgeRemoteError):
            server.poll_once()
    assert calls == [1]


SERVER_A = "a" * 32
SERVER_B = "b" * 32


@pytest.fixture(params=["directory", "remote"])
def leased(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Path, Any]:
    """A ready bridge root and a factory of transports over it: the directory-pair fake and
    the exact remote program must agree on the lease."""
    root = tmp_path.resolve() / "bridge"
    for box in ("outbox", "inbox"):
        (root / box).mkdir(parents=True)

    def make() -> DirectoryTransport | SSHTransport:
        if request.param == "directory":
            return DirectoryTransport(root)
        command = [sys.executable, "-I", "-c", REMOTE_SCRIPT]
        return SSHTransport(command, str(root), environment=ssh_child_environment())

    return root, make


def write_lease(root: Path, server: str, renewed: float, seconds: float = 60.0) -> None:
    body = {"format": "rl-bridge-lease-v1", "server": server, "renewed": renewed}
    (root / "server.lease").write_text(json.dumps({**body, "seconds": seconds}))


def test_one_server_holds_the_lease_and_a_second_is_refused(leased: tuple[Path, Any]) -> None:
    root, make = leased
    first, second = make(), make()
    first.claim(SERVER_A, 60)
    second.claim(SERVER_B, 60)
    assert first.listing().ready
    held = json.loads((root / "server.lease").read_text())
    assert held["server"] == SERVER_A and held["seconds"] == 60
    with pytest.raises(BridgeLeaseHeld) as caught:
        second.listing()
    assert caught.value.holder == SERVER_A and 0 <= caught.value.age_seconds < 5
    assert json.loads((root / "server.lease").read_text()) == held  # the refusal wrote nothing
    first.listing()  # a renewal keeps it
    with pytest.raises(BridgeLeaseHeld):
        second.listing()
    # A restart of the same server (same ID, a new transport) takes its own lease at once.
    again = make()
    again.claim(SERVER_A, 60)
    assert again.listing().ready
    # Without a claim a listing neither takes nor checks a lease.
    plain = make()
    assert plain.listing().ready


def test_a_lapsed_unreadable_or_future_lease_can_be_taken(leased: tuple[Path, Any]) -> None:
    root, make = leased
    now = time.time()
    taker = make()
    taker.claim(SERVER_B, 60)
    for setup in (
        lambda: write_lease(root, SERVER_A, now - 120),  # silent for twice its length
        lambda: (root / "server.lease").write_text("not json"),  # unreadable
        lambda: (root / "server.lease").write_text(json.dumps({"server": SERVER_A})),  # partial
        lambda: write_lease(root, SERVER_A, now + 3600),  # a clock step: dated in the future
    ):
        setup()
        assert taker.listing().ready
        assert json.loads((root / "server.lease").read_text())["server"] == SERVER_B
        (root / "server.lease").unlink()
    write_lease(root, SERVER_A, now - 5, seconds=60)  # live: renewed 5 s into a 60 s lease
    with pytest.raises(BridgeLeaseHeld):
        taker.listing()
    write_lease(root, SERVER_A, now - 5, seconds=2)  # the holder's own length decides
    assert taker.listing().ready


def test_the_lease_files_are_swept_by_neither_the_listing_nor_the_boxes(
    leased: tuple[Path, Any],
) -> None:
    root, make = leased
    transport = make()
    transport.claim(SERVER_A, 60)
    request = BridgeRequest.build(NAME, attempts(), task_file_sha256=RL_SHA, created_unix=0)
    (root / "outbox" / NAME.file_name).write_bytes(request.to_bytes())
    listing = transport.listing()
    assert list(listing.requests) == [NAME.file_name]
    assert {path.name for path in root.iterdir()} >= {"server.lease", "server.lease.lock"}
    assert not [p for p in root.iterdir() if p.name.startswith(".")]  # no temporary left


def test_a_pod_that_is_not_ready_takes_no_lease(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "bridge"
    root.mkdir()
    for transport in (
        DirectoryTransport(root),
        SSHTransport(
            [sys.executable, "-I", "-c", REMOTE_SCRIPT],
            str(root),
            environment=ssh_child_environment(),
        ),
    ):
        transport.claim(SERVER_A, 60)
        assert transport.listing().ready is False
    assert os.listdir(root) == []


def test_the_two_implementations_share_one_lease_file(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "bridge"
    for box in ("outbox", "inbox"):
        (root / box).mkdir(parents=True)
    fake = DirectoryTransport(root)
    fake.claim(SERVER_A, 60)
    fake.listing()
    remote = SSHTransport(
        [sys.executable, "-I", "-c", REMOTE_SCRIPT], str(root), environment=ssh_child_environment()
    )
    remote.claim(SERVER_B, 60)
    with pytest.raises(BridgeLeaseHeld):
        remote.listing()


def test_the_remote_program_refuses_an_invalid_lease_claim(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "bridge"
    for box in ("outbox", "inbox"):
        (root / box).mkdir(parents=True)
    command = [sys.executable, "-I", "-c", REMOTE_SCRIPT]
    for claim in (
        {"server": "short", "seconds": 60},
        {"server": SERVER_A, "seconds": 0},
        {"server": SERVER_A, "seconds": 7200},
        {"server": SERVER_A, "seconds": True},
        {"server": SERVER_A},
        "a string",
    ):
        payload = json.dumps({"root": str(root), "action": "list", "lease": claim})
        result = subprocess.run(command, input=payload.encode(), capture_output=True, check=False)
        assert result.returncode == EXIT_REFUSED, claim
    assert not (root / "server.lease").exists()
    transport = SSHTransport(command, str(root), environment=ssh_child_environment())
    for server, seconds in (("short", 60), (SERVER_A, 0), (SERVER_A, 7200), (SERVER_A, math.inf)):
        with pytest.raises(ValueError):
            transport.claim(server, seconds)


def test_a_malformed_lease_reply_is_a_transport_error() -> None:
    def make(output: bytes) -> SSHTransport:
        transport = SSHTransport(["ssh"], "/r", run=results((0, output, b"")), environment={})
        transport.claim(SERVER_A, 60)
        return transport

    base = {"ready": True, "requests": {}, "replies": []}
    held = {"held": True, "holder": SERVER_A, "age": 0.0}
    assert make(json.dumps({**base, "lease": held}).encode()).listing().ready
    for junk in (
        base,  # a Pod that did not answer the claim must not be assumed to have granted it
        {**base, "lease": {**held, "held": 1}},
        {**base, "lease": {**held, "age": "x"}},
        {**base, "lease": {"held": True}},
        {**base, "lease": "yes"},
    ):
        with pytest.raises(BridgeTransportError, match="malformed"):
            make(json.dumps(junk).encode()).listing()
    refused = {**base, "lease": {"held": False, "holder": SERVER_B, "age": 3.5}}
    with pytest.raises(BridgeLeaseHeld) as caught:
        make(json.dumps(refused).encode()).listing()
    assert caught.value.holder == SERVER_B and caught.value.age_seconds == 3.5
    # A Pod that is not ready answers without a lease, and that is not malformed.
    idle = {"ready": False, "requests": {}, "replies": []}
    assert make(json.dumps(idle).encode()).listing().ready is False


def test_the_claim_travels_in_the_listing_request_only() -> None:
    sent: list[dict[str, Any]] = []

    def run(command: list[str], **options: Any) -> subprocess.CompletedProcess[bytes]:
        sent.append(json.loads(options["input"]))
        reply: dict[str, Any] = {"ready": True, "requests": {}, "replies": []}
        if "lease" in sent[-1]:
            reply["lease"] = {"held": True, "holder": SERVER_A, "age": 0.0}
        return subprocess.CompletedProcess(command, 0, json.dumps(reply).encode(), b"")

    transport = SSHTransport(["ssh"], "/r", run=run, environment={})
    transport.listing()
    transport.claim(SERVER_A, 45)
    transport.listing()
    assert sent[0] == {"root": "/r", "action": "list"}
    assert sent[1] == {"root": "/r", "action": "list", "lease": {"server": SERVER_A, "seconds": 45}}


def test_a_pod_without_a_bridge_root_yet_is_not_ready(tmp_path: Path) -> None:
    command = [sys.executable, "-I", "-c", REMOTE_SCRIPT]
    root = tmp_path.resolve() / "bridge"
    transport = SSHTransport(command, str(root), environment=ssh_child_environment())
    assert transport.listing().ready is False  # no root
    (root / "outbox").mkdir(parents=True)
    assert transport.listing().ready is False  # no inbox yet
    (root / "inbox").mkdir()
    listing = transport.listing()
    assert listing.ready is True and listing.requests == {}
    with pytest.raises(BridgeTransportError):  # only a listing treats a missing root as idle
        SSHTransport(command, str(tmp_path.resolve() / "nowhere")).write_reply(
            NAME.file_name, b"{}"
        )
    assert DirectoryTransport(tmp_path / "absent").listing().ready is False
    with mac_server(tmp_path / "absent", tmp_path / "mac") as server:
        assert server.poll_once() == []
    progress = json.loads((tmp_path / "mac" / "progress.json").read_bytes())
    assert progress["pod_ready"] is False and progress["polls"] == 1
    assert progress["transport_failures"] == 0
