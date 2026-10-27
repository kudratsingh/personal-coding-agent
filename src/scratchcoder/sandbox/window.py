"""The Mac execution-window checks: model-written code runs in Docker Desktop on the Mac.

For the rehearsal, pool completions, RL screening attempts and run 2's rewards and dev
verdicts execute in the Mac's Docker sandbox, on the laptop that holds ``.env``, the Runpod
key, ``gh`` and SSH credentials. An escape from a container reaches Docker Desktop's Linux
VM, its file shares, the engine and its image store, and possibly the SSH-agent socket.
``check`` runs four checks before code runs:

1. **No credentials in the environment:** none of the variables ``.env`` defines (names
   only) and none of the known secret names is set, and the SSH agent variables
   (``SSH_AUTH_SOCK`` and ``SSH_AGENT_PID``) are unset.
2. **No keys in any SSH agent:** Docker Desktop forwards the host's agent into its VM
   (``/run/host-services/ssh-auth.sock``) whatever the child's environment holds, so
   ``ssh-add -l`` is asked of every agent socket the Mac names: the parent shell's
   ``SSH_AUTH_SOCK`` and the login session's (``launchctl getenv SSH_AUTH_SOCK``). Each must
   answer "no identities" (exit 1) or be unreachable (exit 2). ``tools/mac_window_check.py
   exec`` hands its child the parent shell's socket path as ``SHELL_AGENT_SOCKET`` (a path,
   never a key), so a child that repeats the checks (run 2's bridge) still asks it.
3. **The pinned image:** the sandbox image (``scratchcoder.sandbox.runner.IMAGE``) is present
   locally (it is never pulled) with the image ID recorded at the window's start, so a
   tampered image store is caught before a batch runs.
4. **No file sharing:** a container from the pinned image asked to mount each of Docker
   Desktop's removable default shares (``SHARES``: ``/Users``, ``/Volumes``, ``/private`` and
   ``/tmp``) must fail every time. The built-in ``/var/folders`` share (``RESIDUAL_SHARES``)
   cannot be removed: it is probed and recorded as a known residual (ADR 0004) but does not
   fail the window. Each container is removed whatever happens, so no ``scratchcoder-*``
   container is left behind. With ``residual_control`` (run 2's bridge passes) the residual
   is a positive control: it must mount, and a removable share counts as refused only when
   Docker says the mount was denied, so a daemon that fails every container start for
   another reason fails the pass instead of passing it.

``tools/mac_window_check.py`` runs them before a batch; run 2's reward bridge
(``scratchcoder rl-bridge``) runs them at its start and again before scoring a request once
its last pass is older than its interval (``scratchcoder.rl.bridge_mac``). Every call is
bounded (``DOCKER_TIMEOUT_SECONDS``, ``AGENT_TIMEOUT_SECONDS``, ``LAUNCHCTL_TIMEOUT_SECONDS``),
so one pass has a worst case (``pass_seconds_bound``). Nothing here pulls an image, contacts
a network or reads a secret's value.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scratchcoder.common.records import utc_now
from scratchcoder.sandbox.runner import ENDPOINT, IMAGE

SECRET_NAMES = (
    "RUNPOD_API_KEY",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)
SECRET_PREFIXES = ("R2_", "AWS_", "CLOUDFLARE_")
AGENT_NAMES = ("SSH_AUTH_SOCK", "SSH_AGENT_PID")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
DOTENV_NAME = re.compile(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
MOUNT_PROBE = "import os; os.listdir('/x'); print('mounted')"
SHARES = ("/Users", "/Volumes", "/private", "/tmp")
"""Docker Desktop's removable default file shares on macOS: an owner who removes only ``/Users``
still shares the others, so every one is probed and each mount must be refused."""
RESIDUAL_SHARES = ("/var/folders",)
"""Docker Desktop's built-in share, which its settings cannot remove. ADR 0004 records it as a
known residual: the sandbox never asks for a mount, so reaching it needs an escape into the
Docker VM first. It is probed and recorded, but a successful mount does not fail the window."""
NO_IDENTITIES = "no identities"
DOCKER_TIMEOUT_SECONDS = 120.0
"""The bound on one ``docker`` call of ``tools/mac_window_check.py``."""
AGENT_TIMEOUT_SECONDS = 30.0
"""The bound on one ``ssh-add -l``."""
LAUNCHCTL_TIMEOUT_SECONDS = 10.0
"""The bound on ``launchctl getenv SSH_AUTH_SOCK``."""
MAX_AGENT_SOCKETS = 2
"""The shell's socket and the login session's: ``agent_sockets`` names at most these two."""
SHELL_AGENT_SOCKET = "SCRATCHCODER_SHELL_AGENT_SOCKET"
"""The variable through which ``mac_window_check exec`` hands its child the parent shell's
agent socket path (``SSH_AUTH_SOCK`` itself is withheld from the child). Not an agent
variable: no SSH client reads it, so the child holds no agent, only the path to ask."""
MOUNT_DENIED = "mounts denied"
"""Docker Desktop's refusal of a share it does not export (``docker run`` exits 125 with
``Error response from daemon: mounts denied: ...``), matched case-insensitively."""

Run = Callable[[list[str]], "subprocess.CompletedProcess[str]"]
AgentRun = Callable[[str], "subprocess.CompletedProcess[str]"]
"""``ssh-add -l`` against one agent socket."""


def docker_calls() -> int:
    """The ``docker`` calls of one pass: one ``image inspect``, then for each probed share
    one ``run`` and one ``rm -f``."""
    return 1 + 2 * len(SHARES + RESIDUAL_SHARES)


def pass_seconds_bound(docker_timeout: float = DOCKER_TIMEOUT_SECONDS) -> float:
    """The longest one pass of ``check`` can take, every call to its bound: 11 ``docker`` calls,
    two ``ssh-add -l`` and one ``launchctl getenv``. At a 60 s ``docker`` bound (the bridge's)
    that is 11 x 60 + 2 x 30 + 10 = 730 s."""
    return (
        docker_calls() * float(docker_timeout)
        + MAX_AGENT_SOCKETS * AGENT_TIMEOUT_SECONDS
        + LAUNCHCTL_TIMEOUT_SECONDS
    )


def runner(environment: Mapping[str, str], *, timeout: float = DOCKER_TIMEOUT_SECONDS) -> Run:
    """``docker`` calls in ``environment`` (the cleaned one), output captured, bounded."""

    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=dict(environment),
        )

    return run


def ssh_add_list(environ: Mapping[str, str]) -> AgentRun:
    """``ssh-add -l`` against one socket, in a minimal environment; never a key's value."""

    def run(socket: str) -> subprocess.CompletedProcess[str]:
        env = {"PATH": environ.get("PATH", "/usr/bin:/bin"), "SSH_AUTH_SOCK": socket}
        return subprocess.run(
            ["ssh-add", "-l"],
            capture_output=True,
            text=True,
            timeout=AGENT_TIMEOUT_SECONDS,
            check=False,
            env=env,
        )

    return run


def launchd_socket() -> str | None:
    """The login session's agent socket on macOS (``launchctl getenv SSH_AUTH_SOCK``)."""
    if sys.platform != "darwin" or shutil.which("launchctl") is None:
        return None
    found = subprocess.run(
        ["launchctl", "getenv", "SSH_AUTH_SOCK"],
        capture_output=True,
        text=True,
        timeout=LAUNCHCTL_TIMEOUT_SECONDS,
        check=False,
    )
    value = found.stdout.strip()
    return value or None


def agent_sockets(environ: Mapping[str, str], session: str | None) -> list[str]:
    """Every SSH agent socket the Mac names: the shell's (its ``SSH_AUTH_SOCK``, or the path
    ``mac_window_check exec`` handed down as ``SHELL_AGENT_SOCKET``) and the login
    session's."""
    sockets = []
    shell = environ.get("SSH_AUTH_SOCK") or environ.get(SHELL_AGENT_SOCKET)
    for socket in (shell, session):
        if socket and socket not in sockets:
            sockets.append(socket)
    return sockets


def agent_check(sockets: Sequence[str], run_agent: AgentRun) -> dict[str, Any]:
    """Each socket must hold no identity: ``ssh-add -l`` exits 1 saying so, or 2 (no agent)."""
    rows: list[dict[str, Any]] = []
    for socket in sockets:
        try:
            listed = run_agent(socket)
            code = listed.returncode
            empty = code == 2 or (
                code == 1 and NO_IDENTITIES in (listed.stdout + listed.stderr).lower()
            )
        except (OSError, subprocess.SubprocessError) as error:
            code, empty = None, False
            rows.append({"socket": socket, "exit_code": code, "error": type(error).__name__})
            continue
        rows.append({"socket": socket, "exit_code": code, "holds_no_identity": empty})
    return {
        "passed": all(row.get("holds_no_identity") is True for row in rows),
        "sockets": rows,  # Socket paths and exit codes only, never a key.
        "rule": "every SSH agent socket the Mac names holds no identity (ssh-add -l exits 1 or 2)",
    }


def dotenv_names(path: Path | None) -> list[str]:
    """The variable names a ``.env`` file defines; its values are never read into memory."""
    if path is None:
        return []
    names = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = DOTENV_NAME.match(line)
            if match and not line.lstrip().startswith("#"):
                names.append(match.group(1))
    return sorted(set(names))


def withheld(name: str, dotenv: Sequence[str]) -> bool:
    return (
        name in dotenv
        or name in SECRET_NAMES
        or name in AGENT_NAMES
        or name.startswith(SECRET_PREFIXES)
    )


def clean_environment(environ: Mapping[str, str], dotenv: Sequence[str]) -> dict[str, str]:
    """The environment without any credential or agent variable."""
    return {name: value for name, value in environ.items() if not withheld(name, dotenv)}


def check(
    *,
    image_id: str,
    environ: Mapping[str, str],
    dotenv: Sequence[str] = (),
    run: Run,
    docker: str = "docker",
    sockets: Sequence[str] = (),
    run_agent: AgentRun | None = None,
    residual_control: bool = False,
) -> dict[str, Any]:
    """Run the four checks; return a record whose ``status`` is ``passed`` or ``failed``.

    ``sockets`` are the agent sockets to ask (``agent_sockets``), each through ``run_agent``.
    With ``residual_control`` the ``/var/folders`` residual must mount (the positive control)
    and a removable share counts as refused only on Docker's mount-denied answer.
    A call that cannot finish (a ``docker`` call past its bound, a missing binary) raises: a
    check that cannot be made is never read as passed.
    """
    if not IMAGE_ID.fullmatch(image_id):
        raise ValueError("--image-id must be the local image ID, sha256:<64 hex digits>")
    leaked = sorted(name for name in environ if withheld(name, dotenv))
    checks: dict[str, Any] = {
        "environment": {
            "passed": not leaked,
            "set": leaked,  # Names only.
            "rule": "no .env variable, known secret or SSH agent variable is set",
        }
    }
    if sockets and run_agent is None:
        raise ValueError("agent sockets need an ssh-add runner")
    checks["ssh_agent"] = agent_check(sockets, run_agent or ssh_add_list(environ))
    inspected = run([docker, "--host", ENDPOINT, "image", "inspect", "--format", "{{.Id}}", IMAGE])
    found = inspected.stdout.strip() if inspected.returncode == 0 else None
    checks["image"] = {
        "passed": found == image_id,
        "image": IMAGE,
        "expected_id": image_id,
        "found_id": found,
        "rule": "the pinned image is present locally with the recorded image ID; never pulled",
    }
    probes = []
    for share in SHARES + RESIDUAL_SHARES:
        name = f"scratchcoder-window-check-{uuid.uuid4().hex[:12]}"
        try:
            mounted = run(
                [
                    docker,
                    "--host",
                    ENDPOINT,
                    "run",
                    "--rm",
                    "--pull=never",
                    "--network=none",
                    "--name",
                    name,
                    "-v",
                    f"{share}:/x:ro",
                    IMAGE,
                    "python",
                    "-I",
                    "-S",
                    "-c",
                    MOUNT_PROBE,
                ]
            )
        finally:
            run([docker, "--host", ENDPOINT, "rm", "-f", name])
        denied = MOUNT_DENIED in (mounted.stderr or "").lower()
        refused = mounted.returncode != 0 and "mounted" not in mounted.stdout
        probe = {
            "share": share,
            "exit_code": mounted.returncode,
            "refused": refused and (denied or not residual_control),
            "residual": share in RESIDUAL_SHARES,
        }
        if residual_control:
            probe["mount_denied"] = denied
            if share in RESIDUAL_SHARES:
                probe["mounted"] = mounted.returncode == 0 and "mounted" in mounted.stdout
        probes.append(probe)
    passed_sharing = all(probe["refused"] for probe in probes if not probe["residual"])
    rule = (
        "a container from the pinned image cannot mount any removable default Docker "
        "Desktop share; the built-in /var/folders share is recorded as a known residual"
    )
    if residual_control:
        passed_sharing = passed_sharing and all(
            probe["mounted"] for probe in probes if probe["residual"]
        )
        rule += (
            "; it is also the positive control and must mount, and a removable share counts "
            "as refused only when Docker says the mount was denied"
        )
    checks["file_sharing"] = {
        "passed": passed_sharing,
        "shares": probes,
        "rule": rule,
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "schema_version": 1,
        "kind": "mac-execution-window-check",
        "status": "passed" if passed else "failed",
        "checked_at": utc_now(),
        "checks": checks,
    }


def failed_checks(record: Mapping[str, Any]) -> list[str]:
    """The names of the checks a record failed (names only, never a value)."""
    checks = record.get("checks")
    if not isinstance(checks, Mapping):
        return ["record"]
    return sorted(
        name
        for name, item in checks.items()
        if not (isinstance(item, Mapping) and item.get("passed") is True)
    )
