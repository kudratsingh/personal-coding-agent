"""Checks for a Mac execution window: model-written code runs in Docker Desktop on the Mac.

For the rehearsal, pool completions and RL screening attempts execute in the Mac's Docker
sandbox, on the laptop that holds ``.env``, the Runpod key, ``gh`` and SSH credentials. An
escape from a container reaches Docker Desktop's Linux VM, its file shares, the engine and
its image store, and possibly the SSH-agent socket. Before every batch the owner's runbook
therefore checks, with this tool:

``check`` (Mac)
    1. **No credentials in the environment:** none of the variables ``.env`` defines (given
       with ``--dotenv``, names only) and none of the known secret names is set, and the SSH
       agent variables (``SSH_AUTH_SOCK`` and ``SSH_AGENT_PID``) are unset.
    2. **No keys in any SSH agent:** Docker Desktop forwards the host's agent into its VM
       (``/run/host-services/ssh-auth.sock``) whatever the child's environment holds, so
       ``ssh-add -l`` is asked of every agent socket the Mac names: the parent shell's
       ``SSH_AUTH_SOCK`` and the login session's (``launchctl getenv SSH_AUTH_SOCK``). Each
       must answer "no identities" (exit 1) or be unreachable (exit 2).
    3. **The pinned image:** the sandbox image (``scratchcoder.sandbox.runner.IMAGE``) is
       present locally (it is never pulled) with the image ID recorded at the window's start
       (``--image-id``), so a tampered image store is caught before a batch runs.
    4. **No file sharing:** a container from the pinned image asked to mount each of Docker
       Desktop's removable default shares (``SHARES``: ``/Users``, ``/Volumes``, ``/private``
       and ``/tmp``) must fail every time. With file sharing emptied each mount is refused;
       if any succeeds, the window does not start. The built-in ``/var/folders`` share
       (``RESIDUAL_SHARES``) cannot be removed: it is probed and recorded as a known residual
       (ADR 0004) but does not fail the window. Each container is removed whatever happens,
       so no ``scratchcoder-*`` container is left behind.
``exec`` (Mac)
    The same checks, then the given command (``eval-execute`` or ``rl-screen``) in an
    environment without any of those variables: the ``child_environment`` pattern, so the
    executor never holds a credential. Its exit status is returned.

Nothing here pulls an image, contacts a network or reads a secret's value.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json, utc_now
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

Run = Callable[[list[str]], "subprocess.CompletedProcess[str]"]
AgentRun = Callable[[str], "subprocess.CompletedProcess[str]"]
"""``ssh-add -l`` against one agent socket."""


def runner(environment: Mapping[str, str]) -> Run:
    """``docker`` calls in ``environment`` (the cleaned one), output captured, bounded."""

    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=120, check=False, env=dict(environment)
        )

    return run


def ssh_add_list(environ: Mapping[str, str]) -> AgentRun:
    """``ssh-add -l`` against one socket, in a minimal environment; never a key's value."""

    def run(socket: str) -> subprocess.CompletedProcess[str]:
        env = {"PATH": environ.get("PATH", "/usr/bin:/bin"), "SSH_AUTH_SOCK": socket}
        return subprocess.run(
            ["ssh-add", "-l"], capture_output=True, text=True, timeout=30, check=False, env=env
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
        timeout=10,
        check=False,
    )
    value = found.stdout.strip()
    return value or None


def agent_sockets(environ: Mapping[str, str], session: str | None) -> list[str]:
    """Every SSH agent socket the Mac names: the shell's and the login session's."""
    sockets = []
    for socket in (environ.get("SSH_AUTH_SOCK"), session):
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
) -> dict[str, Any]:
    """Run the four checks; return a record whose ``status`` is ``passed`` or ``failed``.

    ``sockets`` are the agent sockets to ask (``agent_sockets``), each through ``run_agent``.
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
        probes.append(
            {
                "share": share,
                "exit_code": mounted.returncode,
                "refused": mounted.returncode != 0 and "mounted" not in mounted.stdout,
                "residual": share in RESIDUAL_SHARES,
            }
        )
    checks["file_sharing"] = {
        "passed": all(probe["refused"] for probe in probes if not probe["residual"]),
        "shares": probes,
        "rule": (
            "a container from the pinned image cannot mount any removable default Docker "
            "Desktop share; the built-in /var/folders share is recorded as a known residual"
        ),
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "schema_version": 1,
        "kind": "mac-execution-window-check",
        "status": "passed" if passed else "failed",
        "checked_at": utc_now(),
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "exec"):
        command = commands.add_parser(name)
        command.add_argument("--image-id", required=True, help="sha256:<64 hex>, recorded once")
        command.add_argument("--dotenv", type=Path, help="the Mac's .env (variable names only)")
        command.add_argument("--receipt", type=Path, help="write the check's record here")
    commands.choices["exec"].add_argument("argv", nargs=argparse.REMAINDER)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    run: Run | None = None,
    environ: Mapping[str, str] | None = None,
    run_agent: AgentRun | None = None,
    session_socket: Callable[[], str | None] = launchd_socket,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    environment = dict(os.environ if environ is None else environ)
    dotenv = dotenv_names(args.dotenv)
    clean = clean_environment(environment, dotenv)
    docker = runner(clean) if run is None else run  # Docker never sees a credential either.
    # The agents are those of the parent shell and the login session, not the cleaned child's:
    # Docker Desktop forwards the session's agent into its VM whatever the child holds.
    sockets = agent_sockets(environment, session_socket())
    agent = ssh_add_list(environment) if run_agent is None else run_agent
    options: dict[str, Any] = {"sockets": sockets, "run_agent": agent}
    try:
        if args.command == "check":
            record = check(
                image_id=args.image_id, environ=environment, dotenv=dotenv, run=docker, **options
            )
        else:
            command = list(args.argv)
            if command[:1] == ["--"]:
                command = command[1:]
            if not command:
                parser.error("exec needs a command after --")
            record = check(
                image_id=args.image_id, environ=clean, dotenv=dotenv, run=docker, **options
            )
            record["command"] = command
    except ValueError as error:
        parser.error(str(error))
    if args.receipt is not None:
        atomic_json(args.receipt, record)
    print(
        json.dumps({"status": record["status"], "checks": record["checks"]}, sort_keys=True),
        flush=True,  # Before the child writes, so a batch's log reads in order.
    )
    if record["status"] != "passed":
        print("mac-window-check: a check failed; nothing was executed", file=sys.stderr)
        return 3
    if args.command == "check":
        return 0
    child = subprocess.run(command, env=clean, check=False)
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
