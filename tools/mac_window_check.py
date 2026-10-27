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
    The same checks, then the given command (``eval-execute``, ``rl-screen`` or
    ``rl-bridge``) in an environment without any of those variables: the
    ``child_environment`` pattern, so the executor never holds a credential. The parent
    shell's agent socket path, when it has one, is handed down as
    ``SCRATCHCODER_SHELL_AGENT_SOCKET`` (a path, which no SSH client reads), so the bridge's
    repeated passes ask that agent too. Its exit status is returned.

The checks themselves live in ``scratchcoder.sandbox.window``, where run 2's reward bridge
repeats them while it runs. Nothing here pulls an image, contacts a network or reads a
secret's value.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json
from scratchcoder.sandbox.window import (
    AGENT_NAMES,
    IMAGE_ID,
    RESIDUAL_SHARES,
    SECRET_NAMES,
    SECRET_PREFIXES,
    SHARES,
    SHELL_AGENT_SOCKET,
    AgentRun,
    Run,
    agent_check,
    agent_sockets,
    check,
    clean_environment,
    dotenv_names,
    launchd_socket,
    runner,
    ssh_add_list,
    withheld,
)

__all__ = [
    "AGENT_NAMES",
    "IMAGE_ID",
    "RESIDUAL_SHARES",
    "SECRET_NAMES",
    "SECRET_PREFIXES",
    "SHARES",
    "AgentRun",
    "Run",
    "agent_check",
    "agent_sockets",
    "build_parser",
    "check",
    "clean_environment",
    "dotenv_names",
    "launchd_socket",
    "main",
    "runner",
    "ssh_add_list",
    "withheld",
]


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
    child_environment = dict(clean)
    if environment.get("SSH_AUTH_SOCK"):  # The path to ask, not an agent (window.py).
        child_environment[SHELL_AGENT_SOCKET] = environment["SSH_AUTH_SOCK"]
    child = subprocess.run(command, env=child_environment, check=False)
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
