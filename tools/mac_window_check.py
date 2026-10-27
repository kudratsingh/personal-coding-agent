"""Checks for a Mac execution window: model-written code runs in Docker Desktop on the Mac.

For the rehearsal, pool completions and RL screening attempts execute in the Mac's Docker
sandbox, on the laptop that holds ``.env``, the Runpod key, ``gh`` and SSH credentials. An
escape from a container reaches Docker Desktop's Linux VM, its file shares, the engine and
its image store, and possibly the SSH-agent socket. Before every batch the owner's runbook
therefore checks, with this tool:

``check`` (Mac)
    1. **No credentials in the environment:** none of the variables ``.env`` defines (given
       with ``--dotenv``, names only) and none of the known secret names is set, and the SSH
       agent is cleared (``SSH_AUTH_SOCK`` and ``SSH_AGENT_PID`` unset).
    2. **The pinned image:** the sandbox image (``scratchcoder.sandbox.runner.IMAGE``) is
       present locally (it is never pulled) with the image ID recorded at the window's start
       (``--image-id``), so a tampered image store is caught before a batch runs.
    3. **No file sharing:** a container from the pinned image asked to mount ``/Users``
       (``docker run -v /Users:/x``) must fail. With Docker Desktop's file sharing emptied the
       mount is refused; if it succeeds, the window does not start. The container is removed
       whatever happens, so no ``scratchcoder-*`` container is left behind.
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

Run = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def runner(environment: Mapping[str, str]) -> Run:
    """``docker`` calls in ``environment`` (the cleaned one), output captured, bounded."""

    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=120, check=False, env=dict(environment)
        )

    return run


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
) -> dict[str, Any]:
    """Run the three checks; return a record whose ``status`` is ``passed`` or ``failed``."""
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
    inspected = run([docker, "--host", ENDPOINT, "image", "inspect", "--format", "{{.Id}}", IMAGE])
    found = inspected.stdout.strip() if inspected.returncode == 0 else None
    checks["image"] = {
        "passed": found == image_id,
        "image": IMAGE,
        "expected_id": image_id,
        "found_id": found,
        "rule": "the pinned image is present locally with the recorded image ID; never pulled",
    }
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
                "/Users:/x:ro",
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
    checks["file_sharing"] = {
        "passed": mounted.returncode != 0 and "mounted" not in mounted.stdout,
        "exit_code": mounted.returncode,
        "rule": "a container from the pinned image cannot mount /Users",
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
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    environment = dict(os.environ if environ is None else environ)
    dotenv = dotenv_names(args.dotenv)
    clean = clean_environment(environment, dotenv)
    docker = runner(clean) if run is None else run  # Docker never sees a credential either.
    try:
        if args.command == "check":
            record = check(image_id=args.image_id, environ=environment, dotenv=dotenv, run=docker)
        else:
            command = list(args.argv)
            if command[:1] == ["--"]:
                command = command[1:]
            if not command:
                parser.error("exec needs a command after --")
            record = check(image_id=args.image_id, environ=clean, dotenv=dotenv, run=docker)
            record["command"] = command
    except ValueError as error:
        parser.error(str(error))
    if args.receipt is not None:
        atomic_json(args.receipt, record)
    print(json.dumps({"status": record["status"], "checks": record["checks"]}, sort_keys=True))
    if record["status"] != "passed":
        print("mac-window-check: a check failed; nothing was executed", file=sys.stderr)
        return 3
    if args.command == "check":
        return 0
    child = subprocess.run(command, env=clean, check=False)
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
