"""``scratchcoder rl-bridge``: serve one Pod's RL rewards and dev verdicts from the Mac.

The command reads the Mac's full task files (the pool's train split for the kept
screening list, the dev split for the 200-task dev subset), refuses to start when a
secret could reach a child process (any ``R2_`` variable, the Runpod key, or a name a
``.env`` file defines; the files are found in the current directory and the checkout, and
at least one must exist unless ``--no-dotenv``), clears the SSH agent variables, and runs
``BridgeServer`` over the rehearsal's SSH options until interrupted, with the required
Mac-side bounds on a reward request and a dev check. It never opens the sealed test or
holdout split, whether named directly or reached through a link. See ``docs/rl.md`` and
``docs/walkthroughs/rl-bridge.md``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.tasks import Task, load_tasks
from scratchcoder.rl.bridge import DEV, REWARD, BridgeRequest
from scratchcoder.rl.bridge_mac import (
    AGENT_NAMES,
    BridgeServer,
    PodTransport,
    SSHTransport,
    environment_refusals,
    ssh_command,
    validate_remote_root,
)
from scratchcoder.rl.dev import dev_subset
from scratchcoder.sandbox.runner import ENDPOINT, IMAGE

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
"""The checkout that holds this module (``src/scratchcoder/rl/`` is three levels down)."""
SEALED_NAMES = frozenset({"test.jsonl", "holdout.jsonl"})
"""Task files the bridge never opens: the pool's sealed test split and its holdouts."""


class BridgeEnvironmentError(RuntimeError):
    """The bridge would start with a secret in its environment, or cannot check for one."""


def resolve_dotenv(
    explicit: Path | None,
    *,
    disabled: bool = False,
    cwd: Path | None = None,
    root: Path | None = None,
) -> list[Path] | None:
    """The ``.env`` files whose variable names the bridge refuses to run with.

    An explicit path must exist. Without one the files named ``.env`` in the current
    directory and in the repository root are used, and at least one must exist: a check that
    silently finds nothing (the command started from another directory) would let a ``.env``
    name through to the Docker children. ``disabled`` (``--no-dotenv``) says there is none
    and returns ``None``."""
    if disabled:
        if explicit is not None:
            raise BridgeEnvironmentError("--dotenv and --no-dotenv exclude each other")
        return None
    if explicit is not None:
        if not explicit.is_file():
            raise BridgeEnvironmentError(f"the --dotenv file {explicit} does not exist")
        return [explicit]
    candidates = [(cwd or Path.cwd()) / ".env", (root or REPOSITORY_ROOT) / ".env"]
    found: list[Path] = []
    for path in candidates:
        if path.is_file() and path.resolve() not in {item.resolve() for item in found}:
            found.append(path)
    if not found:
        raise BridgeEnvironmentError(
            "no .env file found in the current directory or the repository root, so its "
            "variable names cannot be checked: pass --dotenv PATH, or --no-dotenv if there is none"
        )
    return found


def _refuse_sealed(path: Path) -> Path:
    """Refuse a sealed split by the path's own name and by the name it resolves to, so a
    symlink called ``dev.jsonl`` that points at ``test.jsonl`` is refused before it is opened."""
    names = {path.name, path.resolve().name}
    if names & SEALED_NAMES:
        raise ValueError(f"the bridge never opens a sealed split file ({path.name})")
    return path


def read_kept(path: Path) -> list[str]:
    """The kept task IDs of an RL screening record (``rl-screening-v1``, outcome ``kept``)."""
    value = json.loads(path.read_text(encoding="utf-8"))
    kept = value.get("kept") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("format") != "rl-screening-v1"
        or value.get("outcome") != "kept"
        or not isinstance(kept, list)
        or not kept
        or any(not isinstance(item, str) or not item for item in kept)
        or len(set(kept)) != len(kept)
    ):
        raise ValueError(f"{path} is not a kept RL screening record")
    return list(kept)


def load_phase_tasks(
    train_tasks: Path, kept: Path, dev_tasks: Path
) -> tuple[list[Task], list[Task]]:
    """The full records of the kept screening list and of the 200-task dev subset."""
    kept_ids = read_kept(kept)
    wanted = set(kept_ids)
    found = {task.id: task for task in load_tasks(_refuse_sealed(train_tasks)) if task.id in wanted}
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"{len(missing)} kept tasks are not in {train_tasks}, e.g. {missing[0]}")
    rewards = [found[identifier] for identifier in kept_ids]
    if any(task.split != "train" for task in rewards):
        raise ValueError("every kept RL task must be a train-split task")
    dev = dev_subset(list(load_tasks(_refuse_sealed(dev_tasks))))
    return rewards, dev


Run = Callable[..., subprocess.CompletedProcess[bytes]]

IMAGE_CHECK_RETRIES = 3
"""Retries of ``docker image inspect`` after the first try (4 tries, waiting 1, 2 and 4 s)."""


def image_check(
    expected_id: str,
    *,
    run: Run | None = None,
    retries: int = IMAGE_CHECK_RETRIES,
    backoff_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[BridgeRequest], None]:
    """A ``before_scoring`` check: the pinned sandbox image's local ID is ``expected_id``.

    A busy Docker Desktop can fail or stall one ``docker image inspect``; that says nothing
    about the image, so a non-zero exit, a timeout or a failure to start is retried
    ``retries`` times with doubling backoff. Only an inspect that succeeds and names another
    image, or one that keeps failing, raises (the server then stores a ``host_failure``)."""
    if type(retries) is not int or retries < 0:
        raise ValueError("retries must be a nonnegative integer")
    runner: Run = run or subprocess.run
    command = ["docker", "--host", ENDPOINT, "image", "inspect", "--format", "{{.Id}}", IMAGE]

    def check(request: BridgeRequest) -> None:
        problem = ""
        for attempt in range(retries + 1):
            try:
                result = runner(command, capture_output=True, timeout=30, check=False)
            except (subprocess.TimeoutExpired, OSError) as error:
                problem = f"docker image inspect did not finish ({type(error).__name__})"
            else:
                if result.returncode == 0:
                    found = (result.stdout or b"").decode("utf-8", errors="replace").strip()
                    if found != expected_id:
                        raise RuntimeError(
                            f"the pinned sandbox image is not {expected_id} (found {found!r})"
                        )
                    return
                problem = f"docker image inspect exited {result.returncode}"
            if attempt < retries:
                sleep(backoff_seconds * 2.0**attempt)
        raise RuntimeError(f"the pinned sandbox image could not be checked: {problem}")

    return check


def serve_command(
    *,
    host: str,
    port: int,
    identity_file: Path,
    known_hosts: Path,
    remote_root: str,
    train_tasks: Path,
    kept: Path,
    dev_tasks: Path,
    pod_reward_sha256: str,
    pod_dev_sha256: str,
    state: Path,
    workers: int = 8,
    reward_timeout: float,
    dev_timeout: float,
    poll_seconds: float = 2.0,
    ssh_timeout: float = 120.0,
    ssh_retries: int = 4,
    dotenv: Path | Sequence[Path] | None = None,
    image_id: str | None = None,
    max_polls: int | None = None,
    environ: MutableMapping[str, str] = os.environ,
    transport: PodTransport | None = None,
) -> int:
    """Serve one Pod until interrupted (or for ``max_polls`` polls); 0 on a clean stop.

    ``reward_timeout`` and ``dev_timeout`` (seconds, required) are the Mac's own bounds on
    one request. Set each to the Pod's timeout for that kind less the bridge's allowance
    (``docs/rl.md``): the server scores one request at a time, so a request the Pod has
    already withdrawn must not keep the next one waiting. ``dotenv`` lists the ``.env``
    files to check (``resolve_dotenv`` finds them for the command line; ``None`` skips the
    check and says there is none)."""
    for name, bound in (("reward_timeout", reward_timeout), ("dev_timeout", dev_timeout)):
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise ValueError(f"{name} must be a number of seconds")
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError(f"{name} must be positive and finite")
    validate_remote_root(remote_root)
    try:
        refused = environment_refusals(environ, dotenv)
    except FileNotFoundError as error:
        raise BridgeEnvironmentError(f"the .env file {error.filename} does not exist") from error
    if refused:
        raise BridgeEnvironmentError(
            "unset these variables first (start the bridge in a terminal that has not "
            "sourced .env): " + ", ".join(refused)
        )
    for name in AGENT_NAMES:
        environ.pop(name, None)  # the Docker and ssh children get no SSH agent
    rewards, dev = load_phase_tasks(train_tasks, kept, dev_tasks)
    inputs: dict[str, Any] = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in (("train_tasks", train_tasks), ("kept", kept), ("dev_tasks", dev_tasks))
    }
    inputs["counts"] = {REWARD: len(rewards), DEV: len(dev)}
    inputs["image_id"] = image_id
    if transport is None:
        transport = SSHTransport(
            ssh_command(host, port, identity_file, known_hosts), remote_root, timeout=ssh_timeout
        )
    with BridgeServer(
        transport,
        reward_tasks=rewards,
        dev_tasks=dev,
        pod_task_sha256={REWARD: pod_reward_sha256, DEV: pod_dev_sha256},
        state=state,
        before_scoring=None if image_id is None else image_check(image_id),
        workers=workers,
        reward_timeout=reward_timeout,
        dev_timeout=dev_timeout,
        transport_retries=ssh_retries,
        inputs=inputs,
    ) as server:
        sys.stderr.write(
            f"rl-bridge: serving {len(rewards)} reward and {len(dev)} dev tasks; "
            f"progress in {state / 'progress.json'}\n"
        )
        try:
            server.serve(poll_seconds=poll_seconds, max_polls=max_polls)
        except KeyboardInterrupt:
            server.progress.update(state="stopped", current=None)
            return 0
    return 0
