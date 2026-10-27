"""``scratchcoder rl-bridge``: serve one Pod's RL rewards and dev verdicts from the Mac.

The command reads the Mac's full task files (the pool's train split for the kept
screening list, the dev split for the 200-task dev subset), refuses to start when a
secret could reach a child process (any ``R2_`` variable, the Runpod key, or a name the
``.env`` file defines), clears the SSH agent variables, and runs ``BridgeServer`` over the
rehearsal's SSH options until interrupted. It never opens the sealed test or holdout split.
See ``docs/rl.md`` and ``docs/walkthroughs/rl-bridge.md``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, MutableMapping
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
)
from scratchcoder.rl.dev import dev_subset
from scratchcoder.sandbox.runner import ENDPOINT, IMAGE

SEALED_NAMES = frozenset({"test.jsonl", "holdout.jsonl"})
"""Task files the bridge never opens: the pool's sealed test split and its holdouts."""


class BridgeEnvironmentError(RuntimeError):
    """The bridge would start with a secret in its environment."""


def _refuse_sealed(path: Path) -> Path:
    if path.name in SEALED_NAMES:
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


def image_check(expected_id: str, *, run: Run | None = None) -> Callable[[BridgeRequest], None]:
    """A ``before_scoring`` check: the pinned sandbox image's local ID is ``expected_id``."""
    runner: Run = run or subprocess.run
    command = ["docker", "--host", ENDPOINT, "image", "inspect", "--format", "{{.Id}}", IMAGE]

    def check(request: BridgeRequest) -> None:
        result = runner(command, capture_output=True, timeout=30, check=False)
        found = (result.stdout or b"").decode("utf-8", errors="replace").strip()
        if result.returncode != 0 or found != expected_id:
            raise RuntimeError(f"the pinned sandbox image is not {expected_id} (found {found!r})")

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
    reward_timeout: float | None = None,
    poll_seconds: float = 2.0,
    ssh_timeout: float = 120.0,
    ssh_retries: int = 4,
    dotenv: Path | None = Path(".env"),
    image_id: str | None = None,
    max_polls: int | None = None,
    environ: MutableMapping[str, str] = os.environ,
    transport: PodTransport | None = None,
) -> int:
    """Serve one Pod until interrupted (or for ``max_polls`` polls); 0 on a clean stop."""
    refused = environment_refusals(environ, dotenv)
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
