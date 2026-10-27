"""End-to-end tests of the RL reward bridge: the Pod and the Mac in separate threads.

- fakes on both sides, over the local directory pair and over the remote program run by a
  local interpreter in place of ``ssh``;
- the real ``LocalRewardSource`` and evaluation runner in Docker
  (``SCRATCHCODER_SANDBOX_TESTS=1``);
- ``run_rl`` for 2 updates and one dev check, and the more-SFT control, through the
  bridge on the CPU, each compared with the same run scored locally;
- the ``rl-bridge`` command's checks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
from bridge_fixtures import (
    DEV_SHA,
    DEV_TASKS,
    RL_SHA,
    TRAIN,
    CountingDev,
    CountingRewards,
    attempts,
    mac_server,
    pod_client,
)

from scratchcoder.cli import main
from scratchcoder.common.config import ModelConfig
from scratchcoder.data.chat import PYTHON_END, PYTHON_START
from scratchcoder.engine.generation import Completion
from scratchcoder.eval.tasks import Task, save_tasks
from scratchcoder.model.gpt import GPT
from scratchcoder.rl import dev as dev_module
from scratchcoder.rl import loop
from scratchcoder.rl.bridge import (
    BridgeClient,
    BridgeDevEvaluator,
    BridgeDevExecutor,
    BridgeRewardSource,
)
from scratchcoder.rl.bridge_command import (
    BridgeEnvironmentError,
    image_check,
    load_phase_tasks,
    serve_command,
)
from scratchcoder.rl.bridge_mac import (
    REMOTE_SCRIPT,
    BridgeServer,
    DirectoryTransport,
    SSHTransport,
    ssh_child_environment,
)
from scratchcoder.rl.checkpoint import save_weights
from scratchcoder.rl.control import ControlConfig, ControlResult, run_more_sft
from scratchcoder.rl.dev import DevCheck, PoolDevEvaluator
from scratchcoder.rl.fakes import FakeRewardSource
from scratchcoder.rl.loop import CheckRecord, RLConfig, run_rl
from scratchcoder.rl.rewards import NO_PROGRAM, Attempt, AttemptReward
from scratchcoder.rl.rollout import Rollout
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.train import trainer as trainer_module

TOKENIZER = ByteBPE()


@contextmanager
def mac_thread(server: BridgeServer, poll: float = 0.01) -> Iterator[threading.Event]:
    """Serve in a thread until the block ends; re-raise anything the server raised."""
    stop = threading.Event()
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            server.serve(poll_seconds=poll, should_stop=stop.is_set, progress_seconds=0.05)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=serve, name="mac-bridge", daemon=True)
    thread.start()
    try:
        yield stop
    finally:
        stop.set()
        thread.join(timeout=30)
        server.close()
    assert not thread.is_alive()
    if errors:
        raise errors[0]


def pod_round_trips(client: BridgeClient) -> None:
    """Three reward updates and two dev checks; each must equal the local fake scoring."""
    source = BridgeRewardSource(client, task_file_sha256=RL_SHA, timeout=60)
    for update in (1, 2, 3):
        batch = attempts(("t0", "t1", "t2"), group=8)
        assert source.rewards(update, batch) == FakeRewardSource().rewards(update, batch)
    executor = BridgeDevExecutor(client, task_file_sha256=DEV_SHA, timeout=60)
    pairs = [(task, "x" * (index + 1)) for index, task in enumerate(DEV_TASKS)]
    for check in (0, 12):
        with executor.labelled(check):
            assert executor(pairs) == [len(code) % 2 == 0 for _, code in pairs]


def test_end_to_end_over_the_local_directory_pair(tmp_path: Path) -> None:
    pod = tmp_path / "pod"
    rewards, dev = CountingRewards(), CountingDev()
    with pod_client(pod, poll_seconds=0.01) as client:
        server = mac_server(pod, tmp_path / "mac", rewards=rewards, dev=dev)
        with mac_thread(server):
            pod_round_trips(client)
    assert [update for update, _ in rewards.calls] == [1, 2, 3] and len(dev.calls) == 2
    assert len(list((tmp_path / "mac" / "replies").iterdir())) == 5
    assert len(list((pod / "done").iterdir())) == 10
    progress = json.loads((tmp_path / "mac" / "progress.json").read_bytes())
    assert progress["answered"] == 5 and progress["state"] == "stopped"
    assert progress["last_answered"]["index"] == 12


def test_end_to_end_over_the_remote_program(tmp_path: Path) -> None:
    """The same exchange through ``SSHTransport`` running the exact remote program."""
    pod = tmp_path / "pod"
    with pod_client(pod, poll_seconds=0.01) as client:
        transport = SSHTransport(
            [sys.executable, "-I", "-c", REMOTE_SCRIPT],
            str(pod.resolve()),
            environment=ssh_child_environment(),
        )
        server = mac_server(pod, tmp_path / "mac", transport=transport)
        with mac_thread(server, poll=0.02):
            pod_round_trips(client)


def test_a_mac_that_never_answers_ends_in_reward_timeout(tmp_path: Path) -> None:
    from scratchcoder.rl.rewards import RewardTimeout

    with pod_client(tmp_path / "pod", poll_seconds=0.01) as client:
        source = BridgeRewardSource(client, task_file_sha256=RL_SHA, timeout=0.2)
        seen: list[Any] = []
        watcher = threading.Timer(0.1, lambda: seen.append(client.heartbeat()))
        watcher.start()
        with pytest.raises(RewardTimeout):
            source.rewards(1, attempts())
        watcher.join()
    assert seen[0]["status"] == "awaiting_rewards" and seen[0]["index"] == 1
    assert 0.0 < seen[0]["age_seconds"] < 0.2


# ------------------------------------------------------------------- Docker-backed

integration = pytest.mark.skipif(
    os.environ.get("SCRATCHCODER_SANDBOX_TESTS") != "1",
    reason="requires explicitly enabled disposable Docker worker",
)


def io_task(identifier: str, split: str) -> Task:
    cases = [("1 2\n", "3\n"), ("10 5\n", "15\n"), ("-4 4\n", "0\n")]
    return Task(
        id=identifier,
        prompt="Echo the sum.",
        public_tests=json.dumps([{"input": cases[0][0], "output": cases[0][1]}]),
        hidden_tests=json.dumps([{"input": given, "output": wanted} for given, wanted in cases]),
        source="synthetic",
        license="fixture",
        written_by="human",
        split=split,
        mode="io",
    )


def fn_task(identifier: str, split: str) -> Task:
    return Task(
        id=identifier,
        prompt="Add one.",
        public_tests="assert f(0) == 1",
        hidden_tests="assert f(1) == 2\nassert f(5) == 6\nassert f(-3) == -2",
        source="synthetic",
        license="fixture",
        written_by="human",
        split=split,
    )


SOLUTIONS = {
    "fn": "def f(x):\n    return x + 1",
    "io": "a, b = map(int, input().split())\nprint(a + b)",
}
WRONG = {"fn": "def f(x):\n    return x - 1", "io": "print(input())"}


def scratchcoder_containers() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=scratchcoder", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return result.stdout.split()


@integration
def test_end_to_end_with_the_real_reward_channel_and_runner(tmp_path: Path) -> None:
    train = [fn_task("rl/fn", "train"), io_task("rl/io", "train")]
    dev = [fn_task("dev/fn", "dev"), io_task("dev/io", "dev")]
    pod = tmp_path / "pod"
    server = BridgeServer(
        DirectoryTransport(pod),
        reward_tasks=train,
        dev_tasks=dev,
        pod_task_sha256={"reward": RL_SHA, "dev": DEV_SHA},
        state=tmp_path / "mac",
        workers=4,
        reward_timeout=600,
    )
    with pod_client(pod, poll_seconds=0.05) as client:
        with mac_thread(server, poll=0.05):
            source = BridgeRewardSource(client, task_file_sha256=RL_SHA, timeout=600)
            batch = [
                Attempt(task.id, sample, code, 1)
                for task in train
                for sample, code in enumerate(
                    (
                        SOLUTIONS[task.mode if task.mode == "io" else "fn"],
                        WRONG[task.mode if task.mode == "io" else "fn"],
                        None,
                    )
                )
            ]
            replies = source.rewards(1, batch)
            executor = BridgeDevExecutor(client, task_file_sha256=DEV_SHA, timeout=600)
            kind = {"function": "fn", "io": "io"}
            pairs = [
                (task, code)
                for task in dev
                for code in (SOLUTIONS[kind[task.mode]], WRONG[kind[task.mode]])
            ]
            with executor.labelled(0):
                verdicts = executor(pairs)
    for position in range(len(train)):
        reference, wrong, missing = replies[3 * position : 3 * position + 3]
        assert reference.status == "completed" and reference.reward == 1.0
        assert wrong.reward < 1.0
        assert missing == AttemptReward(0.0, NO_PROGRAM)
    assert verdicts == [True, False, True, False]
    assert scratchcoder_containers() == []


# ------------------------------------------------------------- run_rl on the CPU


def scripted_rollouts(
    model: GPT,
    tasks: Sequence[Task],
    prompts: Sequence[Sequence[int]],
    decoder: Any,
    *,
    samples: int,
    seed_of: Callable[[Task, int], int],
    **options: Any,
) -> list[Rollout]:
    """Sample s writes a program of s + 1 letters (as in the RL loop's tests)."""
    rows = []
    for item, prompt in zip(tasks, prompts, strict=True):
        for sample in range(samples):
            body = (PYTHON_START, *([ord("a")] * (sample + 1)), PYTHON_END)
            code = "a" * (sample + 1)
            rows.append(
                Rollout(
                    Attempt(item.id, sample, code, len(body)),
                    tuple(prompt),
                    body,
                    "stop",
                    seed_of(item, sample),
                )
            )
    return rows


def dev_completions(model: GPT, prompts: Sequence[Sequence[int]], **options: Any) -> Any:
    """Greedy writes ``xx``; a sample writes one or two letters by its seed's parity."""
    seeds = options.get("seeds") or [0] * len(prompts)
    rows = []
    for seed in seeds:
        letters = 2 if options.get("seeds") is None or seed % 2 else 1
        rows.append(Completion([PYTHON_START, *([ord("x")] * letters), PYTHON_END], "stop", 1))
    return rows


def rl_config() -> RLConfig:
    return RLConfig(
        updates=2,
        tasks_per_update=2,
        group_size=4,
        max_new_tokens=8,
        micro_batch=3,
        generation_batch=16,
        warmup_updates=1,
        learning_rate=1e-3,
        dev_checks=(0,),
        check_at_stop=False,
    )


def tiny_model() -> GPT:
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=64))


def comparable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key != "seconds"} for row in rows]


def test_run_rl_through_the_bridge_matches_local_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop, "generate_rollouts", scripted_rollouts)
    monkeypatch.setattr(dev_module, "generate_many", dev_completions)
    common: dict[str, Any] = {
        "encoder": TOKENIZER,
        "decoder": TOKENIZER,
        "run_id": "rl-bridge-test",
        "tokenizer_hash": "authored",
        "config": rl_config(),
    }
    local = run_rl(
        tiny_model(),
        TRAIN,
        rewards=FakeRewardSource(),
        evaluator=PoolDevEvaluator(DEV_TASKS, TOKENIZER, TOKENIZER, CountingDev(), batch_size=8),
        output=tmp_path / "local",
        **common,
    )
    pod = tmp_path / "pod"
    rewards, dev = CountingRewards(), CountingDev()
    with pod_client(pod, poll_seconds=0.01) as client:
        executor = BridgeDevExecutor(client, task_file_sha256=DEV_SHA, timeout=60)
        server = mac_server(pod, tmp_path / "mac", rewards=rewards, dev=dev)
        with mac_thread(server):
            bridged = run_rl(
                tiny_model(),
                TRAIN,
                rewards=BridgeRewardSource(client, task_file_sha256=RL_SHA, timeout=60),
                evaluator=BridgeDevEvaluator(
                    DEV_TASKS, TOKENIZER, TOKENIZER, executor, batch_size=8
                ),
                output=tmp_path / "bridged",
                **common,
            )
    assert bridged.stop_reason == local.stop_reason == "completed"
    assert bridged.updates == 2 and bridged.optimizer_steps == local.optimizer_steps
    assert [check.dev.check for check in bridged.checks] == [0]
    assert comparable(bridged.monitoring) == comparable(local.monitoring)
    assert bridged.checks[0].dev.per_task == local.checks[0].dev.per_task
    assert bridged.checks[0].dev.greedy_pass_at_1 == 1.0  # greedy writes "xx": even, passes
    assert 0 < bridged.checks[0].dev.mean_pass_n8 < 1
    assert bridged.checks[0].sha256 == local.checks[0].sha256  # same weights at check 0
    # Two reward updates and one dev check crossed the bridge, each once.
    assert [update for update, _ in rewards.calls] == [1, 2]
    assert len(dev.calls) == 1 and len(dev.calls[0]) == len(DEV_TASKS) * 9
    stored = sorted(path.name for path in (tmp_path / "mac" / "replies").iterdir())
    assert [name.split("-", 2)[2] for name in stored] == [
        "dev-000000.json",
        "reward-000001.json",
        "reward-000002.json",
    ]
    assert len(list((pod / "done").iterdir())) == 6 and client.waiting() is None


def test_the_more_sft_control_takes_its_dev_checks_through_the_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dev_module, "generate_many", dev_completions)
    torch.manual_seed(0)
    parent = tmp_path / "parent.pt"
    model = GPT(ModelConfig(vocab_size=272, width=32, layers=2, heads=2, context=96))
    save_weights(parent, model, tokenizer_hash=TOKENIZER.sha256, run_id="sft", step=0)
    start = CheckRecord(DevCheck(0, 0.1, 0.05, 0.2, 0.1, 200, 0.0, 0.0), 0, str(parent), "0" * 64)

    def control(evaluator: Any, output: Path) -> ControlResult:
        # Each update reads the trainer's clock at its start and end: 1 s of busy time each.
        readings = iter([value for step in range(20) for value in (101.0 * step, 101.0 * step + 1)])
        monkeypatch.setattr(trainer_module, "synchronized_time", lambda device: next(readings))
        return run_more_sft(
            parent=parent,
            tasks=TRAIN,
            encoder=TOKENIZER,
            tokenizer_hash=TOKENIZER.sha256,
            budget_seconds=3.0,
            start=start,
            evaluator=evaluator,
            output=output,
            run_id="more-sft",
            config=ControlConfig(batch_size=2, accumulation_steps=2, warmup_updates=1),
        )

    local = control(
        PoolDevEvaluator(DEV_TASKS, TOKENIZER, TOKENIZER, CountingDev(), batch_size=8),
        tmp_path / "local",
    )
    pod = tmp_path / "pod"
    dev = CountingDev()
    with pod_client(pod, poll_seconds=0.01) as client:
        executor = BridgeDevExecutor(client, task_file_sha256=DEV_SHA, timeout=60)
        with mac_thread(mac_server(pod, tmp_path / "mac", dev=dev)):
            bridged = control(
                BridgeDevEvaluator(DEV_TASKS, TOKENIZER, TOKENIZER, executor, batch_size=8),
                tmp_path / "bridged",
            )
    steps = [check.dev.check for check in bridged.checks]
    assert steps == [check.dev.check for check in local.checks] and steps[0] == 0
    assert len(steps) >= 3  # the start, the pass-1 check and the budget-end check
    assert bridged.updates == local.updates == 3
    for ours, theirs in zip(bridged.checks[1:], local.checks[1:], strict=True):
        assert ours.dev.per_task == theirs.dev.per_task and ours.sha256 == theirs.sha256
    # The start check is RL's, reused; every later check crossed the bridge once, labelled
    # with the control's optimizer step.
    stored = sorted(path.name for path in (tmp_path / "mac" / "replies").iterdir())
    assert [name.split("-", 2)[2] for name in stored] == [
        f"dev-{step:06d}.json" for step in steps[1:]
    ]
    assert len(dev.calls) == len(steps) - 1


# --------------------------------------------------------------------- the command


def dev_pool(count: int = 210) -> list[Task]:
    return [Task(**{**DEV_TASKS[0].__dict__, "id": f"pool-dev-{index}"}) for index in range(count)]


def command_files(tmp_path: Path) -> dict[str, Any]:
    files = {
        "train_tasks": tmp_path / "pool" / "train.jsonl",
        "dev_tasks": tmp_path / "pool" / "dev.jsonl",
        "kept": tmp_path / "screening.json",
    }
    save_tasks(files["train_tasks"], TRAIN)
    save_tasks(files["dev_tasks"], dev_pool())
    record = {"format": "rl-screening-v1", "outcome": "kept", "kept": ["t3", "t1"]}
    files["kept"].write_text(json.dumps(record))
    return files


def test_the_command_serves_with_the_kept_list_and_the_dev_subset(tmp_path: Path) -> None:
    files = command_files(tmp_path)
    rewards, dev = load_phase_tasks(files["train_tasks"], files["kept"], files["dev_tasks"])
    assert [task.id for task in rewards] == ["t3", "t1"] and len(dev) == 200
    pod = tmp_path / "pod"
    for box in ("outbox", "inbox"):
        (pod / box).mkdir(parents=True)
    environ = {"PATH": "/usr/bin", "SSH_AUTH_SOCK": "/tmp/agent"}
    status = serve_command(
        host="unused",
        port=22,
        identity_file=tmp_path / "unused",
        known_hosts=tmp_path / "unused",
        remote_root="/unused",
        pod_reward_sha256=RL_SHA,
        pod_dev_sha256=DEV_SHA,
        state=tmp_path / "mac",
        poll_seconds=0.0,
        dotenv=None,
        max_polls=2,
        environ=environ,
        transport=DirectoryTransport(pod),
        **files,
    )
    assert status == 0 and environ == {"PATH": "/usr/bin"}  # the agent socket is cleared
    progress = json.loads((tmp_path / "mac" / "progress.json").read_bytes())
    assert progress["state"] == "stopped" and progress["polls"] == 2
    assert progress["allowed_tasks"] == {"reward": 2, "dev": 200}
    assert progress["inputs"]["counts"] == {"reward": 2, "dev": 200}
    assert len(progress["inputs"]["dev_tasks"]["sha256"]) == 64


def test_the_command_refuses_secrets_sealed_files_and_bad_records(tmp_path: Path) -> None:
    files = command_files(tmp_path)
    options: dict[str, Any] = {
        "host": "unused",
        "port": 22,
        "identity_file": tmp_path / "unused",
        "known_hosts": tmp_path / "unused",
        "remote_root": "/unused",
        "pod_reward_sha256": RL_SHA,
        "pod_dev_sha256": DEV_SHA,
        "state": tmp_path / "mac",
        "max_polls": 1,
        "transport": DirectoryTransport(tmp_path / "pod"),
    }
    dotenv = tmp_path / ".env"
    dotenv.write_text("MY_TOKEN=secret-value\n")
    for environ in ({"R2_SECRET_ACCESS_KEY": "x"}, {"RUNPOD_API_KEY": "x"}, {"MY_TOKEN": "y"}):
        with pytest.raises(BridgeEnvironmentError) as caught:
            serve_command(**options, **files, environ=dict(environ), dotenv=dotenv)
        assert "secret-value" not in str(caught.value) and "x" not in str(caught.value)[-3:]
    sealed = {**files, "dev_tasks": tmp_path / "pool" / "test.jsonl"}
    with pytest.raises(ValueError, match="sealed"):
        serve_command(**options, **sealed, environ={}, dotenv=None)
    assert not (tmp_path / "pool" / "test.jsonl").exists()  # refused by name, never opened
    files["kept"].write_text(json.dumps({"format": "rl-screening-v1", "outcome": "floor"}))
    with pytest.raises(ValueError, match="kept RL screening record"):
        serve_command(**options, **files, environ={}, dotenv=None)
    record = {"format": "rl-screening-v1", "outcome": "kept", "kept": ["t1", "missing"]}
    files["kept"].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="not in"):
        serve_command(**options, **files, environ={}, dotenv=None)


def test_the_cli_exits_2_when_a_secret_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "do-not-print")
    arguments = [
        "rl-bridge",
        "--host=root@203.0.113.7",
        "--port=22",
        f"--identity-file={tmp_path / 'id'}",
        f"--known-hosts={tmp_path / 'hosts'}",
        "--remote-root=/workspace/bridge",
        f"--train-tasks={tmp_path / 'train.jsonl'}",
        f"--kept={tmp_path / 'kept.json'}",
        f"--dev-tasks={tmp_path / 'dev.jsonl'}",
        f"--pod-reward-sha256={RL_SHA}",
        f"--pod-dev-sha256={DEV_SHA}",
        f"--state={tmp_path / 'mac'}",
        f"--dotenv={tmp_path / 'missing.env'}",
    ]
    assert main(arguments) == 2
    error = capsys.readouterr().err
    assert "R2_ACCESS_KEY_ID" in error and "do-not-print" not in error
    assert not (tmp_path / "mac").exists()


def test_the_image_check_compares_the_local_image_id() -> None:
    def run(stdout: bytes, code: int = 0) -> Any:
        def runner(command: list[str], **options: Any) -> subprocess.CompletedProcess[bytes]:
            assert command[-1].startswith("python@sha256:") and "inspect" in command
            return subprocess.CompletedProcess(command, code, stdout, b"")

        return runner

    expected = "sha256:" + "d" * 64
    request: Any = None
    image_check(expected, run=run(expected.encode() + b"\n"))(request)
    for stdout, code in ((b"sha256:" + b"e" * 64, 0), (expected.encode(), 1)):
        with pytest.raises(RuntimeError, match="pinned sandbox image"):
            image_check(expected, run=run(stdout, code))(request)
