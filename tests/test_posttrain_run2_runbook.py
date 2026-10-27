"""Every tool command in run 2's runbook parses with its real command line, and the runbook's
bounds are the specification's.

The method is run 1's runbook test (``test_posttrain_runbook.py``, itself the rehearsal
runbook's): shell variables get concrete values, no tool runs, each tool command is parsed by
its module's ``main`` with ``ArgumentParser.parse_args`` intercepted, and every shell block
must parse in bash and in an interactive zsh and hold no comment. The unseal record's script
runs on authored reports.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import test_posttrain_runbook as base
from test_rehearsal_runbook import comment_lines

from tools import posttrain_approved, posttrain_ops
from tools.posttrain_run import BRIDGE_LOST_STOP, WAIT_MARGIN_STOP, Specification, launch_fit

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "POSTTRAIN_RUN2_RUNBOOK.md"
ORIGIN = base.ORIGIN
BOUNDS = posttrain_ops.bridge_timeouts(attempts_per_request=256, dev_programs=1_800)
ESTIMATE = 3.53
"""The runbook's example estimate: the run 2 projection's total at the planning speeds."""
LIMITS = posttrain_ops.ceiling_limits(5.80, 0.60, gate_estimate_usd=ESTIMATE)
VALUES = {
    **base.VALUES,
    "RUN": "2",
    "CTRL": "runs/posttrain-controller/2/podpost02",
    "KNOWN_HOSTS": "/Users/owner/.ssh/posttrain_2_known_hosts",
    "SPEC_TEMPLATE": "configs/rehearsal-a40-s-post-2.json",
    "SHARED": "runs/posttrain-shared-2",
    "SPEC": "runs/posttrain-shared-2/rehearsal-a40-s-post-2-gate.json",
    "BINDING": "runs/posttrain-shared-2/binding.json",
    "RUN1_SHARED": "runs/posttrain-shared",
    "RUN1_CTRL": "runs/posttrain-controller/1/podpost01",
    "RUN1_REPORT": "runs/posttrain-controller/1/podpost01/supervisor/final/report.json",
    "RL_EXPORT": "runs/posttrain-shared-2/rl-pod-export",
    "BRIDGE_STATE": "runs/posttrain-shared-2/bridge-state",
    "LEDGER": "runs/posttrain-shared/seal/ledger.jsonl",
    "ART": "runs/posttrain-shared-2/artifacts",
    "CEILING": "5.80",
    "COST_LIMIT": str(LIMITS.worker_cost_stop_usd),
    "WALL": str(LIMITS.wall_seconds),
    "TIMER": str(LIMITS.self_destruct_seconds),
    "WATCHDOG_COST": str(LIMITS.watchdog_cost_stop_usd),
    "REWARD_EXTENSION": str(BOUNDS.awaiting_reward_extension_seconds),
    "DEV_EXTENSION": str(BOUNDS.awaiting_dev_extension_seconds),
    "MAC_REWARD_TIMEOUT": str(BOUNDS.mac_reward_timeout_seconds),
    "MAC_DEV_TIMEOUT": str(BOUNDS.mac_dev_timeout_seconds),
    "POD_REWARD_SHA256": "1" * 64,
    "POD_DEV_SHA256": "2" * 64,
    "BASE_SHA256": "3" * 64,
    "SFT_SHA256": "4" * 64,
    "RL_SHA256": "5" * 64,
    "CONTROL_SHA256": "6" * 64,
    "POD_ID": "podpost02",
    "SELF_DESTRUCT_DEADLINE": str(ORIGIN + LIMITS.self_destruct_seconds),
    "LAUNCH": f"runs/posttrain-controller/2/launch-{ORIGIN}",
}
PLACEHOLDERS = {
    "<run 1 pod id>": "podpost01",
    "<the owner's ceiling in dollars>": "5.80",
    "<reviewed binding sha256>": base.BINDING_SHA256,
    "<the Pod's host>": "203.0.113.7",
    "<the Pod's port>": "40022",
    "<from step 4>": "5600.0",
}


@pytest.fixture(autouse=True)
def run2_runbook(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run 1's parsing helpers, pointed at run 2's runbook and values."""
    monkeypatch.setattr(base, "RUNBOOK", RUNBOOK)
    monkeypatch.setattr(base, "VALUES", VALUES)
    yield


def logical(pattern: str) -> list[str]:
    return [line for line in base.logical_lines() if pattern in line]


def options(argv: list[str]) -> dict[str, str]:
    found = {}
    for index, token in enumerate(argv):
        if (
            token.startswith("--")
            and index + 1 < len(argv)
            and not argv[index + 1].startswith("--")
        ):
            found[token] = argv[index + 1]
    return found


def test_every_tool_command_parses_with_its_real_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = base.commands()
    counts = Counter(module for module, _, _ in found)
    for module, minimum in {
        "tools.posttrain_run": 3,
        "tools.posttrain_ops": 11,
        "tools.posttrain_binding": 4,
        "tools.posttrain_approved": 1,
        "tools.pilot_readiness_receipt": 1,
        "tools.pilot_feasibility": 1,
        "tools.pilot_supervisor": 1,
        "tools.runpod_watchdog": 1,
        "tools.runpod_pod_identity": 4,
        "tools.mac_window_check": 6,
        "tools.benchmark_dispatch": 1,
        "tools.rehearsal_ops": 1,
        "tools.pod_progress": 1,
        "scratchcoder.cli": 7,
    }.items():
        assert counts[module] >= minimum, (
            f"expected {minimum} {module} commands, got {counts[module]}"
        )
    for module, argv, text in found:
        namespace = base.parse(module, argv, monkeypatch)
        assert "SECRET" not in text and "ACCESS_KEY" not in text, text
        if module == "tools.mac_window_check" and namespace.command == "exec":
            assert namespace.argv[:1] == ["--"] and namespace.argv[1:4] == ["uv", "run", "--locked"]


def test_the_runbooks_bounds_are_the_specifications() -> None:
    filled = posttrain_approved.run2_specification(
        {
            "ceiling_usd": 5.80,
            "gate_estimate_usd": ESTIMATE,
            "sft-weights": {
                "key": "posttrain-runs/1/p/sft-weights/checkpoint.pt",
                "size_bytes": 1,
                "sha256": "a" * 64,
            },
            **{
                name: {"size_bytes": 1, "sha256": "a" * 64}
                for name in ("rl-tasks", "rl-dev", "rl-facts", "pool-test")
            },
        }
    )
    derived = posttrain_ops.specification_bridge_timeouts(filled)
    assert derived == BOUNDS
    assert posttrain_ops.specification_limits(filled) == LIMITS
    text = RUNBOOK.read_text()
    assert isinstance(LIMITS, posttrain_ops.EstimateLimits) and LIMITS.limit_usd == 4.23
    assert (
        posttrain_ops.estimate_limit_usd(5.80, 4.84)
        == 5.80
        > posttrain_ops.estimate_limit_usd(5.80, 4.83)
    )
    assert posttrain_ops.ceiling_limits(5.80, 0.60).wall_seconds == 34_800
    for phrase in (
        "**$4.23**",
        "**$4.22**",
        "**25,380 s**",
        "**25,200 s**",
        "Only an estimate of $4.84 or more lets the $5.80 ceiling bind",
        "**900 s**",
        "**5,600 s**",
        "**9,492 s**",
        "**7,854.5 s**",
        "**11,746.5 s**",
        "7,914.5 s (reward), 11,806.5 s (dev)",
        "11,746.5 s, $1.96",
        "20,288 s",
        "about 4,912 s (1.4 hours) of the 25,200 s",
        "5.9 hours and $3.53",
        "cost limit 4.23 USD, wall 25380 s, timer 25200 s, watchdog cost stop 4.22 USD",
    ):
        assert phrase in text, phrase
    lifetime = LIMITS.self_destruct_seconds
    spec = Specification.from_dict(filled)
    needed = launch_fit(spec, remaining_seconds=lifetime)["planning_seconds_needed"]
    assert (round(needed), lifetime - round(needed)) == (20_288, 4_912)
    assert f"{BOUNDS.lost_bridge_seconds:,.0f} s" in text
    assert "| 1,397.5 s |" in text and "| 857 s |" in text
    assert BOUNDS.mac_check_seconds == 857 and BOUNDS.window_check_seconds == 730
    for reason in (BRIDGE_LOST_STOP, WAIT_MARGIN_STOP, "reward_wait", "reward_host_failure"):
        assert reason in text, reason
    assert "60,000 s of the pretraining rehearsal is never used" in text


def test_the_bridge_runs_in_the_window_with_the_derived_mac_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (bridge,) = base.by("scratchcoder.cli", "rl-bridge")
    values = options(bridge)
    spec = posttrain_approved.run2_specification()
    assert values["--reward-timeout"] == "5600.0" and values["--dev-timeout"] == "9492.0"
    for kind in ("reward", "dev"):
        assert (
            float(values[f"--{kind}-timeout"])
            == BOUNDS.timeout(kind) - BOUNDS.bridge_allowance_seconds - BOUNDS.mac_check_seconds
            == BOUNDS.mac_timeout(kind)
        )
    assert values["--remote-root"] == spec["bridge"]["root"] == "/workspace/posttrain-bridge"
    assert values["--image-id"] == VALUES["SANDBOX_IMAGE_ID"] and values["--dotenv"] == ".env"
    assert values["--kept"] == "runs/posttrain-shared/screening.json"
    assert values["--pod-reward-sha256"] == VALUES["POD_REWARD_SHA256"]
    namespace = base.parse("scratchcoder.cli", bridge, monkeypatch)
    assert namespace.workers == 8 and namespace.poll_seconds == spec["bridge"]["mac_poll_seconds"]
    assert namespace.ssh_timeout == spec["bridge"]["ssh_timeout_seconds"]
    assert namespace.ssh_retries == spec["bridge"]["ssh_retries"]
    (line,) = logical("scratchcoder rl-bridge")
    assert line.startswith("uv run --locked python -m tools.mac_window_check exec ")
    text = RUNBOOK.read_text()
    for phrase in (
        "Exit **2**",
        "**Exit 3**",
        "same `--state`",
        "`--no-dotenv`",
        "never source `.env`",
    ):
        assert phrase in text, phrase


def test_the_bridge_and_the_supervisor_start_before_the_watchdogs_idle_clock_runs_out() -> None:
    """The watchdog reads the heartbeat only through the supervisor's mirror, so from the
    receipt's last re-base the bridge, the worker and the supervisor start back to back; the
    owner's slow preparation of the bridge's window comes before the Pod exists."""
    text = RUNBOOK.read_text()
    titles = re.findall(r"^## (\d+)\. (.*)$", text, re.M)
    number = {
        key: next(int(n) for n, title in titles if title.startswith(prefix))
        for key, prefix in (
            ("pod", "Pod creation"),
            ("receipt", "Readiness receipt"),
            ("bridge", "The reward bridge"),
            ("worker", "Worker launch"),
            ("supervisor", "Supervisor"),
        )
    }
    assert [number[key] for key in ("receipt", "bridge", "worker", "supervisor")] == list(
        range(number["receipt"], number["receipt"] + 4)
    )
    pod = text.index(f"## {number['pod']}. Pod creation")
    for phrase in (
        "empties Docker Desktop's file sharing",
        "ssh-add -D",
        "never source `.env` in it",
        '--receipt "$SHARED/window-check-bridge.json"',
    ):
        assert -1 < text.index(phrase) < pod, phrase
    assert text.index("scratchcoder rl-bridge") < text.index("tools.posttrain_run run")
    worker = text[
        text.index(f"## {number['worker']}. ") : text.index(f"## {number['supervisor']}. ")
    ]
    assert "```sh\nwatchdog\n" in worker  # The watchdog is re-based just before the launch.
    supervisor = text[text.index(f"## {number['supervisor']}. ") :]
    supervisor = supervisor[: supervisor.index("\n## ")]
    assert "as soon as the heartbeat exists" in supervisor and "pod_ready True" in supervisor


def test_the_pod_and_mac_commands_name_run_twos_paths_and_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (run,) = base.by("tools.posttrain_run", "run")
    assert options(run)["--specification"] == "/workspace/posttrain/specification.json"
    assert options(run)["--output"] == "/workspace/posttrain-run"
    (watchdog,) = base.by("tools.runpod_watchdog")
    flags = options(watchdog)
    assert flags["--max-wall-seconds"] == "25380" and flags["--max-cost-usd"] == "4.22"
    assert flags["--awaiting-reward-extension-seconds"] == "7914.5"
    assert flags["--awaiting-dev-extension-seconds"] == "11806.5"
    assert flags["--policy"] == f"{VALUES['CTRL']}/policy.json"
    (supervisor,) = base.by("tools.pilot_supervisor")
    assert supervisor[:4] == ["--mode", "posttrain", "--run", "2"]
    (receipt,) = base.by("tools.pilot_readiness_receipt")
    assert options(receipt)["--mac-copy"] == "runs/posttrain-shared-2/mac-copy.json"
    (feasibility,) = base.by("tools.pilot_feasibility")
    (gate,) = base.by("tools.posttrain_approved", "run2-gate")
    assert options(gate)["--pool-test-export"] == "runs/posttrain-shared/exports/test-export.jsonl"
    # The estimate is read from the projection's own output, the one step 4 writes.
    assert options(gate)["--feasibility"] == options(feasibility)["--output"]
    assert options(gate)["--feasibility"] == "runs/posttrain-shared-2/feasibility.json"
    (export,) = base.by("scratchcoder.cli", "rl-pod-export")
    assert options(export)["--dev-tasks"] == "runs/task-pool-20261025/pool/dev.jsonl"
    (build,) = base.by("tools.posttrain_binding", "build")
    inputs = [
        token.split("=", 1)[0] for token in build if "=" in token and not token.startswith("-")
    ]
    spec = posttrain_approved.run2_specification()
    # Every input without a key in the template is given: the staged ones and the SFT
    # weights, whose key the gate fills (the Mac's copy is checked against the pin).
    unkeyed = {name for name, pin in spec["inputs"].items() if pin["key"] is None}
    assert set(inputs) == unkeyed and "tokenizer" not in inputs and "sft-weights" in inputs
    assert options(feasibility)["--ceiling-usd"] == "5.80"
    fields = [
        (options(argv).get("--field") or "").rstrip(")")
        for argv in base.by("tools.posttrain_ops", "limits")
    ]
    for field in (
        "limit_usd",
        "wall_seconds",
        "self_destruct_seconds",
        "watchdog_cost_stop_usd",
        "awaiting_reward_extension_seconds",
        "awaiting_dev_extension_seconds",
        "mac_reward_timeout_seconds",
        "mac_dev_timeout_seconds",
    ):
        assert field in fields, field
    lines = base.logical_lines()
    loops = {}
    for index, line in enumerate(lines):
        match = re.match(r"for name in (.*?); do$", line)
        if match:
            loops[lines[index + 1]] = match.group(1).split()
    sealed = [names for body, names in loops.items() if "eval-seal" in body]
    assert sealed == [
        [f"eval-{arm}-pool-test-{d}" for arm in ("rl", "more-sft") for d in ("greedy", "n8")]
    ]
    dev = [names for body, names in loops.items() if "eval-execute" in body and "dev.jsonl" in body]
    assert dev == [
        [f"eval-{arm}-pool-dev-{d}" for arm in ("rl", "more-sft") for d in ("greedy", "n8")]
    ]
    test = [
        names for body, names in loops.items() if "eval-execute" in body and "test.jsonl" in body
    ]
    assert sorted(name for names in test for name in names) == sorted(
        f"eval-{arm}-pool-test-{d}"
        for arm in ("base", "sft", "rl", "more-sft")
        for d in ("greedy", "n8")
    )
    for body in loops:
        if "test.jsonl" in body:
            assert "--seal-ledger" in body and '--unseal "$SHARED/unseal.json"' in body
    (card,) = base.by("scratchcoder.cli", "scorecard")
    names = [token for token in card if token.startswith(("base=", "sft=", "rl=", "more-sft="))]
    assert [name.split("=")[0] for name in names] == ["base", "sft", "rl", "more-sft"]
    assert (
        card.count("--results") == 16
        and card.count("--benchmark") == 12
        and card.count("--arc") == 4
    )


def test_the_bootstrap_call_uses_only_flags_the_script_knows() -> None:
    script = base.BOOTSTRAP.read_text()
    known = set(re.findall(r"^\s+(--[a-z0-9-]+)\)", script, re.M))
    (line,) = logical("pilot_pod_bootstrap.sh --mode posttrain")
    command = line[line.index("pilot_pod_bootstrap.sh") :]
    command = command[: command.index(";")]
    flags = [token for token in shlex.split(base.substitute(command)) if token.startswith("--")]
    assert set(flags) <= known, set(flags) - known
    assert "--lifetime-seconds 25200" in base.substitute(command) and "--run 2" in base.substitute(
        command
    )


def unseal_script() -> str:
    (block,) = [block for block in base.blocks() if "unseal.json" in block and "<<'PY'" in block]
    return block[block.index("<<'PY'\n") + len("<<'PY'\n") : block.index("\nPY\n")]


def report(run: str, pod: str, **extra: Any) -> dict[str, Any]:
    stages = {"1": ("eval-base", "eval-sft"), "2": ("eval-rl", "eval-more-sft")}[run]
    artifacts = [
        {
            "label": f"{stage}-pool-test-{decoding}",
            "name": "completions.jsonl",
            "key": f"posttrain-runs/{run}/{pod}/{stage}-pool-test-{decoding}/completions.jsonl",
            "sha256": f"{stage}-{decoding}",
            "bytes": 10,
        }
        for stage in stages
        for decoding in ("greedy", "n8")
    ]
    return {"run": run, "pod_id": pod, "artifacts": artifacts, **extra}


def test_the_unseal_record_names_all_four_endpoints_from_the_verified_reports(
    tmp_path: Path,
) -> None:
    run1 = report(
        "1", "p1", inputs={"b-mixed-final": "base-sha"}, sft={"selected": {"sha256": "sft-sha"}}
    )
    run2 = report(
        "2",
        "p2",
        selection={
            "rl": {"selected": {"sha256": "rl-sha"}},
            "more-sft": {"selected": {"sha256": "ctl-sha"}},
        },
    )
    paths = [tmp_path / "r1.json", tmp_path / "r2.json", tmp_path / "unseal.json"]
    paths[0].write_text(json.dumps(run1))
    paths[1].write_text(json.dumps(run2))
    result = subprocess.run(
        ["python3", "-", *map(str, paths)],
        input=unseal_script(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(paths[2].read_text())
    assert record["kind"] == "unseal" and record["schema_version"] == 1 and record["frozen_at"]
    assert record["endpoints"] == {
        "base": {
            "checkpoint_sha256": "base-sha",
            "completions_sha256": ["eval-base-greedy", "eval-base-n8"],
        },
        "sft": {
            "checkpoint_sha256": "sft-sha",
            "completions_sha256": ["eval-sft-greedy", "eval-sft-n8"],
        },
        "rl": {
            "checkpoint_sha256": "rl-sha",
            "completions_sha256": ["eval-rl-greedy", "eval-rl-n8"],
        },
        "more-sft": {
            "checkpoint_sha256": "ctl-sha",
            "completions_sha256": ["eval-more-sft-greedy", "eval-more-sft-n8"],
        },
    }
    again = subprocess.run(
        ["python3", "-", *map(str, paths)],
        input=unseal_script(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert again.returncode != 0  # The record is never rewritten.
    swapped = subprocess.run(
        ["python3", "-", str(paths[1]), str(paths[0]), str(tmp_path / "other.json")],
        input=unseal_script(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert swapped.returncode != 0 and "in that order" in swapped.stderr


def test_the_unseal_record_refuses_one_checkpoint_under_two_endpoints(tmp_path: Path) -> None:
    """Both arms may select the control's start, RL's update-0 check: the record is refused
    before it is written, with the reason, so the owner rules first."""
    run1 = report(
        "1", "p1", inputs={"b-mixed-final": "base-sha"}, sft={"selected": {"sha256": "sft-sha"}}
    )
    shared = {"selected": {"sha256": "rl-check-0-sha"}}
    run2 = report("2", "p2", selection={"rl": shared, "more-sft": shared})
    paths = [tmp_path / "r1.json", tmp_path / "r2.json", tmp_path / "unseal.json"]
    paths[0].write_text(json.dumps(run1))
    paths[1].write_text(json.dumps(run2))
    result = subprocess.run(
        ["python3", "-", *map(str, paths)],
        input=unseal_script(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "endpoints rl and more-sft are one checkpoint" in result.stderr
    assert not paths[2].exists()
    text = RUNBOOK.read_text()
    check = text.index('i=s["identical_endpoints"]')
    assert check < text.index("**Only after the owner freezes all four endpoints**")


def test_no_block_holds_a_comment() -> None:
    found = {line for block in base.blocks() for line in comment_lines(block)}
    assert not found, sorted(found)


@pytest.mark.parametrize("shell", [name for name in ("bash", "zsh") if shutil.which(name)])
def test_every_block_parses_in_the_operators_shells(shell: str, tmp_path: Path) -> None:
    for block in base.blocks():
        pasted = block
        for placeholder, value in PLACEHOLDERS.items():
            pasted = pasted.replace(placeholder, value)
        argv = ["zsh", "-f", "-i", "-n"] if shell == "zsh" else ["bash", "-n"]
        result = subprocess.run(
            argv,
            input=pasted,
            env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "PS1": ""},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (block.splitlines()[0], result.stderr)
        for error in ("unmatched", "parse error", "event not found", "bad pattern"):
            assert error not in result.stderr, (block.splitlines()[0], result.stderr)


def test_the_runbook_covers_every_owner_step_and_the_pod_body_is_run_twos() -> None:
    text = RUNBOOK.read_text()
    for phrase in (
        "revokes the Pod's R2 token",
        "empties Docker Desktop's file sharing",
        "ssh-add -D",
        "two reads at least 30 minutes apart agree",
        "Only after the owner freezes all four endpoints",
        "The sealed test split is not read again for run 2",
        "pod_ready True",
    ):
        assert phrase in text, phrase
    sections = re.findall(r"^## (\d+)\. ", text, re.M)
    assert sections == [str(number) for number in range(1, len(sections) + 1)]
    body = json.loads((ROOT / "configs" / "runpod-pod-a40-s-posttrain-2.json").read_text())
    first = json.loads((ROOT / "configs" / "runpod-pod-a40-s-posttrain.json").read_text())
    assert body == {**first, "name": "posttrain-a40-s-2"}
