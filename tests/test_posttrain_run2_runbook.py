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

import pytest
import test_posttrain_runbook as base
from test_rehearsal_runbook import comment_lines

from tools import posttrain_approved, posttrain_endpoints, posttrain_ops
from tools.pilot_feasibility import posttrain2_report
from tools.posttrain_run import (
    BRIDGE_LOST_STOP,
    SFT_RESULTS,
    WAIT_MARGIN_STOP,
    Specification,
    launch_fit,
)

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "POSTTRAIN_RUN2_RUNBOOK.md"
ORIGIN = base.ORIGIN
BOUNDS = posttrain_ops.bridge_timeouts(attempts_per_request=256, dev_programs=1_800)
ESTIMATE = 3.0658
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
    "RUN2_REPORT": "runs/posttrain-controller/2/podpost02/supervisor/final/report.json",
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
            "sft-results": {
                name: {"key": f"posttrain-runs/1/p/{name}", "size_bytes": 1, "sha256": "a" * 64}
                for name in SFT_RESULTS
            },
        }
    )
    derived = posttrain_ops.specification_bridge_timeouts(filled)
    assert derived == BOUNDS
    assert posttrain_ops.specification_limits(filled) == LIMITS
    text = RUNBOOK.read_text()
    planned = posttrain2_report(posttrain_approved.SPECIFICATION_PATH_2, ceiling_usd=5.80)
    assert ESTIMATE == round(planned["scenarios"]["planning speeds"]["total_usd"], 4)
    assert isinstance(LIMITS, posttrain_ops.EstimateLimits) and LIMITS.limit_usd == 3.67
    assert (
        posttrain_ops.estimate_limit_usd(5.80, 4.84)
        == 5.80
        > posttrain_ops.estimate_limit_usd(5.80, 4.83)
    )
    assert posttrain_ops.ceiling_limits(5.80, 0.60).wall_seconds == 34_800
    for phrase in (
        "**$3.67**",
        "**$3.66**",
        "**22,020 s**",
        "**21,840 s**",
        "Only an estimate of $4.84 or more lets the $5.80 ceiling bind",
        "**900 s**",
        "**5,600 s**",
        "**9,492 s**",
        "**7,854.5 s**",
        "**11,746.5 s**",
        "7,914.5 s (reward), 11,806.5 s (dev)",
        "11,746.5 s, $1.96",
        "8,279 tok/s at batch 128 for the evaluation and 9,268 at batch 256",
        "17,495 s",
        "about 4,345 s (1.2 hours) of the 21,840 s",
        "5.1 hours and $3.07",
        "cost limit 3.67 USD, wall 22020 s, timer 21840 s, watchdog cost stop 3.66 USD",
    ):
        assert phrase in text, phrase
    lifetime = LIMITS.self_destruct_seconds
    spec = Specification.from_dict(filled)
    needed = launch_fit(spec, remaining_seconds=lifetime)["planning_seconds_needed"]
    assert (round(needed), lifetime - round(needed)) == (17_495, 4_345)
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
        # The owner's run 2 gate rulings: the window checks repeat while the bridge runs, and
        # the Pod's R2 token stays in .env under the child-environment rule.
        "**exit 4**",
        "$BRIDGE_STATE/window-checks/",
        "stays in `.env` during the bridge window under the child-environment rule",
        "only through `mac_window_check exec`",
    ):
        assert phrase in text, phrase
    from scratchcoder.rl.bridge_mac import WINDOW_CHECK_INTERVAL_SECONDS

    assert values["--window-check-interval"] == "900"
    assert namespace.window_check_interval == WINDOW_CHECK_INTERVAL_SECONDS == 900.0


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
    assert flags["--max-wall-seconds"] == "22020" and flags["--max-cost-usd"] == "3.66"
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
    loops: dict[str, str] = {}
    for index, line in enumerate(lines):
        match = re.match(r"for name in (.*?); do$", line)
        if match:
            loops[lines[index + 1]] = match.group(1)
    # Run 2's files come from the endpoints plan: only an arm evaluated on the Pod has any.
    listing = 'tools.posttrain_endpoints list --plan "$SHARED/endpoints.json" --list'
    sealed = [names for body, names in loops.items() if "eval-seal" in body]
    assert sealed == [f"$(uv run --locked python -m {listing} run2-pool-test)"]
    dev = [names for body, names in loops.items() if "eval-execute" in body and "dev.jsonl" in body]
    assert dev == [f"$(uv run --locked python -m {listing} run2-pool-dev)"]
    benchmarks = [names for body, names in loops.items() if body.startswith("SUITE=")]
    assert benchmarks == [f"$(uv run --locked python -m {listing} run2-benchmarks)"]
    test = [
        names for body, names in loops.items() if "eval-execute" in body and "test.jsonl" in body
    ]
    assert sorted(test) == sorted(
        [
            " ".join(
                f"eval-{arm}-pool-test-{d}" for arm in ("base", "sft") for d in ("greedy", "n8")
            ),
            f"$(uv run --locked python -m {listing} run2-pool-test)",
        ]
    )
    for body in loops:
        if "test.jsonl" in body:
            assert "--seal-ledger" in body and '--unseal "$SHARED/unseal.json"' in body
    for kind in ("run2-pool-dev", "run2-pool-test", "run2-benchmarks"):
        assert kind in posttrain_endpoints.LISTS
    (plan,) = base.by("tools.posttrain_endpoints", "plan")
    assert options(plan)["--specification"] == VALUES["SPEC"]
    assert options(plan)["--run1-artifacts"] == "runs/posttrain-shared/artifacts"
    (unseal,) = base.by("tools.posttrain_endpoints", "unseal")
    assert options(unseal)["--output"] == "runs/posttrain-shared-2/unseal.json"
    (inputs,) = base.by("tools.posttrain_endpoints", "scorecard-inputs")
    (card,) = base.by("scratchcoder.cli", "scorecard")
    assert options(card)["--inputs"] == options(inputs)["--output"]
    assert "--checkpoint" not in card and "--results" not in card


def test_the_bootstrap_call_uses_only_flags_the_script_knows() -> None:
    script = base.BOOTSTRAP.read_text()
    known = set(re.findall(r"^\s+(--[a-z0-9-]+)\)", script, re.M))
    (line,) = logical("pilot_pod_bootstrap.sh --mode posttrain")
    command = line[line.index("pilot_pod_bootstrap.sh") :]
    command = command[: command.index(";")]
    flags = [token for token in shlex.split(base.substitute(command)) if token.startswith("--")]
    assert set(flags) <= known, set(flags) - known
    assert "--lifetime-seconds 21840" in base.substitute(command) and "--run 2" in base.substitute(
        command
    )


def test_the_unseal_record_and_the_scorecard_come_from_the_endpoints_plan() -> None:
    """The plan is written before anything of run 2 executes; the unseal record only after the
    owner froze the endpoints, from the plan, so one checkpoint is executed once and an arm
    with the SFT weights is mapped to SFT; the scorecard's inputs come from the same plan.
    The reuse itself is tested in test_run2_endpoints.py."""
    text = RUNBOOK.read_text()
    step = text[text.index("## 18. ") : text.index("## 19. ")]
    order = [
        step.index("tools.posttrain_endpoints plan"),
        step.index("--list run2-pool-dev"),
        step.index("**Only after the owner freezes all four endpoints**"),
        step.index("tools.posttrain_endpoints unseal"),
        step.index('--unseal "$SHARED/unseal.json"'),
        step.index("tools.posttrain_endpoints scorecard-inputs"),
        step.index("scratchcoder scorecard --inputs"),
    ]
    assert order == sorted(order)
    for phrase in (
        "reuses-run-1-sft",
        "never as a measurement of its own",
        "`reused_by`",
        "not independent",
    ):
        assert phrase in step, phrase
    assert "identical_endpoints" not in text


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
