"""Every tool command in the post-training runbook parses with its real command line.

Shell variables get concrete values; no tool runs. A tool command is parsed by calling its
module's ``main`` with ``ArgumentParser.parse_args`` intercepted (the method of
``tests/test_rehearsal_runbook.py``), so an unknown flag, a missing required flag or a bad
choice fails here as it would on the Mac or the Pod; ``scratchcoder`` commands go through
``scratchcoder.cli.main`` the same way. Every shell block must parse in bash and in an
interactive zsh and hold no comment, and the bootstrap call must use only flags the script
knows.
"""

import argparse
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from test_rehearsal_runbook import comment_lines

from tools import posttrain_ops
from tools.posttrain_run import PROBE_STOP, PROJECTED_STOP, SPEED_STOP

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "POSTTRAIN_RUNBOOK.md"
BOOTSTRAP = ROOT / "tools" / "pilot_pod_bootstrap.sh"
SPEC = ROOT / "configs" / "rehearsal-a40-s-post-1.json"
NOW = 1_800_000_300
ORIGIN = 1_800_000_000
BINDING_SHA256 = "ab" * 32
VALUES = {
    "RUN": "1",
    "CTRL_ROOT": "runs/posttrain-controller",
    "CTRL": "runs/posttrain-controller/1/podpost01",
    "KNOWN_HOSTS": "/Users/owner/.ssh/posttrain_1_known_hosts",
    "SPEC": "configs/rehearsal-a40-s-post-1.json",
    "SHARED": "runs/posttrain-shared",
    "POD_SSH_KEY": "/Users/owner/.ssh/runpod_ed25519",
    "PRIVATE_REPO": "owner/private-artifacts",
    "PUBLIC_REPO": "owner/public",
    "POOL": "runs/task-pool-20261025/pool",
    "SFT": "runs/sft-data-20261026",
    "ARC_DATA": "runs/arc-review",
    "TOKENIZER": "runs/tokenizer-freeze-65536-20261020/tokenizer.json",
    "BINDING": "runs/posttrain-shared/binding.json",
    "BINDING_SHA256": BINDING_SHA256,
    "WALL": "18000",
    "TIMER": "17820",
    "WATCHDOG_COST": "2.99",
    "EXPORTS": "runs/posttrain-shared/exports",
    "ORIGIN": str(ORIGIN),
    "LAUNCH": f"runs/posttrain-controller/1/launch-{ORIGIN}",
    "POD_ID": "podpost01",
    "SELF_DESTRUCT_DEADLINE": str(ORIGIN + 17_820),
    "POD_HOST": "203.0.113.7",
    "POD_PORT": "40022",
    "BASE": str(NOW),
    "ART": "runs/posttrain-shared/artifacts",
    "SANDBOX_IMAGE_ID": "sha256:" + "d" * 64,
    "SUITE": "humaneval",
    "BUCKETS": "6",
}
TOOL = re.compile(r"(?:uv run --locked(?: --extra [a-z]+)? python|python3) -m (tools\.[a-z_]+)")
CLI = re.compile(r"uv run --locked(?: --extra [a-z]+)? scratchcoder ([a-z-]+)")
PROGRESS = re.compile(r"python3 \S*tools/pod_progress\.py")


class Parsed(BaseException):
    def __init__(self, namespace: argparse.Namespace) -> None:
        super().__init__()
        self.namespace = namespace


def substitute(text: str) -> str:
    text = text.replace("$(date +%s)", str(NOW))

    def value(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        assert name in VALUES, f"the runbook uses ${name}, which this test does not know"
        return VALUES[name]

    return re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}|\$([A-Z_][A-Z0-9_]*)", value, text)


def blocks() -> list[str]:
    return re.findall(r"```sh\n(.*?)```", RUNBOOK.read_text(encoding="utf-8"), re.S)


def logical_lines() -> list[str]:
    lines: list[str] = []
    for block in blocks():
        pending = ""
        for line in block.splitlines():
            if line.rstrip().endswith("\\"):
                pending += line.rstrip()[:-1].rstrip() + " "
                continue
            lines.append((pending + line.strip()).strip())
            pending = ""
    return lines


def inline() -> list[str]:
    prose = re.sub(r"```.*?```", "", RUNBOOK.read_text(encoding="utf-8"), flags=re.S)
    return [span for span in re.findall(r"`([^`]+)`", prose) if TOOL.search(span)]


def end_of(line: str, start: int) -> int:
    """Where one command ends: an unquoted shell operator or the close of its quotes."""
    if line[:start].count("'") % 2:
        close = line.index("'", start)
        separator = line.find(";", start, close)
        return close if separator == -1 else separator
    single = double = escaped = False
    for index in range(start, len(line)):
        character = line[index]
        if escaped:
            escaped = False
        elif character == "\\" and not single:
            escaped = True
        elif character == "'" and not double:
            single = not single
        elif character == '"' and not single:
            double = not double
        elif not single and not double and character in "|;&<>":
            digits = index
            while digits > start and line[digits - 1].isdigit():
                digits -= 1
            if digits < index and line[digits - 1].isspace():
                return digits
            return index
    return len(line)


def commands() -> list[tuple[str, list[str], str]]:
    """(module, argv, text) for every tool and ``scratchcoder`` command in the runbook."""
    found = []
    for line in [*logical_lines(), *inline()]:
        line = re.sub(r"<[^<>\s]*sha256>", "ef" * 32, line)
        line = re.sub(r"<r2_final\.artifact_count>", "45", line)
        for pattern, kind in ((TOOL, "tool"), (CLI, "cli"), (PROGRESS, "progress")):
            for match in pattern.finditer(line):
                text = line[match.start() : end_of(line, match.start())]
                tokens = shlex.split(substitute(text))
                if kind == "tool":
                    index = tokens.index("-m")
                    found.append((tokens[index + 1], tokens[index + 2 :], text))
                elif kind == "cli":
                    index = tokens.index("scratchcoder")
                    found.append(("scratchcoder.cli", tokens[index + 1 :], text))
                else:
                    index = next(i for i, t in enumerate(tokens) if t.endswith("pod_progress.py"))
                    found.append(("tools.pod_progress", tokens[index + 1 :], text))
    return found


def parse(module_name: str, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    module = importlib.import_module(module_name)

    def capturing(name: str) -> Callable[..., argparse.Namespace]:
        original = getattr(argparse.ArgumentParser, name)

        def capture(
            self: argparse.ArgumentParser, args: object = None, namespace: object = None
        ) -> argparse.Namespace:
            raise Parsed(original(self, args, namespace))

        return capture

    with monkeypatch.context() as patch:
        for name in ("parse_args", "parse_intermixed_args"):
            patch.setattr(argparse.ArgumentParser, name, capturing(name))
        try:
            module.main(argv)
        except Parsed as parsed:
            return parsed.namespace
        except SystemExit as error:
            raise AssertionError(f"{module_name} rejects {argv}: exit {error.code}") from error
    raise AssertionError(f"{module_name}.main returned without parsing its arguments")


def by(module: str, first: str | None = None) -> list[list[str]]:
    return [
        argv
        for name, argv, _ in commands()
        if name == module and (first is None or (argv and argv[0] == first))
    ]


def test_every_tool_command_parses_with_its_real_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = commands()
    counts = Counter(module for module, _, _ in found)
    for module, minimum in {
        "tools.posttrain_run": 2,
        "tools.posttrain_ops": 5,
        "tools.posttrain_binding": 4,
        "tools.pilot_readiness_receipt": 1,
        "tools.pilot_feasibility": 1,
        "tools.pilot_supervisor": 1,
        "tools.runpod_watchdog": 1,
        "tools.runpod_pod_identity": 4,
        "tools.mac_window_check": 4,
        "tools.benchmark_dispatch": 1,
        "tools.rehearsal_ops": 1,
        "tools.pod_progress": 1,
        "scratchcoder.cli": 8,
    }.items():
        assert counts[module] >= minimum, (
            f"expected {minimum} {module} commands, got {counts[module]}"
        )
    for module, argv, text in found:
        namespace = parse(module, argv, monkeypatch)
        assert "SECRET" not in text and "ACCESS_KEY" not in text, text
        if module == "tools.mac_window_check" and namespace.command == "exec":
            assert namespace.argv[:1] == ["--"] and namespace.argv[1:4] == ["uv", "run", "--locked"]


def test_the_derived_limits_in_the_runbook_are_the_specifications() -> None:
    derived = posttrain_ops.specification_limits(json.loads(SPEC.read_text()))
    assert (
        str(derived.wall_seconds),
        str(derived.self_destruct_seconds),
        str(derived.watchdog_cost_stop_usd),
    ) == (VALUES["WALL"], VALUES["TIMER"], VALUES["WATCHDOG_COST"])
    assert int(VALUES["SELF_DESTRUCT_DEADLINE"]) == derived.deadline_unix(ORIGIN)
    text = RUNBOOK.read_text()
    for phrase in ("**$3.00**", "**$2.99**", "**18,000 s**", "**17,820 s**", "**900 s**"):
        assert phrase in text, phrase
    for reason in (PROBE_STOP, SPEED_STOP, PROJECTED_STOP):
        assert reason in text, reason
    assert "60,000 s of the pretraining rehearsal is never used" in text


def test_the_pod_commands_name_the_post_training_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    (run,) = by("tools.posttrain_run", "run")
    values = dict(zip(run[1::2], run[2::2], strict=True))
    assert values == {
        "--specification": VALUES["SPEC"],
        "--binding": "/workspace/posttrain/binding.json",
        "--binding-sha256": BINDING_SHA256,
        "--root": "/workspace/posttrain-inputs",
        "--readiness": "/workspace/posttrain/readiness-receipt.json",
        "--preflight": "/workspace/posttrain/preflight-a40/report.json",
        "--revision": "/workspace/posttrain/revision.json",
        "--output": "/workspace/posttrain-run",
    }
    (watchdog,) = by("tools.runpod_watchdog")
    options = dict(zip(watchdog[::2], watchdog[1::2], strict=False))
    assert options["--mode"] == "posttrain" and options["--max-wall-seconds"] == "18000"
    assert options["--max-cost-usd"] == "2.99" and options["--max-transfer-seconds"] == "900"
    (supervisor,) = by("tools.pilot_supervisor")
    assert ["--mode", "posttrain", "--run", "1"] == supervisor[:4]
    assert "/workspace/posttrain-run" in supervisor
    (receipt,) = by("tools.pilot_readiness_receipt")
    assert receipt[0] == "posttrain" and "--mac-copy" in receipt
    executed = [argv for argv in by("tools.mac_window_check", "exec")]
    executors = [argv[argv.index("scratchcoder") + 1] for argv in executed]
    assert sorted(set(executors)) == ["eval-baseline", "eval-execute", "rl-screen"]
    for argv in executed:
        assert "--dotenv" in argv and "--image-id" in argv
    screen = next(argv for argv in executed if "rl-screen" in argv)
    assert "--require-header" in screen and screen[screen.index("--timeout") + 1] == "5600"
    assert by("scratchcoder.cli", "eval-seal") and by("scratchcoder.cli", "eval-execute")
    loops = {}
    lines = logical_lines()
    for index, line in enumerate(lines):
        match = re.match(r"for name in (.*?); do$", line)
        if match:
            loops[lines[index + 1]] = match.group(1).split()
    sealed = [names for body, names in loops.items() if "eval-seal" in body]
    executed = [names for body, names in loops.items() if "eval-execute" in body]
    assert sealed == [
        [f"eval-{arm}-pool-test-{d}" for arm in ("base", "sft") for d in ("greedy", "n8")]
    ]
    # Only dev completions execute on the Mac in run 1: the test split stays sealed.
    assert executed == [
        [f"eval-{arm}-pool-dev-{d}" for arm in ("base", "sft") for d in ("greedy", "n8")]
    ]


def test_the_bootstrap_call_uses_only_flags_the_script_knows() -> None:
    script = BOOTSTRAP.read_text()
    known = set(re.findall(r"^\s+(--[a-z0-9-]+)\)", script, re.M))
    (line,) = [
        line for line in logical_lines() if "pilot_pod_bootstrap.sh --mode posttrain" in line
    ]
    command = line[line.index("pilot_pod_bootstrap.sh") :]
    command = command[: command.index(";")]
    flags = [token for token in shlex.split(substitute(command)) if token.startswith("--")]
    assert set(flags) <= known, set(flags) - known
    for required in (
        "--lifetime-seconds",
        "--idle-seconds",
        "--heartbeat",
        "--binding-sha256",
        "--revision",
        "--archive",
    ):
        assert required in flags, required
    # The timer is given its own lifetime, not the wall: it refuses any later deadline.
    assert "--lifetime-seconds $TIMER" in command
    assert "--lifetime-seconds 17820" in substitute(command)
    assert "--heartbeat /workspace/posttrain-run/heartbeat.json" in command


def test_no_block_holds_a_comment() -> None:
    found = {line for block in blocks() for line in comment_lines(block)}
    assert not found, sorted(found)


@pytest.mark.parametrize("shell", [name for name in ("bash", "zsh") if shutil.which(name)])
def test_every_block_parses_in_the_operators_shells(shell: str, tmp_path: Path) -> None:
    for block in blocks():
        pasted = block.replace("<reviewed binding sha256>", BINDING_SHA256)
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


def test_the_runbook_covers_every_owner_step(tmp_path: Path) -> Any:
    text = RUNBOOK.read_text()
    for phrase in (
        "revokes the Pod's R2 token",
        "empties Docker Desktop's file sharing",
        "ssh-add -D",
        "two reads agree",
        "SHA-256-verified" if "SHA-256-verified" in text else "pinned to its size and SHA-256",
        "The test export is the one launch-time touch of the sealed split",
    ):
        assert phrase in text, phrase
    sections = re.findall(r"^## (\d+)\. ", text, re.M)
    assert sections == [str(number) for number in range(1, len(sections) + 1)]
