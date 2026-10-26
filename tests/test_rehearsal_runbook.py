"""Every tool command in the rehearsal runbook must run as written against the real parsers.

Shell variables get concrete values; no tool is executed. A command is parsed by calling
its module's ``main`` with ``ArgumentParser.parse_args`` intercepted, so an unknown flag,
a missing required flag or a bad choice fails here exactly as it would on the Mac or Pod.

The runbook's own shell and Python (the per-Pod helpers, the step 9 wait loop, the
revision writer and the billing check) are extracted from this file and run against stubs.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tools import rehearsal_ops
from tools.runpod_watchdog import Policy, build_parser, policy_from_args

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "REHEARSAL_RUNBOOK.md"
BOOTSTRAP = ROOT / "tools" / "pilot_pod_bootstrap.sh"
POD_BODY = ROOT / "configs" / "runpod-pod-a40-s-rehearsal.json"
NOW = 1_800_000_300
ORIGIN = 1_800_000_000
BINDING_SHA256 = "ab" * 32
INIT_SHA256 = "cd" * 32
VALUES = {
    "RUN": "A",
    "RUN_LOWER": "a",
    "CTRL_ROOT": "runs/rehearsal-controller",
    "CTRL": "runs/rehearsal-controller/a/podfixture01",
    "KNOWN_HOSTS": "/Users/owner/.ssh/rehearsal_a_known_hosts",
    "SPEC": "configs/rehearsal-a40-s-a.json",
    "SHARED": "runs/rehearsal-shared",
    "POD_SSH_KEY": "/Users/owner/.ssh/runpod_ed25519",
    "PRIVATE_REPO": "owner/private-artifacts",
    "BUILD_DIR": "runs/rehearsal-s-20261025",
    "BINDING": "runs/rehearsal-binding/binding.json",
    "BINDING_SHA256": BINDING_SHA256,
    "INIT_SHA256": INIT_SHA256,
    "INIT_SIZE": "541142016",
    "INIT_KEY": f"rehearsal-init/{INIT_SHA256}/rehearsal-init.pt",
    "ORIGIN": str(ORIGIN),
    "POD_ID": "podfixture01",
    "RUNPOD_POD_ID": "podfixture01",
    "SELF_DESTRUCT_DEADLINE": str(ORIGIN + 45_000 - 180),
    "POD_HOST": "203.0.113.7",
    "POD_PORT": "40022",
    "BASE": str(NOW),
    "BUCKETS": "16",
}
LAUNCHER = re.compile(r"(?:uv run --locked(?: --extra [a-z]+)? python|python3) -m (tools\.[a-z_]+)")
WORKER_RUN_FLAGS = {
    "--specification",
    "--binding",
    "--binding-sha256",
    "--root",
    "--initialization",
    "--initialization-sha256",
    "--readiness",
    "--preflight",
    "--output",
}
"""The shared contract's worker command line, flag for flag."""


class Parsed(BaseException):
    """Raised from the intercepted parse so ``main`` stops before doing anything."""

    def __init__(self, namespace: argparse.Namespace) -> None:
        super().__init__()
        self.namespace = namespace


def _substitute(text: str) -> str:
    text = text.replace("$(date +%s)", str(NOW))

    def value(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        assert name in VALUES, f"the runbook uses ${name}, which this test does not know"
        return VALUES[name]

    return re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}|\$([A-Z_][A-Z0-9_]*)", value, text)


def _code() -> list[str]:
    """Logical lines of the runbook's shell blocks, continuations joined."""
    lines: list[str] = []
    inside = False
    pending = ""
    for line in RUNBOOK.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            inside = line.strip() == "```sh"
            continue
        if not inside:
            continue
        if line.rstrip().endswith("\\"):
            pending += line.rstrip()[:-1].rstrip() + " "
            continue
        lines.append((pending + line.strip()).strip())
        pending = ""
    return lines


def _inline() -> list[str]:
    """Commands quoted inline in the prose, such as `uv run ... -m tools.r2_store status`."""
    text = RUNBOOK.read_text(encoding="utf-8")
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    return [span for span in re.findall(r"`([^`]+)`", prose) if LAUNCHER.search(span)]


def _end(line: str, start: int) -> int:
    """Where one command ends: an unquoted shell operator, or the close of its quotes."""
    if line[:start].count("'") % 2:
        # The command sits inside bash -c '...', up to the quote or the next command.
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
                digits -= 1  # A file descriptor such as the "2" of "2>&1" belongs to it.
            if digits < index and line[digits - 1].isspace():
                return digits
            return index
    return len(line)


def commands() -> list[tuple[str, list[str], str]]:
    """(module, argv, source text) for every tool command in the runbook."""
    found = []
    for line in [*_code(), *_inline()]:
        line = re.sub(r"<[^<>\s]*sha256>", "ef" * 32, line)  # Values the operator pastes.
        for match in LAUNCHER.finditer(line):
            text = line[match.start() : _end(line, match.start())]
            tokens = shlex.split(_substitute(text), comments=True)
            index = tokens.index("-m")
            found.append((tokens[index + 1], tokens[index + 2 :], text))
    return found


def parse(module_name: str, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    module = importlib.import_module(module_name)
    original = argparse.ArgumentParser.parse_args

    def capture(
        self: argparse.ArgumentParser, args: object = None, namespace: object = None
    ) -> argparse.Namespace:
        raise Parsed(original(self, args, namespace))  # type: ignore[call-overload]

    with monkeypatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_args", capture)
        try:
            module.main(argv)
        except Parsed as parsed:
            return parsed.namespace
        except SystemExit as error:
            raise AssertionError(f"{module_name} rejects {argv}: exit {error.code}") from error
    raise AssertionError(f"{module_name}.main returned without parsing its arguments")


def by_module(name: str) -> list[tuple[list[str], str]]:
    return [(argv, text) for module, argv, text in commands() if module == name]


def worker_present() -> bool:
    return importlib.util.find_spec("tools.rehearsal_run") is not None


def test_every_tool_command_parses_with_its_real_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = commands()
    counts = Counter(module for module, _, _ in found)
    for module, minimum in {
        "tools.runpod_watchdog": 1,
        "tools.pilot_supervisor": 1,
        "tools.rehearsal_ops": 2,
        "tools.pilot_readiness_receipt": 1,
        "tools.pilot_feasibility": 1,
        "tools.rehearsal_binding": 1,
        "tools.r2_store": 2,
        "tools.runpod_pod_identity": 3,
        "tools.pilot_preflight": 1,
        "tools.rehearsal_run": 3,
    }.items():
        assert counts[module] >= minimum, f"expected {minimum} {module} commands"
    for module, argv, text in found:
        if module == "tools.rehearsal_run" and not worker_present():
            continue  # Checked against the shared contract below until the runner lands.
        parse(module, argv, monkeypatch)
        assert "SECRET" not in text and "ACCESS_KEY" not in text, text


def test_worker_commands_follow_the_shared_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = by_module("tools.rehearsal_run")
    assert {argv[0] for argv, _ in rows} == {"smoke", "write-initialization", "run"}
    (run,) = [argv for argv, _ in rows if argv[0] == "run"]
    flags = run[1::2]
    assert set(flags) == WORKER_RUN_FLAGS and len(flags) == len(WORKER_RUN_FLAGS)
    values = dict(zip(run[1::2], run[2::2], strict=True))
    assert values["--specification"] == "configs/rehearsal-a40-s-a.json"
    assert values["--binding-sha256"] == BINDING_SHA256
    assert values["--initialization-sha256"] == INIT_SHA256
    assert values["--readiness"] == "/workspace/rehearsal/readiness-receipt.json"
    assert values["--preflight"] == "/workspace/rehearsal/preflight-a40/report.json"
    if worker_present():
        parse("tools.rehearsal_run", run, monkeypatch)


def test_write_initialization_takes_both_specifications(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = by_module("tools.rehearsal_run")
    (initial,) = [argv for argv, _ in rows if argv[0] == "write-initialization"]
    args = parse("tools.rehearsal_run", initial, monkeypatch)
    assert sorted(str(path) for path in args.specification) == [
        "configs/rehearsal-a40-s-a.json",
        "configs/rehearsal-a40-s-b.json",
    ]
    from tools.rehearsal_run import Specification

    assert sorted(Specification.load(ROOT / path).run for path in args.specification) == ["A", "B"]
    # The command refuses one specification alone (both runs must share the init).
    with pytest.raises(SystemExit):
        importlib.import_module("tools.rehearsal_run").main(
            [
                "write-initialization",
                "--specification",
                str(ROOT / "configs/rehearsal-a40-s-a.json"),
                "--binding",
                "unused.json",
                "--binding-sha256",
                BINDING_SHA256,
                "--output",
                "unused.pt",
            ]
        )


def test_receipt_and_supervisor_commands_read_the_files_the_worker_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ((receipt, _),) = by_module("tools.pilot_readiness_receipt")
    assert receipt[0] == "rehearsal"
    namespace = parse("tools.pilot_readiness_receipt", receipt, monkeypatch)
    (run,) = [argv for argv, _ in by_module("tools.rehearsal_run") if argv[0] == "run"]
    worker = dict(zip(run[1::2], run[2::2], strict=True))
    assert str(namespace.specification) == worker["--specification"]  # Same file, same hash.
    assert namespace.binding_sha256 == worker["--binding-sha256"]
    assert namespace.initialization_sha256 == worker["--initialization-sha256"]
    assert namespace.run == VALUES["RUN"]
    assert str(namespace.policy) == f"{VALUES['CTRL']}/policy.json"
    # The receipt reads the Mac copy of the Pod's preflight report the worker reads.
    code = "\n".join(_code())
    assert (
        "podscp root@$POD_HOST:/workspace/rehearsal/preflight-a40/report.json "
        '"$CTRL/preflight-a40-report.json"' in code
    )
    assert str(namespace.preflight) == f"{VALUES['CTRL']}/preflight-a40-report.json"
    assert (
        'podscp "$CTRL/readiness-receipt.json" '
        "root@$POD_HOST:/workspace/rehearsal/readiness-receipt.json" in code
    )
    assert worker["--readiness"] == "/workspace/rehearsal/readiness-receipt.json"


def _watchdog() -> argparse.Namespace:
    rows = by_module("tools.runpod_watchdog")
    assert len(rows) == 1, "the runbook launches the watchdog from one helper only"
    return build_parser().parse_args(rows[0][0])


def _policy_file() -> dict[str, object]:
    text = RUNBOOK.read_text(encoding="utf-8")
    match = re.search(r'cat > "\$CTRL/policy\.json" <<EOF\n(.*?)\nEOF\n', text, re.S)
    assert match, "the runbook writes policy.json with a heredoc"
    value = json.loads(_substitute(match.group(1)))
    assert isinstance(value, dict)
    return value


def test_watchdog_runs_in_rehearsal_mode_with_the_d35_limits() -> None:
    args = _watchdog()
    policy = policy_from_args(args)
    assert policy.mode == "rehearsal"
    assert (policy.max_wall_seconds, policy.max_idle_seconds) == (45_000, 600)
    assert policy.max_cost_usd == rehearsal_ops.WATCHDOG_COST_STOP_USD == 7.54
    assert policy.max_cost_usd < rehearsal_ops.COST_STOP_USD
    assert policy.quoted_total_hourly == 0.60
    # The worker's final drain reads "uploading"; the watchdog allows it this long.
    assert policy.max_transfer_seconds == rehearsal_ops.MAX_TRANSFER_SECONDS == 900
    assert (policy.billing_started_at_unix, policy.idle_baseline_unix) == (ORIGIN, NOW)
    assert args.heartbeat == Path(VALUES["CTRL"]) / "supervisor" / "heartbeat.json"
    assert args.pidfile == Path(VALUES["CTRL"]) / "watchdog.pid"


def test_policy_file_matches_the_watchdog_command_line() -> None:
    # tools.pilot_readiness_receipt and tools.pilot_supervisor both call Policy(**value).
    policy = Policy(**_policy_file())  # type: ignore[arg-type]
    cli = policy_from_args(_watchdog())
    for name in (
        "mode",
        "max_wall_seconds",
        "max_cost_usd",
        "max_idle_seconds",
        "quoted_total_hourly",
        "max_transfer_seconds",
        "billing_started_at_unix",
        "idle_baseline_unix",
    ):
        assert getattr(policy, name) == getattr(cli, name), name


def test_each_runs_watchdog_is_restarted_alone_through_its_pidfile() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    code = "\n".join(_code())
    assert "pkill" not in code and "killall" not in code
    helper = re.search(r"\nwatchdog\(\) \{.*?\n\}\n", text, re.S)
    assert helper, "the runbook defines one watchdog helper"
    body = helper.group(0)
    assert 'OLD=$(cat "$CTRL/watchdog.pid"' in body
    assert 'grep -qF -- "--identity $CTRL/pod-identity.json"' in body
    assert body.index("nohup uv run") < body.index('kill "$OLD"')  # The new one starts first.
    assert code.count("watchdog()") == 1
    assert "\nwatchdog\n" in "\n" + code + "\n"


def test_supervisor_runs_in_rehearsal_mode_for_this_run(monkeypatch: pytest.MonkeyPatch) -> None:
    ((argv, _),) = by_module("tools.pilot_supervisor")
    args = parse("tools.pilot_supervisor", argv, monkeypatch)
    assert (args.mode, args.run) == ("rehearsal", "A")
    assert args.output == Path(VALUES["CTRL"]) / "supervisor"
    assert args.policy == Path(VALUES["CTRL"]) / "policy.json"
    assert str(args.known_hosts) == VALUES["KNOWN_HOSTS"]
    assert not re.search(r"\s", VALUES["KNOWN_HOSTS"])
    (run,) = [argv for argv, _ in by_module("tools.rehearsal_run") if argv[0] == "run"]
    values = dict(zip(run[1::2], run[2::2], strict=True))
    assert args.remote_root == values["--output"] == "/workspace/rehearsal-run"


def _self_destruct() -> list[str]:
    lines = [line for line in _code() if line.startswith("python3 tools/pod_self_destruct.py ")]
    assert len(lines) == 1
    return shlex.split(_substitute(lines[0]))[2:]


def test_self_destruct_command_parses_and_matches_the_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "placeholder")
    args = parse("tools.pod_self_destruct", _self_destruct(), monkeypatch)
    assert args.max_lifetime_seconds == 45_000
    assert args.deadline_unix == ORIGIN + 45_000 - 180
    assert "SELF_DESTRUCT_DEADLINE=$((ORIGIN + 45000 - 180))" in "\n".join(_code())
    script = BOOTSTRAP.read_text(encoding="utf-8")
    assert "LIFETIME_ARGS=(--max-lifetime-seconds 45000)" in script
    for flag in ("--pod-id", "--deadline-unix", "--receipt", "--stop-file"):
        assert flag in script
    assert args.receipt == "/workspace/rehearsal/self-destruct.json"
    assert args.stop_file == "/workspace/rehearsal/TERMINATE_NOW"


def test_bootstrap_command_uses_only_flags_the_script_accepts() -> None:
    (line,) = [line for line in _code() if "tools/pilot_pod_bootstrap.sh" in line]
    match = re.search(r"bash tools/pilot_pod_bootstrap\.sh ([^;']*)", line)
    assert match
    argv = shlex.split(_substitute(match.group(1)))
    script = BOOTSTRAP.read_text(encoding="utf-8")
    accepted = set(re.findall(r"^\s+(--[a-z0-9-]+)\)", script, re.M))
    flags = argv[0::2]
    assert set(flags) <= accepted and len(flags) == len(set(flags))
    values = dict(zip(argv[0::2], argv[1::2], strict=True))
    assert values["--mode"] == "rehearsal" and values["--run"] == "A"
    assert values["--deadline-unix"] == VALUES["SELF_DESTRUCT_DEADLINE"]
    assert values["--workspace"] == "/workspace/rehearsal"
    assert values["--initialization-sha256"] == INIT_SHA256
    assert values["--initialization-size"] == VALUES["INIT_SIZE"]
    assert values["--r2-env"] == "/workspace/rehearsal/r2-pod.env"


def test_r2_pod_key_travels_only_over_ssh_stdin() -> None:
    code = "\n".join(_code())
    (delivery,) = [line for line in _code() if "rehearsal_ops pod-env" in line]
    assert delivery.index("pod-env") < delivery.index("| podssh") < delivery.index("umask 077")
    assert "R2_POD_SECRET_ACCESS_KEY=" not in code and "R2_SECRET_ACCESS_KEY=" not in code
    body = json.loads(POD_BODY.read_text(encoding="utf-8"))
    assert "env" not in body and "R2" not in json.dumps(body)
    assert '--data @"$LAUNCH/create-body.json"' in code


def test_rehearsal_pod_body_is_the_pilot_body_with_a_40_gb_disk() -> None:
    body = json.loads(POD_BODY.read_text(encoding="utf-8"))
    pilot = json.loads((ROOT / "configs" / "runpod-pod-a40-s.json").read_text(encoding="utf-8"))
    assert body["disk"] == rehearsal_ops.CONTAINER_DISK_GB == 40
    assert {key: value for key, value in body.items() if key not in {"name", "disk"}} == {
        key: value for key, value in pilot.items() if key not in {"name", "disk"}
    }
    assert body["gpu"] == {"id": "NVIDIA A40", "count": 1, "minCudaVersion": "12.8"}


def test_per_run_state_is_kept_apart() -> None:
    code = "\n".join(_code())
    assert "export CTRL_ROOT=runs/rehearsal-controller" in code
    assert "export KNOWN_HOSTS=~/.ssh/rehearsal_${RUN_LOWER}_known_hosts" in code
    assert "export SPEC=configs/rehearsal-a40-s-$RUN_LOWER.json" in code
    assert 'ssh-keyscan -p "$POD_PORT" "$POD_HOST" > "$KNOWN_HOSTS"' in code


# Per-Pod controller directories -------------------------------------------------------


def section(number: int) -> str:
    """The runbook text of one numbered step."""
    text = RUNBOOK.read_text(encoding="utf-8")
    match = re.search(rf"\n## {number}\. .*?(?=\n## \d+\. |\Z)", text, re.S)
    assert match, f"no step {number}"
    return match.group(0)


def blocks(number: int) -> list[str]:
    """The shell blocks of one step, in order."""
    return re.findall(r"```sh\n(.*?)```", section(number), re.S)


def heredoc(opening: str, terminator: str) -> str:
    """The body of the heredoc whose first line contains ``opening``."""
    lines = RUNBOOK.read_text(encoding="utf-8").splitlines()
    (start,) = [number for number, line in enumerate(lines) if opening in line]
    end = lines.index(terminator, start + 1)
    return "\n".join(lines[start + 1 : end]) + "\n"


def test_each_pod_gets_a_new_controller_directory_and_step_5_refuses_an_existing_one() -> None:
    code = "\n".join(_code())
    create, *_ = blocks(5)
    assert "export CTRL=$CTRL_ROOT/$RUN_LOWER/$POD_ID" in create
    # The create request runs before the Pod ID exists, from its own launch directory.
    assert create.index("export LAUNCH=$CTRL_ROOT/$RUN_LOWER/launch-$ORIGIN") < create.index("curl")
    # mkdir without -p refuses an existing directory; run.env is written only after it.
    assert 'if mkdir "$CTRL"; then' in create
    guarded = create[create.index('if mkdir "$CTRL"; then') : create.index("\nfi\n")]
    assert '> "$CTRL/run.env"' in guarded and "already exists" in guarded
    assert create.count('"$CTRL/run.env"') == 1
    assert 'mkdir -p "$CTRL"' not in code and "mkdir -p $CTRL" not in code
    # Everything the watchdog, receipt and supervisor read lives under the per-Pod $CTRL.
    assert str(_watchdog().identity).startswith(VALUES["CTRL"] + "/")
    assert VALUES["CTRL"] == f"{VALUES['CTRL_ROOT']}/{VALUES['RUN_LOWER']}/{VALUES['POD_ID']}"


def test_step_5_refusal_leaves_an_existing_directory_untouched(tmp_path: Path) -> None:
    create = blocks(5)[0]
    guarded = create[create.index('if mkdir "$CTRL"; then') : create.index("\nfi\n") + 4]
    control = tmp_path / "runs" / "a" / "podfixture01"
    launch = tmp_path / "runs" / "a" / "launch-1"
    launch.mkdir(parents=True)
    for name in ("create-body.json", "create-response.json"):
        (launch / name).write_text("{}")
    environment = {
        **os.environ,
        "CTRL": str(control),
        "LAUNCH": str(launch),
        "RUN": "A",
        "ORIGIN": "1800000000",
        "POD_ID": "podfixture01",
        "SELF_DESTRUCT_DEADLINE": "1800044820",
    }
    first = subprocess.run(
        ["bash", "-c", guarded], env=environment, capture_output=True, text=True, check=True
    )
    assert first.stdout == "" and not launch.exists()
    assert (control / "run.env").read_text() == (
        "export ORIGIN=1800000000 POD_ID=podfixture01 SELF_DESTRUCT_DEADLINE=1800044820\n"
    )
    assert sorted(path.name for path in control.iterdir()) == [
        "create-body.json",
        "create-response.json",
        "run.env",
    ]
    launch.mkdir()
    again = subprocess.run(
        ["bash", "-c", guarded],
        env={**environment, "ORIGIN": "1800009999"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "already exists" in again.stdout and "never reused" in again.stdout
    assert "1800000000" in (control / "run.env").read_text()  # The first Pod's origin stays.


# The per-Pod helpers file ----------------------------------------------------------------

HELPERS_OPENING = """[ -e "$CTRL/helpers.sh" ] || cat > "$CTRL/helpers.sh" <<'EOF'"""
SHELLS = [shell for shell in ("bash", "zsh") if shutil.which(shell)]


def helpers() -> str:
    return heredoc(HELPERS_OPENING, "EOF")


def test_helpers_are_written_once_per_pod_and_sourced_by_every_terminal() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    code = "\n".join(_code())
    body = helpers()
    for name in ("watchdog() {", "podssh() {", "podscp() {"):
        assert body.count(name) == 1 and code.count(name) == 1, name
    assert "alias " not in code  # Functions in the sourced file, never per-shell aliases.
    # Written once: an existing file is never replaced.
    assert HELPERS_OPENING in code
    reopen = (
        'export CTRL=$CTRL_ROOT/$RUN_LOWER/<pod id>; source "$SHARED/init.env"; '
        'source "$CTRL/run.env"; source "$CTRL/helpers.sh"'
    )
    assert f"`{reopen}`" in section(5)
    assert "reopened-terminal line of step 5" in section(10)
    assert "reopened-terminal line of step 5" in section(14)
    assert 'source "$CTRL/helpers.sh"\nwatchdog\n' in blocks(5)[1]
    # Every later step that calls a helper runs after the file was sourced in step 5.
    first_use = min(code.index(name) for name in ("\npodssh ", "\npodscp "))
    assert code.index('source "$CTRL/helpers.sh"') < first_use
    assert text.count(HELPERS_OPENING) == 1


@pytest.mark.parametrize("shell", SHELLS)
def test_helpers_parse_in_the_operators_shells(tmp_path: Path, shell: str) -> None:
    path = tmp_path / "helpers.sh"
    path.write_text(helpers())
    result = subprocess.run([shell, "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


STUB_RECORDER = """#!/usr/bin/env bash
for argument in "$@"; do printf '%s\\n' "$argument"; done > "$STUB_DIR/$(basename "$0").args"
"""
STUB_WATCHDOG = """#!/usr/bin/env bash
# Stands in for "uv run ... tools.runpod_watchdog": writes its pid to --pidfile, then waits.
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --pidfile ]]; then echo "$$" > "$2"; fi
  shift
done
trap 'exit 0' TERM
while :; do sleep 0.2; done
"""


def helper_environment(tmp_path: Path) -> dict[str, str]:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name, text in (("ssh", STUB_RECORDER), ("scp", STUB_RECORDER), ("uv", STUB_WATCHDOG)):
        (stubs / name).write_text(text)
        (stubs / name).chmod(0o755)
    control = tmp_path / "ctrl"
    control.mkdir()
    (control / "helpers.sh").write_text(helpers())
    return {
        **os.environ,
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
        "STUB_DIR": str(stubs),
        "CTRL": str(control),
        "RUN": "A",
        "ORIGIN": str(ORIGIN),
        "POD_HOST": VALUES["POD_HOST"],
        "POD_PORT": VALUES["POD_PORT"],
        "POD_SSH_KEY": VALUES["POD_SSH_KEY"],
        "KNOWN_HOSTS": VALUES["KNOWN_HOSTS"],
    }


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop(pid: int) -> None:
    if alive(pid):
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize("shell", SHELLS)
def test_sourced_helpers_reach_only_this_pod_and_restart_only_this_watchdog(
    tmp_path: Path, shell: str
) -> None:
    environment = helper_environment(tmp_path)
    control, stubs = Path(environment["CTRL"]), Path(environment["STUB_DIR"])
    bystander = subprocess.Popen(["sleep", "30"])  # Some other process holding an old PID.
    (control / "watchdog.pid").write_text(f"{bystander.pid}\n")
    script = (
        'source "$CTRL/helpers.sh"\n'
        "podssh 'test -s /workspace/rehearsal-run/heartbeat.json'\n"
        'podscp "$CTRL/revision.json" root@$POD_HOST:/workspace/revision.json\n'
        'watchdog && cat "$CTRL/watchdog.pid" > "$CTRL/first.pid"\n'
        'watchdog && cat "$CTRL/watchdog.pid" > "$CTRL/second.pid"\n'
    )
    pids: list[int] = []
    try:
        result = subprocess.run(
            [shell, "-c", script], env=environment, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        pinned = [
            "-i",
            VALUES["POD_SSH_KEY"],
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={VALUES['KNOWN_HOSTS']}",
            "-o",
            "BatchMode=yes",
        ]
        assert (stubs / "ssh.args").read_text().splitlines() == [
            "-F",
            "/dev/null",
            "-p",
            VALUES["POD_PORT"],
            *pinned,
            f"root@{VALUES['POD_HOST']}",
            "test -s /workspace/rehearsal-run/heartbeat.json",
        ]
        assert (stubs / "scp.args").read_text().splitlines() == [
            "-F",
            "/dev/null",
            "-P",
            VALUES["POD_PORT"],
            *pinned,
            f"{control}/revision.json",
            f"root@{VALUES['POD_HOST']}:/workspace/revision.json",
        ]
        first = int((control / "first.pid").read_text())
        second = int((control / "second.pid").read_text())
        pids = [first, second]
        assert first != second and f"run A watchdog pid {second}" in result.stdout
        # A PID file naming a process that is not this Pod's watchdog never kills it.
        assert bystander.poll() is None
        deadline = time.monotonic() + 10
        while alive(first) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not alive(first) and alive(second)  # The restart replaced only its own.
    finally:
        for pid in pids:
            stop(pid)
        bystander.kill()
        bystander.wait()


# Step 9: the wait for the worker's first heartbeat --------------------------------------

LOOP_STUBS = r"""
date() { cat "$CLOCK"; }
sleep() { echo $(( $(cat "$CLOCK") + $1 )) > "$CLOCK"; }
watchdog() { BASE=$(cat "$CLOCK"); echo "watchdog $BASE" >> "$CALLS"; }
podssh() {
  case "$1" in
    "test -s /workspace/rehearsal-run/heartbeat.json") [ "$(cat "$CLOCK")" -ge "$HEARTBEAT_AT" ] ;;
    "grep -q ^WORKER_EXIT= /workspace/rehearsal/run.log") [ "$(cat "$CLOCK")" -ge "$EXIT_AT" ] ;;
    "tail -n 40 /workspace/rehearsal/run.log") echo "TAIL OF RUN LOG" ;;
    *) echo "unexpected podssh $1" >&2; return 99 ;;
  esac
}
"""


def wait_loop() -> str:
    block = blocks(9)[0]
    return block[block.index("GIVE_UP=") :]


def run_wait(
    tmp_path: Path, shell: str, *, heartbeat_at: int, exit_at: int
) -> tuple[str, list[str]]:
    clock, calls = tmp_path / "clock", tmp_path / "calls"
    clock.write_text("1000\n")
    calls.write_text("")
    environment = {
        **os.environ,
        "CLOCK": str(clock),
        "CALLS": str(calls),
        "HEARTBEAT_AT": str(heartbeat_at),
        "EXIT_AT": str(exit_at),
        "RUN": "A",
        "BASE": "1000",
    }
    # The block's last line fails when no heartbeat exists; "exit 0" is reached only if every
    # line of the block parsed and ran.
    result = subprocess.run(
        [shell, "-c", LOOP_STUBS + wait_loop() + "\nexit 0\n"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0 and result.stderr == "", result.stderr
    return result.stdout, calls.read_text().splitlines()


def test_step_9_waits_with_a_deadline_and_rebaselines_like_step_7() -> None:
    loop = wait_loop()
    assert "GIVE_UP=$(( $(date +%s) + 1200 ))" in loop
    assert "if [ $(( $(date +%s) - BASE )) -ge 300 ]; then watchdog; fi" in loop
    assert "^WORKER_EXIT=" in loop and "tail -n 40 /workspace/rehearsal/run.log" in loop
    # Only a heartbeat that exists earns the last re-baseline before step 10.
    assert loop.rstrip().endswith(
        "done\npodssh 'test -s /workspace/rehearsal-run/heartbeat.json' && watchdog"
    )
    (run,) = [text for module, argv, text in commands() if argv[:1] == ["run"]]
    assert "--output /workspace/rehearsal-run" in run
    assert "--output /workspace/rehearsal-run; echo WORKER_EXIT=\\$?'" in blocks(9)[0]
    assert "exec python3" not in blocks(9)[0]  # The shell outlives the worker to log its exit.


@pytest.mark.parametrize("shell", SHELLS)
def test_step_9_returns_once_the_heartbeat_exists(tmp_path: Path, shell: str) -> None:
    out, calls = run_wait(tmp_path, shell, heartbeat_at=1400, exit_at=10**9)
    assert "exited" not in out and "no heartbeat" not in out and "TAIL" not in out
    assert calls[0] == "watchdog 1300"  # Re-baselined after 300 s of waiting...
    assert calls[-1] == "watchdog 1400"  # ...and once more when the heartbeat appears.


@pytest.mark.parametrize("shell", SHELLS)
def test_step_9_stops_and_shows_the_log_when_the_worker_exits(tmp_path: Path, shell: str) -> None:
    out, calls = run_wait(tmp_path, shell, heartbeat_at=10**9, exit_at=1030)
    assert "run A: the worker exited before its first heartbeat:" in out
    assert "TAIL OF RUN LOG" in out
    assert calls == []  # A failed wait never re-baselines the watchdog.


@pytest.mark.parametrize("shell", SHELLS)
def test_step_9_gives_up_after_its_deadline(tmp_path: Path, shell: str) -> None:
    out, calls = run_wait(tmp_path, shell, heartbeat_at=10**9, exit_at=10**9)
    assert "run A: no heartbeat after 20 minutes:" in out and "TAIL OF RUN LOG" in out
    assert calls == ["watchdog 1300", "watchdog 1600", "watchdog 1900", "watchdog 2200"]


# Secrets never reach a command line -------------------------------------------------------

SECRET_VARIABLE = re.compile(r"\$\{?(RUNPOD_API_KEY|[A-Z0-9_]*SECRET[A-Z0-9_]*)\b")
CURL_FROM_STDIN = (
    """printf 'header = "Authorization: Bearer %s"\\n' "$RUNPOD_API_KEY" | curl --config - """
)


def test_no_secret_value_is_ever_a_process_argument() -> None:
    code = _code()
    expanded = [line for line in code if SECRET_VARIABLE.search(line)]
    # The only expansion is into the shell's builtin printf, piped to curl's stdin config.
    assert expanded and all(line.startswith(CURL_FROM_STDIN) for line in expanded), expanded
    assert all(line.count("$RUNPOD_API_KEY") == 1 for line in expanded)
    assert len([line for line in code if "curl" in line]) == len(expanded) == 2  # create, delete
    joined = "\n".join(code)
    assert "Authorization" not in joined.replace(CURL_FROM_STDIN, "")
    assert '-H "Authorization' not in joined and 'grep -c "$' not in joined
    # The leak check feeds the values to grep on stdin; printenv gets only their names.
    (leak,) = [line for line in code if "grep -cFf -" in line]
    assert leak.startswith("for name in R2_POD_SECRET_ACCESS_KEY R2_SECRET_ACCESS_KEY ")
    assert 'printenv "$name"; done | grep -cFf - "$CTRL/bootstrap.log"' in leak


def test_the_leak_check_counts_secret_values_without_naming_them(tmp_path: Path) -> None:
    (leak,) = [line for line in _code() if "grep -cFf -" in line]
    log = tmp_path / "ctrl" / "bootstrap.log"
    log.parent.mkdir()
    secrets = {
        "R2_POD_SECRET_ACCESS_KEY": "pod-secret-1",
        "R2_SECRET_ACCESS_KEY": "mac-secret-2",
        "RUNPOD_API_KEY": "rpa_key_3",
    }
    environment = {**os.environ, **secrets, "CTRL": str(log.parent)}
    for text, expected in (
        ("clean log\nnames only: R2_POD_SECRET_ACCESS_KEY\n", "0"),
        ("leaked mac-secret-2 here\nand rpa_key_3\n", "2"),
    ):
        log.write_text(text)
        result = subprocess.run(
            ["bash", "-c", leak.split("#")[0]], env=environment, capture_output=True, text=True
        )
        assert result.stdout.strip() == expected


# The binding's pinned hash -----------------------------------------------------------------


def test_the_binding_hash_is_pinned_at_review_and_never_recomputed() -> None:
    code = "\n".join(_code())
    assert "export BINDING_SHA256=<reviewed binding sha256>" in code
    assert not re.search(r"BINDING_SHA256=\$\(", code)
    assert not re.search(r"shasum[^\n|]*\$BINDING\"? *\|", code)
    assert 'printf \'%s  %s\\n\' "$BINDING_SHA256" "$BINDING" | shasum -a 256 -c -' in code


# Code identity: revision.json -----------------------------------------------------------------

REVISION_OPENING = "python3 - \"$CTRL\" <<'PY'"


def git(repository: Path, *args: str) -> str:
    environment = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.org",
        "GIT_COMMITTER_NAME": "fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.org",
    }
    return subprocess.run(
        ["git", *args], cwd=repository, env=environment, capture_output=True, text=True, check=True
    ).stdout


def write_revision(repository: Path, control: Path) -> subprocess.CompletedProcess[str]:
    control.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, "-", str(control)],
        input=heredoc(REVISION_OPENING, "PY"),
        cwd=repository,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_the_revision_names_a_clean_commit_and_the_archive_shipped(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "-q")
    (repository / "code.py").write_text("print('trained code')\n")
    (repository / ".gitignore").write_text("ignored/\n")
    git(repository, "add", ".")
    git(repository, "commit", "-q", "-m", "fixture")
    (repository / "ignored").mkdir()
    (repository / "ignored" / "notes.txt").write_text("ignored files keep a tree clean\n")
    commit = git(repository, "rev-parse", "HEAD").strip()
    control = tmp_path / "ctrl"
    result = write_revision(repository, control)
    assert result.returncode == 0, result.stderr
    archive = control / "repo.tar.gz"
    record = json.loads((control / "revision.json").read_text())
    assert record == {
        "git_commit": commit,
        "clean": True,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    with tarfile.open(archive) as bundle:
        assert "code.py" in bundle.getnames()
    # The record has exactly the shape the bootstrap and the readiness receipt accept.
    assert set(record) == {"git_commit", "clean", "archive_sha256"}
    assert re.fullmatch(r"[0-9a-f]{40}", record["git_commit"])
    # A modified or an untracked file is a dirty tree: nothing is archived or recorded.
    for dirty in ("modified", "untracked"):
        if dirty == "modified":
            (repository / "code.py").write_text("print('edited')\n")
        else:
            git(repository, "checkout", "-q", "--", "code.py")
            (repository / "scratch.py").write_text("x = 1\n")
        fresh = tmp_path / f"ctrl-{dirty}"
        refused = write_revision(repository, fresh)
        assert refused.returncode != 0 and "dirty" in refused.stderr
        assert list(fresh.iterdir()) == []


def test_the_revision_is_shipped_where_the_bootstrap_reads_it() -> None:
    code = "\n".join(_code())
    shipping = blocks(6)[0]
    assert shipping.index(REVISION_OPENING) < shipping.index("podscp")
    assert 'podscp "$CTRL/repo.tar.gz" root@$POD_HOST:/workspace/repo.tar.gz' in shipping
    assert 'podscp "$CTRL/revision.json" root@$POD_HOST:/workspace/revision.json' in shipping
    assert "git archive --format=tar.gz" not in code.replace(heredoc(REVISION_OPENING, "PY"), "")
    (line,) = [line for line in _code() if "tools/pilot_pod_bootstrap.sh" in line]
    match = re.search(r"pilot_pod_bootstrap\.sh ([^;']*)", line)
    assert match
    argv = shlex.split(_substitute(match.group(1)))
    values = dict(zip(argv[0::2], argv[1::2], strict=True))
    assert values["--revision"] == "/workspace/revision.json"
    assert values["--archive"] == "/workspace/repo.tar.gz"
    # The Pod's copy, made by the bootstrap, is pulled back and compared byte for byte.
    assert 'cmp "$CTRL/revision.json" "$CTRL/pod-revision.json"' in code
    assert "/workspace/rehearsal/revision.json" in section(9)


def test_the_runbook_explains_the_upload_rate_threshold() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "4 MB/s" not in text and "480 s" not in text
    assert "2 x 1.65 GB / (900 s - 120 s) x 2" in text and "8.46 MB/s" in text
    assert 2 * 1.65e9 / (900 - 120) * 2 == pytest.approx(8.46e6, rel=1e-3)
    assert "REQUIRED_UPLOAD_BYTES_PER_SECOND" in text and "128 MiB" in text
    assert "r2_milestones_unverified" in section(14) and "finalization.json" in section(14)


# Step 15: billing over the whole Pod lifetime -----------------------------------------------

BILLING_OPENING = 'python3 - "$CTRL/billing.json"'
HOUR = 3600


def billing_records(first: int, last: int, pod: str = "podfixture01") -> list[dict[str, Any]]:
    stamp = "%Y-%m-%dT%H:%M:%SZ"
    return [
        {
            "startTime": time.strftime(stamp, time.gmtime(start)),
            "endTime": time.strftime(stamp, time.gmtime(start + HOUR)),
            "podId": pod,
            "totalAmount": 0.6,
            "gpuAmount": 0.59,
            "diskAmount": 0.01,
        }
        for start in range(first, last + 1, HOUR)
    ]


def check_billing(
    tmp_path: Path,
    records: list[dict[str, Any]],
    *,
    buckets: int,
    origin: int,
    teardown: str,
    fetched: float,
    heartbeat: float | None,
    totals: float | None = None,
) -> subprocess.CompletedProcess[str]:
    report = {
        "schema_version": 1,
        "fetched_at_unix": fetched,
        "pod_id": "podfixture01",
        "bucket_size": "hour",
        "records": records,
        "totals": {
            "totalAmount": sum(r["totalAmount"] for r in records) if totals is None else totals
        },
    }
    (tmp_path / "billing.json").write_text(json.dumps(report))
    beat = tmp_path / "heartbeat.json"
    if heartbeat is not None:
        beat.write_text(json.dumps({"updated_at_unix": heartbeat, "pod_id": "podfixture01"}))
    arguments = [tmp_path / "billing.json", "podfixture01", origin, teardown, buckets, beat]
    return subprocess.run(
        [sys.executable, "-", *map(str, arguments)],
        input=heredoc(BILLING_OPENING, "PY"),
        capture_output=True,
        text=True,
    )


def test_step_15_fetches_every_bucket_back_to_the_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    ((argv, _),) = [row for row in by_module("tools.runpod_pod_identity") if row[0][0] == "billing"]
    args = parse("tools.runpod_pod_identity", argv, monkeypatch)
    assert args.last_buckets == int(VALUES["BUCKETS"]) and args.pod_id == VALUES["POD_ID"]
    code = "\n".join(_code())
    assert "export BUCKETS=$(( ($(date +%s) - ORIGIN) / 3600 + 2 ))" in code
    assert (
        "export TEARDOWN=$(date +%s) && printf 'export TEARDOWN=%s\\n' \"$TEARDOWN\" "
        '>> "$CTRL/run.env"' in code
    )
    # The formula always reaches the hour of ORIGIN, wherever in their hours both fall.
    for origin in (ORIGIN, ORIGIN + 1, ORIGIN + HOUR - 1):
        for elapsed in (0, 1, HOUR - 1, HOUR, 45_000, 45_000 + 7 * HOUR - 1):
            now = origin + elapsed
            buckets = elapsed // HOUR + 2
            assert now // HOUR * HOUR - (buckets - 1) * HOUR <= origin // HOUR * HOUR


def test_step_15_accepts_records_that_cover_the_pod_lifetime(tmp_path: Path) -> None:
    origin = 1_800_000_000 + 600
    teardown = origin + 44_500
    fetched = teardown + HOUR
    records = billing_records(origin // HOUR * HOUR, teardown // HOUR * HOUR)
    common = {"origin": origin, "fetched": fetched, "buckets": (fetched - origin) // HOUR + 2}
    result = check_billing(
        tmp_path, records, teardown=str(teardown), heartbeat=origin + 44_000, **common
    )
    assert result.returncode == 0, result.stderr
    assert "billing covers this Pod" in result.stdout
    summary = json.loads(result.stdout.splitlines()[0])
    assert summary["hours"] == len(records) == 13 and summary["total_usd"] == 7.8
    # Without TEARDOWN recorded or a mirrored heartbeat the check still runs.
    assert check_billing(tmp_path, records, teardown="", heartbeat=None, **common).returncode == 0


Records = list[dict[str, Any]]
Settings = dict[str, Any]


def gap(records: Records, settings: Settings) -> tuple[Records, Settings]:
    return records[:6] + records[7:], settings


def lagging(records: Records, settings: Settings) -> tuple[Records, Settings]:
    return records[:-1], settings


def default_buckets(records: Records, settings: Settings) -> tuple[Records, Settings]:
    return records, {**settings, "buckets": 3}  # The tool's default reaches back 3 hours.


def other_pod(records: Records, settings: Settings) -> tuple[Records, Settings]:
    return [*records[:-1], *billing_records(0, 0, "otherpod")], settings


def wrong_totals(records: Records, settings: Settings) -> tuple[Records, Settings]:
    return records, {**settings, "totals": 1.0}


def after_teardown(records: Records, settings: Settings) -> tuple[Records, Settings]:
    late = int(settings["fetched"]) // HOUR * HOUR
    return [*records, *billing_records(late, late)], settings


def over_cap(records: Records, settings: Settings) -> tuple[Records, Settings]:
    return [{**record, "totalAmount": 0.8} for record in records], settings


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (gap, "no record for the hours"),
        (lagging, "no record for the hours"),
        (default_buckets, "did not reach back"),
        (other_pod, "another Pod"),
        (wrong_totals, "provider totals"),
        (after_teardown, "outside this Pod's lifetime"),
        (over_cap, "$10 cap"),
    ],
)
def test_step_15_refuses_records_that_do_not_cover_the_pod(
    tmp_path: Path, edit: Any, message: str
) -> None:
    origin = 1_800_000_000 + 600
    teardown = origin + 44_500
    fetched = teardown + HOUR
    records = billing_records(origin // HOUR * HOUR, teardown // HOUR * HOUR)
    settings = {
        "origin": origin,
        "fetched": fetched,
        "buckets": (fetched - origin) // HOUR + 2,
        "teardown": str(teardown),
        "heartbeat": origin + 44_000,
    }
    records, settings = edit(list(records), settings)
    result = check_billing(tmp_path, records, **settings)
    assert result.returncode == 1 and message in result.stderr, result.stderr
    assert "billing does NOT cover this Pod's lifetime" in result.stderr
