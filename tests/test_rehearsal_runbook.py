"""Every tool command in the rehearsal runbook must run as written against the real parsers.

Shell variables get concrete values; no tool is executed. A command is parsed by calling
its module's ``main`` with ``ArgumentParser.parse_args`` intercepted, so an unknown flag,
a missing required flag or a bad choice fails here exactly as it would on the Mac or Pod.

The runbook's own shell and Python (the step 5 create guard, create request, Pod ID guard
and wait for RUNNING, the per-Pod helpers, the step 7 and step 9 launches and wait loops,
the step 8 timer copy, the revision writer, the step 13 DELETE gate and the billing check)
are extracted from this file and run against stubs, in bash and in zsh where the shell
matters. The step 7 and step 9 launches run as the Pod's shell would run them, to show that
they return at once. No block may hold a comment: an interactive zsh on macOS has
``interactivecomments`` off, so every block must also parse as one.
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
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tools import rehearsal_ops
from tools.pilot_readiness_receipt import SELF_DESTRUCT_FRESH_SECONDS
from tools.runpod_pod_identity import connection_from_pod
from tools.runpod_watchdog import (
    _SAFE_POD_ID,
    Outcome,
    Policy,
    atomic_receipt,
    build_parser,
    policy_from_args,
)

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
    "SELF_DESTRUCT_DEADLINE": str(ORIGIN + 60_000 - 180),
    "POD_HOST": "203.0.113.7",
    "POD_PORT": "40022",
    "BASE": str(NOW),
    "BUCKETS": "16",
}
SHELLS = [shell for shell in ("bash", "zsh") if shutil.which(shell)]
"""The operator's shells: the runbook's own blocks run in each one that is installed."""
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
    assert (policy.max_wall_seconds, policy.max_idle_seconds) == (60_000, 600)
    assert policy.max_cost_usd == rehearsal_ops.WATCHDOG_COST_STOP_USD == 9.99
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
    assert args.max_lifetime_seconds == 60_000
    assert args.deadline_unix == ORIGIN + 60_000 - 180
    assert "SELF_DESTRUCT_DEADLINE=$((ORIGIN + 60000 - 180))" in "\n".join(_code())
    script = BOOTSTRAP.read_text(encoding="utf-8")
    assert "LIFETIME_ARGS=(--max-lifetime-seconds 60000)" in script
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
    pipe = "| podssh_stdin '"  # The one SSH call that keeps its standard input.
    assert delivery.index("pod-env") < delivery.index(pipe) < delivery.index("umask 077")
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
    assert 'elif mkdir "$CTRL"; then' in create
    guarded = create[create.index('elif mkdir "$CTRL"; then') : create.index("\nfi\n")]
    assert '> "$CTRL/run.env"' in guarded and "already exists" in guarded
    assert create.count('"$CTRL/run.env"') == 1
    assert 'mkdir -p "$CTRL"' not in code and "mkdir -p $CTRL" not in code
    # Everything the watchdog, receipt and supervisor read lives under the per-Pod $CTRL.
    assert str(_watchdog().identity).startswith(VALUES["CTRL"] + "/")
    assert VALUES["CTRL"] == f"{VALUES['CTRL_ROOT']}/{VALUES['RUN_LOWER']}/{VALUES['POD_ID']}"


# Step 5: the Pod ID comes from the create response --------------------------------------

POD_ID_PARSER = "export POD_ID=$(python3 -c "


def pod_id_guard() -> str:
    """Step 5 from the Pod ID parser through the controller-directory guard's ``fi``.

    The guard sits inside the create branch, so it is dedented to run on its own.
    """
    create = blocks(5)[0]
    start = create.rindex("\n", 0, create.index(POD_ID_PARSER)) + 1
    indent = create[start : create.index(POD_ID_PARSER)]
    end = create.index(f"\n{indent}fi\n", start) + len(indent) + 4
    return textwrap.dedent(create[start:end])


def test_pod_id_is_parsed_from_the_create_response_never_copied_by_hand() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    create = blocks(5)[0]
    assert "<id printed above>" not in text and "export POD_ID=<" not in text
    assert "tee " not in create  # The response goes to its file only; the parser reads it.
    request = create.index("curl --config -")
    response = '-o "$LAUNCH/create-response.json"'
    assert request < create.index(response) < create.index(POD_ID_PARSER)
    (parser,) = [line for line in create.splitlines() if line.strip().startswith(POD_ID_PARSER)]
    assert parser.endswith(' "$LAUNCH/create-response.json")')
    # The parser accepts exactly the Pod IDs the watchdog accepts.
    assert _SAFE_POD_ID.pattern.removesuffix("\\Z") in parser and "re.fullmatch(" in parser
    # Every later use of POD_ID sits behind the guard's check that it is set.
    guard = pod_id_guard()
    assert guard.startswith(POD_ID_PARSER) and guard.endswith("\nfi\n")
    assert guard.index('if [ -z "$POD_ID" ]; then') < guard.index('elif mkdir "$CTRL"; then')


def run_pod_id_guard(
    tmp_path: Path, shell: str, response: str, *, existing: bool = False
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    root = tmp_path / "runs"
    launch = root / "a" / "launch-1800000000"
    launch.mkdir(parents=True)
    (launch / "create-body.json").write_text('{"name": "rehearsal-a40-s-a"}\n')
    (launch / "create-response.json").write_text(response)
    (launch / "create-http-code").write_text("201\n")
    control = root / "a" / "podfixture01"
    if existing:
        control.mkdir()
        (control / "run.env").write_text("export ORIGIN=1700000000 POD_ID=podfixture01\n")
        (control / "watchdog.pid").write_text("4242\n")
    environment = {
        **os.environ,
        "CTRL_ROOT": str(root),
        "RUN_LOWER": "a",
        "RUN": "A",
        "LAUNCH": str(launch),
        "ORIGIN": "1800000000",
    }
    environment.pop("POD_ID", None)
    environment.pop("CTRL", None)
    script = pod_id_guard() + 'printf "CTRL=%s\\nPOD_ID=%s\\n" "$CTRL" "$POD_ID"\n'
    result = subprocess.run(
        [shell, "-c", script], env=environment, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    return result, launch, control


@pytest.mark.parametrize("shell", SHELLS)
def test_step_5_takes_the_pod_id_from_the_response_and_writes_run_env(
    tmp_path: Path, shell: str
) -> None:
    response = '{"id": "podfixture01", "name": "rehearsal-a40-s-a", "desiredStatus": "RUNNING"}'
    result, launch, control = run_pod_id_guard(tmp_path, shell, response)
    assert not launch.exists()
    assert (control / "run.env").read_text() == (
        "export ORIGIN=1800000000 POD_ID=podfixture01 SELF_DESTRUCT_DEADLINE=1800059820\n"
    )
    assert sorted(path.name for path in control.iterdir()) == [
        "create-body.json",
        "create-http-code",
        "create-response.json",
        "run.env",
    ]
    assert f"run A: Pod podfixture01, controller directory {control}" in result.stdout
    assert f"CTRL={control}\nPOD_ID=podfixture01\n" in result.stdout


@pytest.mark.parametrize("shell", SHELLS)
def test_step_5_refusal_names_this_pod_and_leaves_the_existing_directory_alone(
    tmp_path: Path, shell: str
) -> None:
    result, launch, control = run_pod_id_guard(
        tmp_path, shell, '{"id": "podfixture01"}', existing=True
    )
    message = result.stdout
    assert "already exists, so it is NOT this Pod's" in message and "never reused" in message
    # The Pod to terminate is the one this create response names, and no other.
    assert (
        f"terminate Pod podfixture01, the one {launch}/create-response.json names, "
        "with step 13 in this terminal (TERMINATE=yes); never delete any other Pod." in message
    )
    # The existing directory is untouched, and this terminal no longer points at it.
    assert sorted(path.name for path in control.iterdir()) == ["run.env", "watchdog.pid"]
    assert (control / "run.env").read_text() == "export ORIGIN=1700000000 POD_ID=podfixture01\n"
    assert sorted(path.name for path in launch.iterdir()) == [
        "create-body.json",
        "create-http-code",
        "create-response.json",
    ]
    assert f"CTRL={launch}\n" in message


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    "response",
    [
        '{"error": "There are no longer any instances available"}',
        '{"id": ""}',
        '{"id": null}',
        '{"id": 12345}',
        '{"id": "../b/podfixture02"}',
        '{"id": "pod fixture"}',
        '{"id": "podfixture01\\n"}',
        '{"id": "-podfixture01"}',
        '[{"id": "podfixture01"}]',
        "<html>Bad Gateway</html>",
        "",
    ],
)
def test_step_5_refuses_a_missing_or_malformed_pod_id(
    tmp_path: Path, shell: str, response: str
) -> None:
    result, launch, control = run_pod_id_guard(tmp_path, shell, response)
    assert "names no valid Pod id" in result.stdout
    # A missing ID never means that no Pod exists: the owner checks the console.
    assert "A Pod MAY STILL EXIST" in result.stdout
    assert "a Pod named rehearsal-a40-s-a created at or after " in result.stdout
    assert "terminate any such Pod there, and to decide before any retry" in result.stdout
    assert "Never delete a Pod from this terminal by name or by guess" in result.stdout
    assert "created no Pod" not in result.stdout
    assert "terminate Pod" not in result.stdout  # No ID, so it names no Pod to delete.
    assert f"CTRL={launch}\nPOD_ID=\n" in result.stdout
    assert sorted(path.name for path in launch.parent.iterdir()) == [launch.name]
    assert not (launch / "run.env").exists()


# Step 5: one Pod per run at a time, and a create whose reply failed ---------------------

CREATE_STUBS = r"""
curl() {
  cat > "$STUB_DIR/curl.stdin"
  printf '%s\n' "$@" > "$STUB_DIR/curl.args"
  echo create >> "$STUB_DIR/curl.calls"
  OUT=
  while [ $# -gt 0 ]; do if [ "$1" = -o ]; then OUT=$2; fi; shift; done
  printf '%s' "$RESPONSE" > "$OUT"
  printf '%s\n' "$HTTP_CODE"
  return "$CURL_EXIT"
}
"""
CREATED = '{"id": "podfixture01", "name": "rehearsal-a40-s-a", "status": "PROVISIONING"}'


def create_block() -> str:
    (block,) = [block for block in blocks(5) if "-X POST" in block]
    return block


def run_create(
    tmp_path: Path,
    shell: str,
    *,
    response: str = CREATED,
    http_code: str = "201",
    curl_exit: int = 0,
    pod_id: str | None = None,
    pastes: int = 1,
) -> tuple[str, Path, list[str]]:
    """Paste the create block ``pastes`` times into one shell; curl is a stub."""
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    root = tmp_path / "runs"
    environment = {
        **os.environ,
        "STUB_DIR": str(stubs),
        "CTRL_ROOT": str(root),
        "RUN": "A",
        "RUN_LOWER": "a",
        "RUNPOD_API_KEY": "rpa_fixture_key",
        "RESPONSE": response,
        "HTTP_CODE": http_code,
        "CURL_EXIT": str(curl_exit),
    }
    for name in ("POD_ID", "CTRL", "ORIGIN", "LAUNCH", "SELF_DESTRUCT_DEADLINE", "TERMINATE"):
        environment.pop(name, None)
    if pod_id is not None:
        environment.update(POD_ID=pod_id, CTRL=str(root / "a" / pod_id))
    script = (
        CREATE_STUBS
        + create_block() * pastes
        + 'printf "POD_ID=[%s]\\nCTRL=[%s]\\n" "${POD_ID:-}" "${CTRL:-}"\n'
    )
    # The body is read from configs/ in the checkout; everything written goes under tmp_path.
    result = subprocess.run(
        [shell, "-c", script], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    calls = stubs / "curl.calls"
    return result.stdout, root / "a", calls.read_text().splitlines() if calls.exists() else []


@pytest.mark.parametrize("shell", SHELLS)
def test_step_5_pasted_twice_creates_one_pod_and_records_the_http_code(
    tmp_path: Path, shell: str
) -> None:
    out, runs, calls = run_create(tmp_path, shell, pastes=2)
    control = runs / "podfixture01"
    assert calls == ["create"]  # The second paste sent nothing.
    assert f"run A: Pod podfixture01, controller directory {control}" in out
    assert (
        "run A: NOT creating a Pod: this terminal already names Pod podfixture01 "
        f"({control}), which may still bill" in out
    )
    assert "Confirm Pod podfixture01 gone with step 13 in this terminal." in out
    assert out.endswith(f"POD_ID=[podfixture01]\nCTRL=[{control}]\n")
    assert sorted(path.name for path in runs.iterdir()) == ["podfixture01"]
    assert (control / "create-http-code").read_text() == "201\n"
    assert json.loads((control / "create-body.json").read_text())["name"] == "rehearsal-a40-s-a"
    origin = re.search(r"ORIGIN=(\d+) ", (control / "run.env").read_text())
    assert origin
    launch = runs / f"launch-{origin.group(1)}"
    stubs = tmp_path / "stubs"
    assert (stubs / "curl.args").read_text().splitlines() == [
        "--config",
        "-",
        "-sS",
        "--fail-with-body",
        "-X",
        "POST",
        "https://api.runpod.io/v2/pods",
        "-H",
        "Content-Type: application/json",
        "--data",
        f"@{launch}/create-body.json",
        "-o",
        f"{launch}/create-response.json",
        "-w",
        "%{http_code}\\n",
    ]
    # The key reaches curl on its standard input only.
    stdin = (stubs / "curl.stdin").read_text()
    assert stdin == 'header = "Authorization: Bearer rpa_fixture_key"\n'
    assert "rpa_fixture_key" not in (stubs / "curl.args").read_text()


@pytest.mark.parametrize("shell", SHELLS)
def test_step_5_refuses_to_create_while_this_terminal_names_a_pod(
    tmp_path: Path, shell: str
) -> None:
    out, runs, calls = run_create(tmp_path, shell, pod_id="podearlier01")
    assert calls == [] and not runs.exists()
    assert "run A: NOT creating a Pod: this terminal already names Pod podearlier01" in out
    assert "Confirm Pod podearlier01 gone with step 13 in this terminal." in out
    assert "run unset POD_ID and paste this block again" in out
    assert out.endswith(f"POD_ID=[podearlier01]\nCTRL=[{runs}/podearlier01]\n")


OPEN_RUN_ENV = "export ORIGIN=1700000000 POD_ID=podold01 SELF_DESTRUCT_DEADLINE=1700059820\n"
TEARDOWN_LINE = "export TEARDOWN=1700040000\n"


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    ("directories", "open_one"),
    [
        ({"podold01": OPEN_RUN_ENV}, "podold01"),
        ({"podold01": OPEN_RUN_ENV + "export POD_HOST=203.0.113.9 POD_PORT=1\n"}, "podold01"),
        ({"podold01": None}, "podold01"),
        ({"launch-1700000000": None}, "launch-1700000000"),  # A create never confirmed.
        ({"podold01": OPEN_RUN_ENV + TEARDOWN_LINE, "podold02": OPEN_RUN_ENV}, "podold02"),
    ],
)
def test_step_5_refuses_to_create_while_another_pod_of_the_run_has_no_teardown(
    tmp_path: Path, shell: str, directories: dict[str, str | None], open_one: str
) -> None:
    runs = tmp_path / "runs" / "a"
    for name, run_env in directories.items():
        (runs / name).mkdir(parents=True)
        if run_env is not None:
            (runs / name / "run.env").write_text(run_env)
    out, runs, calls = run_create(tmp_path, shell)
    assert calls == []
    assert f"run A: NOT creating a Pod: no TEARDOWN is recorded in {runs}/{open_one}. " in out
    assert "End each Pod with step 13 in its own terminal" in out
    assert "POD_ID=[]\nCTRL=[]\n" in out
    assert sorted(path.name for path in runs.iterdir()) == sorted(directories)  # Nothing new.
    for name, run_env in directories.items():
        if run_env is not None:
            assert (runs / name / "run.env").read_text() == run_env


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("settled", ["podold01", "launch-1700000000"])
def test_step_5_creates_once_every_earlier_pod_has_its_teardown(
    tmp_path: Path, shell: str, settled: str
) -> None:
    runs = tmp_path / "runs" / "a"
    (runs / settled).mkdir(parents=True)
    run_env = (OPEN_RUN_ENV if settled.startswith("pod") else "") + TEARDOWN_LINE
    (runs / settled / "run.env").write_text(run_env)
    out, runs, calls = run_create(tmp_path, shell)
    assert calls == ["create"] and "NOT creating" not in out
    assert sorted(path.name for path in runs.iterdir()) == sorted([settled, "podfixture01"])


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    ("response", "code", "curl_exit"),
    [
        ("<html>502 Bad Gateway</html>", "502", 22),  # A gateway error after the create.
        ("", "000", 52),  # The connection dropped: no reply at all.
        ('{"error": "internal error"}', "500", 22),
    ],
)
def test_step_5_failed_reply_warns_that_a_pod_may_exist_and_blocks_a_retry(
    tmp_path: Path, shell: str, response: str, code: str, curl_exit: int
) -> None:
    out, runs, calls = run_create(
        tmp_path, shell, response=response, http_code=code, curl_exit=curl_exit, pastes=2
    )
    assert calls == ["create"]  # Pasting again never retries on its own.
    (launch,) = list(runs.iterdir())
    assert launch.name.startswith("launch-")
    assert (
        f"run A: the create request ended with HTTP {code} and "
        f"{launch}/create-response.json names no valid Pod id" in out
    )
    assert "A Pod MAY STILL EXIST" in out
    assert (
        "look in the Runpod console for a Pod named rehearsal-a40-s-a created at or after " in out
    )
    assert "terminate any such Pod there, and to decide before any retry" in out
    assert f"run A: NOT creating a Pod: no TEARDOWN is recorded in {launch}. " in out
    assert out.endswith(f"POD_ID=[]\nCTRL=[{launch}]\n")
    assert sorted(path.name for path in launch.iterdir()) == [
        "create-body.json",
        "create-http-code",
        "create-response.json",
    ]
    assert (launch / "create-http-code").read_text() == f"{code}\n"
    assert (launch / "create-response.json").read_text() == response
    # The settled launch is recorded as step 5 says, and the next paste creates again.
    settle = 'printf \'export TEARDOWN=%s\\n\' "$(date +%s)" >> "$LAUNCH/run.env"'
    assert f"`{settle}`" in section(5)
    (launch / "run.env").write_text(TEARDOWN_LINE)
    (tmp_path / "stubs" / "curl.calls").unlink()
    out, runs, calls = run_create(tmp_path, shell)
    assert calls == ["create"] and "run A: Pod podfixture01" in out


def test_step_5_create_block_guards_before_it_sends_anything() -> None:
    create = create_block()
    first, second = create.splitlines()[:2]
    assert first.startswith('OPEN_PODS=$(find "$CTRL_ROOT/$RUN_LOWER" -mindepth 1 -maxdepth 1')
    assert "grep -qs '^export TEARDOWN=' \"$DIR/run.env\"" in first
    assert "*" not in first  # No glob: zsh would stop the paste on "no matches found".
    assert second == 'if [ -n "${POD_ID:-}" ]; then'
    assert create.index('elif [ -n "$OPEN_PODS" ]; then') < create.index("\nelse\n")
    assert create.index("\nelse\n") < create.index("export ORIGIN=$(date +%s)")
    assert create.index("export ORIGIN=") < create.index("curl --config -")
    assert create.rstrip().endswith("\n  fi\nfi")
    # The step 13 teardown record is what lets a later create through.
    assert 'printf \'export TEARDOWN=%s\\n\' "$TEARDOWN" >> "$CTRL/run.env"' in termination()


# Step 5: the bounded wait for RUNNING ----------------------------------------------------

WATCHDOG_STUBS = r"""
date() { cat "$CLOCK"; }
sleep() { echo $(( $(cat "$CLOCK") + $1 )) > "$CLOCK"; }
watchdog() {
  if [ -n "${WATCHDOG_FAILS:-}" ]; then
    echo "watchdog failed $(cat "$CLOCK")" >> "$CALLS"; return 1
  fi
  BASE=$(cat "$CLOCK"); echo "watchdog $BASE" >> "$CALLS"; echo 4242 > "$CTRL/watchdog.pid"
}
ps() {
  if [ "$1" = -A ]; then
    echo "1 /sbin/launchd"
    if [ -n "${PROCESSES:-}" ]; then printf '%s\n' "$PROCESSES"; fi
  elif [ "$2" = 4242 ] && [ "$4" = ppid= ]; then
    echo 4241
  elif [ "$2" = 4242 ]; then
    echo "python -m tools.runpod_watchdog --mode rehearsal --identity $CTRL/pod-identity.json"
  else
    echo "sleep 30"
  fi
}
"""
"""The clock, this Pod's watchdog (PID 4242, started by ``uv`` as PID 4241) and ``ps`` for
every wait loop. ``ps -A`` lists the processes in ``$PROCESSES`` besides launchd."""

KEEPALIVE_GONE = (
    'keepalive; if [ $? -eq 3 ]; then echo "run $RUN: stop waiting: the Pod is gone '
    '(watchdog receipt above); follow step 14"; break; fi'
)
"""How every wait loop calls keepalive: status 3 means the watchdog has ended this Pod."""
GONE_LINE = "run A: stop waiting: the Pod is gone (watchdog receipt above); follow step 14"
POD_GONE = "STOP run A: POD GONE: the watchdog of Pod podfixture01 ended for good with "
TERMINATED = Outcome("terminated", "idle_gpu_limit_delete_204_get_404", "podfixture01", 990.0)
"""The final receipt of a watchdog that deleted this Pod, as the watchdog writes it."""


def keepalive() -> str:
    """The ``keepalive`` helper, as written into helpers.sh."""
    match = re.search(r"\nkeepalive\(\) \{\n.*?\n\}\n", "\n" + helpers(), re.S)
    assert match, "the helpers define keepalive"
    return match.group(0).lstrip("\n")


def loop_stubs(extra: str) -> str:
    return WATCHDOG_STUBS + keepalive() + extra


def write_receipt(control: Path, receipt: Outcome | str | None) -> str | None:
    """Leave a watchdog receipt in ``control``; returns its text to compare afterwards."""
    if receipt is None:
        return None
    path = control / "watchdog-receipt.json"
    if isinstance(receipt, Outcome):
        atomic_receipt(path, receipt)  # Exactly what tools.runpod_watchdog writes.
    else:
        path.write_text(receipt)
    return path.read_text()


RUNNING_STUBS = r"""
uv() {
  echo "$*" >> "$CAPTURES"
  NOW=$(cat "$CLOCK")
  if [ "$NOW" -lt "$STARTED_AT" ]; then
    printf '%s\n' "$CREATED_CAPTURE" > "$CTRL/pod-connection.json"
  elif [ "$NOW" -lt "$LATER_AT" ]; then
    printf '%s\n' "$STARTED_CAPTURE" > "$CTRL/pod-connection.json"
  else
    printf '%s\n' "$LATER_CAPTURE" > "$CTRL/pod-connection.json"
  fi
}
"""
"""The capture tool: the create reply's shape until ``STARTED_AT``, then the started
container's, then (from ``LATER_AT``) a later shape."""
CREATE_REPLY: dict[str, Any] = {
    "id": "podfixture01",
    "name": "rehearsal-a40-s-a",
    "status": "RUNNING",
    "actions": ["stop", "restart", "terminate"],
    "cost": 0.59,
    "createdAt": "2027-01-15T08:00:00.704Z",
    "startedAt": "2027-01-15T08:00:00.7Z",
    "image": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
    "disk": 40,
    "ports": ["22/tcp"],
    "mounts": {},
    "runtime": None,
    "ssh": {
        "direct": None,
        "proxy": {"host": "ssh.runpod.io", "port": 22, "username": "podfixture01-64411b02"},
    },
}
"""The shape of the pilot's real create reply: ``RUNNING`` before the container is up, with
no runtime and no direct SSH."""
HTTP_PORT = {"ip": "100.65.27.2", "private": 19123, "public": 60084, "type": "http"}
SSH_PORT = {
    "ip": VALUES["POD_HOST"],
    "private": 22,
    "public": int(VALUES["POD_PORT"]),
    "type": "tcp",
}
DIRECT = {"host": VALUES["POD_HOST"], "port": int(VALUES["POD_PORT"]), "username": "root"}


def captured(ports: list[dict[str, Any]] | None, direct: dict[str, Any] | None) -> str:
    """``pod-connection.json`` as ``tools.runpod_pod_identity`` writes it for one reply."""
    reply = {**CREATE_REPLY, "ssh": {**CREATE_REPLY["ssh"], "direct": direct}}
    if ports is not None:
        reply["runtime"] = {"uptime": 30, "ports": ports}
    return json.dumps(connection_from_pod(reply, captured_at=float(NOW)))


CREATED = captured(None, None)
STARTED = captured([HTTP_PORT, SSH_PORT], DIRECT)  # As the pilot's started Pod was captured.
PROXY_ONLY = captured([HTTP_PORT], None)  # Up, with no public TCP port 22 and no direct SSH.
PORT_FIRST = captured([HTTP_PORT, SSH_PORT], None)  # Port 22 mapped, direct address not yet.
INCOMPLETE = captured([HTTP_PORT, SSH_PORT], {"host": None, "port": 40022})


def running_wait() -> str:
    (block,) = [block for block in blocks(5) if '"$CTRL/pod-connection.json"; STATE=$?' in block]
    return block


def test_the_wait_for_running_is_bounded_and_rebaselines_the_watchdog() -> None:
    loop = running_wait()
    assert loop.startswith("GIVE_UP=$(( $(date +%s) + 1200 )); PROXY_ONLY=0\nwhile :; do\n")
    body = loop[loop.index("; do\n") : loop.index("\ndone\n")]
    assert "tools.runpod_pod_identity capture" in body  # Captured again inside the loop.
    assert body.rstrip().endswith("\n  " + KEEPALIVE_GONE)
    assert '"status") == "RUNNING"' in loop and 'd.get("host")' in loop and 'd.get("port")' in loop
    # Proxy-only needs a started container (runtime ports) with no public TCP port 22.
    assert 'c.get("ssh_proxy_only") is True' in loop
    assert 'p=c.get("runtime_ports"); up=isinstance(p, list) and len(p) > 0' in loop
    assert 'e.get("type") == "tcp" and e.get("private") == 22 and bool(e.get("public"))' in loop
    assert "proxy=running and up and not ssh22 and " in loop
    identity = (ROOT / "tools/runpod_pod_identity.py").read_text()
    assert '"ssh_proxy_only": direct is None' in identity
    assert '"runtime_ports": None if runtime is None else runtime.get("ports")' in identity
    assert 'if [ "$PROXY_ONLY" -ge 3 ]; then' in body
    # The wait sits between the helpers (which start the watchdog) and the host-key pin.
    assert blocks(5).index(loop) == 2 and "ssh-keyscan" in blocks(5)[3]
    assert "re-run the capture until it does" not in section(5)
    assert "Runpod reports `RUNNING` from the create reply on, before the container is up" in (
        section(5)
    )
    # A 20-minute start is a fifth of run A's setup allowance; the bootstrap keeps its 45.
    assert 1200 + 2700 < 100 * 60


def test_the_create_reply_fixture_is_the_shape_that_used_to_read_as_proxy_only() -> None:
    created = json.loads(CREATED)
    # RUNNING, no runtime, no direct SSH: before the fix three of these stopped the wait.
    assert created["status"] == "RUNNING" and created["ssh_proxy_only"] is True
    assert created["runtime_ports"] is None and created["ssh_direct"] is None
    started = json.loads(STARTED)
    assert started["ssh_proxy_only"] is False and started["ssh_direct"]["port"] == 40022
    assert started["runtime_ports"] == [HTTP_PORT, SSH_PORT]


def run_running_wait(
    tmp_path: Path,
    shell: str,
    *,
    started_at: int,
    started: str = STARTED,
    later: str | None = None,
    later_at: int = 10**9,
    watchdog_pid: str | None = "4242",
    receipt: Outcome | None = None,
) -> tuple[str, list[str], list[str]]:
    clock, calls, captures = tmp_path / "clock", tmp_path / "calls", tmp_path / "captures"
    clock.write_text("1000\n")
    calls.write_text("")
    captures.write_text("")
    control = tmp_path / "ctrl"
    control.mkdir()
    if watchdog_pid is not None:
        (control / "watchdog.pid").write_text(f"{watchdog_pid}\n")
    receipt_text = write_receipt(control, receipt)
    # The capture right after the create request reads the create reply's shape.
    (control / "pod-connection.json").write_text(CREATED + "\n")
    environment = {
        **os.environ,
        "CLOCK": str(clock),
        "CALLS": str(calls),
        "CAPTURES": str(captures),
        "STARTED_AT": str(started_at),
        "LATER_AT": str(later_at),
        "CREATED_CAPTURE": CREATED,
        "STARTED_CAPTURE": started,
        "LATER_CAPTURE": started if later is None else later,
        "CTRL": str(control),
        "POD_ID": VALUES["POD_ID"],
        "RUN": "A",
        "BASE": "1000",
    }
    result = subprocess.run(
        [shell, "-c", loop_stubs(RUNNING_STUBS) + running_wait() + "\nexit 0\n"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0 and result.stderr == "", result.stderr
    if receipt_text is not None:
        assert (control / "watchdog-receipt.json").read_text() == receipt_text
    return result.stdout, calls.read_text().splitlines(), captures.read_text().splitlines()


NOT_UP = "RUNNING without direct SSH: container not up yet\n"


@pytest.mark.parametrize("shell", SHELLS)
def test_the_wait_for_running_returns_once_the_pod_has_direct_ssh(
    tmp_path: Path, shell: str
) -> None:
    out, calls, captures = run_running_wait(tmp_path, shell, started_at=1400)
    assert out.endswith("RUNNING with direct SSH\n") and "not RUNNING" not in out
    assert "WARNING" not in out and "proxy SSH only" not in out
    assert out.count(NOT_UP) == 20  # Checked at 1000 and after each capture until 1400.
    assert calls == ["watchdog 1300"]  # Re-baselined after 300 s of waiting.
    control = tmp_path / "ctrl"
    assert len(captures) == 20 and set(captures) == {
        "run --locked python -m tools.runpod_pod_identity capture --pod-id podfixture01 "
        f"--identity {control}/pod-identity.json --connection {control}/pod-connection.json"
    }


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    ("started", "later_at"),
    [
        (STARTED, 10**9),  # Up with direct SSH at once.
        (PORT_FIRST, 1200),  # Port 22 mapped first, the direct address 80 s later.
    ],
    ids=["direct-at-once", "port-22-first"],
)
def test_the_wait_for_running_carries_on_while_runpod_says_running_before_the_container(
    tmp_path: Path, shell: str, started: str, later_at: int
) -> None:
    # Runpod reports RUNNING from the create reply on: that shape persists for 120 s while
    # the image is pulled, then the container is up. A healthy Pod is never told to go.
    out, calls, captures = run_running_wait(
        tmp_path, shell, started_at=1120, started=started, later=STARTED, later_at=later_at
    )
    assert out.count(NOT_UP) == 6  # 1000 to 1100: the create reply's shape.
    assert "proxy SSH only" not in out and "terminate" not in out
    assert out.endswith("RUNNING with direct SSH\n")
    assert len(captures) == (6 if later_at > 10**8 else 10) and calls == []


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    ("started_at", "started"),
    [
        (10**9, STARTED),  # The container never comes up.
        (1100, INCOMPLETE),  # Up with an incomplete direct address.
        (1100, PORT_FIRST),  # Port 22 mapped, but no direct address ever.
    ],
    ids=["never-up", "incomplete-direct", "no-direct-ever"],
)
def test_the_wait_for_running_gives_up_after_20_minutes(
    tmp_path: Path, shell: str, started_at: int, started: str
) -> None:
    out, calls, captures = run_running_wait(tmp_path, shell, started_at=started_at, started=started)
    assert (
        "run A: Pod podfixture01 is not RUNNING with direct SSH after 20 minutes; "
        "terminate it (step 13, TERMINATE=yes)" in out
    )
    assert "proxy SSH only" not in out
    assert calls == ["watchdog 1300", "watchdog 1600", "watchdog 1900", "watchdog 2200"]
    assert len(captures) == 60


@pytest.mark.parametrize("shell", SHELLS)
def test_the_wait_for_running_stops_early_for_a_started_pod_with_proxy_ssh_only(
    tmp_path: Path, shell: str
) -> None:
    # The create reply's shape until 1100, then a container that is up with runtime ports
    # but no public TCP port 22 and no direct SSH: a Pod without a public IP.
    out, calls, captures = run_running_wait(tmp_path, shell, started_at=1100, started=PROXY_ONLY)
    assert (
        "run A: Pod podfixture01 is RUNNING with proxy SSH only: it has no public IP, so no "
        "file can reach it; terminate it (step 13, TERMINATE=yes)" in out
    )
    assert "after 20 minutes" not in out
    # Seen up behind the proxy at 1100, 1120 and 1140: three captures in a row.
    assert out.count(NOT_UP) == 5 and out.count("RUNNING with proxy SSH only\n") == 3
    assert len(captures) == 7 and calls == []


@pytest.mark.parametrize("shell", SHELLS)
def test_the_wait_for_running_allows_a_direct_address_that_comes_late(
    tmp_path: Path, shell: str
) -> None:
    # Two proxy-only captures, then the direct address: the wait carries on and succeeds.
    out, calls, captures = run_running_wait(
        tmp_path, shell, started_at=1100, started=PROXY_ONLY, later=STARTED, later_at=1140
    )
    assert out.count("RUNNING with proxy SSH only\n") == 2
    assert out.endswith("RUNNING with direct SSH\n")
    assert "terminate" not in out and len(captures) == 7


@pytest.mark.parametrize("shell", SHELLS)
def test_the_wait_for_running_warns_and_restarts_a_dead_watchdog(
    tmp_path: Path, shell: str
) -> None:
    # watchdog.pid names a process that is not this Pod's watchdog.
    out, calls, _ = run_running_wait(tmp_path, shell, started_at=1400, watchdog_pid="999")
    assert out.count("WARNING run A: NO LIVE WATCHDOG guards Pod podfixture01") == 1
    assert calls == ["watchdog 1020", "watchdog 1320"]  # Restarted on the first pass.


# The per-Pod helpers file ----------------------------------------------------------------

HELPERS_OPENING = """[ -e "$CTRL/helpers.sh" ] || cat > "$CTRL/helpers.sh" <<'EOF'"""


def helpers() -> str:
    return heredoc(HELPERS_OPENING, "EOF")


def test_helpers_are_written_once_per_pod_and_sourced_by_every_terminal() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    code = "\n".join(_code())
    body = helpers()
    for name in ("watchdog() {", "keepalive() {", "podssh() {", "podssh_stdin() {", "podscp() {"):
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
# It writes $STUB_RECEIPT to --receipt as its first poll would, and waits $STUB_LATE_STEPS
# steps of 0.2 s before it writes either.
echo "$$" >> "$STUB_DIR/uv.pids"
trap 'exit 0' TERM
step=0
while [[ $step -lt ${STUB_LATE_STEPS:-0} ]]; do sleep 0.2; step=$((step + 1)); done
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --pidfile ]]; then echo "$$" > "$2"; fi
  if [[ "$1" == --receipt && -n "${STUB_RECEIPT:-}" ]]; then echo "$STUB_RECEIPT" > "$2"; fi
  shift
done
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
        "POD_ID": VALUES["POD_ID"],
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
        # keepalive's real ps checks recognise the live watchdog and see no stray one: no
        # warning, no restart.
        quiet = subprocess.run(
            [shell, "-c", 'source "$CTRL/helpers.sh"\nkeepalive; echo "keepalive $?"\n'],
            env={**environment, "BASE": str(int(time.time()))},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert quiet.stdout == "keepalive 0\n" and quiet.stderr == "", quiet.stderr
        assert int((control / "watchdog.pid").read_text()) == second
    finally:
        for pid in pids:
            stop(pid)
        bystander.kill()
        bystander.wait()


def run_helper_script(
    shell: str, script: str, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Source helpers.sh, make its 1 s waits short, then run ``script``."""
    return subprocess.run(
        [shell, "-c", 'source "$CTRL/helpers.sh"\nsleep() { command sleep 0.02; }\n' + script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )


def wait_until(predicate: Any, seconds: float = 10) -> bool:
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.05)
    return bool(predicate())


def stub_watchdog(
    environment: dict[str, str], pidfile: Path, *wrapper: str
) -> subprocess.Popen[bytes]:
    """A stub watchdog of this Pod writing ``pidfile``, run under ``wrapper`` if given."""
    control, stubs = Path(environment["CTRL"]), Path(environment["STUB_DIR"])
    command = [str(stubs / "uv"), "run", "--locked", "python", "-m", "tools.runpod_watchdog"]
    command += ["--identity", f"{control}/pod-identity.json", "--pidfile", str(pidfile)]
    return subprocess.Popen([*wrapper, *command], env=environment)


@pytest.mark.parametrize("shell", SHELLS)
def test_a_watchdog_restart_keeps_the_receipt_it_replaces(tmp_path: Path, shell: str) -> None:
    environment = helper_environment(tmp_path)
    control = Path(environment["CTRL"])
    verdict = write_receipt(control, TERMINATED)
    # The new watchdog's first poll finds the Pod gone and records only that.
    later = json.dumps({"state": "absent", "reason": "pod_get_404", "pod_id": "podfixture01"})
    result = run_helper_script(shell, "watchdog\n", {**environment, "STUB_RECEIPT": later})
    pids = [int(pid) for pid in (Path(environment["STUB_DIR"]) / "uv.pids").read_text().split()]
    try:
        assert result.returncode == 0, result.stderr
        base = re.search(r"idle baseline (\d+)\n", result.stdout)
        assert base, result.stdout
        (kept,) = control.glob("watchdog-receipt.json.prev-*")
        assert kept.name == f"watchdog-receipt.json.prev-{base.group(1)}"
        assert kept.read_text() == verdict  # Why the last watchdog deleted the Pod.
        assert (control / "watchdog-receipt.json").read_text() == later + "\n"
    finally:
        for pid in pids:
            stop(pid)


@pytest.mark.parametrize("shell", SHELLS)
def test_a_watchdog_that_starts_too_late_is_stopped_and_the_old_one_kept(
    tmp_path: Path, shell: str
) -> None:
    environment = helper_environment(tmp_path)
    control, stubs = Path(environment["CTRL"]), Path(environment["STUB_DIR"])
    pidfile = control / "watchdog.pid"
    old = stub_watchdog(environment, pidfile)
    late: list[int] = []
    try:
        assert wait_until(lambda: pidfile.exists() and pidfile.read_text().strip() == str(old.pid))
        # The new watchdog would write its pidfile only after 60 s, past the helper's wait.
        result = run_helper_script(
            shell, 'watchdog; echo "watchdog $?"\n', {**environment, "STUB_LATE_STEPS": "300"}
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.endswith(
            f"run A: the new watchdog did not start within 60 s and was stopped; "
            f"see {control}/watchdog.log\nwatchdog 1\n"
        )
        late = [int(pid) for pid in (stubs / "uv.pids").read_text().split() if int(pid) != old.pid]
        assert len(late) == 1
        # The process the helper just started is gone, so it can never write its pidfile
        # later and leave an untracked watchdog with an old idle baseline.
        assert wait_until(lambda: not alive(late[0])), "the late watchdog was left running"
        # The watchdog that guarded the Pod still runs and is still the one on record.
        assert old.poll() is None and pidfile.read_text().strip() == str(old.pid)
    finally:
        for pid in late:
            stop(pid)
        old.terminate()
        old.wait()


@pytest.mark.parametrize("shell", SHELLS)
def test_keepalive_warns_about_a_watchdog_the_pidfile_does_not_name(
    tmp_path: Path, shell: str
) -> None:
    environment = helper_environment(tmp_path)
    control = Path(environment["CTRL"])
    pidfile, strayfile = control / "watchdog.pid", control / "stray.pid"
    # The named watchdog runs under a parent whose command line names the Pod too, as
    # "uv run" does; an older watchdog of this Pod still runs unnamed.
    parent = stub_watchdog(environment, pidfile, "bash", "-c", 'bash "$0" "$@"; exit 0')
    stray = stub_watchdog(environment, strayfile)
    pids = [parent.pid, stray.pid]
    try:
        assert wait_until(lambda: pidfile.exists() and strayfile.exists())
        live = int(pidfile.read_text())
        pids.append(live)
        assert live != parent.pid and int(strayfile.read_text()) == stray.pid
        result = run_helper_script(
            shell,
            'keepalive; echo "keepalive $?"\n',
            {**environment, "BASE": str(int(time.time()))},
        )
        assert result.returncode == 0 and result.stderr == "", result.stderr
        assert result.stdout == (
            f"WARNING run A: a watchdog of Pod podfixture01 that {control}/watchdog.pid does "
            f"not name still runs, with an old idle baseline: PID {stray.pid}. Check each "
            "with ps -p <pid> -o command= and stop it with kill <pid>.\nkeepalive 0\n"
        )
        assert stray.poll() is None and alive(live)  # Warned about, never killed.
    finally:
        for pid in pids:
            stop(pid)
        for process in (parent, stray):
            process.wait()


STDIN_READER = """#!/usr/bin/env bash
cat >> "$STUB_DIR/ssh.stdin"
"""
"""An ``ssh`` that reads its standard input, as the real one does for the remote command."""
PASTED = "podssh 'test -e /workspace/rehearsal/bootstrap.log'\necho the next pasted line ran\n"


def paste_into_bash(tmp_path: Path, pasted: str) -> tuple[str, str]:
    """Feed a pasted block to bash on its standard input, as bash 3.2 reads a paste."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    environment = helper_environment(tmp_path)
    stubs, control = Path(environment["STUB_DIR"]), Path(environment["CTRL"])
    (stubs / "ssh").write_text(STDIN_READER)
    (stubs / "ssh.stdin").write_text("")
    # BASH_ENV sources the helpers, so the paste itself is the two lines alone.
    result = subprocess.run(
        ["bash"],
        input=pasted,
        env={**environment, "BASH_ENV": str(control / "helpers.sh")},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0 and result.stderr == "", result.stderr
    return result.stdout, (stubs / "ssh.stdin").read_text()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_podssh_leaves_the_rest_of_a_pasted_block_to_the_shell(tmp_path: Path) -> None:
    out, consumed = paste_into_bash(tmp_path, PASTED)
    assert out == "the next pasted line ran\n" and consumed == ""
    # The same paste through podssh_stdin, which keeps its standard input as podssh once
    # did, loses its second line to ssh: the failure this guards against.
    out, consumed = paste_into_bash(
        tmp_path / "kept", PASTED.replace("podssh ", "podssh_stdin ", 1)
    )
    assert out == "" and consumed == "echo the next pasted line ran\n"


def test_only_the_key_delivery_keeps_podssh_stdin() -> None:
    body = helpers()
    assert 'podssh() {\n  podssh_stdin "$@" < /dev/null\n}\n' in body
    assert body.count("\n  ssh ") == 1  # podssh_stdin holds the one pinned ssh command.
    outside = "\n".join(all_blocks()).replace(body, "")
    (use,) = [line for line in outside.splitlines() if "podssh_stdin" in line]
    assert use.startswith("uv run --locked python -m tools.rehearsal_ops pod-env | podssh_stdin '")


def run_keepalive(
    tmp_path: Path,
    shell: str,
    *,
    watchdog_pid: str | None,
    base: int,
    fails: bool = False,
    receipt: Outcome | str | None = None,
    processes: str = "",
) -> tuple[str, list[str], list[str]]:
    """Two passes of keepalive, 10 s apart, starting at clock 2000; and their statuses."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    clock, calls, statuses = tmp_path / "clock", tmp_path / "calls", tmp_path / "statuses"
    clock.write_text("2000\n")
    calls.write_text("")
    control = tmp_path / "ctrl"
    control.mkdir()
    if watchdog_pid is not None:
        (control / "watchdog.pid").write_text(f"{watchdog_pid}\n")
    receipt_text = write_receipt(control, receipt)
    environment = {
        **os.environ,
        "CLOCK": str(clock),
        "CALLS": str(calls),
        "CTRL": str(control),
        "POD_ID": VALUES["POD_ID"],
        "RUN": "A",
        "BASE": str(base),
        "WATCHDOG_FAILS": "1" if fails else "",
        "PROCESSES": processes.format(ctrl=control),
    }
    record = 'echo $? >> "$STATUSES"\n'
    script = loop_stubs("") + f"keepalive; {record}sleep 10\nkeepalive; {record}"
    result = subprocess.run(
        [shell, "-c", script],
        env={**environment, "STATUSES": str(statuses)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0 and result.stderr == "", result.stderr
    if receipt_text is not None:
        assert (control / "watchdog-receipt.json").read_text() == receipt_text
    return result.stdout, calls.read_text().splitlines(), statuses.read_text().splitlines()


NO_WATCHDOG = (
    "WARNING run A: NO LIVE WATCHDOG guards Pod podfixture01: {ctrl}/watchdog.pid is missing "
    "or names no live watchdog of this Pod."
)
RESTART_FAILED = "WARNING run A: the watchdog restart FAILED; see {ctrl}/watchdog.log."
STRAY = "WARNING run A: a watchdog of Pod podfixture01 that {ctrl}/watchdog.pid does not name"


def test_keepalive_checks_this_pods_watchdog_as_the_helper_does() -> None:
    body = keepalive()
    check = (
        'ps -p "$LIVE" -o command= 2>/dev/null | grep -qF -- "--identity $CTRL/pod-identity.json"'
    )
    assert check in body
    assert body.count("BASE=0") == 2 and body.count("WARNING") == 3
    assert "if [ $(( $(date +%s) - BASE )) -ge 300 ]; then" in body
    # A final receipt of this Pod stops keepalive before it would restart the watchdog.
    assert body.index("watchdog-receipt.json") < body.index("return 3") < body.index("Restarting")
    for state in ("terminated", "absent", "identity_mismatch"):
        assert f'"{state}"' in body
    assert body.count("return 3") == 1 and body.rstrip().endswith("  return 0\n}")
    # Every wait loop stops on that status, and calls keepalive in no other way.
    code = "\n".join(_code())
    assert code.count(KEEPALIVE_GONE) == 3 and code.count("keepalive") == 4


@pytest.mark.parametrize("shell", SHELLS)
def test_keepalive_is_quiet_while_this_pods_watchdog_lives(tmp_path: Path, shell: str) -> None:
    out, calls, statuses = run_keepalive(tmp_path, shell, watchdog_pid="4242", base=1900)
    assert out == "" and calls == [] and statuses == ["0", "0"]
    out, calls, _ = run_keepalive(tmp_path / "old", shell, watchdog_pid="4242", base=1700)
    assert "WARNING" not in out and calls == ["watchdog 2000"]  # 300 s since the baseline.
    # A live watchdog is never second-guessed by a final receipt it has not written yet.
    out, calls, statuses = run_keepalive(
        tmp_path / "live", shell, watchdog_pid="4242", base=1900, receipt=TERMINATED
    )
    assert out == "" and calls == [] and statuses == ["0", "0"]


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("watchdog_pid", [None, "999", "not-a-pid"])
def test_keepalive_warns_and_restarts_a_missing_or_dead_watchdog(
    tmp_path: Path, shell: str, watchdog_pid: str | None
) -> None:
    out, calls, statuses = run_keepalive(tmp_path, shell, watchdog_pid=watchdog_pid, base=1990)
    assert out.count(NO_WATCHDOG.format(ctrl=tmp_path / "ctrl")) == 1
    assert calls == ["watchdog 2000"]  # At once, though the baseline was only 10 s old.
    assert statuses == ["0", "0"] and "POD GONE" not in out


@pytest.mark.parametrize("shell", SHELLS)
def test_keepalive_warns_when_a_restart_fails_and_retries_on_the_next_pass(
    tmp_path: Path, shell: str
) -> None:
    out, calls, statuses = run_keepalive(
        tmp_path, shell, watchdog_pid="4242", base=1700, fails=True
    )
    assert out.count(RESTART_FAILED.format(ctrl=tmp_path / "ctrl")) == 2
    assert calls == ["watchdog failed 2000", "watchdog failed 2010"]  # Not 5 minutes later.
    assert statuses == ["0", "0"]


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("watchdog_pid", [None, "999"])
@pytest.mark.parametrize(
    "receipt",
    [
        TERMINATED,
        Outcome("terminated", "cost_limit_delete_204_get_404", "podfixture01", 990.0, "stale"),
        Outcome("absent", "pod_get_404", "podfixture01", 990.0),
        Outcome("identity_mismatch", "identity_mismatch_no_delete", "podfixture01", 990.0),
    ],
    ids=["idle-delete", "cost-delete-with-detail", "absent", "identity-mismatch"],
)
def test_keepalive_stops_and_never_restarts_a_watchdog_that_ended_the_pod(
    tmp_path: Path, shell: str, watchdog_pid: str | None, receipt: Outcome
) -> None:
    out, calls, statuses = run_keepalive(
        tmp_path, shell, watchdog_pid=watchdog_pid, base=1990, receipt=receipt
    )
    verdict = (
        f"{POD_GONE}state {receipt.state}, reason {receipt.reason}, detail {receipt.detail} "
        f"({tmp_path}/ctrl/watchdog-receipt.json). It is NOT restarted. Follow step 14.\n"
    )
    assert out == verdict * 2 and statuses == ["3", "3"]
    assert calls == []  # A restart would overwrite the receipt with absent/pod_get_404.


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    "receipt",
    [
        Outcome("watching", "within_limits", "podfixture01", 990.0),
        Outcome("unknown", "pod_get_transport_error", "podfixture01", 990.0),
        Outcome("terminated", "idle_gpu_limit_delete_204_get_404", "otherpod01", 990.0),
        "not json",
        '["terminated"]',
    ],
    ids=["watching", "unknown", "another-pods-receipt", "not-json", "not-an-object"],
)
def test_keepalive_restarts_a_watchdog_whose_receipt_is_not_final_for_this_pod(
    tmp_path: Path, shell: str, receipt: Outcome | str
) -> None:
    out, calls, statuses = run_keepalive(
        tmp_path, shell, watchdog_pid=None, base=1990, receipt=receipt
    )
    assert out.count(NO_WATCHDOG.format(ctrl=tmp_path / "ctrl")) == 1 and "POD GONE" not in out
    assert calls == ["watchdog 2000"] and statuses == ["0", "0"]


@pytest.mark.parametrize("shell", SHELLS)
def test_keepalive_names_each_watchdog_of_this_pod_the_pidfile_does_not(
    tmp_path: Path, shell: str
) -> None:
    mine = "tools.runpod_watchdog --mode rehearsal --identity {ctrl}/pod-identity.json"
    processes = "\n".join(
        [
            f"4241 uv run --locked python -m {mine}",
            f"4242 python -m {mine}",
            f"5150 uv run --locked python -m {mine}",
            f"5151 python -m {mine}",
            "6161 python -m tools.runpod_watchdog --identity runs/b/podb/pod-identity.json",
            "7171 python -m tools.pilot_supervisor --identity {ctrl}/pod-identity.json",
        ]
    )
    out, calls, statuses = run_keepalive(
        tmp_path, shell, watchdog_pid="4242", base=1900, processes=processes
    )
    # The named watchdog and its uv parent are this Pod's guard; run B's watchdog and this
    # run's supervisor are not watchdogs of this Pod. Only 5150 and 5151 are strays.
    warning = (
        STRAY.format(ctrl=tmp_path / "ctrl") + " still runs, with an old idle baseline: PID "
        "5150 5151. Check each with ps -p <pid> -o command= and stop it with kill <pid>.\n"
    )
    assert out == warning * 2 and calls == [] and statuses == ["0", "0"]


# Step 7: one bootstrap per Pod, and a wait that notices a failed launch ------------------

BOOTSTRAP_STUBS = r"""
podssh() {
  case "$1" in
    "test -e /workspace/rehearsal/bootstrap.log") [ -n "$LOG_EXISTS" ] ;;
    "test ! -e /workspace/rehearsal/bootstrap.log || exit 1; cd /workspace/repo && "*)
      echo launch >> "$CALLS" ;;
    "grep -q ^BOOTSTRAP_EXIT= /workspace/rehearsal/bootstrap.log")
      [ -e "$CLOCK.exited" ] || [ "$(cat "$CLOCK")" -ge "$EXIT_AT" ] ;;
    "pgrep -f pilot_pod_bootstrap.sh")
      if [ "$(cat "$CLOCK")" -lt "$ALIVE_UNTIL" ]; then echo 4242; return 0; fi
      if [ -n "$EXITS_DURING_PGREP" ]; then : > "$CLOCK.exited"; fi
      return "$PGREP_NONE" ;;
    *) echo "unexpected podssh $1" >&2; return 99 ;;
  esac
}
"""


def bootstrap_wait() -> str:
    """Step 7's launch and wait, up to the end of the wait loop."""
    block = blocks(7)[1]
    return block[: block.index("\ndone\n") + 6]


def test_step_7_never_launches_a_second_bootstrap_and_checks_that_one_runs() -> None:
    wait = bootstrap_wait()
    (launch,) = [line for line in wait.splitlines() if "tools/pilot_pod_bootstrap.sh" in line]
    # Refused from the Mac when bootstrap.log exists, and again on the Pod before the tar.
    assert wait.startswith("if podssh 'test -e /workspace/rehearsal/bootstrap.log'; then\n")
    assert wait.index("\nelse\n") < wait.index(launch) < wait.index("\nfi\n")
    # Each check is its own statement, so only the bootstrap itself goes to the background.
    assert launch.strip().startswith(
        'podssh "test ! -e /workspace/rehearsal/bootstrap.log || exit 1; cd /workspace/repo && '
        "tar xzf /workspace/repo.tar.gz || exit 1; setsid nohup bash -c '"
    )
    assert launch.strip().endswith("' > /workspace/rehearsal/bootstrap.log 2>&1 < /dev/null &\"")
    # The wait checks liveness with the bootstrap script's own name, and keeps re-baselining.
    assert f"podssh 'pgrep -f {BOOTSTRAP.name}' > /dev/null; ALIVE=$?" in wait
    assert '[ "$ALIVE" -eq 1 ]' in wait
    assert f"  sleep 30; {KEEPALIVE_GONE}\ndone\n" in wait and "then watchdog; fi" not in wait
    assert "GIVE_UP=$(( $(date +%s) + 2700 ))" in wait
    assert "move the old log aside" in section(7)
    # A new commit also needs the workspace's old revision moved aside, or exit 6 repeats.
    assert (
        "podssh 'mv /workspace/rehearsal/revision.json "
        "/workspace/rehearsal/revision.json.old-$(date +%s)'" in section(7)
    )
    assert "without this it exits 6 every time" in section(7)


def run_bootstrap_wait(
    tmp_path: Path,
    shell: str,
    *,
    log_exists: bool = False,
    exit_at: int = 10**9,
    alive_until: int = 0,
    pgrep_none: int = 1,
    exits_during_pgrep: bool = False,
    watchdog_pid: str | None = "4242",
    receipt: Outcome | None = None,
) -> tuple[str, list[str]]:
    clock, calls = tmp_path / "clock", tmp_path / "calls"
    clock.write_text("1000\n")
    calls.write_text("")
    control = tmp_path / "ctrl"
    control.mkdir()
    if watchdog_pid is not None:
        (control / "watchdog.pid").write_text(f"{watchdog_pid}\n")
    receipt_text = write_receipt(control, receipt)
    environment = {
        **os.environ,
        "CLOCK": str(clock),
        "CALLS": str(calls),
        "CTRL": str(control),
        "POD_ID": VALUES["POD_ID"],
        "LOG_EXISTS": "1" if log_exists else "",
        "EXIT_AT": str(exit_at),
        "ALIVE_UNTIL": str(alive_until),
        "PGREP_NONE": str(pgrep_none),
        "EXITS_DURING_PGREP": "1" if exits_during_pgrep else "",
        "BASE": "1000",
        **{
            name: VALUES[name]
            for name in (
                "RUN",
                "SELF_DESTRUCT_DEADLINE",
                "BINDING_SHA256",
                "INIT_KEY",
                "INIT_SHA256",
                "INIT_SIZE",
            )
        },
    }
    result = subprocess.run(
        [shell, "-c", loop_stubs(BOOTSTRAP_STUBS) + bootstrap_wait() + "\nexit 0\n"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0 and result.stderr == "", result.stderr
    if receipt_text is not None:
        assert (control / "watchdog-receipt.json").read_text() == receipt_text
    return result.stdout, calls.read_text().splitlines()


NO_BOOTSTRAP = "no bootstrap runs on the Pod and bootstrap.log has no BOOTSTRAP_EXIT line"


@pytest.mark.parametrize("shell", SHELLS)
def test_step_7_launches_once_and_waits_for_the_bootstrap_to_exit(
    tmp_path: Path, shell: str
) -> None:
    out, calls = run_bootstrap_wait(tmp_path, shell, exit_at=1400, alive_until=1400)
    assert calls == ["launch", "watchdog 1300"]
    assert "already exists" not in out and NO_BOOTSTRAP not in out and "45 minutes" not in out
    assert "WARNING" not in out


@pytest.mark.parametrize("shell", SHELLS)
def test_step_7_wait_warns_and_restarts_when_watchdog_pid_is_missing(
    tmp_path: Path, shell: str
) -> None:
    out, calls = run_bootstrap_wait(
        tmp_path, shell, exit_at=1400, alive_until=1400, watchdog_pid=None
    )
    assert out.count(NO_WATCHDOG.format(ctrl=tmp_path / "ctrl")) == 1
    assert calls == ["launch", "watchdog 1030", "watchdog 1330"]  # Restarted on the first pass.


@pytest.mark.parametrize("shell", SHELLS)
def test_step_7_pasted_again_follows_the_first_bootstrap(tmp_path: Path, shell: str) -> None:
    out, calls = run_bootstrap_wait(
        tmp_path, shell, log_exists=True, exit_at=1400, alive_until=1400
    )
    assert calls == ["watchdog 1300"]  # No second launch.
    assert (
        "run A: /workspace/rehearsal/bootstrap.log already exists, so a bootstrap was launched "
        "on this Pod before; not launching a second one." in out
    )
    assert NO_BOOTSTRAP not in out


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("log_exists", [False, True])
def test_step_7_stops_when_no_bootstrap_runs_and_none_exited(
    tmp_path: Path, shell: str, log_exists: bool
) -> None:
    # The launch line failed (or an earlier bootstrap died without its exit line).
    out, calls = run_bootstrap_wait(tmp_path, shell, log_exists=log_exists)
    assert f"run A: {NO_BOOTSTRAP}: the launch failed." in out
    assert "terminate the Pod (step 13, TERMINATE=yes)" in out
    assert calls == ([] if log_exists else ["launch"])  # Stopped at once: no re-baseline.


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("answer", [255, 127])
def test_step_7_keeps_waiting_when_ssh_or_pgrep_cannot_answer(
    tmp_path: Path, shell: str, answer: int
) -> None:
    # 255: SSH failed; 127: no pgrep on the Pod. Neither is evidence that nothing runs.
    out, calls = run_bootstrap_wait(tmp_path, shell, exit_at=1400, pgrep_none=answer)
    assert NO_BOOTSTRAP not in out and calls == ["launch", "watchdog 1300"]


@pytest.mark.parametrize("shell", SHELLS)
def test_step_7_rechecks_the_exit_line_after_pgrep(tmp_path: Path, shell: str) -> None:
    # The bootstrap writes BOOTSTRAP_EXIT and exits between the loop's check and pgrep.
    out, calls = run_bootstrap_wait(tmp_path, shell, exits_during_pgrep=True)
    assert NO_BOOTSTRAP not in out and calls == ["launch"]


@pytest.mark.parametrize("shell", SHELLS)
def test_step_7_gives_up_after_45_minutes(tmp_path: Path, shell: str) -> None:
    out, calls = run_bootstrap_wait(tmp_path, shell, alive_until=10**9)
    assert "run A: bootstrap exceeded 45 minutes; terminate the Pod (step 13, TERMINATE=yes)" in out
    assert calls == ["launch", *(f"watchdog {1000 + 300 * n}" for n in range(1, 10))]


# Steps 7 and 9: each launch returns at once, as the Pod's shell runs it -----------------

LAUNCH_VALUES = (
    "RUN",
    "SELF_DESTRUCT_DEADLINE",
    "BINDING_SHA256",
    "INIT_KEY",
    "INIT_SHA256",
    "INIT_SIZE",
    "SPEC",
)
FAKE_BOOTSTRAP = 'printf "%s\\n" "$@" > "$STUB_DIR/bootstrap.args"\nsleep 5\n'
FAKE_WORKER = """#!/bin/sh
printf '%s\\n' "$@" > "$STUB_DIR/worker.args"
printf '%s\\n' "${R2_FIXTURE_NAME:-unset}" > "$STUB_DIR/worker.env"
sleep 5
"""
POD_STUBS = {
    "setsid": '#!/bin/sh\nexec "$@"\n',
    "tar": '#!/bin/sh\nprintf "%s\\n" "$@" > "$STUB_DIR/tar.args"\nexit "${TAR_EXIT:-0}"\n',
    "python3": FAKE_WORKER,
}
FAST_RETURN_SECONDS = 2.0
"""The fake bootstrap and worker run 5 s; a launch that waits for them takes at least that."""


def launch_branch(block: str) -> str:
    """The ``else`` branch of a block's first ``if``: the launch on the Pod."""
    start = block.index("\nelse\n") + len("\nelse\n")
    return block[start : block.index("\nfi\n", start) + 1]


def remote_launch(block: str) -> str:
    """The command the Pod's shell receives from a launch, as each Mac shell expands it."""
    expanded = set()
    for shell in SHELLS:
        result = subprocess.run(
            [shell, "-c", 'podssh() { printf "%s" "$1"; }\n' + launch_branch(block)],
            env={**os.environ, **{name: VALUES[name] for name in LAUNCH_VALUES}},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0 and result.stderr == "", result.stderr
        expanded.add(result.stdout)
    (command,) = expanded  # Every Mac shell sends the same text.
    return command


def pod_workspace(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A stand-in for the Pod's /workspace, with setsid, tar and python3 stubs on PATH."""
    assert re.fullmatch(r"[\w/.-]+", str(tmp_path)), tmp_path  # Unquoted, as /workspace is.
    workspace = tmp_path / "workspace"
    (workspace / "rehearsal").mkdir(parents=True)
    (workspace / "repo" / "tools").mkdir(parents=True)
    stubs = tmp_path / "pod-bin"
    stubs.mkdir()
    for name, text in POD_STUBS.items():
        (stubs / name).write_text(text)
        (stubs / name).chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
        "STUB_DIR": str(stubs),
    }
    return workspace, environment


def run_on_pod(command: str, environment: dict[str, str]) -> tuple[float, int]:
    """Run a launch as sshd does, with its output on a pipe like the SSH channel.

    ``cat`` returns only once every process holding the pipe has closed it, so the elapsed
    time is how long ``podssh`` would block on the Mac.
    """
    start = time.monotonic()
    result = subprocess.run(
        ["bash", "-c", 'bash -c "$POD_COMMAND" 2>&1 | cat; exit "${PIPESTATUS[0]}"'],
        env={**environment, "POD_COMMAND": command},
        capture_output=True,
        text=True,
        timeout=60,
    )
    return time.monotonic() - start, result.returncode


def wait_for_line(path: Path, prefix: str, seconds: float = 30) -> str:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            for line in path.read_text().splitlines():
                if line.startswith(prefix):
                    return line
        time.sleep(0.1)
    raise AssertionError(f"no {prefix} line in {path}")


def bootstrap_launch() -> str:
    command = remote_launch(blocks(7)[1])
    assert command.startswith(
        "test ! -e /workspace/rehearsal/bootstrap.log || exit 1; cd /workspace/repo && "
        "tar xzf /workspace/repo.tar.gz || exit 1; setsid nohup bash -c 'bash "
        "tools/pilot_pod_bootstrap.sh --mode rehearsal --run A "
    )
    assert command.endswith(
        "; echo BOOTSTRAP_EXIT=$?' > /workspace/rehearsal/bootstrap.log 2>&1 < /dev/null &"
    )
    return command


def test_step_7_launch_returns_at_once_and_the_bootstrap_runs_detached(tmp_path: Path) -> None:
    command = bootstrap_launch()
    workspace, environment = pod_workspace(tmp_path)
    (workspace / "repo.tar.gz").write_bytes(b"")
    (workspace / "repo" / "tools" / BOOTSTRAP.name).write_text(FAKE_BOOTSTRAP)
    elapsed, status = run_on_pod(command.replace("/workspace", str(workspace)), environment)
    assert status == 0
    assert elapsed < FAST_RETURN_SECONDS, f"the launch held the SSH channel for {elapsed:.1f} s"
    log = workspace / "rehearsal" / "bootstrap.log"
    assert wait_for_line(log, "BOOTSTRAP_EXIT=") == "BOOTSTRAP_EXIT=0"
    stubs = Path(environment["STUB_DIR"])
    assert (stubs / "tar.args").read_text().splitlines() == ["xzf", f"{workspace}/repo.tar.gz"]
    argv = (stubs / "bootstrap.args").read_text().splitlines()
    assert argv[:4] == ["--mode", "rehearsal", "--run", "A"]
    assert argv[argv.index("--workspace") + 1] == f"{workspace}/rehearsal"


@pytest.mark.parametrize("case", ["log_exists", "tar_fails"])
def test_step_7_launch_refuses_on_the_pod_without_starting_a_bootstrap(
    tmp_path: Path, case: str
) -> None:
    command = bootstrap_launch()
    workspace, environment = pod_workspace(tmp_path)
    (workspace / "repo" / "tools" / BOOTSTRAP.name).write_text(FAKE_BOOTSTRAP)
    log = workspace / "rehearsal" / "bootstrap.log"
    if case == "log_exists":
        log.write_text("BOOTSTRAP_EXIT=0\n")  # An earlier bootstrap's log.
    else:
        environment["TAR_EXIT"] = "2"
    elapsed, status = run_on_pod(command.replace("/workspace", str(workspace)), environment)
    assert status == 1 and elapsed < FAST_RETURN_SECONDS
    time.sleep(0.5)  # A wrongly started bootstrap would have written its arguments by now.
    stubs = Path(environment["STUB_DIR"])
    assert not (stubs / "bootstrap.args").exists()
    if case == "log_exists":
        assert log.read_text() == "BOOTSTRAP_EXIT=0\n" and not (stubs / "tar.args").exists()
    else:
        assert not log.exists()


def worker_launch() -> str:
    command = remote_launch(blocks(9)[0])
    assert command.startswith(
        "test ! -e /workspace/rehearsal/run.log || exit 1; setsid nohup bash -c 'set -a; "
        ". /workspace/rehearsal/r2-pod.env; set +a; cd /workspace/repo && "
        "python3 -m tools.rehearsal_run run "
    )
    assert command.endswith(
        "; echo WORKER_EXIT=$?'     > /workspace/rehearsal/run.log 2>&1 < /dev/null &"
    )
    return command


def test_step_9_launch_returns_at_once_and_the_worker_runs_detached(tmp_path: Path) -> None:
    command = worker_launch()
    workspace, environment = pod_workspace(tmp_path)
    (workspace / "rehearsal" / "r2-pod.env").write_text("R2_FIXTURE_NAME=loaded\n")
    elapsed, status = run_on_pod(command.replace("/workspace", str(workspace)), environment)
    assert status == 0
    assert elapsed < FAST_RETURN_SECONDS, f"the launch held the SSH channel for {elapsed:.1f} s"
    log = workspace / "rehearsal" / "run.log"
    assert wait_for_line(log, "WORKER_EXIT=") == "WORKER_EXIT=0"
    stubs = Path(environment["STUB_DIR"])
    argv = (stubs / "worker.args").read_text().splitlines()
    assert argv[:3] == ["-m", "tools.rehearsal_run", "run"]
    assert argv[argv.index("--output") + 1] == f"{workspace}/rehearsal-run"
    # The worker saw the key file's variables, loaded on the Pod, never on a command line.
    assert (stubs / "worker.env").read_text() == "loaded\n"


def test_step_9_launch_refuses_a_second_worker_on_the_pod(tmp_path: Path) -> None:
    command = worker_launch()
    workspace, environment = pod_workspace(tmp_path)
    (workspace / "rehearsal" / "r2-pod.env").write_text("R2_FIXTURE_NAME=loaded\n")
    log = workspace / "rehearsal" / "run.log"
    log.write_text("an earlier worker's log\n")
    elapsed, status = run_on_pod(command.replace("/workspace", str(workspace)), environment)
    assert status == 1 and elapsed < FAST_RETURN_SECONDS
    time.sleep(0.5)  # A wrongly started worker would have written its arguments by now.
    assert not (Path(environment["STUB_DIR"]) / "worker.args").exists()
    assert log.read_text() == "an earlier worker's log\n"


def test_no_other_block_sends_a_background_job_over_ssh() -> None:
    # Only the two launches end a podssh command with "&"; both are checked above.
    launches = [line for line in _code() if line.startswith("podssh ") and line.endswith('&"')]
    assert len(launches) == 2
    assert all(" && setsid" not in line and "|| exit 1; " in line for line in launches)


# Step 8: a fresh timer receipt right before the readiness receipt ----------------------

FRESH_TIMER_COPY = (
    'podscp root@$POD_HOST:/workspace/rehearsal/self-destruct.json "$CTRL/self-destruct.json"'
)


def receipt_block() -> str:
    (block,) = [block for block in blocks(8) if "tools.pilot_readiness_receipt" in block]
    return block


def test_step_8_receipt_block_first_copies_a_fresh_timer_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = receipt_block().splitlines()[:2]
    assert first == FRESH_TIMER_COPY
    assert second.startswith("uv run --locked python -m tools.pilot_readiness_receipt rehearsal ")
    # The only step 8 block before it writes the confirmations and nothing else.
    confirmations, receipt = blocks(8)
    assert receipt == receipt_block() and "self-destruct" not in confirmations
    assert confirmations.startswith("cat > \"$CTRL/confirmations.json\" <<'EOF'\n")
    # The receipt tool reads exactly that copy, and refuses one older than 60 s.
    ((argv, _),) = by_module("tools.pilot_readiness_receipt")
    namespace = parse("tools.pilot_readiness_receipt", argv, monkeypatch)
    assert str(namespace.self_destruct_receipt) == f"{VALUES['CTRL']}/self-destruct.json"
    assert SELF_DESTRUCT_FRESH_SECONDS == 60
    assert "more than 60 s earlier, and the copy from step 7 is older than that" in section(8)


@pytest.mark.parametrize("shell", SHELLS)
def test_step_8_copy_reaches_this_pods_live_receipt(tmp_path: Path, shell: str) -> None:
    environment = helper_environment(tmp_path)
    control, stubs = Path(environment["CTRL"]), Path(environment["STUB_DIR"])
    script = 'source "$CTRL/helpers.sh"\n' + FRESH_TIMER_COPY + "\n"
    result = subprocess.run(
        [shell, "-c", script], env=environment, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert (stubs / "scp.args").read_text().splitlines()[-2:] == [
        f"root@{VALUES['POD_HOST']}:/workspace/rehearsal/self-destruct.json",
        f"{control}/self-destruct.json",
    ]


# Step 9: the wait for the worker's first heartbeat --------------------------------------

LOOP_STUBS = r"""
podssh() {
  case "$1" in
    "test -e /workspace/rehearsal/run.log") [ -n "$LOG_EXISTS" ] ;;
    "test ! -e /workspace/rehearsal/run.log || exit 1; setsid nohup bash -c "*)
      echo launch >> "$CALLS" ;;
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
    tmp_path: Path,
    shell: str,
    *,
    heartbeat_at: int,
    exit_at: int,
    whole_block: bool = False,
    log_exists: bool = False,
    watchdog_fails: bool = False,
    watchdog_pid: str | None = "4242",
    receipt: Outcome | None = None,
) -> tuple[str, list[str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    clock, calls = tmp_path / "clock", tmp_path / "calls"
    clock.write_text("1000\n")
    calls.write_text("")
    control = tmp_path / "ctrl"
    control.mkdir()
    if watchdog_pid is not None:
        (control / "watchdog.pid").write_text(f"{watchdog_pid}\n")
    receipt_text = write_receipt(control, receipt)
    environment = {
        **os.environ,
        "CLOCK": str(clock),
        "CALLS": str(calls),
        "CTRL": str(control),
        "POD_ID": VALUES["POD_ID"],
        "HEARTBEAT_AT": str(heartbeat_at),
        "EXIT_AT": str(exit_at),
        "LOG_EXISTS": "1" if log_exists else "",
        "WATCHDOG_FAILS": "1" if watchdog_fails else "",
        "BASE": "1000",
        **{name: VALUES[name] for name in LAUNCH_VALUES},
    }
    code = blocks(9)[0] if whole_block else wait_loop()
    # The block's last line fails when no heartbeat exists; "exit 0" is reached only if every
    # line of the block parsed and ran.
    result = subprocess.run(
        [shell, "-c", loop_stubs(LOOP_STUBS) + code + "\nexit 0\n"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0 and result.stderr == "", result.stderr
    if receipt_text is not None:
        assert (control / "watchdog-receipt.json").read_text() == receipt_text
    return result.stdout, calls.read_text().splitlines()


def test_step_9_waits_with_a_deadline_and_rebaselines_like_step_7() -> None:
    loop = wait_loop()
    assert "GIVE_UP=$(( $(date +%s) + 1200 ))" in loop
    assert f"  sleep 10; {KEEPALIVE_GONE}\ndone\n" in loop and "then watchdog; fi" not in loop
    assert "^WORKER_EXIT=" in loop and "tail -n 40 /workspace/rehearsal/run.log" in loop
    # Refused from the Mac when run.log exists, and again on the Pod before the launch.
    block = blocks(9)[0]
    assert block.startswith("if podssh 'test -e /workspace/rehearsal/run.log'; then\n")
    assert block.index("\nelse\n") < block.index("setsid nohup") < block.index("\nfi\n")
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


@pytest.mark.parametrize("shell", SHELLS)
def test_step_9_launches_once_then_follows_the_first_worker(tmp_path: Path, shell: str) -> None:
    out, calls = run_wait(tmp_path, shell, heartbeat_at=1400, exit_at=10**9, whole_block=True)
    assert calls[0] == "launch" and calls.count("launch") == 1
    out, calls = run_wait(
        tmp_path / "again",
        shell,
        heartbeat_at=1400,
        exit_at=10**9,
        whole_block=True,
        log_exists=True,
    )
    assert "launch" not in calls  # Pasted again: no second worker.
    assert (
        "run A: /workspace/rehearsal/run.log already exists, so a worker was launched on this "
        "Pod before; not launching a second one. The wait below follows that one." in out
    )


@pytest.mark.parametrize("shell", SHELLS)
def test_step_9_wait_warns_on_a_failed_restart_and_retries_on_the_next_pass(
    tmp_path: Path, shell: str
) -> None:
    out, calls = run_wait(tmp_path, shell, heartbeat_at=1400, exit_at=10**9, watchdog_fails=True)
    assert calls[:2] == ["watchdog failed 1300", "watchdog failed 1310"]
    assert out.count(RESTART_FAILED.format(ctrl=tmp_path / "ctrl")) == len(range(1300, 1401, 10))


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("step", [5, 7, 9])
def test_every_wait_loop_stops_once_the_watchdog_has_ended_the_pod(
    tmp_path: Path, shell: str, step: int
) -> None:
    # The watchdog deleted the Pod: it wrote its final receipt, removed its pidfile and exited.
    gone: dict[str, Any] = {"watchdog_pid": None, "receipt": TERMINATED}
    if step == 5:
        out, calls, captures = run_running_wait(tmp_path, shell, started_at=10**9, **gone)
        assert len(captures) == 1  # One pass, then the wait stops.
    elif step == 7:
        out, calls = run_bootstrap_wait(tmp_path, shell, alive_until=10**9, **gone)
        assert calls[0] == "launch"
        calls = calls[1:]
    else:
        out, calls = run_wait(tmp_path, shell, heartbeat_at=10**9, exit_at=10**9, **gone)
    assert out.count(POD_GONE + "state terminated, reason idle_gpu_limit_delete_204_get_404") == 1
    assert out.rstrip().endswith(GONE_LINE)
    # Never restarted: the receipt that says why the Pod went is left as it was.
    assert calls == [] and "NO LIVE WATCHDOG" not in out
    assert "minutes" not in out and "TERMINATE=yes" not in out


# Step 13: confirm by reading; DELETE only on shutdown_ready or TERMINATE=yes --------------

TERMINATE_STUBS = r"""
curl() {
  cat > "$STUB_DIR/curl.stdin"
  printf '%s\n' "$@" > "$STUB_DIR/curl.args"
  : > "$STUB_DIR/deleted"
  echo 204
}
uv() {
  echo "$*" >> "$STUB_DIR/uv.calls"
  if [ "$GONE" = already ] || { [ "$GONE" = after_delete ] && [ -e "$STUB_DIR/deleted" ]; }; then
    echo "runpod-pod-identity: LookupError: resource not found (HTTP 404)" >&2
    return 1
  fi
  echo '{"pod_id": "podfixture01", "status": "RUNNING"}'
}
kill() { echo "$*" >> "$STUB_DIR/kill.calls"; }
ps() { [ "$2" = 4242 ] && printf '%s\n' "$WATCHDOG_COMMAND"; }
"""
DELETE_URL = "https://api.runpod.io/v2/pods/$POD_ID"
DECISION_LINE = "DECISION=${TERMINATE:-}; unset TERMINATE"


def termination() -> str:
    """Step 13's second block: the gated DELETE and the 404-verified teardown record."""
    (block,) = [block for block in blocks(13) if "-X DELETE" in block]
    return block


def test_step_13_confirm_only_reads_and_the_delete_is_gated() -> None:
    confirm, terminate = blocks(13)
    assert terminate == termination()
    assert "DELETE" not in confirm and "curl" not in confirm and "kill" not in confirm
    assert "tools.runpod_pod_identity status" in confirm
    assert '"$CTRL/supervisor/heartbeat.json"' in confirm
    # Every DELETE in the runbook is the one inside step 13's approval branch.
    code = "\n".join(_code())
    assert code.count("-X DELETE") == 1
    gate = 'if [ -n "$POD_ID" ] && [ "$APPROVAL" != none ]; then'
    branch = terminate[terminate.index(gate) : terminate.index("\nelse\n", terminate.index(gate))]
    assert f'-X DELETE "{DELETE_URL}"' in branch
    # The approval reads the heartbeat the watchdog reads, for this Pod only.
    assert _watchdog().heartbeat == Path(VALUES["CTRL"]) / "supervisor" / "heartbeat.json"
    assert '"$CTRL/supervisor/heartbeat.json" "$POD_ID"' in terminate
    assert 'h.get("status") == "shutdown_ready" and h.get("pod_id") == sys.argv[2]' in terminate
    assert 'status="shutdown_ready"' in (ROOT / "tools" / "pilot_supervisor.py").read_text()
    # A shutdown_ready approves the DELETE only with both receipts verified, as the watchdog
    # requires before it deletes on that status.
    receipts = ("mac_receipt_verified", "github_receipt_verified")
    assert all(f'h.get("{name}") is True' in terminate for name in receipts)
    watchdog_source = (ROOT / "tools" / "runpod_watchdog.py").read_text()
    assert all(f'value.get("{name}") is True' in watchdog_source for name in receipts)
    # The decision is read and cleared on the first line, then only DECISION is consulted.
    first, rest = terminate.split("\n", 1)
    assert first == DECISION_LINE and "TERMINATE:-" not in rest and "unset TERMINATE" not in rest
    assert '[ "$DECISION" = yes ] && APPROVAL=TERMINATE=yes' in rest
    # Teardown is recorded only after the status tool's own 404 message.
    identity = (ROOT / "tools" / "runpod_pod_identity.py").read_text()
    assert 'LookupError("resource not found (HTTP 404)")' in identity
    after = terminate[terminate.index("grep -qF 'resource not found (HTTP 404)'") :]
    # Only this Pod's watchdog is stopped, after the same command-line check the helper makes.
    check = (
        'if [ -n "$WATCHDOG_PID" ] && ps -p "$WATCHDOG_PID" -o command= 2>/dev/null | '
        'grep -qF -- "--identity $CTRL/pod-identity.json"; then kill "$WATCHDOG_PID"; fi'
    )
    assert check in after and terminate.count("kill ") == 1
    assert 'grep -qF -- "--identity $CTRL/pod-identity.json"' in helpers()
    assert after.index("export TEARDOWN=") < after.index(check) < after.index("\nelse\n")
    # Each named TERMINATE=yes case is in the text.
    for case in ("Before the supervisor ran.", "no milestone at risk.", "After step 14's rescue."):
        assert case in section(13)


def run_termination(
    tmp_path: Path,
    shell: str,
    *,
    heartbeat: str | None,
    terminate: str | None = None,
    gone: str = "after_delete",
    pod_id: str = "podfixture01",
    watchdog_command: str | None = None,
) -> tuple[str, Path, bool]:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    control = tmp_path / "ctrl"
    (control / "supervisor").mkdir(parents=True)
    if heartbeat is not None:
        (control / "supervisor" / "heartbeat.json").write_text(heartbeat)
    (control / "run.env").write_text("export ORIGIN=1800000000 POD_ID=podfixture01\n")
    (control / "watchdog.pid").write_text("4242\n")  # This Pod's watchdog.
    if watchdog_command is None:
        watchdog_command = (
            "/repo/.venv/bin/python3 -m tools.runpod_watchdog --mode rehearsal "
            f"--identity {control}/pod-identity.json "
            f"--heartbeat {control}/supervisor/heartbeat.json"
        )
    environment = {
        **os.environ,
        "STUB_DIR": str(stubs),
        "CTRL": str(control),
        "POD_ID": pod_id,
        "RUN": "A",
        "GONE": gone,
        "RUNPOD_API_KEY": "rpa_fixture_key",
        "WATCHDOG_COMMAND": watchdog_command,
    }
    environment.pop("TERMINATE", None)
    typed = "" if terminate is None else f"TERMINATE={terminate}\n"  # What the owner types.
    script = (
        TERMINATE_STUBS + typed + termination() + 'printf "TERMINATE=[%s]\\n" "${TERMINATE:-}"\n'
    )
    result = subprocess.run(
        [shell, "-c", script], env=environment, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().endswith("TERMINATE=[]")  # The decision never lingers.
    kills = stubs / "kill.calls"
    stopped = kills.exists()
    assert not stopped or kills.read_text() == "4242\n"  # This Pod's watchdog only.
    return result.stdout, stubs, stopped


def shutdown_ready(pod_id: str = "podfixture01", **fields: object) -> str:
    """The supervisor's mirrored shutdown_ready heartbeat, with both receipts verified."""
    value = {
        "status": "shutdown_ready",
        "pod_id": pod_id,
        "phase": "done",
        "mac_receipt_verified": True,
        "github_receipt_verified": True,
        **fields,
    }
    return json.dumps({name: item for name, item in value.items() if item is not None})


def assert_deleted(stubs: Path, out: str) -> None:
    assert (stubs / "curl.args").read_text().splitlines() == [
        "--config",
        "-",
        "-sS",
        "-o",
        "/dev/null",
        "-w",
        "%{http_code}\\n",
        "-X",
        "DELETE",
        "https://api.runpod.io/v2/pods/podfixture01",
    ]
    # The key reaches curl on its standard input only.
    stdin = (stubs / "curl.stdin").read_text()
    assert stdin == 'header = "Authorization: Bearer rpa_fixture_key"\n'
    assert "rpa_fixture_key" not in (stubs / "curl.args").read_text()
    assert "run A: deleting Pod podfixture01" in out


@pytest.mark.parametrize("shell", SHELLS)
def test_step_13_deletes_after_shutdown_ready_and_records_the_404(
    tmp_path: Path, shell: str
) -> None:
    out, stubs, stopped = run_termination(tmp_path, shell, heartbeat=shutdown_ready())
    assert_deleted(stubs, out)
    assert "(shutdown_ready)" in out and "resource not found (HTTP 404)" in out
    assert re.search(r"\nexport TEARDOWN=\d+\n$", (tmp_path / "ctrl" / "run.env").read_text())
    assert stopped  # This Pod's watchdog is stopped only after the 404.


@pytest.mark.parametrize("shell", SHELLS)
def test_step_13_deletes_on_the_owners_terminate_decision(tmp_path: Path, shell: str) -> None:
    out, stubs, stopped = run_termination(tmp_path, shell, heartbeat=None, terminate="yes")
    assert_deleted(stubs, out)
    assert "(TERMINATE=yes)" in out and stopped


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    ("heartbeat", "terminate", "pod_id"),
    [
        (json.dumps({"status": "running", "pod_id": "podfixture01"}), None, "podfixture01"),
        (json.dumps({"status": "uploading", "pod_id": "podfixture01"}), None, "podfixture01"),
        (json.dumps({"status": "failed", "pod_id": "podfixture01"}), None, "podfixture01"),
        (shutdown_ready("otherpod"), None, "podfixture01"),  # Another Pod's old heartbeat.
        # shutdown_ready without both verified receipts, which the watchdog also refuses.
        (shutdown_ready(mac_receipt_verified=None), None, "podfixture01"),
        (shutdown_ready(github_receipt_verified=None), None, "podfixture01"),
        (shutdown_ready(github_receipt_verified=False), None, "podfixture01"),
        (shutdown_ready(mac_receipt_verified="true"), None, "podfixture01"),
        (shutdown_ready(mac_receipt_verified=1, github_receipt_verified=1), None, "podfixture01"),
        (None, None, "podfixture01"),
        ("not json", None, "podfixture01"),
        ('["shutdown_ready"]', None, "podfixture01"),
        (json.dumps({"status": "running", "pod_id": "podfixture01"}), "YES", "podfixture01"),
        (json.dumps({"status": "running", "pod_id": "podfixture01"}), "y", "podfixture01"),
        (json.dumps({"status": "running", "pod_id": "podfixture01"}), "", "podfixture01"),
        (None, "yes", ""),  # No Pod ID: nothing to delete.
    ],
)
def test_step_13_never_deletes_without_shutdown_ready_or_terminate_yes(
    tmp_path: Path, shell: str, heartbeat: str | None, terminate: str | None, pod_id: str
) -> None:
    out, stubs, stopped = run_termination(
        tmp_path, shell, heartbeat=heartbeat, terminate=terminate, pod_id=pod_id
    )
    assert not (stubs / "curl.args").exists() and "deleting Pod" not in out
    assert "no DELETE sent" in out and "is not confirmed gone" in out
    assert "TEARDOWN" not in (tmp_path / "ctrl" / "run.env").read_text()
    assert not stopped  # A live Pod keeps its watchdog.


@pytest.mark.parametrize("shell", SHELLS)
def test_step_13_records_nothing_while_the_pod_is_still_present(tmp_path: Path, shell: str) -> None:
    out, stubs, stopped = run_termination(tmp_path, shell, heartbeat=shutdown_ready(), gone="never")
    assert_deleted(stubs, out)
    assert "run A: Pod podfixture01 is not confirmed gone" in out
    assert "TEARDOWN" not in (tmp_path / "ctrl" / "run.env").read_text() and not stopped


@pytest.mark.parametrize("shell", SHELLS)
def test_step_13_records_a_pod_the_guards_already_deleted(tmp_path: Path, shell: str) -> None:
    failed = json.dumps({"status": "failed", "pod_id": "podfixture01"})
    out, stubs, stopped = run_termination(tmp_path, shell, heartbeat=failed, gone="already")
    assert not (stubs / "curl.args").exists() and "no DELETE sent" in out
    assert "TEARDOWN=" in (tmp_path / "ctrl" / "run.env").read_text() and stopped


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    "command",
    [
        "sleep 30",  # The PID now belongs to some other process.
        "python3 -m tools.runpod_watchdog --mode rehearsal --identity runs/other/pod-identity.json",
        "",  # No such process.
    ],
)
def test_step_13_never_kills_a_process_that_is_not_this_pods_watchdog(
    tmp_path: Path, shell: str, command: str
) -> None:
    out, stubs, stopped = run_termination(
        tmp_path, shell, heartbeat=shutdown_ready(), watchdog_command=command
    )
    assert_deleted(stubs, out)
    assert "TEARDOWN=" in (tmp_path / "ctrl" / "run.env").read_text()
    assert not stopped


@pytest.mark.parametrize("shell", SHELLS)
def test_step_13_clears_terminate_even_when_the_block_stops_partway(
    tmp_path: Path, shell: str
) -> None:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    control = tmp_path / "ctrl"
    control.mkdir()
    report = 'printf "TERMINATE=[%s] DECISION=[%s]\\n" "${TERMINATE:-}" "${DECISION:-}"'
    # The owner typed TERMINATE=yes; the block then stops at its gate (a crash, a Ctrl-C).
    script = (
        TERMINATE_STUBS
        + f"TERMINATE=yes\ntrap '{report}' EXIT\npython3() {{ exit 3; }}\n"
        + termination()
    )
    environment = {
        **os.environ,
        "STUB_DIR": str(stubs),
        "CTRL": str(control),
        "POD_ID": VALUES["POD_ID"],
        "RUN": "A",
        "RUNPOD_API_KEY": "rpa_fixture_key",
    }
    result = subprocess.run(
        [shell, "-c", script], env=environment, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 3
    assert result.stdout == "TERMINATE=[] DECISION=[yes]\n"
    assert not (stubs / "curl.args").exists()


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
    docs = (ROOT / "docs" / "rehearsal-runs.md").read_text(encoding="utf-8")
    # The two-label, 780 s drain is gone from both pages.
    for old in ("8.46", "780 s", "4.23 MB/s", "13.5 MB/s", "(900 s - 120 s)", "(900 - 120)"):
        assert old not in text and old not in docs, old
    # One label in the 360 s drain with a factor-2 margin.
    assert "1 x 1.65 GB x 2 / 360 s, about **9.17 MB/s**" in text
    assert "1 x 1.65 GB x 2 / 360 s = 9.17 MB/s" in docs and "(9.17 MB/s)" in docs
    rate = 1 * 1.65e9 * 2 / 360
    assert rate == pytest.approx(9.17e6, rel=1e-3)
    assert "about 9.17 million" in section(7)
    # Why one label suffices: a rolling label uploads in at most 180 s at the gated rate.
    assert 1.65e9 / rate == pytest.approx(180) and "at most 180 s" in text
    assert "no rolling save within 900 s of a stage end" in text
    # The 900 s window: a 360 s drain, 120 s of bounded checks to the final heartbeat by
    # 480 s, then the Mac's 420 s SSH-pull reserve.
    assert 360 + 120 + 420 == rehearsal_ops.MAX_TRANSFER_SECONDS
    for phrase in ("**360 s**", "**120 s**", "**420 s**", "final heartbeat by 480 s"):
        assert phrase in text, phrase
    for name in ("`DRAIN_SECONDS`", "`TRANSFER_MARGIN_SECONDS`", "`PULL_RESERVE_SECONDS`"):
        assert name in text and name.strip("`") in docs, name
    # The SSH pull: one 1.65 GB label in about 370 s of the reserve is about 4.5 MB/s, and
    # the pilot's one 1.62 GB pull took 150 s or less.
    assert 1.65e9 / 370 == pytest.approx(4.5e6, rel=0.02)
    for page in (text, docs):
        assert "about 4.5 MB/s" in page and "about 9 MB/s" in page
        assert "150 s or less, about 10.7 MB/s or faster" in page
    split = tuple(
        getattr(rehearsal_ops, name, None)
        for name in ("DRAIN_SECONDS", "TRANSFER_MARGIN_SECONDS", "PULL_RESERVE_SECONDS")
    )
    if split[0] is not None:  # Once the worker side's constants have landed, they agree.
        assert split == (360, 120, 420)
        assert rehearsal_ops.REQUIRED_UPLOAD_BYTES_PER_SECOND == pytest.approx(rate)
    assert "REQUIRED_UPLOAD_BYTES_PER_SECOND" in text and "128 MiB" in text
    assert "r2_milestones_unverified" in section(14) and "finalization.json" in section(14)
    assert "420 s reserve" in section(14) and "every 5 s" in section(14)


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
        for elapsed in (0, 1, HOUR - 1, HOUR, 60_000, 60_000 + 7 * HOUR - 1):
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


# Every block pastes safely into an interactive zsh with interactivecomments off -----------

HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
OPERATORS = ";&|()<>"


def comment_lines(block: str) -> list[str]:
    """The lines of one shell block that hold a comment.

    A ``#`` counts when it starts a word outside quotes, as in a script; ``$#``, ``${#name}``
    and a ``#`` inside a word or quotes do not. In a heredoc body every ``#`` counts: there
    it is a comment of the language the heredoc holds. Quotes may span lines.
    """
    found: list[str] = []
    single = double = False
    heredoc: str | None = None
    for line in block.splitlines():
        if heredoc is not None:
            if line == heredoc:
                heredoc = None
            elif "#" in line:
                found.append(line)
            continue
        pending: str | None = None
        escaped = False
        previous = "\n"
        index = 0
        while index < len(line):
            character = line[index]
            if escaped:
                escaped = False
            elif single:
                single = character != "'"
            elif character == "\\":
                escaped = True
            elif character == '"':
                double = not double
            elif double:
                pass
            elif character == "'":
                single = True
            elif character == "#" and (previous.isspace() or previous in OPERATORS):
                found.append(line)
                break
            elif line.startswith("<<", index) and (match := HEREDOC.match(line, index)):
                pending = match.group(2)
                index = match.end()
                previous = line[index - 1]
                continue
            previous = character
            index += 1
        if pending is not None:
            heredoc = pending
    return found


def all_blocks() -> list[str]:
    return re.findall(r"```sh\n(.*?)```", RUNBOOK.read_text(encoding="utf-8"), re.S)


def test_the_comment_finder_finds_what_an_interactive_zsh_would_misread() -> None:
    misread = [
        "echo a  # expect 204",
        "# a whole-line note",
        "echo a;# after an operator",
        "cat <<'PY'\nx = 1  # a Python comment\nPY",
        "echo 'one\ntwo' # after a quote that spans lines",
    ]
    for text in misread:
        assert comment_lines(text), text
    harmless = [
        'echo "a # b"',
        "echo 'a # b'",
        "echo $# ${#name}",
        "echo a#b",
        "echo \\# escaped",
        "cat <<'PY'\nx = 1\nPY\necho done",
        "printf '%s\\n' \"$RUN\" | grep -c '^#'",
    ]
    for text in harmless:
        assert comment_lines(text) == [], text


def test_no_block_holds_a_comment() -> None:
    found = {line for block in all_blocks() for line in comment_lines(block)}
    assert not found, sorted(found)
    # The prose says so, and keeps the note for the owner's own lines.
    assert "The blocks hold no `#` comments" in section(2)
    assert "run `setopt interactivecomments` in that terminal" in section(2)


def test_every_block_parses_in_an_interactive_zsh_without_comments(tmp_path: Path) -> None:
    if shutil.which("zsh") is None:
        pytest.skip("zsh is not installed")
    for block in all_blocks():
        # The operator pastes the reviewed hash over the step 2 placeholder.
        pasted = block.replace("<reviewed binding sha256>", BINDING_SHA256)
        # -i reads the paste as a terminal session would: interactivecomments is off by
        # default, history expansion is on; -n parses without running anything.
        result = subprocess.run(
            ["zsh", "-f", "-i", "-n"],
            input=pasted,
            env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "PS1": ""},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (block.splitlines()[0], result.stderr)
        for error in ("unmatched", "parse error", "event not found", "bad pattern"):
            assert error not in result.stderr, (block.splitlines()[0], result.stderr)
