"""Every tool command in the rehearsal runbook must run as written against the real parsers.

Shell variables get concrete values; nothing is executed. A command is parsed by calling
its module's ``main`` with ``ArgumentParser.parse_args`` intercepted, so an unknown flag,
a missing required flag or a bad choice fails here exactly as it would on the Mac or Pod.
"""

import argparse
import importlib
import importlib.util
import json
import re
import shlex
from collections import Counter
from pathlib import Path

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
    "CTRL": "runs/rehearsal-a-controller",
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
        return line.index("'", start)  # The command sits inside bash -c '...'.
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
    assert '--data @"$CTRL/create-body.json"' in code


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
    assert "export CTRL=runs/rehearsal-$RUN_LOWER-controller" in code
    assert "export KNOWN_HOSTS=~/.ssh/rehearsal_${RUN_LOWER}_known_hosts" in code
    assert "export SPEC=configs/rehearsal-a40-s-$RUN_LOWER.json" in code
    assert 'ssh-keyscan -p "$POD_PORT" "$POD_HOST" > "$KNOWN_HOSTS"' in code
