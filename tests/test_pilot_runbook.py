"""The runbook's watchdog launch must run exactly as written against the real tools."""

import json
import re
import shlex
from pathlib import Path

from tools.runpod_watchdog import Policy, build_parser, policy_from_args

RUNBOOK = Path(__file__).resolve().parents[1] / "PILOT_RUNBOOK.md"
LAUNCH = "uv run --locked python -m tools.runpod_watchdog"
NOW = 1_800_000_300
ORIGIN = 1_800_000_000


def _substitute(text: str) -> str:
    """Give the runbook's shell expansions concrete values; nothing is executed."""
    text = text.replace("$(date +%s)", str(NOW)).replace("$ORIGIN", str(ORIGIN))
    return text.replace("$CTRL", "ctrl")


def _watchdog_arguments() -> list[str]:
    lines = RUNBOOK.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(LAUNCH + " ")]
    assert len(starts) == 1, "the runbook launches the watchdog exactly once"
    command, index = "", starts[0]
    while True:
        line = lines[index]
        command += line.rstrip("\\").rstrip() + " "
        if not line.rstrip().endswith("\\"):
            break
        index += 1
    command = _substitute(command.split(" > ", 1)[0])  # Drop the log redirect and '&'.
    tokens = shlex.split(command)
    assert tokens[: len(LAUNCH.split())] == LAUNCH.split()
    return tokens[len(LAUNCH.split()) :]


def _policy_file() -> dict[str, object]:
    text = RUNBOOK.read_text(encoding="utf-8")
    match = re.search(r'cat > "\$CTRL/policy\.json" <<EOF\n(.*?)\nEOF\n', text, re.S)
    assert match, "the runbook writes policy.json with a heredoc"
    value = json.loads(_substitute(match.group(1)))
    assert isinstance(value, dict)
    return value


def test_runbook_watchdog_command_parses_with_the_real_command_line() -> None:
    args = build_parser().parse_args(_watchdog_arguments())  # SystemExit on any unknown flag.
    policy = policy_from_args(args)
    assert policy.idle_baseline_unix == NOW
    assert policy.billing_started_at_unix == ORIGIN
    assert (policy.max_wall_seconds, policy.max_idle_seconds, policy.max_transfer_seconds) == (
        5400,
        600,
        1500,
    )


def test_runbook_policy_file_loads_where_the_receipt_and_supervisor_read_it() -> None:
    # tools.pilot_readiness_receipt and tools.pilot_supervisor both call Policy(**value).
    policy = Policy(**_policy_file())  # type: ignore[arg-type]
    cli = policy_from_args(build_parser().parse_args(_watchdog_arguments()))
    for name in (
        "max_wall_seconds",
        "max_cost_usd",
        "max_idle_seconds",
        "quoted_total_hourly",
        "max_transfer_seconds",
        "idle_baseline_unix",
    ):
        assert getattr(policy, name) == getattr(cli, name), name
