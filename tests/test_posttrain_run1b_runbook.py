"""Every tool command in run 1b's runbook parses with its real command line, and its figures
are the specification's.

Run 1's runbook test supplies the machinery (``tests/test_posttrain_runbook.py``): shell
variables get concrete values, each tool's ``main`` is called with ``parse_args``
intercepted, and every block must parse in bash and an interactive zsh with no comment. The
blocks of run 1's runbook that run 1b reuses are run 1's own test's.
"""

import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import test_posttrain_runbook as base
from test_rehearsal_runbook import comment_lines

from tools import posttrain_ops
from tools.pilot_feasibility import posttrain_report
from tools.posttrain_run import FORMAT_GATE_STOP, Specification, launch_fit

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "POSTTRAIN_RUN1B_RUNBOOK.md"
SPEC = ROOT / "configs" / "rehearsal-a40-s-post-1b.json"
ORIGIN = base.ORIGIN
VALUES = {
    **base.VALUES,
    "RUN": "1b",
    "CTRL": "runs/posttrain-controller/1b/podpost1b",
    "KNOWN_HOSTS": "/Users/owner/.ssh/posttrain_1b_known_hosts",
    "SPEC": "configs/rehearsal-a40-s-post-1b.json",
    "SHARED": "runs/posttrain-shared-1b",
    "RUN1_SHARED": "runs/posttrain-shared",
    "SFT": "runs/sft-data-20261027",
    "EXPORTS": "runs/posttrain-shared/exports",
    "BINDING": "runs/posttrain-shared-1b/binding.json",
    "WALL": "9000",
    "TIMER": "8820",
    "WATCHDOG_COST": "1.49",
    "LAUNCH": f"runs/posttrain-controller/1b/launch-{ORIGIN}",
    "POD_ID": "podpost1b",
    "SELF_DESTRUCT_DEADLINE": str(ORIGIN + 8_820),
    "ART": "runs/posttrain-shared-1b/artifacts",
    "RUN1_REPORT": "runs/posttrain-controller/1/podpost01/supervisor/final/report.json",
    "RUN1_SFT_SHA256": "6b" * 32,
}


@pytest.fixture(autouse=True)
def run_1b(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(base, "RUNBOOK", RUNBOOK)
    monkeypatch.setattr(base, "VALUES", VALUES)
    yield


def text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def placeholders(line: str) -> str:
    return line.replace("<run 1 pod id>", "podpost01")


def test_every_tool_command_parses_with_its_real_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = base.logical_lines

    def lines() -> list[str]:
        return [placeholders(line) for line in original()]

    monkeypatch.setattr(base, "logical_lines", lines)
    found = base.commands()
    modules = {module for module, _, _ in found}
    for module in (
        "tools.posttrain_run",
        "tools.posttrain_ops",
        "tools.posttrain_binding",
        "tools.pilot_feasibility",
        "tools.sft_data",
        "tools.runpod_pod_identity",
        "tools.mac_window_check",
        "tools.benchmark_dispatch",
        "scratchcoder.cli",
    ):
        assert module in modules, module
    for module, argv, line in found:
        namespace = base.parse(module, argv, monkeypatch)
        assert "SECRET" not in line and "ACCESS_KEY" not in line, line
        if module == "tools.mac_window_check" and namespace.command == "exec":
            assert namespace.argv[:1] == ["--"] and namespace.argv[1:4] == ["uv", "run", "--locked"]


def test_the_limits_and_figures_are_the_specifications() -> None:
    raw = json.loads(SPEC.read_text())
    derived = posttrain_ops.specification_limits(raw)
    assert (str(derived.wall_seconds), str(derived.self_destruct_seconds)) == ("9000", "8820")
    assert str(derived.watchdog_cost_stop_usd) == VALUES["WATCHDOG_COST"]
    body = text()
    for phrase in ("**$1.50**", "**$1.49**", "**9,000 s**", "**8,820 s**", "**900 s**"):
        assert phrase in body, phrase
    assert "wall 9000 s, timer 8820 s, watchdog cost stop 1.49 USD" in body
    spec = Specification.load(SPEC)
    planned = posttrain_report(SPEC, quoted_gpu_hourly=0.59)["scenarios"]["1x generation"]
    projection = f"about {planned['total_hours']:.2f} hours and ${planned['total_usd']:.2f}"
    assert body.count(projection) == 2, projection
    lifetime = spec.ceiling.self_destruct_seconds
    needed = launch_fit(spec, remaining_seconds=lifetime)["planning_seconds_needed"]
    assert f"The launch rule needs {needed:,.0f} s" in body
    assert f"about {(lifetime - needed) / 60:.0f} minutes of the 8,820 s" in body
    assert FORMAT_GATE_STOP in body and "212 updates" in body
    floors = spec.probe["minimum_tokens_per_second"]
    assert f"{floors['128']:,.0f} generated tok/s at batch 128, {floors['256']:,.0f}" in body


def test_the_binding_names_every_staged_input_of_run_1b() -> None:
    (build,) = base.by("tools.posttrain_binding", "build")
    options = dict(zip(build[1::2], build[2::2], strict=True))
    assert options["--run"] == "1b" and options["--specification"] == VALUES["SPEC"]
    named = {item.split("=")[0] for item in build[build.index("--input") :] if "=" in item}
    staged = {
        name for name, pin in json.loads(SPEC.read_text())["inputs"].items() if pin["key"] is None
    }
    assert named == staged
    (fetch,) = base.by("tools.posttrain_binding", "fetch")
    assert fetch[fetch.index("--only") + 1] == "b-mixed-final"


def test_the_pod_body_and_the_after_run_steps_are_run_1bs() -> None:
    body = json.loads((ROOT / "configs" / "runpod-pod-a40-s-posttrain-1b.json").read_text())
    first = json.loads((ROOT / "configs" / "runpod-pod-a40-s-posttrain.json").read_text())
    assert body == {**first, "name": "posttrain-a40-s-1b"}
    assert "cp configs/runpod-pod-a40-s-posttrain-1b.json" in text()
    lines = [placeholders(line) for line in base.logical_lines()]
    loops = {}
    for index, line in enumerate(lines):
        match = re.match(r"for name in (.*?); do$", line)
        if match:
            loops[lines[index + 1]] = match.group(1).split()
    sealed = [names for body_line, names in loops.items() if "eval-seal" in body_line]
    executed = [names for body_line, names in loops.items() if "eval-execute" in body_line]
    # Only run 1b's SFT files: base's results are run 1's and stand.
    assert sealed == [["eval-sft-pool-test-greedy", "eval-sft-pool-test-n8"]]
    assert executed == [["eval-sft-pool-dev-greedy", "eval-sft-pool-dev-n8"]]
    (withdraw,) = base.by("scratchcoder.cli", "eval-withdraw")
    assert withdraw[withdraw.index("--endpoint") + 1] == "sft"
    assert "formatting failure" in withdraw[withdraw.index("--reason") + 1]
    dispatched = [line for line in lines if "tools.benchmark_dispatch" in line]
    assert dispatched and all('"posttrain-runs/1b/$POD_ID/' in line for line in dispatched)
    (gate,) = base.by("scratchcoder.cli", "sft-format-gate")
    assert gate[gate.index("--rows") + 1] == "runs/sft-data-20261027/code_dev.jsonl"


def test_no_block_holds_a_comment() -> None:
    found = {line for block in base.blocks() for line in comment_lines(block)}
    assert not found, sorted(found)


@pytest.mark.parametrize("shell", [name for name in ("bash", "zsh") if shutil.which(name)])
def test_every_block_parses_in_the_operators_shells(shell: str, tmp_path: Path) -> None:
    for block in base.blocks():
        pasted = placeholders(block.replace("<reviewed binding sha256>", base.BINDING_SHA256))
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


def test_the_sections_are_numbered_in_order() -> None:
    sections = re.findall(r"^## (\d+)\. ", text(), re.M)
    assert sections == [str(number) for number in range(1, len(sections) + 1)]
