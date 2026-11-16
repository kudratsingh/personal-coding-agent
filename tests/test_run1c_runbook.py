"""Every tool command in run 1c's runbook parses with its real command line, its blocks parse
in the operator's shells, and its figures are the specification's.

Run 1's runbook test supplies the machinery (``tests/test_posttrain_runbook.py``), as for run
1b. A Mac document's test: it names the Mac's container runtime, so it does not run on the Pod.
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

from tools import posttrain_approved
from tools.posttrain_run import FORMAT_GATE_STOP

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "POSTTRAIN_RUN1C_RUNBOOK.md"
ORIGIN = base.ORIGIN
PLACEHOLDERS = {
    "<reviewed binding sha256>": base.BINDING_SHA256,
    "<frozen training rows>": "rows/train.jsonl",
}
VALUES = {
    **base.VALUES,
    "RUN": "1c",
    "CTRL": "runs/posttrain-controller/1c/podpost1c",
    "KNOWN_HOSTS": "/Users/owner/.ssh/posttrain_1c_known_hosts",
    "SPEC_TEMPLATE": "configs/rehearsal-a40-s-post-1c.json",
    "SHARED": "runs/posttrain-shared-1c",
    "SPEC": "runs/posttrain-shared-1c/rehearsal-a40-s-post-1c-launch.json",
    "RUN1_SHARED": "runs/posttrain-shared",
    "RUN1B_SHARED": "runs/posttrain-shared-1b",
    "SFT": "runs/sft-data-20261027",
    "EXPORTS": "runs/posttrain-shared/exports",
    "BINDING": "runs/posttrain-shared-1c/binding.json",
    "AGENT_ROWS": "rows/train.jsonl",
    "COST_LIMIT": "0.6",
    "WALL": "3600",
    "TIMER": "3420",
    "WATCHDOG_COST": "0.59",
    "LAUNCH": f"runs/posttrain-controller/1c/launch-{ORIGIN}",
    "POD_ID": "podpost1c",
    "SELF_DESTRUCT_DEADLINE": str(ORIGIN + 3_420),
    "ART": "runs/posttrain-shared-1c/artifacts",
}


@pytest.fixture(autouse=True)
def run_1c(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(base, "RUNBOOK", RUNBOOK)
    monkeypatch.setattr(base, "VALUES", VALUES)
    yield


def text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def filled(line: str) -> str:
    for placeholder, value in PLACEHOLDERS.items():
        line = line.replace(placeholder, value)
    return line


def test_every_tool_command_parses_with_its_real_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = base.logical_lines
    monkeypatch.setattr(base, "logical_lines", lambda: [filled(line) for line in original()])
    found = base.commands()
    modules = {module for module, _, _ in found}
    for module in (
        "tools.posttrain_approved",
        "tools.posttrain_run",
        "tools.posttrain_ops",
        "tools.posttrain_binding",
        "tools.pilot_feasibility",
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


def test_no_block_holds_a_comment() -> None:
    found = {line for block in base.blocks() for line in comment_lines(block)}
    assert not found, sorted(found)


@pytest.mark.parametrize("shell", [name for name in ("bash", "zsh") if shutil.which(name)])
def test_every_block_parses_in_the_operators_shells(shell: str, tmp_path: Path) -> None:
    for block in base.blocks():
        argv = ["zsh", "-f", "-i", "-n"] if shell == "zsh" else ["bash", "-n"]
        result = subprocess.run(
            argv,
            input=filled(block),
            env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "PS1": ""},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (block.splitlines()[0], result.stderr)
        for error in ("unmatched", "parse error", "event not found", "bad pattern"):
            assert error not in result.stderr, (block.splitlines()[0], result.stderr)


def test_the_bind_and_the_binding_are_run_1cs() -> None:
    (bind,) = base.by("tools.posttrain_approved", "run1c-bind")
    options = dict(zip(bind[1::2], bind[2::2], strict=True))
    assert options == {"--sft-train": VALUES["AGENT_ROWS"], "--output": VALUES["SPEC"]}
    (build,) = base.by("tools.posttrain_binding", "build")
    options = dict(zip(build[1::2], build[2::2], strict=True))
    assert options["--run"] == "1c" and options["--specification"] == VALUES["SPEC"]
    named = {item.split("=")[0] for item in build[build.index("--input") :] if "=" in item}
    template = json.loads((ROOT / VALUES["SPEC_TEMPLATE"]).read_text())
    assert named == {name for name, pin in template["inputs"].items() if pin["key"] is None}
    (fetch,) = base.by("tools.posttrain_binding", "fetch")
    assert fetch[fetch.index("--only") + 1] == "sft-weights"
    (check,) = base.by("tools.posttrain_binding", "check-exports")
    assert [item for item in check if item.startswith("pool-")] == [
        f"pool-dev={VALUES['EXPORTS']}/dev-export.jsonl"
    ]
    for module in ("tools.posttrain_ops", "tools.pilot_feasibility"):
        for argv in base.by(module):
            if "--specification" in argv:
                assert argv[argv.index("--specification") + 1] == VALUES["SPEC"], argv


def test_the_figures_are_the_specifications() -> None:
    body = text()
    floors = posttrain_approved.RUN1C_FLOORS
    assert f"**{floors['128']:,.0f}** generated tok/s at batch 128" in body
    assert f"**{floors['256']:,.0f}** at batch 256" in body
    reference = posttrain_approved.RUN1C_SFT_REFERENCE
    assert f"**{reference:,.2f}** slots/s, floor {0.9 * reference:,.2f}" in body
    assert "**$1.50**" in body and "**120% of its estimate**" in body and "**900 s**" in body
    parent = posttrain_approved.RUN1C_PARENT["sha256"]
    assert body.count(parent) == 1
    assert FORMAT_GATE_STOP in body and "cost_limit" in body
    names = [entry["name"] for entry in posttrain_approved.run1c_stages()[3]["files"]]
    assert all(f"`{name}`" in body for name in names)


def test_the_pod_body_is_run_1bs_under_run_1cs_name() -> None:
    body = json.loads((ROOT / "configs" / "runpod-pod-a40-s-posttrain-1c.json").read_text())
    run1b = json.loads((ROOT / "configs" / "runpod-pod-a40-s-posttrain-1b.json").read_text())
    assert body == {**run1b, "name": "posttrain-a40-s-1c"}
    assert "cp configs/runpod-pod-a40-s-posttrain-1c.json" in text()
    assert "posttrain-a40-s-1c" in text()


def test_the_owner_steps_are_all_there() -> None:
    body = text()
    assert "Only after the owner's explicit final go for run 1c on the bound specification" in body
    assert "**Re-run the format gate on the CPU (required).**" in body
    assert "**The local agent evaluation is a separate step.**" in body
    assert body.count("mac_window_check check") == 2
    assert "--specification /workspace/posttrain/specification.json" in body
    assert "eval-seal" not in body and "rl-screen" not in body and "pool-test" not in body
    assert "no digest is guessed here" in body


PUBLIC_RUN_DIRECTORIES = {
    "runs/posttrain-controller",
    "runs/posttrain-shared",
    "runs/posttrain-shared-1b",
    "runs/posttrain-shared-1c",
    "runs/posttrain-parents",
    "runs/posttrain-smoke-1c",
    "runs/task-pool-20261025",
    "runs/sft-data-20261027",
    "runs/tokenizer-freeze-65536-20261020",
}
"""The run directories the runbook may name: those earlier runbooks name and run 1c's own. The
training rows are named only as ``$AGENT_ROWS``."""


def test_the_runbook_names_no_private_identifier() -> None:
    body = text()
    assert not re.search(r"\bD\d\d\b|\bF\d\d\b", body)
    named = set(re.findall(r"(?<![\w-])runs/[A-Za-z0-9._-]+", body))
    assert named <= PUBLIC_RUN_DIRECTORIES, sorted(named - PUBLIC_RUN_DIRECTORIES)
