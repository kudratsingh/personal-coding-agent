"""The pilot stages driver: synthetic tasks and fake executors only; nothing runs code."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import tools.pilot_stages as tool
from scratchcoder.eval.tasks import Task
from scratchcoder.rl.attempts import ScreeningHeader, read_attempts_file
from scratchcoder.rl.rewards import Attempt
from scratchcoder.sandbox.reward import COMPLETED, RewardResult
from scratchcoder.sandbox.runner import Result

# --------------------------------------------------------------------------------------
# Synthetic tasks.
# --------------------------------------------------------------------------------------


def make_task(
    task_id: str,
    *,
    split: str = "train",
    hidden: str | None = None,
    solution: str | None = "def f(x):\n    return x + 1\n",
    mode: str = "function",
    source: str = "nvidia/OpenCodeInstruct",
) -> Task:
    if mode == "io":
        cases = [{"input": "1\n", "output": "2\n"}, {"input": "5\n", "output": "6\n"}]
        tests = json.dumps(cases)
        return Task(
            id=task_id,
            prompt="Read an integer and print it plus one.",
            public_tests=json.dumps(cases[:1]),
            hidden_tests=hidden if hidden is not None else tests,
            source=source,
            license="project-authored",
            written_by="human",
            split=split,
            mode="io",
            solution="print(int(input()) + 1)\n" if solution is not None else None,
        )
    return Task(
        id=task_id,
        prompt="Write f(x) that returns x plus one.",
        public_tests="assert f(0) == 1",
        hidden_tests=hidden
        if hidden is not None
        else "assert f(1) == 2\nassert f(2) == 3\nassert f(5) == 6",
        source=source,
        license="project-authored",
        written_by="human",
        split=split,
        mode=mode,
        entry_point="f",
        solution=solution,
    )


def write_tasks(path: Path, tasks: list[Task]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(task), sort_keys=True) + "\n" for task in tasks), encoding="utf-8"
    )
    return path


def fake_execute(code: str, tests: str, mode: str) -> Result:
    """A deterministic stand-in for the sandbox: references pass, trivial programs fail,
    except where a task's tests only compare with ``None`` or its reference is marked broken."""
    if "BROKEN" in code:
        return Result("failed", "AssertionError\n", 1)
    if mode == "io":
        if code.startswith("pass") or "sys.stdin.read()" in code:
            return Result("failed", "", 1)
        return Result("passed", "", 0)
    if "*args" in code:
        if "return None" in code and "== None" in tests:
            return Result("passed", "", 0)
        return Result("failed", "AssertionError\n", 1)
    return Result("passed", "", 0)


def receipt_for(path: Path, argv: list[str], module: str, *, status: str = "passed") -> Path:
    from scratchcoder.sandbox import runner

    record = {
        "kind": "mac-execution-window-check",
        "status": status,
        "checked_at": datetime.now(UTC).isoformat(),
        "command": ["env", "-u", "VIRTUAL_ENV", "uv", "run", "python", "-m", module, *argv],
        "checks": {
            "image": {
                "passed": True,
                "image": runner.IMAGE,
                "expected_id": "sha256:" + "2" * 64,
                "found_id": "sha256:" + "2" * 64,
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# tools.admission_check: the dev and remainder sets behave exactly as before.
# --------------------------------------------------------------------------------------


def _admission_fixture(root: Path) -> dict[str, Path]:
    dev = [
        make_task("d/sound", split="dev"),
        make_task("d/none", split="dev", hidden="assert f(1) == None\nassert f(2) == None"),
        make_task("d/broken", split="dev", solution="def f(x):\n    return 'BROKEN'\n"),
        make_task("d/quarantined", split="dev"),
        make_task("d/reviewed", split="dev"),
        make_task("d/io", split="dev", mode="io", source="BAAI/TACO"),
    ]
    train = [
        make_task("r/sound"),
        make_task("r/io", mode="io", source="BAAI/TACO"),
        make_task("r/excluded"),
        make_task("r/verifier"),
        make_task("r/none", hidden="assert f(3) == None"),
    ]
    paths = {
        "dev": write_tasks(root / "dev.jsonl", dev),
        "train": write_tasks(root / "train.jsonl", train),
        "dev-adjudication": root / "adjudication.jsonl",
        "screening": root / "screening.json",
        "exclusions": root / "exclusions.json",
        "dev-quarantine": root / "quarantine.json",
        "verifier-filter": root / "verifier-filter.json",
        "cache": root / "cache.jsonl",
    }
    paths["dev-adjudication"].write_text(
        json.dumps({"id": "d/quarantined", "status": "quarantined"})
        + "\n"
        + json.dumps({"id": "d/sound", "status": "clear"})
        + "\n",
        encoding="utf-8",
    )
    paths["screening"].write_text(
        json.dumps({"eligible": [task.id for task in train]}), encoding="utf-8"
    )
    paths["exclusions"].write_text(
        json.dumps({"candidates": [{"id": "r/excluded", "status": "excluded_overlap"}]}),
        encoding="utf-8",
    )
    paths["dev-quarantine"].write_text(
        json.dumps(
            {
                "quarantine_static_clear": [{"id": "d/reviewed", "reasons": ["contradictory"]}],
                "retained_existing": {},
                "provisional_heuristic": [{"id": "d/sound", "reasons": ["heuristic"]}],
                "quarantine_recommended_none_only": {"ids": ["d/none"]},
            }
        ),
        encoding="utf-8",
    )
    paths["verifier-filter"].write_text(
        json.dumps({"verifier_quarantine_clear": ["r/verifier"], "verifier_provisional": []}),
        encoding="utf-8",
    )
    return paths


def admission_golden_digests(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Run ``tools.admission_check`` execute then decide on a fixed fixture, and hash what it
    wrote. Cache rows are sorted (their order is completion order); every other output is
    hashed as written."""
    from scratchcoder.eval import harness
    from tools import admission_check

    monkeypatch.setattr(harness, "isolated_execution", fake_execute)
    paths = _admission_fixture(root)
    common: list[str] = []
    for name in (
        "dev",
        "dev-adjudication",
        "train",
        "screening",
        "exclusions",
        "dev-quarantine",
        "verifier-filter",
        "cache",
    ):
        common += [f"--{name}", str(paths[name])]
    argv = ["execute", *common, "--receipt", str(root / "receipt.json"), "--workers", "1"]
    argv += ["--run-id", "golden"]
    receipt_for(root / "receipt.json", argv, "tools.admission_check")
    assert admission_check.main(argv) == 0
    out = root / "decisions"
    image = "sha256:" + "2" * 64
    assert admission_check.main(["decide", *common, "--image-id", image, "--out", str(out)]) == 0
    # The evaluator identity hashes the verdict code and names the executor; it is replaced
    # by a placeholder so that this test pins admission_check's behaviour, not that code.
    evaluator = admission_check.identities(image).evaluator_sha256

    def normalized(text: str) -> bytes:
        return text.replace(evaluator, "<evaluator>").encode()

    rows = sorted(paths["cache"].read_text(encoding="utf-8").splitlines())
    digests = {"cache": hashlib.sha256(normalized("\n".join(rows))).hexdigest()}
    for name in ("decisions-dev.jsonl", "decisions-remainder.jsonl", "summary.json"):
        text = (out / name).read_text(encoding="utf-8")
        digests[name] = hashlib.sha256(normalized(text)).hexdigest()
    return digests


GOLDEN_ADMISSION = {
    "cache": "928648c1e7d640b860dca618c2670fff978665a6bd6352728c49526132726484",
    "decisions-dev.jsonl": "e902cd754b08834e1eca3778cdba44e3f2c660a4d4490dc050c88c86944285e1",
    "decisions-remainder.jsonl": "f6af9c419159279b05df2c33c2d56c2e30fffd3abfa8a8012906b76db00133ec",
    "summary.json": "dd671fbca29f3562ce0b2b9a5394f1d2af5e70645d1e8a4877f7140542ddcea3",
}
"""Digests of ``tools.admission_check``'s outputs on the fixture above, computed with the
code before it was generalized for the pilot (commit dcc84b8)."""


def test_dev_and_remainder_admission_outputs_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    assert admission_golden_digests(tmp_path, monkeypatch) == GOLDEN_ADMISSION


# --------------------------------------------------------------------------------------
# A synthetic pilot run.
# --------------------------------------------------------------------------------------

INTAKE_ORDER = [
    "unsupported_schema",
    "over_context",
    "oversized_for_sandbox",
    "over_rl_prompt_tokens",
    "choice_contamination",
    "benchmark_contamination",
    "id_in_pool",
    "duplicate_id",
    "dev_contamination",
    "near_duplicate",
    "dev_paraphrase",
    "dev_family",
]


@pytest.fixture(autouse=True)
def _no_agent_and_no_waits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scratchcoder.eval import parallel
    from tools import task_pool

    monkeypatch.setattr(tool, "REGISTRY", tmp_path / "registry" / "pilot-stages-registry.jsonl")
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.setattr(task_pool, "RETRY_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(parallel, "host_backoff", lambda failures: 0.0)


def pilot_task(task_id: str, **fields: Any) -> Task:
    """A pilot candidate whose reference is its own text, so every call is distinct."""
    from dataclasses import replace

    task = make_task(task_id, **fields)
    assert task.solution is not None
    return replace(task, solution=task.solution + f"# {task_id}\n")


def pilot_tasks() -> list[Task]:
    three = "assert f(1) == 2\nassert f(2) == 3\nassert f(5) == 6"
    return [
        pilot_task("p/mixed-1"),
        pilot_task("p/mixed-2"),
        pilot_task("p/zero"),
        pilot_task("p/ones"),
        pilot_task("p/broken", solution="def f(x):\n    return 'BROKEN'\n"),
        pilot_task(
            "p/none",
            hidden="assert f(1) == None\nassert f(2) == None\nassert f(3) == None",
            solution="def f(x):\n    return None\n",
        ),
        pilot_task("p/form", hidden="assert f(1) in [2, 3]\n" + three.split("\n", 1)[1]),
        pilot_task(
            "p/const",
            hidden="assert f(1) == 2\nassert f(2) == 2\nassert f(3) == 2",
            solution="def f(x):\n    return 2\n",
        ),
    ]


def tiny_checkpoint(root: Path) -> tuple[Path, Path]:
    import torch

    from scratchcoder.common.config import ModelConfig
    from scratchcoder.model.gpt import GPT
    from scratchcoder.tokenizer.bpe import ByteBPE

    tokenizer = ByteBPE()
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, width=16, layers=1, heads=2, context=512)
    checkpoint = root / "weights" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "config": {"model": asdict(config)},
            "tokenizer_hash": tokenizer.sha256,
            "model": GPT(config).state_dict(),
        },
        checkpoint,
    )
    return checkpoint, tokenizer_path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pilot_inputs(
    root: Path, tasks: list[Task] | None = None, remainder: list[Task] | None = None
) -> dict[str, Path]:
    tasks = pilot_tasks() if tasks is None else tasks
    remainder = [make_task("r/1"), make_task("r/2")] if remainder is None else remainder
    after = root / "after-scan"
    paths = {
        "pilot": write_tasks(after / "pilot-clear.jsonl", tasks),
        "remainder": write_tasks(after / "remainder-clear.jsonl", remainder),
    }
    (after / "SHA256SUMS").write_text(
        "".join(f"{sha(path)}  {path.name}\n" for path in paths.values()), encoding="utf-8"
    )
    paths["scan_summary"] = root / "scan" / "summary.json"
    paths["scan_summary"].parent.mkdir(parents=True)
    paths["scan_summary"].write_text(
        json.dumps(
            {
                "format": "holdout-integrity-scan-v1",
                "counts": {
                    "subjects": {
                        "pilot": {
                            "clear": len(tasks),
                            "excluded": 1,
                            "flagged": 2,
                            "total": len(tasks) + 3,
                        },
                        "remainder": {
                            "clear": len(remainder),
                            "excluded": 1,
                            "flagged": 1,
                            "total": len(remainder) + 2,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    counts = {name: 0 for name in INTAKE_ORDER}
    counts.update(rows=1000, kept=len(tasks) + 3, unsupported_schema=100, dev_contamination=700)
    counts["near_duplicate"] = 1000 - 100 - 700 - counts["kept"]
    paths["intake_record"] = root / "intake" / "record.json"
    paths["intake_record"].parent.mkdir(parents=True)
    paths["intake_record"].write_text(
        json.dumps(
            {
                "kind": "pilot-intake",
                "sealed_files_opened": False,
                "counts": counts,
                "rules": {"order": INTAKE_ORDER},
            }
        ),
        encoding="utf-8",
    )
    paths["spec"] = root / "spec.json"
    paths["spec"].write_text(
        json.dumps(
            {
                "verification": {
                    "timeout_seconds": 2.0,
                    "io_seconds_per_case": 0.25,
                    "memory_mb": 128,
                    "workers": 4,
                }
            }
        ),
        encoding="utf-8",
    )
    paths["checkpoint"], paths["tokenizer"] = tiny_checkpoint(root)
    paths["checkpoint_record"] = paths["checkpoint"].parent / "record.json"
    paths["checkpoint_record"].write_text(
        json.dumps({"sft": {"selected": {"sha256": sha(paths["checkpoint"])}}}), encoding="utf-8"
    )
    return paths


def init_argv(paths: dict[str, Path], out: Path) -> list[str]:
    argv = ["init", "--out", str(out)]
    for name, path in paths.items():
        argv += [f"--{name.replace('_', '-')}", str(path)]
    return argv


def new_run(root: Path, **inputs: Any) -> tuple[Path, dict[str, Path]]:
    paths = pilot_inputs(root, **inputs)
    run = root / "run"
    assert tool.main(init_argv(paths, run)) == 0
    return run, paths


class Counting:
    """A fake executor that records every call."""

    def __init__(self, behaviour: Callable[[str, str, str], Result] = fake_execute) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.behaviour = behaviour

    def __call__(self, code: str, tests: str, mode: str) -> Result:
        self.calls.append((code, tests, mode))
        return self.behaviour(code, tests, mode)


def execute_stage(run: Path, stage: str, *, receipt: bool = True, **options: Any) -> int:
    argv = [stage, "--run", str(run)]
    if receipt:
        path = run.parent / "receipts" / f"{stage}.json"
        argv += ["--receipt", str(path)]
        receipt_for(path, argv, "tools.pilot_stages")
    return tool.main(argv, options=options)


class FakeGenerator:
    """Deterministic samples by task: ``mixed`` has one solving sample, ``zero`` no program,
    ``ones`` solves every time."""

    tokenizer_sha256 = "a" * 64

    def __init__(self, files: Any = None) -> None:
        self.chunks: list[int] = []

    def __call__(self, tasks: Any) -> list[Attempt]:
        self.chunks.append(len(tasks))
        rows = []
        for task in tasks:
            for sample in range(8):
                if task.id.endswith("zero"):
                    code = None
                elif task.id.endswith("ones") or sample == 0:
                    code = "good"
                else:
                    code = "bad"
                rows.append(Attempt(task.id, sample, code, 5))
        return rows

    def environment(self) -> dict[str, Any]:
        return {"torch_threads": 3, "torch": "x", "numpy": "y"}


def fake_score(task: Task, code: str) -> RewardResult:
    return RewardResult(COMPLETED, 1.0 if code == "good" else 0.0, 1, 1)


def gate_file(
    root: Path,
    checkpoint: Path,
    *,
    passed: bool = True,
    name: str | None = None,
    **changes: Any,
) -> Path:
    """A CPU format-gate record of ``checkpoint`` (and the tokenizer beside its folder),
    written now, in the current gate format; ``changes`` override its fields."""
    from scratchcoder.eval import format_gate

    tokenizer = checkpoint.parent.parent / "tokenizer.json"
    path = root / "gate" / (name or f"gate-{passed}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "format": format_gate.FORMAT,
        "version": format_gate.VERSION,
        "passed": passed,
        "failed_checks": [] if passed else ["close"],
        "inputs": {
            "checkpoint_sha256": sha(checkpoint),
            "tokenizer_sha256": sha(tokenizer) if tokenizer.exists() else "0" * 64,
        },
        "device": "cpu",
        "torch_threads": 4,
        "created_at": datetime.now(UTC).isoformat(),
    }
    record.update(changes)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def through_select(run: Path, executor: Counting | None = None) -> None:
    executor = executor or Counting()
    assert execute_stage(run, "verify", execute=executor) == 0
    assert execute_stage(run, "admit", execute=executor) == 0
    assert tool.main(["select", "--run", str(run)]) == 0


def generate(run: Path, paths: dict[str, Path], **options: Any) -> int:
    options.setdefault("generator", FakeGenerator)
    gate = gate_file(run.parent, paths["checkpoint"])
    return tool.main(["generate", "--run", str(run), "--format-gate", str(gate)], options=options)


def stop_reason(run: Path) -> str:
    return str(json.loads((run / "stop.json").read_text())["reason"])


# --------------------------------------------------------------------------------------
# The whole run.
# --------------------------------------------------------------------------------------


def test_the_whole_pilot_runs_end_to_end_on_fakes(tmp_path: Path) -> None:
    run, paths = new_run(tmp_path)
    executor = Counting()
    through_select(run, executor)
    assert generate(run, paths) == 0
    assert execute_stage(run, "score", score=fake_score) == 0
    assert tool.main(["report", "--run", str(run)]) == 0

    verify = json.loads((run / "verify" / "result.json").read_text())
    assert verify["verified"] == 7 and verify["dropped_by_reason"] == {"public:AssertionError": 1}
    admit = json.loads((run / "admit" / "result.json").read_text())
    assert admit["decisions"]["excluded"] == 1  # the None-only task
    assert admit["after_admission_dropped"] == {"constant_baseline": 1, "not_parseable": 1}
    assert admit["admitted"] == 4 and admit["runs_unresolved"] == 0
    report = json.loads((run / "report" / "report.json").read_text())
    screening = report["screening"]
    assert (screening["N"], screening["E"]) == (4, 2)
    assert screening["z"]["successes"] == 1 and screening["z"]["trials"] == 4
    assert screening["all_one_groups"] == 1
    assert screening["all_zero_sample_classes"]["formatting"]["successes"] == 8
    assert report["P_combined_pool"] == 4 and report["shortfall_to_floor"] == 316
    assert report["remainder"] == {
        "before_scan": 4,
        "excluded": 1,
        "flagged": 1,
        "after_scan": 2,
        "treatment": "flagged and excluded are both removed",
    }
    decision = report["decision"]
    assert decision["rule"] == 4 and decision["K"] == 727 and decision["low_confidence"]
    # K at the Wilson lower bound needs more than 1,000 candidates: no proposal.
    assert decision["decision"] == "no_proposal_owner_choice"
    # Every sandbox call went through the ledger, and the budget is recorded by stage.
    ledger = tool.ledger_counts(run / "budget.jsonl")
    assert (ledger["verify"], ledger["admit"]) == (16, 35)
    assert len(executor.calls) == 16 + 35
    assert ledger["score"] == 3 * 8  # Every sample with a program; the no-program task none.
    assert report["budget"] == dict(ledger)
    text = (run / "report" / "report.md").read_text()
    assert "different numerical executions of the same intended protocol" in text
    assert "planning heuristics" in text
    # Checksums cover the whole run.
    sums = (run / "SHA256SUMS").read_text().splitlines()
    names = {line.split("  ", 1)[1] for line in sums}
    assert {"inputs.json", "budget.jsonl", "score/screening.json", "report/report.json"} <= names
    for line in sums:
        digest, name = line.split("  ", 1)
        assert sha(run / name) == digest


def test_every_stage_refuses_to_run_twice_or_out_of_order(tmp_path: Path) -> None:
    run, paths = new_run(tmp_path)
    assert tool.main(["select", "--run", str(run)]) == tool.EXIT_REFUSED
    assert execute_stage(run, "admit", execute=Counting()) == tool.EXIT_REFUSED
    assert tool.main(["report", "--run", str(run)]) == tool.EXIT_REFUSED
    assert execute_stage(run, "verify", execute=Counting()) == 0
    assert execute_stage(run, "verify", execute=Counting()) == tool.EXIT_EXISTS


def test_init_refuses_an_existing_output_and_inconsistent_inputs(tmp_path: Path) -> None:
    paths = pilot_inputs(tmp_path)
    existing = tmp_path / "exists"
    existing.mkdir()
    assert tool.main(init_argv(paths, existing)) == tool.EXIT_EXISTS
    # A file that differs from its SHA256SUMS line.
    sums = paths["pilot"].parent / "SHA256SUMS"
    original = sums.read_text()
    sums.write_text(original.replace(sha(paths["pilot"]), "0" * 64), encoding="utf-8")
    assert tool.main(init_argv(paths, tmp_path / "a")) == tool.EXIT_REFUSED
    sums.write_text(original, encoding="utf-8")
    # A checkpoint that is not the record's selected one.
    record = paths["checkpoint_record"]
    record.write_text(json.dumps({"sft": {"selected": {"sha256": "1" * 64}}}), encoding="utf-8")
    assert tool.main(init_argv(paths, tmp_path / "b")) == tool.EXIT_REFUSED
    record.write_text(
        json.dumps({"sft": {"selected": {"sha256": sha(paths["checkpoint"])}}}), encoding="utf-8"
    )
    # A scan count that differs from the clear file.
    summary = json.loads(paths["scan_summary"].read_text())
    summary["counts"]["subjects"]["pilot"]["clear"] += 1
    paths["scan_summary"].write_text(json.dumps(summary), encoding="utf-8")
    assert tool.main(init_argv(paths, tmp_path / "c")) == tool.EXIT_REFUSED
    assert not any((tmp_path / name).exists() for name in ("a", "b", "c"))


def test_init_refuses_a_sealed_input_before_opening_it(tmp_path: Path) -> None:
    paths = pilot_inputs(tmp_path)
    sealed = tmp_path / "split" / "holdout.jsonl"
    sealed.parent.mkdir()
    sealed.write_bytes(paths["pilot"].read_bytes())
    paths["pilot"] = sealed
    assert tool.main(init_argv(paths, tmp_path / "run")) == tool.EXIT_SEALED
    assert not (tmp_path / "run").exists()


def test_a_stage_reads_only_unchanged_inputs(tmp_path: Path) -> None:
    run, paths = new_run(tmp_path)
    write_tasks(paths["pilot"], pilot_tasks()[:2])
    executor = Counting()
    assert execute_stage(run, "verify", execute=executor) == tool.EXIT_REFUSED
    assert executor.calls == []


# --------------------------------------------------------------------------------------
# Receipts, locks and the runtime binding.
# --------------------------------------------------------------------------------------


def test_executing_stages_refuse_without_a_passing_receipt_for_this_command(
    tmp_path: Path,
) -> None:
    run, _ = new_run(tmp_path)
    executor = Counting()
    assert execute_stage(run, "verify", receipt=False, execute=executor) == tool.EXIT_REFUSED
    receipt = tmp_path / "receipt.json"
    other = ["verify", "--run", str(tmp_path / "elsewhere"), "--receipt", str(receipt)]
    receipt_for(receipt, other, "tools.pilot_stages")
    argv = ["verify", "--run", str(run), "--receipt", str(receipt)]
    assert tool.main(argv, options={"execute": executor}) == tool.EXIT_REFUSED
    receipt_for(receipt, argv, "tools.pilot_stages", status="failed")
    assert tool.main(argv, options={"execute": executor}) == tool.EXIT_REFUSED
    stale = json.loads(receipt.read_text())
    stale.update(status="passed", checked_at="2020-01-01T00:00:00+00:00")
    receipt.write_text(json.dumps(stale), encoding="utf-8")
    assert tool.main(argv, options={"execute": executor}) == tool.EXIT_REFUSED
    assert executor.calls == []
    assert tool.ledger_counts(run / "budget.jsonl") == {}
    assert not (run / "verify").exists() and not (run / "stop.json").exists()


def test_an_ssh_agent_in_the_environment_refuses_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = new_run(tmp_path)
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent")
    executor = Counting()
    assert execute_stage(run, "verify", execute=executor) == tool.EXIT_REFUSED
    assert executor.calls == []


def test_a_later_stage_under_another_image_is_refused(tmp_path: Path) -> None:
    run, _ = new_run(tmp_path)
    assert execute_stage(run, "verify", execute=Counting()) == 0
    runtime = json.loads((run / "runtime.json").read_text())
    runtime["image_id"] = "sha256:" + "9" * 64
    (run / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
    executor = Counting()
    assert execute_stage(run, "admit", execute=executor) == tool.EXIT_REFUSED
    assert executor.calls == []


def test_another_process_holding_the_run_is_refused(tmp_path: Path) -> None:
    import fcntl

    run, _ = new_run(tmp_path)
    with (run / ".lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert execute_stage(run, "verify", execute=Counting()) == tool.EXIT_LOCKED


# --------------------------------------------------------------------------------------
# The budget ledger and the caps.
# --------------------------------------------------------------------------------------


def test_the_ledger_charges_before_each_call_and_never_passes_a_cap(tmp_path: Path) -> None:
    path = tmp_path / "budget.jsonl"
    path.write_text('{"stage": "verify"}\n{"stage": "verify"}\n{"sta', encoding="utf-8")
    assert tool.ledger_counts(path) == {"verify": 2, "unknown": 1}
    path.write_text('{"stage": "verify"}\n{"stage": "verify"}\n', encoding="utf-8")
    budget = tool.Budget(path, "admit", cap=5, run_cap=4)
    budget.require(2)
    with pytest.raises(tool.CapReached):
        budget.require(3)
    calls: list[int] = []

    def record(value: int) -> int:
        calls.append(value)
        return value

    counted = budget.wrap(record)
    assert counted(1) == 1 and counted(2) == 2
    with pytest.raises(tool.CapReached):
        counted(3)
    assert calls == [1, 2]  # The refused call was not made.
    assert tool.ledger_counts(path) == {"verify": 2, "admit": 2}
    assert budget.record()["calls_this_invocation"] == 2


def test_a_plan_over_the_stage_cap_stops_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = new_run(tmp_path)
    monkeypatch.setitem(tool.CALL_CAPS, "verify", 10)
    executor = Counting()
    assert execute_stage(run, "verify", execute=executor) == tool.EXIT_STOP
    assert executor.calls == [] and stop_reason(run) == "cap_reached"
    # After a stop only the report runs; it draws no decision.
    assert execute_stage(run, "verify", execute=executor) == tool.EXIT_EXISTS
    assert execute_stage(run, "admit", execute=executor) == tool.EXIT_REFUSED
    assert tool.main(["report", "--run", str(run)]) == 0
    report = json.loads((run / "report" / "report.json").read_text())
    assert report["decision"]["rule"] == 0 and report["screening"] is None
    caps = [item for item in report["stopping_criteria"] if "cap" in item["criterion"]]
    assert caps[0]["status"] == "fired"


def test_the_run_budget_counts_retries_and_stops_mid_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = new_run(tmp_path)
    # Distinct calls fit, but every call needs a retry: the cap is reached mid-stage.
    monkeypatch.setattr(tool, "RUN_CALL_CAP", 20)
    tries: dict[tuple[str, str, str], int] = {}

    def flaky(code: str, tests: str, mode: str) -> Result:
        key = (code, tests, mode)
        tries[key] = tries.get(key, 0) + 1
        if tries[key] == 1:
            return Result("infrastructure_error")
        return fake_execute(code, tests, mode)

    executor = Counting(flaky)
    assert execute_stage(run, "verify", execute=executor) == tool.EXIT_STOP
    assert stop_reason(run) == "cap_reached"
    assert len(executor.calls) == 20 == tool.ledger_counts(run / "budget.jsonl")["verify"]


def test_admission_has_its_own_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = new_run(tmp_path)
    assert execute_stage(run, "verify", execute=Counting()) == 0
    monkeypatch.setitem(tool.CALL_CAPS, "admit", 6 * 5 - 1)
    executor = Counting()
    assert execute_stage(run, "admit", execute=executor) == tool.EXIT_STOP
    assert executor.calls == [] and stop_reason(run) == "cap_reached"


def test_rewards_have_their_own_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, paths = new_run(tmp_path)
    through_select(run)
    assert generate(run, paths) == 0
    monkeypatch.setitem(tool.CALL_CAPS, "score", 10)
    assert execute_stage(run, "score", score=fake_score) == tool.EXIT_STOP
    assert stop_reason(run) == "cap_reached"
    assert tool.ledger_counts(run / "budget.jsonl")["score"] == 0


# --------------------------------------------------------------------------------------
# Resume from the caches.
# --------------------------------------------------------------------------------------


class Interrupted(BaseException):
    """Stands in for a killed process (not a stopping criterion)."""


def test_an_interrupted_verification_resumes_from_its_cache(tmp_path: Path) -> None:
    run, _ = new_run(tmp_path)
    count = {"n": 0}

    def dies(code: str, tests: str, mode: str) -> Result:
        count["n"] += 1
        if count["n"] == 5:
            raise Interrupted
        return fake_execute(code, tests, mode)

    first = Counting(dies)
    with pytest.raises(Interrupted):
        execute_stage(run, "verify", execute=first)
    assert not (run / "stop.json").exists() and not (run / "verify" / "result.json").exists()
    cache = next((run / "verify").glob("results-*.jsonl"))
    cached = {json.loads(line)["key"] for line in cache.read_text().splitlines()}
    assert 0 < len(cached) < 16
    second = Counting()
    assert execute_stage(run, "verify", execute=second) == 0
    from tools.task_pool import call_key

    rerun = {call_key(*call) for call in second.calls}
    assert not rerun & cached and len(rerun | cached) == 16
    result = json.loads((run / "verify" / "result.json").read_text())
    assert result["counters"]["cached"] == len(cached) and result["verified"] == 7


def test_an_interrupted_admission_resumes_breadth_first_from_its_cache(tmp_path: Path) -> None:
    run, _ = new_run(tmp_path)
    assert execute_stage(run, "verify", execute=Counting()) == 0
    count = {"n": 0}

    def dies(code: str, tests: str, mode: str) -> Result:
        count["n"] += 1
        if count["n"] == 9:
            raise Interrupted
        return fake_execute(code, tests, mode)

    with pytest.raises(Interrupted):
        execute_stage(run, "admit", execute=Counting(dies))
    rows = [json.loads(line) for line in (run / "admit" / "cache.jsonl").read_text().splitlines()]
    # Breadth first: no control runs before every task's reference runs.
    assert rows and {row["kind"] for row in rows} == {"reference"}
    second = Counting()
    assert execute_stage(run, "admit", execute=second) == 0
    assert len(second.calls) + len(rows) == 7 * 5
    result = json.loads((run / "admit" / "result.json").read_text())
    assert result["runs_unresolved"] == 0 and result["admitted"] == 4


# --------------------------------------------------------------------------------------
# Every stopping criterion.
# --------------------------------------------------------------------------------------


def test_under_twenty_percent_verified_stops(tmp_path: Path) -> None:
    broken = "def f(x):\n    return 'BROKEN'\n"
    tasks = [pilot_task("p/ok")] + [pilot_task(f"p/b{i}", solution=broken) for i in range(5)]
    run, _ = new_run(tmp_path, tasks=tasks)
    assert execute_stage(run, "verify", execute=Counting()) == tool.EXIT_STOP
    assert stop_reason(run) == "under_verified"
    result = json.loads((run / "verify" / "result.json").read_text())
    assert result["status"] == "stopped" and result["verified"] == 1
    assert (run / "verify" / "verified.jsonl").exists()  # Recorded for the report.


def test_a_persistent_sandbox_failure_stops_verification(tmp_path: Path) -> None:
    run, _ = new_run(tmp_path)
    executor = Counting(lambda code, tests, mode: Result("infrastructure_error"))
    assert execute_stage(run, "verify", execute=executor) == tool.EXIT_STOP
    assert stop_reason(run) == "sandbox_failure"


def test_a_missing_supervisor_protection_stops_at_once(tmp_path: Path) -> None:
    run, _ = new_run(tmp_path)
    unprotected = Result("infrastructure_error", detail="supervisor_unprotected")
    executor = Counting(lambda code, tests, mode: unprotected)
    assert execute_stage(run, "verify", execute=executor) == tool.EXIT_STOP
    assert stop_reason(run) == "containment_failure"
    # Never retried, and no queued call starts: only those already running (one per worker).
    assert len(executor.calls) <= 4


def test_too_many_unresolved_admission_runs_stop(tmp_path: Path) -> None:
    run, _ = new_run(tmp_path)
    assert execute_stage(run, "verify", execute=Counting()) == 0

    def host_down_for_one(code: str, tests: str, mode: str) -> Result:
        if "return 2" in code:  # p/const's reference: three unresolved runs of 35.
            return Result("infrastructure_error")
        return fake_execute(code, tests, mode)

    assert execute_stage(run, "admit", execute=Counting(host_down_for_one)) == tool.EXIT_STOP
    assert stop_reason(run) == "admission_unresolved"
    result = json.loads((run / "admit" / "result.json").read_text())
    assert result["runs_unresolved"] == 3 and result["unresolved_share"] > 0.05


def test_a_host_that_stays_down_stops_admission_as_a_sandbox_failure(tmp_path: Path) -> None:
    tasks = [pilot_task(f"p/{i}") for i in range(6)]
    run, _ = new_run(tmp_path, tasks=tasks)
    assert execute_stage(run, "verify", execute=Counting()) == 0
    down = Counting(lambda code, tests, mode: Result("infrastructure_error"))
    assert execute_stage(run, "admit", execute=down) == tool.EXIT_STOP
    assert stop_reason(run) == "sandbox_failure"


def test_a_failing_format_gate_stops_and_another_checkpoints_gate_is_refused(
    tmp_path: Path,
) -> None:
    run, paths = new_run(tmp_path)
    through_select(run)
    other = tmp_path / "other.pt"
    other.write_bytes(b"other weights")
    wrong = gate_file(tmp_path / "wrong", other)
    argv = ["generate", "--run", str(run), "--format-gate", str(wrong)]
    assert tool.main(argv, options={"generator": FakeGenerator}) == tool.EXIT_REFUSED
    failing = gate_file(tmp_path, paths["checkpoint"], passed=False)
    argv = ["generate", "--run", str(run), "--format-gate", str(failing)]
    assert tool.main(argv, options={"generator": FakeGenerator}) == tool.EXIT_STOP
    assert stop_reason(run) == "format_gate_failed"
    assert not (run / "generate" / "attempts.jsonl").exists()


class Clock:
    def __init__(self, chunk_seconds: float) -> None:
        self.now = 0.0
        self.chunk_seconds = chunk_seconds

    def __call__(self) -> float:
        return self.now


class SlowGenerator(FakeGenerator):
    clock: Clock

    def __call__(self, tasks: Any) -> list[Attempt]:
        self.clock.now += self.clock.chunk_seconds
        return super().__call__(tasks)


def slow(clock: Clock) -> type[SlowGenerator]:
    return type("Slow", (SlowGenerator,), {"clock": clock})


def test_a_first_chunk_projecting_past_four_hours_stops_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, paths = new_run(tmp_path)
    through_select(run)
    monkeypatch.setattr(tool, "CHUNK_TASKS", 1)
    clock = Clock(3601.0)  # One of four candidates: 4 h 4 s projected.
    assert generate(run, paths, generator=slow(clock), clock=clock) == tool.EXIT_STOP
    assert stop_reason(run) == "time_projection"
    result = json.loads((run / "generate" / "result.json").read_text())
    assert result["projected_generation_seconds"] == pytest.approx(4 * 3601.0)
    assert result["chunks_done"] == 1 and not (run / "generate" / "attempts.jsonl").exists()


def test_generation_past_four_hours_of_wall_time_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, paths = new_run(tmp_path)
    through_select(run)
    monkeypatch.setattr(tool, "CHUNK_TASKS", 1)

    class Slower(SlowGenerator):
        def __call__(self, tasks: Any) -> list[Attempt]:
            # The first chunk projects 3.2 h; later ones run slower than it projected.
            self.clock.chunk_seconds = 2880.0 if not self.chunks else 4000.0
            return super().__call__(tasks)

    clock = Clock(0.0)
    generator = type("S", (Slower,), {"clock": clock})
    assert generate(run, paths, generator=generator, clock=clock) == tool.EXIT_STOP
    assert stop_reason(run) == "generation_time_cap"


def test_rewards_under_another_grammar_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scratchcoder.sandbox import reward

    run, paths = new_run(tmp_path)
    through_select(run)
    assert generate(run, paths) == 0
    monkeypatch.setattr(reward, "REWARD_GRAMMAR_VERSION", "reward-grammar-v1")
    calls: list[str] = []

    def score(task: Task, code: str) -> RewardResult:
        calls.append(code)
        return fake_score(task, code)

    assert execute_stage(run, "score", score=score) == tool.EXIT_STOP
    assert stop_reason(run) == "grammar_identity" and calls == []


def test_a_persistent_reward_host_failure_stops_scoring(tmp_path: Path) -> None:
    from scratchcoder.sandbox.reward import UNAVAILABLE

    run, paths = new_run(tmp_path)
    through_select(run)
    assert generate(run, paths) == 0

    def down(task: Task, code: str) -> RewardResult:
        return RewardResult(UNAVAILABLE, None)

    import scratchcoder.rl.rewards as rewards

    original = rewards.LocalRewardSource.backoff
    try:
        rewards.LocalRewardSource.backoff = lambda self, retry: 0.0  # type: ignore[method-assign]
        assert execute_stage(run, "score", score=down) == tool.EXIT_STOP
    finally:
        rewards.LocalRewardSource.backoff = original  # type: ignore[method-assign]
    assert stop_reason(run) == "sandbox_failure"


# --------------------------------------------------------------------------------------
# Selection order and the attempts header.
# --------------------------------------------------------------------------------------


def test_selection_takes_the_first_admitted_in_hash_order_and_keeps_a_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scratchcoder.rl.screening import hash_order

    run, _ = new_run(tmp_path)
    monkeypatch.setattr(tool, "CANDIDATE_CAP", 3)
    through_select(run)
    admitted = [json.loads(line)["id"] for line in (run / "admit" / "admitted.jsonl").open()]
    expected = sorted(admitted, key=hash_order)[:3]
    listed = [json.loads(line) for line in (run / "select" / "candidates.jsonl").open()]
    assert listed == [{"id": i, "source": "nvidia/OpenCodeInstruct"} for i in expected]
    order = [json.loads(line) for line in (run / "select" / "order.jsonl").open()]
    assert [(row["position"], row["id"]) for row in order] == list(enumerate(expected))
    assert order[0]["order_key"] == hash_order(expected[0]).hex()
    result = json.loads((run / "select" / "result.json").read_text())
    assert (result["selected"], result["reserve"]) == (3, 1)
    assert result["candidate_order_sha256"] == tool.order_sha256(expected)


def test_the_attempts_header_records_the_backend_threads_versions_order_and_seeds(
    tmp_path: Path,
) -> None:
    from scratchcoder.rl import rollout

    run, paths = new_run(tmp_path)
    through_select(run)
    assert generate(run, paths) == 0
    header, attempts = read_attempts_file(run / "generate" / "attempts.jsonl")
    assert header is not None
    assert (header.device, header.precision, header.batch_size) == ("cpu", "float32", 64)
    assert (header.samples, header.temperature, header.top_p) == (8, 1.0, 1.0)
    assert (header.max_new_tokens, header.seed_base) == (256, 2**40)
    assert header.seed_base == rollout.SCREENING_SEED
    assert header.max_new_tokens == rollout.ROLLOUT_TOKENS
    env = dict(header.environment or {})
    ids = [json.loads(line)["id"] for line in (run / "select" / "candidates.jsonl").open()]
    assert env["torch_threads"] == 3 and env["torch"] == "x" and env["numpy"] == "y"
    assert env["python"] and env["chunk_tasks"] == 64
    assert env["candidate_order_sha256"] == tool.order_sha256(ids)
    seeds = [str(rollout.screening_seed(i, s)) for i in ids for s in range(8)]
    assert env["seeds_sha256"] == hashlib.sha256("\n".join(seeds).encode()).hexdigest()
    assert "different numerical executions" in env["numerics"]
    assert [a.task_id for a in attempts[::8]] == ids
    assert header.task_file_sha256 == sha(run / "select" / "tasks.jsonl")


def test_the_checkpoint_generator_samples_the_protocol_at_the_default_threads(
    tmp_path: Path,
) -> None:
    import torch

    checkpoint, tokenizer = tiny_checkpoint(tmp_path)
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        generator = tool.CheckpointGenerator(checkpoint, tokenizer)
        assert torch.get_num_threads() == 2  # load_checkpoint's one thread is undone.
        tasks = [make_task("p/a"), make_task("p/b")]
        attempts = generator(tasks)
        again = tool.CheckpointGenerator(checkpoint, tokenizer)(tasks)
    finally:
        torch.set_num_threads(before)
    assert [(a.task_id, a.sample) for a in attempts] == [(t.id, s) for t in tasks for s in range(8)]
    assert attempts == again  # Seeded: the same samples again.
    env = generator.environment()
    assert env["torch_threads"] == 2 and env["torch_threads_default"] == 2
    assert env["torch"] == torch.__version__


def test_a_header_without_an_environment_is_unchanged_and_one_with_it_round_trips() -> None:
    values: dict[str, Any] = {
        "task_file_sha256": "1" * 64,
        "candidates_sha256": "2" * 64,
        "checkpoint_sha256": "3" * 64,
        "tokenizer_sha256": "4" * 64,
        "tasks": 1,
        "samples": 8,
        "attempts": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 256,
        "seed_base": 2**40,
        "precision": "bfloat16",
        "device": "cuda:0",
        "batch_size": 128,
        "generation_seconds": 1.0,
        "created_at": "2026-10-28T00:00:00Z",
    }
    plain = ScreeningHeader(**values).to_json()
    assert "environment" not in plain
    assert ScreeningHeader.from_json(plain).to_json() == plain
    rich = ScreeningHeader(**values, environment={"torch_threads": 6}).to_json()
    assert ScreeningHeader.from_json(rich).environment == {"torch_threads": 6}
    with pytest.raises(ValueError):
        ScreeningHeader.from_json({**plain, "environment": None})
    with pytest.raises(ValueError):
        ScreeningHeader.from_json({**plain, "unknown": 1})
    with pytest.raises(ValueError):
        ScreeningHeader(**values, environment={"x": float("nan")})


# --------------------------------------------------------------------------------------
# The report's arithmetic and every recommendation branch.
# --------------------------------------------------------------------------------------


def screening_record(groups: dict[str, list[tuple[float, str, int]]]) -> dict[str, Any]:
    """A minimal rl-screening-v1 record: per task, its 8 (reward, status, unfinished)."""
    tasks, rewards, eligible = [], [], []
    for task_id, samples in groups.items():
        values = [reward for reward, _, _ in samples]
        reason = "equal_rewards" if len(set(values)) == 1 else None
        if reason is None:
            eligible.append(task_id)
        tasks.append({"id": task_id, "rewards": values, "reason": reason})
        rewards += [
            {"task_id": task_id, "sample": index, "reward": r, "status": s, "unfinished": u}
            for index, (r, s, u) in enumerate(samples)
        ]
    by_reason = {"kept": len(eligible), "equal_rewards": len(tasks) - len(eligible)}
    return {
        "tasks": tasks,
        "rewards": rewards,
        "eligible": eligible,
        "counts": {"by_reason": by_reason},
    }


def test_screening_metrics_count_y_z_and_the_failure_classes() -> None:
    zero_formatting = [(0.0, "no_program", 0)] * 6 + [(0.0, "completed", 0)] * 2
    zero_mixed = [(0.0, "resource_kill", 0), (0.0, "completed", 2)] + [(0.0, "completed", 0)] * 6
    record = screening_record(
        {
            "a": [(1.0, "completed", 0)] + [(0.0, "completed", 0)] * 7,
            "b": zero_formatting,
            "c": zero_mixed,
            "d": [(1.0, "completed", 0)] * 8,
            "e": [(0.5, "completed", 0)] * 8,
        }
    )
    metrics = tool.screening_metrics(record)
    assert (metrics["N"], metrics["E"]) == (5, 1)
    assert metrics["y"]["share"] == pytest.approx(0.2)
    low, high = metrics["y"]["wilson95"]
    assert low < 0.2 < high
    assert (metrics["z"]["successes"], metrics["z"]["trials"]) == (2, 5)
    assert (metrics["all_one_groups"], metrics["other_equal_groups"]) == (1, 1)
    classes = metrics["all_zero_sample_classes"]
    assert classes["formatting"]["successes"] == 6
    assert classes["execution"]["successes"] == 2
    assert classes["genuine"]["successes"] == 8
    assert classes["formatting"]["trials"] == 16
    assert metrics["pass_at_8"]["successes"] == 2


BASE: dict[str, Any] = {
    "stopped": None,
    "pool": 200,
    "screened": 1000,
    "eligible": 200,
    "y_lo": 0.176,
    "z": 0.5,
    "f": 0.1,
    "x": 0.01,
    "reserve": 0,
    "rate": None,
    "funnel": [("intake", 10, 5), ("screening", 5, 2)],
    "drops": {},
}


def test_rule_zero_reports_a_stop_without_a_decision() -> None:
    stop = {"stage": "verify", "criterion": "x"}
    assert tool.recommend(**{**BASE, "stopped": stop, "pool": 999})["rule"] == 0


def test_rule_one_stops_intake_at_the_floor() -> None:
    result = tool.recommend(**{**BASE, "pool": 320})
    assert (result["rule"], result["decision"]) == (1, "stop_intake")
    assert "authorizes nothing" in result["text"]


def test_rule_two_names_the_step_that_removed_the_candidates() -> None:
    funnel = [("intake", 10, 5), ("verification", 5, 0), ("screening", 0, 0)]
    result = tool.recommend(**{**BASE, "screened": 0, "eligible": 0, "funnel": funnel})
    assert (result["rule"], result["step"]) == (2, "verification")
    result = tool.recommend(**{**BASE, "eligible": 0, "funnel": [("screening", 9, 0)]})
    assert (result["rule"], result["step"]) == (2, "screening")


@pytest.mark.parametrize(
    "z,f,x,diagnosis",
    [
        (0.6, 0.5, 0.2, "formatting"),
        (0.7, 0.4, 0.1, "execution"),
        (0.9, 0.4, 0.05, "difficulty"),
    ],
)
def test_rule_three_diagnoses_a_failing_model(z: float, f: float, x: float, diagnosis: str) -> None:
    result = tool.recommend(**{**BASE, "eligible": 149, "z": z, "f": f, "x": x})
    assert (result["rule"], result["diagnosis"]) == (3, diagnosis)


def test_rule_three_reports_the_dominant_drop_reason_below_sixty_percent_all_zero() -> None:
    drops = {"constant_baseline": 4, "equal_rewards_all_one": 9, "not_parseable": 2}
    result = tool.recommend(**{**BASE, "eligible": 100, "z": 0.3, "drops": drops})
    assert (result["rule"], result["diagnosis"]) == (3, "equal_rewards_all_one")
    assert "too easy" in result["text"]
    result = tool.recommend(**{**BASE, "eligible": 100, "z": None, "drops": {"not_parseable": 1}})
    assert result["diagnosis"] == "not_parseable"


def test_rule_four_screens_the_reserve_when_it_suffices() -> None:
    # (320 - 200) * 1.15 / 0.2 = 690 exactly.
    result = tool.recommend(**{**BASE, "reserve": 690})
    assert (result["rule"], result["decision"], result["K"]) == (4, "screen_reserve", 690)
    assert result["K_at_y_lo"] == math.ceil(120 * 1.15 / 0.176)
    assert not result["low_confidence"]
    assert tool.recommend(**{**BASE, "reserve": 689})["decision"] != "screen_reserve"


def test_rule_four_proposes_one_further_slice_only_when_it_fits() -> None:
    from fractions import Fraction

    rate = Fraction(1, 10)  # One admitted candidate per ten intake rows.
    result = tool.recommend(**{**BASE, "pool": 300, "reserve": 10, "rate": rate})
    # K = ceil(20 * 1.15 / 0.2) = 115; rows = (115 - 10) / 0.1 = 1,050.
    assert (result["decision"], result["K"], result["rows_needed"]) == ("further_slice", 115, 1050)
    assert result["start_row"] == 30_000
    # A slice over 20,000 rows, or K at y_lo over 1,000 candidates: no proposal.
    small = Fraction(1, 1000)
    assert tool.recommend(**{**BASE, "rate": small})["decision"] == "no_proposal_owner_choice"
    wide = tool.recommend(**{**BASE, "pool": 300, "rate": rate, "y_lo": 0.02})
    assert wide["decision"] == "no_proposal_owner_choice"
    assert tool.recommend(**{**BASE, "rate": None})["decision"] == "no_proposal_owner_choice"


def test_rule_four_flags_fewer_than_three_hundred_screened_as_low_confidence() -> None:
    result = tool.recommend(**{**BASE, "screened": 299, "eligible": 60, "reserve": 10_000})
    assert result["low_confidence"] and "Low confidence" in result["text"]


def test_a_yield_of_exactly_fifteen_percent_is_sized_not_diagnosed() -> None:
    assert tool.recommend(**{**BASE, "eligible": 150})["rule"] == 4
    assert tool.recommend(**{**BASE, "eligible": 149})["rule"] == 3


# --------------------------------------------------------------------------------------
# Review follow-ups: the gate's identity, durable generation time, the run's wall time,
# reruns on the same inputs, containment, draining and zero candidates.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"device": "cuda:0"}, "not on the CPU"),
        ({"version": 1}, "format and version"),
        ({"format": "other"}, "format and version"),
        ({"created_at": "2020-01-01T00:00:00+00:00"}, "before this run"),
        ({"inputs": {"checkpoint_sha256": None, "tokenizer_sha256": "0" * 64}}, "checkpoint"),
    ],
)
def test_only_a_current_cpu_gate_of_this_checkpoint_and_tokenizer_is_accepted(
    tmp_path: Path, changes: dict[str, Any], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    run, paths = new_run(tmp_path)
    through_select(run)
    if "inputs" in changes:
        changes["inputs"]["checkpoint_sha256"] = sha(paths["checkpoint"])
        message = "another tokenizer"
    gate = gate_file(tmp_path, paths["checkpoint"], **changes)
    argv = ["generate", "--run", str(run), "--format-gate", str(gate)]
    assert tool.main(argv, options={"generator": FakeGenerator}) == tool.EXIT_REFUSED
    assert message in capsys.readouterr().err
    assert not (run / "stop.json").exists() and not (run / "generate" / tool.GATE_PIN).exists()


def test_a_passing_cuda_gate_of_the_same_checkpoint_is_refused(tmp_path: Path) -> None:
    run, paths = new_run(tmp_path)
    through_select(run)
    cuda = gate_file(tmp_path, paths["checkpoint"], device="cuda:0", torch_threads=None)
    argv = ["generate", "--run", str(run), "--format-gate", str(cuda)]
    assert tool.main(argv, options={"generator": FakeGenerator}) == tool.EXIT_REFUSED
    assert not (run / "generate" / "result.json").exists()


def test_the_gate_is_pinned_on_first_use_and_recorded_with_its_device_and_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, paths = new_run(tmp_path)
    through_select(run)
    first = gate_file(tmp_path, paths["checkpoint"], name="first.json")

    class Dies(FakeGenerator):
        def __call__(self, tasks: Any) -> list[Attempt]:
            raise Interrupted

    argv = ["generate", "--run", str(run), "--format-gate", str(first)]
    with pytest.raises(Interrupted):
        tool.main(argv, options={"generator": Dies})
    second = gate_file(tmp_path, paths["checkpoint"], name="second.json")
    other = ["generate", "--run", str(run), "--format-gate", str(second)]
    assert tool.main(other, options={"generator": FakeGenerator}) == tool.EXIT_REFUSED
    assert tool.main(argv, options={"generator": FakeGenerator}) == 0
    result = json.loads((run / "generate" / "result.json").read_text())
    assert result["gate"]["device"] == "cpu" and result["gate"]["torch_threads"] == 4
    assert result["gate"]["sha256"] == sha(first)


def run_generate(run: Path, paths: dict[str, Path], **options: Any) -> int:
    gate = run.parent / "gate" / "pinned.json"
    if not gate.exists():
        gate_file(run.parent, paths["checkpoint"], name="pinned.json")
    argv = ["generate", "--run", str(run), "--format-gate", str(gate)]
    return tool.main(argv, options=options)


def test_interrupted_generation_time_counts_towards_the_four_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, paths = new_run(tmp_path)
    through_select(run)
    monkeypatch.setattr(tool, "CHUNK_TASKS", 1)
    clock = Clock(1000.0)

    class DiesSecond(SlowGenerator):
        def __call__(self, tasks: Any) -> list[Attempt]:
            if self.chunks:
                raise Interrupted
            return super().__call__(tasks)

    with pytest.raises(Interrupted):
        run_generate(run, paths, generator=type("D", (DiesSecond,), {"clock": clock}), clock=clock)
    # One recorded chunk, plus the longest chunk for the one it was in when interrupted.
    assert tool.generation_seconds_used(tool.RunDir(run)) == 2000.0
    assert run_generate(run, paths, generator=slow(clock), clock=clock) == 0
    result = json.loads((run / "generate" / "result.json").read_text())
    assert result["prior_generation_seconds"] == 2000.0
    assert result["projected_generation_seconds"] == 2000.0 + 4 * 1000.0
    assert tool.generation_seconds_used(tool.RunDir(run)) == 2000.0 + 4 * 1000.0


def test_generation_already_at_four_hours_stops_on_entry(tmp_path: Path) -> None:
    run, paths = new_run(tmp_path)
    through_select(run)
    log = run / "generate" / tool.GENERATION_LOG
    log.parent.mkdir()
    log.write_text(
        '{"event": "start"}\n{"event": "chunk", "index": 0, "seconds": 7200.0}\n'
        '{"event": "chunk", "index": 1, "seconds": 7200.0}\n{"event": "end"}\n',
        encoding="utf-8",
    )
    built: list[int] = []

    def factory(files: Any) -> FakeGenerator:
        built.append(1)
        return FakeGenerator()

    assert run_generate(run, paths, generator=factory) == tool.EXIT_STOP
    assert stop_reason(run) == "generation_time_cap" and built == []


def test_a_chunk_that_could_overrun_the_remaining_time_is_not_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, paths = new_run(tmp_path)
    through_select(run)
    monkeypatch.setattr(tool, "CHUNK_TASKS", 1)
    clock = Clock(0.0)

    class Slowing(SlowGenerator):
        def __call__(self, tasks: Any) -> list[Attempt]:
            # 3,000 s projects 3.3 h; then 6,000 s: 9,000 s used, and another 6,000 s chunk
            # would pass 4 h, so the third is never started.
            self.clock.chunk_seconds = 3000.0 if not self.chunks else 6000.0
            return super().__call__(tasks)

    made: list[Slowing] = []

    def factory(files: Any) -> Slowing:
        generator = Slowing()
        generator.clock = clock
        made.append(generator)
        return generator

    assert run_generate(run, paths, generator=factory, clock=clock) == tool.EXIT_STOP
    assert stop_reason(run) == "generation_time_cap"
    assert made[0].chunks == [1, 1] and clock.now == 9000.0


def _age(run: Path, seconds: float) -> None:
    """Move the run's init time ``seconds`` into the past."""
    from datetime import timedelta

    inputs = json.loads((run / "inputs.json").read_text())
    for name in ("created_at", "chain_started_at"):
        moved = datetime.fromisoformat(inputs[name]) - timedelta(seconds=seconds)
        inputs[name] = moved.isoformat()
    (run / "inputs.json").write_text(json.dumps(inputs), encoding="utf-8")


def test_a_stage_after_twelve_hours_of_run_wall_time_stops(tmp_path: Path) -> None:
    run, _ = new_run(tmp_path)
    _age(run, 12 * 3600 + 1)
    executor = Counting()
    assert execute_stage(run, "verify", execute=executor) == tool.EXIT_STOP
    stop = json.loads((run / "stop.json").read_text())
    assert stop["reason"] == "cap_reached" and "run wall time" in stop["detail"]
    assert executor.calls == []
    assert tool.main(["report", "--run", str(run)]) == 0
    report = json.loads((run / "report" / "report.json").read_text())
    wall = [i for i in report["stopping_criteria"] if "12 h end-to-end" in i["criterion"]]
    assert wall[0]["status"] == "fired"


def test_calls_and_chunks_stop_at_the_run_wall_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import timedelta

    deadline = datetime(2026, 1, 1, tzinfo=UTC)
    moments = [deadline - timedelta(seconds=1), deadline]
    budget = tool.Budget(
        tmp_path / "budget.jsonl", "verify", cap=10, deadline=deadline, now=lambda: moments[0]
    )
    budget.charge()
    moments[0] = moments[1]
    with pytest.raises(tool.CapReached, match="run wall time"):
        budget.charge()
    # A chunk is not started when the longest measured chunk would reach the deadline.
    run, paths = new_run(tmp_path / "chunks")
    through_select(run)
    monkeypatch.setattr(tool, "CHUNK_TASKS", 1)
    end = tool.run_deadline(tool.RunDir(run))
    clock = Clock(100.0)
    late = lambda: end - timedelta(seconds=50)  # noqa: E731
    assert run_generate(run, paths, generator=slow(clock), clock=clock, now=late) == tool.EXIT_STOP
    stop = json.loads((run / "stop.json").read_text())
    assert stop["reason"] == "cap_reached" and "run wall time" in stop["detail"]
    assert clock.now == 100.0  # Only the first chunk ran.


def failed_run(root: Path) -> Path:
    run, _ = new_run(root)
    executor = Counting(lambda code, tests, mode: Result("infrastructure_error"))
    assert execute_stage(run, "verify", execute=executor) == tool.EXIT_STOP
    assert stop_reason(run) == "sandbox_failure"
    return run


def test_a_second_run_on_the_same_inputs_needs_to_supersede_a_host_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = failed_run(tmp_path)
    paths = pilot_inputs(tmp_path / "same")
    # The same pilot bytes are the same inputs, wherever the files are.
    paths["pilot"] = tmp_path / "after-scan" / "pilot-clear.jsonl"
    paths["remainder"] = tmp_path / "after-scan" / "remainder-clear.jsonl"
    second = tmp_path / "run2"
    assert tool.main(init_argv(paths, second)) == tool.EXIT_REFUSED
    wrong = [*init_argv(paths, second), "--supersedes", str(tmp_path / "elsewhere")]
    assert tool.main(wrong) == tool.EXIT_REFUSED
    assert tool.main([*init_argv(paths, second), "--supersedes", str(first)]) == 0
    # Caps hold across the rerun: the new ledger starts with the old run's calls.
    before = tool.ledger_counts(first / "budget.jsonl")
    assert tool.ledger_counts(second / "budget.jsonl") == before
    carried = json.loads((second / "inputs.json").read_text())["carried"]
    assert carried["from"] == str(first.resolve()) and carried["stop_reason"] == "sandbox_failure"
    monkeypatch.setitem(tool.CALL_CAPS, "verify", before["verify"] + 15)
    executor = Counting()
    assert execute_stage(second, "verify", execute=executor) == tool.EXIT_STOP
    assert stop_reason(second) == "cap_reached" and executor.calls == []
    # Only the latest run can be superseded, and only after a host failure.
    third = [*init_argv(paths, tmp_path / "run3"), "--supersedes", str(first)]
    assert tool.main(third) == tool.EXIT_REFUSED
    third[-1] = str(second)
    assert tool.main(third) == tool.EXIT_REFUSED  # second stopped on a cap.
    assert not (tmp_path / "run3").exists()


def test_a_run_that_did_not_stop_on_a_host_failure_cannot_be_superseded(tmp_path: Path) -> None:
    run, paths = new_run(tmp_path)
    again = [*init_argv(paths, tmp_path / "run2"), "--supersedes", str(run)]
    assert tool.main(again) == tool.EXIT_REFUSED  # Not stopped: resume it instead.
    assert execute_stage(run, "verify", execute=Counting()) == 0
    assert tool.main(again) == tool.EXIT_REFUSED


def test_carried_generation_time_counts_in_the_new_run(tmp_path: Path) -> None:
    run, _ = new_run(tmp_path)
    inputs = json.loads((run / "inputs.json").read_text())
    inputs["carried"] = {"generation_seconds": 5000.0}
    (run / "inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
    assert tool.generation_seconds_used(tool.RunDir(run)) == 5000.0


def test_after_a_containment_failure_no_queued_admission_call_starts(tmp_path: Path) -> None:
    tasks = [pilot_task(f"p/{i}") for i in range(12)]
    run, _ = new_run(tmp_path, tasks=tasks)
    assert execute_stage(run, "verify", execute=Counting()) == 0
    unprotected = Result("infrastructure_error", detail="supervisor_unprotected")
    executor = Counting(lambda code, tests, mode: unprotected)
    assert execute_stage(run, "admit", execute=executor) == tool.EXIT_STOP
    assert stop_reason(run) == "containment_failure"
    assert len(executor.calls) <= tool.ADMIT_WORKERS
    budget = tool.Budget(tmp_path / "b.jsonl", "admit", cap=5)
    budget.trip()
    with pytest.raises(tool.ContainmentFailure):
        budget.charge()


def test_a_scoring_stop_waits_for_calls_in_flight_before_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    import time

    from scratchcoder.rl.rewards import LocalRewardSource
    from scratchcoder.sandbox.reward import UNAVAILABLE

    monkeypatch.setattr(LocalRewardSource, "backoff", lambda self, retry: 0.0)
    run, paths = new_run(tmp_path)
    through_select(run)
    assert generate(run, paths) == 0
    lock = threading.Lock()
    state = {"started": 0, "finished": 0}

    def score(task: Task, code: str) -> RewardResult:
        if code == "good":
            return RewardResult(UNAVAILABLE, None)  # A host that stays down.
        with lock:
            state["started"] += 1
        time.sleep(0.3)
        with lock:
            state["finished"] += 1
        return fake_score(task, code)

    assert execute_stage(run, "score", score=score) == tool.EXIT_STOP
    assert stop_reason(run) == "sandbox_failure"
    # Every call that started finished before the stop was recorded and the lock released.
    assert state["started"] == state["finished"]
    stop = json.loads((run / "stop.json").read_text())
    assert stop["budget"]["calls_in_flight"] == 0
    assert stop["budget"]["run_calls"] == sum(tool.ledger_counts(run / "budget.jsonl").values())


def test_zero_cleared_candidates_complete_every_stage_and_report_rule_two(tmp_path: Path) -> None:
    run, paths = new_run(tmp_path, tasks=[])
    through_select(run)
    assert generate(run, paths) == 0
    assert execute_stage(run, "score", score=fake_score) == 0
    assert tool.main(["report", "--run", str(run)]) == 0
    verify = json.loads((run / "verify" / "result.json").read_text())
    assert verify["status"] == "completed" and verify["verified_share"] is None
    report = json.loads((run / "report" / "report.json").read_text())
    assert report["decision"]["rule"] == 2
    assert report["decision"]["step"] == "integrity_scan"
    assert not (run / "stop.json").exists()


# --------------------------------------------------------------------------------------
# Second review: one registry, live runs, containment precedence, mid-chunk charging,
# the chain's wall time and torn or missing records.
# --------------------------------------------------------------------------------------


def test_a_run_in_another_directory_on_the_same_inputs_is_refused(tmp_path: Path) -> None:
    run, paths = new_run(tmp_path)
    elsewhere = tmp_path / "other-parent" / "deep" / "run"
    assert tool.main(init_argv(paths, elsewhere)) == tool.EXIT_REFUSED
    assert not elsewhere.exists()
    # The same tasks in another order and other bytes are the same inputs.
    shuffled = list(reversed(pilot_tasks()))
    other = pilot_inputs(tmp_path / "copy", tasks=shuffled)
    assert tool.main(init_argv(other, tmp_path / "copy" / "run")) == tool.EXIT_REFUSED
    assert (
        tool.content_key(shuffled) == json.loads((run / "inputs.json").read_text())["registry_key"]
    )


def test_every_stage_refuses_a_run_that_is_not_the_live_registered_one(tmp_path: Path) -> None:
    import shutil

    first = failed_run(tmp_path)
    copy = tmp_path / "copied-run"
    shutil.copytree(first, copy)
    (copy / "stop.json").unlink()
    (copy / "verify" / "result.json").unlink()
    assert execute_stage(copy, "verify", execute=Counting()) == tool.EXIT_REFUSED
    paths = pilot_inputs(tmp_path / "same")
    paths["pilot"] = tmp_path / "after-scan" / "pilot-clear.jsonl"
    second = tmp_path / "run2"
    assert tool.main([*init_argv(paths, second), "--supersedes", str(first)]) == 0
    # The superseded run is no longer live: even its report is refused.
    assert tool.main(["report", "--run", str(first)]) == tool.EXIT_REFUSED
    assert execute_stage(second, "verify", execute=Counting()) == 0


def test_the_registry_entry_is_written_before_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = pilot_inputs(tmp_path)
    real = tool._write_new

    def fails(path: Path, data: bytes) -> str:
        if path.name == "inputs.json":
            assert tool.REGISTRY.read_text().count('"key"') == 1  # Already durable.
            raise OSError("disk full")
        return real(path, data)

    monkeypatch.setattr(tool, "_write_new", fails)
    with pytest.raises(OSError):
        tool.main(init_argv(paths, tmp_path / "run"))
    monkeypatch.setattr(tool, "_write_new", real)
    # The half-made run is not silently replaced: superseding it is refused for inspection.
    again = [*init_argv(paths, tmp_path / "run2"), "--supersedes", str(tmp_path / "run")]
    assert tool.main(again) == tool.EXIT_REFUSED
    assert tool.main(init_argv(paths, tmp_path / "run3")) == tool.EXIT_REFUSED


def test_a_containment_failure_wins_over_whatever_ended_the_stage(tmp_path: Path) -> None:
    run, _ = new_run(tmp_path)

    def races(context: tool.Context) -> dict[str, Any]:
        assert context.budget is not None
        context.budget.trip()  # One thread saw a missing protection ...
        raise tool.Stop("sandbox_failure", "... while another's host failure ended the stage")

    argv = ["verify", "--run", str(run), "--receipt", str(tmp_path / "r.json")]
    receipt_for(tmp_path / "r.json", argv, "tools.pilot_stages")
    status = tool.run_stage("verify", run, races, argv=argv, receipt=tmp_path / "r.json")
    assert status == tool.EXIT_STOP and stop_reason(run) == "containment_failure"


def test_a_stop_showing_a_tripped_budget_cannot_be_superseded(tmp_path: Path) -> None:
    first = failed_run(tmp_path)
    stop = json.loads((first / "stop.json").read_text())
    stop["budget"]["tripped"] = True
    (first / "stop.json").write_text(json.dumps(stop), encoding="utf-8")
    paths = pilot_inputs(tmp_path / "same")
    paths["pilot"] = tmp_path / "after-scan" / "pilot-clear.jsonl"
    again = [*init_argv(paths, tmp_path / "run2"), "--supersedes", str(first)]
    assert tool.main(again) == tool.EXIT_REFUSED


def _log_rows(run: Path, rows: list[dict[str, Any]], torn: str = "") -> None:
    log = run / "generate" / tool.GENERATION_LOG
    log.parent.mkdir(exist_ok=True)
    log.write_text("".join(json.dumps(row) + "\n" for row in rows) + torn, encoding="utf-8")


def test_an_interruption_inside_a_chunk_is_charged_its_time_since_the_chunk_started(
    tmp_path: Path,
) -> None:
    from datetime import timedelta

    run, _ = new_run(tmp_path)
    t = datetime(2026, 11, 15, tzinfo=UTC)

    def at(seconds: float) -> str:
        return (t + timedelta(seconds=seconds)).isoformat()

    directory = tool.RunDir(run)
    # Killed in the first chunk: no chunk finished, yet the time since it started counts.
    _log_rows(run, [{"event": "start", "at": at(0)}, {"event": "chunk_start", "at": at(0)}])
    assert tool.generation_seconds_used(directory, t + timedelta(seconds=5000)) == 5000.0
    # Killed in a later chunk, then restarted: charged until the next invocation's start.
    _log_rows(
        run,
        [
            {"event": "start", "at": at(0)},
            {"event": "chunk_start", "at": at(0)},
            {"event": "chunk", "seconds": 1000.0, "at": at(1000)},
            {"event": "chunk_start", "at": at(1000)},
            {"event": "start", "at": at(4000)},
            {"event": "end", "at": at(4001)},
        ],
    )
    assert tool.generation_seconds_used(directory, t) == 1000.0 + 3000.0
    # Killed between chunks: the longest chunk is still charged.
    _log_rows(
        run,
        [
            {"event": "start", "at": at(0)},
            {"event": "chunk_start", "at": at(0)},
            {"event": "chunk", "seconds": 1000.0, "at": at(1000)},
        ],
    )
    assert tool.generation_seconds_used(directory, t + timedelta(seconds=9999)) == 2000.0


def test_a_torn_generation_log_line_keeps_the_invocation_open(tmp_path: Path) -> None:
    run, _ = new_run(tmp_path)
    rows: list[dict[str, Any]] = [
        {"event": "start", "at": "2026-11-15T00:00:00+00:00"},
        {"event": "chunk", "seconds": 700.0, "at": "2026-11-15T00:11:40+00:00"},
        {"event": "end", "at": "2026-11-15T00:11:41+00:00"},
    ]
    _log_rows(run, rows, torn='{"event": "sta')
    assert tool.generation_seconds_used(tool.RunDir(run)) == 700.0 + 700.0


def test_the_wall_time_runs_from_the_first_init_of_the_chain(tmp_path: Path) -> None:
    first = failed_run(tmp_path)
    _age(first, 11 * 3600)
    started = datetime.fromisoformat(json.loads((first / "inputs.json").read_text())["created_at"])
    paths = pilot_inputs(tmp_path / "same")
    paths["pilot"] = tmp_path / "after-scan" / "pilot-clear.jsonl"
    second = tmp_path / "run2"
    assert tool.main([*init_argv(paths, second), "--supersedes", str(first)]) == 0
    deadline = tool.run_deadline(tool.RunDir(second))
    assert (deadline - started).total_seconds() == 12 * 3600  # The gap counts too.
    _age(second, 3600)  # The chain is now past 12 h.
    assert execute_stage(second, "verify", execute=Counting()) == tool.EXIT_STOP
    assert "run wall time" in json.loads((second / "stop.json").read_text())["detail"]


def test_a_gate_without_a_time_zone_is_refused_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run, paths = new_run(tmp_path)
    through_select(run)
    naive = gate_file(tmp_path, paths["checkpoint"], created_at="2099-01-01T00:00:00")
    argv = ["generate", "--run", str(run), "--format-gate", str(naive)]
    assert tool.main(argv, options={"generator": FakeGenerator}) == tool.EXIT_REFUSED
    assert "time-zoned" in capsys.readouterr().err


def test_missing_ledgers_and_logs_are_refused(tmp_path: Path) -> None:
    run, paths = new_run(tmp_path)
    through_select(run)
    assert run_generate(run, paths, generator=FakeGenerator) == 0
    # A pinned gate without its generation log, and a run without its budget ledger.
    other, other_paths = new_run(tmp_path / "b", tasks=[pilot_task("p/solo")])
    through_select(other)
    (other / "generate").mkdir()
    (other / "generate" / tool.GATE_PIN).write_text("{}", encoding="utf-8")
    assert run_generate(other, other_paths, generator=FakeGenerator) == tool.EXIT_REFUSED
    (run / "budget.jsonl").unlink()
    assert execute_stage(run, "score", score=fake_score) == tool.EXIT_REFUSED


def test_a_torn_registry_line_is_refused_cleanly(tmp_path: Path) -> None:
    paths = pilot_inputs(tmp_path)
    tool.REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    tool.REGISTRY.write_text('{"key": "abc", "ru', encoding="utf-8")
    assert tool.main(init_argv(paths, tmp_path / "run")) == tool.EXIT_REFUSED
    assert not (tmp_path / "run").exists()


def test_a_torn_generation_log_line_does_not_wedge_later_appends(tmp_path: Path) -> None:
    log = tmp_path / "generation.jsonl"
    tool._append_log(log, {"event": "start", "prior_generation_seconds": 0.0})
    tool._append_log(log, {"event": "chunk_start", "index": 0})
    with log.open("a", encoding="utf-8") as stream:
        stream.write('{"event": "chu')  # killed mid-append
    tool._append_log(log, {"event": "start", "prior_generation_seconds": 0.0})
    rows, torn = tool._jsonl_rows(log)
    assert not torn
    assert [row["event"] for row in rows] == ["start", "chunk_start", "start"]


def test_the_registry_is_anchored_to_the_main_checkout(tmp_path: Path) -> None:
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (main / ".git" / "worktrees" / "wt" / "commondir").write_text("../..\n")
    worktree = tmp_path / "elsewhere" / "wt"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n")
    assert tool._main_checkout(worktree) == main.resolve()
    assert tool._main_checkout(main) == main
    assert tool._main_checkout(tmp_path / "nogit") == tmp_path / "nogit"
