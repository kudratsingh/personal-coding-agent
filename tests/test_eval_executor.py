"""The Mac executor and the test seal, with fake execution (no generated code runs here)."""

import fcntl
import hashlib
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from eval_fixtures import (
    CHECKPOINT,
    TOKENIZER,
    io_task,
    pool_task,
    save_task_file,
    write_fixture_completions,
)

from scratchcoder.cli import main
from scratchcoder.eval import execution_policy, executor, harness
from scratchcoder.eval.completions import Sampling, read_completions
from scratchcoder.eval.executor import execute_completions, read_results, run_test_baseline
from scratchcoder.eval.parallel import HostUnavailableError
from scratchcoder.eval.scorecard import run_from_results
from scratchcoder.eval.seal import SealError, SealLedger, UnsealRecord, seal_completions
from scratchcoder.eval.tasks import Task, load_tasks
from scratchcoder.sandbox import budget, runner
from scratchcoder.sandbox.runner import Limits, Result
from scratchcoder.tokenizer.bpe import ByteBPE

GOOD = "def add_one(x):\n    return x + 1  # good"
BAD = "def add_one(x):\n    return x"


class FakeSandbox:
    """Passes programs marked ``# good``; can stall or fail the host for chosen programs, or
    raise ``KeyboardInterrupt`` on the call after ``interrupt_after`` calls."""

    def __init__(
        self,
        *,
        stall: int = 0,
        down: Callable[[str], bool] | None = None,
        interrupt_after: int | None = None,
    ) -> None:
        self.calls: list[str] = []
        self.stall = stall
        self.down = down or (lambda code: False)
        self.interrupt_after = interrupt_after
        self.lock = threading.Lock()

    def __call__(self, code: str, tests: str, mode: str) -> Result:
        with self.lock:
            if self.interrupt_after is not None and len(self.calls) >= self.interrupt_after:
                raise KeyboardInterrupt
            self.calls.append(code)
            if self.down(code):
                return Result("unavailable")
            if self.stall:
                self.stall -= 1
                return Result("timeout", host_stalled=True, elapsed_seconds=12.0)
        return Result("passed" if "# good" in code else "failed", elapsed_seconds=0.25)


def no_sleep(_: float) -> None:
    return None


def dev_fixture(root: Path) -> tuple[Path, Path, list[Task]]:
    tasks = [pool_task("t/1"), pool_task("t/2", source="synthetic-b"), io_task("t/3")]
    tasks_path = save_task_file(root / "dev.jsonl", tasks)
    programs = {"t/1": GOOD, "t/2": BAD, "t/3": None}
    completions = root / "completions.jsonl"
    write_fixture_completions(completions, tasks_path, tasks, lambda task, _: programs[task.id])
    return completions, tasks_path, tasks


def test_execution_scores_extracts_audits_and_records(tmp_path: Path) -> None:
    completions, tasks_path, _ = dev_fixture(tmp_path)
    sandbox = FakeSandbox()
    results = tmp_path / "results.jsonl"
    summary = execute_completions(
        completions, tasks_path, results, tokenizer=TOKENIZER, execute=sandbox, sleep=no_sleep
    )
    assert summary["attempts"] == 3 and summary["scored"] == 3 and summary["passed"] == 1
    assert summary["unresolved"] == 0 and summary["pending"] == 0
    assert sorted(sandbox.calls) == sorted([GOOD, BAD])  # No block: scored without running.
    parsed = read_results(results)
    assert parsed.complete and parsed.torn_bytes == 0
    assert parsed.header["completions_sha256"] == read_completions(completions).sha256
    assert parsed.header["extraction_checked_with_tokenizer"] is True
    first, second, third = (parsed.attempts[(task, 0)] for task in ("t/1", "t/2", "t/3"))
    assert first.status == "passed" and first.audit_flags == ()
    assert first.program_sha256 == hashlib.sha256(GOOD.encode()).hexdigest()
    assert second.status == "failed" and second.audit_flags is None
    assert third.status == "no_program" and third.program_sha256 is None
    assert (first.source, second.source, third.written_by) == (
        "synthetic-a",
        "synthetic-b",
        "human",
    )
    assert parsed.sandbox_seconds == pytest.approx(0.5)


def test_passing_attempts_record_the_static_audit(tmp_path: Path) -> None:
    tasks = [pool_task("t/1")]
    tasks_path = save_task_file(tmp_path / "dev.jsonl", tasks)
    sneaky = "import sys\ndef add_one(x):\n    sys._getframe()\n    return x + 1  # good"
    completions = tmp_path / "completions.jsonl"
    write_fixture_completions(completions, tasks_path, tasks, lambda *_: sneaky)
    results = tmp_path / "results.jsonl"
    summary = execute_completions(
        completions, tasks_path, results, tokenizer=TOKENIZER, execute=FakeSandbox()
    )
    assert summary["audit_flagged_passes"] == 1
    flags = read_results(results).attempts[("t/1", 0)].audit_flags
    assert flags is not None and any("getframe" in flag for flag in flags)


def test_host_failures_retry_then_stay_unresolved_and_resume_later(tmp_path: Path) -> None:
    completions, tasks_path, _ = dev_fixture(tmp_path)
    results = tmp_path / "results.jsonl"
    waits: list[float] = []
    flaky = FakeSandbox(stall=1, down=lambda code: code == BAD)
    summary = execute_completions(
        completions, tasks_path, results, tokenizer=TOKENIZER, execute=flaky, sleep=waits.append
    )
    assert summary["unresolved"] == 1 and summary["scored"] == 2 and summary["passed"] == 1
    parsed = read_results(results)
    assert set(parsed.unresolved) == {("t/2", 0)} and not parsed.complete
    assert parsed.unresolved[("t/2", 0)]["last_status"] == "unavailable"
    assert flaky.calls.count(BAD) == 4 and max(waits) <= 30
    assert parsed.attempts[("t/1", 0)].host_failures == 1  # The stall was retried, not scored.
    healthy = FakeSandbox()
    again = execute_completions(
        completions, tasks_path, results, tokenizer=TOKENIZER, execute=healthy, sleep=no_sleep
    )
    assert healthy.calls == [BAD]  # Only the unresolved attempt runs again.
    assert again["unresolved"] == 0 and again["scored"] == 3
    final = read_results(results)
    assert final.complete and final.attempts[("t/2", 0)].status == "failed"
    assert [window["kind"] for window in final.windows] == ["window", "window_end"] * 2


def test_an_interrupted_run_resumes_from_the_last_complete_line(tmp_path: Path) -> None:
    tasks = [pool_task(f"t/{index}") for index in range(12)]
    tasks_path = save_task_file(tmp_path / "dev.jsonl", tasks)
    completions = tmp_path / "completions.jsonl"
    write_fixture_completions(completions, tasks_path, tasks, lambda *_: GOOD)
    results = tmp_path / "results.jsonl"
    # The resumed run must use the same execute as the interrupted one: here both are a
    # FakeSandbox, so the execution policy matches.
    interrupted = FakeSandbox(interrupt_after=5)
    with pytest.raises(KeyboardInterrupt):
        execute_completions(
            completions, tasks_path, results, tokenizer=TOKENIZER, execute=interrupted, workers=1
        )
    done = len(read_results(results).attempts)
    assert 1 <= done <= 5
    with results.open("ab") as stream:
        stream.write(b'{"kind": "attempt", "task_id": "t/9"')  # A torn write.
    assert read_results(results).torn_bytes > 0
    sandbox = FakeSandbox()
    summary = execute_completions(
        completions, tasks_path, results, tokenizer=TOKENIZER, execute=sandbox
    )
    assert len(sandbox.calls) == 12 - done and summary["scored"] == 12
    assert read_results(results).torn_bytes == 0 and read_results(results).complete


def test_a_down_host_stops_the_window_and_keeps_the_file_resumable(tmp_path: Path) -> None:
    tasks = [pool_task(f"t/{index}") for index in range(40)]
    tasks_path = save_task_file(tmp_path / "dev.jsonl", tasks)
    completions = tmp_path / "completions.jsonl"
    write_fixture_completions(completions, tasks_path, tasks, lambda *_: GOOD)
    results = tmp_path / "results.jsonl"
    with pytest.raises(HostUnavailableError):
        execute_completions(
            completions,
            tasks_path,
            results,
            tokenizer=TOKENIZER,
            execute=FakeSandbox(down=lambda code: True),
            sleep=no_sleep,
            workers=2,
            max_consecutive_unresolved=4,
        )
    parsed = read_results(results)
    assert not parsed.attempts and 4 <= len(parsed.unresolved) < 40
    assert parsed.windows[-1]["kind"] == "window_end"


def test_inconsistent_inputs_are_refused(tmp_path: Path) -> None:
    completions, tasks_path, tasks = dev_fixture(tmp_path)
    results = tmp_path / "results.jsonl"
    with pytest.raises(ValueError, match="tokenizer"):
        execute_completions(completions, tasks_path, results, execute=FakeSandbox())
    other = ByteBPE.train(["abababab"], 273)
    with pytest.raises(ValueError, match="tokenizer differs"):
        execute_completions(completions, tasks_path, results, tokenizer=other)
    assert not results.exists()
    lines = completions.read_text().splitlines()
    row = json.loads(lines[1])
    row["program"] = "print('not what the tokens say')"
    forged = tmp_path / "forged.jsonl"
    forged.write_text("\n".join([lines[0], json.dumps(row), *lines[2:]]) + "\n")
    with pytest.raises(ValueError, match="extraction"):
        execute_completions(forged, tasks_path, results, tokenizer=TOKENIZER)
    execute_completions(
        completions, tasks_path, results, tokenizer=TOKENIZER, execute=FakeSandbox()
    )
    second = tmp_path / "second.jsonl"
    write_fixture_completions(second, tasks_path, tasks, lambda *_: GOOD)
    with pytest.raises(ValueError, match="other completions"):
        execute_completions(second, tasks_path, results, tokenizer=TOKENIZER)
    with pytest.raises(ValueError, match="test split only"):
        execute_completions(
            completions,
            tasks_path,
            tmp_path / "r2.jsonl",
            tokenizer=TOKENIZER,
            unseal=tmp_path / "unseal.json",
        )


# --- the test seal -------------------------------------------------------------------


def sealed_fixture(root: Path, checkpoint: str = CHECKPOINT) -> tuple[Path, Path, list[Task]]:
    tasks = [
        pool_task("t/1", split="test"),
        pool_task("t/2", split="test", hidden_tests="assert add_one(2) == [i for i in (3,)]"),
    ]
    tasks_path = root / "test.jsonl"
    if not tasks_path.exists():
        save_task_file(tasks_path, tasks)
    completions = root / f"test-{checkpoint[:4]}.jsonl"
    write_fixture_completions(
        completions, tasks_path, tasks, lambda *_: GOOD, split="test", checkpoint=checkpoint
    )
    return completions, tasks_path, tasks


def unseal_record(path: Path, endpoints: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "unseal",
                "schema_version": 1,
                "frozen_at": "2026-11-01",
                "endpoints": endpoints,
            }
        )
    )
    return path


def four_endpoints(root: Path, ledger: Path) -> tuple[dict[str, Any], dict[str, Path], Path]:
    endpoints: dict[str, Any] = {}
    files: dict[str, Path] = {}
    tasks_path = root / "test.jsonl"
    for name, letter in (("base", "a"), ("sft", "b"), ("rl", "c"), ("more-sft", "d")):
        completions, tasks_path, _ = sealed_fixture(root, letter * 64)
        digest = seal_completions(completions, ledger)["completions_sha256"]
        endpoints[name] = {"checkpoint_sha256": letter * 64, "completions_sha256": [digest]}
        files[name] = completions
    return endpoints, files, tasks_path


def test_the_test_split_is_refused_without_a_seal(tmp_path: Path) -> None:
    completions, tasks_path, _ = sealed_fixture(tmp_path)
    with pytest.raises(SealError, match="sealed"):
        execute_completions(completions, tasks_path, tmp_path / "r.jsonl", tokenizer=TOKENIZER)
    ledger = tmp_path / "ledger.jsonl"
    record = unseal_record(
        tmp_path / "unseal.json",
        {
            name: {"checkpoint_sha256": letter * 64, "completions_sha256": [letter * 64]}
            for name, letter in (("base", "1"), ("sft", "2"), ("rl", "3"), ("more-sft", "4"))
        },
    )
    with pytest.raises(SealError, match="never sealed"):
        execute_completions(
            completions,
            tasks_path,
            tmp_path / "r.jsonl",
            tokenizer=TOKENIZER,
            seal_ledger=ledger,
            unseal=record,
        )
    assert not (tmp_path / "r.jsonl").exists()


def test_each_checkpoints_test_completions_execute_once(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    calls: list[str] = []

    def sandbox(code: str, tests: str, mode: str) -> Result:
        calls.append(tests)
        return Result("passed", stdout="")

    results = tmp_path / "sft-results.jsonl"
    summary = execute_completions(
        files["sft"],
        tasks_path,
        results,
        tokenizer=TOKENIZER,
        execute=sandbox,
        seal_ledger=ledger,
        unseal=record,
    )
    assert summary["scored"] == 2 and summary["test_baseline"] is not None
    events = [event["event"] for event in SealLedger(ledger).events()]
    assert events == ["sealed"] * 4 + ["opened", "completed", "baseline"]
    baseline = json.loads(Path(summary["test_baseline"]).read_text())
    assert baseline["split"] == "test" and baseline["summary"]["tasks"] == 2
    assert baseline["summary"]["methods"] == {"executed": 1, "reward-channel": 1}
    for target in (results, tmp_path / "again.jsonl"):
        with pytest.raises(SealError, match="already executed"):
            execute_completions(
                files["sft"],
                tasks_path,
                target,
                tokenizer=TOKENIZER,
                execute=sandbox,
                seal_ledger=ledger,
                unseal=record,
            )
    assert not (tmp_path / "again.jsonl").exists()
    # Another endpoint's read reuses the logged baseline instead of computing it again.
    before = len(calls)
    second = execute_completions(
        files["rl"],
        tasks_path,
        tmp_path / "rl-results.jsonl",
        tokenizer=TOKENIZER,
        execute=sandbox,
        seal_ledger=ledger,
        unseal=record,
    )
    assert len(calls) - before == 2 and second["test_baseline"] == summary["test_baseline"]


def test_an_interrupted_test_read_resumes_but_a_second_read_is_refused(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    results = tmp_path / "base-results.jsonl"
    options: dict[str, Any] = {"tokenizer": TOKENIZER, "seal_ledger": ledger, "unseal": record}
    execute_completions(
        files["base"],
        tasks_path,
        results,
        execute=FakeSandbox(down=lambda code: True),
        sleep=no_sleep,
        **options,
    )
    with pytest.raises(SealError, match="already opened"):
        execute_completions(
            files["base"], tasks_path, tmp_path / "other.jsonl", execute=FakeSandbox(), **options
        )
    finished = execute_completions(
        files["base"], tasks_path, results, execute=FakeSandbox(), **options
    )
    assert finished["scored"] == 2
    events = [event["event"] for event in SealLedger(ledger).events()]
    assert events[4:] == ["opened", "resumed", "completed", "baseline"]


@pytest.mark.parametrize(
    "change",
    [
        lambda e: e.pop("rl"),
        lambda e: e.update(extra=e["rl"]),
        lambda e: e["rl"].update(checkpoint_sha256=e["sft"]["checkpoint_sha256"]),
        lambda e: e["rl"].update(completions_sha256=e["sft"]["completions_sha256"]),
        lambda e: e["rl"].update(completions_sha256=[]),
        lambda e: e["rl"].update(checkpoint_sha256="xyz"),
    ],
)
def test_malformed_unseal_records_are_refused(tmp_path: Path, change: Any) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, _, _ = four_endpoints(tmp_path, ledger)
    change(endpoints)
    with pytest.raises(SealError):
        UnsealRecord.load(unseal_record(tmp_path / "unseal.json", endpoints))


def test_an_arm_with_the_sft_weights_reuses_its_reads_and_one_checkpoint_executes_once(
    tmp_path: Path,
) -> None:
    """Run 2: an arm whose selected weights equal the SFT weights bit for bit was not
    evaluated again. The unseal record maps it to ``sft``: the SFT file executes once, its
    read names the arm, and the arm has no completions of its own to read. An arm the run never
    evaluated is recorded as such."""
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    endpoints["rl"] = {"reuses": "sft", "selected_checkpoint_sha256": "e" * 64}
    record = UnsealRecord.load(unseal_record(tmp_path / "unseal.json", endpoints))
    assert [endpoint.name for endpoint in record.endpoints] == ["base", "sft", "more-sft"]
    assert record.reused_by("sft") == ["rl"] and record.reused_by("base") == []
    summary = execute_completions(
        files["sft"],
        tasks_path,
        tmp_path / "sft-results.jsonl",
        tokenizer=TOKENIZER,
        execute=FakeSandbox(),
        seal_ledger=ledger,
        unseal=tmp_path / "unseal.json",
    )
    assert summary["scored"] == 2
    (opened,) = [event for event in SealLedger(ledger).events() if event["event"] == "opened"]
    assert opened["endpoint"] == "sft" and opened["reused_by"] == ["rl"]
    with pytest.raises(SealError, match="does not list"):  # rl's own file is never read.
        execute_completions(
            files["rl"],
            tasks_path,
            tmp_path / "rl-results.jsonl",
            tokenizer=TOKENIZER,
            execute=FakeSandbox(),
            seal_ledger=ledger,
            unseal=tmp_path / "unseal.json",
        )
    absent = {**endpoints, "more-sft": {"not_evaluated": "the Mac was lost during the control"}}
    loaded = UnsealRecord.load(unseal_record(tmp_path / "absent.json", absent))
    assert loaded.not_evaluated == (("more-sft", "the Mac was lost during the control"),)
    assert [endpoint.name for endpoint in loaded.endpoints] == ["base", "sft"]


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda e: e.update(base={"reuses": "sft", "selected_checkpoint_sha256": "e" * 64}), ""),
        (lambda e: e.update(rl={"reuses": "rl", "selected_checkpoint_sha256": "e" * 64}), ""),
        (  # Only SFT's reads stand for an arm: never base's.
            lambda e: e.update(
                rl={"reuses": "base", "selected_checkpoint_sha256": "e" * 64},
                **{"more-sft": {"reuses": "base", "selected_checkpoint_sha256": "e" * 64}},
            ),
            "only SFT's reads",
        ),
        (lambda e: e.update(rl={"reuses": "sft"}), "selected checkpoint"),
        (lambda e: e.update(rl={"reuses": "sft", "selected_checkpoint_sha256": "x"}), ""),
        (
            lambda e: e.update(
                rl={"reuses": "more-sft", "selected_checkpoint_sha256": "e" * 64},
                **{"more-sft": {"not_evaluated": "lost"}},
            ),
            "only SFT's reads",
        ),
        (lambda e: e.update(rl={"not_evaluated": ""}), "reason"),
        (lambda e: e.update(sft={"not_evaluated": "no"}), ""),
    ],
)
def test_a_reuse_names_an_executing_endpoint_and_only_an_arm_reuses(
    tmp_path: Path, change: Any, message: str
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, _, _ = four_endpoints(tmp_path, ledger)
    change(endpoints)
    with pytest.raises(SealError, match=message or None):
        UnsealRecord.load(unseal_record(tmp_path / "unseal.json", endpoints))


def test_the_unseal_record_must_list_the_file_under_its_checkpoint(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    swapped = json.loads(json.dumps(endpoints))
    swapped["sft"]["completions_sha256"], swapped["rl"]["completions_sha256"] = (
        endpoints["rl"]["completions_sha256"],
        endpoints["sft"]["completions_sha256"],
    )
    with pytest.raises(SealError, match="another checkpoint"):
        execute_completions(
            files["sft"],
            tasks_path,
            tmp_path / "r.jsonl",
            tokenizer=TOKENIZER,
            seal_ledger=ledger,
            unseal=unseal_record(tmp_path / "swapped.json", swapped),
        )
    unsealed_extra = json.loads(json.dumps(endpoints))
    unsealed_extra["base"]["completions_sha256"].append("9" * 64)
    with pytest.raises(SealError, match="not sealed"):
        execute_completions(
            files["sft"],
            tasks_path,
            tmp_path / "r.jsonl",
            tokenizer=TOKENIZER,
            seal_ledger=ledger,
            unseal=unseal_record(tmp_path / "extra.json", unsealed_extra),
        )
    assert [event["event"] for event in SealLedger(ledger).events()] == ["sealed"] * 4


def test_the_ledger_is_hash_chained_and_seals_only_test_completions(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    completions, _, _ = sealed_fixture(tmp_path)
    first = seal_completions(completions, ledger)
    assert seal_completions(completions, ledger) == first  # Idempotent.
    assert len(SealLedger(ledger).events()) == 1
    dev, _, _ = dev_fixture(tmp_path)
    with pytest.raises(SealError, match="only pool test"):
        seal_completions(dev, ledger)
    other, _, _ = sealed_fixture(tmp_path, "e" * 64)
    seal_completions(other, ledger)
    lines = ledger.read_text().splitlines()
    edited = json.loads(lines[0])
    edited["rows"] = 99
    ledger.write_text("\n".join([json.dumps(edited, sort_keys=True), lines[1]]) + "\n")
    with pytest.raises(SealError, match="chain broken"):
        SealLedger(ledger).events()
    ledger.write_text(lines[1] + "\n")  # Dropping the first line breaks the chain too.
    with pytest.raises(SealError, match="chain broken"):
        SealLedger(ledger).events()


def test_cli_seal_and_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    completions, tasks_path, _ = dev_fixture(tmp_path)
    monkeypatch.setattr(harness, "run", lambda code, tests, limits, *, mode: Result("passed"))
    tokenizer = tmp_path / "tokenizer.json"
    TOKENIZER.save(tokenizer)
    results = tmp_path / "results.jsonl"
    arguments = [
        "eval-execute",
        "--completions",
        str(completions),
        "--tasks",
        str(tasks_path),
        "--results",
        str(results),
        "--tokenizer",
        str(tokenizer),
        "--workers",
        "2",
    ]
    assert main(arguments) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["scored"] == 3 and printed["passed"] == 2
    sealed, _, _ = sealed_fixture(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    assert main(["eval-seal", "--completions", str(sealed), "--ledger", str(ledger)]) == 0
    event = json.loads(capsys.readouterr().out)
    assert event["event"] == "sealed"
    assert event["completions_sha256"] == hashlib.sha256(sealed.read_bytes()).hexdigest()


def test_greedy_and_sampled_files_share_one_task_file(tmp_path: Path) -> None:
    completions, tasks_path, tasks = dev_fixture(tmp_path)
    sampled = tmp_path / "sampled.jsonl"
    write_fixture_completions(
        sampled,
        tasks_path,
        tasks,
        lambda task, sample: GOOD if sample % 2 else BAD,
        sampling=Sampling(samples=4, temperature=0.8, top_p=0.95),
    )
    summary = execute_completions(
        sampled,
        tasks_path,
        tmp_path / "sampled-results.jsonl",
        tokenizer=TOKENIZER,
        execute=FakeSandbox(),
    )
    assert summary["attempts"] == 12 and summary["passed"] == 6
    header = read_results(tmp_path / "sampled-results.jsonl").completions_header
    assert header.sampling.samples == 4


def test_two_runs_never_append_to_one_results_file(tmp_path: Path) -> None:
    completions, tasks_path, _ = dev_fixture(tmp_path)
    results = tmp_path / "results.jsonl"
    sandbox = FakeSandbox(down=lambda code: code == BAD)
    execute_completions(
        completions, tasks_path, results, tokenizer=TOKENIZER, execute=sandbox, sleep=no_sleep
    )
    with results.open("ab") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="another eval-execute"):
            execute_completions(
                completions, tasks_path, results, tokenizer=TOKENIZER, execute=FakeSandbox()
            )
    assert len(read_results(results).unresolved) == 1


# --- review findings: regeneration, ledger tail, baseline retry, preflight, environment --


def options(ledger: Path, record: Path, **more: Any) -> dict[str, Any]:
    return {
        "tokenizer": TOKENIZER,
        "seal_ledger": ledger,
        "unseal": record,
        "sleep": no_sleep,
        **more,
    }


def regenerated(
    root: Path, name: str, checkpoint: str, *, sampling: Sampling | None = None
) -> Path:
    """Another completions file for ``checkpoint`` (different answers, so another SHA-256)."""
    tasks_path = root / "test.jsonl"
    tasks = list(load_tasks(tasks_path))
    path = root / name
    write_fixture_completions(
        path,
        tasks_path,
        tasks,
        lambda *_: BAD,
        split="test",
        checkpoint=checkpoint,
        sampling=sampling,
    )
    return path


def test_a_regenerated_file_never_buys_a_second_read(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    # Sealed before any read opens, then offered under a second unseal record.
    again = regenerated(tmp_path, "sft-again.jsonl", "b" * 64)
    again_sha = seal_completions(again, ledger)["completions_sha256"]
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    execute_completions(
        files["sft"],
        tasks_path,
        tmp_path / "sft.jsonl",
        execute=FakeSandbox(),
        **options(ledger, record),
    )
    second = json.loads(json.dumps(endpoints))
    second["sft"]["completions_sha256"] = [again_sha]
    other = unseal_record(tmp_path / "other-unseal.json", second)
    with pytest.raises(SealError, match="another unseal record"):
        execute_completions(
            again,
            tasks_path,
            tmp_path / "again.jsonl",
            execute=FakeSandbox(),
            **options(ledger, other),
        )
    assert not (tmp_path / "again.jsonl").exists()
    # Sealed after the read opened: refused outright, and sealing a known file stays a no-op.
    late = regenerated(tmp_path, "rl-late.jsonl", "c" * 64)
    with pytest.raises(SealError, match="already opened"):
        seal_completions(late, ledger)
    assert seal_completions(files["sft"], ledger)["event"] == "sealed"
    assert [event["event"] for event in SealLedger(ledger).events()].count("opened") == 1


def test_the_unseal_record_cannot_list_two_files_of_one_decoding(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    again = regenerated(tmp_path, "sft-again.jsonl", "b" * 64)
    endpoints["sft"]["completions_sha256"].append(
        seal_completions(again, ledger)["completions_sha256"]
    )
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    for name in ("sft", "rl"):  # Refused for every endpoint's read, before anything is logged.
        with pytest.raises(SealError, match="endpoint sft: .* two completions files"):
            execute_completions(
                files[name],
                tasks_path,
                tmp_path / f"{name}.jsonl",
                execute=FakeSandbox(),
                **options(ledger, record),
            )
    assert [event["event"] for event in SealLedger(ledger).events()] == ["sealed"] * 5


def test_one_checkpoint_may_list_a_greedy_and_a_sampled_file(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    sampled = regenerated(
        tmp_path, "sft-sampled.jsonl", "b" * 64, sampling=Sampling(samples=2, temperature=0.8)
    )
    endpoints["sft"]["completions_sha256"].append(
        seal_completions(sampled, ledger)["completions_sha256"]
    )
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    for name, source in (("greedy", files["sft"]), ("sampled", sampled)):
        summary = execute_completions(
            source,
            tasks_path,
            tmp_path / f"{name}.jsonl",
            execute=FakeSandbox(),
            **options(ledger, record),
        )
        assert summary["unresolved"] == 0
    events = [event["event"] for event in SealLedger(ledger).events()]
    assert events.count("opened") == 2 and events.count("completed") == 2


def test_removing_the_ledgers_tail_or_the_ledger_is_detected(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    results = tmp_path / "sft.jsonl"
    execute_completions(
        files["sft"], tasks_path, results, execute=FakeSandbox(), **options(ledger, record)
    )
    window = read_results(results).windows[0]
    assert window["ledger"]["events"] == 5  # Four seals, then this read's opening.
    head = SealLedger(ledger).head()
    assert head["events"] == 7 and json.loads(Path(f"{ledger}.head").read_text())["events"] == 7
    full, head_file = ledger.read_bytes(), Path(f"{ledger}.head").read_bytes()
    # Dropping the last three events leaves a valid chain; the head record still catches it.
    ledger.write_bytes(b"".join(full.splitlines(keepends=True)[:4]))
    with pytest.raises(SealError, match="events were removed"):
        SealLedger(ledger).events()
    with pytest.raises(SealError, match="events were removed"):
        execute_completions(
            files["sft"],
            tasks_path,
            tmp_path / "again.jsonl",
            execute=FakeSandbox(),
            **options(ledger, record),
        )
    assert not (tmp_path / "again.jsonl").exists()
    # A ledger whose head record is gone, or a head record whose ledger is gone.
    ledger.write_bytes(full)
    Path(f"{ledger}.head").unlink()
    with pytest.raises(SealError, match="no head record"):
        SealLedger(ledger).events()
    Path(f"{ledger}.head").write_bytes(head_file)
    ledger.unlink()
    with pytest.raises(SealError, match="ledger is missing"):
        SealLedger(ledger).events()
    # A head record that disagrees with the chain.
    ledger.write_bytes(full)
    Path(f"{ledger}.head").write_text(
        json.dumps(
            {"format": "scratchcoder-seal-ledger-head", "events": 7, "head_sha256": "0" * 64}
        )
    )
    with pytest.raises(SealError, match="does not match its head record"):
        SealLedger(ledger).events()
    Path(f"{ledger}.head").write_text("{}")
    with pytest.raises(SealError, match="head record"):
        SealLedger(ledger).events()


def test_a_crash_between_the_ledger_line_and_its_head_record_is_repaired(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    first, _, _ = sealed_fixture(tmp_path)
    seal_completions(first, ledger)
    head = Path(f"{ledger}.head")
    stale = head.read_bytes()
    second, _, _ = sealed_fixture(tmp_path, "e" * 64)
    seal_completions(second, ledger)
    head.write_bytes(stale)  # The second line reached the ledger; its head record did not.
    assert len(SealLedger(ledger).events()) == 2
    third, _, _ = sealed_fixture(tmp_path, "f" * 64)
    seal_completions(third, ledger)
    assert json.loads(head.read_text())["events"] == 3


def test_a_refused_test_read_never_opens_the_task_file(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    missing = tmp_path / "nowhere" / "test.jsonl"  # Opening it would raise FileNotFoundError.
    stranger, _, _ = sealed_fixture(tmp_path, "9" * 64)
    with pytest.raises(SealError, match="never sealed"):
        execute_completions(
            stranger,
            missing,
            tmp_path / "r.jsonl",
            execute=FakeSandbox(),
            **options(ledger, record),
        )
    with pytest.raises(SealError, match="sealed"):
        execute_completions(files["sft"], missing, tmp_path / "r.jsonl", tokenizer=TOKENIZER)
    execute_completions(
        files["sft"],
        tasks_path,
        tmp_path / "sft.jsonl",
        execute=FakeSandbox(),
        **options(ledger, record),
    )
    with pytest.raises(SealError, match="already executed"):
        execute_completions(
            files["sft"],
            missing,
            tmp_path / "r.jsonl",
            execute=FakeSandbox(),
            **options(ledger, record),
        )
    assert not (tmp_path / "r.jsonl").exists()


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            (
                "scratchcoder.eval.harness",
                "return result.host_stalled or result.status not in SCORED_STATUSES",
                "return result.status not in SCORED_STATUSES",
            ),
            r"another execution policy \(code: scratchcoder\.eval\.harness differ",
        ),
        (
            ("scratchcoder.eval.harness", "def is_host_failure(", "def is_host_error("),
            "is_host_failure not found",
        ),
    ],
)
def test_a_refused_test_resume_never_opens_the_task_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edit: tuple[str, str, str], message: str
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    results = tmp_path / "base-results.jsonl"
    execute_completions(
        files["base"],
        tasks_path,
        results,
        execute=FakeSandbox(down=lambda code: True),
        **options(ledger, record),
    )
    before, events = results.read_bytes(), SealLedger(ledger).events()
    missing = tmp_path / "nowhere" / "test.jsonl"  # Opening it would raise FileNotFoundError.

    def never(*_: Any) -> None:
        raise AssertionError("the task file was read")

    with monkeypatch.context() as patch:
        patch.setattr(executor, "validate_against_tasks", never)
        edit_sources(patch, {edit[0]: [(edit[1], edit[2])]})
        with pytest.raises(ValueError, match=message):
            execute_completions(
                files["base"], missing, results, execute=FakeSandbox(), **options(ledger, record)
            )
    assert results.read_bytes() == before and SealLedger(ledger).events() == events
    finished = execute_completions(
        files["base"], tasks_path, results, execute=FakeSandbox(), **options(ledger, record)
    )
    assert finished["scored"] == 2


def test_the_test_split_needs_the_tokenizer_before_a_read_is_spent(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    with pytest.raises(ValueError, match="needs the tokenizer"):
        execute_completions(
            files["base"],
            tasks_path,
            tmp_path / "r.jsonl",
            execute=FakeSandbox(),
            seal_ledger=ledger,
            unseal=record,
        )
    assert [event["event"] for event in SealLedger(ledger).events()] == ["sealed"] * 4
    assert not (tmp_path / "r.jsonl").exists()


def constant_hostile(code: str, tests: str, mode: str) -> Result:
    """Passes candidates marked ``# good``; the sandbox is down for constant-answer programs."""
    if "# good" in code:
        return Result("passed")
    return Result("unavailable")


def half_done(root: Path) -> tuple[Path, Path, Path]:
    """A results file with two scored attempts and one unresolved (the host was down for it)."""
    completions, tasks_path, _ = dev_fixture(root)
    results = root / "results.jsonl"
    execute_completions(
        completions,
        tasks_path,
        results,
        tokenizer=TOKENIZER,
        execute=FakeSandbox(down=lambda code: code == BAD),
        sleep=no_sleep,
    )
    return completions, tasks_path, results


def edit_sources(monkeypatch: pytest.MonkeyPatch, edits: dict[str, list[tuple[str, str]]]) -> None:
    """Make the policy see edited source for some modules, as if the files had been changed."""
    original = execution_policy.module_source

    def source(name: str) -> str:
        text = original(name)
        for old, new in edits.get(name, []):
            assert text.count(old) == 1, (name, old)
            text = text.replace(old, new)
        return text

    monkeypatch.setattr(execution_policy, "module_source", source)


HARNESS, PARALLEL = "scratchcoder.eval.harness", "scratchcoder.eval.parallel"
RUNNER, PROGRAM = "scratchcoder.sandbox.runner", "scratchcoder.sandbox.program"
BUDGET, REWARD = "scratchcoder.sandbox.budget", "scratchcoder.sandbox.reward"
EXECUTOR = "scratchcoder.eval.executor"

Change = Callable[[pytest.MonkeyPatch], dict[str, Any]]


def patched(target: object, name: str, value: object) -> Change:
    """Replace ``target.name`` in memory; no extra options for the run."""

    def apply(mp: pytest.MonkeyPatch) -> dict[str, Any]:
        mp.setattr(target, name, value)
        return {}

    return apply


def source_edit(module: str, old: str, new: str) -> Change:
    """Make the policy see ``module``'s file with ``old`` replaced by ``new``."""

    def apply(mp: pytest.MonkeyPatch) -> dict[str, Any]:
        edit_sources(mp, {module: [(old, new)]})
        return {}

    return apply


def together(*changes: Change) -> Change:
    """Several changes at once, as one real file edit is seen both in memory and in source."""

    def apply(mp: pytest.MonkeyPatch) -> dict[str, Any]:
        options: dict[str, Any] = {}
        for change in changes:
            options |= change(mp)
        return options

    return apply


def per_case_seconds(seconds_per_case: float) -> Callable[[int], float]:
    return lambda cases: budget.io_seconds(cases, seconds_per_case=seconds_per_case)


POLICY_CHANGES: dict[str, tuple[Change, str]] = {
    "function timeout": (
        patched(harness, "execution_limits", lambda tests, mode: Limits(timeout_seconds=60.0)),
        r"\(deadline_rule; limits differ",
    ),
    "I/O budget": (patched(harness, "io_seconds", lambda cases: 30.0), r"\(limits differ"),
    # Only the recorded rule text reads the module constant at call time.
    "I/O deadline rule text": (
        patched(budget, "IO_SECONDS_PER_CASE", 0.5),
        r"\(deadline_rule differ",
    ),
    # Editing the constant in budget.py: a new process sees new source, constant and limits.
    "I/O per-case constant edited": (
        together(
            source_edit(BUDGET, "IO_SECONDS_PER_CASE = 0.25", "IO_SECONDS_PER_CASE = 0.5"),
            patched(budget, "IO_SECONDS_PER_CASE", 0.5),
            patched(harness, "io_seconds", per_case_seconds(0.5)),
        ),
        r"\(code: scratchcoder\.sandbox\.budget; deadline_rule; limits differ",
    ),
    "scored statuses": (
        patched(harness, "SCORED_STATUSES", frozenset({"passed", "failed"})),
        r"\(scored_statuses differ",
    ),
    "input limit": (patched(executor, "CALL_INPUT_LIMIT", 10), r"\(input_limit differ"),
    "image": (patched(runner, "IMAGE", "python@sha256:" + "0" * 64), r"\(image differ"),
    "host retries": (lambda mp: {"retries": 1}, r"host_backoff_seconds; host_retries differ"),
    "execute": (lambda mp: {"execute": constant_hostile}, r"\(execute differ"),
    "host-status rule code": (
        source_edit(
            HARNESS,
            "return result.host_stalled or result.status not in SCORED_STATUSES",
            "return result.status not in SCORED_STATUSES",
        ),
        r"\(code: scratchcoder\.eval\.harness differ",
    ),
    "oversized rule code": (
        source_edit(
            PARALLEL,
            "if _size(call.code + call.tests) > input_limit:",
            "if _size(call.code + call.tests) >= input_limit:",
        ),
        r"\(code: scratchcoder\.eval\.parallel differ",
    ),
    "supervisor code": (
        source_edit(
            PROGRAM,
            "if not last_output.rstrip().endswith(marker):",
            "if marker not in last_output:",
        ),
        r"\(code: scratchcoder\.sandbox\.program differ",
    ),
    "runner code": (
        source_edit(RUNNER, "HOST_WATCHDOG_SECONDS = 10.0", "HOST_WATCHDOG_SECONDS = 20.0"),
        r"\(code: scratchcoder\.sandbox\.runner differ",
    ),
    "pass rule code": (
        source_edit(
            EXECUTOR,
            'return status == "passed"',
            'return status in ("passed", "oversized")',
        ),
        r"\(code: scratchcoder\.eval\.executor differ",
    ),
    "unexecuted status code": (
        source_edit(
            EXECUTOR,
            "return CONTEXT_LIMIT if stop_reason == CONTEXT_LIMIT else NO_PROGRAM",
            "return NO_PROGRAM",
        ),
        r"\(code: scratchcoder\.eval\.executor differ",
    ),
    "executor input limit code": (
        source_edit(
            EXECUTOR,
            "max_consecutive_unresolved=max_consecutive_unresolved,\n"
            "        input_limit=CALL_INPUT_LIMIT,",
            "max_consecutive_unresolved=max_consecutive_unresolved,",
        ),
        r"\(code: scratchcoder\.eval\.executor differ",
    ),
    "budget code": (
        source_edit(
            BUDGET,
            "seconds += seconds_per_case * cases",
            "seconds += 2 * seconds_per_case * cases",
        ),
        r"\(code: scratchcoder\.sandbox\.budget differ",
    ),
}


@pytest.mark.parametrize("change", sorted(POLICY_CHANGES))
def test_a_resumed_run_refuses_another_execution_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    completions, tasks_path, results = half_done(tmp_path)
    before = results.read_bytes()
    apply, message = POLICY_CHANGES[change]
    with monkeypatch.context() as other:
        options: dict[str, Any] = {"execute": FakeSandbox(), **apply(other)}
        with pytest.raises(ValueError, match=message):
            execute_completions(
                completions, tasks_path, results, tokenizer=TOKENIZER, sleep=no_sleep, **options
            )
    assert results.read_bytes() == before  # The refused run appended nothing.
    healthy = FakeSandbox()
    execute_completions(
        completions, tasks_path, results, tokenizer=TOKENIZER, execute=healthy, sleep=no_sleep
    )
    assert healthy.calls == [BAD] and read_results(results).complete


def test_documentation_and_unlisted_code_changes_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completions, tasks_path, results = half_done(tmp_path)
    recorded = read_results(results).header["execution"]["policy_sha256"]
    edit_sources(
        monkeypatch,
        {
            HARNESS: [
                ("A result that says nothing about", "A result that tells nothing about"),
                ("def host_backoff(", "# A new comment.\ndef host_backoff("),
                # evaluate() is not a listed unit: it never runs in eval-execute.
                ("nonempty unique tasks", "distinct tasks"),
            ],
            PARALLEL: [("Run sandbox calls in parallel", "Run the sandbox calls in parallel")],
            PROGRAM: [("executed ONLY inside", "run ONLY inside")],
            # Docstring of a listed executor unit, and code outside the listed units.
            EXECUTOR: [
                ("The status to score for an outcome", "The status scored for an outcome"),
                (
                    'raise ValueError("the results file belongs to other completions")',
                    'raise ValueError("the results file is for other completions")',
                ),
            ],
            BUDGET: [("The largest deadline the runner", "The longest deadline the runner")],
        },
    )
    healthy = FakeSandbox()
    execute_completions(
        completions, tasks_path, results, tokenizer=TOKENIZER, execute=healthy, sleep=no_sleep
    )
    final = read_results(results)
    assert healthy.calls == [BAD] and final.complete
    assert final.header["execution"]["policy_sha256"] == recorded
    starts = [window for window in final.windows if window["kind"] == "window"]
    assert [window["execution_policy_sha256"] for window in starts] == [recorded] * 2


def test_an_audit_change_resumes_and_is_recorded_per_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completions, tasks_path, results = half_done(tmp_path)
    assert len(read_results(results).audit_policies) == 1
    edit_sources(monkeypatch, {REWARD: [("def audit_attempt(", "def audit_attempt_(")]})
    execute_completions(
        completions, tasks_path, results, tokenizer=TOKENIZER, execute=FakeSandbox()
    )
    final = read_results(results)
    assert final.complete and len(final.audit_policies) == 2  # Flag counts mix two audits.


def test_a_moved_verdict_definition_is_an_error_not_a_silent_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completions, tasks_path, _ = dev_fixture(tmp_path)
    edit_sources(monkeypatch, {HARNESS: [("def is_host_failure(", "def is_host_error(")]})
    with pytest.raises(ValueError, match="is_host_failure not found"):
        execute_completions(
            completions,
            tasks_path,
            tmp_path / "r.jsonl",
            tokenizer=TOKENIZER,
            execute=FakeSandbox(),
        )
    assert not (tmp_path / "r.jsonl").exists()


def test_an_interrupted_and_resumed_run_equals_an_uninterrupted_one(tmp_path: Path) -> None:
    tasks = [pool_task(f"t/{index:02}") for index in range(12)] + [io_task("t/io")]
    tasks_path = save_task_file(tmp_path / "dev.jsonl", tasks)
    programs = {task.id: [GOOD, BAD, None][index % 3] for index, task in enumerate(tasks)}
    completions = tmp_path / "completions.jsonl"
    write_fixture_completions(completions, tasks_path, tasks, lambda task, _: programs[task.id])
    whole = tmp_path / "whole.jsonl"
    execute_completions(
        completions, tasks_path, whole, tokenizer=TOKENIZER, execute=FakeSandbox(), workers=1
    )
    pieces = tmp_path / "pieces.jsonl"
    for stop in (2, 3):
        with pytest.raises(KeyboardInterrupt):
            execute_completions(
                completions,
                tasks_path,
                pieces,
                tokenizer=TOKENIZER,
                execute=FakeSandbox(interrupt_after=stop),
                workers=1,
            )
    execute_completions(
        completions, tasks_path, pieces, tokenizer=TOKENIZER, execute=FakeSandbox(), workers=1
    )
    one, resumed = read_results(whole), read_results(pieces)
    assert one.complete and resumed.complete
    assert resumed.attempts == one.attempts
    assert sum(window["kind"] == "window" for window in resumed.windows) == 3
    assert resumed.header["execution"] == one.header["execution"]
    assert {window.get("execution_policy_sha256") for window in resumed.windows} == {
        one.header["execution"]["policy_sha256"],
        None,  # window_end records carry counts only.
    }


def schema_one(results: Path) -> None:
    """Rewrite a results header the way schema 1 wrote it (four compared fields)."""
    lines = results.read_bytes().splitlines(keepends=True)
    header = json.loads(lines[0])
    header["schema_version"] = 1
    header["execution"] = {
        "execute": "scratchcoder.eval.harness.isolated_execution",
        "image": runner.IMAGE,
        "deadline_rule": "function: 2 s; I/O: 2 s + 0.25 s per case, at most 60 s; 128 MiB",
        "runner_sha256": "1" * 64,
        "program_sha256": "2" * 64,
        "host_retries": 3,
        "host_backoff": "2 s doubling, at most 30 s",
        "audit": "scratchcoder.sandbox.reward.audit_attempt on passing attempts",
    }
    lines[0] = json.dumps(header, sort_keys=True).encode() + b"\n"
    for number, line in enumerate(lines[1:], 1):
        record = json.loads(line)
        if record["kind"] == "window":
            del record["execution_policy_sha256"], record["audit_policy_sha256"]
            lines[number] = json.dumps(record, sort_keys=True).encode() + b"\n"
    results.write_bytes(b"".join(lines))


def test_a_schema_one_file_stays_readable_but_is_never_resumed(tmp_path: Path) -> None:
    completions, tasks_path, results = half_done(tmp_path)
    schema_one(results)
    before = results.read_bytes()
    old = read_results(results)
    assert old.schema_version == 1 and old.audit_policies == set()
    assert len(old.attempts) == 2 and set(old.unresolved) == {("t/2", 0)}
    run = run_from_results(results)  # The scorecard still reads it.
    assert run.outcomes["t/1"] == [True] and run.unresolved == {"t/2": 1}
    sandbox = FakeSandbox()
    with pytest.raises(ValueError, match="schema 1, which recorded only a partial"):
        execute_completions(completions, tasks_path, results, tokenizer=TOKENIZER, execute=sandbox)
    assert results.read_bytes() == before and sandbox.calls == []
    fresh = tmp_path / "fresh.jsonl"
    execute_completions(completions, tasks_path, fresh, tokenizer=TOKENIZER, execute=sandbox)
    assert read_results(fresh).schema_version == 2 and read_results(fresh).complete


def test_an_unlogged_test_baseline_is_retried_after_every_read_finished(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    with pytest.raises(SealError, match="no completed read"):
        run_test_baseline(tasks_path, ledger, record)
    for name in ("base", "sft", "rl", "more-sft"):
        summary = execute_completions(
            files[name],
            tasks_path,
            tmp_path / f"{name}.jsonl",
            execute=constant_hostile,
            **options(ledger, record),
        )
        assert summary["scored"] == 2 and summary["test_baseline"] is None
    events = [event["event"] for event in SealLedger(ledger).events()]
    assert "baseline" not in events and events.count("completed") == 4
    healthy_calls: list[str] = []

    def healthy(code: str, tests: str, mode: str) -> Result:
        healthy_calls.append(code)
        return Result("passed")

    outcome = run_test_baseline(tasks_path, ledger, record, execute=healthy, sleep=no_sleep)
    assert outcome["recorded"] is True and outcome["test_baseline"] is not None
    saved = json.loads(Path(outcome["test_baseline"]).read_text())
    assert saved["split"] == "test" and saved["summary"]["tasks"] == 2
    assert [event["event"] for event in SealLedger(ledger).events()].count("baseline") == 1
    calls = len(healthy_calls)
    assert run_test_baseline(tasks_path, ledger, record, execute=healthy)["recorded"] is True
    assert len(healthy_calls) == calls  # Already logged: nothing runs again.
    # The gate also pins the unseal record and the task file.
    other = tmp_path / "other"
    other.mkdir()
    _, other_tasks, _ = sealed_fixture(other, "d" * 64)
    other_tasks.write_text(other_tasks.read_text().replace("Write add_one", "Write add_two"))
    with pytest.raises(SealError, match="not the test task file"):
        run_test_baseline(other_tasks, ledger, record)
    changed = json.loads(json.dumps(endpoints))
    changed["base"]["completions_sha256"] = ["1" * 64]
    with pytest.raises(SealError, match="not sealed"):
        run_test_baseline(tasks_path, ledger, unseal_record(tmp_path / "changed.json", changed))


def test_a_baseline_file_saved_without_its_ledger_event_is_recorded_not_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    original = SealLedger.record_baseline
    broken = {"on": True}

    def flaky(self: SealLedger, **fields: Any) -> dict[str, Any]:
        if broken["on"]:
            raise OSError("disk full")
        return original(self, **fields)

    monkeypatch.setattr(SealLedger, "record_baseline", flaky)
    calls: list[str] = []

    def sandbox(code: str, tests: str, mode: str) -> Result:
        calls.append(code)
        return Result("passed")

    with pytest.raises(OSError, match="disk full"):
        execute_completions(
            files["sft"],
            tasks_path,
            tmp_path / "sft.jsonl",
            execute=sandbox,
            **options(ledger, record),
        )
    saved = next(tmp_path.glob("pool-test-baseline-*.json"))
    first = saved.read_bytes()
    assert "baseline" not in [event["event"] for event in SealLedger(ledger).events()]
    broken["on"] = False
    before = len(calls)
    summary = execute_completions(
        files["rl"], tasks_path, tmp_path / "rl.jsonl", execute=sandbox, **options(ledger, record)
    )
    assert summary["test_baseline"] == str(saved) and saved.read_bytes() == first
    assert len(calls) - before == 2  # Only the rl attempts ran; the baseline was not recomputed.
    event = SealLedger(ledger).baseline_event(read_completions(files["rl"]).header.task_file_sha256)
    assert event is not None and event["sha256"] == hashlib.sha256(first).hexdigest()


def test_a_planted_baseline_for_another_task_file_is_refused(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    digest = read_completions(files["sft"]).header.task_file_sha256
    planted = tmp_path / f"pool-test-baseline-{digest[:16]}.json"
    planted.write_text(json.dumps({"split": "test", "task_file_sha256": "0" * 64}))
    with pytest.raises(SealError, match="not the test baseline of this task file"):
        execute_completions(
            files["sft"],
            tasks_path,
            tmp_path / "sft.jsonl",
            execute=FakeSandbox(),
            **options(ledger, record),
        )
    assert "baseline" not in [event["event"] for event in SealLedger(ledger).events()]


def test_cli_test_baseline_retry_exits_zero_once_logged_and_three_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, tasks_path = four_endpoints(tmp_path, ledger)
    record = unseal_record(tmp_path / "unseal.json", endpoints)
    arguments = [
        "eval-test-baseline",
        "--tasks",
        str(tasks_path),
        "--seal-ledger",
        str(ledger),
        "--unseal",
        str(record),
        "--workers",
        "2",
    ]
    with pytest.raises(SealError, match="no completed read"):
        main(arguments)
    for name in endpoints:
        execute_completions(
            files[name],
            tasks_path,
            tmp_path / f"{name}.jsonl",
            execute=constant_hostile,
            **options(ledger, record),
        )
    with monkeypatch.context() as unresolved:
        unresolved.setattr(
            executor,
            "run_test_baseline",
            lambda *_, **__: {"recorded": False, "test_baseline": None},
        )
        assert main(arguments) == 3
    capsys.readouterr()
    monkeypatch.setattr(harness, "run", lambda code, tests, limits, *, mode: Result("passed"))
    assert main(arguments) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["recorded"] is True and Path(printed["test_baseline"]).is_file()
