"""The Mac executor and the test seal, with fake execution (no generated code runs here)."""

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
from scratchcoder.eval import harness
from scratchcoder.eval.completions import Sampling, read_completions
from scratchcoder.eval.executor import execute_completions, read_results
from scratchcoder.eval.parallel import HostUnavailableError
from scratchcoder.eval.seal import SealError, SealLedger, UnsealRecord, seal_completions
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.runner import Result
from scratchcoder.tokenizer.bpe import ByteBPE

GOOD = "def add_one(x):\n    return x + 1  # good"
BAD = "def add_one(x):\n    return x"


class FakeSandbox:
    """Passes programs marked ``# good``; can stall or fail the host for chosen programs."""

    def __init__(self, *, stall: int = 0, down: Callable[[str], bool] | None = None) -> None:
        self.calls: list[str] = []
        self.stall = stall
        self.down = down or (lambda code: False)
        self.lock = threading.Lock()

    def __call__(self, code: str, tests: str, mode: str) -> Result:
        with self.lock:
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
    seen = 0

    def interrupted(code: str, tests: str, mode: str) -> Result:
        nonlocal seen
        seen += 1
        if seen > 5:
            raise KeyboardInterrupt
        return Result("passed")

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
