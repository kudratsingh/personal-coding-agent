"""Pod task exports: hidden tests never leave the Mac, and execution re-derives the export.

Fixture task files only; the real dev and test splits are exported by the runbook at launch.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from eval_fixtures import TOKENIZER, io_task, pool_task, save_task_file, write_fixture_completions
from test_eval_executor import GOOD, FakeSandbox, four_endpoints, no_sleep, unseal_record

from scratchcoder.cli import main
from scratchcoder.eval import export
from scratchcoder.eval.baseline import pool_baseline_report
from scratchcoder.eval.completions import read_completions, validate_against_tasks
from scratchcoder.eval.executor import execute_completions, read_results, run_test_baseline
from scratchcoder.eval.export import (
    ExportError,
    export_bytes,
    is_exported,
    placeholder,
    task_file_identity,
    write_export,
)
from scratchcoder.eval.scorecard import _pool_baseline_for
from scratchcoder.eval.seal import SealError, seal_completions
from scratchcoder.eval.tasks import Task, load_tasks
from scratchcoder.rl import command
from scratchcoder.rl.attempts import ScreeningHeader, read_attempts_file, write_attempts
from scratchcoder.rl.command import screen_attempts
from scratchcoder.rl.rewards import Attempt, LocalRewardSource
from scratchcoder.sandbox.reward import COMPLETED, RewardResult
from scratchcoder.sandbox.runner import Result


def dev_tasks() -> list[Task]:
    return [pool_task("t/1"), pool_task("t/2", source="synthetic-b"), io_task("t/3", cases=5)]


def exported_file(tmp_path: Path, tasks: list[Task], name: str = "dev") -> tuple[Path, Path]:
    full = save_task_file(tmp_path / f"{name}.jsonl", tasks)
    output = tmp_path / "pod" / f"{name}-export.jsonl"
    write_export("pool", tasks_path=full, output=output, record=tmp_path / f"{name}-record.json")
    return full, output


def test_an_export_keeps_everything_but_the_hidden_tests(tmp_path: Path) -> None:
    tasks = dev_tasks()
    full, output = exported_file(tmp_path, tasks)
    shipped = list(load_tasks(output))  # Still a valid task file: generation reads it unchanged.
    assert [task.id for task in shipped] == [task.id for task in tasks]
    for original, copy in zip(tasks, shipped, strict=True):
        assert copy.hidden_tests == placeholder(original.mode, original.hidden_tests)
        assert is_exported(copy) and not is_exported(original)
        assert (copy.prompt, copy.public_tests, copy.solution, copy.mode, copy.split) == (
            original.prompt,
            original.public_tests,
            original.solution,
            original.mode,
            original.split,
        )
    text = output.read_text()
    for task in tasks:
        assert task.hidden_tests not in text and json.dumps(task.hidden_tests) not in text
    record = json.loads((tmp_path / "dev-record.json").read_text())
    assert record["source"]["sha256"] == export.file_sha256(full)
    assert record["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert record["tasks"] == 3 and record["by_mode"] == {"function": 2, "io": 1}
    assert "created_at" not in record  # Deterministic: the same inputs give the same record.


def test_exports_are_deterministic_and_never_replaced(tmp_path: Path) -> None:
    full, output = exported_file(tmp_path, dev_tasks())
    again = tmp_path / "again.jsonl"
    write_export("pool", tasks_path=full, output=again, record=tmp_path / "again-record.json")
    assert again.read_bytes() == output.read_bytes() == export_bytes(full)
    with pytest.raises(FileExistsError):
        write_export("pool", tasks_path=full, output=again, record=tmp_path / "new-record.json")
    with pytest.raises(FileExistsError):
        write_export(
            "pool",
            tasks_path=full,
            output=tmp_path / "x.jsonl",
            record=tmp_path / "again-record.json",
        )
    with pytest.raises(ExportError, match="already a Pod export"):
        export_bytes(output)


def test_a_candidates_export_follows_the_candidate_file(tmp_path: Path) -> None:
    train = [pool_task(f"r/{index}", split="train") for index in range(4)]
    full = save_task_file(tmp_path / "train.jsonl", train)
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        "".join(json.dumps({"id": i, "source": "synthetic-a"}) + "\n" for i in ("r/2", "r/0"))
    )
    output = tmp_path / "candidates-export.jsonl"
    record = write_export(
        "candidates",
        tasks_path=full,
        candidates_path=candidates,
        output=output,
        record=tmp_path / "candidates-record.json",
    )
    assert [task.id for task in load_tasks(output)] == ["r/2", "r/0"]
    assert record["candidates"]["sha256"] == export.file_sha256(candidates)
    identity = task_file_identity(full, export.file_sha256(output), candidates_path=candidates)
    assert identity == "export"
    with pytest.raises(ValueError, match="only it"):
        write_export(
            "pool",
            tasks_path=full,
            candidates_path=candidates,
            output=tmp_path / "y",
            record=tmp_path / "z",
        )


def test_execution_rederives_the_export_from_the_full_file(tmp_path: Path) -> None:
    tasks = dev_tasks()
    full, shipped = exported_file(tmp_path, tasks)
    programs = {
        "t/1": GOOD,
        "t/2": "def add_one(x):\n    return x",
        "t/3": "print(int(input()) + 1)",
    }
    completions = tmp_path / "completions.jsonl"
    # Generated on the Pod: the header names the export's SHA-256, not the full file's.
    write_fixture_completions(completions, shipped, tasks, lambda task, _: programs[task.id])
    header = read_completions(completions).header
    assert header.task_file_sha256 == export.file_sha256(shipped) != export.file_sha256(full)
    assert [task.id for task in validate_against_tasks(read_completions(completions), full)] == [
        "t/1",
        "t/2",
        "t/3",
    ]
    seen: list[tuple[str, str]] = []

    class Recording(FakeSandbox):
        def __call__(self, code: str, tests: str, mode: str) -> Result:
            seen.append((tests, mode))
            return super().__call__(code, tests, mode)

    summary = execute_completions(
        completions,
        full,
        tmp_path / "results.jsonl",
        tokenizer=TOKENIZER,
        execute=Recording(),
        sleep=no_sleep,
    )
    assert summary["scored"] == 3 and summary["passed"] == 1
    # The real hidden tests ran, on the Mac; no placeholder ever reached the sandbox.
    assert {tests for tests, _ in seen} == {task.hidden_tests for task in tasks}
    assert read_results(tmp_path / "results.jsonl").header["task_file_sha256"] == (
        header.task_file_sha256
    )


def test_execution_refuses_the_export_itself_and_any_other_full_file(tmp_path: Path) -> None:
    tasks = dev_tasks()
    full, shipped = exported_file(tmp_path, tasks)
    completions = tmp_path / "completions.jsonl"
    write_fixture_completions(completions, shipped, tasks, lambda *_: GOOD)
    with pytest.raises(ValueError, match="Pod export with placeholder"):
        execute_completions(completions, shipped, tmp_path / "r1.jsonl", execute=FakeSandbox())
    other = save_task_file(tmp_path / "other.jsonl", [*tasks[:2], io_task("t/3", cases=6)])
    with pytest.raises(ValueError, match="SHA-256 differs"):
        execute_completions(completions, other, tmp_path / "r2.jsonl", execute=FakeSandbox())
    assert not (tmp_path / "r1.jsonl").exists() and not (tmp_path / "r2.jsonl").exists()
    # Completions generated from the full file still validate against it, as before.
    direct = tmp_path / "direct.jsonl"
    write_fixture_completions(direct, full, tasks, lambda *_: GOOD)
    assert len(validate_against_tasks(read_completions(direct), full)) == 3


def test_a_dev_baseline_matches_runs_generated_from_its_export(tmp_path: Path) -> None:
    tasks = [pool_task("t/1"), pool_task("t/2")]
    full, shipped = exported_file(tmp_path, tasks)

    def sandbox(code: str, tests: str, mode: str) -> Result:
        return Result("failed")

    report = pool_baseline_report(full, split="dev", execute=sandbox, sleep=no_sleep)
    assert report["pod_export_sha256"] == export.file_sha256(shipped)
    completions = tmp_path / "completions.jsonl"
    write_fixture_completions(completions, shipped, tasks, lambda *_: GOOD)

    class Run:
        header = read_completions(completions).header

    assert _pool_baseline_for(Run(), [report]) is report  # type: ignore[arg-type]


def test_a_sealed_read_and_its_baseline_retry_accept_the_derived_export(tmp_path: Path) -> None:
    """Sealed test completions generated from an export execute against the full test file,
    and the baseline retry accepts that file by re-deriving the export."""
    ledger = tmp_path / "ledger.jsonl"
    endpoints, files, full = four_endpoints(tmp_path, ledger)
    # Replace each endpoint's file with one generated from the test file's export.
    shipped = tmp_path / "pod" / "test-export.jsonl"
    write_export("pool", tasks_path=full, output=shipped, record=tmp_path / "test-record.json")
    tasks = list(load_tasks(full))
    ledger2 = tmp_path / "ledger2.jsonl"
    endpoints2: dict[str, Any] = {}
    for name, letter in (("base", "a"), ("sft", "b"), ("rl", "c"), ("more-sft", "d")):
        path = tmp_path / f"export-{name}.jsonl"
        write_fixture_completions(
            path, shipped, tasks, lambda *_: GOOD, split="test", checkpoint=letter * 64
        )
        files[name] = path
        digest = seal_completions(path, ledger2)["completions_sha256"]
        endpoints2[name] = {"checkpoint_sha256": letter * 64, "completions_sha256": [digest]}
    record = unseal_record(tmp_path / "unseal2.json", endpoints2)

    def hostile(code: str, tests: str, mode: str) -> Result:
        return Result("timeout", host_stalled=True)  # The baseline's calls stay unresolved.

    calls: list[str] = []

    def sandbox(code: str, tests: str, mode: str) -> Result:
        calls.append(tests)
        return Result("passed")

    for name in ("base", "sft", "rl", "more-sft"):
        summary = execute_completions(
            files[name],
            full,
            tmp_path / f"{name}-results.jsonl",
            tokenizer=TOKENIZER,
            execute=sandbox,
            seal_ledger=ledger2,
            unseal=record,
        )
        assert summary["scored"] == 2
    assert calls and not any(export.WITHHELD in tests for tests in calls)
    outcome = run_test_baseline(full, ledger2, record, execute=hostile, sleep=no_sleep)
    assert outcome["task_file_sha256"] == export.file_sha256(shipped)
    with pytest.raises(SealError, match="not the test task file"):
        run_test_baseline(shipped, ledger2, record, execute=hostile, sleep=no_sleep)


def test_cli_writes_a_pool_export(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    full = save_task_file(tmp_path / "dev.jsonl", dev_tasks())
    output, record = tmp_path / "export.jsonl", tmp_path / "record.json"
    arguments = ["pod-task-export", "--kind", "pool", "--tasks", str(full)]
    assert main([*arguments, "--output", str(output), "--record", str(record)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"]["sha256"] == export.file_sha256(output)
    assert output.read_bytes() == export_bytes(full)


# --- screening attempts generated on a Pod ------------------------------------------------


def screening_fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[Task]]:
    train = [
        pool_task(
            "a", split="train", hidden_tests="assert add_one(6) == 7\nassert add_one(7) == 8"
        ),
        pool_task(
            "b", split="train", hidden_tests="assert add_one(1) == 2\nassert add_one(2) == 3"
        ),
    ]
    full = save_task_file(tmp_path / "train.jsonl", train)
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        "".join(json.dumps({"id": t.id, "source": t.source}) + "\n" for t in train)
    )
    shipped = tmp_path / "pod" / "candidates-export.jsonl"
    write_export(
        "candidates",
        tasks_path=full,
        candidates_path=candidates,
        output=shipped,
        record=tmp_path / "candidates-record.json",
    )
    return full, candidates, shipped, train


def attempts_header(shipped: Path, candidates: Path, **changes: Any) -> ScreeningHeader:
    values: dict[str, Any] = {
        "task_file_sha256": export.file_sha256(shipped),
        "candidates_sha256": export.file_sha256(candidates),
        "checkpoint_sha256": "c" * 64,
        "tokenizer_sha256": TOKENIZER.sha256,
        "tasks": 2,
        "samples": 8,
        "attempts": 16,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 256,
        "seed_base": 2**40,
        "precision": "bfloat16",
        "device": "cuda:0",
        "batch_size": 64,
        "generation_seconds": 1.5,
        "created_at": "2026-10-27T00:00:00Z",
    }
    values.update(changes)
    return ScreeningHeader(**values)


def fake_rewards(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    scored: list[str] = []

    def score(item: Task, code: str) -> RewardResult:
        scored.append(item.hidden_tests)
        return RewardResult(COMPLETED, 0.5 if "x" in code else 0.0, 1, 2)

    def local(tasks: Any, **options: Any) -> LocalRewardSource:
        return LocalRewardSource(tasks, score=score, **options)

    monkeypatch.setattr(command, "LocalRewardSource", local)
    return scored


def test_rl_screen_binds_pod_attempts_to_the_full_train_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full, candidates, shipped, train = screening_fixture(tmp_path)
    scored = fake_rewards(monkeypatch)
    attempts = [
        Attempt(t.id, s, "x" * (s + 1) if t.id == "a" else "y", 3) for t in train for s in range(8)
    ]
    path = tmp_path / "attempts.jsonl"
    write_attempts(path, attempts_header(shipped, candidates), attempts)
    header, read = read_attempts_file(path)
    assert header is not None and read == attempts
    output = tmp_path / "screening.json"
    screen_attempts(
        tasks_path=full,
        candidates_path=candidates,
        attempts_path=path,
        output=output,
        timeout=60.0,
        floor=1,
        require_header=True,
    )
    record = json.loads(output.read_text())
    assert record["inputs"]["header_task_file"] == "export"
    assert set(scored) == {task.hidden_tests for task in train}  # The Mac's real tests.


@pytest.mark.parametrize(
    "problem",
    [
        "other_candidates",
        "other_tasks",
        "missing_sample",
        "export_given",
        "no_header",
        "other_temperature",
        "other_cap",
        "other_seed",
    ],
)
def test_rl_screen_refuses_attempts_it_cannot_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    full, candidates, shipped, train = screening_fixture(tmp_path)
    scored = fake_rewards(monkeypatch)
    attempts = [Attempt(t.id, s, "x", 3) for t in train for s in range(8)]
    header = attempts_header(shipped, candidates)
    tasks_path = full
    if problem == "other_candidates":
        header = attempts_header(shipped, candidates, candidates_sha256="0" * 64)
    elif problem == "other_tasks":
        header = attempts_header(shipped, candidates, task_file_sha256="1" * 64)
    elif problem == "missing_sample":
        attempts = [*attempts[:15], Attempt("a", 9, "x", 3)]
    elif problem == "export_given":
        tasks_path = shipped
    elif problem == "other_temperature":
        header = attempts_header(shipped, candidates, temperature=0.8)
    elif problem == "other_cap":
        header = attempts_header(shipped, candidates, max_new_tokens=512)
    elif problem == "other_seed":
        header = attempts_header(shipped, candidates, seed_base=0)
    path = tmp_path / "attempts.jsonl"
    if problem == "no_header":
        path.write_text("".join(json.dumps(item.to_json()) + "\n" for item in attempts))
    else:
        write_attempts(path, header, attempts)
    with pytest.raises(ValueError):
        screen_attempts(
            tasks_path=tasks_path,
            candidates_path=candidates,
            attempts_path=path,
            output=tmp_path / "screening.json",
            timeout=60.0,
            floor=1,
            require_header=True,
        )
    assert not scored and not (tmp_path / "screening.json").exists()


def test_an_arc_export_writes_choice_lines_in_id_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scratchcoder.data import arc
    from scratchcoder.eval.choices import ChoiceTask, load_choices

    questions = [
        ChoiceTask(
            id=identifier,
            question=f"Question {identifier}?",
            labels=("A", "B"),
            choices=("yes", "no"),
            answer="A",
            source="allenai/ai2_arc",
            source_revision=arc.REVISION,
            license="CC-BY-SA-4.0",
            split="test",
        )
        for identifier in ("q2", "q1")
    ]
    metadata = {"data_manifests": {"test": {"sha256": "f" * 64}}, "source_split_counts": {}}
    monkeypatch.setattr(arc, "read_arc_easy", lambda directory: (questions, metadata))
    output = tmp_path / "arc-easy-test.jsonl"
    record = write_export(
        "arc", arc_directory=tmp_path, output=output, record=tmp_path / "arc-record.json"
    )
    assert [question.id for question in load_choices(output)] == ["q1", "q2"]
    assert record["questions"] == 2 and record["source"]["manifests"] == metadata["data_manifests"]
    with pytest.raises(ValueError, match="data directory only"):
        write_export("arc", tasks_path=output, output=tmp_path / "a", record=tmp_path / "b")


def test_the_screening_protocol_is_the_rollouts() -> None:
    from scratchcoder.rl.attempts import SCREENING_PROTOCOL
    from scratchcoder.rl.rollout import ROLLOUT_TOKENS, SCREENING_SEED
    from scratchcoder.rl.screening import SAMPLES

    assert SCREENING_PROTOCOL == {
        "samples": SAMPLES,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": ROLLOUT_TOKENS,
        "seed_base": SCREENING_SEED,
    }
