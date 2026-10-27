"""``scratchcoder rl-pod-export``: run 2's withheld RL and dev-subset files and the facts file.

Authored fixture tasks only (6b's tasks with marker hidden tests); no real pool file is read.
Every Pod file is scanned for a hidden test with 6b's marker scan.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from bridge_fixtures import DEV_TASKS, TRAIN
from test_rl_bridge_end_to_end import HIDDEN, assert_no_hidden_test_in, with_hidden

from scratchcoder.cli import main
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.tasks import Task, load_tasks, save_tasks
from scratchcoder.rl.dev import dev_subset
from scratchcoder.rl.facts import facts_bytes, load_facts, task_facts
from scratchcoder.rl.pod_export import FILES, RECORD, read_record, write_rl_pod_export
from scratchcoder.rl.withheld import is_withheld, withheld_task
from tools.posttrain_binding import require_export

KEPT = ["t3", "t1", "t5", "t0"]


def sources(root: Path, *, kept: list[str] = KEPT) -> dict[str, Path]:
    files = {
        "train_tasks": root / "pool" / "train.jsonl",
        "dev_tasks": root / "pool" / "dev.jsonl",
        "kept": root / "screening.json",
    }
    save_tasks(files["train_tasks"], with_hidden(TRAIN))
    save_tasks(files["dev_tasks"], with_hidden(DEV_TASKS))
    files["kept"].write_text(
        json.dumps({"format": "rl-screening-v1", "outcome": "kept", "kept": kept})
    )
    return files


def resave(path: Path, tasks: list[Task]) -> None:
    path.unlink()
    save_tasks(path, tasks)


def export(root: Path, files: dict[str, Path], name: str = "export") -> Path:
    write_rl_pod_export(**files, output=root / name, dev_size=3)
    return root / name


def test_the_export_writes_withheld_tasks_the_subset_and_the_facts(tmp_path: Path) -> None:
    files = sources(tmp_path)
    output = export(tmp_path, files)
    record = read_record(output / RECORD)
    rl = list(load_tasks(output / FILES["rl-tasks"]))
    dev = list(load_tasks(output / FILES["rl-dev"]))
    full_train = {task.id: task for task in with_hidden(TRAIN)}
    full_dev = with_hidden(DEV_TASKS)
    assert [task.id for task in rl] == KEPT  # The screening record's order.
    assert rl == [withheld_task(full_train[identifier]) for identifier in KEPT]
    assert all(is_withheld(task) and task.solution for task in rl)  # References kept.
    subset = dev_subset(full_dev, 3)
    assert dev == [withheld_task(task) for task in subset]
    assert all(is_withheld(task) and task.solution is None for task in dev)
    facts_raw = (output / FILES["rl-facts"]).read_bytes()
    facts = load_facts(facts_raw, sha256=record["files"]["rl-facts"]["sha256"])
    assert facts == task_facts([full_train[identifier] for identifier in KEPT] + subset)
    assert facts_raw == facts_bytes(facts)
    for name, file in FILES.items():
        row = record["files"][name]
        assert (row["sha256"], row["bytes"]) == (
            file_sha256(output / file),
            (output / file).stat().st_size,
        )
    assert record["files"]["rl-tasks"]["tasks"] == 4 and record["files"]["rl-dev"]["tasks"] == 3
    assert record["sources"]["train_tasks"]["sha256"] == file_sha256(files["train_tasks"])
    assert record["dev_subset"] == {"size": 3, "salt": "rl-dev-subset-v1"}


def test_no_hidden_test_reaches_any_pod_file(tmp_path: Path) -> None:
    output = export(tmp_path, sources(tmp_path))
    assert_no_hidden_test_in(output)  # 6b's scan: a marker unique to every hidden test.
    # The scan is live: the full train file the Mac keeps holds the markers.
    with pytest.raises(AssertionError):
        assert_no_hidden_test_in(tmp_path / "pool")
    for name, kind in (
        ("rl-tasks", "rl-tasks"),
        ("rl-dev", "rl-dev-tasks"),
        ("rl-facts", "rl-facts"),
    ):
        require_export(name, kind, output / FILES[name])


def test_the_export_is_deterministic(tmp_path: Path) -> None:
    files = sources(tmp_path)
    first, second = export(tmp_path, files, "first"), export(tmp_path, files, "second")
    for file in (*FILES.values(), RECORD):
        if file == RECORD:
            one, two = (
                json.loads((first / file).read_text()),
                json.loads((second / file).read_text()),
            )
            assert one["files"] == two["files"] and one["sources"] == two["sources"]
        else:
            assert (first / file).read_bytes() == (second / file).read_bytes(), file


def test_the_export_refuses_sealed_files_missing_or_unfit_tasks_and_an_existing_output(
    tmp_path: Path,
) -> None:
    files = sources(tmp_path)
    sealed = tmp_path / "pool" / "test.jsonl"
    save_tasks(sealed, with_hidden(DEV_TASKS))
    link = tmp_path / "dev-link.jsonl"
    link.symlink_to(sealed)
    for change in ({"dev_tasks": sealed}, {"dev_tasks": link}, {"train_tasks": sealed}):
        with pytest.raises(ValueError, match="sealed"):
            write_rl_pod_export(**{**files, **change}, output=tmp_path / "x", dev_size=3)
    missing = sources(tmp_path / "missing", kept=["t1", "nope"])
    with pytest.raises(ValueError, match="not in"):
        export(tmp_path / "missing", missing)
    # A kept task without a reference, or whose hidden tests the channel cannot parse.
    unfit = tmp_path / "unfit"
    files = sources(unfit)
    tasks = with_hidden(TRAIN)
    resave(files["train_tasks"], [replace(tasks[0], solution=None), *tasks[1:]])
    with pytest.raises(ValueError, match="reference"):
        write_rl_pod_export(**files, output=unfit / "a", dev_size=3)
    resave(
        files["train_tasks"],
        [replace(tasks[0], hidden_tests="y = f(1)\nassert y == 2"), *tasks[1:]],
    )
    with pytest.raises(ValueError, match="parse"):
        write_rl_pod_export(**files, output=unfit / "b", dev_size=3)
    resave(files["train_tasks"], [withheld_task(task) for task in tasks])
    with pytest.raises(ValueError, match="withheld"):
        write_rl_pod_export(**files, output=unfit / "c", dev_size=3)
    with pytest.raises(FileExistsError):
        write_rl_pod_export(**sources(tmp_path / "again"), output=tmp_path / "again", dev_size=3)


def test_the_binding_refuses_full_or_mixed_files_as_run_two_inputs(tmp_path: Path) -> None:
    output = export(tmp_path, sources(tmp_path))
    with pytest.raises(ValueError, match="hidden tests"):
        require_export("rl-tasks", "rl-tasks", tmp_path / "pool" / "train.jsonl")
    held_dev = [withheld_task(task) for task in with_hidden(DEV_TASKS)]
    referenced = tmp_path / "referenced.jsonl"
    save_tasks(referenced, [replace(held_dev[0], solution="pass"), *held_dev[1:]])
    with pytest.raises(ValueError, match="with a reference"):
        require_export("rl-dev", "rl-dev-tasks", referenced)
    with pytest.raises(ValueError, match="not train-split"):
        require_export("rl-tasks", "rl-tasks", output / FILES["rl-dev"])
    broken = tmp_path / "facts.json"
    broken.write_text('{"format": "rl-task-facts-v1", "tasks": {}}')
    with pytest.raises(ValueError, match="not a facts file"):
        require_export("rl-facts", "rl-facts", broken)
    assert require_export("rl-tasks", "rl-tasks", output / FILES["rl-tasks"]) == 4


def test_the_command_writes_the_export_and_prints_its_hashes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI uses the protocol's 200-task subset, so it is given a dev split of 210 tasks."""
    files = sources(tmp_path)
    dev = [replace(with_hidden(DEV_TASKS)[0], id=f"pool-dev-{index}") for index in range(210)]
    resave(files["dev_tasks"], dev)
    argv = [
        "rl-pod-export",
        "--train-tasks",
        str(files["train_tasks"]),
        "--kept",
        str(files["kept"]),
        "--dev-tasks",
        str(files["dev_tasks"]),
        "--output",
        str(tmp_path / "cli"),
    ]
    assert main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == set(FILES)
    for name, row in printed.items():
        assert row["name"] == FILES[name]
        assert row["sha256"] == file_sha256(tmp_path / "cli" / row["name"])
    assert len(list(load_tasks(tmp_path / "cli" / FILES["rl-dev"]))) == 200
    assert HIDDEN  # The markers the scans above look for.
