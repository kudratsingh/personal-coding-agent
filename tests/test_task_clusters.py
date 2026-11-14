"""The static candidate manifest: synthetic files, plus a local-only check on the real ones."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data.clusters import find_edges, validate_manifest
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.tasks import Task
from tools import task_clusters

ROOT = Path(__file__).resolve().parents[1]
REAL = {
    "train": ROOT / "runs/task-pool-20261025/pool/train.jsonl",
    "dev": ROOT / "runs/task-pool-20261025/pool/dev.jsonl",
    "sft": ROOT / "runs/sft-data-20261027/train.jsonl",
    "rl_candidates": ROOT / "runs/sft-data-20261027/rl_candidates.jsonl",
    "screening": ROOT / "runs/posttrain-shared-1b/screening.json",
}


def task(identifier: str, split: str, prompt: str, solution: str) -> Task:
    return Task(
        id=identifier,
        prompt=prompt,
        public_tests="assert True",
        hidden_tests="assert True\nassert True",
        source="nvidia/OpenCodeInstruct",
        license="CC-BY-4.0",
        written_by="model",
        split=split,
        solution=solution,
    )


SQUARE = "def square(x):\n    return x * x\n"
CUBE = "def cube(x):\n    return x * x * x\n"
TWICE = "def twice(x):\n    return x + x\n"


def write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def synthetic(tmp_path: Path) -> dict[str, Path]:
    train = [
        task("t/square", "train", "Return the square of an integer value.", SQUARE),
        task("t/cube", "train", "Raise x to the third power and return it.", CUBE),
        task("t/twice", "train", "Double the argument.", TWICE),
        task("t/other", "train", "Reverse a string.", "def rev(s):\n    return s[::-1]\n"),
    ]
    dev = [
        task("d/square", "dev", "Compute x multiplied by itself.", SQUARE),
        task("d/cube", "dev", "Give the cube of a number.", CUBE),
        task("d/twice", "dev", "Sum a number with itself.", TWICE.replace("twice", "double")),
        task("d/alone", "dev", "Count vowels.", "def vowels(s):\n    return 0\n"),
    ]
    screening = {
        "format": "rl-screening-v1",
        "eligible": ["t/cube", "t/other"],
        "rule": {"floor": 320},
        "tasks": [
            {"id": "t/cube", "reason": None},
            {"id": "t/twice", "reason": "equal_rewards"},
            {"id": "t/other", "reason": None},
        ],
    }
    paths = {
        "train": write(tmp_path / "train.jsonl", [asdict(t) for t in train]),
        "dev": write(tmp_path / "dev.jsonl", [asdict(t) for t in dev]),
        "sft": write(tmp_path / "sft.jsonl", [{"id": "t/square", "ids": [1]}, {"id": "x"}]),
        "rl_candidates": write(
            tmp_path / "rl.jsonl", [{"id": i} for i in ("t/cube", "t/twice", "t/other")]
        ),
        "screening": tmp_path / "screening.json",
    }
    paths["screening"].write_text(json.dumps(screening))
    return paths


def arguments(paths: dict[str, Path], output: Path, root: Path) -> list[str]:
    return [
        "candidates",
        "--train",
        str(paths["train"]),
        "--dev",
        str(paths["dev"]),
        "--sft",
        str(paths["sft"]),
        "--rl-candidates",
        str(paths["rl_candidates"]),
        "--screening",
        str(paths["screening"]),
        "--output",
        str(output),
        "--root",
        str(root),
    ]


@pytest.mark.parametrize(
    "name", ["test.jsonl", "holdout.jsonl", "rl_only.jsonl", "TEST.jsonl", "test-v2.jsonl"]
)
def test_sealed_and_holdout_files_are_refused_by_name(name: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="never read"):
        task_clusters.refuse_sealed(tmp_path / name)
    assert task_clusters.refuse_sealed(tmp_path / "dev.jsonl") == tmp_path / "dev.jsonl"
    assert task_clusters.refuse_sealed(tmp_path / "testing-notes.jsonl")


def test_candidate_manifest_is_pending_traceable_and_changes_no_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = synthetic(tmp_path)
    before = {name: file_sha256(path) for name, path in paths.items()}
    output = tmp_path / "clusters"
    assert task_clusters.main(arguments(paths, output, tmp_path)) == 0
    assert {name: file_sha256(path) for name, path in paths.items()} == before
    manifest = json.loads((output / "manifest.json").read_text())
    validate_manifest(manifest)
    assert manifest["kind"] == "candidate"
    statuses = {c["adjudication"]["status"] for c in manifest["clusters"]}
    assert statuses <= {"pending", "unclear_quarantined"}
    inputs = manifest["provenance"]["inputs"]
    assert inputs["pool_dev"] == {"path": "dev.jsonl", "sha256": before["dev"]}
    assert manifest["provenance"]["tool"]["rules_version"] == "task-clusters-v1"
    primary = manifest["summary"]["views"]["primary"]["direct"]
    assert primary["dev_tasks_matching_sft"] == 1
    assert primary["dev_tasks_matching_pool_train"] == 2
    assert primary["distinct_rl_candidates_matched"] == 1
    expanded = manifest["summary"]["views"]["expanded"]["direct"]
    # The renamed "double" reference joins only through the canonicalized AST.
    assert expanded["dev_tasks_matching_pool_train"] == 3
    quarantine = manifest["screening_quarantine"]
    assert [c["id"] for c in quarantine["candidates"]] == ["t/cube"]
    assert quarantine["candidates"][0]["eligible"] is True
    assert quarantine["candidates"][0]["screening_reason"] == "kept"
    assert quarantine["recount"] == {
        "eligible_before": 2,
        "quarantined": 1,
        "quarantined_eligible": 1,
        "eligible_after": 1,
        "floor": 320,
        "floor_met": False,
        "note": "before any further filtering; run 2 remains blocked",
    }
    extra = quarantine["additional_expanded_candidates"]
    assert [c["id"] for c in extra["candidates"]] == ["t/twice"]
    assert extra["candidates"][0]["status"] == "listed_for_adjudication_not_quarantined"
    exposed = [c for c in manifest["clusters"] if "training_exposed_dev" in c["categories"]]
    assert [m["id"] for m in exposed[0]["members"]] == ["t/square", "d/square"]
    assert manifest["existing_weights"]["pending_clusters"] == [exposed[0]["cluster_id"]]
    record = json.loads((output / "record.json").read_text())
    assert record["manifest_sha256"] == file_sha256(output / "manifest.json")
    assert record["paid_runs"] == 0
    with pytest.raises(SystemExit, match="never overwritten"):
        task_clusters.main(arguments(paths, output, tmp_path))
    sealed = {**paths, "dev": paths["dev"].rename(tmp_path / "test.jsonl")}
    with pytest.raises(ValueError, match="never read"):
        task_clusters.main(arguments(sealed, tmp_path / "other", tmp_path))
    capsys.readouterr()


def test_adjudication_is_logged_and_checked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = synthetic(tmp_path)
    output = tmp_path / "clusters"
    task_clusters.main(arguments(paths, output, tmp_path))
    manifest_path = output / "manifest.json"
    cluster = json.loads(manifest_path.read_text())["clusters"][0]["cluster_id"]
    base = ["adjudicate", "--manifest", str(manifest_path), "--cluster", cluster]
    with pytest.raises(ValueError, match="behaviour result"):
        task_clusters.main(
            [*base, "--status", "confirmed_equivalent", "--reviewer", "r", "--rationale", "x"]
        )
    assert (
        task_clusters.main(
            [
                *base,
                "--status",
                "unclear_quarantined",
                "--reviewer",
                "owner",
                "--rationale",
                "prompts differ in edge-case handling",
                "--uncertainty",
                "negative inputs unspecified in one prompt",
            ]
        )
        == 0
    )
    log = (output / "adjudication-log.jsonl").read_text().splitlines()
    assert len(log) == 1 and json.loads(log[0])["status"] == "unclear_quarantined"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["kind"] == "partially_adjudicated"
    assert task_clusters.main(["validate", "--manifest", str(manifest_path)]) == 0
    capsys.readouterr()


@pytest.mark.skipif(
    not all(path.is_file() for path in REAL.values()),
    reason="local-only: needs the owner's ignored pool, SFT and screening files",
)
def test_local_files_reproduce_the_audit_overlap_counts() -> None:
    items, info = task_clusters.load_items(
        REAL["train"], REAL["dev"], REAL["sft"], REAL["rl_candidates"]
    )
    assert info["sft_rows_in_pool_train"] == 2871 and info["rl_candidates"] == 1000
    edges = find_edges(items)
    screening = json.loads(REAL["screening"].read_text())
    view = task_clusters.view_counts(items, edges, task_clusters.PRIMARY, screening["eligible"])
    assert view["direct"] == {
        "dev_tasks_matching_sft": 17,
        "dev_tasks_matching_pool_train": 38,
        "dev_tasks_matching_rl_candidates": 16,
        "dev_tasks_matching_rl_candidates_and_sft": 2,
        "dev_train_edges": 57,
        "distinct_train_partners": 45,
        "dev_sft_edges": 21,
        "distinct_sft_partners": 17,
        "distinct_rl_candidates_matched": 9,
        "eligible_rl_candidates_matched": 2,
    }
    assert view["family"]["dev_internal_groups"] == 55
    assert view["family"]["dev_internal_group_members"] == 140
    quarantine = task_clusters.screening_quarantine(items, edges, screening, {})
    assert quarantine["recount"]["eligible_before"] == 161
    assert quarantine["recount"]["eligible_after"] == 159
