"""The run-1 input binding: build on the Mac, stage at the go, fetch and verify on the Pod.

The smoke's fixtures stand in for the inputs and an in-memory bucket for R2."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools import posttrain_binding
from tools.posttrain_binding import (
    binding_sha256,
    build_binding,
    check_fetch_receipt,
    fetch,
    load_binding,
    stage,
    verify,
)
from tools.posttrain_smoke import EXISTING, Fixtures, fixtures, smoke_specification
from tools.rehearsal_run import MemoryBucket

COMMIT = "1" * 40


@pytest.fixture
def found(tmp_path: Path) -> Fixtures:
    return fixtures(tmp_path / "inputs")


@pytest.fixture(autouse=True)
def clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(posttrain_binding, "git_state", lambda: (COMMIT, False))


def specification(found: Fixtures, tmp_path: Path, **pins: dict[str, Any]) -> Path:
    raw = smoke_specification(found)
    for name, changes in pins.items():
        raw["inputs"][name].update(changes)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(raw, indent=2) + "\n")
    return path


def staged_inputs(found: Fixtures) -> dict[str, Path]:
    return {name: path for name, path in found.inputs.items() if name not in EXISTING}


def built(found: Fixtures, tmp_path: Path) -> tuple[dict[str, Any], Path]:
    output = tmp_path / "binding.json"
    binding = build_binding(specification(found, tmp_path), staged_inputs(found), output, run="1")
    return binding, output


def r2_with_parents(found: Fixtures) -> MemoryBucket:
    bucket = MemoryBucket()
    for name, key in EXISTING.items():
        data = found.inputs[name].read_bytes()
        bucket.objects[key] = data
        bucket.digests[key] = hashlib.sha256(data).hexdigest()
    return bucket


def test_the_binding_pins_every_input_by_key_size_and_sha256(
    found: Fixtures, tmp_path: Path
) -> None:
    binding, output = built(found, tmp_path)
    loaded = load_binding(output, expected_sha256=binding_sha256(binding))
    assert loaded == binding and set(loaded["inputs"]) == set(found.inputs)
    for name, entry in loaded["inputs"].items():
        path = found.inputs[name]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["size_bytes"] == path.stat().st_size
        if name in EXISTING:
            assert (entry["origin"], entry["key"]) == ("existing", EXISTING[name])
        else:
            assert entry["origin"] == "staged"
            assert entry["key"] == f"posttrain-inputs/{entry['sha256']}/{path.name}"
    assert loaded["binder"] == {"git_commit": COMMIT, "dirty": False}
    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_binding(output, expected_sha256="0" * 64)
    with pytest.raises(FileExistsError):
        build_binding(specification(found, tmp_path), staged_inputs(found), output, run="1")


def test_the_binding_refuses_inputs_that_differ_from_the_specification(
    found: Fixtures, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = specification(found, tmp_path, **{"sft-train": {"sha256": "0" * 64}})
    with pytest.raises(ValueError, match="sha256 differs"):
        build_binding(other, staged_inputs(found), tmp_path / "a.json", run="1")
    missing = {k: v for k, v in staged_inputs(found).items() if k != "pool-test"}
    with pytest.raises(ValueError, match="pool-test is staged"):
        build_binding(specification(found, tmp_path), missing, tmp_path / "b.json", run="1")
    copies = {**staged_inputs(found), "b-mixed-final": found.inputs["a-stage1-final"]}
    with pytest.raises(ValueError, match="Mac copy differs"):
        build_binding(specification(found, tmp_path), copies, tmp_path / "c.json", run="1")
    monkeypatch.setattr(posttrain_binding, "git_state", lambda: (COMMIT, True))
    with pytest.raises(ValueError, match="clean tree"):
        build_binding(
            specification(found, tmp_path), staged_inputs(found), tmp_path / "d.json", run="1"
        )


def test_staging_uploads_the_staged_inputs_and_heads_the_existing_ones(
    found: Fixtures, tmp_path: Path
) -> None:
    binding, _ = built(found, tmp_path)
    bucket = r2_with_parents(found)
    record = stage(binding, bucket.store(), cap_usd=5.0, receipt=tmp_path / "stage.json")
    assert {row["key"] for row in record["files"]} == {
        entry["key"] for entry in binding["inputs"].values() if entry["origin"] == "staged"
    }
    assert {row["name"] for row in record["existing_verified"]} == set(EXISTING)
    for entry in binding["inputs"].values():
        assert bucket.digests[entry["key"]] == entry["sha256"]
    bucket.digests[EXISTING["b-mixed-final"]] = "f" * 64
    with pytest.raises(ValueError, match="differs"):
        stage(binding, bucket.store(), cap_usd=5.0, receipt=tmp_path / "again.json")


def test_the_pod_fetches_and_verifies_every_input_pinned(found: Fixtures, tmp_path: Path) -> None:
    binding, _ = built(found, tmp_path)
    bucket = r2_with_parents(found)
    stage(binding, bucket.store(), cap_usd=5.0, receipt=tmp_path / "stage.json")
    root = tmp_path / "pod"
    record = fetch(
        binding, root, bucket.store(), receipt=tmp_path / "fetch.json", progress=lambda _: None
    )
    check_fetch_receipt(json.loads((tmp_path / "fetch.json").read_text()), binding)
    assert record["bytes_transferred"] == sum(e["size_bytes"] for e in binding["inputs"].values())
    verified = verify(binding, root)
    assert set(verified["paths"]) == set(binding["inputs"])
    for name, path in verified["paths"].items():
        assert Path(path).read_bytes() == found.inputs[name].read_bytes()
    # A second fetch keeps the pinned files; a changed one stops it.
    again = fetch(
        binding, root, bucket.store(), receipt=tmp_path / "fetch2.json", progress=lambda _: None
    )
    assert {row["method"] for row in again["inputs"]} == {"present"}
    Path(verified["paths"]["sft-train"]).write_text("changed\n")
    with pytest.raises(ValueError, match="other bytes"):
        fetch(
            binding, root, bucket.store(), receipt=tmp_path / "fetch3.json", progress=lambda _: None
        )
    with pytest.raises(ValueError, match="differs"):
        verify(binding, root)


def test_the_macs_parent_copies_are_a_partial_fetch_its_receipt_names(
    found: Fixtures, tmp_path: Path
) -> None:
    binding, _ = built(found, tmp_path)
    bucket = r2_with_parents(found)
    parents = ["a-stage1-final", "a-stage2-final", "b-mixed-final"]
    fetch(
        binding,
        tmp_path / "mac",
        bucket.store(),
        receipt=tmp_path / "copies.json",
        only=parents,
        progress=lambda _: None,
    )
    record = json.loads((tmp_path / "copies.json").read_text())
    check_fetch_receipt(record, binding, names=parents)
    with pytest.raises(ValueError, match="exactly"):
        check_fetch_receipt(record, binding)
    assert verify(binding, tmp_path / "mac", only=parents)["inputs"].keys() == set(parents)


def test_a_full_task_file_is_never_bound_as_a_pod_input(found: Fixtures, tmp_path: Path) -> None:
    """The test split's export is pinned only at the go, so the spec's pin cannot catch a full
    file there: the binding itself refuses any pool or candidates file with hidden tests."""
    unpinned = specification(found, tmp_path, **{"pool-test": {"size_bytes": None, "sha256": None}})
    full = {**staged_inputs(found), "pool-test": found.full_tasks["test"]}
    output = tmp_path / "binding.json"
    with pytest.raises(ValueError, match="pool-test is not a Pod export") as error:
        build_binding(unpinned, full, output, run="1")
    hidden = json.loads(found.full_tasks["test"].read_text().splitlines()[0])["hidden_tests"]
    assert hidden not in str(error.value)  # The message names task IDs, never tests.
    assert not output.exists()
    # The exported file binds; a full train file as the candidates export does not.
    build_binding(unpinned, staged_inputs(found), tmp_path / "exported.json", run="1")
    pinned_off = specification(
        found, tmp_path, **{"rl-candidates": {"size_bytes": None, "sha256": None}}
    )
    with pytest.raises(ValueError, match="rl-candidates is not a Pod export"):
        build_binding(
            pinned_off,
            {**staged_inputs(found), "rl-candidates": found.full_tasks["train"]},
            tmp_path / "c.json",
            run="1",
        )


def test_an_export_of_another_size_than_planned_is_refused(found: Fixtures, tmp_path: Path) -> None:
    raw = json.loads(specification(found, tmp_path).read_text())
    for plan in raw["stages"]:
        for entry in plan.get("files", []):
            if entry["tasks"] == "pool-dev":
                entry["planned_tasks"] = 2_000
    path = tmp_path / "planned.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="pool-dev holds 3 tasks; the specification plans 2000"):
        build_binding(path, staged_inputs(found), tmp_path / "binding.json", run="1")


def test_the_pod_refuses_a_bound_file_with_hidden_tests(found: Fixtures, tmp_path: Path) -> None:
    binding, _ = built(found, tmp_path)
    root = tmp_path / "pod"
    for name, entry in binding["inputs"].items():
        target = root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(found.inputs[name].read_bytes())
    verify(binding, root)
    # Bytes that match a (mistaken) pin but carry hidden tests are still refused on the Pod.
    full = found.full_tasks["test"].read_bytes()
    target = root / binding["inputs"]["pool-test"]["path"]
    target.write_bytes(full)
    entry = binding["inputs"]["pool-test"]
    entry.update(size_bytes=len(full), sha256=hashlib.sha256(full).hexdigest())
    entry["key"] = posttrain_binding.staged_key(entry["sha256"], target.name)
    with pytest.raises(ValueError, match="not a Pod export"):
        verify(binding, root)


def test_the_runbook_export_check_names_counts_and_refuses_full_files(
    found: Fixtures, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = specification(found, tmp_path)
    argv = ["check-exports", "--specification", str(spec)]
    exported = [f"--input=pool-test={found.inputs['pool-test']}"]
    assert posttrain_binding.main([*argv, *exported]) == 0
    assert json.loads(capsys.readouterr().out) == {"exports": {"pool-test": 3}, "hidden_tests": 0}
    with pytest.raises(ValueError, match="not a Pod export"):
        posttrain_binding.main([*argv, f"--input=pool-test={found.full_tasks['test']}"])
    with pytest.raises(ValueError, match="not a pool or candidates input"):
        posttrain_binding.main([*argv, f"--input=sft-train={found.inputs['sft-train']}"])
