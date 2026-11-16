"""Small pool splits, synthetic instance directories and builds for the dev-family check.
Everything is text written to disk; nothing here runs the code it writes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scratchcoder.agent.cases import REGISTRY_NAME, build_registry, registry_sha256
from scratchcoder.eval.tasks import Task


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def pool_task(identifier: str, prompt: str, solution: str | None, split: str) -> Task:
    return Task(
        id=identifier,
        prompt=prompt,
        public_tests="assert True",
        hidden_tests="assert 1 == 1",
        source="example/source",
        license="CC-BY-4.0",
        written_by="model",
        split=split,
        solution=solution,
    )


def write_pool(path: Path, tasks: Sequence[Task]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(task)) + "\n" for task in tasks), encoding="utf-8")
    return path


def write_instance(
    root: Path,
    identifier: str,
    *,
    template: str,
    task: str,
    reference: str,
    original: str | None = None,
    family: str = "fam",
    module: str = "toolbox.py",
    others: Mapping[str, str] | None = None,
) -> None:
    """One instance directory: ``module`` holds the target alone (original in the
    repository, ``reference`` in the fix); ``others`` are further repository modules."""
    original = original or reference.replace("return", "return 1 +", 1)
    directory = root / identifier
    (directory / "repo").mkdir(parents=True)
    (directory / "reference").mkdir()
    (directory / "hidden").mkdir()
    (directory / "repo" / module).write_text(original, encoding="utf-8")
    (directory / "repo" / f"test_{module}").write_text("def test_x():\n    pass\n")
    for name, text in (others or {}).items():
        (directory / "repo" / name).write_text(text, encoding="utf-8")
    (directory / "reference" / module).write_text(reference, encoding="utf-8")
    (directory / "hidden" / "correctness.py").write_text("pass\n")
    (directory / "hidden" / "preservation.py").write_text("pass\n")
    case = {
        "id": identifier,
        "split": "dev",
        "category": "debugging",
        "workflow_type": "W1",
        "task": task,
        "repo": "repo",
        "reference": {module: f"reference/{module}"},
    }
    synth = {
        "id": identifier,
        "template_id": template,
        "family": family,
        "workflow_type": "W1",
        "mutants": [],
        "reference_target": reference,
        "original_target": original,
    }
    (directory / "case.json").write_text(json.dumps(case), encoding="utf-8")
    (directory / "synth.json").write_text(json.dumps(synth), encoding="utf-8")


def register(root: Path) -> Path:
    (root / REGISTRY_NAME).write_text(json.dumps(build_registry(root)), encoding="utf-8")
    return root


def case_sha(root: Path, identifier: str) -> str:
    """The SHA-256 of an instance's ``case.json`` (what a row's ``spec_sha256`` names)."""
    return hashlib.sha256((root / identifier / "case.json").read_bytes()).hexdigest()


def write_build(path: Path, outcomes: Mapping[str, bool], instances: Path) -> Path:
    """A build of the registered set ``instances``: its record names that registry."""
    path.mkdir(parents=True, exist_ok=True)
    record = {"instances_registry_sha256": registry_sha256(instances)}
    (path / "record.json").write_text(json.dumps(record), encoding="utf-8")
    (path / "instances.jsonl").write_text(
        "".join(
            json.dumps({"instance": name, "accepted": accepted}) + "\n"
            for name, accepted in outcomes.items()
        ),
        encoding="utf-8",
    )
    return path


def write_specs(path: Path, specs: Mapping[str, str]) -> Path:
    document = {"templates": [{"template_id": key, "spec": text} for key, text in specs.items()]}
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def meta_line(
    instance: str,
    template: str,
    kind: str,
    user_text: str,
    reply: str,
    family: str = "fam",
    *,
    spec_sha256: str,
) -> dict[str, Any]:
    return {
        "id": f"agent-sft1/{instance}/{kind}",
        "kind": "edit" if kind == "edit" else "repair",
        "instance": instance,
        "template_id": template,
        "family": family,
        "workflow_type": "W1",
        "mutation_kind": None,
        "prompt_sha256": f"{instance}/{kind}",
        "spec_sha256": spec_sha256,
        "user_text": user_text,
        "reply": reply,
        "loss_tokens": 8,
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def dev_family_argv(
    dev: Path,
    train: Path,
    judged: Sequence[tuple[Path, Path]],
    specs: Sequence[Path],
    meta: Path,
    decoded: Path,
    out: Path,
    bridges: Sequence[Path] = (),
) -> list[str]:
    argv = ["dev-family", "--dev", str(dev), "--train", str(train)]
    for instances, build in judged:
        argv += ["--judged", str(instances), str(build)]
    if bridges:
        argv += ["--bridge-instances", *map(str, bridges)]
    argv += ["--specs", *map(str, specs), "--rows-meta", str(meta), "--decoded", str(decoded)]
    return [*argv, "--out", str(out)]
