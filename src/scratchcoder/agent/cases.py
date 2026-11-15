"""Development cases for the agent loop: loading, registry hashes and static validation.

A case directory holds ``case.json`` (schema ``agent-case-v1``), the repository under
``repo/`` (flat modules; tests are ``test_*.py``), the hidden checks (correctness and
preservation, plain assertion scripts that import the repository's modules), and the
reference fix (the changed files in full). A registry (``registry.json`` beside the case
directories) lists every file of every case with its SHA-256; loading refuses any file whose
bytes differ, any unlisted file and any held-out location.

Static validation runs nothing: it parses every file and checks that the reference fix
differs from the original repository only inside the declared target functions, by applying
each reference function through the same editor the agent uses
(:func:`scratchcoder.agent.workspace.apply_function`) and requiring the result to equal the
reference text byte for byte.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scratchcoder.agent import workspace

CASE_SCHEMA = "agent-case-v1"
REGISTRY_SCHEMA = "agent-case-registry-v1"
REGISTRY_NAME = "registry.json"
MAX_FILE_BYTES = 16384
FORBIDDEN_PARTS = ("heldout", "heldout-results")
"""Held-out cases are sealed; no development tool reads them."""


class CaseError(ValueError):
    """A case or registry is malformed, changed since registration, or sealed."""


@dataclass(frozen=True)
class Budgets:
    max_edit_turns: int = 3
    max_feedback_runs: int = 6
    max_generated_tokens: int = 2000
    max_wall_seconds: float = 600.0

    def __post_init__(self) -> None:
        if not (
            1 <= self.max_edit_turns <= 10
            and 2 <= self.max_feedback_runs <= 20
            and 1 <= self.max_generated_tokens <= 20000
            and 1 <= self.max_wall_seconds <= 3600
        ):
            raise CaseError("budgets out of range")


@dataclass(frozen=True)
class Case:
    id: str
    title: str
    category: str
    workflow_type: str
    structure: str
    task: str
    files: Mapping[str, str]
    """The original repository: file name -> source text, tests included."""
    allowed_files: tuple[str, ...]
    public_tests: tuple[str, ...]
    public_fails_before_fix: bool
    correctness: str
    """Hidden correctness checks (source). Never shown to a policy, never logged."""
    preservation: str
    """Hidden preservation checks (source). Never shown to a policy, never logged."""
    reference: Mapping[str, str]
    """The reference fix: changed file name -> full fixed text."""
    targets: tuple[str, ...]
    """``path::function`` keys the reference fix changes."""
    budgets: Budgets
    spec_sha256: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sources(self) -> list[str]:
        return sorted(path for path in self.files if not workspace.is_test_file(path))

    def reference_files(self) -> dict[str, str]:
        return {**self.files, **self.reference}

    def reference_function(self, key: str) -> str:
        path, name = key.split("::")
        text = self.reference[path]
        return workspace.function_source(text, workspace.functions(text, path)[name])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _forbidden(path: Path) -> bool:
    return any(part.lower() in FORBIDDEN_PARTS for part in path.resolve().parts)


def _case_files(directory: Path) -> list[Path]:
    found = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise CaseError(f"symlink in a case directory: {path}")
        if path.is_file() and "__pycache__" not in path.parts and path.name != ".DS_Store":
            found.append(path)
    return found


def build_registry(root: Path) -> dict[str, Any]:
    """The registry of every case directory under ``root`` (each holds ``case.json``)."""
    if _forbidden(root):
        raise CaseError("refusing a held-out location")
    cases = []
    for directory in sorted(path for path in root.iterdir() if (path / "case.json").is_file()):
        spec = json.loads((directory / "case.json").read_text(encoding="utf-8"))
        files = {
            str(path.relative_to(directory)): _sha256(path.read_bytes())
            for path in _case_files(directory)
        }
        cases.append(
            {
                "id": spec["id"],
                "directory": directory.name,
                "split": spec["split"],
                "category": spec["category"],
                "workflow_type": spec["workflow_type"],
                "files": files,
            }
        )
    if not cases:
        raise CaseError("no case directories")
    return {"schema": REGISTRY_SCHEMA, "cases": cases}


def registry_sha256(root: Path) -> str:
    return _sha256((root / REGISTRY_NAME).read_bytes())


def load_cases(root: Path, *, only: set[str] | None = None) -> list[Case]:
    """Every registered case under ``root`` (or those in ``only``), hash-checked and
    statically validated, in registry order."""
    if _forbidden(root):
        raise CaseError("refusing a held-out location")
    registry = json.loads((root / REGISTRY_NAME).read_text(encoding="utf-8"))
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise CaseError("unknown registry schema")
    cases = []
    for entry in registry["cases"]:
        if only is not None and entry["id"] not in only:
            continue
        if entry.get("split") != "dev":
            raise CaseError(f"{entry['id']}: only dev cases are loaded")
        directory = root / entry["directory"]
        present = {str(path.relative_to(directory)): path for path in _case_files(directory)}
        if set(present) != set(entry["files"]):
            raise CaseError(f"{entry['id']}: files differ from the registry")
        for name, digest in entry["files"].items():
            if _sha256(present[name].read_bytes()) != digest:
                raise CaseError(f"{entry['id']}: {name} changed since registration")
        case = load_case(directory)
        if case.id != entry["id"]:
            raise CaseError(f"{entry['id']}: case.json names {case.id}")
        cases.append(case)
    if only is not None and {case.id for case in cases} != only:
        raise CaseError("unknown case IDs requested")
    return cases


def _text(path: Path) -> str:
    data = path.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise CaseError(f"{path.name} is larger than {MAX_FILE_BYTES} bytes")
    return data.decode("utf-8")


def load_case(directory: Path) -> Case:
    raw = (directory / "case.json").read_bytes()
    spec = json.loads(raw)
    if spec.get("schema") != CASE_SCHEMA or spec.get("split") != "dev":
        raise CaseError(f"{directory.name}: not an {CASE_SCHEMA} dev case")
    repo = directory / spec["repo"]
    files = {path.name: _text(path) for path in sorted(repo.glob("*.py"))}
    reference = {name: _text(directory / source) for name, source in spec["reference"].items()}
    case = Case(
        id=spec["id"],
        title=spec["title"],
        category=spec["category"],
        workflow_type=spec["workflow_type"],
        structure=spec["structure"],
        task=spec["task"],
        files=files,
        allowed_files=tuple(spec["allowed_files"]),
        public_tests=tuple(spec["public_tests"]),
        public_fails_before_fix=bool(spec["public_fails_before_fix"]),
        correctness=_text(directory / spec["hidden"]["correctness"]),
        preservation=_text(directory / spec["hidden"]["preservation"]),
        reference=reference,
        targets=tuple(spec["targets"]),
        budgets=Budgets(**spec["budgets"]),
        spec_sha256=_sha256(raw),
        provenance=spec.get("provenance", {}),
    )
    validate_static(case)
    return case


def validate_static(case: Case) -> None:
    """Checks that need no execution; raises :class:`CaseError` on the first problem."""
    try:
        for path in case.files:
            workspace.check_file_name(path)
        for path, text in [*case.files.items(), *case.reference.items()]:
            workspace.parse(text, path)
        ast.parse(case.correctness)
        ast.parse(case.preservation)
    except (SyntaxError, ValueError) as error:
        raise CaseError(f"{case.id}: {error}") from None
    if not case.public_tests or any(
        path not in case.files or not workspace.is_test_file(path) for path in case.public_tests
    ):
        raise CaseError(f"{case.id}: public tests must be test_*.py files of the repository")
    if not case.allowed_files or any(
        path not in case.files or workspace.is_test_file(path) for path in case.allowed_files
    ):
        raise CaseError(f"{case.id}: allowed files must be non-test files of the repository")
    if not case.reference or any(path not in case.allowed_files for path in case.reference):
        raise CaseError(f"{case.id}: the reference may change only allowed files")
    edited = dict(case.files)
    for key in case.targets:
        path, _, name = key.partition("::")
        if path not in case.reference or name not in workspace.functions(case.files[path], path):
            raise CaseError(f"{case.id}: target {key} is not a function of a changed file")
        edit = workspace.Edit(path, name, case.reference_function(key))
        try:
            edited = workspace.apply_function(edited, edit, case.allowed_files)
        except workspace.EditRefused as error:
            raise CaseError(f"{case.id}: reference edit of {key} refused: {error}") from None
    if edited != case.reference_files():
        raise CaseError(f"{case.id}: the reference changes code outside its target functions")
    if edited == dict(case.files):
        raise CaseError(f"{case.id}: the reference fix changes nothing")
