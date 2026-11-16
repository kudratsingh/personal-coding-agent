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
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scratchcoder.agent import workspace

CASE_SCHEMA = "agent-case-v1"
REGISTRY_SCHEMA = "agent-case-registry-v1"
REGISTRY_NAME = "registry.json"
MAX_FILE_BYTES = 16384
MAX_CASE_ID = 32
CASE_ID = re.compile(r"(?!.*--)[A-Za-z0-9](?:[A-Za-z0-9_-]{0,30}[A-Za-z0-9_])?\Z")
"""A case ID is the identity of a case in every output: file names
(``{case}--{policy}--{condition}--{decoding}``), scorecard cells and comparisons. It is 1 to
:data:`MAX_CASE_ID` characters of a safe file-name alphabet, starts with a letter or digit,
does not end with a hyphen and never contains the ``--`` separator, so a file name splits back
into one case ID. IDs that are equal ignoring case are refused together
(:func:`check_case_ids`): they would share a file on a case-insensitive file system."""
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


def check_case_id(case_id: object) -> str:
    """``case_id`` if it is a well-formed case ID (:data:`CASE_ID`); raises :class:`CaseError`."""
    if not isinstance(case_id, str) or not CASE_ID.match(case_id):
        raise CaseError(
            f"case id {case_id!r} must be 1-{MAX_CASE_ID} letters, digits, '_' or '-', start "
            "with a letter or digit, not end with '-' and not contain '--'"
        )
    return case_id


def check_case_ids(case_ids: Iterable[object], *, what: str = "case IDs") -> list[str]:
    """Every ID well formed and distinct, also ignoring case; raises :class:`CaseError`.

    Results are keyed by case ID, so two distinct cases under one ID (or under IDs that differ
    only in case) would merge in a scorecard or overwrite each other's files."""
    seen: dict[str, str] = {}
    found = []
    for case_id in case_ids:
        checked = check_case_id(case_id)
        folded = checked.casefold()
        if folded in seen:
            other = seen[folded]
            kind = "repeated" if other == checked else f"equal ignoring case to {other!r}"
            raise CaseError(f"{what}: {checked!r} is {kind}")
        seen[folded] = checked
        found.append(checked)
    return found


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
    check_case_ids((case["id"] for case in cases), what="registry")
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
    # Every registered ID, not only the requested ones: a run's outputs must never be able to
    # mix two cases, whichever subset it loads.
    check_case_ids((entry.get("id") for entry in registry["cases"]), what="registry")
    directories = [str(entry.get("directory", "")).casefold() for entry in registry["cases"]]
    if len(set(directories)) != len(directories):
        raise CaseError("registry: two cases share a directory (ignoring case)")
    cases = []
    for entry in registry["cases"]:
        if only is not None and entry["id"] not in only:
            continue
        if entry.get("split") != "dev":
            raise CaseError(f"{entry['id']}: only dev cases are loaded")
        directory = _inside(root, entry["directory"], what="case directory")
        if directory.parent != root.resolve():
            raise CaseError(f"{entry['id']}: a case directory must be a direct child of the root")
        present = {str(path.relative_to(directory)): path for path in _case_files(directory)}
        if set(present) != set(entry["files"]):
            raise CaseError(f"{entry['id']}: files differ from the registry")
        for name, digest in entry["files"].items():
            if _sha256(present[name].read_bytes()) != digest:
                raise CaseError(f"{entry['id']}: {name} changed since registration")
        case = load_case(directory, registered=entry["files"])
        if case.id != entry["id"]:
            raise CaseError(f"{entry['id']}: case.json names {case.id}")
        cases.append(case)
    if only is not None and {case.id for case in cases} != only:
        raise CaseError("unknown case IDs requested")
    return cases


def _inside(base: Path, relative: object, *, what: str) -> Path:
    """``base / relative`` resolved, refused unless it stays inside ``base`` and outside any
    held-out location."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise CaseError(f"{what} must be a relative path")
    root = base.resolve()
    resolved = (root / relative).resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise CaseError(f"{what} escapes its directory: {relative}")
    if _forbidden(resolved):
        raise CaseError("refusing a held-out location")
    return resolved


def load_case(directory: Path, *, registered: Mapping[str, str] | None = None) -> Case:
    """Load one case. With ``registered`` (relative path -> SHA-256, from the registry), every
    file read must be listed there and its bytes, as read, must match the listed hash."""
    root = directory.resolve()
    if _forbidden(root):
        raise CaseError("refusing a held-out location")

    def read(relative: str, what: str) -> bytes:
        path = _inside(root, relative, what=what)
        data = path.read_bytes()
        name = str(path.relative_to(root))
        if registered is not None and (name not in registered or _sha256(data) != registered[name]):
            raise CaseError(f"{root.name}: {name} is not the registered file")
        if len(data) > MAX_FILE_BYTES:
            raise CaseError(f"{name} is larger than {MAX_FILE_BYTES} bytes")
        return data

    def text(relative: str, what: str) -> str:
        decoded = read(relative, what).decode("utf-8")
        if "\r" in decoded:
            raise CaseError(f"{root.name}: {relative} has a carriage return")
        return decoded

    raw = read("case.json", "case.json")
    spec = json.loads(raw)
    if spec.get("schema") != CASE_SCHEMA or spec.get("split") != "dev":
        raise CaseError(f"{root.name}: not an {CASE_SCHEMA} dev case")
    try:
        case_id = check_case_id(spec.get("id"))
    except CaseError as error:
        raise CaseError(f"{root.name}: {error}") from None
    repo = _inside(root, spec["repo"], what="repo")
    files = {
        path.name: text(str(path.relative_to(root)), "repo file")
        for path in sorted(repo.glob("*.py"))
    }
    reference = {name: text(source, "reference file") for name, source in spec["reference"].items()}
    case = Case(
        id=case_id,
        title=spec["title"],
        category=spec["category"],
        workflow_type=spec["workflow_type"],
        structure=spec["structure"],
        task=spec["task"],
        files=files,
        allowed_files=tuple(spec["allowed_files"]),
        public_tests=tuple(spec["public_tests"]),
        public_fails_before_fix=bool(spec["public_fails_before_fix"]),
        correctness=text(spec["hidden"]["correctness"], "hidden check file"),
        preservation=text(spec["hidden"]["preservation"], "hidden check file"),
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
    if not case.public_fails_before_fix:
        raise CaseError(
            f"{case.id}: harness v0 needs public tests that fail before the fix (the loop "
            "stops when they pass, so the policy would never be asked)"
        )
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
