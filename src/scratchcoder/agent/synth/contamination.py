"""``agent-case-contamination``: keep evaluation cases and their families out of training rows.

A row (an agent row, a generated instance, or a replay row) is **rejected** when any of these
holds; the output names rejected row IDs only, never the matching case, signal or count:

1. **Case n-grams.** It shares at least one normalized word 13-gram (lower-cased ``\\w+``
   tokens, :class:`scratchcoder.data.filters.ContaminationIndex`) with any text of any
   registered evaluation case: every registered file (repository, tests, hidden checks,
   reference fix) and the task statement.
2. **Reference AST.** A function it defines has the same syntax tree as a function of an
   evaluation case's reference fix, after removing docstrings and renaming the function, its
   parameters and its local variables in order of appearance.
3. **Defined names.** The Jaccard similarity of its top-level defined names with an
   evaluation case's defined names is at least 0.25.
4. **Family blocklist.** It defines a name, or (for a repository) has a module, on the
   evaluation-family blocklist (``families.json``).
5. **Benchmarks.** It shares a 13-gram with a benchmark prompt or reference solution
   (HumanEval, HumanEval+, MBPP+), counting only 13-grams with at least three word tokens.
6. **Development tasks.** The same with the pool's development split (prompt and solution).
   This is an n-gram check only (:data:`DEV_METHOD`): a row on the same concept as a
   development task in different words is not caught, and the task-family rule used for
   replay rows is not applied here. Every result says so in ``index.dev_method``.

A result records its input file's SHA-256 and kind (``input_sha256``, ``input_kind``) and
the registry SHA-256 of every evaluation-case root it was checked against, so that a later
step can refuse a result that was made for other rows or against fewer cases
(:func:`require_result`).

Evaluation cases are read through :func:`scratchcoder.agent.cases.load_cases` (every file is
checked against its registry hash; held-out locations are refused). Nothing is executed.
"""

from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scratchcoder.agent import cases, workspace
from scratchcoder.agent.synth.instances import Blocklist
from scratchcoder.data.filters import ContaminationIndex

SCHEMA = "agent-case-contamination-v1"
DEV_METHOD = (
    "13-gram overlap (at least three word tokens) with the pool development split's prompts "
    "and solutions; the development task-family rule is not applied to these rows"
)
NGRAM = 13
JACCARD = 0.25
EXTERNAL_MIN_ALPHA = 3


@dataclass(frozen=True)
class RowText:
    id: str
    texts: tuple[str, ...]
    code: tuple[str, ...]
    """Python sources the row holds (parsed for names and syntax trees; never run)."""
    modules: tuple[str, ...] = ()


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            return body[1:]
    return body


def normalized_dump(function: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """The function's tree without docstrings, its own name, parameters and stored local
    names renamed ``v0, v1, ...`` in order of first appearance (other names kept)."""
    node = copy.deepcopy(function)
    for item in ast.walk(node):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            item.body = _strip_docstring(item.body) or [ast.Pass()]
    local: list[str] = [node.name]
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    for extra in (node.args.vararg, node.args.kwarg):
        if extra is not None:
            arguments.append(extra)
    local += [arg.arg for arg in arguments]
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
            if item.id not in local:
                local.append(item.id)
    mapping = {name: f"v{index}" for index, name in enumerate(dict.fromkeys(local))}
    node.name = mapping[node.name]
    for arg in arguments:
        arg.arg = mapping[arg.arg]
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and item.id in mapping:
            item.id = mapping[item.id]
        if isinstance(item, ast.arg) and item.arg in mapping:
            item.arg = mapping[item.arg]
    return ast.dump(node, annotate_fields=False)


def _parse(text: str) -> ast.Module | None:
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def functions_of(code: Iterable[str]) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    for text in code:
        tree = _parse(text)
        if tree is None:
            continue
        for item in ast.walk(tree):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield item


def defined_names(code: Iterable[str]) -> set[str]:
    """Top-level function, class and assigned names of the sources that parse."""
    found: set[str] = set()
    for text in code:
        tree = _parse(text)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.add(node.name)
            elif isinstance(node, ast.Assign):
                found |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return found


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Index:
    shingles: ContaminationIndex
    reference_trees: set[str]
    case_names: list[frozenset[str]]
    blocklist: Blocklist | None = None
    benchmarks: ContaminationIndex | None = None
    dev: ContaminationIndex | None = None
    identity: dict[str, Any] = field(default_factory=dict)

    def rejects(self, row: RowText) -> bool:
        texts = [text for text in row.texts if text]
        if any(self.shingles.contains(text) for text in texts):
            return True
        if any(normalized_dump(item) in self.reference_trees for item in functions_of(row.code)):
            return True
        names = defined_names(row.code)
        for case in self.case_names:
            union = names | case
            if union and len(names & case) / len(union) >= JACCARD:
                return True
        if self.blocklist is not None:
            defined = names | {item.name for item in functions_of(row.code)}
            if defined & self.blocklist.names:
                return True
            modules = {name.removesuffix(".py") for name in row.modules}
            modules |= {name.removeprefix("test_") for name in modules}
            if modules & self.blocklist.modules:
                return True
        for external in (self.benchmarks, self.dev):
            if external is not None and any(external.contains(text) for text in texts):
                return True
        return False


def case_texts(case: cases.Case) -> list[str]:
    return [
        case.task,
        case.title,
        *case.files.values(),
        *case.reference.values(),
        case.correctness,
        case.preservation,
    ]


def build_index(
    case_roots: Sequence[Path],
    *,
    families: Path | None = None,
    benchmarks: Sequence[Path] = (),
    dev: Path | None = None,
) -> Index:
    """The index of every registered evaluation case under ``case_roots``, the family
    blocklist and the external n-gram sets."""
    texts: list[str] = []
    trees: set[str] = set()
    names: list[frozenset[str]] = []
    identity: dict[str, Any] = {"case_roots": []}
    for root in case_roots:
        loaded = cases.load_cases(root)
        identity["case_roots"].append({"registry_sha256": cases.registry_sha256(root)})
        for case in loaded:
            texts.extend(case_texts(case))
            for key in case.targets:
                trees |= {
                    normalized_dump(item) for item in functions_of([case.reference_function(key)])
                }
            sources = [
                text for path, text in case.files.items() if not workspace.is_test_file(path)
            ]
            names.append(frozenset(defined_names([*sources, *case.reference.values()])))
    index = Index(ContaminationIndex(texts, NGRAM), trees, names)
    if families is not None:
        index.blocklist = Blocklist.load(families)
        identity["families_sha256"] = _sha_file(families)
    if benchmarks:
        index.benchmarks = ContaminationIndex(
            benchmark_texts(benchmarks), NGRAM, min_alpha_tokens=EXTERNAL_MIN_ALPHA
        )
        identity["benchmarks"] = {path.name: _sha_file(path) for path in benchmarks}
    if dev is not None:
        index.dev = ContaminationIndex(pool_texts(dev), NGRAM, min_alpha_tokens=EXTERNAL_MIN_ALPHA)
        identity["dev_sha256"] = _sha_file(dev)
        identity["dev_method"] = DEV_METHOD
    index.identity = identity
    return index


BENCHMARK_FIELDS = ("prompt", "canonical_solution", "code")


def benchmark_texts(paths: Sequence[Path]) -> list[str]:
    """Prompt and reference-solution fields of gzipped benchmark JSONL files."""
    found: list[str] = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    found.extend(str(row[name]) for name in BENCHMARK_FIELDS if row.get(name))
    return found


def pool_texts(path: Path) -> list[str]:
    """Prompt and solution of every task of a pool split file."""
    found: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                found.extend(str(row[name]) for name in ("prompt", "solution") if row.get(name))
    return found


def instance_rows(root: Path) -> list[RowText]:
    """One row per generated instance directory under ``root``: every file it holds."""
    found = []
    for directory in sorted(path for path in root.iterdir() if (path / "case.json").is_file()):
        spec = json.loads((directory / "case.json").read_text(encoding="utf-8"))
        texts = [spec["task"]]
        code = []
        modules = []
        for path in sorted(directory.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            texts.append(text)
            if path.parent.name in {"repo", "reference"}:
                code.append(text)
                modules.append(path.name)
        found.append(RowText(spec["id"], tuple(texts), tuple(code), tuple(modules)))
    return found


def meta_rows(path: Path) -> list[RowText]:
    """Rows of a build's ``rows-meta.jsonl`` (user text and reply of each row)."""
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            found.append(RowText(row["id"], (row["user_text"], row["reply"]), (row["reply"],)))
    return found


def decoded_rows(path: Path) -> list[RowText]:
    """Rows of a decoded replay index (``id``, ``user`` turns, ``assistant`` turns)."""
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            user = [str(text) for text in row.get("user", [])]
            assistant = [str(text) for text in row.get("assistant", [])]
            found.append(RowText(str(row["id"]), (*user, *assistant), tuple(assistant)))
    return found


def check(index: Index, rows: Iterable[RowText]) -> dict[str, Any]:
    """Status-only result: the rejected row IDs, sorted, and how many rows were checked."""
    checked = 0
    rejected = []
    for row in rows:
        checked += 1
        if index.rejects(row):
            rejected.append(row.id)
    return {"schema": SCHEMA, "checked": checked, "rejected": sorted(rejected)}


def load_result(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a contamination result")
    return dict(raw)


def rejected_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return set(load_result(path)["rejected"])


def registries(result: Mapping[str, Any]) -> set[str]:
    return {str(root["registry_sha256"]) for root in result.get("index", {}).get("case_roots", [])}


def require_result(
    result: Mapping[str, Any], *, input_path: Path, kind: str, case_registries: Iterable[str]
) -> None:
    """Refuse a result unless it was computed on ``input_path`` as it is now (same SHA-256,
    same kind of input) and against exactly the registries ``case_registries`` names."""
    if result.get("input_kind") != kind:
        raise ValueError(f"the contamination result checked {result.get('input_kind')}, not {kind}")
    if result.get("input_sha256") != _sha_file(input_path):
        raise ValueError(f"the contamination result was not computed on {input_path} as it is")
    wanted = set(case_registries)
    if not wanted:
        raise ValueError("name the evaluation-case registries the result must cover")
    if registries(result) != wanted:
        raise ValueError("the contamination result does not cover exactly the named case sets")


def describe(index: Index) -> Mapping[str, Any]:
    return dict(index.identity)
