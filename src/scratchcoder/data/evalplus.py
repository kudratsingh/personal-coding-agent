"""Pinned evaluation-only OriginFmt releases; preserve upstream executable tests."""

import ast
import gzip
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scratchcoder.data.manifest import ManifestEntry
from scratchcoder.eval.tasks import Task


@dataclass(frozen=True)
class EvalPlusRelease:
    suite: str
    manifest: ManifestEntry
    task_count: int

    def validate(self) -> None:
        self.manifest.validate()
        if self.suite not in {"humaneval-plus", "mbpp-plus"} or self.manifest.purpose != "eval":
            raise ValueError("EvalPlus releases are evaluation-only")
        if type(self.task_count) is not int or self.task_count < 1:
            raise ValueError("expected release task count must be positive")


def release(suite: str) -> EvalPlusRelease:
    if suite == "humaneval-plus":
        repo, revision, name = (
            "humanevalplus_release",
            "200defce9e3429d28ca215b6dd061c0f7f31c18b",
            "HumanEvalPlus-OriginFmt.jsonl.gz",
        )
        size, digest, count = (
            1350689,
            "daa7661c8189924068069b0872a440b491edb60f8bdf431d5957adc88d18bae5",
            164,
        )
        license_name, credit = "Apache-2.0 AND MIT", "EvalPlus Team; OpenAI HumanEval (MIT)"
    elif suite == "mbpp-plus":
        repo, revision, name = (
            "mbppplus_release",
            "dadf43da556a00f7bacd71cb154f2932757d9144",
            "MbppPlus-OriginFmt.jsonl.gz",
        )
        size, digest, count = (
            549938,
            "7a5aec547b134fa297ee6c45f829298aa3f58de02547b8d8a5c3d85d9cb10efc",
            378,
        )
        license_name, credit = (
            "Apache-2.0 AND CC-BY-4.0",
            "EvalPlus Team; Google Research MBPP, Austin et al. (2021), CC-BY-4.0",
        )
    else:
        raise ValueError("unsupported EvalPlus suite")
    return EvalPlusRelease(
        suite,
        ManifestEntry(
            source=f"evalplus/{repo}",
            revision=revision,
            url=f"https://raw.githubusercontent.com/evalplus/{repo}/{revision}/{name}",
            license=license_name,
            attribution=credit,
            sha256=digest,
            size_bytes=size,
            purpose="eval",
            license_verified=True,
        ),
        count,
    )


def origin_task(row: dict[str, Any], specification: EvalPlusRelease) -> Task:
    specification.validate()
    manifest = specification.manifest
    common = {
        "source": manifest.source,
        "source_revision": manifest.revision,
        "license": manifest.license,
        "written_by": "human",
        "split": "eval",
    }
    if specification.suite == "humaneval-plus":
        if set(row) != {"task_id", "prompt", "canonical_solution", "entry_point", "test"}:
            raise ValueError("unsupported HumanEval+ OriginFmt schema")
        identifier, entry = row["task_id"], row["entry_point"]
        if not isinstance(identifier, str) or not re.fullmatch(r"HumanEval/[0-9]+", identifier):
            raise ValueError("invalid HumanEval+ ID")
        if not isinstance(entry, str) or not entry.isidentifier():
            raise ValueError("invalid HumanEval+ entry point")
        if not all(
            isinstance(row[key], str) and row[key].strip()
            for key in ("prompt", "test", "canonical_solution")
        ):
            raise ValueError("nonempty HumanEval+ source text required")
        nodes = ast.parse(row["test"]).body
        if sum(isinstance(node, ast.FunctionDef) and node.name == "check" for node in nodes) != 1:
            raise ValueError("HumanEval+ requires the upstream check function")
        return Task(
            id=identifier.casefold(),
            prompt=row["prompt"],
            public_tests=f"assert callable({entry})",
            hidden_tests=row["test"] + f"\ncheck({entry})\n",
            solution=row["prompt"] + row["canonical_solution"],
            entry_point=entry,
            **common,
        )
    if set(row) != {
        "task_id",
        "code",
        "prompt",
        "source_file",
        "test_imports",
        "test_list",
        "test",
    }:
        raise ValueError("unsupported MBPP+ OriginFmt schema")
    identifier = row["task_id"]
    if type(identifier) is not int or identifier < 0:
        raise ValueError("MBPP+ requires an original integer task ID")
    for key in ("test_imports", "test_list"):
        if not isinstance(row[key], list) or any(not isinstance(item, str) for item in row[key]):
            raise ValueError("MBPP+ imports/tests must be string arrays")
    if not row["test_list"] or not all(
        isinstance(row[key], str) and row[key].strip() for key in ("code", "prompt", "test")
    ):
        raise ValueError("MBPP+ source code, prompt, public and hidden tests are required")
    imports = "\n".join(row["test_imports"])
    if any(not isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.parse(imports).body):
        raise ValueError("MBPP+ import preamble must contain only imports")
    ast.parse(row["test"])
    return Task(
        id=f"mbpp/{identifier}",
        prompt=row["prompt"],
        public_tests=imports + "\n" + "\n".join(row["test_list"]),
        hidden_tests=row["test"],
        solution=imports + "\n" + row["code"],
        **common,
    )


def read_release(
    path: Path, specification: EvalPlusRelease, *, expanded_limit: int = 64 * 1024 * 1024
) -> list[Task]:
    specification.validate()
    if type(expanded_limit) is not int or expanded_limit < 1:
        raise ValueError("positive expanded byte limit required")
    if path.stat().st_size != specification.manifest.size_bytes:
        raise ValueError("release compressed size differs")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != specification.manifest.sha256:
        raise ValueError("release SHA-256 differs")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        expanded = stream.read(expanded_limit + 1)
    if len(expanded) > expanded_limit:
        raise ValueError("release exceeds expanded byte limit")
    tasks = [
        origin_task(json.loads(line), specification)
        for line in expanded.splitlines()
        if line.strip()
    ]
    if len(tasks) != specification.task_count or len({task.id for task in tasks}) != len(tasks):
        raise ValueError("release task count or unique IDs differ")
    return tasks
