"""Normalized task records and explicit source-field mapping."""

import json
import keyword
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    public_tests: str
    hidden_tests: str
    source: str
    license: str
    written_by: str
    split: str
    mode: str = "function"
    entry_point: str | None = None
    solution: str | None = None
    source_revision: str | None = None

    def __post_init__(self) -> None:
        strings = (
            self.id,
            self.prompt,
            self.public_tests,
            self.hidden_tests,
            self.source,
            self.license,
            self.written_by,
            self.split,
            self.mode,
        )
        if any(not isinstance(value, str) or not value.strip() for value in strings):
            raise ValueError("task text and provenance fields must be nonempty strings")
        if self.solution is not None and not isinstance(self.solution, str):
            raise ValueError("reference solution must be text or null")
        if self.entry_point is not None and (
            not isinstance(self.entry_point, str)
            or not self.entry_point.isidentifier()
            or keyword.iskeyword(self.entry_point)
        ):
            raise ValueError("entry point must be a Python identifier")
        if self.source_revision is not None and (
            not isinstance(self.source_revision, str)
            or not re.fullmatch(r"[0-9a-f]{40,64}", self.source_revision)
        ):
            raise ValueError("source revision must be immutable")
        if not all(
            (self.id, self.prompt, self.public_tests, self.hidden_tests, self.source, self.license)
        ):
            raise ValueError("task identity, prompt, provenance, and separate tests are required")
        if self.split not in {"train", "dev", "test", "eval"} or self.mode not in {
            "function",
            "io",
        }:
            raise ValueError("invalid task split or mode")
        if self.written_by not in {"human", "model"}:
            raise ValueError("task authorship must be recorded")
        if self.mode == "io":
            for text in (self.public_tests, self.hidden_tests):
                cases = json.loads(text)
                if (
                    not isinstance(cases, list)
                    or not cases
                    or any(
                        not isinstance(case, dict)
                        or set(case) != {"input", "output"}
                        or not all(isinstance(value, str) for value in case.values())
                        for case in cases
                    )
                ):
                    raise ValueError("invalid I/O cases")

    def public_prompt(self) -> str:
        return self.prompt + "\n\nPublic tests:\n" + self.public_tests


def load_tasks(path: Path) -> Iterator[Task]:
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                task = Task(**json.loads(line))
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid task on line {number}") from error
            if task.id in seen:
                raise ValueError(f"duplicate task id: {task.id}")
            seen.add(task.id)
            yield task


def save_tasks(path: Path, tasks: Iterable[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for task in tasks:
            stream.write(json.dumps(asdict(task), sort_keys=True) + "\n")


def map_source_row(
    row: Mapping[str, Any], fields: Mapping[str, str], *, metadata: Mapping[str, Any]
) -> Task:
    """Require declared field mappings; never guess a dataset's current schema."""
    data = dict(metadata)
    for task_field, source_field in fields.items():
        if source_field not in row:
            raise ValueError(f"missing source field: {source_field}")
        data[task_field] = row[source_field]
    return Task(**data)
