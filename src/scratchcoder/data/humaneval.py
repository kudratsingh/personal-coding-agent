"""The pinned original HumanEval release (164 tasks), evaluation-only.

This is the release the reference gate (G3) scored: OpenAI's ``HumanEval.jsonl.gz`` at a
pinned revision. The loader only parses text; it never executes it.
"""

import gzip
import hashlib
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scratchcoder.data.manifest import ManifestEntry
from scratchcoder.eval.tasks import Task

DATA_REVISION = "6d43fb980f9fee3c892a914eda09951f772ad10d"
DATA_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
DATA_BYTES = 44877
DATA_URL = (
    f"https://raw.githubusercontent.com/openai/human-eval/{DATA_REVISION}/data/HumanEval.jsonl.gz"
)
TASK_COUNT = 164
SOURCE = "openai/human-eval"


def manifest() -> ManifestEntry:
    """The pinned download, for :func:`scratchcoder.data.manifest.download`."""
    return ManifestEntry(
        source=SOURCE,
        revision=DATA_REVISION,
        url=DATA_URL,
        license="MIT",
        attribution="OpenAI HumanEval",
        sha256=DATA_SHA256,
        size_bytes=DATA_BYTES,
        purpose="eval",
        license_verified=True,
    )


def normalize(rows: Sequence[dict[str, Any]]) -> list[Task]:
    """Map release rows to tasks: the hidden test is the release's ``check`` plus its call."""
    tasks = []
    for row in rows:
        entry = row["entry_point"]
        if not isinstance(entry, str) or not entry.isidentifier():
            raise ValueError("invalid benchmark entry point")
        tasks.append(
            Task(
                id=row["task_id"],
                prompt=row["prompt"],
                public_tests=f"assert callable({entry})",
                hidden_tests=row["test"] + f"\ncheck({entry})",
                solution=row["prompt"] + row["canonical_solution"],
                entry_point=entry,
                source=SOURCE,
                license="MIT",
                written_by="human",
                split="eval",
            )
        )
    return tasks


def parse_release(raw: bytes) -> list[Task]:
    """The 164 tasks of the pinned release bytes; any other bytes are rejected."""
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("benchmark content hash differs from pinned bytes")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        rows = [json.loads(line) for line in stream.read(16 * 1024 * 1024).splitlines()]
    tasks = normalize(rows)
    if len(tasks) != TASK_COUNT or len({task.id for task in tasks}) != TASK_COUNT:
        raise ValueError("expected the complete 164-task benchmark")
    return tasks


def read_humaneval(path: Path) -> list[Task]:
    """Read the pinned release from a local file, checking its size and SHA-256."""
    if path.stat().st_size != DATA_BYTES:
        raise ValueError("benchmark size differs from the pinned release")
    return parse_release(path.read_bytes())
