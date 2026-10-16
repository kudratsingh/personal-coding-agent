"""Pinned ARC-Easy test intake; other splits are read only for exclusion checks."""

import hashlib
import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scratchcoder.data.manifest import ManifestEntry
from scratchcoder.eval.choices import ChoiceTask, adapt_arc_row

REVISION = "210d026faf9955653af8916fad021475a3f00453"


@dataclass(frozen=True)
class ArcSplit:
    split: str
    manifest: ManifestEntry
    task_count: int

    def validate(self) -> None:
        self.manifest.validate()
        if (
            self.split not in {"train", "validation", "test"}
            or self.manifest.source != "allenai/ai2_arc"
            or self.manifest.purpose != "eval"
            or type(self.task_count) is not int
            or self.task_count < 1
            or self.task_count > 3000
            or not 0 < self.manifest.size_bytes <= 2 * 1024 * 1024
        ):
            raise ValueError("ARC-Easy requires evaluation-only split specifications")

    @property
    def filename(self) -> str:
        return f"{self.split}-00000-of-00001.parquet"


def arc_splits() -> dict[str, ArcSplit]:
    identities = {
        "train": (2251, 330598, "b315db8a4be597dc7daa50a4e70d48dd7c990c32085629e6ccd8c926beaa80b5"),
        "validation": (
            570,
            86080,
            "ed890ff1e4cef7a7140d3a30dcea3ed2c9d467c6458f447ad9ef0176d8dcbb74",
        ),
        "test": (2376, 346257, "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177"),
    }
    return {
        split: ArcSplit(
            split,
            ManifestEntry(
                source="allenai/ai2_arc",
                revision=REVISION,
                url=(
                    f"https://huggingface.co/datasets/allenai/ai2_arc/resolve/{REVISION}"
                    f"/ARC-Easy/{split}-00000-of-00001.parquet"
                ),
                license="CC-BY-SA-4.0",
                attribution="Allen Institute for AI; Clark et al. (2018), AI2 Reasoning Challenge",
                sha256=digest,
                size_bytes=size,
                purpose="eval",
                license_verified=True,
            ),
            count,
        )
        for split, (count, size, digest) in identities.items()
    }


def _read_split(path: Path, specification: ArcSplit) -> list[ChoiceTask]:
    specification.validate()
    manifest = specification.manifest
    if path.stat().st_size != manifest.size_bytes:
        raise ValueError("ARC split byte size differs")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.sha256:
        raise ValueError("ARC split SHA-256 differs")
    try:
        parquet = importlib.import_module("pyarrow.parquet")
    except ImportError as error:
        raise RuntimeError("ARC Parquet intake requires the locked evaluation extra") from error
    with parquet.ParquetFile(
        path,
        arrow_extensions_enabled=False,
        thrift_string_size_limit=1024 * 1024,
        thrift_container_size_limit=10000,
    ) as source:
        metadata = source.metadata
        if (
            metadata.num_rows != specification.task_count
            or metadata.num_row_groups > 8
            or set(source.schema_arrow.names) != {"id", "question", "choices", "answerKey"}
            or len(source.schema_arrow.names) != 4
            or sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
            > 16 * 1024 * 1024
        ):
            raise ValueError("ARC split schema, count or expanded size differs")
        table = source.read(use_threads=False)
        if table.nbytes > 16 * 1024 * 1024:
            raise ValueError("ARC expanded table exceeds bound")
        rows = table.to_pylist()
    tasks = []
    for row in rows:
        if (
            set(row) != {"id", "question", "choices", "answerKey"}
            or not isinstance(row["choices"], dict)
            or set(row["choices"]) != {"label", "text"}
        ):
            raise ValueError("unsupported ARC row schema")
        # Non-test splits are validation/exclusion-only, never returned for scoring/training.
        tasks.append(
            adapt_arc_row(row, manifest, split="test" if specification.split == "test" else "eval")
        )
    if len(tasks) != specification.task_count or len({task.id for task in tasks}) != len(tasks):
        raise ValueError("ARC split count or unique IDs differ")
    return tasks


def read_arc_easy(directory: Path) -> tuple[list[ChoiceTask], dict[str, Any]]:
    specifications = arc_splits()
    splits = {
        name: _read_split(directory / spec.filename, spec) for name, spec in specifications.items()
    }
    seen: set[str] = set()
    for tasks in splits.values():
        identifiers = {task.id for task in tasks}
        if seen & identifiers:
            raise ValueError("ARC IDs overlap across source splits")
        seen.update(identifiers)
    return splits["test"], {
        "suite": "ARC-Easy",
        "source_split": "test",
        "data_manifests": {name: asdict(spec.manifest) for name, spec in specifications.items()},
        "source_split_counts": {name: len(tasks) for name, tasks in splits.items()},
        "cross_split_id_overlap": 0,
        "other_splits_usage": "schema-validation-and-ID-exclusion-only",
        "official_scoring_protocol_equivalent": False,
    }
