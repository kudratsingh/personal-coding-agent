"""Bounded evaluation-only pilot of one pinned OpenCodeInstruct shard prefix."""

import hashlib
import importlib
import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scratchcoder.data.adapters import adapt_row
from scratchcoder.data.filters import ContaminationIndex
from scratchcoder.data.manifest import ManifestEntry
from scratchcoder.data.pool import build_pool, split_summary
from scratchcoder.eval.choices import ChoiceTask
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.diagnostics import failure_category
from scratchcoder.sandbox.runner import Result

TEXT_FIELDS = frozenset(
    {
        "id",
        "input",
        "output",
        "domain",
        "generation_algorithm",
        "llm_judgement",
        "unit_tests",
        "tests_execution_status",
    }
)


@dataclass(frozen=True)
class OpenCodeSample:
    manifest: ManifestEntry
    file_rows: int = 100000
    sample_rows: int = 200
    filename: str = "train-00000-of-00050.parquet"

    def validate(self) -> None:
        self.manifest.validate()
        if (
            self.manifest.source != "nvidia/OpenCodeInstruct"
            or self.manifest.purpose != "eval"
            or not 0 < self.manifest.size_bytes <= 160 * 1024 * 1024
            or type(self.file_rows) is not int
            or not 1 <= self.file_rows <= 100000
            or type(self.sample_rows) is not int
            or not 1 <= self.sample_rows <= min(200, self.file_rows)
            or self.filename != "train-00000-of-00050.parquet"
        ):
            raise ValueError("bounded evaluation-only OpenCodeInstruct prefix required")


def opencode_sample() -> OpenCodeSample:
    revision = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
    return OpenCodeSample(
        ManifestEntry(
            source="nvidia/OpenCodeInstruct",
            revision=revision,
            url=f"https://huggingface.co/datasets/nvidia/OpenCodeInstruct/resolve/{revision}/data/train-00000-of-00050.parquet",
            license="CC-BY-4.0",
            attribution="NVIDIA; Ahmad et al. (2025), OpenCodeInstruct",
            sha256="342757e0c6b706c8f68cf0a867ead5c417d0273726453dbcfc934f5c3c6ca891",
            size_bytes=133476478,
            purpose="eval",
            license_verified=True,
        )
    )


def _validate_rows(rows: Sequence[dict[str, Any]], specification: OpenCodeSample) -> None:
    if len(rows) != specification.sample_rows:
        raise ValueError("OpenCode sample count differs")
    for row in rows:
        if (
            set(row) != TEXT_FIELDS | {"average_test_score"}
            or any(
                not isinstance(row[name], str) or len(row[name]) > 128000 for name in TEXT_FIELDS
            )
            or any(not row[name].strip() for name in ("id", "input", "output", "unit_tests"))
            or type(row["average_test_score"]) is not float
            or not math.isfinite(row["average_test_score"])
            or not 0 <= row["average_test_score"] <= 1
        ):
            raise ValueError("unsupported OpenCode row schema")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("OpenCode sample IDs must be unique")


def read_opencode_sample(path: Path, specification: OpenCodeSample) -> list[dict[str, Any]]:
    specification.validate()
    if path.stat().st_size != specification.manifest.size_bytes:
        raise ValueError("OpenCode shard byte size differs")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != specification.manifest.sha256:
        raise ValueError("OpenCode shard SHA-256 differs")
    try:
        arrow = importlib.import_module("pyarrow")
        parquet = importlib.import_module("pyarrow.parquet")
    except ImportError as error:
        raise RuntimeError(
            "OpenCode Parquet intake requires the locked evaluation extra"
        ) from error
    with parquet.ParquetFile(
        path,
        arrow_extensions_enabled=False,
        thrift_string_size_limit=1024 * 1024,
        thrift_container_size_limit=10000,
    ) as source:
        metadata, schema = source.metadata, source.schema_arrow
        if (
            metadata.num_rows != specification.file_rows
            or not 1 <= metadata.num_row_groups <= 8
            or len(schema.names) != 9
            or set(schema.names) != TEXT_FIELDS | {"average_test_score"}
            or any(not arrow.types.is_string(schema.field(name).type) for name in TEXT_FIELDS)
            or not arrow.types.is_float64(schema.field("average_test_score").type)
            or sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
            > 512 * 1024 * 1024
        ):
            raise ValueError("OpenCode shard schema/count/expanded bounds differ")
        # Deliberately ordered prefix, not a random or representative source sample.
        batch = next(source.iter_batches(batch_size=specification.sample_rows, use_threads=False))
        if batch.nbytes > 4 * 1024 * 1024:
            raise ValueError("OpenCode sample batch exceeds bound")
        rows: list[dict[str, Any]] = batch.to_pylist()
    _validate_rows(rows, specification)
    return rows


def _task_hash(tasks: Sequence[Task | ChoiceTask]) -> str:
    return hashlib.sha256(
        json.dumps(
            [asdict(task) for task in sorted(tasks, key=lambda task: task.id)],
            sort_keys=True,
        ).encode()
    ).hexdigest()


def audit_opencode(
    rows: Sequence[dict[str, Any]],
    specification: OpenCodeSample,
    benchmarks: Sequence[Task],
    choices: Sequence[ChoiceTask],
    *,
    execute: Callable[[str, str, str], Result],
) -> dict[str, Any]:
    specification.validate()
    _validate_rows(rows, specification)
    if not benchmarks or not choices:
        raise ValueError("explicit benchmark/held-out protection required")
    choice_index = ContaminationIndex(
        text for task in choices for text in (task.question, task.public_prompt())
    )
    counts: Counter[str] = Counter()
    candidates = []
    for row in rows:
        counts["seen"] += 1
        try:
            task = adapt_row(
                specification.manifest.source, row, specification.manifest, split="eval"
            )
        except (ValueError, SyntaxError):
            counts["unsupported_assertion_schema"] += 1
            continue
        if any(
            choice_index.contains(text)
            for text in (
                task.prompt,
                task.solution or "",
                task.public_tests,
                task.hidden_tests,
            )
        ):
            counts["choice_contamination"] += 1
            continue
        candidates.append(task)
        counts["normalized_for_pool_checks"] += 1
    outcomes: Counter[str] = Counter()

    def check(code: str, tests: str, mode: str) -> Result:
        result = execute(code, tests, mode)
        outcomes[failure_category(result)] += 1
        return result

    kept, pool = build_pool(candidates, benchmarks, execute=check)
    if (
        pool.get("seen", 0) != len(candidates)
        or len(rows) != sum(count for name, count in counts.items() if name != "seen")
        or (
            len(candidates)
            != sum(
                pool.get(name, 0)
                for name in (
                    "kept",
                    "invalid_solution",
                    "missing_solution",
                    "contamination_or_duplicate",
                )
            )
        )
    ):
        raise RuntimeError("OpenCode sample accounting differs")
    return {
        "status": "completed",
        "suite": "OpenCodeInstruct-prefix-reference-pilot",
        "training_admission": False,
        "complete_final_holdout_coverage": False,
        "representative_source_sample": False,
        "selection": "first-shard-first-rows-in-file-order",
        "source_manifest": asdict(specification.manifest),
        "source_file_rows": specification.file_rows,
        "sample_rows": len(rows),
        "intake_counts": dict(counts),
        "pool_counts": pool,
        "reference_execution_categories": dict(outcomes),
        "audit_kept_ids": sorted(task.id for task in kept),
        "audit_split_summary": split_summary(kept),
        "task_hash": _task_hash(kept),
        "protected_code_task_count": len(benchmarks),
        "protected_code_task_hash": _task_hash(benchmarks),
        "protected_choice_task_count": len(choices),
        "protected_choice_task_hash": _task_hash(choices),
        "upstream_scores_usage": "schema-validation-only; never reference-pass evidence",
        "choice_protection": "question-and-public-options-text; no standalone-answer-label-index",
        "test_policy": "complete-supported-assertion-lists; one-public-rest-hidden; exact-dedup",
    }
