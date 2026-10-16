"""Source-component reference audit; never an approved training-pool export."""

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

from scratchcoder.data.adapters import adapt_row, canonical_id
from scratchcoder.data.filters import ContaminationIndex
from scratchcoder.data.mbpp import MbppRelease
from scratchcoder.data.pool import build_pool, split_summary
from scratchcoder.eval.choices import ChoiceTask
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.diagnostics import failure_category
from scratchcoder.sandbox.runner import Result


def audit_mbpp(
    rows: Sequence[dict[str, Any]],
    specification: MbppRelease,
    benchmarks: Sequence[Task],
    choices: Sequence[ChoiceTask],
    *,
    execute: Callable[[str, str, str], Result],
) -> dict[str, Any]:
    specification.validate()
    if len(rows) != specification.task_count or not benchmarks or not choices:
        raise ValueError("complete source and explicit held-out protection required")
    protected_ids = {canonical_id(task.id) for task in benchmarks}
    source_ids = {f"mbpp/{row['task_id']}" for row in rows}
    if len(source_ids) != len(rows) or any(
        identifier.startswith("mbpp/") and identifier not in source_ids
        for identifier in protected_ids
    ):
        raise ValueError("unique complete MBPP source IDs must cover the protected MBPP IDs")
    choice_index = ContaminationIndex(
        text for task in choices for text in (task.question, task.public_prompt())
    )
    candidates = []
    counts: Counter[str] = Counter()
    for row in rows:
        counts["seen"] += 1
        if f"mbpp/{row['task_id']}" in protected_ids:
            counts["benchmark_id_excluded"] += 1
            continue
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
    if counts["seen"] != sum(count for name, count in counts.items() if name != "seen"):
        raise RuntimeError("MBPP intake accounting differs")
    if pool.get("seen", 0) != len(candidates) or len(candidates) != sum(
        pool.get(reason, 0)
        for reason in ("kept", "invalid_solution", "missing_solution", "contamination_or_duplicate")
    ):
        raise RuntimeError("MBPP pool accounting differs")
    return {
        "status": "completed",
        "suite": "MBPP-reference-intake-audit",
        "training_admission": False,
        "complete_final_holdout_coverage": False,
        "source_manifest": asdict(specification.manifest),
        "source_rows": len(rows),
        "source_test_counts": {
            "primary": sum(len(row["test_list"]) for row in rows),
            "challenge": sum(len(row["challenge_test_list"]) for row in rows),
        },
        "intake_counts": dict(counts),
        "pool_counts": pool,
        "original_id_partition_counts": {
            "evaluation_11_510": sum(11 <= row["task_id"] <= 510 for row in rows),
            "prompting_1_10": sum(1 <= row["task_id"] <= 10 for row in rows),
            "training_511_1000": sum(511 <= row["task_id"] <= 1000 for row in rows),
        },
        "reference_execution_categories": dict(outcomes),
        "audit_kept_ids": sorted(task.id for task in kept),
        "audit_split_summary": split_summary(kept),
        "task_hash": hashlib.sha256(
            json.dumps(
                [asdict(task) for task in sorted(kept, key=lambda task: task.id)], sort_keys=True
            ).encode()
        ).hexdigest(),
        "protected_code_task_count": len(benchmarks),
        "protected_code_task_hash": hashlib.sha256(
            json.dumps(
                [asdict(task) for task in sorted(benchmarks, key=lambda task: task.id)],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "protected_choice_task_count": len(choices),
        "protected_choice_task_hash": hashlib.sha256(
            json.dumps(
                [asdict(task) for task in sorted(choices, key=lambda task: task.id)], sort_keys=True
            ).encode()
        ).hexdigest(),
        "choice_protection": "question-and-public-options-text; no standalone-answer-label-index",
        "test_policy": (
            "primary-and-challenge-assertions; one-public-rest-hidden; exact-assertion-dedup"
        ),
    }
