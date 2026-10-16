"""Explicit, fingerprint-bound deviations from the pinned OriginFmt test scripts."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from typing import Any

from scratchcoder.data.evalplus import EvalPlusRelease, release
from scratchcoder.eval.tasks import Task

FIND_ZERO_VERSION = "find-zero-residual-v1"
FIND_ZERO_ORIGINAL_SHA256 = "41ab5c51c15d24d07317ba9153761d568c3c249234f3442223c3f4511b88fd01"
BROKEN_ASSERTION = "assert _poly(*candidate(*inp), inp) <= 0.0001"
CORRECTED_ASSERTION = "assert abs(_poly(*inp, candidate(*inp))) <= 0.0001"
REFERENCE_REVISION = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"


def correct_find_zero(
    tasks: Sequence[Task], specification: EvalPlusRelease
) -> tuple[list[Task], dict[str, Any]]:
    """Change one reviewed expression; reject any unreviewed release or test bytes."""
    if specification != release("humaneval-plus"):
        raise ValueError("find-zero correction requires the exact pinned HumanEval+ release")
    matches = [task for task in tasks if task.id == "humaneval/32"]
    if len(matches) != 1:
        raise ValueError("find-zero correction requires exactly one task 32")
    task = matches[0]
    if (
        task.source != specification.manifest.source
        or task.source_revision != specification.manifest.revision
        or task.entry_point != "find_zero"
        or task.mode != "function"
        or task.split != "eval"
    ):
        raise ValueError("find-zero task provenance differs")
    original_hash = hashlib.sha256(task.hidden_tests.encode()).hexdigest()
    if original_hash != FIND_ZERO_ORIGINAL_SHA256:
        raise ValueError("find-zero original hidden-script SHA-256 differs")
    if task.hidden_tests.count(BROKEN_ASSERTION) != 1:
        raise ValueError("find-zero requires exactly one original assertion")
    corrected = replace(
        task, hidden_tests=task.hidden_tests.replace(BROKEN_ASSERTION, CORRECTED_ASSERTION, 1)
    )
    return [corrected if item.id == task.id else item for item in tasks], {
        "version": FIND_ZERO_VERSION,
        "task_id": task.id,
        "original_release_sha256": specification.manifest.sha256,
        "original_hidden_script_sha256": original_hash,
        "effective_hidden_script_sha256": hashlib.sha256(
            corrected.hidden_tests.encode()
        ).hexdigest(),
        "effective_task_sha256": hashlib.sha256(
            json.dumps(asdict(corrected), sort_keys=True).encode()
        ).hexdigest(),
        "reference_evaluator_revision": REFERENCE_REVISION,
    }
