"""Held-out multiple-choice scoring without answer-key exposure or code execution."""

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scratchcoder.common.records import git_state, utc_now
from scratchcoder.data.manifest import ManifestEntry
from scratchcoder.eval.harness import ContextLimitError
from scratchcoder.eval.metrics import accuracy, paired_bootstrap


@dataclass(frozen=True)
class ChoiceTask:
    id: str
    question: str
    labels: tuple[str, ...]
    choices: tuple[str, ...]
    answer: str
    source: str
    source_revision: str
    license: str
    split: str = "eval"

    def __post_init__(self) -> None:
        strings = (self.id, self.question, self.answer, self.source, self.license)
        if any(not isinstance(value, str) or not value.strip() for value in strings):
            raise ValueError("choice identity, question, answer, and provenance are required")
        if (
            not isinstance(self.labels, tuple)
            or not isinstance(self.choices, tuple)
            or not 2 <= len(self.labels) <= 26
            or len(self.labels) != len(self.choices)
            or any(
                not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9]+", label)
                for label in self.labels
            )
            or len(set(self.labels)) != len(self.labels)
            or any(not isinstance(text, str) or not text.strip() for text in self.choices)
            or self.answer not in self.labels
        ):
            raise ValueError("unique aligned choice labels, text, and a valid answer are required")
        if not isinstance(self.source_revision, str) or not re.fullmatch(
            r"[0-9a-f]{40,64}", self.source_revision
        ):
            raise ValueError("immutable source revision is required")
        if self.split not in {"dev", "test", "eval"}:
            raise ValueError("choice evaluation requires a held-out split")

    def public_prompt(self) -> str:
        options = "\n".join(
            f"{label}. {text}" for label, text in zip(self.labels, self.choices, strict=True)
        )
        return f"Question: {self.question}\n{options}\nAnswer:"


def load_choices(path: Path) -> Iterator[ChoiceTask]:
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row["labels"], list) or not isinstance(row["choices"], list):
                    raise ValueError("choice fields must be arrays")
                row["labels"], row["choices"] = tuple(row["labels"]), tuple(row["choices"])
                task = ChoiceTask(**row)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"invalid choice task on line {number}") from error
            if task.id in seen:
                raise ValueError("duplicate choice task ID")
            seen.add(task.id)
            yield task


def adapt_arc_row(
    row: Mapping[str, Any], provenance: ManifestEntry, *, split: str = "eval"
) -> ChoiceTask:
    """Convert explicit question/choices/answerKey schema; never download or infer rights."""
    provenance.validate()
    if provenance.purpose != "eval" or provenance.source != "allenai/ai2_arc":
        raise ValueError("ARC requires evaluation-only provenance")
    options = row["choices"]
    if (
        not isinstance(options, Mapping)
        or not isinstance(options["label"], list)
        or not isinstance(options["text"], list)
    ):
        raise ValueError("unsupported ARC choices schema")
    return ChoiceTask(
        id=row["id"],
        question=row["question"],
        labels=tuple(options["label"]),
        choices=tuple(options["text"]),
        answer=row["answerKey"],
        source=provenance.source,
        source_revision=provenance.revision,
        license=provenance.license,
        split=split,
    )


def evaluate_choices(
    tasks: Sequence[ChoiceTask],
    score: Callable[[str, list[str]], Sequence[float]],
    *,
    checkpoint_hash: str,
    tokenizer_hash: str,
    seed: int = 0,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    if not tasks or len({task.id for task in tasks}) != len(tasks):
        raise ValueError("nonempty unique choice tasks required")
    if not checkpoint_hash or not tokenizer_hash:
        raise ValueError("checkpoint and tokenizer identities are required")
    ordered = sorted(tasks, key=lambda task: task.id)
    predictions: list[str | None] = []
    rows: list[dict[str, Any]] = []
    for task in ordered:
        try:
            # The scorer receives ONLY public text and label continuations, not Task/answer.
            values = list(score(task.public_prompt(), [" " + label for label in task.labels]))
            if len(values) != len(task.labels) or any(not math.isfinite(value) for value in values):
                raise ValueError("aligned finite choice scores are required")
            predicted = task.labels[max(range(len(values)), key=values.__getitem__)]
            status = "scored"
        except ContextLimitError:
            values, predicted, status = [], None, "context_limit"
        predictions.append(predicted)
        rows.append(
            {
                "id": task.id,
                "source": task.source,
                "prediction": predicted,
                "correct": predicted == task.answer,
                "status": status,
                "label_scores": values,
            }
        )
    commit, dirty = git_state()
    return {
        "schema_version": 1,
        "status": "completed",
        "suite": "multiple-choice",
        "created_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "checkpoint_hash": checkpoint_hash,
        "tokenizer_hash": tokenizer_hash,
        "task_hash": hashlib.sha256(
            json.dumps([asdict(task) for task in ordered], sort_keys=True).encode()
        ).hexdigest(),
        "task_count": len(tasks),
        "configuration": {
            "scoring": "sum_log_probability_of_space_prefixed_label",
            "tie_break": "first-listed-label",
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
        },
        "metrics": {
            "accuracy": accuracy(predictions, [task.answer for task in ordered]),
            "confidence_interval": asdict(
                paired_bootstrap(
                    [0.0] * len(tasks),
                    [float(row["correct"]) for row in rows],
                    seed=seed,
                    samples=bootstrap_samples,
                )
            ),
        },
        "per_task": rows,
    }
