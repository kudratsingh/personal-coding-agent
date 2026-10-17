"""Canonical preflight before interpreting official standalone-suite model outputs."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from scratchcoder.data.evalplus import EvalPlusRelease
from scratchcoder.data.evalplus_corrections import correct_find_zero
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.eval.command import checkpoint_generator
from scratchcoder.eval.harness import EvaluationConfig, evaluate, save_report
from scratchcoder.eval.prompt import prompt_protocol
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.diagnostics import failure_category
from scratchcoder.sandbox.runner import Limits, Result, Runtime


def evaluate_suite(
    tasks: Sequence[Task],
    specification: EvalPlusRelease,
    *,
    execute: Callable[[str, str, str], Result],
    runtime: Runtime,
    limits: Limits,
    output: Path,
    checkpoint: Path | None = None,
    tokenizer_path: Path | None = None,
    config: EvaluationConfig | None = None,
    correct_find_zero_oracle: bool = False,
) -> dict[str, Any]:
    specification.validate()
    if output.exists():
        raise ValueError("suite output must be a new directory")
    if len(tasks) != specification.task_count or any(
        task.split != "eval"
        or not task.solution
        or task.source != specification.manifest.source
        or task.source_revision != specification.manifest.revision
        for task in tasks
    ):
        raise ValueError("complete held-out canonical suite required")
    if (checkpoint is None) != (tokenizer_path is None):
        raise ValueError("checkpoint and tokenizer must be supplied together")
    corrections = []
    if correct_find_zero_oracle:
        tasks, correction = correct_find_zero(tasks, specification)
        corrections.append(correction)
    metadata = {
        "suite": specification.suite,
        "data_manifest": asdict(specification.manifest),
        "runtime": asdict(runtime),
        "limits": asdict(limits),
        "upstream_resource_policy_equivalent": False,
        "oracle_policy": (
            "OriginFmt-with-find-zero-residual-v1" if corrections else "unmodified-OriginFmt-tests"
        ),
        "oracle_corrections": corrections,
    }
    categories: list[str] = []

    def check_reference(code: str, tests: str, mode: str) -> Result:
        result = execute(code, tests, mode)
        categories.append(failure_category(result))
        return result

    canonical = evaluate(
        tasks,
        lambda task, _: cast(str, task.solution),
        config=EvaluationConfig(),
        checkpoint_hash="canonical:" + specification.manifest.sha256,
        tokenizer_hash="not-applicable:canonical-source",
        execute=check_reference,
    )
    canonical["canonical_diagnostics"] = {
        row["id"]: category
        for row, category in zip(canonical["per_task"], categories, strict=True)
        if category != "passed"
    }
    canonical["benchmark"] = dict(metadata, kind="canonical-preflight")
    save_report(canonical, output / "canonical", scorecard=output / "scorecard.jsonl")
    if canonical["metrics"]["greedy_pass_at_1"] != 1.0:
        raise RuntimeError("canonical preflight failed; model evaluation blocked; report preserved")
    reports = {"canonical": canonical}
    if checkpoint is not None and tokenizer_path is not None:
        model, tokenizer = load_checkpoint(checkpoint, tokenizer_path)
        options = config or EvaluationConfig()
        report = evaluate(
            tasks,
            checkpoint_generator(model, tokenizer, options),
            config=options,
            checkpoint_hash=file_sha256(checkpoint),
            tokenizer_hash=tokenizer.sha256,
            execute=execute,
        )
        report.update(prompt_protocol(options.prompt_mode))
        report["benchmark"] = dict(
            metadata,
            kind="project-checkpoint",
            prompt_policy=report["protocol"]["prompt_format"],
            prompt_stripped=report["prompt_stripped"],
        )
        save_report(report, output / "checkpoint", scorecard=output / "scorecard.jsonl")
        reports["checkpoint"] = report
    return reports
