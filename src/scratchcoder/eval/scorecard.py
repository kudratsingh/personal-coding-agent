"""The evaluation scorecard: every checkpoint on every suite, with controls beside it.

Inputs are the pipeline's own records: pool results files (``eval-execute``), benchmark
artifacts from GitHub Actions (``tools/benchmark_score.py``), ARC-Easy reports
(``eval-choice``), and constant-answer baselines (``eval-baseline`` for pool dev, the logged
test baseline for pool test, the benchmark artifacts, and the ARC labels).

One row per checkpoint and completions file (or ARC report) carries:

- the score: greedy pass@1, or for ``n`` samples the mean pass rate and pass@n
  (``scratchcoder.eval.metrics.pass_at_k``); ARC-Easy accuracy;
- the constant-answer baseline of the same suite beside it (owner ruling): the mean
  best-constant reward and the share of tasks a constant answer fully passes; for ARC, the
  best fixed answer position and chance;
- pool slices by source and by human- versus model-written tests;
- unresolved attempts and tasks (host failures that outlasted their retries), never folded
  into the score: a task with an unresolved attempt is left out of that row's metrics and
  counted, and the row is marked incomplete;
- GPU-minutes (generation) and sandbox-minutes (execution);
- the checkpoint, tokenizer, task-file and protocol hashes and the precision.

Named checkpoint pairs on the same tasks get a paired task bootstrap
(``scratchcoder.eval.metrics.paired_bootstrap``) with a fixed seed; a pair whose protocols
differ (SFT versus base on the pool) is labelled cross-protocol. The leak flag marks any
checkpoint above 20% greedy pass@1 on HumanEval, treated as a leak until shown otherwise.
It has three values: ``True`` (above the threshold, even on a partial row), ``False`` (a
complete greedy HumanEval row at or below it) and ``None`` (no complete greedy HumanEval
row, so nothing can be said); a checkpoint never evaluated on HumanEval is listed under
``not_evaluated`` rather than reported as clean.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from scratchcoder.common.records import git_state, utc_now
from scratchcoder.eval.baseline import arc_constant_baseline
from scratchcoder.eval.choices import ChoiceTask, load_choices
from scratchcoder.eval.completions import CompletionsHeader
from scratchcoder.eval.executor import read_results
from scratchcoder.eval.metrics import paired_bootstrap, pass_at_k
from scratchcoder.eval.protocols import HUMANEVAL, record_sha256

LEAK_SUITE = HUMANEVAL
LEAK_THRESHOLD = 0.20
DEFAULT_PAIRS = (("sft", "base"), ("rl", "sft"), ("more-sft", "sft"), ("rl", "more-sft"))
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0
SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Run:
    """One scored completions file: per-task resolved outcomes plus identity and costs."""

    checkpoint_sha256: str
    suite: str
    split: str
    header: CompletionsHeader
    source_path: str
    source_sha256: str
    outcomes: dict[str, list[bool]]
    unresolved: dict[str, int]
    meta: dict[str, tuple[str, str]]
    sandbox_seconds: float
    audit_flagged: int
    completions_sha256: str
    constant_baseline: dict[str, Any] | None = None

    @property
    def decoding(self) -> str:
        return "greedy" if self.header.sampling.greedy else "sampled"

    @property
    def samples(self) -> int:
        return self.header.sampling.samples

    def complete_tasks(self) -> dict[str, list[bool]]:
        """Tasks whose every attempt is resolved; only these enter metrics and pairs."""
        return {
            task: outcomes
            for task, outcomes in self.outcomes.items()
            if not self.unresolved.get(task) and len(outcomes) == self.samples
        }


def run_from_results(path: Path) -> Run:
    results = read_results(path)
    header = results.completions_header
    tasks = results.header.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"{path}: the results header lists no tasks")
    meta = {item["task_id"]: (item["source"], item["written_by"]) for item in tasks}
    outcomes: dict[str, list[bool]] = {task: [] for task in meta}
    unresolved = {task: header.sampling.samples for task in meta}
    for (task_id, _), result in results.attempts.items():
        outcomes[task_id].append(result.passed)
        unresolved[task_id] -= 1
    return Run(
        header.checkpoint_sha256,
        header.suite,
        header.split,
        header,
        str(path),
        results.sha256,
        outcomes,
        {task: count for task, count in unresolved.items() if count},
        meta,
        results.sandbox_seconds,
        sum(bool(result.audit_flags) for result in results.attempts.values()),
        str(results.header["completions_sha256"]),
    )


def run_from_benchmark(path: Path) -> Run:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("kind") != "benchmark-results" or report.get("status") != "completed":
        raise ValueError(f"{path}: not a completed benchmark results file")
    header = CompletionsHeader.from_json(report["completions"])
    outcomes: dict[str, list[bool]] = defaultdict(list)
    unresolved: dict[str, int] = defaultdict(int)
    for attempt in report["attempts"]:
        if attempt["resolved"]:
            outcomes[attempt["task_id"]].append(bool(attempt["passed"]))
        else:
            outcomes.setdefault(attempt["task_id"], [])
            unresolved[attempt["task_id"]] += 1
    meta = {task: (header.suite, "human") for task in outcomes}
    return Run(
        header.checkpoint_sha256,
        header.suite,
        header.split,
        header,
        str(path),
        _file_sha256(path),
        dict(outcomes),
        dict(unresolved),
        meta,
        float(report.get("sandbox_seconds", 0.0)),
        int(report["summary"].get("audit_flagged_passes", 0)),
        str(report["completions_sha256"]),
        report.get("constant_baseline"),
    )


def _metrics(run: Run, tasks: Sequence[str]) -> dict[str, Any]:
    complete = run.complete_tasks()
    scored = [task for task in tasks if task in complete]
    if not scored:
        return {"tasks": 0}
    if run.decoding == "greedy":
        return {
            "tasks": len(scored),
            "pass_at_1": sum(complete[t][0] for t in scored) / len(scored),
        }
    n = run.samples
    return {
        "tasks": len(scored),
        "samples": n,
        "mean_pass": sum(sum(complete[t]) / n for t in scored) / len(scored),
        "pass_at_k": {
            str(k): sum(pass_at_k(n, sum(complete[t]), k) for t in scored) / len(scored)
            for k in sorted({1, n})
        },
    }


def _task_scores(run: Run, metric: str) -> dict[str, float]:
    complete = run.complete_tasks()
    if metric == "pass_at_1":
        return {task: float(outcomes[0]) for task, outcomes in complete.items()}
    if metric == "mean_pass":
        return {task: sum(outcomes) / len(outcomes) for task, outcomes in complete.items()}
    n = int(metric.removeprefix("pass_at_"))
    return {task: pass_at_k(len(outcomes), sum(outcomes), n) for task, outcomes in complete.items()}


def _pool_baseline_for(run: Run, baselines: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The baseline of the run's task file, or of the full file whose Pod export it is."""
    for baseline in baselines:
        if run.header.task_file_sha256 in (
            baseline.get("task_file_sha256"),
            baseline.get("pod_export_sha256"),
        ):
            return baseline
    return None


def _baseline_summary(run: Run, baselines: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if run.suite == "pool":
        baseline = _pool_baseline_for(run, baselines)
        if baseline is None:
            return {"available": False, "reason": "no baseline for this task file"}
        summary = baseline["summary"]
        return {
            "available": True,
            "mean_reward": summary["mean_reward"],
            "full_pass_share": summary["full_pass_share"],
            "covered": summary["covered"],
            "uncovered": summary["uncovered"],
            "definition": baseline["definition"],
        }
    if run.constant_baseline is None:
        return {"available": False, "reason": "the benchmark run scored no constant answers"}
    found = run.constant_baseline
    return {
        "available": True,
        "mean_reward": found["mean_reward"],
        "full_pass_share": found["full_pass_share"],
        "covered": found["covered"],
        "uncovered": found["uncovered"],
        "definition": found["definition"],
    }


def _slices(run: Run, baselines: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[str]]] = {"by_source": {}, "by_authorship": {}}
    for task, (source, author) in sorted(run.meta.items()):
        groups["by_source"].setdefault(source, []).append(task)
        groups["by_authorship"].setdefault(author, []).append(task)
    baseline = _pool_baseline_for(run, baselines) if run.suite == "pool" else None
    complete = run.complete_tasks()
    sliced: dict[str, Any] = {}
    for name, members in groups.items():
        sliced[name] = {}
        for key, tasks in members.items():
            entry = _metrics(run, tasks)
            entry["unresolved_tasks"] = sum(1 for task in tasks if task not in complete)
            if baseline is not None:
                summary_key = "by_source" if name == "by_source" else "by_authorship"
                found = baseline["summary"][summary_key].get(key)
                if found is not None:
                    entry["constant_baseline_mean_reward"] = found["mean_reward"]
                    entry["constant_baseline_full_pass_share"] = found["full_pass_share"]
            sliced[name][key] = entry
    return sliced


def _row(name: str, run: Run, baselines: Sequence[dict[str, Any]]) -> dict[str, Any]:
    header = run.header
    tasks = sorted(run.outcomes)
    metrics = _metrics(run, tasks)
    complete = run.complete_tasks()
    row: dict[str, Any] = {
        "checkpoint": name,
        "checkpoint_sha256": run.checkpoint_sha256,
        "suite": run.suite,
        "split": run.split,
        "decoding": run.decoding,
        "sampling": asdict(header.sampling),
        "protocol": header.protocol,
        "protocol_sha256": header.protocol_sha256,
        "tokenizer_sha256": header.tokenizer_sha256,
        "task_file_sha256": header.task_file_sha256,
        "precision": header.precision,
        "device": header.device,
        "completions_sha256": run.completions_sha256,
        "source": run.source_path,
        "source_sha256": run.source_sha256,
        "tasks": len(tasks),
        "excluded_tasks": [dict(item) for item in header.excluded_tasks],
        "scored_tasks": len(complete),
        "unresolved_tasks": len(tasks) - len(complete),
        "unresolved_attempts": sum(run.unresolved.values()),
        "complete": len(complete) == len(tasks),
        "metrics": metrics,
        "constant_baseline": _baseline_summary(run, baselines),
        "gpu_minutes": header.generation_seconds / 60,
        "sandbox_minutes": run.sandbox_seconds / 60,
        "audit_flagged_passes": run.audit_flagged,
    }
    if run.suite == "pool":
        row["slices"] = _slices(run, baselines)
    if run.suite == LEAK_SUITE and run.decoding == "greedy":
        score = metrics.get("pass_at_1")
        if score is not None and score > LEAK_THRESHOLD:
            row["leak_flag"] = True
        else:
            row["leak_flag"] = False if score is not None and row["complete"] else None
    return row


@dataclass
class ArcRun:
    checkpoint_sha256: str
    report: dict[str, Any]
    source_path: str
    source_sha256: str
    correct: dict[str, bool] = field(default_factory=dict)


def _arc_task_hash(tasks: Sequence[ChoiceTask]) -> str:
    ordered = sorted(tasks, key=lambda task: task.id)
    return hashlib.sha256(
        json.dumps([asdict(task) for task in ordered], sort_keys=True).encode()
    ).hexdigest()


def _arc_row(name: str, run: ArcRun, baseline: dict[str, Any] | None) -> dict[str, Any]:
    report = run.report
    rows = report["per_task"]
    return {
        "checkpoint": name,
        "checkpoint_sha256": run.checkpoint_sha256,
        "suite": "arc-easy",
        "split": "test",
        "decoding": "label-likelihood",
        "protocol": report["configuration"]["scoring"],
        "protocol_sha256": record_sha256(report["configuration"]),
        "tokenizer_sha256": report["tokenizer_hash"],
        "task_file_sha256": report["task_hash"],
        "precision": report.get("precision", "float32 (eval-choice default)"),
        "source": run.source_path,
        "source_sha256": run.source_sha256,
        "tasks": len(rows),
        "context_limit_tasks": sum(row["status"] == "context_limit" for row in rows),
        "metrics": {"accuracy": report["metrics"]["accuracy"]},
        "constant_baseline": (
            {"available": False, "reason": "no ARC task file given"}
            if baseline is None
            else {
                "available": True,
                "best_position": baseline["best_position"],
                "best_position_accuracy": baseline["best_position_accuracy"],
                "chance": baseline["chance"],
            }
        ),
        "gpu_minutes": None,
        "sandbox_minutes": 0.0,
    }


def _direction(lower: float, upper: float) -> str:
    if lower > 0:
        return "first higher"
    if upper < 0:
        return "second higher"
    return "interval includes 0"


def _compare(
    first: str,
    second: str,
    after: Mapping[str, float],
    before: Mapping[str, float],
    *,
    suite: str,
    split: str,
    metric: str,
    cross_protocol: bool,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    shared = sorted(set(after) & set(before))
    excluded = len(set(after) ^ set(before))
    entry: dict[str, Any] = {
        "pair": f"{first} vs {second}",
        "first": first,
        "second": second,
        "suite": suite,
        "split": split,
        "metric": metric,
        "cross_protocol": cross_protocol,
        "tasks": len(shared),
        "excluded_tasks": excluded,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }
    if not shared:
        entry["difference"] = None
        return entry
    interval = paired_bootstrap(
        [before[task] for task in shared],
        [after[task] for task in shared],
        samples=samples,
        seed=seed,
    )
    entry.update(
        first_score=sum(after[task] for task in shared) / len(shared),
        second_score=sum(before[task] for task in shared) / len(shared),
        difference=interval.mean,
        lower=interval.lower,
        upper=interval.upper,
        confidence=interval.confidence,
        direction=_direction(interval.lower, interval.upper),
    )
    return entry


def build_scorecard(
    *,
    checkpoints: Mapping[str, str],
    results: Sequence[Path] = (),
    benchmarks: Sequence[Path] = (),
    arc_reports: Sequence[Path] = (),
    arc_tasks: Path | None = None,
    pool_baselines: Sequence[Path] = (),
    pairs: Sequence[tuple[str, str]] | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Build the scorecard. ``checkpoints`` maps names (``base``, ``sft``, ...) to SHA-256."""
    by_sha = {sha: name for name, sha in checkpoints.items()}
    if len(by_sha) != len(checkpoints):
        raise ValueError("two checkpoint names share one SHA-256")

    def name_of(sha: str, source: object) -> str:
        if sha not in by_sha:
            raise ValueError(f"{source}: checkpoint {sha[:12]} is not named with --checkpoint")
        return by_sha[sha]

    baselines = [json.loads(path.read_text(encoding="utf-8")) for path in pool_baselines]
    for path, baseline in zip(pool_baselines, baselines, strict=True):
        if baseline.get("kind") != "constant-baseline" or baseline.get("suite") != "pool":
            raise ValueError(f"{path}: not a pool constant-baseline report")
    runs = [run_from_results(path) for path in results] + [
        run_from_benchmark(path) for path in benchmarks
    ]
    seen: set[tuple[str, str, str, str]] = set()
    rows = []
    for run in runs:
        name = name_of(run.checkpoint_sha256, run.source_path)
        identity = (name, run.suite, run.split, run.decoding)
        if identity in seen:
            raise ValueError(f"two {run.decoding} inputs for {name} on {run.suite} {run.split}")
        seen.add(identity)
        rows.append((name, run, _row(name, run, baselines)))
    arc_baseline = None
    arc_hash = None
    if arc_tasks is not None:
        choice_tasks = list(load_choices(arc_tasks)) if arc_tasks.is_file() else _arc_dir(arc_tasks)
        arc_baseline = arc_constant_baseline(choice_tasks)
        arc_hash = _arc_task_hash(choice_tasks)
    arc_runs: dict[str, ArcRun] = {}
    arc_rows = []
    for path in arc_reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("suite") != "multiple-choice" or report.get("status") != "completed":
            raise ValueError(f"{path}: not a completed eval-choice report")
        if arc_hash is not None and report["task_hash"] != arc_hash:
            raise ValueError(f"{path}: the ARC report was scored on other tasks")
        name = name_of(report["checkpoint_hash"], path)
        if name in arc_runs:
            raise ValueError(f"two ARC reports for {name}")
        arc = ArcRun(
            report["checkpoint_hash"],
            report,
            str(path),
            _file_sha256(path),
            {row["id"]: bool(row["correct"]) for row in report["per_task"]},
        )
        arc_runs[name] = arc
        arc_rows.append(_arc_row(name, arc, arc_baseline))
    comparisons = []
    named_pairs = list(DEFAULT_PAIRS if pairs is None else pairs)
    for first, second in named_pairs:
        for suite_key in sorted({(run.suite, run.split, run.decoding) for _, run, _ in rows}):
            left = [run for name, run, _ in rows if name == first and _key(run) == suite_key]
            right = [run for name, run, _ in rows if name == second and _key(run) == suite_key]
            if not left or not right:
                continue
            a, b = left[0], right[0]
            if a.header.task_file_sha256 != b.header.task_file_sha256:
                raise ValueError(f"{first} and {second} were scored on different task files")
            metrics = ["pass_at_1"] if a.decoding == "greedy" else ["mean_pass"]
            if a.decoding == "sampled" and a.samples == b.samples:
                metrics.append(f"pass_at_{a.samples}")
            for metric in metrics:
                comparisons.append(
                    _compare(
                        first,
                        second,
                        _task_scores(a, metric),
                        _task_scores(b, metric),
                        suite=a.suite,
                        split=a.split,
                        metric=metric,
                        cross_protocol=a.header.protocol != b.header.protocol,
                        samples=bootstrap_samples,
                        seed=bootstrap_seed,
                    )
                )
        if first in arc_runs and second in arc_runs:
            comparisons.append(
                _compare(
                    first,
                    second,
                    {task: float(value) for task, value in arc_runs[first].correct.items()},
                    {task: float(value) for task, value in arc_runs[second].correct.items()},
                    suite="arc-easy",
                    split="test",
                    metric="accuracy",
                    cross_protocol=False,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
            )
    all_rows = [row for _, _, row in rows] + arc_rows
    costs: dict[tuple[str, str, str], dict[str, float]] = {}
    for item in all_rows:
        total = costs.setdefault(
            (item["checkpoint"], item["suite"], item["split"]),
            {"gpu_minutes": 0.0, "sandbox_minutes": 0.0},
        )
        total["gpu_minutes"] += item["gpu_minutes"] or 0.0
        total["sandbox_minutes"] += item["sandbox_minutes"] or 0.0
    leak: dict[str, bool | None] = {}
    for name in checkpoints:
        marks = [
            row["leak_flag"] for row in all_rows if row["checkpoint"] == name and "leak_flag" in row
        ]
        if any(mark is True for mark in marks):
            leak[name] = True
        else:
            leak[name] = False if any(mark is False for mark in marks) else None
    commit, dirty = git_state()
    return {
        "kind": "scorecard",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "checkpoints": dict(checkpoints),
        "rows": sorted(all_rows, key=lambda row: (row["suite"], row["split"], row["checkpoint"])),
        "comparisons": comparisons,
        "costs": [
            {"checkpoint": name, "suite": suite, "split": split, **total}
            for (name, suite, split), total in sorted(costs.items())
        ],
        "leak": {
            "rule": f"greedy pass@1 above {LEAK_THRESHOLD:.0%} on {LEAK_SUITE} is a leak until "
            "shown otherwise",
            "flags": leak,
            "any": any(flag is True for flag in leak.values()),
            "not_evaluated": [name for name, flag in leak.items() if flag is None],
        },
        "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed, "confidence": 0.95},
        "inputs": [
            {"path": str(path), "sha256": _file_sha256(path)}
            for path in [*results, *benchmarks, *arc_reports, *pool_baselines]
            + ([arc_tasks] if arc_tasks is not None and arc_tasks.is_file() else [])
        ],
    }


def _key(run: Run) -> tuple[str, str, str]:
    return run.suite, run.split, run.decoding


def _arc_dir(directory: Path) -> list[ChoiceTask]:
    from scratchcoder.data.arc import read_arc_easy

    tasks, _ = read_arc_easy(directory)
    return tasks


def _percent(value: object) -> str:
    return "n/a" if not isinstance(value, int | float) else f"{100 * value:.2f}%"


def render_markdown(scorecard: Mapping[str, Any]) -> str:
    """A readable summary of a scorecard: one table per suite, then the comparisons."""
    lines = ["# Evaluation scorecard", ""]
    leak = scorecard["leak"]
    lines.append(f"Leak rule: {leak['rule']}. Flagged: {leak['any']}.")
    if leak["not_evaluated"]:
        lines.append(
            "Leak check not possible (no complete greedy HumanEval row): "
            + ", ".join(leak["not_evaluated"])
            + "."
        )
    lines.append("")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scorecard["rows"]:
        groups[(row["suite"], row["split"])].append(row)
    for (suite, split), rows in sorted(groups.items()):
        lines += [f"## {suite} ({split})", ""]
        lines.append(
            "| Checkpoint | Decoding | Protocol | Score | Constant baseline (mean / full) "
            "| Unresolved | GPU min | Sandbox min | Checkpoint / tokenizer / tasks / protocol "
            "| Precision |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for row in rows:
            metrics = row["metrics"]
            if "pass_at_1" in metrics:
                score = f"pass@1 {_percent(metrics['pass_at_1'])}"
            elif "mean_pass" in metrics:
                k = str(metrics["samples"])
                score = (
                    f"mean {_percent(metrics['mean_pass'])}, "
                    f"pass@{k} {_percent(metrics['pass_at_k'][k])}"
                )
            elif "accuracy" in metrics:
                score = f"accuracy {_percent(metrics['accuracy'])}"
            else:
                score = "n/a"
            baseline = row["constant_baseline"]
            if not baseline.get("available"):
                control = "n/a"
            elif "chance" in baseline:
                control = (
                    f"position {_percent(baseline['best_position_accuracy'])}, "
                    f"chance {_percent(baseline['chance'])}"
                )
            else:
                control = (
                    f"{_percent(baseline['mean_reward'])} / {_percent(baseline['full_pass_share'])}"
                )
                if baseline.get("uncovered"):
                    control += f" ({baseline['uncovered']} uncovered)"
            unresolved = row.get("unresolved_attempts", 0)
            gpu = row.get("gpu_minutes")
            hashes = " / ".join(
                str(row.get(key) or "")[:12]
                for key in ("checkpoint_sha256", "tokenizer_sha256", "task_file_sha256")
            )
            hashes += " / " + str(row.get("protocol_sha256", ""))[:12]
            leak_mark = " (LEAK FLAG)" if row.get("leak_flag") is True else ""
            if "leak_flag" in row and row["leak_flag"] is None:
                leak_mark = " (leak check inconclusive)"
            lines.append(
                f"| {row['checkpoint']}{leak_mark} | {row['decoding']} | {row['protocol']} | "
                f"{score} | {control} | {unresolved} | "
                f"{'n/a' if gpu is None else f'{gpu:.1f}'} | {row['sandbox_minutes']:.1f} | "
                f"{hashes} | {row['precision']} |"
            )
        lines.append("")
    if scorecard["comparisons"]:
        lines += ["## Paired comparisons", ""]
        lines.append(
            "| Pair | Suite | Metric | Difference | 95% interval | Direction | Tasks | Note |"
        )
        lines.append("| --- | --- | --- | ---: | --- | --- | ---: | --- |")
        for item in scorecard["comparisons"]:
            if item.get("difference") is None:
                interval, difference, direction = "n/a", "n/a", "no shared tasks"
            else:
                difference = f"{100 * item['difference']:+.2f} pts"
                interval = f"[{100 * item['lower']:+.2f}, {100 * item['upper']:+.2f}]"
                direction = item["direction"]
            note = "cross-protocol" if item["cross_protocol"] else ""
            lines.append(
                f"| {item['pair']} | {item['suite']} {item['split']} | {item['metric']} | "
                f"{difference} | {interval} | {direction} | {item['tasks']} | {note} |"
            )
        lines.append("")
    return "\n".join(lines)
