"""Read-only reward-signal report of an RL screening record; it never executes code.

``scratchcoder rl-screen`` keeps a candidate when its 8 sampled rewards are not all equal
(among other rules). Unequal once is a weak promise: most rewards are partial credit and many
groups differ in a single sample. This tool reads the screening record (``rl-screening-v1``)
and, optionally, the screening attempts file, and reports by source:

- the eligible rate (kept or over the cap) with a Wilson 95% interval;
- the zero / partial / full reward mix of the eligible tasks' attempts;
- how many samples of each eligible group differ from its modal reward;
- each eligible task's best sampled reward against its best constant answer's;
- the split-half replication of "unequal" among the parseable tasks below the constant-answer
  threshold (samples 0-3 against 4-7);
- a **forecast** of the share of fresh groups that would be informative at the screening
  policy: a beta-binomial fit to P(reward > 0) over those tasks (empirical Bayes), its
  posterior predictive for a fresh group of the same size mixing zero and nonzero rewards,
  checked against the observed split-half replication; and, for comparison, the naive figure
  from resampling each task's 8 observed rewards;
- from the attempts file, if given: the share of attempts without a program and at the token
  cap. Only the JSON fields are read; ``code`` is tested for ``None`` and never run.

It also reports how many more candidates would reach the floor **if** they were eligible at the
observed rate. That figure is an **extrapolation** from one screen, not a requirement or a
guarantee. Screening more tasks from a source already accepted for screening is a different
decision from admitting a new source or lifting a source's exclusion (the AceCode exclusion
stays in force); this tool decides neither.

Usage: ``python -m tools.rl_reward_signal --screening <record.json> [--attempts <jsonl>]
--output <new json>``. The output must be a new file outside the screening record's
directory, so a recorded run is never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

FORMAT = "rl-reward-signal-v1"
SCREENING_FORMAT = "rl-screening-v1"
ALL = "all"
Z95 = 1.959963984540054
FORECAST_LABEL = (
    "forecast from one screen at the screening checkpoint's policy, under a beta-binomial "
    "model of P(reward > 0); not a measurement of fresh groups during training, where the "
    "policy changes"
)
EXTRAPOLATION_LABEL = (
    "extrapolation from the observed eligibility rate of one screen; not a requirement and "
    "not a guarantee: new candidates may be eligible at another rate"
)
ALL_SCOPE = "every source together: new candidates eligible at the pooled rate"
SOURCE_SCOPE = (
    "this source alone: as if every new candidate came from it, at its own rate; the whole "
    "shortfall of the floor, not this source's share of it"
)
SOURCE_NOTE = (
    "screening more tasks from a source already accepted for screening is a different "
    "decision from admitting a new source or lifting a source's exclusion; the AceCode "
    "exclusion stays in force, and this report decides neither"
)


# Reading --------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_screening(path: Path) -> dict[str, Any]:
    """The screening record, checked for the fields this report reads."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("format") != SCREENING_FORMAT:
        raise ValueError(f"{path} is not an {SCREENING_FORMAT} record")
    rule = record.get("rule")
    tasks = record.get("tasks")
    if not isinstance(rule, dict) or not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{path} has no rule or no tasks")
    samples = int(rule["samples"])
    seen: set[str] = set()
    for row in tasks:
        rewards = row.get("rewards")
        if not isinstance(rewards, list) or len(rewards) != samples:
            raise ValueError(f"task {row.get('id')!r} does not have {samples} rewards")
        if not all(isinstance(value, int | float) and math.isfinite(value) for value in rewards):
            raise ValueError(f"task {row.get('id')!r} has a non-finite reward")
        if row["id"] in seen:
            raise ValueError(f"task {row['id']!r} appears twice")
        seen.add(row["id"])
    if not set(record.get("eligible", [])) <= seen:
        raise ValueError("the eligible list names a task the record does not hold")
    return record


def load_attempts(path: Path, *, task_ids: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The attempts file's header and rows (task, sample, token count, whether a program was
    extracted). The program text is never kept and never run."""
    header: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not an object")
            if value.get("kind") == "header":
                if header is not None or rows:
                    raise ValueError(f"{path}:{number}: the header must come first, once")
                header = value
                continue
            task_id, sample = value.get("task_id"), value.get("sample")
            if task_id not in task_ids:
                raise ValueError(f"{path}:{number}: an attempt for a task the record lacks")
            tokens = value.get("completion_tokens")
            if not isinstance(sample, int) or not isinstance(tokens, int) or tokens < 0:
                raise ValueError(f"{path}:{number}: malformed attempt")
            rows.append(
                {
                    "task_id": task_id,
                    "sample": sample,
                    "completion_tokens": tokens,
                    "has_program": value.get("code") is not None,
                }
            )
    if header is None:
        raise ValueError(f"{path} has no header")
    return header, rows


# Statistics -----------------------------------------------------------------------------


def wilson(successes: int, trials: int, z: float = Z95) -> tuple[float, float] | None:
    """The Wilson score interval of a binomial proportion; ``None`` for no trials."""
    if trials <= 0:
        return None
    p = successes / trials
    centre = p + z * z / (2 * trials)
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    scale = 1 + z * z / trials
    return ((centre - spread) / scale, (centre + spread) / scale)


def modal_off(rewards: Sequence[float]) -> int:
    """Samples whose reward differs from the group's most common reward."""
    return len(rewards) - Counter(rewards).most_common(1)[0][1]


def reward_mix(rewards: Iterable[float]) -> dict[str, Any]:
    values = list(rewards)
    zero = sum(value == 0 for value in values)
    full = sum(value >= 1 for value in values)
    partial = len(values) - zero - full
    total = len(values)
    return {
        "attempts": total,
        "zero": zero,
        "partial": partial,
        "full": full,
        "zero_share": zero / total if total else None,
        "partial_share": partial / total if total else None,
        "full_share": full / total if total else None,
    }


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_binomial_fit(counts: Sequence[int], trials: int) -> tuple[float, float]:
    """Maximum-likelihood beta prior (a, b) on a per-task success probability from each
    task's successes out of ``trials``: a log-spaced grid, refined three times."""
    if not counts:
        raise ValueError("the fit needs at least one task")
    histogram = Counter(counts)

    def loglik(a: float, b: float) -> float:
        base = _log_beta(a, b)
        return math.fsum(
            number * (_log_beta(k + a, trials - k + b) - base) for k, number in histogram.items()
        )

    low_a, high_a, low_b, high_b = math.log(1e-3), math.log(1e3), math.log(1e-3), math.log(1e3)
    best = (0.0, 0.0)
    for _ in range(4):
        grid_a = [low_a + (high_a - low_a) * i / 40 for i in range(41)]
        grid_b = [low_b + (high_b - low_b) * i / 40 for i in range(41)]
        score, best = max(
            (loglik(math.exp(x), math.exp(y)), (x, y)) for x in grid_a for y in grid_b
        )
        step_a, step_b = (high_a - low_a) / 40, (high_b - low_b) / 40
        low_a, high_a = best[0] - 2 * step_a, best[0] + 2 * step_a
        low_b, high_b = best[1] - 2 * step_b, best[1] + 2 * step_b
    return math.exp(best[0]), math.exp(best[1])


def mixed_probability(a: float, b: float, successes: int, trials: int, fresh: int) -> float:
    """Posterior predictive probability that ``fresh`` new draws hold both a success and a
    failure, after ``successes`` of ``trials`` under a Beta(a, b) prior."""
    pa, pb = a + successes, b + trials - successes
    none = math.exp(_log_beta(pa, pb + fresh) - _log_beta(pa, pb))
    every = math.exp(_log_beta(pa + fresh, pb) - _log_beta(pa, pb))
    return max(0.0, 1.0 - none - every)


def resample_unequal(rewards: Sequence[float], fresh: int) -> float:
    """The chance a group of ``fresh`` draws from the observed rewards is not all equal."""
    counts = Counter(rewards)
    return 1.0 - math.fsum((n / len(rewards)) ** fresh for n in counts.values())


def split_half(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Of the groups unequal in their first half, the share also unequal in the second, and
    the reverse."""
    half = len(rows[0]["rewards"]) // 2 if rows else 0
    first = [row for row in rows if len(set(row["rewards"][:half])) > 1]
    both = [row for row in first if len(set(row["rewards"][half:])) > 1]
    second = [row for row in rows if len(set(row["rewards"][half:])) > 1]
    back = [row for row in second if len(set(row["rewards"][:half])) > 1]
    return {
        "half": half,
        "unequal_first_half": len(first),
        "also_unequal_second_half": len(both),
        "replication": len(both) / len(first) if first else None,
        "unequal_second_half": len(second),
        "also_unequal_first_half": len(back),
        "reverse_replication": len(back) / len(second) if second else None,
    }


# The report -----------------------------------------------------------------------------


def below_threshold(row: Mapping[str, Any], threshold: float) -> bool:
    baseline = row.get("baseline")
    return bool(row.get("parseable")) and baseline is not None and baseline < threshold


def source_report(
    rows: Sequence[Mapping[str, Any]],
    eligible: set[str],
    *,
    threshold: float,
    prior: tuple[float, float],
    samples: int,
    floor: int,
) -> dict[str, Any]:
    """One source's (or every source's) statistics."""
    candidates = len(rows)
    chosen = [row for row in rows if row["id"] in eligible]
    pool = [row for row in rows if below_threshold(row, threshold)]
    reasons = Counter(row.get("reason") or "kept" for row in rows)
    count = len(chosen)
    rate = count / candidates if candidates else None
    interval = wilson(count, candidates)
    best = Counter(
        "below"
        if max(row["rewards"]) < row["baseline"] - 1e-12
        else "equal"
        if abs(max(row["rewards"]) - row["baseline"]) <= 1e-12
        else "above"
        for row in chosen
    )
    a, b = prior
    mixed = [
        mixed_probability(a, b, sum(value > 0 for value in row["rewards"]), samples, samples)
        for row in chosen
    ]
    naive = [resample_unequal(row["rewards"], samples) for row in chosen]
    deviations = [statistics.pstdev(row["rewards"]) for row in chosen]
    report: dict[str, Any] = {
        "candidates": candidates,
        "reasons": dict(sorted(reasons.items())),
        "parseable_below_threshold": len(pool),
        "eligible": count,
        "eligible_rate": rate,
        "eligible_rate_wilson_95": interval,
        "eligible_rate_among_parseable_below_threshold": count / len(pool) if pool else None,
        "reward_mix_eligible": reward_mix(r for row in chosen for r in row["rewards"]),
        "reward_mix_parseable_below_threshold": reward_mix(
            r for row in pool for r in row["rewards"]
        ),
        "eligible_tasks_with_a_full_reward": sum(max(row["rewards"]) >= 1 for row in chosen),
        "samples_off_modal_reward": {
            str(k): v
            for k, v in sorted(Counter(modal_off(row["rewards"]) for row in chosen).items())
        },
        "median_within_group_std": statistics.median(deviations) if deviations else None,
        "best_sampled_vs_constant_baseline": {
            key: best.get(key, 0) for key in ("below", "equal", "above")
        },
        "attempts_above_own_baseline": sum(
            value > row["baseline"] + 1e-12 for row in chosen for value in row["rewards"]
        ),
        "split_half": split_half(pool),
        "informative_fresh_groups": {
            "forecast": FORECAST_LABEL,
            "beta_binomial_mean": statistics.fmean(mixed) if mixed else None,
            "beta_binomial_median": statistics.median(mixed) if mixed else None,
            "naive_resample_mean": statistics.fmean(naive) if naive else None,
            "expected_informative_tasks": math.fsum(mixed) if mixed else 0.0,
        },
    }
    report["extrapolation"] = extrapolate(count, candidates, floor=floor)
    return report


def extrapolate(eligible: int, candidates: int, *, floor: int) -> dict[str, Any]:
    """More candidates that would reach ``floor`` eligible **if** they were eligible at the
    observed rate (and its Wilson interval). An extrapolation, never a requirement."""
    short = max(0, floor - eligible)
    interval = wilson(eligible, candidates)
    if eligible == 0 or interval is None:
        return {"label": EXTRAPOLATION_LABEL, "short_of_floor": short, "more_candidates": None}

    def needed(rate: float) -> int | None:
        return None if rate <= 0 else math.ceil(short / rate)

    return {
        "label": EXTRAPOLATION_LABEL,
        "floor": floor,
        "short_of_floor": short,
        "more_candidates_at_observed_rate": needed(eligible / candidates),
        "more_candidates_at_wilson_95": [needed(interval[1]), needed(interval[0])],
    }


def attempts_report(
    header: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
    eligible: set[str],
) -> dict[str, Any]:
    """No-program and at-cap shares, by source, from the attempts' JSON fields only."""
    cap = int(header["max_new_tokens"])
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        source = str(by_id[row["task_id"]]["source"])
        groups.setdefault(source, []).append(row)
        groups.setdefault(ALL, []).append(row)
        if row["task_id"] in eligible:
            groups.setdefault(f"{ALL} eligible", []).append(row)

    def summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        total = len(items)
        return {
            "attempts": total,
            "no_program_share": sum(not item["has_program"] for item in items) / total,
            "at_token_cap_share": sum(item["completion_tokens"] >= cap for item in items) / total,
            "mean_completion_tokens": statistics.fmean(item["completion_tokens"] for item in items),
        }

    return {
        "max_new_tokens": cap,
        "header_attempts": header.get("attempts"),
        "read_fields": ["task_id", "sample", "completion_tokens", "code is None"],
        "executed": False,
        "by_source": {name: summary(items) for name, items in sorted(groups.items())},
    }


def analyze(
    record: Mapping[str, Any],
    *,
    attempts: tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """The reward-signal report of a screening record (and its attempts, if given)."""
    rule = record["rule"]
    threshold, samples, floor = float(rule["threshold"]), int(rule["samples"]), int(rule["floor"])
    tasks: list[Mapping[str, Any]] = record["tasks"]
    eligible = set(record["eligible"])
    pool = [row for row in tasks if below_threshold(row, threshold)]
    counts = [sum(value > 0 for value in row["rewards"]) for row in pool]
    prior = beta_binomial_fit(counts, samples)
    half = samples // 2
    halves = [row for row in pool if 0 < sum(value > 0 for value in row["rewards"][:half]) < half]
    predicted = (
        statistics.fmean(
            mixed_probability(
                prior[0], prior[1], sum(value > 0 for value in row["rewards"][:half]), half, half
            )
            for row in halves
        )
        if halves
        else None
    )
    observed = (
        sum(0 < sum(value > 0 for value in row["rewards"][half:]) < half for row in halves)
        / len(halves)
        if halves
        else None
    )
    options: dict[str, Any] = {
        "threshold": threshold,
        "prior": prior,
        "samples": samples,
        "floor": floor,
    }
    sources: dict[str, Any] = {ALL: source_report(tasks, eligible, **options)}
    sources[ALL]["extrapolation"]["scope"] = ALL_SCOPE
    for name in sorted({str(row["source"]) for row in tasks}):
        rows = [row for row in tasks if row["source"] == name]
        sources[name] = source_report(rows, eligible, **options)
        sources[name]["extrapolation"]["scope"] = SOURCE_SCOPE
    report: dict[str, Any] = {
        "format": FORMAT,
        "rule": dict(rule),
        "outcome": record.get("outcome"),
        "kept": None if record.get("kept") is None else len(record["kept"]),
        "eligible": len(eligible),
        "model": {
            "kind": "beta-binomial on P(reward > 0), fitted by maximum likelihood over the "
            "parseable tasks below the constant-answer threshold",
            "tasks": len(pool),
            "a": prior[0],
            "b": prior[1],
            "mean": prior[0] / (prior[0] + prior[1]),
            "check": {
                "what": "zero/nonzero mixing in samples 4-7 given mixing in samples 0-3",
                "selected": len(halves),
                "predicted": predicted,
                "observed": observed,
            },
            "note": "a fresh group is counted informative here when it mixes zero and nonzero "
            "rewards; a group of unequal nonzero rewards is informative too, so this is a "
            "lower bound when such groups exist",
        },
        "sources": sources,
        "notes": {
            "forecast": FORECAST_LABEL,
            "extrapolation": EXTRAPOLATION_LABEL,
            "sources": SOURCE_NOTE,
            "executed_code": False,
        },
    }
    if attempts is not None:
        header, attempt_rows = attempts
        by_id = {str(row["id"]): row for row in tasks}
        report["attempts"] = attempts_report(header, attempt_rows, by_id, eligible)
    return report


def markdown(report: Mapping[str, Any]) -> str:
    """The report's main figures as Markdown tables."""
    lines = [
        "| Source | Candidates | Eligible (rate, Wilson 95%) | Zero / partial / full (eligible "
        "attempts) | Best vs constant: below / equal / above | Informative fresh groups "
        "(forecast; naive) |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for name, row in report["sources"].items():
        mix = row["reward_mix_eligible"]
        interval = row["eligible_rate_wilson_95"]
        best = row["best_sampled_vs_constant_baseline"]
        groups = row["informative_fresh_groups"]
        rate = (
            f"{row['eligible']} ({100 * row['eligible_rate']:.1f}%, "
            f"{100 * interval[0]:.1f}-{100 * interval[1]:.1f}%)"
        )
        forecast = (
            "-"
            if groups["beta_binomial_mean"] is None
            else f"{100 * groups['beta_binomial_mean']:.1f}%; "
            f"{100 * groups['naive_resample_mean']:.1f}%"
        )
        lines.append(
            f"| {name} | {row['candidates']} | {rate} | {mix['zero']} / {mix['partial']} / "
            f"{mix['full']} | {best['below']} / {best['equal']} / {best['above']} | {forecast} |"
        )
    lines += [
        "",
        "| Source | Short of floor | More candidates if eligible at the observed rate "
        "(Wilson 95%) | Scope |",
        "| --- | ---: | --- | --- |",
    ]
    for name, row in report["sources"].items():
        extra = row["extrapolation"]
        if extra.get("more_candidates_at_observed_rate") is None:
            lines.append(
                f"| {name} | {extra['short_of_floor']} | no eligible task observed | "
                f"{extra['scope']} |"
            )
            continue
        low, high = extra["more_candidates_at_wilson_95"]
        lines.append(
            f"| {name} | {extra['short_of_floor']} | "
            f"{extra['more_candidates_at_observed_rate']:,} ({low:,}-{high:,}) | "
            f"{extra['scope']} |"
        )
    notes = report["notes"]
    lines += [
        "",
        f"Informative fresh groups: {notes['forecast']}.",
        f"More candidates: {notes['extrapolation']}; {notes['sources']}.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--screening", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, help="the screening attempts (read only)")
    parser.add_argument("--output", type=Path, help="a new JSON file for the report")
    args = parser.parse_args(argv)
    if args.output is not None:
        if args.output.exists() or args.output.is_symlink():
            parser.error("the output must be a new file")
        if args.output.resolve().parent == args.screening.resolve().parent:
            parser.error("write the report outside the screening record's directory")
    try:
        record = load_screening(args.screening)
        attempts = None
        if args.attempts is not None:
            attempts = load_attempts(
                args.attempts, task_ids={str(r["id"]) for r in record["tasks"]}
            )
        report = analyze(record, attempts=attempts)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    report["inputs"] = {
        "screening_sha256": sha256_file(args.screening),
        "attempts_sha256": None if args.attempts is None else sha256_file(args.attempts),
    }
    if args.output is not None:
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(markdown(report), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
