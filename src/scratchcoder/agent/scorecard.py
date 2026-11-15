"""The development scorecard: per case and condition, what was solved, how, and at what cost.

Counts only; no rate is reported without its denominator. Unscored instances (host failures
that outlasted the retries) are counted separately and never as failures. Oracle rows check
the harness, not the model, and are reported apart from model rows.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from scratchcoder.agent.loop import FAILURE_CLASSES, InstanceResult


def _cell(results: Sequence[InstanceResult]) -> dict[str, Any]:
    scored = [result for result in results if result.scored]
    return {
        "instances": len(results),
        "scored": len(scored),
        "solved": sum(result.solved for result in scored),
        "failure_classes": dict(
            sorted(Counter(r.failure_class for r in scored if r.failure_class).items())
        ),
        "unscored": len(results) - len(scored),
        "turns_used": sum(len(result.turns) for result in results),
        "edits_applied": sum(result.edits_applied for result in results),
        "generated_tokens": sum(result.generated_tokens for result in results),
        "generation_seconds": round(sum(result.generation_seconds for result in results), 3),
        "cached_generation_seconds": round(
            sum(result.cached_generation_seconds for result in results), 3
        ),
        "attributed_generation_seconds": round(
            sum(r.generation_seconds + r.cached_generation_seconds for r in results), 3
        ),
        "sandbox_calls": sum(result.sandbox_calls for result in results),
        "sandbox_seconds": round(sum(result.sandbox_seconds for result in results), 3),
        "wall_seconds": round(sum(result.wall_seconds for result in results), 3),
        "flagged_solved": sum(result.solved and bool(result.audit_flags) for result in results),
    }


def build(results: Sequence[InstanceResult], identity: dict[str, Any]) -> dict[str, Any]:
    cases = sorted({result.case_id for result in results})
    groups: dict[str, dict[str, Any]] = {}
    for policy in sorted({result.policy for result in results}):
        for condition in sorted({r.condition for r in results if r.policy == policy}):
            label = f"{policy}/{condition}"
            chosen = [r for r in results if r.policy == policy and r.condition == condition]
            sampled = [r for r in chosen if r.decoding != "greedy"]
            greedy = [r for r in chosen if r.decoding == "greedy"]
            groups[label] = {
                "all": _cell(chosen),
                "sampled": _cell(sampled),
                "greedy": _cell(greedy),
                "per_case": {
                    case: {
                        "sampled": _cell([r for r in sampled if r.case_id == case]),
                        "greedy": _cell([r for r in greedy if r.case_id == case]),
                    }
                    for case in cases
                    if any(r.case_id == case for r in chosen)
                },
            }
    return {
        "schema": "agent-dev-scorecard-v1",
        "identity": identity,
        "failure_class_vocabulary": list(FAILURE_CLASSES),
        "groups": groups,
        "instances": [result.summary() for result in results],
    }


def markdown(card: dict[str, Any]) -> str:
    """A compact table for review: one row per case per policy/condition."""
    lines = [
        "# Agent development scorecard",
        "",
        "Dev cases only. Solved = public tests, hidden correctness checks and preservation "
        "checks all `passed` (supervisor status). Sampled cells count seeds; greedy is one "
        "instance. Generation seconds are net (time spent); reused generations add their "
        "original time in brackets, so net + cached is comparable across conditions.",
        "",
        "| Policy/condition | Case | Sampled solved | Greedy solved | Failure classes "
        "(scored, unsolved) | Turns | Gen tokens | Gen s net (+ cached) | Sandbox calls | "
        "Sandbox s |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, group in card["groups"].items():
        for case, cells in group["per_case"].items():
            sampled, greedy = cells["sampled"], cells["greedy"]
            classes = Counter(sampled["failure_classes"]) + Counter(greedy["failure_classes"])
            total = Counter(
                {
                    key: sampled[key] + greedy[key]
                    for key in (
                        "turns_used",
                        "generated_tokens",
                        "sandbox_calls",
                    )
                }
            )
            gen = sampled["generation_seconds"] + greedy["generation_seconds"]
            cached = sampled["cached_generation_seconds"] + greedy["cached_generation_seconds"]
            box = sampled["sandbox_seconds"] + greedy["sandbox_seconds"]
            unscored = sampled["unscored"] + greedy["unscored"]
            lines.append(
                f"| {label} | {case} | {sampled['solved']}/{sampled['scored']} | "
                f"{greedy['solved']}/{greedy['scored']} | "
                + (", ".join(f"{k} {v}" for k, v in sorted(classes.items())) or "-")
                + (f" (unscored {unscored})" if unscored else "")
                + f" | {total['turns_used']} | {total['generated_tokens']} | "
                f"{gen:.1f} (+{cached:.1f}) | "
                f"{total['sandbox_calls']} | {box:.1f} |"
            )
    return "\n".join(lines) + "\n"
