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

GENERATION_STAGES = (
    "generations",
    "fresh_generations",
    "closed_block",
    "recovered",
    "parseable",
    "one_shown_function",
    "target_function",
    "applied_edit",
    "public_passed",
)
INSTANCE_STAGES = (
    "instances",
    "closed_or_recovered",
    "parseable",
    "one_shown_function",
    "target_function",
    "applied_edit",
    "public_passed",
    "solved",
)


def funnel(results: Sequence[InstanceResult]) -> dict[str, Any]:
    """How far replies got, per generation (every turn; ``fresh_generations`` excludes cache
    reuse) and per instance (any turn reaching the stage; ``solved`` = public, correctness and
    preservation all passed). Recovered blocks are counted apart from closed ones."""
    turns = [turn for result in results for turn in result.turns]
    generations = {
        "generations": len(turns),
        "fresh_generations": sum(not turn.cached for turn in turns),
        "closed_block": sum(turn.closed_block for turn in turns),
        "recovered": sum(turn.recovered for turn in turns),
        "parseable": sum(turn.parseable for turn in turns),
        "one_shown_function": sum(len(turn.shown_defined) == 1 for turn in turns),
        "target_function": sum(turn.target_defined for turn in turns),
        "applied_edit": sum(turn.edit == "applied" for turn in turns),
        "public_passed": sum(turn.feedback == "passed" for turn in turns),
    }
    instances = {
        "instances": len(results),
        "closed_or_recovered": sum(
            any(t.closed_block or t.recovered for t in r.turns) for r in results
        ),
        "parseable": sum(any(t.parseable for t in r.turns) for r in results),
        "one_shown_function": sum(any(len(t.shown_defined) == 1 for t in r.turns) for r in results),
        "target_function": sum(any(t.target_defined for t in r.turns) for r in results),
        "applied_edit": sum(any(t.edit == "applied" for t in r.turns) for r in results),
        "public_passed": sum(r.grade.get("public") == "passed" for r in results),
        "solved": sum(r.scored and r.solved for r in results),
    }
    return {
        "per_generation": generations,
        "per_instance": instances,
        "refusals": dict(sorted(Counter(t.edit for t in turns if t.edit != "applied").items())),
        "name_failures": dict(
            sorted(Counter(t.name_failure for t in turns if t.name_failure).items())
        ),
        "harness_defects_per_generation": dict(
            sorted(Counter(flag for t in turns for flag in t.audit).items())
        ),
        "instances_with_harness_defects": dict(
            sorted(Counter(flag for r in results for flag in r.harness_defects).items())
        ),
    }


def _cell(results: Sequence[InstanceResult]) -> dict[str, Any]:
    scored = [result for result in results if result.scored]
    return {
        "funnel": funnel(results),
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
    lines += [
        "",
        "## Funnel",
        "",
        "Per generation (every turn; fresh excludes reuse), then per instance (any turn).",
        "",
        "| Policy/condition | Level | " + " | ".join(GENERATION_STAGES) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in GENERATION_STAGES) + " |",
    ]
    for label, group in card["groups"].items():
        stages = group["all"]["funnel"]["per_generation"]
        lines.append(
            f"| {label} | generation | "
            + " | ".join(str(stages[s]) for s in GENERATION_STAGES)
            + " |"
        )
    lines += [
        "",
        "| Policy/condition | "
        + " | ".join(INSTANCE_STAGES)
        + " | name failures | harness defects |",
        "| --- | " + " | ".join("---:" for _ in INSTANCE_STAGES) + " | --- | --- |",
    ]
    for label, group in card["groups"].items():
        found = group["all"]["funnel"]
        stages = found["per_instance"]
        names = ", ".join(f"{k} {v}" for k, v in found["name_failures"].items()) or "-"
        defects = ", ".join(f"{k} {v}" for k, v in found["instances_with_harness_defects"].items())
        lines.append(
            f"| {label} | "
            + " | ".join(str(stages[s]) for s in INSTANCE_STAGES)
            + f" | {names} | {defects or '-'} |"
        )
    return "\n".join(lines) + "\n"


COMPARE_GENERATION = (
    "fresh_generations",
    "closed_block",
    "recovered",
    "parseable",
    "one_shown_function",
    "target_function",
    "applied_edit",
    "public_passed",
)


def _inferred_config(identity: dict[str, Any]) -> dict[str, Any]:
    """The configuration of a scorecard written before configurations were recorded: the
    defaults, with the sampled decoding read from its decodings."""
    sampled = [row for row in identity.get("decodings", []) if row.get("label") != "greedy"]
    return {
        "prompt": "harness",
        "sampled_temperature": sampled[0]["temperature"] if sampled else None,
        "sampled_top_p": sampled[0]["top_p"] if sampled else None,
        "recovery": "off",
        "inferred": True,
    }


def comparison(cards: Sequence[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """One row per run and model condition: configuration, funnel, solved and cost. A run
    without funnel data (an earlier scorecard) shows ``None`` for those columns."""
    rows = []
    for name, card in cards:
        identity = card.get("identity", {})
        config = identity.get("config") or _inferred_config(identity)
        for label, group in card["groups"].items():
            if not label.startswith("model/"):
                continue
            everything = group["all"]
            per_generation = everything.get("funnel", {}).get("per_generation", {})
            rows.append(
                {
                    "run": name,
                    "condition": label.split("/", 1)[1],
                    "config": config,
                    "funnel": {stage: per_generation.get(stage) for stage in COMPARE_GENERATION},
                    "sampled_solved": [group["sampled"]["solved"], group["sampled"]["scored"]],
                    "greedy_solved": [group["greedy"]["solved"], group["greedy"]["scored"]],
                    "unscored": everything["unscored"],
                    "failure_classes": everything["failure_classes"],
                    "name_failures": everything.get("funnel", {}).get("name_failures"),
                    "harness_defects": everything.get("funnel", {}).get(
                        "instances_with_harness_defects"
                    ),
                    "generation_seconds": everything.get(
                        "attributed_generation_seconds", everything["generation_seconds"]
                    ),
                    "sandbox_calls": everything["sandbox_calls"],
                    "wall_seconds": everything["wall_seconds"],
                }
            )
    return {"schema": "agent-dev-comparison-v1", "rows": rows}


def _counts(values: dict[str, int] | None) -> str:
    if values is None:
        return "n/a"
    return ", ".join(f"{key} {value}" for key, value in values.items()) or "-"


def comparison_markdown(table: dict[str, Any]) -> str:
    head = (
        "| Run | Condition | Prompt | T / top-p | Recovery | "
        + " | ".join(COMPARE_GENERATION)
        + " | Sampled solved | Greedy solved | Failure classes | Name failures | "
        "Harness defects | Gen s | Sandbox calls | Wall s |"
    )
    lines = [
        "# Agent development comparison",
        "",
        "Funnel columns count generations (fresh = excluding reuse). Solved counts instances: "
        "public tests, correctness and preservation checks all passed.",
        "",
        head,
        "| "
        + " | ".join(["---"] * 5 + ["---:"] * (len(COMPARE_GENERATION) + 2))
        + " | "
        + " | ".join(["---"] * 3 + ["---:"] * 3)
        + " |",
    ]
    for row in table["rows"]:
        config = row["config"]
        funnel_cells = [
            "n/a" if row["funnel"][stage] is None else str(row["funnel"][stage])
            for stage in COMPARE_GENERATION
        ]
        lines.append(
            f"| {row['run']} | {row['condition']} | {config['prompt']} | "
            f"{config['sampled_temperature']} / {config['sampled_top_p']} | {config['recovery']} | "
            + " | ".join(funnel_cells)
            + f" | {row['sampled_solved'][0]}/{row['sampled_solved'][1]}"
            f" | {row['greedy_solved'][0]}/{row['greedy_solved'][1]}"
            f" | {_counts(row['failure_classes'])} | {_counts(row['name_failures'])}"
            f" | {_counts(row['harness_defects'])} | {row['generation_seconds']:.1f}"
            f" | {row['sandbox_calls']} | {row['wall_seconds']:.1f} |"
        )
    return "\n".join(lines) + "\n"
