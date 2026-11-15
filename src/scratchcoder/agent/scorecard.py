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
    "closed_or_recovered",
    "parseable",
    "one_shown_function",
    "target_function",
    "applied_target_edit",
    "public_passed",
)
"""Nested stages: each counts only turns that reached the stage before it."""
GENERATION_SIDE = ("fresh_generations", "closed_block", "recovered", "applied_non_target_edit")
INSTANCE_STAGES = (
    "instances",
    "closed_or_recovered",
    "parseable",
    "one_shown_function",
    "target_function",
    "applied_target_edit",
    "public_passed",
    "solved",
)
"""Nested stages per instance; public and solved count only instances with a target edit."""
INSTANCE_SIDE = ("applied_non_target_edit", "solved_without_target_edit")


def _turn_stages(turn: Any) -> dict[str, bool]:
    reached: dict[str, bool] = {}
    reached["closed_or_recovered"] = turn.closed_block or turn.recovered
    reached["parseable"] = reached["closed_or_recovered"] and turn.parseable
    reached["one_shown_function"] = reached["parseable"] and len(turn.shown_defined) == 1
    reached["target_function"] = reached["one_shown_function"] and turn.target_defined
    reached["applied_target_edit"] = reached["target_function"] and turn.edit == "applied"
    reached["public_passed"] = reached["applied_target_edit"] and turn.feedback == "passed"
    return reached


def funnel(results: Sequence[InstanceResult]) -> dict[str, Any]:
    """How far replies got, as nested stages: per generation (every turn) and per instance
    (any turn reaching the stage; public and solved also need a target edit). Side counts:
    fresh generations (no cache reuse), closed and recovered blocks apart, applied edits to a
    function that is not a target, and instances solved without editing a target."""
    turns = [turn for result in results for turn in result.turns]
    staged = [_turn_stages(turn) for turn in turns]
    generations: dict[str, int] = {"generations": len(turns)}
    for stage in GENERATION_STAGES[1:]:
        generations[stage] = sum(row[stage] for row in staged)
    generations.update(
        fresh_generations=sum(not turn.cached for turn in turns),
        closed_block=sum(turn.closed_block for turn in turns),
        recovered=sum(turn.recovered for turn in turns),
        applied_non_target_edit=sum(
            turn.edit == "applied" and not row["applied_target_edit"]
            for turn, row in zip(turns, staged, strict=True)
        ),
    )
    instances: dict[str, int] = {"instances": len(results)}
    per_result = [[_turn_stages(turn) for turn in result.turns] for result in results]
    for stage in INSTANCE_STAGES[1:6]:
        instances[stage] = sum(any(row[stage] for row in rows) for rows in per_result)
    targeted = [any(row["applied_target_edit"] for row in rows) for rows in per_result]
    instances["public_passed"] = sum(
        hit and result.grade.get("public") == "passed"
        for hit, result in zip(targeted, results, strict=True)
    )
    instances["solved"] = sum(
        hit and result.scored and result.solved
        for hit, result in zip(targeted, results, strict=True)
    )
    instances["applied_non_target_edit"] = sum(
        any(t.edit == "applied" for t in result.turns) and not hit
        for hit, result in zip(targeted, results, strict=True)
    )
    instances["solved_without_target_edit"] = sum(
        result.scored and result.solved and not hit
        for hit, result in zip(targeted, results, strict=True)
    )
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
        "Nested stages (each counts only what reached the one before), per generation, then "
        "per instance; side columns follow the stages.",
        "",
        "| Policy/condition | Level | " + " | ".join(GENERATION_STAGES + GENERATION_SIDE) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in GENERATION_STAGES + GENERATION_SIDE) + " |",
    ]
    for label, group in card["groups"].items():
        stages = group["all"]["funnel"]["per_generation"]
        lines.append(
            f"| {label} | generation | "
            + " | ".join(str(stages[s]) for s in GENERATION_STAGES + GENERATION_SIDE)
            + " |"
        )
    lines += [
        "",
        "| Policy/condition | "
        + " | ".join(INSTANCE_STAGES + INSTANCE_SIDE)
        + " | name failures | harness defects |",
        "| --- | " + " | ".join("---:" for _ in INSTANCE_STAGES + INSTANCE_SIDE) + " | --- | --- |",
    ]
    for label, group in card["groups"].items():
        found = group["all"]["funnel"]
        stages = found["per_instance"]
        names = ", ".join(f"{k} {v}" for k, v in found["name_failures"].items()) or "-"
        defects = ", ".join(f"{k} {v}" for k, v in found["instances_with_harness_defects"].items())
        lines.append(
            f"| {label} | "
            + " | ".join(str(stages[s]) for s in INSTANCE_STAGES + INSTANCE_SIDE)
            + f" | {names} | {defects or '-'} |"
        )
    return "\n".join(lines) + "\n"


COMPARE_GENERATION = (*GENERATION_STAGES, *GENERATION_SIDE)


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


INVARIANT_FIELDS = (
    "git_commit",
    "dirty",
    "valid_cases",
    "cases_registry_sha256",
    "checkpoint_sha256",
    "tokenizer_sha256",
    "seed_base",
    "decoding_rows",
    "conditions",
    "budgets",
    "threads",
    "batch_size",
    "supervisor_sha256",
    "runtime_image_id",
    "precision",
    "device",
    "protocol",
)
"""Identity fields every compared run must share: only the configuration may differ."""
FACTORS = {
    "prompt": ("prompt",),
    "sampled_decoding": ("sampled_temperature", "sampled_top_p"),
    "recovery": ("recovery",),
}
"""Configuration factors. The sampled temperature and top-p move together as one factor."""


def invariants(identity: dict[str, Any]) -> dict[str, Any]:
    values = {field: identity.get(field) for field in INVARIANT_FIELDS}
    values["decoding_rows"] = [
        [row.get("label"), row.get("index")]
        + ([row.get("temperature"), row.get("top_p")] if row.get("label") == "greedy" else [])
        for row in identity.get("decodings", [])
    ]
    return values


def _label(changed: list[str], first: bool) -> str:
    if first:
        return "baseline"
    if not changed:
        return "same configuration as baseline"
    if len(changed) == 1:
        return f"single factor: {changed[0]}"
    return "combined: " + " + ".join(changed)


def comparison(cards: Sequence[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """One row per run and model condition: configuration, funnel, solved and cost. The first
    run is the baseline. Every run must share :data:`INVARIANT_FIELDS` with it, or its rows are
    marked non-comparable (with the fields named); each row lists the configuration fields and
    factors that differ from the baseline. A run without funnel data (an earlier scorecard)
    shows ``None`` for those columns."""
    rows = []
    if not cards:
        raise ValueError("compare needs at least one run")
    base_identity = cards[0][1].get("identity", {})
    base_invariants = invariants(base_identity)
    base_config = base_identity.get("config") or _inferred_config(base_identity)
    for position, (name, card) in enumerate(cards):
        identity = card.get("identity", {})
        config = identity.get("config") or _inferred_config(identity)
        own = invariants(identity)
        noncomparable = [
            field for field in INVARIANT_FIELDS if own[field] != base_invariants[field]
        ]
        fields = [
            field
            for names in FACTORS.values()
            for field in names
            if config.get(field) != base_config.get(field)
        ]
        changed = [
            factor
            for factor, names in FACTORS.items()
            if any(config.get(field) != base_config.get(field) for field in names)
        ]
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
                    "label": _label(changed, position == 0),
                    "differs_from_baseline": fields,
                    "factors_changed": changed,
                    "comparable": not noncomparable,
                    "noncomparable_fields": noncomparable,
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
    return {
        "schema": "agent-dev-comparison-v2",
        "baseline": cards[0][0],
        "comparable": all(row["comparable"] for row in rows),
        "invariant_fields": list(INVARIANT_FIELDS),
        "rows": rows,
    }


def _counts(values: dict[str, int] | None) -> str:
    if values is None:
        return "n/a"
    return ", ".join(f"{key} {value}" for key, value in values.items()) or "-"


def comparison_markdown(table: dict[str, Any]) -> str:
    head = (
        "| Run | Condition | Label | Comparable | Prompt | T / top-p | Recovery | "
        + " | ".join(COMPARE_GENERATION)
        + " | Sampled solved | Greedy solved | Failure classes | Name failures | "
        "Harness defects | Gen s | Sandbox calls | Wall s |"
    )
    lines = [
        "# Agent development comparison",
        "",
        f"Baseline: {table['baseline']}. Every run must share the baseline's code commit "
        "(and clean tree), cases, case "
        "registry, checkpoint, tokenizer, seeds, decoding rows, conditions, budgets, threads, "
        "batch size, supervisor and runtime image; only the configuration may differ. Funnel "
        "columns are nested generation stages, then side counts. Solved counts instances: "
        "public tests, correctness and preservation checks all passed.",
        "",
        head,
        "| "
        + " | ".join(["---"] * 7 + ["---:"] * (len(COMPARE_GENERATION) + 2))
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
            f"| {row['run']} | {row['condition']} | {row['label']} | "
            + ("yes" if row["comparable"] else "NO: " + ", ".join(row["noncomparable_fields"]))
            + f" | {config['prompt']} | "
            f"{config['sampled_temperature']} / {config['sampled_top_p']} | {config['recovery']} | "
            + " | ".join(funnel_cells)
            + f" | {row['sampled_solved'][0]}/{row['sampled_solved'][1]}"
            f" | {row['greedy_solved'][0]}/{row['greedy_solved'][1]}"
            f" | {_counts(row['failure_classes'])} | {_counts(row['name_failures'])}"
            f" | {_counts(row['harness_defects'])} | {row['generation_seconds']:.1f}"
            f" | {row['sandbox_calls']} | {row['wall_seconds']:.1f} |"
        )
    return "\n".join(lines) + "\n"
