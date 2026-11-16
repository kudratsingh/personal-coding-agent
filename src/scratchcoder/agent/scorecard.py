"""The development scorecard: per case and condition, what was solved, how, and at what cost.

Counts only; no count is reported without its denominator. Unscored instances (host failures
that outlasted the retries) are counted separately and never as failures. Oracle rows check
the harness, not the model, and are reported apart from model rows.

Results are keyed by case ID: a scorecard refuses two results for the same instance and case
IDs that are malformed or equal ignoring case (:func:`scratchcoder.agent.cases.check_case_ids`),
so two distinct cases can never merge into one cell. Under ``--prefill-name`` the harness writes
the function name, so name-following is labelled harness-supplied, never model task-following.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from scratchcoder.agent.cases import CaseError, check_case_ids
from scratchcoder.agent.loop import FAILURE_CLASSES, InstanceResult

HARNESS_SUPPLIED = "harness-supplied (name prefill)"
MODEL_WRITTEN = "model"
PREFILL_NOTE = (
    "Name prefill on: before each generation the harness wrote a python block opener and "
    "`def <name>(` for the nearest shown function. The one_shown_function and target_function "
    "stages and the name failures therefore count harness-supplied names, not model "
    "task-following. It is skipped for a decorated function (the editor would drop the "
    "decorators). Prefill tokens are counted apart from generated tokens."
)

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
        "generations": sum(len(result.turns) for result in results),
        "prefilled_generations": sum(
            turn.prefill_tokens > 0 for result in results for turn in result.turns
        ),
        "prefill_tokens": sum(turn.prefill_tokens for result in results for turn in result.turns),
        "prefill_skipped_generations": sum(
            bool(turn.prefill_skipped) for result in results for turn in result.turns
        ),
        "name_following": (
            HARNESS_SUPPLIED
            if any(turn.prefill_tokens for result in results for turn in result.turns)
            else MODEL_WRITTEN
        ),
    }


def check_instances(results: Sequence[InstanceResult]) -> None:
    """Refuse results that would merge in a scorecard: malformed case IDs, IDs equal ignoring
    case, and two results for one (case, policy, condition, decoding) instance."""
    check_case_ids(sorted({result.case_id for result in results}), what="results")
    keys = Counter((r.case_id, r.policy, r.condition, r.decoding) for r in results)
    repeated = sorted(key for key, count in keys.items() if count > 1)
    if repeated:
        raise CaseError(f"results: instance {'--'.join(repeated[0])} appears more than once")


def build(
    results: Sequence[InstanceResult],
    identity: dict[str, Any],
    *,
    prefill: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The scorecard; ``prefill`` is the after-the-run report of :func:`prefill_report`."""
    check_instances(results)
    if identity.get("cases") is not None:
        check_case_ids(identity["cases"], what="identity cases")
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
    card: dict[str, Any] = {
        "schema": "agent-dev-scorecard-v1",
        "identity": identity,
        "failure_class_vocabulary": list(FAILURE_CLASSES),
        "groups": groups,
        "instances": [result.summary() for result in results],
    }
    if prefill is not None:
        card["prefill_report"] = prefill
    return card


def prefill_report(
    traced: Iterable[tuple[str, Sequence[Mapping[str, Any]]]],
    targets: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """For reporting only, after the run: per policy/condition, how often the prefilled name
    (read back from each instance's trace) equals the name of one of the case's targets.

    ``traced`` holds (group label, trace events) per instance; ``targets`` maps a case ID to its
    ``path::function`` targets. Nothing here reaches a policy: the run is over."""
    report: dict[str, dict[str, int]] = {}
    for label, events in traced:
        start = next(event for event in events if event.get("event") == "start")
        names = {key.split("::", 1)[1] for key in targets[start["case"]]}
        prompts = [event for event in events if event.get("event") == "prompt"]
        prefilled = [
            event["prefill_function"] for event in prompts if event.get("prefill_function")
        ]
        skipped = [event for event in prompts if event.get("prefill_skipped")]
        hits = [key for key in prefilled if key.split("::", 1)[1] in names]
        row = report.setdefault(
            label,
            {
                "instances": 0,
                "instances_prefilled": 0,
                "instances_prefilled_target_name": 0,
                "generations": 0,
                "generations_prefilled": 0,
                "generations_prefilled_target_name": 0,
                "generations_prefill_skipped": 0,
            },
        )
        row["instances"] += 1
        row["instances_prefilled"] += bool(prefilled)
        row["instances_prefilled_target_name"] += bool(hits)
        row["generations"] += len(prompts)
        row["generations_prefilled"] += len(prefilled)
        row["generations_prefilled_target_name"] += len(hits)
        row["generations_prefill_skipped"] += len(skipped)
    return dict(sorted(report.items()))


def _of(count: int, total: int, unit: str = "") -> str:
    return f"{count} of {total}" + (f" {unit}" if unit else "")


def _counts_of(values: Mapping[str, int] | None, total: int | None, unit: str) -> str:
    """``key k of N unit`` for each count, or ``-``; ``n/a`` for a missing count."""
    if values is None:
        return "n/a"
    if not values:
        return "-"
    if total is None:
        return ", ".join(
            f"{key} {value} (denominator not recorded)" for key, value in values.items()
        )
    return ", ".join(f"{key} {_of(value, total, unit)}" for key, value in values.items())


DEFAULT_DECODING = (1.0, 1.0)
"""The sampled temperature and top-p of the first development configuration."""


def decoding_text(temperature: object, top_p: object) -> str:
    """The sampled decoding named precisely, against the default (T 1.0, top-p 1.0): for
    example ``lower temperature plus nucleus sampling (T 0.6, top-p 0.95)``."""
    values = f"(T {temperature}, top-p {top_p})"
    if not isinstance(temperature, int | float) or not isinstance(top_p, int | float):
        return f"sampled decoding {values}"
    base_t, base_p = DEFAULT_DECODING
    parts = []
    if temperature != base_t:
        parts.append("lower temperature" if temperature < base_t else "higher temperature")
    if top_p != base_p:
        parts.append("nucleus sampling")
    return (" plus ".join(parts) or "default sampling") + " " + values


def _config_line(identity: Mapping[str, Any]) -> list[str]:
    config = identity.get("config")
    if not config:
        return []
    return [
        f"Configuration: prompt {config.get('prompt')}, sampled decoding "
        f"{decoding_text(config.get('sampled_temperature'), config.get('sampled_top_p'))}, "
        f"recovery {config.get('recovery')}, name prefill "
        f"{'on' if config.get('prefill_name') else 'off'}.",
        "",
    ]


def markdown(card: dict[str, Any]) -> str:
    """A compact table for review: one row per case per policy/condition. Every count is
    written with its denominator."""
    lines = [
        "# Agent development scorecard",
        "",
        *_config_line(card.get("identity", {})),
        "Dev cases only. Solved = public tests, hidden correctness checks and preservation "
        "checks all `passed` (supervisor status), out of the scored instances. Sampled cells "
        "count seeds; greedy is one instance. Failure classes count scored, unsolved instances "
        "out of the scored instances of the row. Generation seconds are net (time spent); "
        "reused generations add their original time in brackets, so net + cached is "
        "comparable across conditions.",
        "",
        "| Policy/condition | Case | Sampled solved | Greedy solved | Failure classes "
        "(scored, unsolved) | Turns | Gen tokens | Gen s net (+ cached) | Sandbox calls | "
        "Sandbox s |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    prefilled = any(group["all"].get("prefilled_generations") for group in card["groups"].values())
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
            scored = sampled["scored"] + greedy["scored"]
            instances = sampled["instances"] + greedy["instances"]
            lines.append(
                f"| {label} | {case} | {sampled['solved']}/{sampled['scored']} | "
                f"{greedy['solved']}/{greedy['scored']} | "
                + _counts_of(dict(sorted(classes.items())), scored, "scored")
                + (f" (unscored {_of(unscored, instances)})" if unscored else "")
                + f" | {total['turns_used']} | {total['generated_tokens']} | "
                f"{gen:.1f} (+{cached:.1f}) | "
                f"{total['sandbox_calls']} | {box:.1f} |"
            )
    lines += [
        "",
        "## Funnel",
        "",
        "Nested stages (each counts only what reached the one before), per generation, then "
        "per instance; side columns follow the stages. Every cell is a count of all "
        "generations (or all instances) of its row.",
        "",
    ]
    if prefilled:
        lines += [PREFILL_NOTE, ""]
    lines += [
        "| Policy/condition | Level | Name following | "
        + " | ".join(GENERATION_STAGES + GENERATION_SIDE)
        + " |",
        "| --- | --- | --- | "
        + " | ".join("---:" for _ in GENERATION_STAGES + GENERATION_SIDE)
        + " |",
    ]
    for label, group in card["groups"].items():
        stages = group["all"]["funnel"]["per_generation"]
        whole = stages["generations"]
        lines.append(
            f"| {label} | generation | {group['all'].get('name_following', MODEL_WRITTEN)} | "
            + " | ".join(
                str(whole) if s == "generations" else _of(stages[s], whole)
                for s in GENERATION_STAGES + GENERATION_SIDE
            )
            + " |"
        )
    lines += [
        "",
        "| Policy/condition | "
        + " | ".join(INSTANCE_STAGES + INSTANCE_SIDE)
        + " | name failures (of generations) | harness defects (of instances) |",
        "| --- | " + " | ".join("---:" for _ in INSTANCE_STAGES + INSTANCE_SIDE) + " | --- | --- |",
    ]
    for label, group in card["groups"].items():
        found = group["all"]["funnel"]
        stages = found["per_instance"]
        whole = stages["instances"]
        generations = found["per_generation"]["generations"]
        lines.append(
            f"| {label} | "
            + " | ".join(
                str(whole) if s == "instances" else _of(stages[s], whole)
                for s in INSTANCE_STAGES + INSTANCE_SIDE
            )
            + f" | {_counts_of(found['name_failures'], generations, 'generations')}"
            + f" | {_counts_of(found['instances_with_harness_defects'], whole, 'instances')} |"
        )
    lines += _prefill_lines(card)
    return "\n".join(lines) + "\n"


def _prefill_lines(card: Mapping[str, Any]) -> list[str]:
    lines = []
    prefilled = [
        (label, group["all"])
        for label, group in card["groups"].items()
        if group["all"].get("prefilled_generations")
        or group["all"].get("prefill_skipped_generations")
    ]
    if prefilled:
        lines += [
            "",
            "## Name prefill",
            "",
            "| Policy/condition | Prefilled generations | Skipped (decorated) | Prefill tokens | "
            "Generated tokens (without prefill) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for label, cell in prefilled:
            lines.append(
                f"| {label} | {_of(cell['prefilled_generations'], cell['generations'])} | "
                f"{_of(cell.get('prefill_skipped_generations', 0), cell['generations'])} | "
                f"{cell['prefill_tokens']} | {cell['generated_tokens']} |"
            )
    report = card.get("prefill_report")
    if report:
        lines += [
            "",
            "Reporting only, read from the traces after the run: how often the prefilled name "
            "equals the name of a target of the case (the harness never consults the targets).",
            "",
            "| Policy/condition | Instances with a prefilled target name | Generations with a "
            "prefilled target name |",
            "| --- | ---: | ---: |",
        ]
        for label, row in report.items():
            lines.append(
                f"| {label} | {_of(row['instances_prefilled_target_name'], row['instances'])} | "
                f"{_of(row['generations_prefilled_target_name'], row['generations'])} |"
            )
    return lines


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
        "prefill_name": False,
        "inferred": True,
    }


def _config(identity: dict[str, Any]) -> dict[str, Any]:
    """The recorded configuration; a scorecard written before the name prefill existed ran
    without it."""
    config = identity.get("config") or _inferred_config(identity)
    return {"prefill_name": False, **config}


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
    "prefill_name": ("prefill_name",),
}
"""Configuration factors. The sampled temperature and top-p move together as one factor. The
name prefill is a harness aid: a run may differ from the baseline in it only when it is the one
factor that differs (``prefill_name_with_another_factor`` otherwise makes the row
non-comparable)."""


def invariants(identity: dict[str, Any]) -> dict[str, Any]:
    values = {field: identity.get(field) for field in INVARIANT_FIELDS}
    values["decoding_rows"] = [
        [row.get("label"), row.get("index")]
        + ([row.get("temperature"), row.get("top_p")] if row.get("label") == "greedy" else [])
        for row in identity.get("decodings", [])
    ]
    return values


def _factor_text(factor: str, config: Mapping[str, Any]) -> str:
    if factor == "sampled_decoding":
        return "sampled decoding, " + decoding_text(
            config.get("sampled_temperature"), config.get("sampled_top_p")
        )
    if factor == "prefill_name":
        return "prefill_name (harness-supplied function name)"
    return factor


def _label(changed: list[str], first: bool, config: Mapping[str, Any] | None = None) -> str:
    if first:
        return "baseline"
    if not changed:
        return "same configuration as baseline"
    texts = [_factor_text(factor, config or {}) for factor in changed]
    if len(texts) == 1:
        return f"single factor: {texts[0]}"
    return "combined: " + " + ".join(texts)


def _card_case_ids(card: Mapping[str, Any]) -> list[str]:
    identity = card.get("identity", {})
    found = {*identity.get("cases", []), *identity.get("valid_cases", [])}
    found |= {row["case_id"] for row in card.get("instances", []) if "case_id" in row}
    return sorted(found)


def check_cards(cards: Sequence[tuple[str, Mapping[str, Any]]]) -> None:
    """Refuse a comparison whose rows could merge: two runs with one name, or a run whose case
    IDs are malformed or equal ignoring case."""
    names = [name for name, _ in cards]
    if len(set(names)) != len(names):
        raise CaseError("compare: two runs share a name; rows are keyed by run name")
    for name, card in cards:
        check_case_ids(_card_case_ids(card), what=f"run {name}")


def comparison(cards: Sequence[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """One row per run and model condition: configuration, funnel, solved and cost. The first
    run is the baseline. Every run must share :data:`INVARIANT_FIELDS` with it, or its rows are
    marked non-comparable (with the fields named); each row lists the configuration fields and
    factors that differ from the baseline. A run without funnel data (an earlier scorecard)
    shows ``None`` for those columns."""
    rows = []
    if not cards:
        raise ValueError("compare needs at least one run")
    check_cards(cards)
    base_identity = cards[0][1].get("identity", {})
    base_invariants = invariants(base_identity)
    base_config = _config(base_identity)
    for position, (name, card) in enumerate(cards):
        identity = card.get("identity", {})
        config = _config(identity)
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
        if "prefill_name" in changed and len(changed) > 1:
            noncomparable.append("prefill_name_with_another_factor")
        report = card.get("prefill_report") or {}
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
                    "label": _label(changed, position == 0, config),
                    "differs_from_baseline": fields,
                    "factors_changed": changed,
                    "comparable": not noncomparable,
                    "noncomparable_fields": noncomparable,
                    "funnel": {stage: per_generation.get(stage) for stage in COMPARE_GENERATION},
                    "instances": everything.get("instances"),
                    "scored": everything.get("scored"),
                    "name_following": everything.get("name_following", MODEL_WRITTEN),
                    "generations": everything.get("generations"),
                    "prefilled_generations": everything.get("prefilled_generations", 0),
                    "prefill_skipped_generations": everything.get("prefill_skipped_generations", 0),
                    "prefill_tokens": everything.get("prefill_tokens", 0),
                    "prefill_report": report.get(label),
                    "generated_tokens": everything.get("generated_tokens"),
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
        "schema": "agent-dev-comparison-v3",
        "baseline": cards[0][0],
        "comparable": all(row["comparable"] for row in rows),
        "invariant_fields": list(INVARIANT_FIELDS),
        "rows": rows,
    }


def comparison_markdown(table: dict[str, Any]) -> str:
    head = (
        "| Run | Condition | Label | Comparable | Prompt | Sampled decoding | Recovery | "
        "Name prefill | Prefilled generations | Name following | "
        + " | ".join(COMPARE_GENERATION)
        + " | Sampled solved | Greedy solved | Failure classes | Name failures | "
        "Harness defects | Prefill tokens | Gen s | Sandbox calls | Wall s |"
    )
    prefilled = any(
        row.get("prefilled_generations") or row.get("prefill_skipped_generations")
        for row in table["rows"]
    )
    lines = [
        "# Agent development comparison",
        "",
        f"Baseline: {table['baseline']}. Every run must share the baseline's code commit "
        "(and clean tree), cases, case "
        "registry, checkpoint, tokenizer, seeds, decoding rows, conditions, budgets, threads, "
        "batch size, supervisor and runtime image; only the configuration may differ, and the "
        "name prefill only as the one differing factor. Funnel columns are nested generation "
        "stages, then side counts, each out of the row's generations. Solved counts instances: "
        "public tests, correctness and preservation checks all passed. Failure classes count "
        "scored instances, name failures generations, harness defects instances, each with its "
        "denominator.",
        "",
    ]
    if prefilled:
        lines += [PREFILL_NOTE, ""]
    lines += [
        head,
        "| "
        + " | ".join(["---"] * 10 + ["---:"] * (len(COMPARE_GENERATION) + 2))
        + " | "
        + " | ".join(["---"] * 3 + ["---:"] * 4)
        + " |",
    ]
    for row in table["rows"]:
        config = row["config"]
        generations = row["funnel"].get("generations")
        funnel_cells = [
            "n/a"
            if row["funnel"][stage] is None
            else str(row["funnel"][stage])
            if stage == "generations" or generations is None
            else _of(row["funnel"][stage], generations)
            for stage in COMPARE_GENERATION
        ]
        lines.append(
            f"| {row['run']} | {row['condition']} | {row['label']} | "
            + ("yes" if row["comparable"] else "NO: " + ", ".join(row["noncomparable_fields"]))
            + f" | {config['prompt']} | "
            f"{decoding_text(config['sampled_temperature'], config['sampled_top_p'])} | "
            f"{config['recovery']} | {'on' if config.get('prefill_name') else 'off'} | "
            f"{_prefilled_of(row)} | {row.get('name_following', MODEL_WRITTEN)} | "
            + " | ".join(funnel_cells)
            + f" | {row['sampled_solved'][0]}/{row['sampled_solved'][1]}"
            f" | {row['greedy_solved'][0]}/{row['greedy_solved'][1]}"
            f" | {_counts_of(row['failure_classes'], row.get('scored'), 'scored')}"
            f" | {_counts_of(row['name_failures'], generations, 'generations')}"
            f" | {_counts_of(row['harness_defects'], row.get('instances'), 'instances')}"
            f" | {row.get('prefill_tokens', 0)} | {row['generation_seconds']:.1f}"
            f" | {row['sandbox_calls']} | {row['wall_seconds']:.1f} |"
        )
    if prefilled:
        lines += _compare_prefill_lines(table["rows"])
    return "\n".join(lines) + "\n"


def _prefilled_of(row: Mapping[str, Any]) -> str:
    """``k of N generations`` prefilled, or ``n/a`` for a run without a generation count."""
    generations = row.get("generations")
    if generations is None:
        return "n/a"
    return _of(row.get("prefilled_generations", 0), generations, "generations")


def _compare_prefill_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Per row: how often the prefill applied and, read from the traces after the run, how
    often the prefilled name was a target's, each with its denominator."""
    lines = [
        "",
        "## Name prefill applicability",
        "",
        "How often the harness wrote a name (reporting only; the harness never consults the "
        "targets). A prefilled target name is a name the harness supplied, not one the model "
        "chose.",
        "",
        "| Run | Condition | Prefilled generations | Skipped (decorated) | Prefill tokens | "
        "Instances with a prefilled target name | Generations with a prefilled target name |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        generations = row.get("generations")
        skipped = (
            "n/a"
            if generations is None
            else _of(row.get("prefill_skipped_generations", 0), generations, "generations")
        )
        report = row.get("prefill_report")
        instances, named = (
            ("n/a", "n/a")
            if not report
            else (
                _of(report["instances_prefilled_target_name"], report["instances"], "instances"),
                _of(
                    report["generations_prefilled_target_name"],
                    report["generations"],
                    "generations",
                ),
            )
        )
        lines.append(
            f"| {row['run']} | {row['condition']} | {_prefilled_of(row)} | {skipped} | "
            f"{row.get('prefill_tokens', 0)} | {instances} | {named} |"
        )
    return lines
