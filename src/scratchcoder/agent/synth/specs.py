"""``specs.json``: what an independent author needs to write each template's oracle.

Per template: its ID, family, workflow type and target concept; the natural-language
behaviour, input domain and return type of the target; the instance constants the behaviour
refers to; and, per generated instance, the function and module names, the exact signature
and the constants (``params``). Plus the oracle file format
(:data:`scratchcoder.agent.synth.independent.FORMAT`).

It holds no generator code, no test inputs and no expected values, so the oracle is written
from the specification alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from scratchcoder.agent.synth import GENERATOR_VERSION, independent
from scratchcoder.agent.synth.template import WORKFLOWS, Catalog, Template

SCHEMA = "agent-synth-specs-v1"
NOTE = (
    "Names in a template's spec are generic. Each instance renames the function, its module "
    "and some parameters (see 'instances'); the behaviour and the positional order of the "
    "arguments are the same for every instance of a template. 'params' holds the instance's "
    "constants, which the spec refers to as params['name']."
)


def param_keys(template: Template) -> list[str]:
    """The family constants the target's source refers to."""
    func = template.family.func(template.target)
    return sorted(
        key
        for key in template.family.consts
        if any(f"${key}" in variant for variant in func.variants)
    )


def specs(catalog: Catalog, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_template: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_template.setdefault(str(record["template_id"]), []).append(record)
    templates = []
    for template in catalog.templates:
        keys = param_keys(template)
        found = sorted(by_template.get(template.id, []), key=lambda item: str(item["id"]))
        templates.append(
            {
                "template_id": template.id,
                "family": template.family.id,
                "family_domain": template.family.domain,
                "workflow_type": template.workflow,
                "workflow": WORKFLOWS[template.workflow],
                "concept": template.concept,
                "spec": template.spec,
                "input_domain": template.domain,
                "returns": template.returns,
                "param_keys": keys,
                "oracle_file": f"independent/{template.id}.py",
                "instances": [
                    {
                        "instance": record["id"],
                        "module": str(record["target"]).split("::")[0],
                        "function": str(record["target"]).split("::")[1],
                        "signature": record["signature"],
                        "params": record["oracle_params"],
                    }
                    for record in found
                ],
            }
        )
    return {
        "schema": SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "note": NOTE,
        "oracle_format": independent.FORMAT,
        "templates": templates,
    }
