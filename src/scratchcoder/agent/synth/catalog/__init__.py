"""The template catalog: every family and template of the generator, in a fixed order.

Each family module defines ``FAMILY`` and ``TEMPLATES``. The catalog checks, without running
anything emitted, that template and family IDs are distinct, that bug kinds (which the spec
functions read as flags) are distinct within a family, and that every workflow type is one of
:data:`scratchcoder.agent.synth.template.WORKFLOWS`.
"""

from __future__ import annotations

from scratchcoder.agent.synth.catalog import (
    dice,
    digits,
    flags,
    graph,
    grid,
    jobs,
    matrix,
    points,
    poly,
    rpn,
    runs,
    sensor,
    stock,
    votes,
)
from scratchcoder.agent.synth.template import WORKFLOWS, Catalog, Template, TemplateError

_MODULES = (
    stock,
    grid,
    sensor,
    dice,
    graph,
    jobs,
    votes,
    matrix,
    digits,
    runs,
    points,
    poly,
    rpn,
    flags,
)


def _check(templates: tuple[Template, ...]) -> None:
    ids = [template.id for template in templates]
    if len(set(ids)) != len(ids):
        raise TemplateError("repeated template IDs")
    flags: dict[str, dict[str, str]] = {}
    for template in templates:
        if template.workflow not in WORKFLOWS:
            raise TemplateError(f"{template.id}: unknown workflow {template.workflow}")
        if template.workflow == "W4" and template.caller:
            raise TemplateError(f"{template.id}: a stub case reaches its target directly")
        if template.workflow == "W3" and not template.caller:
            raise TemplateError(f"{template.id}: a cross-module case needs a caller")
        seen = flags.setdefault(template.family.id, {})
        for kind, bug in template.bugs.items():
            if bug.stub:
                continue
            if seen.get(kind, template.target) != template.target:
                raise TemplateError(f"{template.id}: bug kind {kind} is used by another target")
            seen[kind] = template.target


def catalog() -> Catalog:
    families = tuple(module.FAMILY for module in _MODULES)
    templates = tuple(template for module in _MODULES for template in module.TEMPLATES)
    if len({family.id for family in families}) != len(families):
        raise TemplateError("repeated family IDs")
    _check(templates)
    return Catalog(families, templates)
