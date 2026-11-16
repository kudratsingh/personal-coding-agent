"""The template model: families of small repositories, their functions, bugs and specs.

A :class:`Family` is one repository domain: its module slots (each with candidate module
names), the identifier and constant pools a seed draws from, and its functions. A
:class:`Func` is one top-level function: a source template in canonical form (the text
``ast.unparse`` prints, with ``$placeholders`` for names and constants), a **spec function**
(project code that computes the function's result, used only to write expected values as
literals) and the public and hidden inputs its tests use.

A :class:`Template` picks one function of a family as the target, the workflow type, the
other functions of the repository, and the **bug kinds**: each is the original (buggy or
stub) state of the target, given as exact text substitutions in the reference source (or a
stub body), with the spec function told the bug kind so that it can model the buggy
behaviour. That is how the generator knows, without running anything it emits, which public
tests fail before the fix, which hidden checks the original fails, and (for tracebacks) which
exception the original raises.

Nothing here executes emitted text: sources are rendered with :class:`string.Template` and
parsed with :mod:`ast`; the spec functions are this module's caller's own code.
"""

from __future__ import annotations

import ast
import copy
import random
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

Params = Mapping[str, Any]
Args = tuple[Any, ...]
Spec = Callable[..., Any]
"""``spec(params, *args, bug=None)``: the function's result for ``args`` under the instance
constants ``params``; with ``bug`` set to a bug kind of a template whose target this
function is (or depends on), the result of the original buggy code instead."""
Labeled = Sequence[tuple[str, Args]]
LabeledInputs = Labeled | Callable[[Params], Labeled]
Inputs = Sequence[Args] | Callable[[Params], Sequence[Args]]

WORKFLOWS = {
    "W1": "failing-test bug fix",
    "W3": "cross-module bug (the cause is in a helper another module calls)",
    "W4": "small feature: implement a stub from its docstring",
    "W5": "debug from a traceback in the task text",
}
CATEGORY = {"W1": "bug_fix", "W3": "bug_fix", "W4": "feature", "W5": "debugging"}
STUBS = ("raise", "none")


class TemplateError(ValueError):
    """A template is malformed (an authoring error, found without running anything)."""


@dataclass(frozen=True)
class Func:
    role: str
    """The function's key in its family; its name placeholder is ``$<role>``."""
    module: str
    """The module slot that holds it."""
    source: str | tuple[str, ...]
    """Canonical source template, or several structural variants of it."""
    spec: Spec
    public: LabeledInputs = ()
    """Public test inputs as (label, args); the test is ``test_<name>_<label>``."""
    hidden: Inputs = ()
    """Hidden check inputs."""
    uses: tuple[str, ...] = ()
    """Roles this function calls (imports across modules are derived from it)."""
    imports: tuple[str, ...] = ()
    """Standard-library import lines the function needs."""

    @property
    def variants(self) -> tuple[str, ...]:
        return (self.source,) if isinstance(self.source, str) else tuple(self.source)


@dataclass(frozen=True)
class Family:
    id: str
    domain: str
    """The repository domain, in a few words."""
    modules: Mapping[str, Sequence[str]]
    """Module slot -> candidate module names (without ``.py``), in file order."""
    docs: Mapping[str, str]
    """Module slot -> the module docstring."""
    names: Mapping[str, Sequence[str]]
    """Placeholder -> candidate identifiers (function roles and local names)."""
    consts: Mapping[str, Sequence[Any]]
    """Placeholder -> candidate constant values (rendered with ``repr``)."""
    funcs: tuple[Func, ...]

    def func(self, role: str) -> Func:
        for item in self.funcs:
            if item.role == role:
                return item
        raise TemplateError(f"{self.id}: no function {role!r}")


@dataclass(frozen=True)
class Bug:
    edits: tuple[tuple[str, str], ...] = ()
    """(old, new) text substitutions in the rendered reference target; each old occurs once."""
    stub: str = ""
    """For a feature (W4): ``raise`` (``raise NotImplementedError``) or ``none``
    (``return None``) after the docstring."""


@dataclass(frozen=True)
class Template:
    id: str
    family: Family
    workflow: str
    target: str
    concept: str
    """The target concept, in a few words."""
    bugs: Mapping[str, Bug]
    spec: str
    """Natural-language behaviour of the target, for an independent author."""
    domain: str
    """Natural-language input domain of the target."""
    returns: str
    """Natural-language return type of the target."""
    include: tuple[str, ...] = ()
    """The other functions of the repository (their dependencies are added)."""
    caller: str = ""
    """For a cross-module case: the role whose public tests exercise the target."""

    def roles(self) -> list[str]:
        """Every function of the repository, in family order (dependencies included)."""
        wanted = {self.target, *self.include, *([self.caller] if self.caller else [])}
        pending = list(wanted)
        while pending:
            for used in self.family.func(pending.pop()).uses:
                if used not in wanted:
                    wanted.add(used)
                    pending.append(used)
        return [item.role for item in self.family.funcs if item.role in wanted]

    def dependents(self) -> set[str]:
        """Roles whose behaviour depends on the target (the target included)."""
        roles = self.roles()
        found = {self.target}
        changed = True
        while changed:
            changed = False
            for role in roles:
                if role not in found and set(self.family.func(role).uses) & found:
                    found.add(role)
                    changed = True
        return found


def canonical(text: str) -> str:
    """The text ``ast.unparse`` prints for one function definition."""
    tree = ast.parse(text)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise TemplateError("a function source must hold exactly one function definition")
    return ast.unparse(tree.body[0])


def _require_canonical(text: str, where: str) -> str:
    try:
        found = canonical(text)
    except SyntaxError as error:
        raise TemplateError(f"{where}: does not parse: {error}") from None
    if found != text:
        raise TemplateError(f"{where}: not in canonical form; ast.unparse gives:\n{found}")
    return text


@dataclass(frozen=True)
class Rendering:
    """The seed's choices and the rendered reference sources."""

    names: dict[str, str]
    modules: dict[str, str]
    params: dict[str, Any]
    variants: dict[str, int]
    sources: dict[str, str]
    """Role -> canonical reference source."""

    def mapping(self) -> dict[str, str]:
        found = dict(self.names)
        found.update({key: repr(value) for key, value in self.params.items()})
        return found


def substitute(text: str, mapping: Mapping[str, str], where: str) -> str:
    try:
        return string.Template(text).substitute(mapping)
    except (KeyError, ValueError) as error:
        raise TemplateError(f"{where}: placeholder {error}") from None


def apply_edits(text: str, edits: Sequence[tuple[str, str]], mapping: Mapping[str, str]) -> str:
    """``text`` with each (old, new) substitution applied; each old must occur exactly once."""
    for old, new in edits:
        old_text = substitute(old, mapping, "bug edit")
        new_text = substitute(new, mapping, "bug edit")
        if text.count(old_text) != 1:
            raise TemplateError(f"bug edit {old_text!r} does not occur exactly once")
        text = text.replace(old_text, new_text)
    return text


def stub_source(reference: str, kind: str) -> str:
    """The reference function with its body replaced by a stub (the docstring is kept)."""
    if kind not in STUBS:
        raise TemplateError(f"unknown stub {kind!r}")
    node = ast.parse(reference).body[0]
    assert isinstance(node, ast.FunctionDef)
    stub = copy.deepcopy(node)
    body: list[ast.stmt] = []
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        if isinstance(first.value.value, str):
            body.append(copy.deepcopy(first))
    if kind == "raise":
        body.append(ast.Raise(ast.Name("NotImplementedError", ast.Load())))
    else:
        body.append(ast.Return(ast.Constant(None)))
    stub.body = body
    return ast.unparse(ast.fix_missing_locations(stub))


def _choose_distinct(
    rng: random.Random, pools: Mapping[str, Sequence[str]], taken: set[str], what: str
) -> dict[str, str]:
    chosen: dict[str, str] = {}
    for key in sorted(pools):
        options = [value for value in pools[key] if value not in taken]
        if not options:
            raise TemplateError(f"no distinct {what} left for {key!r}")
        value = rng.choice(options)
        taken.add(value)
        chosen[key] = value
    return chosen


def bug_text(template: Template, rendering: Rendering, kind: str) -> str:
    """The original (buggy or stub) source of the target under bug ``kind``."""
    bug = template.bugs[kind]
    reference = rendering.sources[template.target]
    if bug.stub:
        return stub_source(reference, bug.stub)
    text = apply_edits(reference, bug.edits, rendering.mapping())
    return _require_canonical(text, f"{template.id}/{kind}")


def render(template: Template, rng: random.Random, kind: str) -> Rendering:
    """The seed's choices (in a fixed order) and the reference sources they give. A
    structural variant of the target is chosen among those the bug's edits apply to."""
    family = template.family
    if kind not in template.bugs:
        raise TemplateError(f"{template.id}: unknown bug kind {kind!r}")
    names = _choose_distinct(rng, family.names, set(), "identifier")
    modules = _choose_distinct(rng, family.modules, set(names.values()), "module name")
    params = {key: rng.choice(list(family.consts[key])) for key in sorted(family.consts)}
    mapping = dict(names)
    mapping.update({key: repr(value) for key, value in params.items()})
    variants: dict[str, int] = {}
    sources: dict[str, str] = {}
    for role in template.roles():
        func = family.func(role)
        options = list(range(len(func.variants)))
        if role == template.target and template.bugs[kind].edits:
            options = [
                index
                for index in options
                if _edits_apply(func.variants[index], template.bugs[kind].edits, mapping)
            ]
            if not options:
                raise TemplateError(f"{template.id}/{kind}: no variant takes the bug edits")
        index = rng.choice(options)
        text = substitute(func.variants[index], mapping, f"{family.id}/{role}")
        _require_canonical(text, f"{family.id}/{role}[{index}]")
        node = ast.parse(text).body[0]
        assert isinstance(node, ast.FunctionDef)
        if node.name != names[role]:
            raise TemplateError(f"{family.id}/{role}: defines {node.name}, not ${role}")
        variants[role] = index
        sources[role] = text
    return Rendering(names, modules, params, variants, sources)


def _edits_apply(source: str, edits: Sequence[tuple[str, str]], mapping: Mapping[str, str]) -> bool:
    text = substitute(source, mapping, "variant")
    try:
        apply_edits(text, edits, mapping)
    except TemplateError:
        return False
    return True


@dataclass(frozen=True)
class Outcome:
    """What a call returns or raises, as the spec function models it."""

    raised: str = ""
    """The exception type name, or empty for a return."""
    message: str = ""
    value: Any = None
    frames: tuple[str, ...] = ()
    """For a raise: the roles whose spec functions were on the stack, outermost first. A
    spec function mirrors its function's calls, so these are the repository frames of the
    original's traceback."""

    def same(self, other: Outcome) -> bool:
        if self.raised or other.raised:
            return (self.raised, self.message) == (other.raised, other.message)
        return type(self.value) is type(other.value) and self.value == other.value


def outcome(
    template: Template, role: str, params: Params, args: Args, bug: str | None = None
) -> Outcome:
    """The modelled outcome of ``role(*args)``; with ``bug``, in the original repository.
    Arguments are deep-copied first, so a spec that mutates its inputs changes nothing."""
    call_bug = bug if bug is not None and role in template.dependents() else None
    if call_bug is not None and template.bugs[call_bug].stub:
        if role != template.target:
            raise TemplateError(f"{template.id}: a stub target may not have dependents")
        if template.bugs[call_bug].stub == "raise":
            return Outcome("NotImplementedError", "")
        return Outcome(value=None)
    spec = template.family.func(role).spec
    try:
        value = spec(params, *copy.deepcopy(args), bug=call_bug)
    except Exception as error:
        if bug is None:
            raise TemplateError(f"{template.id}: the reference spec of {role} raised") from error
        return Outcome(type(error).__name__, str(error), frames=_spec_frames(template, error))
    return Outcome(value=value)


def _spec_frames(template: Template, error: BaseException) -> tuple[str, ...]:
    codes = {item.spec.__code__: item.role for item in template.family.funcs}
    frames: list[str] = []
    trace = error.__traceback__
    while trace is not None:
        role = codes.get(trace.tb_frame.f_code)
        if role is not None:
            frames.append(role)
        trace = trace.tb_next
    return tuple(frames)


def inputs(source: LabeledInputs | Inputs, params: Params) -> list[Any]:
    found = source(params) if callable(source) else source
    return list(found)


@dataclass(frozen=True)
class Catalog:
    families: tuple[Family, ...]
    templates: tuple[Template, ...] = field(default_factory=tuple)

    def template(self, template_id: str) -> Template:
        for item in self.templates:
            if item.id == template_id:
                return item
        raise TemplateError(f"unknown template {template_id!r}")
