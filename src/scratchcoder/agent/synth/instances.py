"""One generated instance: a harness v0 case directory, its mutants and its static checks.

:func:`build` turns (template, bug kind, index, seed base) into an :class:`Instance`:

1. The seed (a hash of the generator version, seed base, template, bug kind and index) picks
   names, module names, constants and variants (:func:`.template.render`).
2. Modules are assembled from the rendered functions: a module docstring, standard-library
   imports, imports of functions from the other modules, then the functions.
3. The original repository holds the bug kind's version of the target; the reference fix
   holds the reference version.
4. Public tests (``test_<module>.py``) and the hidden checks are written as literals from the
   spec functions. A public test whose modelled outcome under the bug differs from the
   reference outcome is a *designed failing test*; there must be at least one, and the hidden
   correctness checks must include at least one the original fails. Preservation checks
   are the checks the original passes by construction: those of functions that do not
   depend on the target, and the target's own checks whose outcome the bug leaves unchanged.
5. The task statement is written for the workflow type; a traceback (W5) names the frames
   and the exception the spec function raised.
6. Wrong attempts are AST mutants of the reference target (:mod:`.mutation`), each distinct
   from the original and from the other mutants.

:func:`static_check` then applies every check that needs no execution: the case loads
through :func:`scratchcoder.agent.cases.validate_static`; every function of every version is
at most :data:`MAX_FUNCTION_LINES` lines and :data:`scratchcoder.agent.prompt.FUNCTION_BUDGET`
tokens; the task fits its budget; the turn-1 prompt passes :func:`scratchcoder.agent.fit.
plan_case` (the ``tools.agent_dev_run plan`` rule) and shows the target, from every public test
and from the designed failing ones; and no defined name, test name or module name is on the
evaluation-family blocklist.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scratchcoder.agent import cases, prompt, workspace
from scratchcoder.agent.fit import plan_case
from scratchcoder.agent.synth import GENERATOR_VERSION, mutation
from scratchcoder.agent.synth.specs import param_keys
from scratchcoder.agent.synth.template import (
    CATEGORY,
    Outcome,
    Rendering,
    Template,
    TemplateError,
    bug_text,
    inputs,
    outcome,
    render,
)
from scratchcoder.data.chat import TextEncoder

MAX_FUNCTION_LINES = 20
MIN_CORRECTNESS_CHECKS = 6
MAX_TASK_TOKENS = prompt.BUDGETS["task"]
BUDGETS = {
    "max_edit_turns": 3,
    "max_feedback_runs": 6,
    "max_generated_tokens": 2000,
    "max_wall_seconds": 600,
}
"""The harness v0 per-instance budgets, as the development cases use them."""
INSTANCE_SCHEMA = "agent-synth-instance-v1"
CASE_SET = "agent-sft1-synthetic-train"
"""``case.json`` keeps ``split: dev`` (the loader's only unsealed value); this field marks
the case as generated training material, never an evaluation case."""


def instance_seed(seed_base: int, template_id: str, bug: str, index: int) -> int:
    text = f"{GENERATOR_VERSION}|{seed_base}|{template_id}|{bug}|{index}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def instance_id(template_id: str, bug: str, index: int) -> str:
    return f"{template_id}-{bug}-{index:03d}"


def _literal(value: Any) -> str:
    text = repr(value)
    try:
        back = ast.literal_eval(text)
    except (ValueError, SyntaxError) as error:
        raise TemplateError(f"not a literal: {text}") from error
    if back != value or type(back) is not type(value):
        raise TemplateError(f"does not round-trip as a literal: {text}")
    return text


def _call(name: str, args: Sequence[Any]) -> str:
    return f"{name}({', '.join(_literal(arg) for arg in args)})"


@dataclass(frozen=True)
class Check:
    role: str
    args: tuple[Any, ...]
    expected: Any
    exposes: bool
    """The original's modelled outcome differs (the check fails before the fix)."""


@dataclass(frozen=True)
class PublicTest:
    name: str
    role: str
    args: tuple[Any, ...]
    expected: Any
    bug: Outcome


@dataclass
class Instance:
    id: str
    template: Template
    bug: str
    index: int
    seed: int
    rendering: Rendering
    files: dict[str, str]
    reference: dict[str, str]
    correctness: str
    preservation: str
    task: str
    target_key: str
    allowed: tuple[str, ...]
    public_tests: tuple[str, ...]
    tests: list[PublicTest]
    checks: list[Check]
    designed_failing: list[str]
    traceback: dict[str, str] | None
    original_target: str
    mutants: list[mutation.Mutant] = field(default_factory=list)
    control_constant: Any = None

    @property
    def reference_target(self) -> str:
        return self.rendering.sources[self.template.target]

    def module_file(self, role: str) -> str:
        slot = self.template.family.func(role).module
        return self.rendering.modules[slot] + ".py"

    def name(self, role: str) -> str:
        return self.rendering.names[role]

    def case(self) -> cases.Case:
        return cases.Case(
            id=self.id,
            title=self.title,
            category=CATEGORY[self.template.workflow],
            workflow_type=self.template.workflow,
            structure=self.structure,
            task=self.task,
            files=dict(self.files),
            allowed_files=self.allowed,
            public_tests=self.public_tests,
            public_fails_before_fix=True,
            correctness=self.correctness,
            preservation=self.preservation,
            reference=dict(self.reference),
            targets=(self.target_key,),
            budgets=cases.Budgets(**BUDGETS),
            spec_sha256="",
        )

    @property
    def title(self) -> str:
        return f"{self.template.concept} ({self.template.id}, {self.bug})"

    @property
    def structure(self) -> str:
        if self.template.caller:
            return (
                f"cross-module (caller in {self.module_file(self.template.caller)}, cause in "
                f"{self.module_file(self.template.target)})"
            )
        if self.template.workflow == "W4":
            return "single module, replace a stub"
        return "single module"

    def params(self) -> dict[str, Any]:
        return dict(self.rendering.params)

    def signature(self) -> str:
        node = mutation.function_node(self.reference_target)
        return f"def {node.name}({ast.unparse(node.args)})"


def _module_text(
    template: Template, rendering: Rendering, slot: str, sources: Mapping[str, str]
) -> str:
    family = template.family
    roles = [role for role in template.roles() if family.func(role).module == slot]
    stdlib = sorted({line for role in roles for line in family.func(role).imports})
    imported: dict[str, set[str]] = {}
    for role in roles:
        for used in family.func(role).uses:
            other = family.func(used).module
            if other != slot:
                imported.setdefault(rendering.modules[other], set()).add(rendering.names[used])
    lines = list(stdlib)
    lines += [
        f"from {module} import {', '.join(sorted(imported[module]))}" for module in sorted(imported)
    ]
    parts = [f'"""{family.docs[slot]}"""']
    if lines:
        parts.append("\n".join(lines))
    body = "\n\n\n".join(sources[role] for role in roles)
    return "\n\n".join(parts) + "\n\n\n" + body + "\n"


def _slots(template: Template) -> list[str]:
    used = {template.family.func(role).module for role in template.roles()}
    return [slot for slot in template.family.modules if slot in used]


def _phrase(rng: random.Random, options: Sequence[str], fields: Mapping[str, str]) -> list[str]:
    """Every phrasing filled in, the seed's choice first."""
    first = rng.randrange(len(options))
    order = [options[first], *(text for i, text in enumerate(options) if i != first)]
    return [text.format(**fields) for text in order]


W1_PHRASES = (
    "Bug: {call} returns {got} instead of {want}. Fix the bug.",
    "{call} gives {got}, but the expected result is {want}. Find and fix the bug.",
    "The test {test} fails: {call} should return {want} but returns {got}. Fix the code.",
)
W1_RAISE_PHRASES = (
    "Bug: {call} raises {exc} instead of returning {want}. Fix the bug.",
    "The test {test} fails: {call} raises {exc}; it should return {want}. Fix the code.",
)
W1_SHORT_PHRASES = (
    "The public test {test} fails. Fix the bug it exposes.",
    "Bug: the test {test} fails. Fix the code under test.",
)
W3_PHRASES = (
    "Bug: {call} returns {got} instead of {want}. Fix the cause of the bug.",
    "{call} gives {got}, but the expected result is {want}. Find the cause and fix it.",
)
W3_RAISE_PHRASES = (
    "Bug: {call} raises {exc} instead of returning {want}. Fix the cause of the bug.",
)
W3_SHORT_PHRASES = (
    "The public test {test} fails. Fix the cause of the failure.",
    "Bug: the test {test} fails. Find the cause and fix it.",
)
W4_PHRASES = (
    "Feature: implement {name} as its docstring describes.",
    "Implement the stub {name} in {file} following its docstring.",
    "{name} in {file} is a stub. Implement it as documented.",
)
W5_PHRASES = (
    "Bug report: {call} fails:\n{traceback}\nFix the cause.",
    "Crash report for {call}:\n{traceback}\nFind the cause and fix it.",
)
W5_SHORT_PHRASES = ("Bug report: the test {test} fails:\n{traceback}\nFix the cause.",)
MAX_CALL_CHARS = 70


def _task(instance: Instance, rng: random.Random, encoder: TextEncoder | None) -> str:
    template = instance.template
    workflow = template.workflow
    first = next(test for test in instance.tests if test.name in instance.designed_failing)
    call = _call(instance.name(first.role), first.args)
    fields = {
        "call": call,
        "test": first.name,
        "want": _literal(first.expected),
        "got": "" if first.bug.raised else _literal(first.bug.value),
        "exc": first.bug.raised,
        "name": instance.name(template.target),
        "file": instance.module_file(template.target),
        "traceback": (instance.traceback or {}).get("text", ""),
    }
    if workflow == "W4":
        options = _phrase(rng, W4_PHRASES, fields)
    elif workflow == "W5":
        options = _phrase(rng, W5_PHRASES, fields) + _phrase(rng, W5_SHORT_PHRASES, fields)
    else:
        long_set, raise_set, short_set = (
            (W1_PHRASES, W1_RAISE_PHRASES, W1_SHORT_PHRASES)
            if workflow == "W1"
            else (W3_PHRASES, W3_RAISE_PHRASES, W3_SHORT_PHRASES)
        )
        chosen = raise_set if first.bug.raised else long_set
        options = _phrase(rng, chosen, fields) + _phrase(rng, short_set, fields)
    if len(call) + len(fields["want"]) + len(fields["got"]) > MAX_CALL_CHARS * 2:
        options = [text for text in options if call not in text] or options
    elif len(call) > MAX_CALL_CHARS:
        options = [text for text in options if call not in text] or options
    for text in options:
        if encoder is None or len(encoder.encode(text)) <= MAX_TASK_TOKENS:
            return text
    return options[0]


def _traceback(instance: Instance, test: PublicTest) -> dict[str, str]:
    """The task's traceback: the test frame, then the repository frames the spec function's
    own traceback went through (a spec mirrors its function's calls), then the error."""
    chain = list(test.bug.frames)
    if not chain:
        raise TemplateError(f"{instance.id}: the spec raised outside every spec function")
    test_file = instance.public_tests[0]
    frames = [f'  File "{test_file}", in {test.name}']
    frames += [f'  File "{instance.module_file(role)}", in {instance.name(role)}' for role in chain]
    error = f"{test.bug.raised}: {test.bug.message}" if test.bug.message else test.bug.raised
    text = "\n".join(["Traceback (most recent call last):", *frames, error])
    return {
        "text": text,
        "error_line": error,
        "raise_file": instance.module_file(chain[-1]),
        "raise_function": instance.name(chain[-1]),
        "test": test.name,
    }


def build(
    template: Template,
    bug: str,
    index: int,
    seed_base: int,
    *,
    mutation_kinds: Sequence[str] = (),
    encoder: TextEncoder | None = None,
) -> Instance:
    """Build one instance; raises :class:`TemplateError` when the template cannot give one."""
    seed = instance_seed(seed_base, template.id, bug, index)
    rng = random.Random(seed)
    rendering = render(template, rng, bug)
    family = template.family
    params = rendering.params
    original_target = bug_text(template, rendering, bug)
    slots = _slots(template)
    reference_sources = dict(rendering.sources)
    original_sources = {**reference_sources, template.target: original_target}
    files: dict[str, str] = {}
    reference: dict[str, str] = {}
    target_slot = family.func(template.target).module
    for slot in slots:
        path = rendering.modules[slot] + ".py"
        files[path] = _module_text(template, rendering, slot, original_sources)
        if slot == target_slot:
            reference[path] = _module_text(template, rendering, slot, reference_sources)
    roles = template.roles()
    dependents = template.dependents()
    lead = template.caller or template.target
    ordered = [lead] + [role for role in roles if role != lead]
    tests: list[PublicTest] = []
    for role in ordered:
        if template.caller and role == template.target:
            continue  # A cross-module case reaches its target only through the caller.
        for label, args in inputs(family.func(role).public, params):
            expected = outcome(template, role, params, args)
            buggy = outcome(template, role, params, args, bug)
            tests.append(
                PublicTest(
                    f"test_{rendering.names[role]}_{label}", role, args, expected.value, buggy
                )
            )
    names = [test.name for test in tests]
    if len(set(names)) != len(names):
        raise TemplateError(f"{template.id}: repeated public test names")
    lead_file = rendering.modules[family.func(lead).module] + ".py"
    test_path = "test_" + lead_file
    imports: dict[str, set[str]] = {}
    for test in tests:
        module = rendering.modules[family.func(test.role).module]
        imports.setdefault(module, set()).add(rendering.names[test.role])
    header = "\n".join(
        f"from {module} import {', '.join(sorted(imports[module]))}" for module in sorted(imports)
    )
    bodies = [
        f"def {test.name}():\n    got = {_call(rendering.names[test.role], test.args)}\n"
        f"    assert got == {_literal(test.expected)}, repr(got)"
        for test in tests
    ]
    files[test_path] = header + "\n\n\n" + "\n\n\n".join(bodies) + "\n"
    designed = [test.name for test in tests if not test.bug.same(Outcome(value=test.expected))]
    checks: list[Check] = []
    for role in ordered:
        for args in inputs(family.func(role).hidden, params):
            expected = outcome(template, role, params, args)
            buggy = outcome(template, role, params, args, bug)
            checks.append(Check(role, tuple(args), expected.value, not buggy.same(expected)))
    correctness = [check for check in checks if check.role in dependents]
    preservation = [check for check in checks if check.role not in dependents or not check.exposes]
    target_checks = [check for check in correctness if check.role == template.target]

    def script(selected: list[Check]) -> str:
        imported: dict[str, set[str]] = {}
        for check in selected:
            module = rendering.modules[family.func(check.role).module]
            imported.setdefault(module, set()).add(rendering.names[check.role])
        head = "\n".join(
            f"from {module} import {', '.join(sorted(imported[module]))}"
            for module in sorted(imported)
        )
        lines = [
            f"assert {_call(rendering.names[check.role], check.args)} == "
            f"{_literal(check.expected)}, {('check ' + str(number))!r}"
            for number, check in enumerate(selected, 1)
        ]
        return head + "\n\n" + "\n".join(lines) + "\n"

    allowed = sorted(
        {
            rendering.modules[family.func(role).module] + ".py"
            for role in [template.target, *([template.caller] if template.caller else [])]
        }
    )
    instance = Instance(
        id=instance_id(template.id, bug, index),
        template=template,
        bug=bug,
        index=index,
        seed=seed,
        rendering=rendering,
        files=files,
        reference=reference,
        correctness=script(correctness),
        preservation=script(preservation) if preservation else "",
        task="",
        target_key=f"{rendering.modules[target_slot]}.py::{rendering.names[template.target]}",
        allowed=tuple(allowed),
        public_tests=(test_path,),
        tests=tests,
        checks=checks,
        designed_failing=designed,
        traceback=None,
        original_target=original_target,
        control_constant=target_checks[0].expected if target_checks else None,
    )
    if not designed:
        raise TemplateError(f"{instance.id}: no public test fails before the fix")
    if not any(check.exposes for check in correctness):
        raise TemplateError(f"{instance.id}: the original passes every correctness check")
    if len(target_checks) < MIN_CORRECTNESS_CHECKS:
        raise TemplateError(f"{instance.id}: fewer than {MIN_CORRECTNESS_CHECKS} target checks")
    if not preservation:
        raise TemplateError(f"{instance.id}: no preservation checks")
    if template.workflow == "W5":
        first = next(test for test in tests if test.name in designed)
        if not first.bug.raised:
            raise TemplateError(f"{instance.id}: a traceback case's first failing test must raise")
        instance.traceback = _traceback(instance, first)
    instance.task = _task(instance, rng, encoder)
    forbid = [original_target]
    for kind in mutation_kinds:
        chosen = mutation.choose(instance.reference_target, kind, rng, forbid=forbid)
        instance.mutants.append(chosen)
        forbid.append(chosen.source)
    return instance


@dataclass(frozen=True)
class Blocklist:
    """Evaluation-family names and modules a generated repository must not use."""

    names: frozenset[str]
    modules: frozenset[str]

    @classmethod
    def load(cls, path: Path) -> Blocklist:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema") != "agent-eval-families-v1":
            raise ValueError("unknown families schema")
        names = set(raw.get("all_defined_source_names", []))
        names |= set(raw.get("all_target_functions", []))
        names |= set(raw.get("all_test_function_names", []))
        modules = {name.removesuffix(".py") for name in raw.get("all_source_modules", [])}
        for case in raw.get("cases", []):
            for defined in case.get("defined_names", {}).values():
                names |= set(defined)
            for tests in case.get("test_modules", {}).values():
                names |= set(tests)
            modules |= {name.removesuffix(".py") for name in case.get("source_modules", [])}
        return cls(frozenset(names), frozenset(modules))


def defined_names(files: Mapping[str, str]) -> set[str]:
    """Top-level function, class and assigned names of every file."""
    found: set[str] = set()
    for path, text in files.items():
        for node in workspace.parse(text, path).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.add(node.name)
            elif isinstance(node, ast.Assign):
                found |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return found


def _functions(text: str, path: str) -> list[str]:
    return [
        workspace.function_source(text, function)
        for function in workspace.functions(text, path).values()
    ]


def static_check(
    instance: Instance, encoder: TextEncoder, blocklist: Blocklist | None = None
) -> dict[str, Any]:
    """Every check that needs no execution; ``problems`` is empty when the instance is kept."""
    problems: list[str] = []
    case = instance.case()
    try:
        cases.check_case_id(case.id)
        cases.validate_static(case)
    except cases.CaseError as error:
        return {"problems": [f"case: {error}"]}
    sources = [
        *(
            item
            for path, text in case.files.items()
            if not workspace.is_test_file(path)
            for item in _functions(text, path)
        ),
        *(item for path, text in case.reference.items() for item in _functions(text, path)),
        *(item.source for item in instance.mutants),
    ]
    for source in sources:
        lines = source.count("\n") + 1
        tokens = len(encoder.encode(source))
        if lines > MAX_FUNCTION_LINES or tokens > prompt.FUNCTION_BUDGET:
            name = source.split("(")[0].removeprefix("def ")
            problems.append(f"function {name}: {lines} lines, {tokens} tokens")
    task_tokens = len(encoder.encode(case.task))
    if task_tokens > MAX_TASK_TOKENS:
        problems.append(f"task: {task_tokens} tokens")
    plans: dict[str, Any] = {}
    for label, failing in (("all_tests", None), ("designed_failing", instance.designed_failing)):
        try:
            plan = plan_case(case, encoder, failing=failing)
        except prompt.ContextLimit as error:
            problems.append(f"prompt ({label}): {error}")
            continue
        plans[label] = plan
        if not plan["fits"]:
            problems.append(f"prompt ({label}): does not fit the section 3.5 bound")
        if not plan["targets_shown"]:
            problems.append(f"prompt ({label}): the target is not shown")
    if len(instance.mutants) != len({item.source for item in instance.mutants}):
        problems.append("repeated mutants")
    if blocklist is not None:
        names = defined_names(case.files) | defined_names(case.reference)
        for text in case.files.values():
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.FunctionDef):
                    names.add(node.name)
        if names & blocklist.names:
            problems.append("a defined name is on the evaluation-family blocklist")
        modules = {path.removesuffix(".py") for path in case.files}
        modules |= {path.removeprefix("test_") for path in modules}
        if modules & blocklist.modules:
            problems.append("a module name is on the evaluation-family blocklist")
    return {"problems": problems, "plan": plans}


def mutation_cycle(kinds: Sequence[str], start: int, count: int) -> list[str]:
    """``count`` distinct kinds from ``kinds`` (cyclic from ``start``), for round-robin use."""
    if not kinds:
        return []
    count = min(count, len(kinds))
    return [kinds[(start + offset) % len(kinds)] for offset in range(count)]


def record(instance: Instance, static: Mapping[str, Any]) -> dict[str, Any]:
    """The instance's ``synth.json``: everything needed to rebuild and audit it."""
    template = instance.template
    return {
        "schema": INSTANCE_SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "id": instance.id,
        "template_id": template.id,
        "family": template.family.id,
        "domain": template.family.domain,
        "concept": template.concept,
        "workflow_type": template.workflow,
        "bug_kind": instance.bug,
        "index": instance.index,
        "seed": instance.seed,
        "params": instance.params(),
        "oracle_params": {key: instance.params()[key] for key in param_keys(template)},
        "names": instance.rendering.names,
        "modules": instance.rendering.modules,
        "variants": instance.rendering.variants,
        "target": instance.target_key,
        "signature": instance.signature(),
        "designed_failing_tests": instance.designed_failing,
        "traceback": instance.traceback,
        "original_target": instance.original_target,
        "reference_target": instance.reference_target,
        "mutants": [
            {
                "kind": item.kind,
                "site": item.site,
                "source": item.source,
                "sha256": hashlib.sha256(item.source.encode()).hexdigest(),
            }
            for item in instance.mutants
        ],
        "control_constant": instance.control_constant,
        "checks": {
            "correctness": sum(1 for c in instance.checks if c.role in template.dependents()),
            "correctness_exposing": sum(
                1 for c in instance.checks if c.role in template.dependents() and c.exposes
            ),
        },
        "static": {"plan": static.get("plan", {})},
    }


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write(root: Path, instance: Instance, static: Mapping[str, Any]) -> dict[str, str]:
    """Write the case directory ``root/<id>/`` (agent-case-v1 plus ``synth.json``); returns
    the SHA-256 of every file written, by relative path."""
    directory = root / instance.id
    directory.mkdir(parents=True, exist_ok=False)
    written: dict[str, str] = {}

    def put(relative: str, text: str) -> None:
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written[relative] = _sha(text)

    for path, text in sorted(instance.files.items()):
        put(f"repo/{path}", text)
    for path, text in sorted(instance.reference.items()):
        put(f"reference/{path}", text)
    put("hidden/correctness.py", instance.correctness)
    put("hidden/preservation.py", instance.preservation)
    template = instance.template
    spec = {
        "schema": cases.CASE_SCHEMA,
        "id": instance.id,
        "title": instance.title,
        "category": CATEGORY[template.workflow],
        "workflow_type": template.workflow,
        "structure": instance.structure,
        "task": instance.task,
        "allowed_files": list(instance.allowed),
        "public_tests": list(instance.public_tests),
        "reference": {path: f"reference/{path}" for path in sorted(instance.reference)},
        "targets": [instance.target_key],
        "split": "dev",
        "set": CASE_SET,
        "repo": "repo",
        "public_fails_before_fix": True,
        "hidden": {
            "correctness": "hidden/correctness.py",
            "preservation": "hidden/preservation.py",
        },
        "budgets": BUDGETS,
        "provenance": {
            "author": f"generator {GENERATOR_VERSION} (scratchcoder.agent.synth)",
            "source": "generated from a project template; not adapted from any dataset, "
            "benchmark, pool task or public repository",
            "template_id": template.id,
            "family": template.family.id,
            "seed": instance.seed,
        },
    }
    put("case.json", json.dumps(spec, indent=2, sort_keys=True) + "\n")
    put("synth.json", json.dumps(record(instance, static), indent=1, sort_keys=True) + "\n")
    return written


def plan_instances(templates: Iterable[Template], per_bug: int) -> list[tuple[Template, str, int]]:
    """(template, bug kind, index) for ``per_bug`` instances of every bug kind, in order."""
    return [
        (template, bug, index)
        for template in templates
        for bug in template.bugs
        for index in range(per_bug)
    ]
