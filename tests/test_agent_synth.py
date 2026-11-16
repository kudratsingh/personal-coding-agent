"""The synthetic agent-data generator: templates, instances, mutants and static checks.

Nothing here executes generated source: instances are parsed with ``ast``; expected values
come from the spec functions (project code)."""

from __future__ import annotations

import ast
import json
import random
from collections import Counter
from pathlib import Path

import pytest
from synth_fixtures import TOY, W1, W3, W4, W5, Chunks, families_json

from scratchcoder.agent import cases, workspace
from scratchcoder.agent.synth import instances, mutation
from scratchcoder.agent.synth.catalog import catalog
from scratchcoder.agent.synth.template import (
    Bug,
    Func,
    Template,
    TemplateError,
    canonical,
    outcome,
    render,
)


def test_the_catalog_meets_the_breadth_rules() -> None:
    found = catalog()
    assert len(found.templates) >= 40
    assert len({template.family.id for template in found.templates}) >= 12
    assert {template.workflow for template in found.templates} == {"W1", "W3", "W4", "W5"}
    for template in found.templates:
        assert len(template.bugs) >= 2, template.id
        assert 2 <= len({template.family.func(r).module for r in template.roles()}) <= 3
        if template.workflow == "W3":
            assert template.caller and template.family.func(template.caller).module != (
                template.family.func(template.target).module
            )
        assert cases.CASE_ID.match(instances.instance_id(template.id, max(template.bugs), 999))


def test_every_template_and_bug_kind_builds_with_parseable_canonical_sources() -> None:
    encoder = Chunks()
    for template in catalog().templates:
        for bug in template.bugs:
            instance = instances.build(template, bug, 0, 0, encoder=encoder)
            for path, text in [*instance.files.items(), *instance.reference.items()]:
                tree = ast.parse(text)
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        source = workspace.function_source(
                            text, workspace.functions(text, path)[node.name]
                        )
                        assert source == canonical(source)
                        assert source.count("\n") + 1 <= instances.MAX_FUNCTION_LINES
            ast.parse(instance.correctness)
            ast.parse(instance.preservation)
            assert instance.designed_failing
            assert any(check.exposes for check in instance.checks)
            cases.validate_static(instance.case())
            kinds = mutation.applicable(
                instance.reference_target, forbid=[instance.original_target]
            )
            assert len(kinds) >= 2, (template.id, kinds)


def test_instances_are_deterministic_and_seeds_vary_them() -> None:
    first = instances.build(W1, "factor_up", 0, 7)
    again = instances.build(W1, "factor_up", 0, 7)
    assert first.files == again.files and first.task == again.task
    assert first.seed != instances.build(W1, "factor_up", 1, 7).seed
    variants = {
        tuple(sorted(instances.build(W1, "factor_up", index, 7).files.items()))
        for index in range(8)
    }
    assert len(variants) > 1


def test_expected_values_come_from_the_spec_and_the_bug_model() -> None:
    instance = instances.build(W1, "factor_up", 0, 0)
    k = instance.params()["k"]
    name = instance.name("scale")
    test_text = instance.files[instance.public_tests[0]]
    assert f"got = {name}(2)\n    assert got == {2 * k}, repr(got)" in test_text
    assert instance.designed_failing == [f"test_{name}_two"]  # 0 * (k + 1) == 0 * k
    assert f"assert {name}(5) == {5 * k}, 'check 3'" in instance.correctness
    # The check whose outcome the bug leaves unchanged (x = 0) is also a preservation check.
    assert f"{name}(0) == 0" in instance.preservation
    assert f"{name}(2)" in instance.task and str(2 * (k + 1)) in instance.task


def test_cross_module_case_reaches_its_target_only_through_the_caller() -> None:
    instance = instances.build(W3, "factor_up", 0, 0)
    test_text = instance.files[instance.public_tests[0]]
    caller = instance.name("scale_all")
    assert f"def test_{caller}_pair" in test_text
    assert f"def test_{instance.name('scale')}_" not in test_text
    assert set(instance.allowed) == {
        instance.module_file("scale"),
        instance.module_file("scale_all"),
    }
    assert instance.structure.startswith("cross-module")
    assert (
        f"from {instance.rendering.modules['main']} import"
        in instance.files[instance.module_file("scale_all")]
    )


def test_a_stub_keeps_the_docstring_and_fails_every_target_test() -> None:
    raised = instances.build(W4, "stub_raise", 0, 0)
    none = instances.build(W4, "stub_none", 0, 0)
    assert raised.original_target.endswith('"""\n    raise NotImplementedError')
    assert none.original_target.endswith('"""\n    return None')
    assert raised.designed_failing == [f"test_{raised.name('first_item')}_one"]
    assert raised.task.endswith(("describes.", "docstring.", "documented."))


def test_a_traceback_task_names_the_frames_the_spec_raised_through() -> None:
    instance = instances.build(W5, "wrong_index", 0, 0)
    assert instance.traceback is not None
    assert instance.traceback["error_line"] == "IndexError: list index out of range"
    assert instance.traceback["raise_function"] == instance.name("first_item")
    assert "Traceback (most recent call last):" in instance.task
    assert f'File "{instance.module_file("first_item")}", in {instance.name("first_item")}' in (
        instance.task
    )


def test_bug_edits_must_apply_exactly_once_and_sources_must_be_canonical() -> None:
    broken = Template(
        id="toyx",
        family=TOY,
        workflow="W1",
        target="scale",
        concept="c",
        bugs={"missing": Bug(edits=(("no such text", "x"),))},
        spec="s",
        domain="d",
        returns="int",
        include=("offset",),
    )
    with pytest.raises(TemplateError, match="no variant takes the bug edits"):
        instances.build(broken, "missing", 0, 0)
    loose = TOY.__class__(
        **{
            **TOY.__dict__,
            "funcs": (
                Func("scale", "main", "def $scale(x):\n    return (x * $k)", TOY.funcs[0].spec),
                *TOY.funcs[1:],
            ),
        }
    )
    template = Template(**{**W1.__dict__, "family": loose})
    with pytest.raises(TemplateError, match="canonical"):
        render(template, random.Random(0), "factor_up")


def test_the_reference_spec_must_not_raise() -> None:
    with pytest.raises(TemplateError, match="reference spec"):
        outcome(W5, "first_item", {"k": 3}, ([],))


def _target_spec(template_id: str, *args: object, **params: object) -> object:
    """The target's spec function (project code) on ``args``, with the template's constants
    set to their first candidates unless given."""
    template = next(item for item in catalog().templates if item.id == template_id)
    values = {key: choices[0] for key, choices in template.family.consts.items()}
    values.update(params)
    return template.family.func(template.target).spec(values, *args)


@pytest.mark.parametrize(
    ("template_id", "args", "params", "expected", "phrase"),
    [
        ("stock1", ({}, "a", 0), {}, 0, "even when amount is 0"),
        ("stock2", ({}, "a", 0), {}, 0, "added with the value 0"),
        ("stock2", ({"a": 1}, "b", 1), {}, -1, "no key is added"),
        ("grid2", ([[1, 1], [1, 1]], -1, 0), {}, 0, "does not wrap around"),
        ("grid2", ([[1, 1], [1, 1]], 0, -1), {}, 0, "does not wrap around"),
        ("grid3", ([[1, 0]],), {}, [[1], [0]], "rows are new lists"),
        ("sensor2", ([],), {}, [], "An empty list gives []"),
        ("graph1", ({"a": ["a", "b", "b"], "b": []},), {}, {"a": 1, "b": 2}, "self-loop"),
        ("graph2", ([("b", "a")],), {}, {"b": ["a"], "a": []}, "only ever a target"),
        ("graph3", ({"a": ["b"]}, "z"), {}, ["z"], "gives [start]"),
        ("jobs2", (["a", "b", "a"],), {"gap": 2}, 4, "previous list entry + 1"),
        ("jobs2", (["a", "a"],), {"gap": 1}, 3, "0 for the first job"),
        ("votes1", (["b", "a", "b", "a"],), {}, "a", "Python str order"),
        ("votes2", ([["a", "c"], ["a"]],), {}, [["c"], []], "never first is not counted"),
        ("votes2", ([["a"], ["b", "a"], ["b"]],), {}, [[], ["b"], ["b"]], "stays as []"),
        ("digits3", (0,), {}, True, "The number 0"),
        ("points3", ([(3, 4)],), {}, 0, "a single point gives 0"),
        ("poly2", ([7, 2, 5], 0), {}, 7, "0 included"),
        ("rpn1", ([2, 7, "-", 3, "//"],), {}, -2, "floor division"),
        ("rpn1", ([1, 0, 3, "-", "//"],), {}, -1, "floor division"),
        ("rpn2", ([1, "+", 2, 3, "+"],), {}, False, "at once, whatever follows"),
        ("rpn2", ([],), {}, False, "empty list gives False"),
        ("rpn3", ([],), {}, 0, "the empty list"),
        ("flags1", (63,), {"order": ["r", "w", "x"]}, ["r", "w", "x"], "are ignored"),
        ("flags1", (5,), {"order": ("r", "w", "x")}, ["r", "x"], "a list in the instance"),
        ("flags2", (["r", "r", "rs", "w"],), {"order": ("r", "w", "x")}, 3, "exact equality"),
    ],
)
def test_spec_text_states_each_edge_case_the_spec_function_settles(
    template_id: str,
    args: tuple[object, ...],
    params: dict[str, object],
    expected: object,
    phrase: str,
) -> None:
    """Edge cases an independent reader found ambiguous: the spec text (what the oracle author
    reads) must state the case, and the spec function (what the expected values come from)
    must give the stated answer, with the stated type."""
    template = next(item for item in catalog().templates if item.id == template_id)
    assert phrase in template.spec + " " + template.domain
    found = _target_spec(template_id, *args, **params)
    assert type(found) is type(expected) and found == expected


def test_spec_functions_leave_their_inputs_unchanged() -> None:
    levels = {"a": 1}
    grid = [[1, 0], [0, 1]]
    ballots = [["a", "b"], ["b"]]
    _target_spec("stock2", levels, "a", 1)
    rotated = _target_spec("grid3", grid)
    _target_spec("votes2", ballots)
    assert levels == {"a": 1} and grid == [[1, 0], [0, 1]] and ballots == [["a", "b"], ["b"]]
    assert isinstance(rotated, list) and all(row is not grid[0] for row in rotated)


def test_mutants_keep_name_and_signature_and_differ_from_reference_and_forbidden() -> None:
    source = (
        "def pick(values, limit=3):\n"
        "    best = 0\n"
        "    for i in range(1, len(values)):\n"
        "        if values[i] > values[best] and values[i] < limit:\n"
        "            best = i\n"
        "    return max(best, 0)"
    )
    reference = mutation.function_node(source)
    seen: Counter[str] = Counter()
    for kind in mutation.MUTATION_KINDS:
        for item in mutation.mutants(source, kind):
            node = mutation.function_node(item.source)
            assert node.name == "pick"
            assert ast.dump(node.args) == ast.dump(reference.args)
            assert ast.dump(node) != ast.dump(reference)
            seen[kind] += 1
    assert set(seen) == set(mutation.MUTATION_KINDS)
    first = mutation.mutants(source, "flip_comparison")[0]
    rest = mutation.mutants(source, "flip_comparison", forbid=[first.source])
    assert first.source not in {item.source for item in rest}
    with pytest.raises(mutation.MutationError):
        mutation.choose("def f():\n    return None", "swap_arguments", random.Random(0))


def test_drop_branch_removes_an_if_or_keeps_only_its_else() -> None:
    source = "def f(x):\n    if x:\n        return 1\n    else:\n        return 2"
    found = {item.source for item in mutation.mutants(source, "drop_branch")}
    assert found == {"def f(x):\n    return 2"}


def test_negative_controls_have_trivial_bodies() -> None:
    source = 'def f(a, b):\n    """Doc."""\n    return a + b'
    assert mutation.control(source, "none") == "def f(a, b):\n    return None"
    assert mutation.control(source, "delete") == "def f(a, b):\n    pass"
    assert mutation.control(source, "echo") == "def f(a, b):\n    return a"
    assert (
        mutation.control(source, "constant", [1, (2, 3)]) == "def f(a, b):\n    return [1, (2, 3)]"
    )


def test_static_check_applies_the_plan_rule_sizes_and_blocklist(tmp_path: Path) -> None:
    encoder = Chunks()
    instance = instances.build(
        W1, "factor_up", 0, 0, mutation_kinds=["off_by_one"], encoder=encoder
    )
    clean = instances.Blocklist.load(families_json(tmp_path / "families.json"))
    result = instances.static_check(instance, encoder, clean)
    assert result["problems"] == []
    assert result["plan"]["all_tests"]["targets_shown"] and result["plan"]["all_tests"]["fits"]
    blocked = instances.Blocklist.load(
        families_json(tmp_path / "blocked.json", names=[instance.name("offset")])
    )
    assert instances.static_check(instance, encoder, blocked)["problems"] == [
        "a defined name is on the evaluation-family blocklist"
    ]


def test_an_instance_writes_a_loadable_registered_case(tmp_path: Path) -> None:
    encoder = Chunks()
    instance = instances.build(
        W3, "factor_up", 0, 0, mutation_kinds=["off_by_one"], encoder=encoder
    )
    static = instances.static_check(instance, encoder)
    instances.write(tmp_path, instance, static)
    (tmp_path / cases.REGISTRY_NAME).write_text(
        json.dumps(cases.build_registry(tmp_path)), encoding="utf-8"
    )
    (loaded,) = cases.load_cases(tmp_path)
    assert loaded.id == instance.id
    assert loaded.targets == (instance.target_key,)
    assert loaded.reference_function(instance.target_key) == instance.reference_target
    record = json.loads((tmp_path / instance.id / "synth.json").read_text())
    assert record["mutants"][0]["kind"] == "off_by_one"
    assert record["oracle_params"] == {"k": instance.params()["k"]}


def test_mutation_cycle_spreads_kinds() -> None:
    kinds = ["a", "b", "c"]
    assert instances.mutation_cycle(kinds, 0, 2) == ["a", "b"]
    assert instances.mutation_cycle(kinds, 2, 2) == ["c", "a"]
    assert instances.mutation_cycle(kinds, 0, 5) == ["a", "b", "c"]
    assert instances.mutation_cycle([], 0, 2) == []
