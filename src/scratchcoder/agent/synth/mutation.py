"""AST mutants of a reference function (the wrong attempts of repair rows) and the negative
controls. Everything here is a syntax-tree transformation; nothing is executed.

A mutant is the reference function with one small change at one site. It is kept only if
its text parses back into one function with the same name and the same signature (the
``arguments`` node, defaults included), and its tree differs from the reference's and from
every tree in ``forbid`` (the original buggy version, other mutants). Trees are compared
after a small canonical rewrite (:func:`canonical_dump`): ``not a < b`` is ``a >= b`` (one
comparison, its logical inverse) and ``x % 2 == 1`` is ``x % 2 != 0``, so a negation that
only re-expresses the reference or the original bug is not a new mutant.

Some sites are skipped because the change is known, from the code alone, to give a mutant
that cannot fail (behaves like the reference on every valid input) or that cannot finish:

- ``swap_arguments`` on a symmetric call: ``max``/``min``, a helper the caller names as
  symmetric (``symmetric``), and a ``zip`` of two sequences whose pair is used only
  symmetrically (``a == b``, ``a != b``, ``a + b``, ``a * b``, ...);
- ``off_by_one`` that turns ``len(x[0])`` into ``len(x[-1])`` (the same on rectangular data),
  on the default of ``d.get(key, default)`` when ``d`` was built by a dict comprehension
  (every key is already present), and that writes ``-0``;
- ``wrong_operator`` on the step of ``f(x[i], x[(i + 1) % n])`` with ``f`` symmetric (over a
  cycle, ``i - 1`` gives the same total);
- in a ``while`` loop: ``flip_comparison``, ``negate_condition`` and every operator inside the
  loop test; the same in an ``if`` that guards adding to the loop's worklist; and
  ``wrong_operator``, ``swap_arguments`` and ``off_by_one`` in a statement that updates a
  name the loop test reads (except an ``x op= constant`` step whose new constant is not 0
  or 1).

Whether a mutant also *behaves* differently on the public tests is not decided here: the
row builder keeps a repair row only when the sandbox shows the mutant failing them.
"""

from __future__ import annotations

import ast
import copy
import random
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

MUTATION_KINDS = (
    "off_by_one",
    "flip_comparison",
    "wrong_operator",
    "drop_branch",
    "wrong_return",
    "swap_arguments",
    "wrong_initial_value",
    "negate_condition",
)
CONTROL_KINDS = ("none", "delete", "echo", "constant")
"""Negative controls: ``return None``; the body deleted (``pass``); return the first
parameter; return a constant (the first hidden check's expected value)."""

_COMPARE = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}
_BINARY = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Add,
    ast.FloorDiv: ast.Mod,
    ast.Mod: ast.FloorDiv,
    ast.BitAnd: ast.BitOr,
    ast.BitOr: ast.BitAnd,
    ast.BitXor: ast.BitOr,
    ast.LShift: ast.RShift,
    ast.RShift: ast.LShift,
}


_INVERSE = {
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}
"""The logical inverse of each comparison (``not a < b`` is ``a >= b``)."""
SYMMETRIC_CALLS = frozenset({"max", "min"})
"""Builtins whose two positional arguments can be swapped without changing the result."""
_COMMUTATIVE = (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)
_WORKLIST_ADDS = frozenset({"append", "add", "extend", "insert", "appendleft"})


class MutationError(ValueError):
    pass


def function_node(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise MutationError("expected one function definition")
    return tree.body[0]


def _body_nodes(function: ast.FunctionDef) -> Iterator[ast.AST]:
    """Every node of the body (breadth first), the docstring excluded."""
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    queue: list[ast.AST] = list(body)
    while queue:
        node = queue.pop(0)
        yield node
        queue.extend(ast.iter_child_nodes(node))


def _statement_lists(function: ast.FunctionDef) -> Iterator[list[ast.stmt]]:
    for node in [function, *_body_nodes(function)]:
        for name in ("body", "orelse", "finalbody"):
            value = getattr(node, name, None)
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                yield value


def _initial_assign(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and (node.value.value is None or isinstance(node.value.value, (bool, int)))
    )


def _initial_constants(function: ast.FunctionDef) -> set[int]:
    """ids of constants that are the whole value of a ``name = constant`` assignment."""
    found = set()
    for node in _body_nodes(function):
        if _initial_assign(node):
            assert isinstance(node, ast.Assign)
            found.add(id(node.value))
    return found


def _is_int(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and type(node.value) is int
        and not isinstance(node.value, bool)
    )


Site = tuple[int, Any]
"""(position of the node in the body's breadth-first order, the kind's variant)."""


@dataclass
class _Context:
    """What the skip rules need to know about where a node sits."""

    parents: dict[int, ast.AST] = field(default_factory=dict)
    loop_test: set[int] = field(default_factory=set)
    """Nodes inside a ``while`` test or inside the test of an ``if`` that guards adding to
    the loop's worklist."""
    guards: set[int] = field(default_factory=set)
    """``if`` statements that guard adding to a ``while`` loop's worklist."""
    loop_update: set[int] = field(default_factory=set)
    """Nodes inside a statement that assigns a name a ``while`` test reads."""
    prefilled: set[str] = field(default_factory=set)
    """Names assigned a dict comprehension."""


def _base_name(target: ast.AST) -> str | None:
    while isinstance(target, (ast.Subscript, ast.Attribute)):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def _context(function: ast.FunctionDef) -> _Context:
    found = _Context()
    for node in ast.walk(function):
        for child in ast.iter_child_nodes(node):
            found.parents[id(child)] = node
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.DictComp)
        ):
            found.prefilled.add(node.targets[0].id)
    for loop in (node for node in ast.walk(function) if isinstance(node, ast.While)):
        names = {item.id for item in ast.walk(loop.test) if isinstance(item, ast.Name)}
        found.loop_test |= {id(item) for item in ast.walk(loop.test)}
        for statement in (item for body in loop.body for item in ast.walk(body)):
            if isinstance(statement, (ast.Assign, ast.AugAssign)):
                targets = (
                    statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                )
                if any(_base_name(target) in names for target in targets):
                    found.loop_update |= {id(item) for item in ast.walk(statement)}
            if isinstance(statement, ast.If) and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in _WORKLIST_ADDS
                and _base_name(call.func.value) in names
                for branch in statement.body
                for call in ast.walk(branch)
            ):
                found.guards.add(id(statement))
                found.loop_test |= {id(item) for item in ast.walk(statement.test)}
    return found


def _name_of(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _symmetric_zip(function: ast.FunctionDef, call: ast.Call) -> bool:
    """``zip(a, b)`` whose pair is only used symmetrically, so swapping a and b changes
    nothing."""
    if _name_of(call) != "zip" or len(call.args) != 2 or call.keywords:
        return False
    scope: ast.AST | None = None
    target: ast.AST | None = None
    for node in ast.walk(function):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for generator in node.generators:
                if generator.iter is call:
                    scope, target = node, generator.target
        elif isinstance(node, ast.For) and node.iter is call:
            scope, target = node, node.target
    if (
        scope is None
        or not isinstance(target, ast.Tuple)
        or len(target.elts) != 2
        or not all(isinstance(item, ast.Name) for item in target.elts)
    ):
        return False
    pair = {item.id for item in target.elts if isinstance(item, ast.Name)}
    if len(pair) != 2:
        return False
    parents = {id(child): node for node in ast.walk(scope) for child in ast.iter_child_nodes(node)}
    uses = [
        item
        for item in ast.walk(scope)
        if isinstance(item, ast.Name) and item.id in pair and isinstance(item.ctx, ast.Load)
    ]
    if not uses:
        return False
    for use in uses:
        parent = parents.get(id(use))
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, _COMMUTATIVE):
            operands = [parent.left, parent.right]
        elif (
            isinstance(parent, ast.Compare)
            and len(parent.ops) == 1
            and isinstance(parent.ops[0], (ast.Eq, ast.NotEq))
        ):
            operands = [parent.left, parent.comparators[0]]
        else:
            return False
        names = {item.id for item in operands if isinstance(item, ast.Name)}
        if names != pair:
            return False
    return True


def _skip_off_by_one(context: _Context, node: ast.Constant, variant: int) -> bool:
    assert type(node.value) is int
    result = node.value + variant
    parent = context.parents.get(id(node))
    grand = context.parents.get(id(parent)) if parent is not None else None
    if (
        isinstance(parent, ast.Subscript)
        and parent.slice is node
        and isinstance(grand, ast.Call)
        and _name_of(grand) == "len"
        and result == -1
    ):
        return True  # len(x[0]) -> len(x[-1]): the same on rectangular data.
    if (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Attribute)
        and parent.func.attr == "get"
        and len(parent.args) == 2
        and parent.args[1] is node
        and _base_name(parent.func.value) in context.prefilled
    ):
        return True  # Every key is present, so the default is never used.
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.USub):
        if result == 0:
            return True  # "-0": not code anyone writes.
    if id(node) in context.loop_test:
        return True
    if id(node) in context.loop_update:
        step = isinstance(parent, ast.AugAssign) and parent.value is node
        return not step or result in (0, 1)
    return False


def _cyclic_step(context: _Context, node: ast.AST, symmetric: Collection[str]) -> bool:
    """``i + 1`` in ``f(x[i], x[(i + 1) % n])`` with ``f`` symmetric: summed over a cycle,
    ``(i - 1) % n`` gives the same total."""
    parent = context.parents.get(id(node))
    subscript = context.parents.get(id(parent)) if parent is not None else None
    call = context.parents.get(id(subscript)) if subscript is not None else None
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, (ast.Add, ast.Sub))
        and isinstance(parent, ast.BinOp)
        and isinstance(parent.op, ast.Mod)
        and parent.left is node
        and isinstance(subscript, ast.Subscript)
        and subscript.slice is parent
        and isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in symmetric
    )


def sites(function: ast.FunctionDef, kind: str, *, symmetric: Collection[str] = ()) -> list[Site]:
    """The mutation sites of ``kind`` in ``function``, in a fixed order, without the sites
    the skip rules exclude. ``symmetric`` names helpers whose two arguments commute."""
    nodes = list(_body_nodes(function))
    context = _context(function)
    found: list[Site] = []
    if kind == "off_by_one":
        initial = _initial_constants(function)
        for pos, node in enumerate(nodes):
            if _is_int(node) and id(node) not in initial:
                assert isinstance(node, ast.Constant)
                found += [
                    (pos, variant)
                    for variant in (1, -1)
                    if not _skip_off_by_one(context, node, variant)
                ]
    elif kind == "flip_comparison":
        for pos, node in enumerate(nodes):
            if isinstance(node, ast.Compare) and id(node) not in context.loop_test:
                found += [(pos, i) for i, op in enumerate(node.ops) if type(op) in _COMPARE]
    elif kind == "wrong_operator":
        for pos, node in enumerate(nodes):
            if id(node) in context.loop_test or id(node) in context.loop_update:
                continue
            if _cyclic_step(context, node, symmetric):
                continue
            if isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in _BINARY:
                found.append((pos, None))
            elif isinstance(node, ast.BoolOp):
                found.append((pos, None))
    elif kind == "drop_branch":
        found = [(pos, None) for pos, node in enumerate(nodes) if isinstance(node, ast.If)]
    elif kind == "wrong_return":
        names = _local_names(function)
        for pos, node in enumerate(nodes):
            if isinstance(node, ast.Return) and node.value is not None:
                current = ast.dump(node.value)
                options: list[Any] = [
                    name for name in names if ast.dump(ast.Name(name, ast.Load())) != current
                ][:3]
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
                    options.insert(0, not node.value.value)
                found += [(pos, option) for option in options]
    elif kind == "swap_arguments":
        for pos, node in enumerate(nodes):
            if (
                isinstance(node, ast.Call)
                and len(node.args) >= 2
                and ast.dump(node.args[0]) != ast.dump(node.args[1])
                and not any(isinstance(arg, ast.Starred) for arg in node.args[:2])
                and id(node) not in context.loop_update
                and not (
                    isinstance(node.func, ast.Name)
                    and (node.func.id in SYMMETRIC_CALLS or node.func.id in symmetric)
                    and len(node.args) == 2
                )
                and not _symmetric_zip(function, node)
            ):
                found.append((pos, None))
    elif kind == "wrong_initial_value":
        found = [(pos, None) for pos, node in enumerate(nodes) if _initial_assign(node)]
    elif kind == "negate_condition":
        found = [
            (pos, None)
            for pos, node in enumerate(nodes)
            if isinstance(node, (ast.If, ast.IfExp)) and id(node) not in context.guards
        ]
    else:
        raise MutationError(f"unknown mutation kind {kind!r}")
    return found


def apply_site(function: ast.FunctionDef, kind: str, site: Site) -> None:
    """Apply one site of ``kind`` to ``function`` in place. ``function`` is a fresh copy of
    the tree :func:`sites` enumerated, so the positions match."""
    pos, variant = site
    node = list(_body_nodes(function))[pos]
    if kind == "off_by_one":
        assert isinstance(node, ast.Constant) and isinstance(node.value, int)
        node.value = node.value + int(variant)
    elif kind == "flip_comparison":
        assert isinstance(node, ast.Compare)
        node.ops[variant] = _COMPARE[type(node.ops[variant])]()
    elif kind == "wrong_operator":
        if isinstance(node, ast.BoolOp):
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        else:
            assert isinstance(node, (ast.BinOp, ast.AugAssign))
            node.op = _BINARY[type(node.op)]()
    elif kind == "drop_branch":
        for block in _statement_lists(function):
            for index, statement in enumerate(block):
                if statement is node:
                    assert isinstance(statement, ast.If)
                    block[index : index + 1] = list(statement.orelse)
                    if not block:
                        block.append(ast.Pass())
                    return
    elif kind == "wrong_return":
        assert isinstance(node, ast.Return)
        node.value = (
            ast.Constant(variant) if isinstance(variant, bool) else ast.Name(variant, ast.Load())
        )
    elif kind == "swap_arguments":
        assert isinstance(node, ast.Call)
        node.args[0], node.args[1] = node.args[1], node.args[0]
    elif kind == "wrong_initial_value":
        assert isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        value = node.value.value
        if isinstance(value, bool):
            node.value.value = not value
        elif value is None:
            node.value.value = 0
        else:
            node.value.value = 1 if value == 0 else 0
    elif kind == "negate_condition":
        assert isinstance(node, (ast.If, ast.While, ast.IfExp))
        if isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not):
            node.test = node.test.operand
        else:
            node.test = ast.UnaryOp(ast.Not(), node.test)
    else:
        raise MutationError(f"unknown mutation kind {kind!r}")


def _local_names(function: ast.FunctionDef) -> list[str]:
    """Parameters, then names the body assigns, in first-seen order."""
    names = [arg.arg for arg in [*function.args.posonlyargs, *function.args.args]]
    for node in _body_nodes(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id not in names:
            names.append(node.id)
    return names


@dataclass(frozen=True)
class Mutant:
    kind: str
    site: int
    source: str


class _Canonical(ast.NodeTransformer):
    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        inner = node.operand
        if (
            isinstance(node.op, ast.Not)
            and isinstance(inner, ast.Compare)
            and len(inner.ops) == 1
            and type(inner.ops[0]) in _INVERSE
        ):
            return ast.Compare(inner.left, [_INVERSE[type(inner.ops[0])]()], inner.comparators)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        left, right = node.left, node.comparators[0] if node.comparators else None
        if (
            len(node.ops) == 1
            and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
            and isinstance(left, ast.BinOp)
            and isinstance(left.op, ast.Mod)
            and _is_int(left.right)
            and getattr(left.right, "value", None) == 2
            and right is not None
            and _is_int(right)
            and getattr(right, "value", None) == 1
        ):
            flipped = ast.NotEq() if isinstance(node.ops[0], ast.Eq) else ast.Eq()
            return ast.Compare(left, [flipped], [ast.Constant(0)])
        return node


def canonical_dump(node: ast.AST) -> str:
    """The tree with ``not (a OP b)`` rewritten to ``a INVERSE_OP b`` and ``x % 2 == 1`` to
    ``x % 2 != 0`` (and their ``!=`` forms), dumped: two functions with equal canonical
    dumps behave the same."""
    return ast.dump(_Canonical().visit(copy.deepcopy(node)))


def _valid(reference: ast.FunctionDef, text: str, forbid: Iterable[str]) -> bool:
    try:
        node = function_node(text)
    except (SyntaxError, MutationError, ValueError, RecursionError):
        return False
    if node.name != reference.name or ast.dump(node.args) != ast.dump(reference.args):
        return False
    dump = canonical_dump(node)
    return dump != canonical_dump(reference) and all(dump != item for item in forbid)


def mutants(
    source: str,
    kind: str,
    *,
    forbid: Sequence[str] = (),
    symmetric: Collection[str] = (),
) -> list[Mutant]:
    """Every valid mutant of ``kind`` (site order). ``forbid`` holds sources whose trees a
    mutant must not equal; ``symmetric`` names helpers whose two arguments commute."""
    reference = function_node(source)
    forbidden = [canonical_dump(function_node(item)) for item in forbid]
    found = []
    seen: set[str] = set()
    for index, site in enumerate(sites(reference, kind, symmetric=symmetric)):
        copy_node = copy.deepcopy(reference)
        apply_site(copy_node, kind, site)
        text = ast.unparse(ast.fix_missing_locations(copy_node))
        if not _valid(reference, text, forbidden):
            continue
        dump = canonical_dump(function_node(text))
        if dump in seen:
            continue
        seen.add(dump)
        found.append(Mutant(kind, index, text))
    return found


def applicable(
    source: str, *, forbid: Sequence[str] = (), symmetric: Collection[str] = ()
) -> list[str]:
    """The mutation kinds with at least one valid mutant."""
    return [
        kind for kind in MUTATION_KINDS if mutants(source, kind, forbid=forbid, symmetric=symmetric)
    ]


def choose(
    source: str,
    kind: str,
    rng: random.Random,
    *,
    forbid: Sequence[str] = (),
    symmetric: Collection[str] = (),
) -> Mutant:
    found = mutants(source, kind, forbid=forbid, symmetric=symmetric)
    if not found:
        raise MutationError(f"no valid {kind} mutant")
    return rng.choice(found)


def control(source: str, kind: str, constant: Any = None) -> str:
    """A negative control of the function: same name and signature, a trivial body."""
    if kind not in CONTROL_KINDS:
        raise MutationError(f"unknown control {kind!r}")
    node = copy.deepcopy(function_node(source))
    node.decorator_list = []
    params = [arg.arg for arg in [*node.args.posonlyargs, *node.args.args]]
    body: ast.stmt
    if kind == "delete":
        body = ast.Pass()
    elif kind == "echo":
        body = ast.Return(ast.Name(params[0], ast.Load()) if params else ast.Constant(None))
    elif kind == "constant":
        body = ast.Return(_literal(constant))
    else:
        body = ast.Return(ast.Constant(None))
    node.body = [body]
    return ast.unparse(ast.fix_missing_locations(node))


def _literal(value: Any) -> ast.expr:
    tree = ast.parse(repr(value), mode="eval")
    if ast.literal_eval(tree) != value:
        raise MutationError("the constant control needs a literal value")
    return tree.body
