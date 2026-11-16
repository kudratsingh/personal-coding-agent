"""AST mutants of a reference function (the wrong attempts of repair rows) and the negative
controls. Everything here is a syntax-tree transformation; nothing is executed.

A mutant is the reference function with one small change at one site. It is kept only if
its text parses back into one function with the same name and the same signature (the
``arguments`` node, defaults included), and its tree differs from the reference's and from
every tree in ``forbid`` (the original buggy version, other mutants). Whether a mutant also
*behaves* differently is not decided here: the row builder keeps a repair row only when the
sandbox shows the mutant failing the public tests.
"""

from __future__ import annotations

import ast
import copy
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
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


def sites(function: ast.FunctionDef, kind: str) -> list[Site]:
    """The mutation sites of ``kind`` in ``function``, in a fixed order."""
    nodes = list(_body_nodes(function))
    found: list[Site] = []
    if kind == "off_by_one":
        initial = _initial_constants(function)
        for pos, node in enumerate(nodes):
            if _is_int(node) and id(node) not in initial:
                found += [(pos, 1), (pos, -1)]
    elif kind == "flip_comparison":
        for pos, node in enumerate(nodes):
            if isinstance(node, ast.Compare):
                found += [(pos, i) for i, op in enumerate(node.ops) if type(op) in _COMPARE]
    elif kind == "wrong_operator":
        for pos, node in enumerate(nodes):
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
            ):
                found.append((pos, None))
    elif kind == "wrong_initial_value":
        found = [(pos, None) for pos, node in enumerate(nodes) if _initial_assign(node)]
    elif kind == "negate_condition":
        found = [
            (pos, None)
            for pos, node in enumerate(nodes)
            if isinstance(node, (ast.If, ast.While, ast.IfExp))
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


def _valid(reference: ast.FunctionDef, text: str, forbid: Iterable[str]) -> bool:
    try:
        node = function_node(text)
    except (SyntaxError, MutationError, ValueError, RecursionError):
        return False
    if node.name != reference.name or ast.dump(node.args) != ast.dump(reference.args):
        return False
    dump = ast.dump(node)
    return dump != ast.dump(reference) and all(dump != item for item in forbid)


def mutants(source: str, kind: str, *, forbid: Sequence[str] = ()) -> list[Mutant]:
    """Every valid mutant of ``kind`` (site order). ``forbid`` holds sources whose trees a
    mutant must not equal."""
    reference = function_node(source)
    forbidden = [ast.dump(function_node(item)) for item in forbid]
    found = []
    seen: set[str] = set()
    for index, site in enumerate(sites(reference, kind)):
        copy_node = copy.deepcopy(reference)
        apply_site(copy_node, kind, site)
        text = ast.unparse(ast.fix_missing_locations(copy_node))
        if text in seen or not _valid(reference, text, forbidden):
            continue
        seen.add(text)
        found.append(Mutant(kind, index, text))
    return found


def applicable(source: str, *, forbid: Sequence[str] = ()) -> list[str]:
    """The mutation kinds with at least one valid mutant."""
    return [kind for kind in MUTATION_KINDS if mutants(source, kind, forbid=forbid)]


def choose(source: str, kind: str, rng: random.Random, *, forbid: Sequence[str] = ()) -> Mutant:
    found = mutants(source, kind, forbid=forbid)
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
