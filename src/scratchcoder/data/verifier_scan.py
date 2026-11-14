"""Static, text-only scan of task records for verifier-quality risks.

Diagnostic only: nothing in training, evaluation or the sandbox imports this module. It
parses task text with :mod:`ast` and regular expressions and never executes, compiles for
execution, imports or otherwise runs any task, reference or generated code. A hit is a
provisional, heuristic flag that carries its rationale; it is not an executed verdict, and a
task with no hit is not thereby shown to be sound.

Input records are the normalized task rows of :class:`scratchcoder.eval.tasks.Task`
(``id``, ``prompt``, ``mode``, ``hidden_tests``, and the reference text in ``solution``;
``reference`` is accepted as an alias; ``entry_point`` is used only as a fallback name when a
record has no reference).

Rules (``RULESET_VERSION`` changes whenever any rule changes):

* R1 ``declared_checks``: count of top-level asserts that mention a name the reference defines
  at its top level (``def``, ``class`` or plain assignment; imports do not count). A metric
  reported on every function task, not a hit; see :func:`few_declared_checks`.
* R2 ``candidate_independent_assert``: a top-level assert that mentions no such name.
* R3 ``none_only``: every top-level hidden assert compares with ``None`` (``== None`` or
  ``is None``).
* R4 ``side_effect_prompt_none_only``: R3 together with a side-effect keyword in the prompt
  (``SIDE_EFFECT_KEYWORDS``).
* R5 ``raise_in_prompt``: the prompt names a raise or an exception type. No current test
  form can check one. Applies to both modes.
* R6 ``tautological_compare``: a comparison inside an assert whose operands are all literals.
* R7 ``set_order``: the reference builds a list or tuple by iterating a set without
  ``sorted``, and a test compares a call with an ordered list or tuple literal of two or more
  elements. Sub-tags ``hash_order_candidate`` (string elements) and ``impl_order_dependent``.
  Names are bound per function scope (a nested function is its own scope), a list that is
  sorted in place (``name.sort()``) is skipped, and ``for x in <set>: out.append(x)`` with
  ``return out`` counts. Known false negatives: sets that arrive as arguments, other
  accumulation forms (``out += [x]``, ``out.extend``, ``insert``).
* R8 ``reference_global``: a test reads a module-level name the reference assigns (not a
  ``def`` or ``class``). Sub-tags ``reference_only_global`` and ``prompt_named_global`` (the
  prompt mentions the identifier).
* R9 ``rounding_clause``: the prompt says "rounded to N decimal places" and an expected float
  has more than N decimals. For I/O tasks the expected output text is searched instead.
* R10 ``order_free_clause``: the prompt says order does not matter and a test compares an
  unwrapped call with an ordered list or tuple literal of two or more elements.
* R11 ``redefinable_outer_bool_check``: a check the reward channel would parse (``CALL is
  True/False/None`` in either orientation, ``not CALL``, ``CALL == <representable literal>``)
  whose outer call is a builtin the reference does not define, or an attribute of a module the
  tests import. Sub-tags ``builtin_outer`` and ``module_attr_outer``. The parser is mirrored
  from ``scratchcoder.sandbox.reward`` (which is not imported here: it pulls in subprocess).
  Each hit carries ``task_parseable``: true only when the whole task is parseable by
  ``parse_function_checks`` (every top-level statement an import or a supported assert, and
  at least one check), ignoring its byte-size limits. Only ``task_parseable`` true is
  reachable reward-channel exposure; summaries count the two separately.

Mode handling: assert-based rules (R1 to R3, R4, R6 to R8, R10, R11) apply to function tasks
only; an I/O task's hidden tests are JSON cases, not asserts. The prompt rule R5 applies to
both modes, and R9 applies to both (against asserts or against expected output text). R10
needs an ordered assertion, so it is function-only.
"""

import ast
import builtins
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

RULESET_VERSION = "verifier-scan-v2"

RULES: dict[str, str] = {
    "R1": "declared_checks",
    "R2": "candidate_independent_assert",
    "R3": "none_only",
    "R4": "side_effect_prompt_none_only",
    "R5": "raise_in_prompt",
    "R6": "tautological_compare",
    "R7": "set_order",
    "R8": "reference_global",
    "R9": "rounding_clause",
    "R10": "order_free_clause",
    "R11": "redefinable_outer_bool_check",
}
"""Rule id to rule name. R1 is a metric on every function task; the others are hits."""

SIDE_EFFECT_KEYWORDS: tuple[str, ...] = (
    "print",
    "prints",
    "printed",
    "printing",
    "display",
    "displays",
    "displayed",
    "writes",
    "write to",
    "write into",
    "write out",
    "log",
    "logs",
    "logged",
    "logging",
    "output to",
    "outputs to",
)
"""Prompt words that say the function acts through output rather than a return value. The
bare imperative "write" is excluded because "write a function" opens most prompts."""

_SIDE_EFFECT = re.compile(
    r"\b("
    + "|".join(
        # The bare noun "log" is also math ("log(n)", "log n", "log in time"): skip those.
        re.escape(word) + (r"(?!\s*\(|\s+(?:in|n|base|of)\b)" if word == "log" else "")
        for word in SIDE_EFFECT_KEYWORDS
    )
    + r")\b",
    re.IGNORECASE,
)
_RAISE_NOISE = re.compile(r"\bwith\s+the\s+exception\s+of\b|\bexception\s+of\b", re.IGNORECASE)
_RAISE = re.compile(
    r"\b(?:raises?|raised|raising|throws?|exceptions?)\b|\b\w+(?:Error|Exception)\b",
    re.IGNORECASE,
)
_ORDER_FREE = re.compile(
    r"\bin\s+any\s+order\b|\bany\s+order\b|\bregardless\s+of\s+(?:the\s+)?order\b"
    r"|\border\b[^.\n]{0,40}?\s*\b(?:does\s+not|doesn't|doesnt|is\s+not|isn't|isnt|are\s+not"
    r"|aren't)\s+(?:matter|important|relevant|significant)\b"
    r"|\border\s+(?:is\s+)?(?:irrelevant|unimportant)\b|\bunordered\b"
    r"|\bno\s+particular\s+order\b|\bwithout\s+regard\s+to\s+order\b",
    re.IGNORECASE,
)
_ROUNDING = re.compile(
    r"\bround(?:ed|ing|s)?\b[^.\n]{0,40}?\b(\d+|one|two|three|four|five|six)\s+"
    r"(?:decimal\s+)?(?:places?|digits?|decimals?)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_DECIMAL_TOKEN = re.compile(r"\d+\.(\d+)")

_WRAPPERS = frozenset({"sorted", "set", "frozenset", "Counter", "len", "sum", "min", "max"})
"""Outer calls that normalize or reduce a value, so an ordered comparison is not order-bound."""
_ORDER_SINKS = frozenset({"sorted", "set", "frozenset", "sum", "len", "min", "max", "any", "all"})
_SET_METHODS = frozenset({"intersection", "union", "difference", "symmetric_difference"})
_SET_OPERATORS = (ast.BitAnd, ast.BitOr, ast.Sub, ast.BitXor)
_BUILTIN_NAMES = frozenset(name for name in dir(builtins) if not name.startswith("_"))
_FEW_CHECKS = 2


@dataclass(frozen=True)
class Hit:
    """One rule hit: the rule id and name, an optional sub-tag, and short evidence."""

    rule: str
    tag: str | None
    evidence: str
    task_parseable: bool | None = None
    """R11 only: whether the whole task is parseable by the reward parser."""

    @property
    def name(self) -> str:
        return RULES[self.rule]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence,
            "name": self.name,
            "rule": self.rule,
            "tag": self.tag,
            "task_parseable": self.task_parseable,
        }


@dataclass(frozen=True)
class TaskScan:
    """The scan of one task: R1 metrics, hits in stable order, and scan notes."""

    id: str
    mode: str
    declared_checks: int | None
    total_asserts: int | None
    hits: tuple[Hit, ...]
    notes: tuple[str, ...]

    def to_row(self) -> dict[str, Any]:
        return {
            "declared_checks": self.declared_checks,
            "hits": [hit.to_dict() for hit in self.hits],
            "id": self.id,
            "mode": self.mode,
            "notes": list(self.notes),
            "total_asserts": self.total_asserts,
        }


def few_declared_checks(scan: TaskScan, limit: int = _FEW_CHECKS) -> bool:
    """Whether a function task has at most ``limit`` declared checks (R1 bucket)."""
    return scan.declared_checks is not None and scan.declared_checks <= limit


# --------------------------------------------------------------------------------------
# Small AST helpers.
# --------------------------------------------------------------------------------------


def _parse(text: str) -> ast.Module | None:
    """Parse ``text`` to a tree, or ``None`` if it is not valid Python."""
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None


def _snippet(node: ast.AST, limit: int = 80) -> str:
    text = ast.unparse(node).replace("\n", " ")
    line = getattr(node, "lineno", 0)
    return f"L{line}: {text[:limit]}"


def _evidence(nodes: list[ast.AST]) -> str:
    shown = "; ".join(_snippet(node) for node in nodes[:3])
    return f"{len(nodes)} found; {shown}" if len(nodes) > 1 else shown


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
    }


def _stored_names(tree: ast.AST) -> set[str]:
    """Every name a tree binds (stores, parameters, imports, defs); used to find free reads."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _is_literal(node: ast.expr) -> bool:
    try:
        ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False
    return True


def _top_asserts(tree: ast.Module) -> list[ast.Assert]:
    return [node for node in tree.body if isinstance(node, ast.Assert)]


def _target_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for item in target.elts for name in _target_names(item)]
    return []


@dataclass(frozen=True)
class _Reference:
    tree: ast.Module
    defined: frozenset[str]
    assigned: frozenset[str]


def _reference_info(tree: ast.Module) -> _Reference:
    defined: set[str] = set()
    assigned: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                assigned.update(_target_names(target))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            assigned.update(_target_names(node.target))
    funcs = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    return _Reference(tree, frozenset(defined | assigned), frozenset(assigned - funcs))


# --------------------------------------------------------------------------------------
# Test-shape helpers shared by several rules.
# --------------------------------------------------------------------------------------


def _is_none_compare(test: ast.expr) -> bool:
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return False
    if not isinstance(test.ops[0], (ast.Eq, ast.Is)):
        return False
    sides = (test.left, test.comparators[0])
    flags = [isinstance(side, ast.Constant) and side.value is None for side in sides]
    return flags.count(True) == 1


def _is_ordered_literal(node: ast.expr) -> bool:
    return isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) >= 2 and _is_literal(node)


def _unwrapped_call(node: ast.expr) -> bool:
    """A call that is not wrapped in a normalizing or reducing outer call."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    outer = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
    return outer not in _WRAPPERS


def _ordered_compares(asserts: Iterable[ast.Assert]) -> list[ast.Compare]:
    """Asserts of the form ``CALL == [a, b, ...]`` (two or more elements, unwrapped call).

    Only an assert whose whole test is the comparison counts, so ``== [x, y] or == [y, x]``
    (an order-tolerant test) is not collected.
    """
    found: list[ast.Compare] = []
    for node in asserts:
        test = node.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
            continue
        if not isinstance(test.ops[0], ast.Eq):
            continue
        left, right = test.left, test.comparators[0]
        if (_is_ordered_literal(right) and _unwrapped_call(left)) or (
            _is_ordered_literal(left) and _unwrapped_call(right)
        ):
            found.append(test)
    return found


def _has_string(node: ast.expr) -> bool:
    return any(
        isinstance(sub, ast.Constant) and isinstance(sub.value, str) for sub in ast.walk(node)
    )


# --------------------------------------------------------------------------------------
# Rules.
# --------------------------------------------------------------------------------------


def _r1_r2(asserts: list[ast.Assert], defined: frozenset[str]) -> tuple[int, list[ast.AST]]:
    """Return (R1 declared count, R2 candidate-independent asserts)."""
    declared = 0
    independent: list[ast.AST] = []
    for node in asserts:
        if _loaded_names(node.test) & defined:
            declared += 1
        else:
            independent.append(node)
    return declared, independent


def _r4(prompt: str) -> str | None:
    match = _SIDE_EFFECT.search(prompt)
    return match.group(1).lower() if match else None


def _r6(asserts: list[ast.Assert]) -> list[ast.AST]:
    found: list[ast.AST] = []
    for node in asserts:
        for sub in ast.walk(node.test):
            if isinstance(sub, ast.Compare) and all(
                _is_literal(operand) for operand in (sub.left, *sub.comparators)
            ):
                found.append(sub)
    return found


def _is_set_expr(node: ast.expr, set_names: set[str]) -> bool:
    if isinstance(node, (ast.Set, ast.SetComp)):
        return True
    if isinstance(node, ast.Name):
        return node.id in set_names
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in {"set", "frozenset"}:
            return True
        return isinstance(func, ast.Attribute) and func.attr in _SET_METHODS
    if isinstance(node, ast.BinOp) and isinstance(node.op, _SET_OPERATORS):
        return _is_set_expr(node.left, set_names) or _is_set_expr(node.right, set_names)
    return False


_SCOPE_BREAKS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _walk_scope(scope: ast.AST) -> list[ast.AST]:
    """Nodes of one scope: the body of ``scope`` without descending into nested functions or
    classes (each is its own scope). The scope node itself is not included."""
    found: list[ast.AST] = []
    stack: list[ast.AST] = list(reversed(getattr(scope, "body", [])))
    while stack:
        node = stack.pop()
        found.append(node)
        if isinstance(node, _SCOPE_BREAKS):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))
    return found


def _set_names(nodes: list[ast.AST]) -> set[str]:
    """Names bound in one scope to set expressions, to a fixed point."""
    names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if isinstance(node, ast.Assign):
                value: ast.expr | None = node.value
                targets = [name for target in node.targets for name in _target_names(target)]
            elif isinstance(node, ast.AnnAssign):
                value, targets = node.value, _target_names(node.target)
            elif isinstance(node, ast.AugAssign) and isinstance(node.op, _SET_OPERATORS):
                value, targets = node.value, _target_names(node.target)
            else:
                continue
            if value is not None and _is_set_expr(value, names):
                for name in targets:
                    if name not in names:
                        names.add(name)
                        changed = True
    return names


def _is_method_call(node: ast.AST, method: str) -> str | None:
    """The receiver name of ``name.method(...)``, else ``None``."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and isinstance(node.func.value, ast.Name)
    ):
        return node.func.value.id
    return None


def _loop_accumulations(
    nodes: list[ast.AST], names: set[str], sorted_in_place: set[str]
) -> list[ast.AST]:
    """``for x in <set>: out.append(x)`` where ``out`` is returned and never sorted."""
    returned = {
        node.value.id
        for node in nodes
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name)
    }
    found: list[ast.AST] = []
    for node in nodes:
        if not (isinstance(node, ast.For) and _is_set_expr(node.iter, names)):
            continue
        for sub in ast.walk(node):
            receiver = _is_method_call(sub, "append")
            if receiver in returned and receiver not in sorted_in_place:
                found.append(node)
                break
    return found


def _set_iterations(reference: ast.Module) -> list[ast.AST]:
    """Places where the reference turns set iteration order into a list or tuple."""
    scopes: list[ast.AST] = [reference]
    scopes.extend(
        node
        for node in ast.walk(reference)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    found: dict[tuple[int, int], ast.AST] = {}
    for scope in scopes:
        nodes = _walk_scope(scope)
        names = _set_names(nodes)
        sorted_in_place = {
            receiver for node in nodes if (receiver := _is_method_call(node, "sort")) is not None
        }
        assigned_to: dict[int, set[str]] = {}
        for node in nodes:
            if isinstance(node, ast.Assign):
                assigned_to[id(node.value)] = {
                    name for target in node.targets for name in _target_names(target)
                }
        skipped: set[int] = set()
        for node in nodes:
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _ORDER_SINKS
            ):
                for argument in node.args:
                    skipped.update(id(sub) for sub in ast.walk(argument))
        for node in nodes:
            if id(node) in skipped:
                continue
            hit = False
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"list", "tuple"}
                and len(node.args) == 1
            ):
                argument = node.args[0]
                hit = _is_set_expr(argument, names) or (
                    isinstance(argument, ast.GeneratorExp)
                    and any(_is_set_expr(gen.iter, names) for gen in argument.generators)
                )
            elif isinstance(node, ast.ListComp):
                hit = any(_is_set_expr(gen.iter, names) for gen in node.generators)
            if hit and assigned_to.get(id(node), set()) & sorted_in_place:
                hit = False
            if hit:
                found[(getattr(node, "lineno", 0), getattr(node, "col_offset", 0))] = node
        for node in _loop_accumulations(nodes, names, sorted_in_place):
            found[(getattr(node, "lineno", 0), getattr(node, "col_offset", 0))] = node
    return [found[key] for key in sorted(found)]


def _r8(
    reference: _Reference, tests: ast.Module, asserts: list[ast.Assert], prompt: str
) -> dict[str, list[str]]:
    """Reference-assigned module names read by the tests, keyed by sub-tag."""
    bound = _stored_names(tests)
    by_tag: dict[str, list[str]] = {}
    read = set().union(*(_loaded_names(node.test) for node in asserts)) if asserts else set()
    for name in sorted(read & reference.assigned):
        if name in bound or name.startswith("__"):
            continue
        named = re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", prompt)
        tag = "prompt_named_global" if named else "reference_only_global"
        by_tag.setdefault(tag, []).append(name)
    return by_tag


def _decimals(value: float) -> int:
    exponent = Decimal(repr(value)).as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


def _rounding_places(prompt: str) -> int | None:
    match = _ROUNDING.search(prompt)
    if match is None:
        return None
    word = match.group(1).lower()
    return int(word) if word.isdigit() else _NUMBER_WORDS[word]


def _expected_floats(asserts: list[ast.Assert]) -> list[tuple[ast.AST, float]]:
    """Float literals on the literal side of an ``==`` inside an assert."""
    found: list[tuple[ast.AST, float]] = []
    for node in asserts:
        for sub in ast.walk(node.test):
            if not isinstance(sub, ast.Compare):
                continue
            operands = (sub.left, *sub.comparators)
            for index, operator in enumerate(sub.ops):
                if not isinstance(operator, ast.Eq):
                    continue
                for operand in operands[index : index + 2]:
                    if _is_literal(operand):
                        found.extend(
                            (item, item.value)
                            for item in ast.walk(operand)
                            if isinstance(item, ast.Constant) and isinstance(item.value, float)
                        )
    return found


def _io_outputs(hidden_tests: str) -> list[str]:
    try:
        cases = json.loads(hidden_tests)
    except ValueError:
        return []
    if not isinstance(cases, list):
        return []
    return [
        case["output"]
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("output"), str)
    ]


def _r11(
    asserts: list[ast.Assert], reference: _Reference | None, tests: ast.Module
) -> dict[str, list[ast.AST]]:
    """Reward-parseable checks whose outer call can be redefined, keyed by sub-tag."""
    defined = reference.defined if reference else frozenset()
    modules = {
        (alias.asname or alias.name).split(".")[0]
        for node in tests.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    found: dict[str, list[ast.AST]] = {}
    for node in asserts:
        call = _check_call(node.test)
        if call is None or not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Name):
            if func.id in _BUILTIN_NAMES and func.id not in defined:
                found.setdefault("builtin_outer", []).append(node)
        elif isinstance(func, ast.Attribute):
            root = func.value
            if isinstance(root, ast.Name) and root.id in modules and root.id not in defined:
                found.setdefault("module_attr_outer", []).append(node)
    return found


def _representable(value: object, depth: int = 0) -> bool:
    """Mirror of ``reward._representable``: built only from the types the driver can report."""
    if depth > 32:
        return False
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_representable(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return all(
            _representable(key, depth + 1) and _representable(item, depth + 1)
            for key, item in value.items()
        )
    return False


def _safe_literal(node: ast.expr) -> bool:
    """Mirror of ``reward._safe_literal`` (the value is not needed here)."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False
    return _representable(value)


def _call_is_safe(node: ast.expr) -> bool:
    """Mirror of ``reward._call_is_safe``: a CALL is not a constant, boolean operation,
    negation or comparison, and contains no comparison."""
    if isinstance(node, (ast.BoolOp, ast.Constant)):
        return False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return False
    return not any(isinstance(sub, ast.Compare) for sub in ast.walk(node))


def _check_call(test: ast.expr) -> ast.expr | None:
    """The CALL of an assert the reward channel parses, mirroring
    ``reward._parse_predicate``: ``CALL == <representable literal>`` (either side),
    ``CALL is None/True/False`` (either side) and ``not CALL``; else ``None``."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return test.operand if _call_is_safe(test.operand) else None
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return None
    left, right = test.left, test.comparators[0]
    operator = test.ops[0]
    if isinstance(operator, ast.Eq):
        left_ok, right_ok = _safe_literal(left), _safe_literal(right)
        if right_ok and not left_ok and _call_is_safe(left):
            return left
        if left_ok and not right_ok and _call_is_safe(right):
            return right
        return None
    if isinstance(operator, ast.Is):
        for expression, other in ((left, right), (right, left)):
            if (
                isinstance(other, ast.Constant)
                and (other.value is None or other.value is True or other.value is False)
                and _call_is_safe(expression)
            ):
                return expression
    return None


def _task_parseable(tests: ast.Module) -> bool:
    """Whether ``reward.parse_function_checks`` would accept the whole task, ignoring its byte
    size limits: every top-level statement is an import or an assert whose predicate parses,
    and there is at least one assert."""
    checks = 0
    for node in tests.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if not isinstance(node, ast.Assert) or _check_call(node.test) is None:
            return False
        checks += 1
    return checks > 0


# --------------------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------------------


def _text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f"task record needs a text field: {key}")
    return value


def scan_task(record: Mapping[str, Any]) -> TaskScan:
    """Scan one task record. Reads text only; never runs any of it. A task whose text is so
    deep or large that the scan exhausts the stack or memory yields a ``scan_error`` note and
    no hits instead of failing the run."""
    task_id = _text(record, "id")
    mode = record.get("mode") or "function"
    try:
        return _scan_task(record)
    except (RecursionError, MemoryError):
        return TaskScan(task_id, str(mode), None, None, (), ("scan_error",))


def _scan_task(record: Mapping[str, Any]) -> TaskScan:
    task_id = _text(record, "id")
    prompt = _text(record, "prompt")
    hidden = _text(record, "hidden_tests")
    mode = record.get("mode") or "function"
    if mode not in {"function", "io"}:
        raise ValueError(f"unknown task mode: {mode!r}")
    solution = record.get("solution", record.get("reference"))
    notes: list[str] = []
    hits: list[Hit] = []

    raised = _RAISE.search(_RAISE_NOISE.sub(" ", prompt))
    if raised is not None:
        hits.append(Hit("R5", None, f"prompt text {raised.group(0)!r}"))

    places = _rounding_places(prompt)

    if mode == "io":
        if places is not None:
            long = [
                token.group(0)
                for text in _io_outputs(hidden)
                for token in _DECIMAL_TOKEN.finditer(text)
                if len(token.group(1)) > places
            ]
            if long:
                hits.append(Hit("R9", None, f"{places} places; expected output {long[0]}"))
        return TaskScan(task_id, mode, None, None, _ordered(hits), tuple(notes))

    tests = _parse(hidden)
    if tests is None:
        notes.append("hidden_tests_unparseable")
        return TaskScan(task_id, mode, None, None, _ordered(hits), tuple(notes))
    asserts = _top_asserts(tests)

    reference: _Reference | None = None
    if isinstance(solution, str) and solution.strip():
        parsed = _parse(solution)
        if parsed is None:
            notes.append("reference_unparseable")
        else:
            reference = _reference_info(parsed)
    else:
        notes.append("reference_missing")

    defined: frozenset[str] = reference.defined if reference else frozenset()
    entry = record.get("entry_point")
    if reference is None and isinstance(entry, str):
        defined = frozenset({entry})

    declared: int | None = None
    if defined:
        declared, independent = _r1_r2(asserts, defined)
        if independent:
            hits.append(Hit("R2", None, _evidence(independent)))
    else:
        notes.append("no_reference_names")

    if asserts and all(_is_none_compare(node.test) for node in asserts):
        hits.append(Hit("R3", None, f"{len(asserts)} of {len(asserts)} asserts compare with None"))
        keyword = _r4(prompt)
        if keyword is not None:
            hits.append(Hit("R4", None, f"prompt keyword {keyword!r}"))

    tautologies = _r6(asserts)
    if tautologies:
        hits.append(Hit("R6", None, _evidence(tautologies)))

    ordered = _ordered_compares(asserts)
    if reference is not None and ordered:
        iterations = _set_iterations(reference.tree)
        if iterations:
            tags: dict[str, ast.AST] = {}
            for compare in ordered:
                literal = (
                    compare.comparators[0]
                    if _is_ordered_literal(compare.comparators[0])
                    else compare.left
                )
                tag = "hash_order_candidate" if _has_string(literal) else "impl_order_dependent"
                tags.setdefault(tag, compare)
            for tag in sorted(tags):
                hits.append(
                    Hit(
                        "R7",
                        tag,
                        f"reference {_snippet(iterations[0])} | test {_snippet(tags[tag])}",
                    )
                )

    if reference is not None:
        for tag, names in sorted(_r8(reference, tests, asserts, prompt).items()):
            hits.append(Hit("R8", tag, "reads " + ", ".join(names[:5])))

    if places is not None:
        long_floats = [
            (node, value) for node, value in _expected_floats(asserts) if _decimals(value) > places
        ]
        if long_floats:
            node, value = long_floats[0]
            hits.append(Hit("R9", None, f"{places} places; expected {value!r} ({_snippet(node)})"))

    if ordered and _ORDER_FREE.search(prompt):
        hits.append(Hit("R10", None, _evidence(list(ordered))))

    exposed = _r11(asserts, reference, tests)
    if exposed:
        parseable = _task_parseable(tests)
        for tag, nodes in sorted(exposed.items()):
            hits.append(Hit("R11", tag, _evidence(nodes), parseable))

    return TaskScan(task_id, mode, declared, len(asserts), _ordered(hits), tuple(notes))


def _ordered(hits: list[Hit]) -> tuple[Hit, ...]:
    return tuple(sorted(hits, key=lambda hit: (int(hit.rule[1:]), hit.tag or "")))


def summarize(scans: Iterable[TaskScan]) -> dict[str, Any]:
    """Deterministic counts for a set of scans: per rule, per rule/sub-tag, and the R1 buckets."""
    rules: dict[str, int] = {}
    tags: dict[str, int] = {}
    notes: dict[str, int] = {}
    histogram: dict[str, int] = {}
    total = functions = with_hits = few = 0
    for scan in scans:
        total += 1
        with_hits += bool(scan.hits)
        for hit in scan.hits:
            rules[hit.name] = rules.get(hit.name, 0) + 1
            if hit.tag:
                key = f"{hit.name}/{hit.tag}"
                tags[key] = tags.get(key, 0) + 1
                if hit.task_parseable is not None:
                    split = f"{key}/task_parseable={str(hit.task_parseable).lower()}"
                    tags[split] = tags.get(split, 0) + 1
        for note in scan.notes:
            notes[note] = notes.get(note, 0) + 1
        if scan.mode == "function":
            functions += 1
            if scan.declared_checks is not None:
                bucket = str(scan.declared_checks) if scan.declared_checks < 3 else "3+"
                histogram[bucket] = histogram.get(bucket, 0) + 1
                few += few_declared_checks(scan)
    return {
        "declared_checks_histogram": dict(sorted(histogram.items())),
        "declared_checks_le2": few,
        "function_tasks": functions,
        "notes": dict(sorted(notes.items())),
        "rule_counts": dict(sorted(rules.items())),
        "tag_counts": dict(sorted(tags.items())),
        "tasks": total,
        "tasks_with_hits": with_hits,
    }
