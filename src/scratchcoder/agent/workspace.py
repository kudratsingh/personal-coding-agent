"""Host-side static analysis and editing of a small flat repository. Nothing here executes code.

A workspace maps file names (flat, ``name.py``) to source text. Test files are named
``test_*.py``. Every operation parses with :func:`ast.parse`, which builds a syntax tree and
runs nothing.

- :func:`locate` follows calls from the failing public tests through in-repository imports
  and returns the reachable top-level functions, nearest first.
- :func:`apply_function` replaces one top-level function's source span (decorators included)
  with a model-written function of the same name, and refuses anything else.
- :func:`unified_diff` renders the change for review.
"""

from __future__ import annotations

import ast
import difflib
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

Files = Mapping[str, str]
FILE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\.py\Z")
LOCATE_DEPTH = 3
"""How many calls deep :func:`locate` follows from a test: test -> f -> helper -> helper."""


def module_name(path: str) -> str:
    """``textkit.py`` -> ``textkit``; refuses names that are not flat module files."""
    if not FILE_NAME.match(path):
        raise ValueError(f"not a flat module file name: {path!r}")
    return path[:-3]


def is_test_file(path: str) -> bool:
    return path.startswith("test_")


def check_file_name(path: str) -> None:
    """A workspace file must be a flat module whose name shadows no standard-library module
    and no name the packed program reserves (``_sc``)."""
    name = module_name(path)
    if name in sys.stdlib_module_names or name.startswith("_sc") or name == "__main__":
        raise ValueError(f"workspace module name is reserved: {path!r}")


@dataclass(frozen=True)
class Function:
    """A top-level function: its file, name and 1-based inclusive line span (decorators
    included)."""

    path: str
    name: str
    start: int
    end: int

    @property
    def key(self) -> str:
        return f"{self.path}::{self.name}"


def parse(text: str, path: str) -> ast.Module:
    return ast.parse(text, filename=path)


def functions(text: str, path: str) -> dict[str, Function]:
    """The top-level (synchronous) functions of a module by name; a later definition of the
    same name wins, as it would at import."""
    found: dict[str, Function] = {}
    for node in parse(text, path).body:
        if isinstance(node, ast.FunctionDef):
            start = min([node.lineno, *(item.lineno for item in node.decorator_list)])
            assert node.end_lineno is not None
            found[node.name] = Function(path, node.name, start, node.end_lineno)
    return found


def _span(node: ast.FunctionDef) -> tuple[int, int]:
    assert node.end_lineno is not None
    return min([node.lineno, *(item.lineno for item in node.decorator_list)]), node.end_lineno


def function_source(text: str, function: Function) -> str:
    lines = text.split("\n")
    return "\n".join(lines[function.start - 1 : function.end])


def _imports(tree: ast.Module, files: Files) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """In-repository imports of a module: ``import m`` (local name -> file) and
    ``from m import f`` (local name -> (file, f))."""
    modules: dict[str, str] = {}
    names: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                path = alias.name + ".py"
                if path in files and "." not in alias.name:
                    modules[alias.asname or alias.name] = path
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            path = node.module + ".py"
            if path in files:
                for alias in node.names:
                    if alias.name != "*":
                        names[alias.asname or alias.name] = (path, alias.name)
    return modules, names


def _called(node: ast.AST) -> list[tuple[str | None, str]]:
    """Calls inside ``node`` as (module local name or None, attribute or name), in source
    order: ``f(x)`` -> (None, "f"); ``m.f(x)`` -> ("m", "f")."""
    calls: list[tuple[str | None, str]] = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        target = item.func
        if isinstance(target, ast.Name):
            calls.append((None, target.id))
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            calls.append((target.value.id, target.attr))
    return sorted(set(calls), key=calls.index)


@dataclass(frozen=True)
class Candidate:
    function: Function
    distance: int
    """0 for a function a test calls directly, 1 for one that function calls, and so on."""


def locate(
    files: Files,
    test_paths: Sequence[str],
    failing: Iterable[str] | None,
    allowed: Iterable[str],
    *,
    depth: int = LOCATE_DEPTH,
) -> list[Candidate]:
    """The functions reachable from the failing tests, restricted to ``allowed`` files.

    ``failing`` holds test names (``test_x``) from feedback; when it is empty or ``None``
    every test function of ``test_paths`` is a start. Calls resolve through the calling
    module's own top-level functions and its imports of other workspace modules; calls of
    anything else (builtins, methods, the standard library) are not followed. The result is
    ordered by distance from a test, then by first discovery.
    """
    allowed_set = set(allowed)
    trees = {path: parse(text, path) for path, text in files.items()}
    defined = {path: functions(text, path) for path, text in files.items()}
    imports = {path: _imports(tree, files) for path, tree in trees.items()}
    wanted = set(failing or ())
    starts: list[tuple[str, ast.FunctionDef]] = []
    for path in test_paths:
        for node in trees[path].body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                if not wanted or node.name in wanted:
                    starts.append((path, node))
    if wanted and not starts:  # Feedback named no test we know: fall back to all of them.
        return locate(files, test_paths, None, allowed_set, depth=depth)

    def resolve(path: str, module: str | None, name: str) -> Function | None:
        modules, names = imports[path]
        if module is None:
            if name in defined[path]:
                return defined[path][name]
            if name in names:
                source, original = names[name]
                return defined[source].get(original)
            return None
        if module in modules:
            return defined[modules[module]].get(name)
        return None

    seen: dict[str, Candidate] = {}
    order: list[str] = []
    frontier: list[tuple[str, ast.AST]] = list(starts)
    level = 0
    while frontier and level <= depth:
        following: list[tuple[str, ast.AST]] = []
        for path, item in frontier:
            for module, name in _called(item):
                found = resolve(path, module, name)
                if found is None or found.key in seen:
                    continue
                if is_test_file(found.path):  # A test-module helper: follow, not a target.
                    seen[found.key] = Candidate(found, level)
                    following.append((found.path, _definition(trees[found.path], found.name)))
                    continue
                seen[found.key] = Candidate(found, level)
                order.append(found.key)
                following.append((found.path, _definition(trees[found.path], found.name)))
        frontier = following
        level += 1
    return [seen[key] for key in order if seen[key].function.path in allowed_set]


def _definition(tree: ast.Module, name: str) -> ast.FunctionDef:
    last = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            last = node
    assert last is not None
    return last


class EditRefused(ValueError):
    """The host refused a proposed edit; ``reason`` is one of :data:`REFUSALS`."""

    def __init__(self, reason: str, detail: str = "") -> None:
        if reason not in REFUSALS:
            raise ValueError(f"unknown refusal {reason!r}")
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


FORMAT_REFUSALS = ("no_python_block", "unparsable")
"""The reply is not usable code: a formatting failure."""
INTERFACE_REFUSALS = (
    "no_function",
    "multiple_functions",
    "extra_statements",
    "unknown_name",
    "ambiguous_name",
    "not_allowed",
    "breaks_module",
)
"""The reply parsed but is not an edit the host accepts: a tool-interface failure."""
REFUSALS = FORMAT_REFUSALS + INTERFACE_REFUSALS

REFUSAL_TEXT = {
    "no_python_block": "the reply had no complete python block",
    "unparsable": "the code in the python block does not parse",
    "no_function": "the python block defines no function",
    "multiple_functions": "write one function at a time, without new helper functions",
    "extra_statements": "the function uses code from outside it that the file does not have",
    "unknown_name": "the function is not one of the functions shown",
    "ambiguous_name": "more than one shown function has that name",
    "not_allowed": "that file may not be edited",
    "breaks_module": "the edited file no longer parses",
}


@dataclass(frozen=True)
class Edit:
    path: str
    name: str
    source: str
    """The new function's lines exactly as the reply wrote them (decorators included)."""
    dropped: tuple[str, ...] = ()
    """First lines of the reply's other top-level statements, which were not written."""


def _bound(node: ast.stmt) -> set[str]:
    """Names a top-level statement binds (imports, definitions, assignment targets)."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {
            (alias.asname or alias.name).split(".")[0] for alias in node.names if alias.name != "*"
        }
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del))
    }


def _used(function: ast.FunctionDef) -> set[str]:
    return {item.id for item in ast.walk(function) if isinstance(item, ast.Name)}


def parse_reply(
    code: str | None, candidates: Sequence[Function], files: Files, allowed: Iterable[str]
) -> Edit:
    """Turn the code of a reply into an edit of one candidate function, or refuse it.

    The code must parse and define exactly one top-level function whose name is one of the
    candidates (the functions the prompt showed), in an allowed non-test file. Only that
    function's lines are ever written to the workspace, so nothing else in the reply can
    change the repository. The reply's other top-level statements are dropped (and recorded)
    when the function does not depend on them: unused imports, example usage, unrelated
    definitions. A reply whose function needs something else from it (a helper it defines,
    an import the target module lacks) is refused, because dropping that would break the
    function.
    """
    if code is None:
        raise EditRefused("no_python_block")
    # Python ends a line at "\r\n", "\r" or "\n"; the host splits on "\n" only. Normalizing
    # first keeps the syntax tree's line numbers and the host's lines the same lines.
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError) as error:
        raise EditRefused("unparsable", type(error).__name__) from None
    defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not defs:
        raise EditRefused("no_function")
    names = {item.name for item in candidates}
    shown = [node for node in defs if node.name in names]
    if not shown:
        raise EditRefused("unknown_name", ", ".join(node.name for node in defs))
    if len(shown) > 1:
        raise EditRefused("multiple_functions", ", ".join(node.name for node in shown))
    function = shown[0]
    matches = [item for item in candidates if item.name == function.name]
    if len({item.key for item in matches}) > 1:
        raise EditRefused("ambiguous_name", function.name)
    target = matches[0]
    if target.path not in set(allowed) or is_test_file(target.path):
        raise EditRefused("not_allowed", target.path)
    module = parse(files[target.path], target.path)
    existing = {ast.unparse(node) for node in module.body}
    used = _used(function)
    dropped: list[str] = []
    for node in tree.body:
        if node is function:
            continue
        needed = _bound(node) & used
        label = ast.unparse(node).split("\n")[0][:80]
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if needed and ast.unparse(node) not in existing:
                raise EditRefused("extra_statements", f"new import the function uses: {label}")
        elif needed:
            reason = (
                "multiple_functions" if isinstance(node, ast.FunctionDef) else "extra_statements"
            )
            raise EditRefused(reason, f"the function depends on: {label}")
        dropped.append(label)
    lines = code.split("\n")
    start = min([function.lineno, *(item.lineno for item in function.decorator_list)])
    assert function.end_lineno is not None
    # A top-level definition starts in column 0, so its lines are taken as they are (not
    # stripped: a line may end inside a string literal).
    source = "\n".join(lines[start - 1 : function.end_lineno])
    return Edit(target.path, target.name, source, tuple(dropped))


def apply_function(files: Files, edit: Edit, allowed: Iterable[str]) -> dict[str, str]:
    """The workspace with ``edit`` applied: the target's span replaced by the new source.

    Refused unless the file is allowed and not a test, the function exists there, and the
    edited file parses into the same top-level statements: every statement other than the
    target is unchanged (compared as syntax trees), and the target's place holds one function
    of the same name whose span is exactly the inserted lines. So an edit can change one
    function and nothing else, whatever the reply's text holds."""
    if edit.path not in set(allowed) or is_test_file(edit.path) or edit.path not in files:
        raise EditRefused("not_allowed", edit.path)
    if "\r" in edit.source:
        raise EditRefused("unparsable", "carriage return in the function source")
    text = files[edit.path]
    defined = functions(text, edit.path)
    if edit.name not in defined:
        raise EditRefused("unknown_name", edit.name)
    target = defined[edit.name]
    lines = text.split("\n")
    inserted = edit.source.split("\n")
    replaced = "\n".join([*lines[: target.start - 1], *inserted, *lines[target.end :]])
    try:
        before_body = parse(text, edit.path).body
        after_body = parse(replaced, edit.path).body
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        raise EditRefused("breaks_module") from None
    index = next(
        position
        for position, node in enumerate(before_body)
        if isinstance(node, ast.FunctionDef) and _span(node) == (target.start, target.end)
    )
    if len(after_body) != len(before_body):
        raise EditRefused("breaks_module", "top-level statements changed")
    for position, (old, new) in enumerate(zip(before_body, after_body, strict=True)):
        if position != index and ast.dump(old) != ast.dump(new):
            raise EditRefused("breaks_module", "code outside the function changed")
    new = after_body[index]
    expected = (target.start, target.start + len(inserted) - 1)
    if not isinstance(new, ast.FunctionDef) or new.name != edit.name or _span(new) != expected:
        raise EditRefused("breaks_module", "the inserted lines are not exactly one function")
    updated = dict(files)
    updated[edit.path] = replaced
    return updated


def touched(before: Files, after: Files) -> list[str]:
    return sorted(path for path in after if before.get(path) != after[path])


def unified_diff(before: Files, after: Files) -> str:
    """A unified diff of every changed file, ``a/`` and ``b/`` prefixed, in name order."""
    chunks: list[str] = []
    for path in touched(before, after):
        chunks.extend(
            difflib.unified_diff(
                before.get(path, "").splitlines(keepends=True),
                after[path].splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
    return "".join(line if line.endswith("\n") else line + "\n" for line in chunks)


def changed_lines(before: str, after: str) -> list[str]:
    """The removed and added lines between two versions (no context), for a short summary."""
    return [
        line
        for line in difflib.unified_diff(before.split("\n"), after.split("\n"), lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]


def repository_map(files: Files) -> list[str]:
    """One line per non-test module: ``path: f, g, h`` (top-level functions in order)."""
    rows = []
    for path in sorted(files):
        if is_test_file(path):
            continue
        rows.append(f"{path}: " + ", ".join(functions(files[path], path)))
    return rows
