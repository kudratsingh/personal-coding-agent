"""Host-side facts about each reply, and an independent re-check of every refusal.

:func:`reply_facts` records how far a reply got (parseable, which functions it defines, which
of them were shown). :func:`name_failure` splits ``unknown_name`` refusals into the model not
following the request (it wrote a function that was not shown) and a harness defect (locate
left out the function the case needs). :func:`audit_refusal` re-derives each refusal from the
reply rules with separate, simpler code and flags any contradiction, so a harness bug shows up
as a flagged defect instead of passing as a model failure. Nothing here executes code.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from typing import Any

from scratchcoder.agent import workspace
from scratchcoder.data.chat import PYTHON_END, PYTHON_START, is_special

NAME_FAILURES = (
    "invented_name",
    "public_test_name",
    "unshown_repo_function",
    "locate_omitted_target",
)
"""``unknown_name`` refusals: the first three are model failures (task following: a new name,
a name of a public test function, a repository function that was not shown), the last a
harness defect (locate left out the function the case needs)."""
HARNESS_DEFECTS = (
    "closed_block_missed",
    "parse_contradiction",
    "function_contradiction",
    "name_contradiction",
    "single_function_refused",
    "import_already_present",
    "locate_omitted_target",
    "target_not_shown",
)


def _tree(code: str | None) -> ast.Module | None:
    if code is None:
        return None
    try:
        return ast.parse(code.replace("\r\n", "\n").replace("\r", "\n"))
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None


def reply_facts(code: str | None, shown: Sequence[workspace.Function]) -> dict[str, Any]:
    tree = _tree(code)
    defined = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)] if tree else []
    names = {item.name for item in shown}
    return {
        "parseable": tree is not None,
        "defined": defined,
        "shown_defined": [name for name in defined if name in names],
    }


def name_failure(
    defined: Sequence[str],
    shown: Sequence[workspace.Function],
    files: Mapping[str, str],
    targets: Sequence[str],
) -> str:
    shown_names = {item.name for item in shown}
    target_names = {key.split("::")[1] for key in targets}
    repo = {
        name
        for path, text in files.items()
        if not workspace.is_test_file(path)
        for name in workspace.functions(text, path)
    }
    extra = [name for name in defined if name not in shown_names]
    if any(name in target_names for name in extra):
        return "locate_omitted_target"
    if any(name in repo for name in extra):
        return "unshown_repo_function"
    tests = {
        name
        for path, text in files.items()
        if workspace.is_test_file(path)
        for name in workspace.functions(text, path)
    }
    if any(name in tests for name in extra):
        return "public_test_name"
    return "invented_name"


def _closed_block(tokens: Sequence[int]) -> bool:
    """The chat-v1 rule, written again: the first python start (nothing special before it)
    is followed, before any other special token, by a python end."""
    if PYTHON_START not in tokens:
        return False
    start = list(tokens).index(PYTHON_START)
    if any(is_special(token) for token in tokens[:start]):
        return False
    for token in tokens[start + 1 :]:
        if is_special(token):
            return token == PYTHON_END
    return False


_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _stored(node: ast.AST) -> set[str]:
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del))
    }


def _bound_by_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names if a.name != "*"}
    return names


def audit_refusal(
    reason: str,
    *,
    tokens: Sequence[int],
    code: str | None,
    shown: Sequence[workspace.Function],
    files: Mapping[str, str],
    targets: Sequence[str],
) -> list[str]:
    """Contradictions between a refusal and the reply rules (empty when consistent)."""
    flags: list[str] = []
    tree = _tree(code)
    shown_names = {item.name for item in shown}
    if reason == "no_python_block":
        if tokens and _closed_block(tokens):
            flags.append("closed_block_missed")
        return flags
    if reason == "unparsable":
        if tree is not None:
            flags.append("parse_contradiction")
        return flags
    if tree is None:
        return flags
    defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    named = [node for node in defs if node.name in shown_names]
    if reason == "no_function" and defs:
        flags.append("function_contradiction")
    if reason == "unknown_name":
        if named:
            flags.append("name_contradiction")
        if name_failure([d.name for d in defs], shown, files, targets) == "locate_omitted_target":
            flags.append("locate_omitted_target")
    if reason in {"multiple_functions", "extra_statements"} and len(named) == 1:
        function = named[0]
        used = {item.id for item in ast.walk(function) if isinstance(item, ast.Name)}
        helpers = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node is not function
            and node.name in used
        ]
        needed_imports = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and {(a.asname or a.name).split(".")[0] for a in node.names} & used
        ]
        assigned = [
            node
            for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom, *_DEFINITIONS))
            and _stored(node) & used
        ]
        if reason == "multiple_functions" and not helpers:
            flags.append("single_function_refused")
        if reason == "extra_statements" and not helpers and not assigned and needed_imports:
            target = next(item for item in shown if item.name == function.name)
            present = _bound_by_imports(workspace.parse(files[target.path], target.path))
            wanted = set().union(
                *({(a.asname or a.name).split(".")[0] for a in n.names} for n in needed_imports)
            )
            if wanted & used <= present:
                flags.append("import_already_present")
    return flags
