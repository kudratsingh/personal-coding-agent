"""The independent oracle: its file format, static checks and embedding in a grading call.

A second author, who sees only each template's natural-language specification
(``specs.json``), writes one file per template, ``<template_id>.py``, defining:

- ``inputs(params, rng)``: a list of 1 to :data:`MAX_INPUTS` argument tuples for the target,
  drawn from the specification's input domain with ``rng`` (a :class:`random.Random`);
- ``oracle(params, *args)``: the target's expected return value for ``args``;
- optionally ``check_properties(params, fn, rng)``: raises :class:`AssertionError` when the
  function ``fn`` (the instance's target) breaks a property of the specification.

``params`` is the instance's constants (``specs.json`` lists them per instance). The file uses
the standard library only and never imports the repository.

The host **never imports or runs** this file. :func:`load` reads it as text and checks it with
:mod:`ast`; :func:`block` embeds the text as a string literal in the grading call's test
string, where it is compiled and run inside the sandbox next to the reference function. The
differential check compares the reference's result with the oracle's (same type and equal)
on every input; the property check runs after it. Any failure fails the whole grading call,
so the generator's references and tests and the independent author's oracle must agree for
an instance to be accepted: a mistake shared by the generator's spec function and its
reference cannot validate itself.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_BYTES = 65536
MAX_INPUTS = 200
REQUIRED = ("inputs", "oracle")
OPTIONAL = ("check_properties",)
FORMAT = {
    "file": "independent/<template_id>.py, one per template, UTF-8, at most 64 KiB",
    "imports": "standard library only; never the repository's modules",
    "defines": {
        "inputs(params, rng)": (
            f"returns a list of 1..{MAX_INPUTS} tuples, each the positional arguments of one "
            "call of the target, drawn from the template's input domain; rng is a "
            "random.Random seeded by the build, the only randomness to use"
        ),
        "oracle(params, *args)": (
            "returns the target's expected return value for args, per the template's "
            "specification (same type: list vs tuple and bool vs int matter); it must not "
            "mutate params or args"
        ),
        "check_properties(params, fn, rng)": (
            "optional; calls fn (the instance's target function) on inputs of its own choice "
            "and raises AssertionError when a property stated by the specification fails"
        ),
    },
    "params": "the instance's constants: the 'params' object listed for the instance",
    "run": (
        "never on the host: the text is embedded in the grading call and runs in the sandbox; "
        "every input must give an equal result of the same type from the target and from "
        "oracle, then check_properties (if defined) must return normally"
    ),
}


class OracleError(ValueError):
    pass


@dataclass(frozen=True)
class Oracle:
    template_id: str
    text: str
    sha256: str
    has_properties: bool


def path_for(directory: Path, template_id: str) -> Path:
    return directory / f"{template_id}.py"


def check_text(text: str, template_id: str) -> Oracle:
    """Static checks only (the text is parsed, never run)."""
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise OracleError(f"{template_id}: oracle file is larger than {MAX_BYTES} bytes")
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        raise OracleError(f"{template_id}: oracle file does not parse: {error}") from None
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = [name for name in REQUIRED if name not in defined]
    if missing:
        raise OracleError(f"{template_id}: oracle file lacks {', '.join(missing)}")
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise OracleError(f"{template_id}: relative import in the oracle file")
            modules = [node.module or ""]
        for module in modules:
            if module.split(".")[0] not in sys.stdlib_module_names:
                raise OracleError(f"{template_id}: oracle imports a non-stdlib module {module}")
    return Oracle(
        template_id,
        text,
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "check_properties" in defined,
    )


def load(directory: Path, template_id: str) -> Oracle:
    """Read (never import) and check the oracle file of ``template_id``."""
    path = path_for(directory, template_id)
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"{template_id}: no oracle file at {path}")
    return check_text(path.read_text(encoding="utf-8"), template_id)


def block(oracle: Oracle, module: str, function: str, params: Mapping[str, Any], seed: int) -> str:
    """The test-string block that runs the oracle's checks against ``module.function``."""
    names = {"module": module, "function": function}
    for value in names.values():
        if not value.isidentifier():
            raise OracleError(f"not an identifier: {value!r}")
    params_text = repr(dict(params))
    if ast.literal_eval(params_text) != dict(params):
        raise OracleError("instance params must be literals")
    filename = f"independent/{oracle.template_id}.py"
    return (
        f"_SC_INDEPENDENT = {oracle.text!r}\n"
        "import copy as _sc_copy\n"
        "import random as _sc_random\n"
        f"from {module} import {function} as _sc_target\n"
        "_sc_space = {'__name__': '_sc_independent'}\n"
        f"exec(compile(_SC_INDEPENDENT, {filename!r}, 'exec'), _sc_space)\n"
        f"_sc_params = {params_text}\n"
        "_sc_inputs = list(_sc_space['inputs'](_sc_copy.deepcopy(_sc_params), "
        f"_sc_random.Random({seed})))\n"
        f"assert 1 <= len(_sc_inputs) <= {MAX_INPUTS}, 'independent: input count'\n"
        "for _sc_args in _sc_inputs:\n"
        "    _sc_want = _sc_space['oracle'](_sc_copy.deepcopy(_sc_params), "
        "*_sc_copy.deepcopy(tuple(_sc_args)))\n"
        "    _sc_got = _sc_target(*_sc_copy.deepcopy(tuple(_sc_args)))\n"
        "    assert type(_sc_got) is type(_sc_want) and _sc_got == _sc_want, "
        "'independent: differential check'\n"
        "if 'check_properties' in _sc_space:\n"
        "    _sc_space['check_properties'](_sc_copy.deepcopy(_sc_params), _sc_target, "
        f"_sc_random.Random({seed + 1}))\n"
    )


def grading_script(correctness: str, preservation: str, independent: str) -> str:
    """One test string holding every check that decides acceptance: the hidden correctness
    and preservation checks, then the independent block. Any failing part fails the call."""
    parts = [
        "# hidden correctness checks\n" + correctness,
        "# hidden preservation checks\n" + preservation,
        "# independent oracle checks\n" + independent,
    ]
    return "\n".join(part if part.endswith("\n") else part + "\n" for part in parts)
