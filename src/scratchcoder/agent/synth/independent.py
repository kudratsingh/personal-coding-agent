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
string, where it is compiled and run inside the sandbox next to the reference function.
:func:`pin` verifies the whole directory against its ``SHA256SUMS`` manifest first; the
manifest's own hash identifies the oracle set in every validation record. The
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
SUMS = "SHA256SUMS"
"""The directory's manifest, in ``sha256sum`` format: one ``<hex>  <name>`` line per other
file. Its own SHA-256 pins the directory in every validation record."""
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
    "manifest": (
        "SHA256SUMS in the same directory, sha256sum format, listing every other file of the "
        "directory; its own SHA-256 pins the directory in every validation record"
    ),
    "copies": (
        "the target and oracle each get their own deep copy of every argument tuple and of "
        "params, so a target that changes its arguments does not affect the oracle"
    ),
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


@dataclass(frozen=True)
class Pin:
    """The independent directory as its ``SHA256SUMS`` file lists it, every hash verified."""

    sums_sha256: str
    """SHA-256 of the ``SHA256SUMS`` file: the one value that identifies the whole directory."""
    files: Mapping[str, str]
    """File name -> SHA-256, every file of the directory except ``SHA256SUMS``."""

    def record(self) -> dict[str, Any]:
        return {"sums_sha256": self.sums_sha256, "files": dict(sorted(self.files.items()))}


def path_for(directory: Path, template_id: str) -> Path:
    return directory / f"{template_id}.py"


def _sums_line(line: str) -> tuple[str, str]:
    digest, sep, name = line.partition("  ")
    if not sep and " *" in line:  # The binary-mode marker of sha256sum.
        digest, _, name = line.partition(" *")
    valid_digest = len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest)
    if not valid_digest or not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise OracleError(f"{SUMS}: malformed line {line!r}")
    return name, digest


def pin(directory: Path, expected: str | None = None) -> Pin:
    """Verify ``directory`` against its ``SHA256SUMS`` (files are hashed, never run).

    Every entry of the directory other than ``SHA256SUMS`` must be a regular file listed
    exactly once with its correct hash, and every listed file must exist; with ``expected``,
    the hash of ``SHA256SUMS`` itself must equal it."""
    sums = directory / SUMS
    if sums.is_symlink() or not sums.is_file():
        raise OracleError(f"no {SUMS} in {directory}")
    raw = sums.read_bytes()
    sums_sha256 = hashlib.sha256(raw).hexdigest()
    if expected is not None and sums_sha256 != expected.lower():
        raise OracleError(f"{SUMS} hash {sums_sha256} is not the pinned {expected}")
    listed: dict[str, str] = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        name, digest = _sums_line(line)
        if name in listed or name == SUMS:
            raise OracleError(f"{SUMS}: {name} listed twice or lists itself")
        listed[name] = digest
    present = {entry.name for entry in directory.iterdir() if entry.name != SUMS}
    if present - set(listed):
        raise OracleError(f"{SUMS} does not list {', '.join(sorted(present - set(listed)))}")
    if set(listed) - present:
        raise OracleError(f"{SUMS} lists missing {', '.join(sorted(set(listed) - present))}")
    for name, digest in listed.items():
        entry = directory / name
        if entry.is_symlink() or not entry.is_file():
            raise OracleError(f"{name} is not a regular file")
        if hashlib.sha256(entry.read_bytes()).hexdigest() != digest:
            raise OracleError(f"{name} does not match its {SUMS} hash")
    return Pin(sums_sha256, listed)


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


def load(directory: Path, template_id: str, pinned: Pin | None = None) -> Oracle:
    """Read (never import) and check the oracle file of ``template_id``; with ``pinned``,
    its bytes must have the hash the pinned ``SHA256SUMS`` lists for it."""
    path = path_for(directory, template_id)
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"{template_id}: no oracle file at {path}")
    raw = path.read_bytes()
    if pinned is not None and pinned.files.get(path.name) != hashlib.sha256(raw).hexdigest():
        raise OracleError(f"{template_id}: oracle file does not match the pinned {SUMS}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise OracleError(f"{template_id}: oracle file is not UTF-8") from None
    return check_text(text, template_id)


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
