"""Generic fixtures for the synthetic agent-data generator: a chunk encoder, a toy template
family and a scripted executor that decides every call by reading the packed sources as a
literal (``ast.literal_eval``), never by running them."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_fixtures import packed_sources

from scratchcoder.agent import execution, workspace
from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template
from scratchcoder.sandbox.runner import Result


class Chunks:
    """A reversible test encoder: one token per run of up to four characters (never a
    special token), so token counts are close to a byte-pair tokenizer's on code."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}
        self.texts: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        found = []
        for start in range(0, len(text), 4):
            piece = text[start : start + 4]
            if piece not in self.ids:
                token = 1000 + len(self.ids)
                self.ids[piece] = token
                self.texts[token] = piece
            found.append(self.ids[piece])
        return found

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.texts.get(token, "") for token in ids)


def _scale(p: Params, x: int, bug: Any = None) -> int:
    factor = int(p["k"]) + 1 if bug == "factor_up" else int(p["k"])
    return x * factor + (1 if bug == "plus_one" else 0)


def _scale_all(p: Params, xs: list[int], bug: Any = None) -> list[int]:
    return [_scale(p, x, bug) for x in xs]


def _first_item(p: Params, xs: list[int], bug: Any = None) -> int:
    return xs[1] if bug == "wrong_index" else xs[0]


def _offset(p: Params, x: int, bug: Any = None) -> int:
    return x + 100


TOY = Family(
    id="toy",
    domain="toy integer helpers (test fixture)",
    modules={"main": ("toolbox", "kit"), "use": ("usage", "callers")},
    docs={"main": "Toy helpers.", "use": "Toy callers."},
    names={
        "scale": ("scale", "times_k"),
        "scale_all": ("scale_all", "times_k_all"),
        "first_item": ("first_item", "head_of"),
        "offset": ("offset", "shifted"),
    },
    consts={"k": (3, 4)},
    funcs=(
        Func(
            "scale",
            "main",
            'def $scale(x):\n    """x times $k."""\n    return x * $k',
            _scale,
            public=[("two", (2,)), ("zero", (0,))],
            hidden=[(1,), (2,), (5,), (0,), (7,), (10,)],
        ),
        Func(
            "scale_all",
            "use",
            'def $scale_all(xs):\n    """Each value times $k."""\n'
            "    return [$scale(x) for x in xs]",
            _scale_all,
            public=[("pair", ([1, 2],)), ("empty", ([],))],
            hidden=[([],), ([3],), ([0, 1],)],
            uses=("scale",),
        ),
        Func(
            "first_item",
            "main",
            'def $first_item(xs):\n    """The first value."""\n    return xs[0]',
            _first_item,
            public=[("one", ([9],))],
            hidden=[([1],), ([2, 3],), ([4, 5, 6],), ([7],), ([8, 9],), ([0],)],
        ),
        Func(
            "offset",
            "use",
            'def $offset(x):\n    """x plus 100."""\n    return x + 100',
            _offset,
            public=[("small", (1,))],
            hidden=[(0,), (5,)],
        ),
    ),
)

W1 = Template(
    id="toy1",
    family=TOY,
    workflow="W1",
    target="scale",
    concept="multiply by a constant",
    bugs={
        "factor_up": Bug(edits=(("x * $k", "x * ($k + 1)"),)),
        "plus_one": Bug(edits=(("return x * $k", "return x * $k + 1"),)),
    },
    spec="scale(x) returns x * params['k'].",
    domain="x: int in 0..20.",
    returns="int",
    include=("offset",),
)
W3 = Template(
    id="toy3",
    family=TOY,
    workflow="W3",
    target="scale",
    concept="multiply by a constant through a caller",
    bugs={"factor_up": Bug(edits=(("x * $k", "x * ($k + 1)"),))},
    spec="scale(x) returns x * params['k'].",
    domain="x: int in 0..20.",
    returns="int",
    include=("offset",),
    caller="scale_all",
)
W4 = Template(
    id="toy4",
    family=TOY,
    workflow="W4",
    target="first_item",
    concept="first value",
    bugs={"stub_raise": Bug(stub="raise"), "stub_none": Bug(stub="none")},
    spec="first_item(xs) returns xs[0].",
    domain="xs: non-empty list of ints.",
    returns="int",
    include=("offset",),
)
W5 = Template(
    id="toy5",
    family=TOY,
    workflow="W5",
    target="first_item",
    concept="first value with a traceback",
    bugs={"wrong_index": Bug(edits=(("xs[0]", "xs[1]"),))},
    spec="first_item(xs) returns xs[0].",
    domain="xs: non-empty list of ints.",
    returns="int",
    include=("offset",),
)


def families_json(path: Path, names: Sequence[str] = ("blocked_name",)) -> Path:
    """A synthetic evaluation-family blocklist."""
    path.write_text(
        json.dumps(
            {
                "schema": "agent-eval-families-v1",
                "all_source_modules": ["blockedmod.py"],
                "all_target_functions": list(names),
                "all_defined_source_names": list(names),
                "all_test_function_names": ["test_blocked"],
                "cases": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def target_text(sources: dict[str, str], module: str, name: str) -> str | None:
    text = sources.get(module)
    if text is None:
        return None
    found = workspace.functions(text, module + ".py").get(name)
    return None if found is None else workspace.function_source(text, found)


class SynthExecutor:
    """Scripted sandbox for one instance: the target function decides every verdict.

    - public driver: passes when the target equals ``reference``; otherwise fails, reporting
      ``failing`` (test names) with ``frames`` and ``error``;
    - a check script: passes when the target equals ``reference``.

    ``public_passes`` lists target texts (such as an equivalent mutant) for which the public
    tests pass anyway. Records every call."""

    def __init__(
        self,
        module: str,
        name: str,
        reference: str,
        failing: Sequence[str],
        *,
        test_module: str,
        frames: Sequence[list[object]] = (),
        error: tuple[str, str] = ("AssertionError", "wrong"),
        public_passes: Sequence[str] = (),
    ) -> None:
        self.module, self.name, self.reference = module, name, reference
        self.failing = list(failing)
        self.test_module = test_module
        self.frames = [list(frame) for frame in frames]
        self.error = error
        self.public_passes = list(public_passes)
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, program: str, tests: str, mode: str) -> Result:
        self.calls.append((program, tests, mode))
        current = target_text(packed_sources(program), self.module, self.name)
        fixed = current == self.reference
        if execution.REPORT_PREFIX not in tests:
            return Result("passed" if fixed else "failed", elapsed_seconds=0.1)
        if fixed or current in self.public_passes:
            passed = {"module": self.test_module, "test": self.failing[0], "outcome": "pass"}
            return Result("passed", execution.REPORT_PREFIX + json.dumps(passed) + "\n", 0, 0.1)
        lines = []
        for test in self.failing:
            row: dict[str, object] = {
                "module": self.test_module,
                "test": test,
                "outcome": "fail",
                "type": self.error[0],
                "message": self.error[1],
                "frames": self.frames,
            }
            lines.append(execution.REPORT_PREFIX + json.dumps(row))
        return Result("failed", "\n".join(lines) + "\n", 1, 0.1)
