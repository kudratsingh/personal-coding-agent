"""Family ``poly``: polynomials as coefficient lists, lowest power first."""

from __future__ import annotations

from typing import Any

from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template


def _evaluate(p: Params, cs: list[int], x: int, bug: Any = None) -> int:
    acc = 1 if bug == "start_one" else 0
    for c in cs if bug == "not_reversed" else reversed(cs):
        acc = acc + x * c if bug == "plus_times" else acc * x + c
    return acc


def _derivative(p: Params, cs: list[int], bug: Any = None) -> list[int]:
    start = 0 if bug == "from_zero" else 1
    if bug == "no_power":
        return [cs[i] for i in range(start, len(cs))]
    if bug == "shifted":
        return [i * cs[i - 1] for i in range(start, len(cs))]
    return [i * cs[i] for i in range(start, len(cs))]


def _trim(p: Params, cs: list[int], bug: Any = None) -> list[int]:
    out = list(cs)
    while out and out[-1] == 0:
        out.pop()
    return out


def _add(p: Params, a: list[int], b: list[int], bug: Any = None) -> list[int]:
    n = max(len(a), len(b))
    out = [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]
    return _trim(p, out, bug)


def _degree(p: Params, cs: list[int], bug: Any = None) -> int:
    return len(_trim(p, cs, bug)) - 1


def _slope(p: Params, cs: list[int], x: int, bug: Any = None) -> int:
    return _evaluate(p, _derivative(p, cs, bug), x, bug)


def _scale(p: Params, cs: list[int], bug: Any = None) -> list[int]:
    return _trim(p, [c * p["factor"] for c in cs], bug)


FAMILY = Family(
    id="poly",
    domain="polynomials as integer coefficient lists: evaluation, derivatives, sums",
    modules={"main": ("poly", "polynomial", "coeffs"), "calc": ("calculus", "poly_ops", "algebra")},
    docs={
        "main": "Polynomials as coefficient lists, lowest power first.",
        "calc": "Calculus helpers.",
    },
    names={
        "evaluate": ("evaluate", "eval_at", "value_at"),
        "derivative": ("derivative", "differentiate", "slope_coeffs"),
        "trim": ("trim", "drop_zeros", "strip_zeros"),
        "add": ("add_poly", "poly_sum", "combine"),
        "degree": ("degree", "poly_degree", "order_of"),
        "slope": ("slope_at", "rate_at", "tangent_slope"),
        "scale": ("scale", "times_const", "multiply_by"),
        "cs": ("cs", "coeffs_in", "terms"),
    },
    consts={"factor": (2, 3, -1)},
    funcs=(
        Func(
            "evaluate",
            "main",
            "def $evaluate($cs, x):\n"
            '    """Value at x of the polynomial with coefficients $cs (lowest power first)."""\n'
            "    acc = 0\n"
            "    for c in reversed($cs):\n"
            "        acc = acc * x + c\n"
            "    return acc",
            _evaluate,
            public=[
                ("quadratic", ([1, 2, 3], 2)),
                ("cubic", ([0, -1, 0, 1], 3)),
                ("linear", ([5, 1], -4)),
            ],
            hidden=[
                ([], 5),
                ([7], 3),
                ([1, 1], 0),
                ([2, 0, 1], -1),
                ([0, 0, 0, 2], 2),
                ([3, -2, 1], 4),
                ([1, 1, 1, 1], 1),
                ([-5, 0, 2], 3),
            ],
        ),
        Func(
            "derivative",
            "main",
            "def $derivative($cs):\n"
            '    """Coefficients of the derivative, lowest power first; [] for a constant."""\n'
            "    return [i * $cs[i] for i in range(1, len($cs))]",
            _derivative,
            hidden=[
                ([],),
                ([4],),
                ([4, 3],),
                ([1, 2, 3],),
                ([0, 0, 5],),
                ([2, -1, 0, 4],),
                ([1, 1, 1, 1, 1],),
            ],
        ),
        Func(
            "trim",
            "main",
            "def $trim($cs):\n"
            '    """The coefficients without trailing zeros (the zero polynomial is [])."""\n'
            "    out = list($cs)\n"
            "    while out and out[-1] == 0:\n"
            "        out.pop()\n"
            "    return out",
            _trim,
            public=[("zeros", ([1, 0, 2, 0, 0],))],
            hidden=[([],), ([0, 0],), ([0, 3],)],
        ),
        Func(
            "add",
            "main",
            "def $add(first, second):\n"
            '    """Sum of two polynomials, without trailing zeros.\n\n'
            "    Coefficients of the same power are added; the shorter list counts as having\n"
            "    zeros for its missing higher powers. Trailing zero coefficients are removed\n"
            '    from the result.\n    """\n'
            "    n = max(len(first), len(second))\n"
            "    out = []\n"
            "    for i in range(n):\n"
            "        a = first[i] if i < len(first) else 0\n"
            "        b = second[i] if i < len(second) else 0\n"
            "        out.append(a + b)\n"
            "    return $trim(out)",
            _add,
            public=[
                ("same_length", ([1, 2], [3, 4])),
                ("longer_second", ([1], [0, 0, 5])),
                ("cancel", ([2, 3], [1, -3])),
            ],
            hidden=[
                ([], []),
                ([4], []),
                ([], [0, 2]),
                ([1, 2, 3], [1]),
                ([0, 0, 1], [0, 0, -1]),
                ([5, -5], [-5, 5]),
                ([1, 1, 1], [2, 2, 2]),
                ([0, 7], [3]),
            ],
            uses=("trim",),
        ),
        Func(
            "degree",
            "main",
            "def $degree($cs):\n"
            '    """Highest power with a nonzero coefficient; -1 for the zero polynomial."""\n'
            "    return len($trim($cs)) - 1",
            _degree,
            hidden=[([],), ([0, 4],), ([3, 0],)],
            uses=("trim",),
        ),
        Func(
            "slope",
            "calc",
            "def $slope($cs, x):\n"
            '    """Slope of the polynomial at x."""\n'
            "    return $evaluate($derivative($cs), x)",
            _slope,
            public=[("quadratic", ([1, 2, 3], 2)), ("cubic", ([0, 0, 0, 1], 1))],
            hidden=[([5], 3), ([5, 2], 7), ([0, 0, 1], -3), ([1, -4, 0, 2], 2), ([3, 1, 1], 0)],
            uses=("evaluate", "derivative"),
        ),
        Func(
            "scale",
            "calc",
            "def $scale($cs):\n"
            '    """Every coefficient times $factor, without trailing zeros."""\n'
            "    return $trim([c * $factor for c in $cs])",
            _scale,
            public=[("simple", ([1, 0, 2],))],
            hidden=[([],), ([0, 0],), ([3, -1],)],
            uses=("trim",),
        ),
    ),
)

TEMPLATES = (
    Template(
        id="poly1",
        family=FAMILY,
        workflow="W3",
        target="derivative",
        concept="derivative coefficients",
        bugs={
            "from_zero": Bug(edits=(("range(1, len($cs))", "range(len($cs))"),)),
            "no_power": Bug(edits=(("i * $cs[i]", "$cs[i]"),)),
            "shifted": Bug(edits=(("i * $cs[i]", "i * $cs[i - 1]"),)),
        },
        spec=(
            "derivative(cs) takes polynomial coefficients, lowest power first (cs[i] is the "
            "coefficient of x**i), and returns the derivative's coefficients the same way: "
            "the list [i * cs[i] for i = 1 .. len(cs) - 1]. It does not remove zeros; a "
            "constant or empty list gives []."
        ),
        domain="cs: list of 0..6 ints in -9..9.",
        returns="list of int",
        include=("add",),
        caller="slope",
    ),
    Template(
        id="poly2",
        family=FAMILY,
        workflow="W1",
        target="evaluate",
        concept="Horner evaluation",
        bugs={
            "not_reversed": Bug(edits=(("for c in reversed($cs):", "for c in $cs:"),)),
            "start_one": Bug(edits=(("    acc = 0\n", "    acc = 1\n"),)),
            "plus_times": Bug(edits=(("acc = acc * x + c", "acc = acc + x * c"),)),
        },
        spec=(
            "evaluate(cs, x) takes polynomial coefficients, lowest power first (cs[i] is the "
            "coefficient of x**i), and an int x, and returns the sum of cs[i] * x**i (0 for an "
            "empty list). x**0 is 1 for every x, 0 included, so at x == 0 the result is cs[0]."
        ),
        domain="cs: list of 0..6 ints in -9..9; x: int in -5..5.",
        returns="int",
        include=("trim", "scale"),
    ),
    Template(
        id="poly3",
        family=FAMILY,
        workflow="W4",
        target="add",
        concept="sum of two polynomials, trimmed",
        bugs={"stub_raise": Bug(stub="raise"), "stub_none": Bug(stub="none")},
        spec=(
            "add(first, second) takes two coefficient lists (lowest power first) and returns "
            "a new list whose element i is first[i] + second[i], a missing element counting as "
            "0, for i up to the longer length; then trailing zeros are removed (so the zero "
            "polynomial is [])."
        ),
        domain="first, second: lists of 0..6 ints in -9..9.",
        returns="list of int",
        include=("degree", "scale"),
    ),
)
