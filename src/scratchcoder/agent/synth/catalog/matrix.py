"""Family ``matrix``: integer matrices as lists of rows."""

from __future__ import annotations

from typing import Any

from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template

Matrix = list[list[int]]


def _transpose(p: Params, mt: Matrix, bug: Any = None) -> Matrix:
    if bug == "no_swap":
        return [list(row) for row in mt]
    rows = mt[:-1] if bug == "drop_last" else mt
    return [[row[c] for row in rows] for c in range(len(mt[0]))]


def _diag(p: Params, mt: Matrix, bug: Any = None) -> int:
    n = len(mt)
    total = 0
    for i in range(n):
        total += mt[i][i] + mt[i][n - i if bug == "anti_shift" else n - 1 - i]
    if bug != "double_centre" and n % 2 == (0 if bug == "even_centre" else 1):
        total -= mt[n // 2][n // 2]
    return total


def _col_sums(p: Params, mt: Matrix, bug: Any = None) -> list[int]:
    return [sum(col) for col in _transpose(p, mt, bug)]


def _mat_vec(p: Params, mt: Matrix, vec: list[int], bug: Any = None) -> list[int]:
    return [sum(a * b for a, b in zip(row, vec, strict=False)) for row in mt]


def _scaled(p: Params, mt: Matrix, bug: Any = None) -> Matrix:
    return [[p["factor"] * x for x in row] for row in mt]


M2 = [[1, 2], [3, 4]]
M3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
W = [[1, 0, 2], [0, 3, 1]]
M4 = [[2, 0, 0, 1], [0, 3, 1, 0], [0, 1, 4, 0], [5, 0, 0, 6]]

FAMILY = Family(
    id="matrix",
    domain="integer matrices as lists of rows: transpose, diagonals, products, scaling",
    modules={"main": ("matrix", "mat", "matrices"), "ops": ("ops", "linalg", "products")},
    docs={"main": "Integer matrices stored as lists of rows.", "ops": "Matrix operations."},
    names={
        "transpose": ("transpose", "flip_axes", "swap_axes"),
        "diag": ("diagonal_sum", "cross_sum", "both_diagonals"),
        "col_sums": ("column_sums", "col_totals", "sum_columns"),
        "mat_vec": ("mat_vec", "apply_matrix", "times_vector"),
        "scaled": ("scaled", "times_factor", "multiply_all"),
        "mt": ("m", "mat_rows", "grid"),
    },
    consts={"factor": (2, 3, 10)},
    funcs=(
        Func(
            "transpose",
            "main",
            "def $transpose($mt):\n"
            '    """Rows become columns: entry [r][c] moves to [c][r]."""\n'
            "    return [[row[c] for row in $mt] for c in range(len($mt[0]))]",
            _transpose,
            hidden=[([[7]],), (M2,), (M3,), (W,), ([[1], [2], [3]],), ([[1, 2, 3]],)],
        ),
        Func(
            "diag",
            "main",
            "def $diag($mt):\n"
            '    """Sum of both diagonals of a square matrix, the centre cell counted once."""\n'
            "    n = len($mt)\n"
            "    total = 0\n"
            "    for i in range(n):\n"
            "        total += $mt[i][i] + $mt[i][n - 1 - i]\n"
            "    if n % 2 == 1:\n"
            "        total -= $mt[n // 2][n // 2]\n"
            "    return total",
            _diag,
            public=[("odd", (M3,)), ("even", (M2,)), ("four", (M4,))],
            hidden=[
                ([[7]],),
                ([[0, 0], [0, 0]],),
                ([[1, 1], [1, 1]],),
                ([[2, 0, 1], [0, 5, 0], [3, 0, 4]],),
                (M4,),
                ([[1, 2, 3, 4, 5]] * 5,),
                ([[9, 8], [7, 6]],),
            ],
        ),
        Func(
            "col_sums",
            "ops",
            "def $col_sums($mt):\n"
            '    """Sum of each column."""\n'
            "    return [sum(col) for col in $transpose($mt)]",
            _col_sums,
            public=[("square", (M2,)), ("wide", (W,))],
            hidden=[([[7]],), (M3,), ([[1], [2], [3]],), ([[1, 2, 3]],), (M4,)],
            uses=("transpose",),
        ),
        Func(
            "mat_vec",
            "ops",
            "def $mat_vec($mt, vec):\n"
            '    """The matrix times a column vector, as a list with one entry per row."""\n'
            "    out = []\n"
            "    for row in $mt:\n"
            "        acc = 0\n"
            "        for a, b in zip(row, vec):\n"
            "            acc += a * b\n"
            "        out.append(acc)\n"
            "    return out",
            _mat_vec,
            public=[("square", (M2, [1, 1])), ("wide", (W, [2, 0, 1])), ("zero", (M3, [0, 0, 0]))],
            hidden=[
                ([[5]], [3]),
                (M2, [0, 1]),
                (M2, [2, 3]),
                (M3, [1, 0, 1]),
                (W, [1, 1, 1]),
                ([[1, 2, 3]], [4, 5, 6]),
                ([[1], [2]], [3]),
            ],
        ),
        Func(
            "scaled",
            "main",
            "def $scaled($mt):\n"
            '    """Every entry multiplied by $factor."""\n'
            "    return [[$factor * x for x in row] for row in $mt]",
            _scaled,
            public=[("small", (M2,))],
            hidden=[([[0]],), (W,)],
        ),
    ),
)

TEMPLATES = (
    Template(
        id="matrix1",
        family=FAMILY,
        workflow="W1",
        target="diag",
        concept="sum of both diagonals, centre counted once",
        bugs={
            "double_centre": Bug(
                edits=(("    if n % 2 == 1:\n        total -= $mt[n // 2][n // 2]\n", ""),)
            ),
            "anti_shift": Bug(edits=(("$mt[i][n - 1 - i]", "$mt[i][n - i]"),)),
            "even_centre": Bug(edits=(("if n % 2 == 1:", "if n % 2 == 0:"),)),
        },
        spec=(
            "diag(m) takes a square n x n matrix (list of n rows of n ints, n >= 1) and returns "
            "the sum of m[i][i] and m[i][n - 1 - i] over all i, minus m[n // 2][n // 2] once "
            "when n is odd (so a cell on both diagonals is counted once)."
        ),
        domain="m: square matrix of size 1..5 with ints in -9..9.",
        returns="int",
        include=("scaled", "col_sums"),
    ),
    Template(
        id="matrix2",
        family=FAMILY,
        workflow="W3",
        target="transpose",
        concept="matrix transpose",
        bugs={
            "no_swap": Bug(
                edits=(
                    (
                        "return [[row[c] for row in $mt] for c in range(len($mt[0]))]",
                        "return [list(row) for row in $mt]",
                    ),
                )
            ),
            "drop_last": Bug(edits=(("for row in $mt]", "for row in $mt[:-1]]"),)),
        },
        spec=(
            "transpose(m) takes a matrix with r >= 1 rows of c >= 1 ints each and returns a new "
            "matrix with c rows of r ints, where result[j][i] == m[i][j]."
        ),
        domain="m: matrix with 1..4 rows of equal length 1..4, ints in -9..9.",
        returns="list of lists of int",
        include=("scaled",),
        caller="col_sums",
    ),
    Template(
        id="matrix3",
        family=FAMILY,
        workflow="W4",
        target="mat_vec",
        concept="matrix times a column vector",
        bugs={"stub_raise": Bug(stub="raise"), "stub_none": Bug(stub="none")},
        spec=(
            "mat_vec(m, vec) takes a matrix (list of rows of ints) and a list of ints of the "
            "same length as each row, and returns a list with one int per row: the sum of "
            "row[j] * vec[j] over j."
        ),
        domain=(
            "m: 1..4 rows of equal length 1..4, ints in -9..9; vec: list of ints in -9..9, "
            "one per column."
        ),
        returns="list of int",
        include=("diag", "col_sums"),
    ),
)
