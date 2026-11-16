"""Family ``grid``: cell grids as lists of rows of 0/1 values."""

from __future__ import annotations

from typing import Any

from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template

Grid = list[list[int]]


def _inside(p: Params, cg: Grid, r: int, c: int, bug: Any = None) -> bool:
    return 0 <= r < len(cg) and 0 <= c < len(cg[0])


def _neighbours(p: Params, cg: Grid, r: int, c: int, bug: Any = None) -> int:
    count = 0
    columns = (0, 1) if bug == "no_left_col" else (-1, 0, 1)
    for dr in (-1, 0, 1):
        for dc in columns:
            if bug == "self":
                keep = True
            elif bug == "diagonal_only":
                keep = bool(dr and dc)
            else:
                keep = bool(dr or dc)
            if keep and _inside(p, cg, r + dr, c + dc):
                count += cg[r + dr][c + dc]
    return count


def _rotate(p: Params, cg: Grid, bug: Any = None) -> Grid:
    height = len(cg)
    return [[cg[height - 1 - r][c] for r in range(height)] for c in range(len(cg[0]))]


def _region(p: Params, cg: Grid, r: int, c: int, bug: Any = None) -> int:
    if not _inside(p, cg, r, c) or (not cg[r][c] and bug != "dead_start"):
        return 0
    seen = {(r, c)}
    todo = [(r, c)]
    steps = 3 if bug == "no_step_left" else 4
    while todo:
        y, x = todo.pop()
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1))[:steps]:
            if _inside(p, cg, ny, nx) and cg[ny][nx] and (ny, nx) not in seen:
                seen.add((ny, nx))
                todo.append((ny, nx))
    return len(seen) - 1 if bug == "count_less" else len(seen)


def _row_counts(p: Params, cg: Grid, bug: Any = None) -> list[int]:
    return [sum(row) for row in cg]


def _largest(p: Params, cg: Grid, bug: Any = None) -> int:
    best = 0
    for r in range(len(cg)):
        for c in range(len(cg[0])):
            best = max(best, _region(p, cg, r, c, bug))
    return best


def _busiest(p: Params, cg: Grid, bug: Any = None) -> tuple[int, int]:
    best = (0, 0)
    most = -1
    for r in range(len(cg)):
        for c in range(len(cg[0])):
            here = _neighbours(p, cg, r, c, bug)
            if here > most:
                most = here
                best = (r, c)
    return best


A = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
B = [[0, 1, 0, 1], [1, 1, 1, 0], [0, 1, 0, 0]]
C = [[1, 0, 1], [1, 0, 1], [1, 1, 1]]
D = [[0, 0], [0, 0]]
E = [[1]]
F = [[0, 1, 1, 0, 1], [1, 1, 0, 0, 1]]
H = [[1, 1, 1], [0, 0, 1], [1, 1, 1]]

FAMILY = Family(
    id="grid",
    domain="rectangular grids of 0/1 cells: bounds, neighbour counts, connected regions",
    modules={"main": ("grid", "board", "cells"), "scan": ("scan", "survey", "regions")},
    docs={"main": "Cell grids stored as lists of rows of 0 and 1.", "scan": "Whole-grid scans."},
    names={
        "inside": ("in_bounds", "on_board", "within_grid"),
        "neighbours": ("live_neighbours", "count_around", "near_count"),
        "rotate": ("rotate_right", "turn_clockwise", "rotate_cw"),
        "region": ("region_size", "area_of", "patch_size"),
        "row_counts": ("row_counts", "filled_per_row", "row_fill"),
        "largest": ("largest_region", "biggest_patch", "widest_area"),
        "busiest": ("busiest_cell", "crowded_cell", "most_crowded"),
        "cg": ("cells", "rows", "board_rows"),
    },
    consts={},
    funcs=(
        Func(
            "inside",
            "main",
            "def $inside($cg, r, c):\n"
            '    """True when row r and column c lie inside the grid."""\n'
            "    return 0 <= r < len($cg) and 0 <= c < len($cg[0])",
            _inside,
            hidden=[(A, 0, 0), (A, 3, 0), (A, 0, -1), (B, 2, 3), (B, 2, 4)],
        ),
        Func(
            "neighbours",
            "main",
            "def $neighbours($cg, r, c):\n"
            '    """Live cells among the eight neighbours of the cell at row r, column c."""\n'
            "    count = 0\n"
            "    for dr in (-1, 0, 1):\n"
            "        for dc in (-1, 0, 1):\n"
            "            if (dr or dc) and $inside($cg, r + dr, c + dc):\n"
            "                count += $cg[r + dr][c + dc]\n"
            "    return count",
            _neighbours,
            public=[("middle", (B, 1, 1)), ("corner", (A, 0, 0)), ("edge", (C, 1, 2))],
            hidden=[
                (A, 1, 1),
                (A, 2, 2),
                (B, 0, 0),
                (B, 2, 3),
                (C, 1, 1),
                (C, 0, 0),
                (E, 0, 0),
                (F, 0, 2),
                (F, 1, 4),
            ],
            uses=("inside",),
        ),
        Func(
            "rotate",
            "main",
            "def $rotate($cg):\n"
            '    """The grid turned a quarter turn clockwise: the first column, read from the\n'
            '    bottom row up, becomes the first row."""\n'
            "    height = len($cg)\n"
            "    return [[$cg[height - 1 - r][c] for r in range(height)] for c in "
            "range(len($cg[0]))]",
            _rotate,
            public=[("square", (A,)), ("wide", (F,)), ("single", (E,))],
            hidden=[(B,), (C,), (D,), ([[1, 0]],), ([[1], [0], [0]],), ([[0, 1], [1, 1]],)],
        ),
        Func(
            "region",
            "main",
            "def $region($cg, r, c):\n"
            '    """Live cells connected to (r, c) by up, down, left and right steps."""\n'
            "    if not $inside($cg, r, c) or not $cg[r][c]:\n"
            "        return 0\n"
            "    seen = {(r, c)}\n"
            "    todo = [(r, c)]\n"
            "    while todo:\n"
            "        y, x = todo.pop()\n"
            "        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):\n"
            "            if $inside($cg, ny, nx) and $cg[ny][nx] and ((ny, nx) not in seen):\n"
            "                seen.add((ny, nx))\n"
            "                todo.append((ny, nx))\n"
            "    return len(seen)",
            _region,
            hidden=[
                (A, 0, 0),
                (A, 2, 2),
                (A, 2, 0),
                (B, 1, 1),
                (B, 0, 3),
                (C, 0, 2),
                (D, 0, 0),
                (F, 0, 2),
                (F, 1, 0),
                ([[1, 1], [1, 1]], 1, 1),
                (H, 0, 0),
                (H, 2, 0),
            ],
            uses=("inside",),
        ),
        Func(
            "row_counts",
            "main",
            "def $row_counts($cg):\n"
            '    """Live cells in each row."""\n'
            "    return [sum(row) for row in $cg]",
            _row_counts,
            public=[("rows", (B,))],
            hidden=[(A,), (D,), (E,)],
        ),
        Func(
            "largest",
            "scan",
            "def $largest($cg):\n"
            '    """Size of the largest group of connected live cells (0 when none)."""\n'
            "    best = 0\n"
            "    for r in range(len($cg)):\n"
            "        for c in range(len($cg[0])):\n"
            "            best = max(best, $region($cg, r, c))\n"
            "    return best",
            _largest,
            public=[("ring", (C,)), ("hook", (H,)), ("pieces", (F,)), ("empty", (D,))],
            hidden=[(A,), (B,), (E,), ([[0, 1], [0, 0]],), ([[1, 0, 1], [0, 1, 0]],), (F,)],
            uses=("region",),
        ),
        Func(
            "busiest",
            "scan",
            "def $busiest($cg):\n"
            '    """(row, column) of the cell with the most live neighbours; the first in row\n'
            '    order wins ties."""\n'
            "    best = (0, 0)\n"
            "    most = -1\n"
            "    for r in range(len($cg)):\n"
            "        for c in range(len($cg[0])):\n"
            "            here = $neighbours($cg, r, c)\n"
            "            if here > most:\n"
            "                most = here\n"
            "                best = (r, c)\n"
            "    return best",
            _busiest,
            public=[("plus", (B,))],
            hidden=[(A,), (C,), (D,), (F,)],
            uses=("neighbours",),
        ),
    ),
)

TEMPLATES = (
    Template(
        id="grid1",
        family=FAMILY,
        workflow="W1",
        target="neighbours",
        concept="eight-neighbour live count with bounds",
        bugs={
            "self": Bug(edits=(("(dr or dc) and $inside(", "$inside("),)),
            "no_left_col": Bug(edits=(("for dc in (-1, 0, 1):", "for dc in (0, 1):"),)),
            "diagonal_only": Bug(edits=(("(dr or dc)", "(dr and dc)"),)),
        },
        spec=(
            "neighbours(grid, r, c) counts live cells around a cell. grid is a non-empty list "
            "of equal-length rows of 0/1 ints. Consider the eight positions (r + dr, c + dc) "
            "for dr, dc in {-1, 0, 1}, excluding (r, c) itself; ignore positions outside the "
            "grid (row index outside 0..len(grid)-1 or column outside 0..len(grid[0])-1). "
            "Return the sum of grid values at the remaining positions."
        ),
        domain=("grid: 1..6 rows of equal length 1..6, values 0 or 1; r, c: a valid cell of grid."),
        returns="int",
        include=("row_counts", "largest"),
    ),
    Template(
        id="grid2",
        family=FAMILY,
        workflow="W3",
        target="region",
        concept="four-connected region size",
        bugs={
            "count_less": Bug(edits=(("return len(seen)", "return len(seen) - 1"),)),
            "no_step_left": Bug(
                edits=(
                    (
                        "((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1))",
                        "((y + 1, x), (y - 1, x), (y, x + 1))",
                    ),
                )
            ),
            "dead_start": Bug(
                edits=(
                    ("if not $inside($cg, r, c) or not $cg[r][c]:", "if not $inside($cg, r, c):"),
                )
            ),
        },
        spec=(
            "region(grid, r, c) returns the size of the group of live cells containing (r, c). "
            "grid is a non-empty list of equal-length rows of 0/1 ints. If (r, c) is outside "
            "the grid or grid[r][c] is 0, return 0. Otherwise return the number of live cells "
            "reachable from (r, c) by repeated steps up, down, left or right through live "
            "cells, (r, c) included. Diagonal steps do not connect."
        ),
        domain="grid: 1..6 rows of equal length 1..6, values 0 or 1; r, c: ints in -1..6.",
        returns="int",
        caller="largest",
    ),
    Template(
        id="grid3",
        family=FAMILY,
        workflow="W4",
        target="rotate",
        concept="quarter-turn clockwise rotation",
        bugs={"stub_raise": Bug(stub="raise"), "stub_none": Bug(stub="none")},
        spec=(
            "rotate(grid) returns a new grid: the input turned a quarter turn clockwise. For an "
            "input with h rows and w columns the result has w rows and h columns, and "
            "result[i][j] == grid[h - 1 - j][i]. The input is not changed."
        ),
        domain="grid: 1..5 rows of equal length 1..5, values 0 or 1.",
        returns="list of lists of int",
        include=("row_counts", "busiest"),
    ),
)
