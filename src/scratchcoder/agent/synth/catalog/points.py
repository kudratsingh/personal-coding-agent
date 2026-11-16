"""Family ``points``: integer lattice points as (x, y) tuples."""

from __future__ import annotations

from typing import Any

from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template

Point = tuple[int, int]


def _manhattan(p: Params, a: Point, b: Point, bug: Any = None) -> int:
    dx = abs(a[0] - b[0])
    if bug == "signed_y":
        return dx + (a[1] - b[1])
    if bug == "x_for_y":
        return dx + abs(a[0] - b[1])
    if bug == "max_not_sum":
        return max(dx, abs(a[1] - b[1]))
    return dx + abs(a[1] - b[1])


def _nearest(p: Params, origin: Point, pts: list[Point], bug: Any = None) -> Any:
    best = None
    for pt in pts:
        if best is None or _manhattan(p, origin, pt, bug) < _manhattan(p, origin, best, bug):
            best = pt
    return best


def _quadrant(p: Params, pt: Point, bug: Any = None) -> int:
    x, y = pt
    if (x == 0 and y == 0) if bug == "axis_and" else (x == 0 or y == 0):
        return 0
    if x > 0:
        if bug == "swap_right":
            return 4 if y > 0 else 1
        return 1 if y > 0 else 4
    if bug == "swap_left":
        return 3 if y > 0 else 2
    return 2 if y > 0 else 3


def _centroid(p: Params, pts: list[Point], bug: Any = None) -> Point:
    n = len(pts)
    return (sum(q[0] for q in pts) // n, sum(q[1] for q in pts) // n)


def _within(p: Params, origin: Point, pts: list[Point], bug: Any = None) -> list[Point]:
    return [q for q in pts if _manhattan(p, origin, q, bug) <= p["radius"]]


def _walk(p: Params, pts: list[Point], bug: Any = None) -> int:
    total = 0
    for i in range(len(pts)):
        total += _manhattan(p, pts[i], pts[(i + 1) % len(pts)], bug)
    return total


FAMILY = Family(
    id="points",
    domain="integer lattice points: grid distance, nearest point, quadrants, closed walks",
    modules={"main": ("points", "lattice", "plane"), "search": ("nearest", "lookup", "search")},
    docs={"main": "Integer points stored as (x, y) tuples.", "search": "Searches over points."},
    names={
        "manhattan": ("manhattan", "taxi_distance", "block_distance"),
        "nearest": ("nearest_point", "closest", "closest_point"),
        "quadrant": ("quadrant", "quarter_of", "sector_of"),
        "centroid": ("centroid", "mean_point", "center_of"),
        "within": ("within_radius", "points_near", "nearby"),
        "walk": ("walk_length", "loop_length", "route_steps"),
    },
    consts={"radius": (2, 3, 5)},
    funcs=(
        Func(
            "manhattan",
            "main",
            "def $manhattan(a, b):\n"
            '    """Grid distance between two (x, y) points: |dx| + |dy|."""\n'
            "    return abs(a[0] - b[0]) + abs(a[1] - b[1])",
            _manhattan,
            hidden=[
                ((0, 0), (0, 0)),
                ((0, 0), (3, 4)),
                ((3, 4), (0, 0)),
                ((-2, 5), (1, -1)),
                ((5, 1), (5, 9)),
                ((1, 7), (7, 1)),
                ((-3, -3), (-1, -8)),
            ],
        ),
        Func(
            "nearest",
            "search",
            "def $nearest(origin, pts):\n"
            '    """The point of pts closest to origin by grid distance; the earliest wins\n'
            '    ties; None when pts is empty."""\n'
            "    best = None\n"
            "    for pt in pts:\n"
            "        if best is None or $manhattan(origin, pt) < $manhattan(origin, best):\n"
            "            best = pt\n"
            "    return best",
            _nearest,
            public=[
                ("below", ((0, 0), [(0, 5), (1, -2), (4, 4)])),
                ("diagonal", ((2, 2), [(6, 2), (4, 4), (2, 7)])),
                ("column", ((0, 5), [(0, 0), (0, 6)])),
                ("empty", ((1, 1), [])),
            ],
            hidden=[
                ((0, 0), [(3, 3)]),
                ((0, 0), [(0, -4), (3, 0)]),
                ((5, 5), [(5, 9), (9, 5), (6, 6)]),
                ((1, 1), [(1, 6), (5, 1), (0, 0)]),
                ((-2, 0), [(2, 0), (-2, -3), (0, 3)]),
                ((0, 0), [(4, 0), (0, 4), (-4, 0)]),
            ],
            uses=("manhattan",),
        ),
        Func(
            "quadrant",
            "main",
            "def $quadrant(pt):\n"
            '    """Quadrant 1 to 4, counter-clockwise from x > 0, y > 0; 0 for a point on an\n'
            '    axis."""\n'
            "    x, y = pt\n"
            "    if x == 0 or y == 0:\n"
            "        return 0\n"
            "    if x > 0:\n"
            "        return 1 if y > 0 else 4\n"
            "    return 2 if y > 0 else 3",
            _quadrant,
            public=[
                ("upper_right", ((3, 4),)),
                ("lower_left", ((-1, -1),)),
                ("axis", ((0, 5),)),
                ("lower_right", ((2, -6),)),
            ],
            hidden=[
                ((0, 0),),
                ((7, 0),),
                ((1, 1),),
                ((-4, 2),),
                ((-4, -2),),
                ((4, -2),),
                ((0, -3),),
                ((-9, 9),),
            ],
        ),
        Func(
            "centroid",
            "main",
            "def $centroid(pts):\n"
            '    """(mean x, mean y) of a non-empty list of points, each rounded down."""\n'
            "    n = len(pts)\n"
            "    return (sum([q[0] for q in pts]) // n, sum([q[1] for q in pts]) // n)",
            _centroid,
            public=[("square", ([(0, 0), (2, 0), (2, 2), (0, 2)],))],
            hidden=[([(5, 5)],), ([(0, 0), (1, 1)],), ([(-3, 1), (2, 2), (0, 0)],)],
        ),
        Func(
            "within",
            "search",
            "def $within(origin, pts):\n"
            '    """Points of pts at grid distance $radius or less from origin, in order."""\n'
            "    return [q for q in pts if $manhattan(origin, q) <= $radius]",
            _within,
            public=lambda p: [("edge", ((0, 0), [(p["radius"], 0), (p["radius"], 1)]))],
            hidden=lambda p: [
                ((0, 0), []),
                ((1, 1), [(1, 1), (9, 9)]),
                ((0, 0), [(0, -p["radius"])]),
            ],
            uses=("manhattan",),
        ),
        Func(
            "walk",
            "main",
            "def $walk(pts):\n"
            '    """Grid length of the closed walk through the points in order and back to the\n'
            "    first one (each step costs the grid distance |dx| + |dy| between the two\n"
            '    points; 0 for fewer than two points)."""\n'
            "    total = 0\n"
            "    for i in range(len(pts)):\n"
            "        total += $manhattan(pts[i], pts[(i + 1) % len(pts)])\n"
            "    return total",
            _walk,
            public=[
                ("square", ([(0, 0), (3, 0), (3, 3), (0, 3)],)),
                ("line", ([(1, 1), (4, 5)],)),
                ("single", ([(2, 9)],)),
            ],
            hidden=[
                ([],),
                ([(0, 0)],),
                ([(0, 0), (0, 0)],),
                ([(0, 0), (1, 0), (1, 1)],),
                ([(-2, -2), (2, -2), (2, 2), (-2, 2)],),
                ([(5, 1), (1, 5), (5, 5)],),
                ([(0, 0), (10, 0), (0, 0)],),
            ],
            uses=("manhattan",),
        ),
    ),
)

TEMPLATES = (
    Template(
        id="points1",
        family=FAMILY,
        workflow="W3",
        target="manhattan",
        concept="grid (taxicab) distance",
        bugs={
            "signed_y": Bug(edits=(("abs(a[1] - b[1])", "(a[1] - b[1])"),)),
            "x_for_y": Bug(edits=(("abs(a[1] - b[1])", "abs(a[0] - b[1])"),)),
            "max_not_sum": Bug(
                edits=(
                    (
                        "abs(a[0] - b[0]) + abs(a[1] - b[1])",
                        "max(abs(a[0] - b[0]), abs(a[1] - b[1]))",
                    ),
                )
            ),
        },
        spec=(
            "manhattan(a, b) takes two (x, y) tuples of ints and returns |a[0] - b[0]| + "
            "|a[1] - b[1]|."
        ),
        domain="a, b: (int, int) tuples with coordinates in -10..10.",
        returns="int",
        include=("centroid",),
        caller="nearest",
    ),
    Template(
        id="points2",
        family=FAMILY,
        workflow="W1",
        target="quadrant",
        concept="quadrant of a lattice point",
        bugs={
            "axis_and": Bug(edits=(("if x == 0 or y == 0:", "if x == 0 and y == 0:"),)),
            "swap_right": Bug(edits=(("return 1 if y > 0 else 4", "return 4 if y > 0 else 1"),)),
            "swap_left": Bug(edits=(("return 2 if y > 0 else 3", "return 3 if y > 0 else 2"),)),
        },
        spec=(
            "quadrant(pt) takes an (x, y) tuple of ints and returns 0 when x == 0 or y == 0; "
            "otherwise 1 for x > 0 and y > 0, 2 for x < 0 and y > 0, 3 for x < 0 and y < 0, "
            "and 4 for x > 0 and y < 0."
        ),
        domain="pt: (int, int) tuple with coordinates in -10..10.",
        returns="int",
        include=("centroid", "within"),
    ),
    Template(
        id="points3",
        family=FAMILY,
        workflow="W4",
        target="walk",
        concept="length of a closed walk through points",
        bugs={"stub_raise": Bug(stub="raise"), "stub_none": Bug(stub="none")},
        spec=(
            "walk(pts) takes a list of (x, y) tuples of ints and returns the sum, over i in "
            "0..len(pts)-1, of the grid distance |dx| + |dy| from pts[i] to pts[(i + 1) % "
            "len(pts)] (the walk closes back to the first point). An empty list gives 0, and "
            "a single point gives 0 (its one step returns to itself)."
        ),
        domain="pts: list of 0..6 (int, int) tuples with coordinates in -10..10.",
        returns="int",
        include=("quadrant", "within"),
    ),
)
