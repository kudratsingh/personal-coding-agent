"""Family ``runs``: run-length encoding of integer lists."""

from __future__ import annotations

from typing import Any

from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template


def _encode(p: Params, values: list[int], bug: Any = None) -> list[list[int]]:
    out: list[list[int]] = []
    for v in values:
        if bug == "merge_all":
            same = bool(out)
        else:
            same = bool(out) and (out[-1][0] != v if bug == "inverted" else out[-1][0] == v)
        if same:
            out[-1][1] += 1
        else:
            out.append([v, 0 if bug == "start_zero" else 1])
    return out


def _decode(p: Params, pairs: list[Any], bug: Any = None) -> list[int]:
    out: list[int] = []
    for v, n in pairs[0] if bug == "unpack_first" else pairs:
        out.extend(v * n if bug == "no_list" else [v] * n)
    return out


def _expanded_sum(p: Params, pairs: list[Any], bug: Any = None) -> int:
    return sum(_decode(p, pairs, bug))


def _longest_run(p: Params, values: list[int], bug: Any = None) -> Any:
    best: Any = None
    for v, n in _encode(p, values, bug):
        if best is None or (n >= best[1] if bug == "latest_tie" else n > best[1]):
            best = (n, v) if bug == "swapped" else (v, n)
    return best


def _run_count(p: Params, values: list[int], bug: Any = None) -> int:
    return len(_encode(p, values, bug))


FAMILY = Family(
    id="runs",
    domain="run-length encoding of integer lists: encode, decode, longest run",
    modules={"main": ("runs", "rle", "packing"), "stats": ("run_stats", "streaks", "summaries")},
    docs={"main": "Run-length encoding of lists of ints.", "stats": "Statistics over runs."},
    names={
        "encode": ("encode_runs", "run_length", "pack_runs"),
        "decode": ("decode_runs", "expand_runs", "unpack_runs"),
        "expanded_sum": ("expanded_sum", "sum_unpacked", "decoded_sum"),
        "longest_run": ("longest_run", "max_run", "biggest_run"),
        "run_count": ("run_count", "count_runs", "number_of_runs"),
    },
    consts={},
    funcs=(
        Func(
            "encode",
            "main",
            "def $encode(values):\n"
            '    """[value, count] pairs for each run of equal neighbouring values."""\n'
            "    out = []\n"
            "    for v in values:\n"
            "        if out and out[-1][0] == v:\n"
            "            out[-1][1] += 1\n"
            "        else:\n"
            "            out.append([v, 1])\n"
            "    return out",
            _encode,
            hidden=[
                ([],),
                ([4],),
                ([4, 4],),
                ([1, 2, 3],),
                ([5, 5, 1, 5],),
                ([0, 0, 0, 2, 2, 0],),
                ([7, 8, 8, 8, 9, 9],),
            ],
        ),
        Func(
            "decode",
            "main",
            "def $decode(pairs):\n"
            '    """The values that [value, count] pairs stand for."""\n'
            "    out = []\n"
            "    for v, n in pairs:\n"
            "        out.extend([v] * n)\n"
            "    return out",
            _decode,
            hidden=[
                ([],),
                ([[3, 1]],),
                ([[3, 0]],),
                ([[1, 2], [2, 1]],),
                ([[9, 3], [9, 1]],),
                ([[0, 4], [5, 2], [0, 1]],),
            ],
        ),
        Func(
            "expanded_sum",
            "stats",
            "def $expanded_sum(pairs):\n"
            '    """Sum of the values that [value, count] pairs stand for."""\n'
            "    return sum($decode(pairs))",
            _expanded_sum,
            public=[("two", ([[2, 3], [5, 1]],)), ("empty", ([],))],
            hidden=[([[1, 1]],), ([[4, 0], [6, 2]],), ([[1, 5], [2, 5]],), ([],)],
            uses=("decode",),
        ),
        Func(
            "longest_run",
            "stats",
            "def $longest_run(values):\n"
            '    """(value, length) of the longest run of equal values; the earliest run wins\n'
            '    ties; None for no values."""\n'
            "    best = None\n"
            "    for v, n in $encode(values):\n"
            "        if best is None or n > best[1]:\n"
            "            best = (v, n)\n"
            "    return best",
            _longest_run,
            public=[("clear", ([1, 3, 3, 3, 2],)), ("tie", ([4, 4, 6, 6],)), ("none", ([],))],
            hidden=[
                ([5],),
                ([1, 2, 3],),
                ([2, 2, 2, 1, 1, 1],),
                ([9, 8, 8, 7, 7, 7],),
                ([0, 0, 1, 0, 0, 0],),
                ([3, 3, 4, 4, 4, 3, 3, 3],),
            ],
            uses=("encode",),
        ),
        Func(
            "run_count",
            "stats",
            "def $run_count(values):\n"
            '    """Number of runs of equal neighbouring values."""\n'
            "    return len($encode(values))",
            _run_count,
            public=[("three", ([1, 1, 2, 1],)), ("solo", ([8],))],
            hidden=[
                ([],),
                ([2, 2, 2],),
                ([1, 2, 1, 2],),
                ([3, 3, 1, 1, 3],),
                ([0, 1],),
                ([6, 6, 6, 5],),
            ],
            uses=("encode",),
        ),
    ),
)

TEMPLATES = (
    Template(
        id="runs1",
        family=FAMILY,
        workflow="W3",
        target="encode",
        concept="run-length encoding",
        bugs={
            "inverted": Bug(edits=(("out[-1][0] == v", "out[-1][0] != v"),)),
            "start_zero": Bug(edits=(("out.append([v, 1])", "out.append([v, 0])"),)),
            "merge_all": Bug(edits=(("if out and out[-1][0] == v:", "if out:"),)),
        },
        spec=(
            "encode(values) returns a list of [value, count] lists (lists, not tuples), one per "
            "maximal run of equal adjacent elements of values, in order; [] for no values."
        ),
        domain="values: list of 0..10 ints in 0..9.",
        returns="list of [int, int] lists",
        include=("expanded_sum",),
        caller="longest_run",
    ),
    Template(
        id="runs2",
        family=FAMILY,
        workflow="W5",
        target="decode",
        concept="expand run-length pairs",
        bugs={
            "no_list": Bug(edits=(("out.extend([v] * n)", "out.extend(v * n)"),)),
            "unpack_first": Bug(edits=(("for v, n in pairs:", "for v, n in pairs[0]:"),)),
        },
        spec=(
            "decode(pairs) takes a list of [value, count] pairs (count >= 0) and returns the "
            "list holding each value repeated count times, pairs in order."
        ),
        domain="pairs: list of 0..5 [int, int] lists, values in 0..9, counts in 0..4.",
        returns="list of int",
        include=("run_count",),
        caller="expanded_sum",
    ),
    Template(
        id="runs3",
        family=FAMILY,
        workflow="W1",
        target="longest_run",
        concept="longest run with an earliest-first tie-break",
        bugs={
            "latest_tie": Bug(edits=(("n > best[1]", "n >= best[1]"),)),
            "swapped": Bug(edits=(("best = (v, n)", "best = (n, v)"),)),
        },
        spec=(
            "longest_run(values) returns None for an empty list. Otherwise it returns a tuple "
            "(value, length) for the longest maximal run of equal adjacent elements; when "
            "several runs share the greatest length, the first such run."
        ),
        domain="values: list of 0..10 ints in 0..9.",
        returns="tuple (int, int) or None",
        include=("decode",),
    ),
)
