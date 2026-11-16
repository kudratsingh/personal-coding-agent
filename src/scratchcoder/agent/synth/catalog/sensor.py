"""Family ``sensor``: series of integer sensor readings."""

from __future__ import annotations

from typing import Any

from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template


def _smooth(p: Params, xs: list[int], bug: Any = None) -> list[int]:
    width = p["width"]
    stop = len(xs) - width + (0 if bug == "last_window" else 1)
    span = width - 1 if bug == "short_slice" else width
    divisor = width + 1 if bug == "divisor" else width
    return [sum(xs[i : i + span]) // divisor for i in range(stop)]


def _rises(p: Params, xs: list[int], bug: Any = None) -> int:
    level = p["level"]
    return sum(1 for prev, cur in zip(xs, xs[1:], strict=False) if cur > level and prev <= level)


def _longest(p: Params, xs: list[int], bug: Any = None) -> int:
    best = run = 0
    for x in xs:
        if x > p["level"]:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _clip(p: Params, xs: list[int], bug: Any = None) -> list[int]:
    jump = p["jump"]
    out: list[int] = []
    for x in xs:
        if out and x - out[-1] > jump:
            x = out[-1] + jump
        elif out and out[-1] - x > jump:
            x = out[-1] - jump
        out.append(x)
    return out


def _peak(p: Params, xs: list[int], bug: Any = None) -> int:
    best = len(xs) if bug == "start_len" else 0
    stop = len(xs) + 1 if bug == "range_end" else len(xs)
    for i in range(1, stop):
        if xs[i] > xs[best]:
            best = xs[i] if bug == "value_kept" else i
    return best


def _alarm(p: Params, xs: list[int], bug: Any = None) -> Any:
    if not xs:
        return None
    i = _peak(p, xs, bug)
    if xs[i] > p["level"]:
        return (i, xs[i])
    return None


FAMILY = Family(
    id="sensor",
    domain="series of integer sensor readings: windows, thresholds, spikes, peaks",
    modules={"main": ("readings", "sensor", "samples"), "alerts": ("alerts", "alarms", "monitor")},
    docs={"main": "Helpers for a series of integer sensor readings.", "alerts": "Alert rules."},
    names={
        "smooth": ("smooth", "rolling_mean", "window_mean"),
        "rises": ("rising_edges", "crossings", "upward_crossings"),
        "longest": ("longest_high", "longest_above", "max_high_run"),
        "clip": ("clip_spikes", "cap_spikes", "limit_jumps"),
        "peak": ("peak_index", "highest_at", "argmax_reading"),
        "alarm": ("alarm_reading", "alert_for", "check_alarm"),
        "xs": ("values", "series", "samples_in"),
    },
    consts={"width": (2, 3, 4), "level": (10, 20, 50), "jump": (5, 8)},
    funcs=(
        Func(
            "smooth",
            "main",
            "def $smooth($xs):\n"
            '    """Mean of each full window of $width consecutive values, rounded down."""\n'
            "    out = []\n"
            "    for i in range(len($xs) - $width + 1):\n"
            "        out.append(sum($xs[i:i + $width]) // $width)\n"
            "    return out",
            _smooth,
            public=lambda p: [
                ("steady", ([6] * (p["width"] + 2),)),
                ("ramp", (list(range(1, p["width"] + 4)),)),
                ("exact", (list(range(10, 10 + p["width"])),)),
            ],
            hidden=lambda p: [
                ([],),
                ([5] * (p["width"] - 1),),
                ([3] * p["width"],),
                (list(range(0, 30, 3)),),
                ([9, 1] * p["width"],),
                ([0, 0, 7, 7, 7, 0, 0],),
                ([100, 1, 50, 2, 25, 3],),
                (list(range(20, 0, -2)),),
            ],
        ),
        Func(
            "rises",
            "main",
            "def $rises($xs):\n"
            '    """How many times a value is above $level while the one before it is not."""\n'
            "    count = 0\n"
            "    for prev, cur in zip($xs, $xs[1:]):\n"
            "        if cur > $level and prev <= $level:\n"
            "            count += 1\n"
            "    return count",
            _rises,
            public=lambda p: [("two", ([0, p["level"] + 1, 0, p["level"] + 5],))],
            hidden=lambda p: [([],), ([p["level"]] * 3,), ([p["level"], p["level"] + 1],)],
        ),
        Func(
            "longest",
            "main",
            "def $longest($xs):\n"
            '    """Length of the longest run of consecutive values above $level."""\n'
            "    best = 0\n"
            "    run = 0\n"
            "    for x in $xs:\n"
            "        if x > $level:\n"
            "            run += 1\n"
            "            best = max(best, run)\n"
            "        else:\n"
            "            run = 0\n"
            "    return best",
            _longest,
            public=lambda p: [("runs", ([p["level"] + 1] * 2 + [0] + [p["level"] + 2] * 3,))],
            hidden=lambda p: [([],), ([p["level"]] * 4,), ([p["level"] + 1],)],
        ),
        Func(
            "clip",
            "main",
            "def $clip($xs):\n"
            '    """A new list where no value moves more than $jump from the value kept before '
            "it.\n\n"
            "    The first value is kept as it is. A later value more than $jump above the\n"
            "    last kept value becomes that value plus $jump; one more than $jump below it\n"
            '    becomes that value minus $jump; any other value is kept as it is.\n    """\n'
            "    out = []\n"
            "    for x in $xs:\n"
            "        if out and x - out[-1] > $jump:\n"
            "            x = out[-1] + $jump\n"
            "        elif out and out[-1] - x > $jump:\n"
            "            x = out[-1] - $jump\n"
            "        out.append(x)\n"
            "    return out",
            _clip,
            public=lambda p: [
                ("calm", ([3, 4, 2],)),
                ("spike", ([10, 10 + p["jump"] * 3, 10],)),
                ("drop", ([50, 50 - p["jump"] - 1, 50 - p["jump"] - 1],)),
            ],
            hidden=lambda p: [
                ([],),
                ([7],),
                ([0, p["jump"]],),
                ([0, p["jump"] + 1],),
                ([0, -p["jump"] - 1],),
                ([100, 0, 0, 0],),
                ([0, 100, 100, 100],),
                ([5, 5 + p["jump"], 5],),
            ],
        ),
        Func(
            "peak",
            "main",
            "def $peak($xs):\n"
            '    """Index of the largest value; the first one wins ties."""\n'
            "    best = 0\n"
            "    for i in range(1, len($xs)):\n"
            "        if $xs[i] > $xs[best]:\n"
            "            best = i\n"
            "    return best",
            _peak,
            hidden=[
                ([4],),
                ([1, 2, 3],),
                ([3, 2, 1],),
                ([2, 9, 9, 1],),
                ([0, 0, 0],),
                ([5, 1, 5, 7, 2],),
            ],
        ),
        Func(
            "alarm",
            "alerts",
            "def $alarm($xs):\n"
            '    """(index, value) of the highest reading when it is above $level, else None."""\n'
            "    if not $xs:\n"
            "        return None\n"
            "    i = $peak($xs)\n"
            "    if $xs[i] > $level:\n"
            "        return (i, $xs[i])\n"
            "    return None",
            _alarm,
            public=lambda p: [
                ("spike", ([3, p["level"] + 40, 7],)),
                ("quiet", ([1, 2, 1],)),
                ("empty", ([],)),
            ],
            hidden=lambda p: [
                ([p["level"] + 1],),
                ([p["level"]],),
                ([1, p["level"] + 3, p["level"] + 3],),
                ([p["level"] + 9, 2, 4],),
                ([0, 1, 2, p["level"] + 2],),
                ([],),
            ],
            uses=("peak",),
        ),
    ),
)

TEMPLATES = (
    Template(
        id="sensor1",
        family=FAMILY,
        workflow="W1",
        target="smooth",
        concept="rounded-down means of full sliding windows",
        bugs={
            "last_window": Bug(
                edits=(("range(len($xs) - $width + 1)", "range(len($xs) - $width)"),)
            ),
            "short_slice": Bug(edits=(("$xs[i:i + $width]", "$xs[i:i + $width - 1]"),)),
            "divisor": Bug(edits=((") // $width)", ") // ($width + 1))"),)),
        },
        spec=(
            "smooth(values) returns, for each start index i with i + params['width'] <= "
            "len(values), the sum of values[i:i + params['width']] floor-divided (//) by "
            "params['width'], in order of i. Fewer than params['width'] values give []."
        ),
        domain="values: list of 0..12 ints in -100..200.",
        returns="list of int",
        include=("rises", "alarm"),
    ),
    Template(
        id="sensor2",
        family=FAMILY,
        workflow="W4",
        target="clip",
        concept="limit the jump between consecutive kept values",
        bugs={"stub_raise": Bug(stub="raise"), "stub_none": Bug(stub="none")},
        spec=(
            "clip(values) returns a new list of the same length. The first value is copied. "
            "Each later value x is compared with k, the previous value of the result: if "
            "x - k > params['jump'] the result gets k + params['jump']; if k - x > "
            "params['jump'] it gets k - params['jump']; otherwise it gets x. An empty list "
            "gives []. The input list is not changed."
        ),
        domain="values: list of 0..12 ints in -200..200.",
        returns="list of int",
        include=("longest", "alarm"),
    ),
    Template(
        id="sensor3",
        family=FAMILY,
        workflow="W5",
        target="peak",
        concept="index of the first maximum",
        bugs={
            "range_end": Bug(edits=(("range(1, len($xs))", "range(1, len($xs) + 1)"),)),
            "start_len": Bug(edits=(("    best = 0\n", "    best = len($xs)\n"),)),
            "value_kept": Bug(edits=(("            best = i", "            best = $xs[i]"),)),
        },
        spec=(
            "peak(values) returns the index of the largest value of a non-empty list; when "
            "several positions hold the largest value, the smallest such index."
        ),
        domain="values: non-empty list of 1..12 ints in -100..200.",
        returns="int",
        include=("smooth",),
        caller="alarm",
    ),
)
