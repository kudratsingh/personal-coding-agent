"""Family ``dice``: scoring rolls of six-sided dice and ranking players."""

from __future__ import annotations

from typing import Any

from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template


def _face_counts(p: Params, rl: list[int], bug: Any = None) -> list[int]:
    counts = [0] * (5 if bug == "short_list" else 6)
    for face in rl:
        counts[face if bug == "no_offset" else face - 1] += 1
    return counts


def _streak(p: Params, rl: list[int], bug: Any = None) -> int:
    later = rl[2:] if bug == "skip_two" else rl[1:]
    each = 1 if bug == "unit_bonus" else p["bonus"]
    return sum(
        each for a, b in zip(rl, later, strict=False) if (a != b if bug == "unequal" else a == b)
    )


def _score(p: Params, rl: list[int], bug: Any = None) -> int:
    counts = _face_counts(p, rl, bug)
    points = sum(rl)
    if max(counts) >= 3:
        points += p["bonus"]
    return points


def _final(p: Params, rl: list[int], bug: Any = None) -> int:
    return _streak(p, rl, bug) + _score(p, rl, bug)


def _ranks(p: Params, points: list[int], bug: Any = None) -> list[int]:
    if bug == "ascending":
        order = sorted(set(points))
    elif bug == "keeps_ties":
        order = sorted(points, reverse=True)
    else:
        order = sorted(set(points), reverse=True)
    offset = 0 if bug == "from_zero" else 1
    return [order.index(x) + offset for x in points]


def _best(p: Params, players: list[Any], bug: Any = None) -> Any:
    top = None
    top_points = -1
    for name, roll in players:
        points = _score(p, roll, bug)
        if points > top_points:
            top, top_points = name, points
    return top


FAMILY = Family(
    id="dice",
    domain="dice games: face counts, roll points, streak bonuses and player ranks",
    modules={"main": ("dice", "rolls", "faces"), "play": ("scoring", "game", "points")},
    docs={"main": "Facts about a roll of six-sided dice.", "play": "Scoring rules of the game."},
    names={
        "face_counts": ("face_counts", "tally_faces", "count_faces"),
        "streak": ("streak_bonus", "run_bonus", "chain_bonus"),
        "score": ("score_roll", "roll_points", "points_for"),
        "final": ("final_points", "full_score", "round_points"),
        "ranks": ("dense_ranks", "rank_scores", "placings"),
        "best": ("best_player", "top_player", "round_leader"),
        "rl": ("roll", "throw", "dice_values"),
    },
    consts={"bonus": (10, 15, 25)},
    funcs=(
        Func(
            "face_counts",
            "main",
            "def $face_counts($rl):\n"
            '    """How many dice show each face 1 to 6, as a list of six counts."""\n'
            "    counts = [0] * 6\n"
            "    for face in $rl:\n"
            "        counts[face - 1] += 1\n"
            "    return counts",
            _face_counts,
            hidden=[
                ([],),
                ([1],),
                ([6],),
                ([6, 6, 6],),
                ([1, 2, 3, 4, 5, 6],),
                ([2, 2, 5],),
                ([3, 1, 3, 1, 3],),
            ],
        ),
        Func(
            "streak",
            "main",
            "def $streak($rl):\n"
            '    """$bonus points for each die that shows the same face as the die before it."""\n'
            "    return sum([$bonus for a, b in zip($rl, $rl[1:]) if a == b])",
            _streak,
            public=[("pairs", ([2, 2, 5, 5, 5],))],
            hidden=[
                ([],),
                ([4],),
                ([4, 4],),
                ([1, 2, 1, 2],),
                ([3, 3, 3, 3],),
                ([6, 5, 5, 6, 6],),
                ([1, 1, 2, 2, 1],),
            ],
        ),
        Func(
            "score",
            "play",
            "def $score($rl):\n"
            '    """Points of a roll: the sum of the dice, plus $bonus for three or more of a\n'
            '    kind."""\n'
            "    counts = $face_counts($rl)\n"
            "    points = sum($rl)\n"
            "    if max(counts) >= 3:\n"
            "        points += $bonus\n"
            "    return points",
            _score,
            public=[("triple_six", ([6, 2, 6, 6],)), ("plain", ([1, 2, 3],))],
            hidden=[
                ([1, 1, 1],),
                ([6, 6, 6],),
                ([5, 6, 5, 6],),
                ([2, 3, 4, 5, 6],),
                ([4, 4, 4, 4, 1],),
                ([6],),
                ([6, 5],),
            ],
            uses=("face_counts",),
        ),
        Func(
            "final",
            "play",
            "def $final($rl):\n"
            '    """Points of a roll including the streak bonus."""\n'
            "    return $streak($rl) + $score($rl)",
            _final,
            public=[("streaky", ([3, 3, 1, 4, 4],)), ("triple", ([5, 5, 5, 2],))],
            hidden=[
                ([],),
                ([1],),
                ([2, 2],),
                ([6, 1, 6, 1],),
                ([4, 4, 4, 4],),
                ([1, 2, 2, 3, 3],),
            ],
            uses=("score", "streak"),
        ),
        Func(
            "ranks",
            "play",
            "def $ranks(points):\n"
            '    """Dense rank of each score: 1 for the highest, equal scores share a rank and\n'
            '    the next lower score gets the next rank."""\n'
            "    order = sorted(set(points), reverse=True)\n"
            "    return [order.index(x) + 1 for x in points]",
            _ranks,
            public=[("ties", ([30, 50, 30, 10],)), ("distinct", ([7, 9, 8],))],
            hidden=[
                ([],),
                ([5],),
                ([5, 5],),
                ([1, 2, 3, 4],),
                ([40, 40, 20, 20, 10],),
                ([0, 100, 50, 100],),
                ([9, 3, 9, 3, 6],),
            ],
        ),
        Func(
            "best",
            "play",
            "def $best(players):\n"
            '    """Name of the player whose roll scores most; the earliest player wins ties."""\n'
            "    top = None\n"
            "    top_points = -1\n"
            "    for name, roll in players:\n"
            "        points = $score(roll)\n"
            "        if points > top_points:\n"
            "            top = name\n"
            "            top_points = points\n"
            "    return top",
            _best,
            public=[("two", ([("ann", [1, 2]), ("bo", [6, 6, 6])],))],
            hidden=[([],), ([("x", [3])],), ([("x", [2, 2]), ("y", [4])],)],
            uses=("score",),
        ),
    ),
)

TEMPLATES = (
    Template(
        id="dice1",
        family=FAMILY,
        workflow="W5",
        target="face_counts",
        concept="count dice faces 1..6",
        bugs={
            "no_offset": Bug(edits=(("counts[face - 1]", "counts[face]"),)),
            "short_list": Bug(edits=(("[0] * 6", "[0] * 5"),)),
        },
        spec=(
            "face_counts(roll) returns a list of six ints: element i is how many entries of "
            "roll equal i + 1."
        ),
        domain="roll: list of 0..8 ints, each in 1..6.",
        returns="list of 6 ints",
        include=("ranks",),
        caller="score",
    ),
    Template(
        id="dice2",
        family=FAMILY,
        workflow="W1",
        target="ranks",
        concept="dense ranking of scores",
        bugs={
            "ascending": Bug(edits=(("sorted(set(points), reverse=True)", "sorted(set(points))"),)),
            "from_zero": Bug(edits=(("order.index(x) + 1", "order.index(x)"),)),
            "keeps_ties": Bug(
                edits=(("sorted(set(points), reverse=True)", "sorted(points, reverse=True)"),)
            ),
        },
        spec=(
            "ranks(points) returns a list the same length as points: the dense rank of each "
            "score. Let distinct be the distinct values of points in decreasing order; the rank "
            "of a score is its 1-based position in distinct (so the highest score has rank 1, "
            "equal scores share a rank and there are no gaps)."
        ),
        domain="points: list of 0..10 ints in 0..200.",
        returns="list of int",
        include=("best", "streak"),
    ),
    Template(
        id="dice3",
        family=FAMILY,
        workflow="W3",
        target="streak",
        concept="bonus for each die equal to the previous one",
        bugs={
            "unequal": Bug(edits=(("if a == b", "if a != b"),)),
            "skip_two": Bug(edits=(("zip($rl, $rl[1:])", "zip($rl, $rl[2:])"),)),
            "unit_bonus": Bug(edits=(("sum([$bonus for", "sum([1 for"),)),
        },
        spec=(
            "streak(roll) returns params['bonus'] times the number of positions i >= 1 with "
            "roll[i] == roll[i - 1]."
        ),
        domain="roll: list of 0..8 ints, each in 1..6.",
        returns="int",
        caller="final",
    ),
)
