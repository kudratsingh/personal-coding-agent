"""Family ``votes``: ballot tallies, winners, approval and elimination rounds."""

from __future__ import annotations

from typing import Any

from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template


def _tally(p: Params, bl: list[str], bug: Any = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in bl:
        if bug == "set_one":
            counts[name] = 1
        else:
            counts[name] = counts.get(name, 1 if bug == "start_one" else 0) + 1
    return counts


def _winner(p: Params, bl: list[str], bug: Any = None) -> Any:
    counts = _tally(p, bl, bug)
    best = None
    for name in counts if bug == "unsorted" else sorted(counts):
        if best is None:
            best = name
        elif bug == "fewest":
            if counts[name] < counts[best]:
                best = name
        elif counts[name] >= counts[best] if bug == "last_tie" else counts[name] > counts[best]:
            best = name
    return best


def _approve(p: Params, yes: int, total: int, bug: Any = None) -> bool:
    if total == 0:
        return False
    return bool(yes * 100 >= p["quota"] * total)


def _eliminate(p: Params, ranked: list[list[str]], bug: Any = None) -> list[list[str]]:
    firsts = _tally(p, [b[0] for b in ranked if b], bug)
    weakest = min(sorted(firsts, reverse=True), key=lambda name: firsts[name])
    return [[name for name in b if name != weakest] for b in ranked]


def _share(p: Params, bl: list[str], bug: Any = None) -> Any:
    if not bl:
        return None
    top = _winner(p, bl, bug)
    return (top, _tally(p, bl, bug)[top] * 100 // len(bl))


FAMILY = Family(
    id="votes",
    domain="ballots and tallies: plurality winners, tie-breaks, approval quotas, elimination",
    modules={"main": ("votes", "ballots", "poll"), "count": ("counting", "results", "outcome")},
    docs={"main": "Ballot tallies.", "count": "Election results."},
    names={
        "tally": ("tally", "count_votes", "vote_counts"),
        "winner": ("winner", "plurality", "top_choice"),
        "approve": ("approved", "passes_quota", "accepted"),
        "eliminate": ("drop_weakest", "eliminate", "remove_last"),
        "share": ("leader_share", "winner_share", "top_share"),
        "bl": ("ballots", "choices", "picks"),
    },
    consts={"quota": (50, 60, 75)},
    funcs=(
        Func(
            "tally",
            "main",
            "def $tally($bl):\n"
            '    """Votes per candidate from a list of first choices."""\n'
            "    counts = {}\n"
            "    for name in $bl:\n"
            "        counts[name] = counts.get(name, 0) + 1\n"
            "    return counts",
            _tally,
            hidden=[
                ([],),
                (["ann"],),
                (["ann", "ann"],),
                (["ann", "bob", "ann"],),
                (["c", "b", "a", "b"],),
                (["x", "y", "z"],),
            ],
        ),
        Func(
            "winner",
            "main",
            "def $winner($bl):\n"
            '    """The candidate with the most votes; ties go to the name that sorts first;\n'
            '    None when there are no votes."""\n'
            "    counts = $tally($bl)\n"
            "    best = None\n"
            "    for name in sorted(counts):\n"
            "        if best is None or counts[name] > counts[best]:\n"
            "            best = name\n"
            "    return best",
            _winner,
            public=[
                ("clear", (["kim", "lee", "kim"],)),
                ("tie", (["zed", "amy", "zed", "amy"],)),
                ("none", ([],)),
            ],
            hidden=[
                (["a"],),
                (["b", "a"],),
                (["c", "c", "b", "b", "a"],),
                (["m", "n", "n"],),
                (["q", "p", "q", "p", "r", "r"],),
                (["y", "x", "y"],),
                (["z", "y", "x"],),
            ],
            uses=("tally",),
        ),
        Func(
            "approve",
            "main",
            "def $approve(yes, total):\n"
            '    """True when yes votes are at least $quota percent of total (False for no\n'
            '    votes)."""\n'
            "    if total == 0:\n"
            "        return False\n"
            "    return yes * 100 >= $quota * total",
            _approve,
            public=lambda p: [("exact", (p["quota"], 100))],
            hidden=lambda p: [(0, 0), (p["quota"] - 1, 100), (1, 1)],
        ),
        Func(
            "eliminate",
            "main",
            "def $eliminate(ranked):\n"
            '    """Ballots with the weakest candidate removed.\n\n'
            "    Each ballot is a list of names, most preferred first. Among the names that are\n"
            "    first on at least one ballot, the weakest has the fewest first places; on a tie\n"
            "    it is the name that sorts last. It is removed from every ballot.\n"
            '    """\n'
            "    firsts = $tally([b[0] for b in ranked if b])\n"
            "    weakest = min(sorted(firsts, reverse=True), key=lambda name: firsts[name])\n"
            "    return [[name for name in b if name != weakest] for b in ranked]",
            _eliminate,
            public=[
                ("clear", ([["a", "b"], ["a", "c"], ["b", "a"]],)),
                ("tie", ([["x", "y"], ["y", "x"]],)),
                ("three", ([["p", "q", "r"], ["q", "r"], ["q", "p"], ["r", "p"]],)),
            ],
            hidden=[
                ([["a"]],),
                ([["a"], ["b"], ["b"]],),
                ([["c", "a"], ["b", "a"], ["a", "c"]],),
                ([["m", "n"], [], ["n", "m"], ["n"]],),
                ([["z", "y", "x"], ["y", "z"], ["x", "z"], ["x"]],),
                ([["k", "j"], ["j", "k"], ["k"], ["j"]],),
            ],
            uses=("tally",),
        ),
        Func(
            "share",
            "count",
            "def $share($bl):\n"
            '    """(winner, percent of the votes the winner got, rounded down), or None\n'
            '    without votes."""\n'
            "    if not $bl:\n"
            "        return None\n"
            "    top = $winner($bl)\n"
            "    return (top, $tally($bl)[top] * 100 // len($bl))",
            _share,
            public=[
                ("majority", (["ivy", "joe", "ivy"],)),
                ("split", (["ivy", "joe", "kai", "joe"],)),
                ("empty", ([],)),
            ],
            hidden=[
                (["a"],),
                (["a", "b"],),
                (["b", "b", "a"],),
                (["c", "a", "b", "c", "a", "c"],),
                (["x", "y", "z", "x", "y", "x", "w"],),
                ([],),
            ],
            uses=("winner", "tally"),
        ),
    ),
)

TEMPLATES = (
    Template(
        id="votes1",
        family=FAMILY,
        workflow="W1",
        target="winner",
        concept="plurality winner with an alphabetical tie-break",
        bugs={
            "last_tie": Bug(
                edits=(("counts[name] > counts[best]", "counts[name] >= counts[best]"),)
            ),
            "unsorted": Bug(edits=(("for name in sorted(counts):", "for name in counts:"),)),
            "fewest": Bug(edits=(("counts[name] > counts[best]", "counts[name] < counts[best]"),)),
        },
        spec=(
            "winner(ballots) takes a list of candidate names, one per vote. Return None for an "
            "empty list. Otherwise return the name with the most votes; when several names "
            "share the most votes, the one that is smallest in Python str order."
        ),
        domain="ballots: list of 0..8 short lowercase str names.",
        returns="str or None",
        include=("approve", "share"),
    ),
    Template(
        id="votes2",
        family=FAMILY,
        workflow="W4",
        target="eliminate",
        concept="remove the weakest candidate from ranked ballots",
        bugs={"stub_raise": Bug(stub="raise"), "stub_none": Bug(stub="none")},
        spec=(
            "eliminate(ranked) takes a list of ballots, each a list of str names (most "
            "preferred first, possibly empty), with at least one non-empty ballot. Count, for "
            "each name that is the first element of at least one ballot, how many ballots it is "
            "first on; a name that is never first is not counted and cannot be the weakest. "
            "The weakest is the counted name with the smallest count; among names with that "
            "count, the largest in Python str order. Return a new list of new ballot lists, in "
            "the same order, each with every occurrence of the weakest name removed; a ballot "
            "that becomes empty stays as []. The input is not changed."
        ),
        domain=(
            "ranked: list of 1..6 lists of 0..3 distinct short lowercase names; at least one "
            "non-empty."
        ),
        returns="list of lists of str",
        include=("approve", "share"),
    ),
    Template(
        id="votes3",
        family=FAMILY,
        workflow="W3",
        target="tally",
        concept="vote count per candidate",
        bugs={
            "start_one": Bug(edits=(("counts.get(name, 0) + 1", "counts.get(name, 1) + 1"),)),
            "set_one": Bug(edits=(("counts[name] = counts.get(name, 0) + 1", "counts[name] = 1"),)),
        },
        spec=(
            "tally(ballots) takes a list of names, one per vote, and returns a dict mapping each "
            "name that occurs to the number of times it occurs."
        ),
        domain="ballots: list of 0..8 short lowercase str names.",
        returns="dict str -> int",
        caller="share",
    ),
)
