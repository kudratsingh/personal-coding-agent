"""Family ``graph``: directed graphs as successor-list dicts."""

from __future__ import annotations

from collections import deque
from typing import Any

from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template

Adj = dict[str, list[str]]


def _adjacency(p: Params, edges: list[Any], bug: Any = None) -> Adj:
    out: dict[str, list[str]] = {}
    for a, b in edges:
        if bug == "reversed":
            out.setdefault(b, []).append(a)
            out.setdefault(a, [])
        elif bug == "overwrite":
            out[a] = [b]
            out.setdefault(b, [])
        else:
            out.setdefault(a, []).append(b)
            out.setdefault(b, [])
    return {node: sorted(succ) for node, succ in out.items()}


def _in_degree(p: Params, gr: Adj, bug: Any = None) -> dict[str, int]:
    counts: dict[str, int] = {} if bug == "no_init" else {node: 0 for node in gr}
    for node in gr:
        succ = gr[node][1:] if bug == "skip_first" else gr[node]
        for nxt in succ:
            key = node if bug == "counts_source" else nxt
            counts[key] = counts.get(key, 0) + 1
    return counts


def _reach(p: Params, gr: Adj, start: str, bug: Any = None) -> list[str]:
    seen: Any = start if bug == "seen_str" else {start}
    todo = [start]
    while todo:
        node = todo.pop(1) if bug == "pop_one" else todo.pop()
        for nxt in gr[node] if bug == "no_get" else gr.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return sorted(seen)


def _hops(p: Params, gr: Adj, start: str, goal: str, bug: Any = None) -> int:
    dist = {start: 0}
    todo = deque([start])
    while todo:
        node = todo.popleft()
        if node == goal:
            return dist[node]
        for nxt in gr.get(node, []):
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                todo.append(nxt)
    return -1


def _sinks(p: Params, gr: Adj, bug: Any = None) -> list[str]:
    return sorted(node for node, succ in gr.items() if not succ)


def _trip(p: Params, edges: list[Any], start: str, goal: str, bug: Any = None) -> int:
    return _hops(p, _adjacency(p, edges, bug), start, goal, bug)


def _spread(p: Params, gr: Adj, start: str, bug: Any = None) -> int:
    return len(_reach(p, gr, start, bug)) - 1


G1 = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
G2 = {"s": ["t"], "t": ["u"], "u": ["s"], "v": ["s"]}
G3 = {"p": [], "q": ["p"], "r": ["p", "q"]}
G4 = {"x": ["y"], "y": []}

FAMILY = Family(
    id="graph",
    domain="directed graphs as dicts of successor lists: degrees, reachability, hop counts",
    modules={"main": ("graph", "network", "links"), "paths": ("paths", "routes", "travel")},
    docs={"main": "Directed graphs stored as successor lists.", "paths": "Questions about paths."},
    names={
        "adjacency": ("adjacency", "build_links", "edge_map"),
        "in_degree": ("in_degrees", "fan_in", "incoming"),
        "reach": ("reachable", "reach_from", "visit_all"),
        "hops": ("hop_count", "hops_between", "fewest_edges"),
        "sinks": ("sinks", "dead_ends", "terminal_nodes"),
        "trip": ("trip_length", "route_hops", "edge_distance"),
        "spread": ("spread_of", "reach_count", "downstream"),
        "gr": ("adj", "succ_map", "outgoing"),
    },
    consts={},
    funcs=(
        Func(
            "adjacency",
            "main",
            "def $adjacency(edges):\n"
            '    """Sorted successor lists from (source, target) edges; every node is a key."""\n'
            "    out = {}\n"
            "    for a, b in edges:\n"
            "        out.setdefault(a, []).append(b)\n"
            "        out.setdefault(b, [])\n"
            "    return {node: sorted(succ) for node, succ in out.items()}",
            _adjacency,
            hidden=[
                ([],),
                ([("a", "b")],),
                ([("a", "b"), ("a", "c"), ("c", "b")],),
                ([("x", "x")],),
                ([("b", "a"), ("a", "b")],),
                ([("m", "n"), ("m", "o"), ("o", "p"), ("n", "p")],),
            ],
        ),
        Func(
            "in_degree",
            "main",
            "def $in_degree($gr):\n"
            '    """How many edges enter each node (every key of the graph is counted)."""\n'
            "    counts = {node: 0 for node in $gr}\n"
            "    for node in $gr:\n"
            "        for nxt in $gr[node]:\n"
            "            counts[nxt] = counts.get(nxt, 0) + 1\n"
            "    return counts",
            _in_degree,
            public=[("diamond", (G1,)), ("cycle", (G2,)), ("chain", (G4,))],
            hidden=[
                ({},),
                (G3,),
                ({"a": ["a"]},),
                ({"a": ["b", "c"], "b": ["c"], "c": []},),
                ({"k": ["m"]},),
                ({"u": ["v", "w"], "v": ["w"], "w": ["u"]},),
            ],
        ),
        Func(
            "reach",
            "main",
            "def $reach($gr, start):\n"
            '    """Sorted nodes reachable from start, start included."""\n'
            "    seen = {start}\n"
            "    todo = [start]\n"
            "    while todo:\n"
            "        node = todo.pop()\n"
            "        for nxt in $gr.get(node, []):\n"
            "            if nxt not in seen:\n"
            "                seen.add(nxt)\n"
            "                todo.append(nxt)\n"
            "    return sorted(seen)",
            _reach,
            hidden=[
                (G1, "a"),
                (G1, "d"),
                (G2, "s"),
                (G2, "v"),
                (G3, "r"),
                ({}, "z"),
                ({"a": ["b"]}, "a"),
                ({"a": ["b"], "b": ["c"]}, "b"),
            ],
        ),
        Func(
            "hops",
            "main",
            "def $hops($gr, start, goal):\n"
            '    """Fewest edges from start to goal, or -1 when goal cannot be reached."""\n'
            "    dist = {start: 0}\n"
            "    todo = deque([start])\n"
            "    while todo:\n"
            "        node = todo.popleft()\n"
            "        if node == goal:\n"
            "            return dist[node]\n"
            "        for nxt in $gr.get(node, []):\n"
            "            if nxt not in dist:\n"
            "                dist[nxt] = dist[node] + 1\n"
            "                todo.append(nxt)\n"
            "    return -1",
            _hops,
            public=[("two", (G1, "a", "d"))],
            hidden=[(G1, "a", "a"), (G1, "d", "a"), (G2, "v", "u"), (G3, "r", "p")],
            imports=("from collections import deque",),
        ),
        Func(
            "sinks",
            "main",
            "def $sinks($gr):\n"
            '    """Sorted nodes with no outgoing edges."""\n'
            "    return sorted([node for node, succ in $gr.items() if not succ])",
            _sinks,
            public=[("diamond", (G1,))],
            hidden=[({},), (G2,), (G3,)],
        ),
        Func(
            "trip",
            "paths",
            "def $trip(edges, start, goal):\n"
            '    """Fewest edges between two nodes of an edge list, or -1."""\n'
            "    links = $adjacency(edges)\n"
            "    return $hops(links, start, goal)",
            _trip,
            public=[
                ("forward", ([("a", "b"), ("b", "c")], "a", "c")),
                ("branch", ([("a", "b"), ("a", "c"), ("b", "d")], "a", "d")),
                ("backward", ([("a", "b")], "b", "a")),
            ],
            hidden=[
                ([], "a", "a"),
                ([("a", "b")], "a", "b"),
                ([("x", "y"), ("y", "z"), ("x", "z")], "x", "z"),
                ([("q", "r"), ("q", "s"), ("s", "t"), ("r", "t")], "q", "t"),
                ([("b", "a"), ("c", "b")], "c", "a"),
                ([("m", "n")], "n", "m"),
            ],
            uses=("hops", "adjacency"),
        ),
        Func(
            "spread",
            "paths",
            "def $spread($gr, start):\n"
            '    """How many other nodes can be reached from start."""\n'
            "    return len($reach($gr, start)) - 1",
            _spread,
            public=[
                ("chain", ({"a": ["b"], "b": ["c"]}, "a")),
                ("branch", ({"r": ["s", "t"], "s": [], "t": []}, "r")),
            ],
            hidden=[
                ({"a": ["b"], "b": ["c"]}, "a"),
                ({"a": ["b", "c"]}, "a"),
                (G1, "a"),
                (G2, "u"),
                ({}, "z"),
                (G3, "p"),
            ],
            uses=("reach",),
        ),
    ),
)

TEMPLATES = (
    Template(
        id="graph1",
        family=FAMILY,
        workflow="W1",
        target="in_degree",
        concept="in-degree of every node",
        bugs={
            "no_init": Bug(edits=(("counts = {node: 0 for node in $gr}", "counts = {}"),)),
            "counts_source": Bug(
                edits=(
                    (
                        "counts[nxt] = counts.get(nxt, 0) + 1",
                        "counts[node] = counts.get(node, 0) + 1",
                    ),
                )
            ),
            "skip_first": Bug(edits=(("for nxt in $gr[node]:", "for nxt in $gr[node][1:]:"),)),
        },
        spec=(
            "in_degree(graph) maps every key of graph, and every node that appears in a "
            "successor list, to the number of times it appears in the successor lists of all "
            "keys (0 when it appears in none)."
        ),
        domain=(
            "graph: dict mapping short str node names to lists of str node names (0..6 keys); "
            "every listed node is also a key."
        ),
        returns="dict str -> int",
        include=("sinks", "trip"),
    ),
    Template(
        id="graph2",
        family=FAMILY,
        workflow="W3",
        target="adjacency",
        concept="successor lists from an edge list",
        bugs={
            "reversed": Bug(
                edits=(
                    ("out.setdefault(a, []).append(b)", "out.setdefault(b, []).append(a)"),
                    ("out.setdefault(b, [])\n", "out.setdefault(a, [])\n"),
                )
            ),
            "overwrite": Bug(edits=(("out.setdefault(a, []).append(b)", "out[a] = [b]"),)),
        },
        spec=(
            "adjacency(edges) builds successor lists. edges is a list of (source, target) pairs "
            "of str nodes. The result is a dict whose keys are every node that occurs in any "
            "edge; each value is the sorted list of targets of the edges leaving that key "
            "(with repeats if an edge repeats; [] when none leave it)."
        ),
        domain="edges: list of 0..6 (str, str) tuples of short lowercase node names.",
        returns="dict str -> list of str",
        include=("in_degree",),
        caller="trip",
    ),
    Template(
        id="graph3",
        family=FAMILY,
        workflow="W5",
        target="reach",
        concept="nodes reachable from a start node",
        bugs={
            "no_get": Bug(edits=(("$gr.get(node, [])", "$gr[node]"),)),
            "pop_one": Bug(edits=(("todo.pop()", "todo.pop(1)"),)),
            "seen_str": Bug(edits=(("seen = {start}", "seen = start"),)),
        },
        spec=(
            "reach(graph, start) returns the sorted list of nodes reachable from start by "
            "following successor lists any number of times, start included. A node that is "
            "not a key of graph has no successors."
        ),
        domain=(
            "graph: dict mapping short str node names to lists of str node names (0..6 keys; "
            "listed nodes need not be keys); start: str."
        ),
        returns="list of str",
        include=("in_degree",),
        caller="spread",
    ),
)
