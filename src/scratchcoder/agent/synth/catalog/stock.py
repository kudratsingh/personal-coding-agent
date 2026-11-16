"""Family ``stock``: warehouse stock levels kept as item -> count dicts."""

from __future__ import annotations

from typing import Any

from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template


def _restock(p: Params, lv: dict[str, int], item: str, amount: int, bug: Any = None) -> int:
    current = lv[item] if bug == "keyerror" else lv.get(item, 0)
    cap = p["cap"] - 1 if bug == "cap_minus" else p["cap"]
    pick = max if bug == "max_for_min" else min
    return int(pick(current + amount, cap))


def _reserve(p: Params, lv: dict[str, int], item: str, amount: int, bug: Any = None) -> int:
    current = lv.get(item, 0)
    return -1 if amount > current else current - amount


def _low(p: Params, lv: dict[str, int], bug: Any = None) -> list[str]:
    return sorted(item for item, count in lv.items() if count < p["reorder"])


def _shortfall(p: Params, lv: dict[str, int], item: str, want: int, bug: Any = None) -> int:
    have = lv.get(item, 1 if bug == "default_one" else 0)
    if have >= (0 if bug == "always_zero" else want):
        return 0
    return have - want if bug == "reversed" else want - have


def _plan(p: Params, lv: dict[str, int], wants: dict[str, int], bug: Any = None) -> list[Any]:
    lines = []
    for item in sorted(wants):
        units = _shortfall(p, lv, item, wants[item], bug)
        if units > 0:
            lines.append((item, units))
    return lines


FAMILY = Family(
    id="stock",
    domain="warehouse stock levels (item -> unit count dicts): capacity, reserving, reordering",
    modules={"main": ("stock", "inventory", "depot"), "orders": ("orders", "purchasing", "supply")},
    docs={"main": "Stock levels per item.", "orders": "Reorder planning."},
    names={
        "restock": ("restock", "add_units", "receive_units"),
        "reserve": ("reserve", "take_units", "allocate"),
        "low": ("low_items", "short_items", "below_reorder"),
        "shortfall": ("shortfall", "missing_units", "deficit"),
        "plan": ("plan_orders", "order_lines", "reorder_plan"),
        "lv": ("levels", "counts", "on_hand"),
    },
    consts={"cap": (30, 40, 50, 60), "reorder": (4, 5, 8)},
    funcs=(
        Func(
            "restock",
            "main",
            "def $restock($lv, item, amount):\n"
            '    """Add amount units of item, never above the capacity $cap; return the new '
            'level."""\n'
            "    current = $lv.get(item, 0)\n"
            "    new = min(current + amount, $cap)\n"
            "    $lv[item] = new\n"
            "    return new",
            _restock,
            public=lambda p: [
                ("below_cap", ({"bolt": 3, "nut": 9}, "bolt", 4)),
                ("new_item", ({"nut": 2}, "washer", 5)),
                ("over_cap", ({"bolt": p["cap"] - 2}, "bolt", 7)),
            ],
            hidden=lambda p: [
                ({}, "pin", 1),
                ({"pin": 1}, "pin", 0),
                ({"a": p["cap"]}, "a", 1),
                ({"a": p["cap"] - 1}, "a", 1),
                ({"a": 0, "b": 5}, "b", 2),
                ({"x": 10}, "y", p["cap"] + 5),
                ({"x": 3}, "x", p["cap"]),
                ({"q": 7}, "q", 3),
            ],
        ),
        Func(
            "reserve",
            "main",
            "def $reserve($lv, item, amount):\n"
            '    """Take amount units of item and return the units left.\n\n'
            "    When fewer than amount units are in stock (an unknown item has none),\n"
            '    take nothing and return -1.\n    """\n'
            "    current = $lv.get(item, 0)\n"
            "    if amount > current:\n"
            "        return -1\n"
            "    $lv[item] = current - amount\n"
            "    return current - amount",
            _reserve,
            public=[
                ("enough", ({"gear": 9}, "gear", 4)),
                ("exact", ({"gear": 3}, "gear", 3)),
                ("too_few", ({"gear": 2, "cog": 8}, "gear", 5)),
                ("unknown", ({"cog": 1}, "belt", 1)),
            ],
            hidden=[
                ({}, "a", 0),
                ({}, "a", 1),
                ({"a": 5}, "a", 5),
                ({"a": 5}, "a", 6),
                ({"a": 5, "b": 1}, "b", 1),
                ({"a": 10}, "a", 0),
                ({"a": 7}, "a", 2),
                ({"z": 4}, "y", 2),
            ],
        ),
        Func(
            "low",
            "main",
            "def $low($lv):\n"
            '    """Items below the reorder point $reorder, sorted by name."""\n'
            "    return sorted([item for item, count in $lv.items() if count < $reorder])",
            _low,
            public=lambda p: [("mixed", ({"a": 1, "c": p["reorder"], "b": 20},))],
            hidden=lambda p: [
                ({},),
                ({"z": 0, "y": 0},),
                ({"m": p["reorder"] - 1, "n": p["reorder"]},),
            ],
        ),
        Func(
            "shortfall",
            "main",
            "def $shortfall($lv, item, want):\n"
            '    """Units still needed for item to reach want (0 when there are enough)."""\n'
            "    have = $lv.get(item, 0)\n"
            "    if have >= want:\n"
            "        return 0\n"
            "    return want - have",
            _shortfall,
            public=[("short", ({"rod": 2}, "rod", 5))],
            hidden=[
                ({"rod": 2}, "rod", 5),
                ({}, "rod", 3),
                ({"rod": 9}, "rod", 4),
                ({"rod": 4}, "rod", 4),
                ({"a": 0}, "a", 1),
                ({"a": 1, "b": 2}, "b", 7),
                ({"a": 3}, "c", 0),
            ],
        ),
        Func(
            "plan",
            "orders",
            "def $plan($lv, wants):\n"
            '    """Order lines (item, units) for every wanted item that is short, by item '
            'name."""\n'
            "    lines = []\n"
            "    for item in sorted(wants):\n"
            "        units = $shortfall($lv, item, wants[item])\n"
            "        if units > 0:\n"
            "            lines.append((item, units))\n"
            "    return lines",
            _plan,
            public=[
                ("short", ({"rod": 2, "cap": 9}, {"rod": 5, "cap": 4})),
                ("missing", ({"rod": 1}, {"tube": 3})),
                ("enough", ({"rod": 6}, {"rod": 6})),
            ],
            hidden=[
                ({}, {}),
                ({}, {"a": 2, "b": 1}),
                ({"a": 5}, {"a": 2}),
                ({"a": 1, "b": 1}, {"b": 4, "a": 3}),
                ({"x": 2}, {"x": 2, "y": 1}),
                ({"k": 3}, {"k": 10}),
            ],
            uses=("shortfall",),
        ),
    ),
)

TEMPLATES = (
    Template(
        id="stock1",
        family=FAMILY,
        workflow="W1",
        target="restock",
        concept="capacity-capped restock",
        bugs={
            "cap_minus": Bug(
                edits=(("min(current + amount, $cap)", "min(current + amount, $cap - 1)"),)
            ),
            "keyerror": Bug(edits=(("$lv.get(item, 0)", "$lv[item]"),)),
            "max_for_min": Bug(edits=(("min(", "max("),)),
        },
        spec=(
            "restock(levels, item, amount) adds stock. Let current be levels[item] when item is "
            "a key of levels, otherwise 0. The new level is min(current + amount, params['cap']). "
            "The function stores the new level in levels[item] and returns it; when item is "
            "absent the key is added, even when amount is 0."
        ),
        domain=(
            "levels: dict mapping short lowercase item names (str) to ints 0..params['cap']; "
            "item: str (may be absent from levels); amount: int >= 0."
        ),
        returns="int",
        include=("low", "plan"),
    ),
    Template(
        id="stock2",
        family=FAMILY,
        workflow="W4",
        target="reserve",
        concept="reserve units only when enough are in stock",
        bugs={"stub_raise": Bug(stub="raise"), "stub_none": Bug(stub="none")},
        spec=(
            "reserve(levels, item, amount) takes units out of stock. Let current be levels[item] "
            "when item is a key of levels, otherwise 0. If amount > current it leaves levels "
            "unchanged (no key is added) and returns -1. Otherwise it sets levels[item] = "
            "current - amount and returns current - amount; when item is absent (so amount is "
            "0) the key is added with the value 0."
        ),
        domain=(
            "levels: dict mapping short lowercase item names (str) to ints >= 0; item: str (may "
            "be absent); amount: int >= 0."
        ),
        returns="int",
        include=("restock", "low", "plan"),
    ),
    Template(
        id="stock3",
        family=FAMILY,
        workflow="W3",
        target="shortfall",
        concept="shortfall to a wanted level",
        bugs={
            "reversed": Bug(edits=(("return want - have", "return have - want"),)),
            "default_one": Bug(edits=(("$lv.get(item, 0)", "$lv.get(item, 1)"),)),
            "always_zero": Bug(edits=(("if have >= want:", "if have >= 0:"),)),
        },
        spec=(
            "shortfall(levels, item, want) returns how many units are still needed. Let have be "
            "levels[item] when item is a key of levels, otherwise 0. Return 0 when have >= want, "
            "otherwise want - have. It does not change levels."
        ),
        domain=(
            "levels: dict mapping short lowercase item names (str) to ints >= 0; item: str (may "
            "be absent); want: int >= 0."
        ),
        returns="int",
        include=("restock",),
        caller="plan",
    ),
)
