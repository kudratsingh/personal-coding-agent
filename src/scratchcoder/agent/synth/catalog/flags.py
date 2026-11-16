"""Family ``flags``: permission flags packed into integer bit masks."""

from __future__ import annotations

from typing import Any

from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template


def _mask_of(p: Params, names: list[str], bug: Any = None) -> int:
    mask = 0
    for i, flag in enumerate(p["order"]):
        present = flag not in names if bug == "inverted" else flag in names
        if present:
            bit = 1 << (i + 1 if bug == "shift_one" else i)
            mask = bit if bug == "keeps_last" else mask | bit
    return mask


def _names_of(p: Params, mask: int, bug: Any = None) -> list[str]:
    order = list(reversed(p["order"])) if bug == "reversed" else p["order"]
    shift = 1 if bug == "shift_one_more" else 0
    probe = 2 if bug == "wrong_bit" else 1
    return [flag for i, flag in enumerate(order) if mask >> (i + shift) & probe]


def _bits_set(p: Params, mask: int, bug: Any = None) -> int:
    return bin(mask).count("1")


def _has_all(p: Params, mask: int, need: int, bug: Any = None) -> bool:
    return mask & need == need


def _grant(p: Params, mask: int, names: list[str], bug: Any = None) -> int:
    return mask | _mask_of(p, names, bug)


def _can(p: Params, mask: int, names: list[str], bug: Any = None) -> bool:
    return _has_all(p, mask, _mask_of(p, names, bug), bug)


ORDERS = (
    ("read", "write", "run"),
    ("view", "edit", "share", "admin"),
    ("open", "close", "lock", "seal"),
)

FAMILY = Family(
    id="flags",
    domain="permission flags as integer bit masks: masks from names, names from masks, checks",
    modules={"main": ("flags", "perms", "access"), "roles": ("roles", "grants", "policies")},
    docs={"main": "Permission flags stored as bits of an int.", "roles": "Role permission checks."},
    names={
        "mask_of": ("mask_of", "to_mask", "flags_mask"),
        "names_of": ("flag_names", "from_mask", "names_in"),
        "bits_set": ("bit_count", "count_bits", "ones_in"),
        "has_all": ("has_all", "allows_all", "covers"),
        "grant": ("grant", "add_flags", "with_flags"),
        "can": ("can_do", "permits", "allowed_to"),
    },
    consts={"order": ORDERS},
    funcs=(
        Func(
            "mask_of",
            "main",
            "def $mask_of(names):\n"
            '    """Bit mask of the named flags: bit i stands for flag i of $order."""\n'
            "    mask = 0\n"
            "    for i, flag in enumerate($order):\n"
            "        if flag in names:\n"
            "            mask |= 1 << i\n"
            "    return mask",
            _mask_of,
            hidden=lambda p: [
                ([],),
                ([p["order"][0]],),
                ([p["order"][-1]],),
                (list(p["order"]),),
                ([p["order"][1], p["order"][0]],),
                (["unknown"],),
                ([p["order"][2], "unknown"],),
            ],
        ),
        Func(
            "names_of",
            "main",
            "def $names_of(mask):\n"
            '    """Names of the flags whose bits are set in mask, in the order of $order."""\n'
            "    return [flag for i, flag in enumerate($order) if mask >> i & 1]",
            _names_of,
            public=[("first", (1,)), ("two", (6,)), ("none", (0,))],
            hidden=lambda p: [
                (2,),
                (3,),
                (5,),
                (4,),
                ((1 << len(p["order"])) - 1,),
                (1 << (len(p["order"]) - 1),),
                (1 << len(p["order"]),),
            ],
        ),
        Func(
            "bits_set",
            "main",
            "def $bits_set(mask):\n"
            '    """Number of set bits of a non-negative int."""\n'
            "    count = 0\n"
            "    while mask:\n"
            "        count += mask & 1\n"
            "        mask >>= 1\n"
            "    return count",
            _bits_set,
            public=[("seven", (7,)), ("sparse", (10,)), ("zero", (0,))],
            hidden=[(1,), (2,), (255,), (256,), (1023,), (12345,), (2**20 + 1,)],
        ),
        Func(
            "has_all",
            "main",
            "def $has_all(mask, need):\n"
            '    """True when every bit set in need is also set in mask."""\n'
            "    return mask & need == need",
            _has_all,
            public=[("subset", (7, 5))],
            hidden=[(0, 0), (4, 6), (6, 2)],
        ),
        Func(
            "grant",
            "roles",
            "def $grant(mask, names):\n"
            '    """mask with the named flags added."""\n'
            "    return mask | $mask_of(names)",
            _grant,
            public=lambda p: [
                ("add_two", (0, [p["order"][0], p["order"][2]])),
                ("already", (1, [p["order"][0]])),
            ],
            hidden=lambda p: [
                (0, []),
                (2, [p["order"][0]]),
                (0, list(p["order"])),
                (4, [p["order"][1], "unknown"]),
                (1, [p["order"][-1]]),
                (0, [p["order"][1]]),
            ],
            uses=("mask_of",),
        ),
        Func(
            "can",
            "roles",
            "def $can(mask, names):\n"
            '    """True when mask holds every named flag."""\n'
            "    return $has_all(mask, $mask_of(names))",
            _can,
            public=lambda p: [("yes", (3, [p["order"][1]])), ("no", (1, [p["order"][1]]))],
            hidden=lambda p: [
                (0, []),
                (7, [p["order"][0], p["order"][2]]),
                (4, [p["order"][0]]),
                (2, [p["order"][1]]),
                (5, [p["order"][2]]),
                (1, [p["order"][0], p["order"][1]]),
            ],
            uses=("has_all", "mask_of"),
        ),
    ),
)

TEMPLATES = (
    Template(
        id="flags1",
        family=FAMILY,
        workflow="W1",
        target="names_of",
        concept="flag names from a bit mask",
        bugs={
            "shift_one_more": Bug(edits=(("mask >> i & 1", "mask >> i + 1 & 1"),)),
            "wrong_bit": Bug(edits=(("mask >> i & 1", "mask >> i & 2"),)),
            "reversed": Bug(edits=(("enumerate($order)", "enumerate(reversed($order))"),)),
        },
        spec=(
            "names_of(mask) returns, in the order of params['order'] (a sequence of 3 or 4 "
            "distinct str; a list in the instance params, a tuple constant in the code), the "
            "names params['order'][i] whose bit i (value 1 << i) is set in the non-negative int "
            "mask. Set bits at positions i >= len(params['order']) are ignored."
        ),
        domain="mask: int in 0..63.",
        returns="list of str",
        include=("bits_set", "can"),
    ),
    Template(
        id="flags2",
        family=FAMILY,
        workflow="W3",
        target="mask_of",
        concept="bit mask from flag names",
        bugs={
            "shift_one": Bug(edits=(("1 << i", "1 << i + 1"),)),
            "keeps_last": Bug(edits=(("mask |= 1 << i", "mask = 1 << i"),)),
            "inverted": Bug(edits=(("if flag in names:", "if flag not in names:"),)),
        },
        spec=(
            "mask_of(names) takes a list of str and returns the int whose bit i (value 1 << i) "
            "is set exactly when params['order'][i] is in names; names not in params['order'] "
            "are ignored. Names match by exact equality only (a prefix or plural of a flag "
            "name does not match), and a repeated name does not change the result."
        ),
        domain=(
            "names: list of 0..4 str, each one of params['order'] or another short lowercase word."
        ),
        returns="int",
        include=("bits_set",),
        caller="grant",
    ),
    Template(
        id="flags3",
        family=FAMILY,
        workflow="W4",
        target="bits_set",
        concept="population count of a mask",
        bugs={"stub_raise": Bug(stub="raise"), "stub_none": Bug(stub="none")},
        spec=(
            "bits_set(mask) returns the number of 1 bits in the binary form of the "
            "non-negative int mask."
        ),
        domain="mask: int in 0..2**24.",
        returns="int",
        include=("names_of", "can"),
    ),
)
