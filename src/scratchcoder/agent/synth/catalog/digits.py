"""Family ``digits``: arithmetic on the digits of non-negative integers."""

from __future__ import annotations

from typing import Any

from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template


def _dsum(p: Params, n: int, bug: Any = None) -> int:
    total = 1 if bug == "start_one" else 0
    modulus = 9 if bug == "mod_nine" else 10
    while n > (9 if bug == "drop_lead" else 0):
        total += n % modulus
        n //= 10
    return total


def _droot(p: Params, n: int, bug: Any = None) -> int:
    while n >= 10:
        n = _dsum(p, n, bug)
    return n


def _to_base(p: Params, n: int, bug: Any = None) -> list[int]:
    base = p["base"]
    if n == 0 and bug != "no_zero":
        return [0]
    out = []
    while n > 0:
        out.append(n % base)
        n //= 10 if bug == "div_ten" else base
    return out if bug == "not_reversed" else out[::-1]


def _same_both_ways(p: Params, n: int, bug: Any = None) -> bool:
    digs = _to_base(p, n, bug)
    return digs == digs[::-1]


def _luhn(p: Params, number: int, bug: Any = None) -> bool:
    total = 0
    for i, ch in enumerate(reversed(str(number))):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _persist(p: Params, n: int, bug: Any = None) -> int:
    steps = 0
    while n >= 10:
        prod = 1
        for ch in str(n):
            prod *= int(ch)
        n = prod
        steps += 1
    return steps


FAMILY = Family(
    id="digits",
    domain="digits of non-negative integers: digit sums, roots, bases, checksums",
    modules={"main": ("digits", "numeric", "figures"), "checks": ("checks", "verify", "validity")},
    docs={"main": "Digit arithmetic on non-negative integers.", "checks": "Number checks."},
    names={
        "dsum": ("digit_sum", "sum_digits", "add_digits"),
        "droot": ("digital_root", "repeated_sum", "root_digit"),
        "to_base": ("to_base", "base_digits", "in_base"),
        "same": ("same_both_ways", "base_mirror", "symmetric_in_base"),
        "luhn": ("luhn_ok", "check_digit_ok", "luhn_valid"),
        "persist": ("persistence", "multiply_steps", "product_steps"),
    },
    consts={"base": (2, 3, 5, 7, 8)},
    funcs=(
        Func(
            "dsum",
            "main",
            "def $dsum(n):\n"
            '    """Sum of the decimal digits of a non-negative integer."""\n'
            "    total = 0\n"
            "    while n > 0:\n"
            "        total += n % 10\n"
            "        n //= 10\n"
            "    return total",
            _dsum,
            hidden=[(0,), (7,), (10,), (99,), (123,), (9081,), (500005,)],
        ),
        Func(
            "to_base",
            "main",
            "def $to_base(n):\n"
            '    """Digits of a non-negative integer in base $base, most significant first."""\n'
            "    if n == 0:\n"
            "        return [0]\n"
            "    out = []\n"
            "    while n > 0:\n"
            "        out.append(n % $base)\n"
            "        n //= $base\n"
            "    return out[::-1]",
            _to_base,
            public=lambda p: [
                ("zero", (0,)),
                ("small", (p["base"] + 1,)),
                ("larger", (p["base"] ** 3 + 2 * p["base"],)),
            ],
            hidden=lambda p: [
                (1,),
                (p["base"] - 1,),
                (p["base"],),
                (p["base"] ** 2,),
                (100,),
                (255,),
                (1000,),
                (0,),
            ],
        ),
        Func(
            "persist",
            "main",
            "def $persist(n):\n"
            '    """How many times the digits of n must be multiplied together to reach one\n'
            '    digit."""\n'
            "    steps = 0\n"
            "    while n >= 10:\n"
            "        prod = 1\n"
            "        for ch in str(n):\n"
            "            prod *= int(ch)\n"
            "        n = prod\n"
            "        steps += 1\n"
            "    return steps",
            _persist,
            public=[("three", (39,))],
            hidden=[(4,), (25,), (999,)],
        ),
        Func(
            "droot",
            "checks",
            "def $droot(n):\n"
            '    """Digit sum repeated until one digit remains."""\n'
            "    while n >= 10:\n"
            "        n = $dsum(n)\n"
            "    return n",
            _droot,
            public=[("two_steps", (987,)), ("one_step", (34,)), ("single", (5,))],
            hidden=[(0,), (9,), (10,), (19,), (99999,), (123456,), (1000000,)],
            uses=("dsum",),
        ),
        Func(
            "same",
            "checks",
            "def $same(n):\n"
            '    """True when the base $base digits of n read the same both ways."""\n'
            "    digs = $to_base(n)\n"
            "    return digs == digs[::-1]",
            _same_both_ways,
            public=lambda p: [("mirror", (p["base"] ** 2 + 1,))],
            hidden=lambda p: [(0,), (p["base"] + 1,), (p["base"] ** 2 + p["base"],)],
            uses=("to_base",),
        ),
        Func(
            "luhn",
            "checks",
            "def $luhn(number):\n"
            '    """True when the decimal digits of number pass the Luhn checksum.\n\n'
            "    From the rightmost digit leftwards, every second digit (the 2nd, 4th, ...) is\n"
            "    doubled, and 9 is taken off a doubled value above 9. The number passes when\n"
            '    the sum of all digits so treated is a multiple of 10.\n    """\n'
            "    total = 0\n"
            "    for i, ch in enumerate(reversed(str(number))):\n"
            "        d = int(ch)\n"
            "        if i % 2 == 1:\n"
            "            d = d * 2\n"
            "            if d > 9:\n"
            "                d -= 9\n"
            "        total += d\n"
            "    return total % 10 == 0",
            _luhn,
            public=[("valid", (79927398713,)), ("invalid", (79927398710,)), ("short", (18,))],
            hidden=[
                (0,),
                (5,),
                (42,),
                (59,),
                (4539578763621486,),
                (1234567812345670,),
                (1234567812345678,),
                (91,),
            ],
        ),
    ),
)

TEMPLATES = (
    Template(
        id="digits1",
        family=FAMILY,
        workflow="W1",
        target="to_base",
        concept="digits of an integer in a small base",
        bugs={
            "not_reversed": Bug(edits=(("return out[::-1]", "return out"),)),
            "no_zero": Bug(edits=(("    if n == 0:\n        return [0]\n", ""),)),
            "div_ten": Bug(edits=(("n //= $base", "n //= 10"),)),
        },
        spec=(
            "to_base(n) returns the digits of the non-negative int n written in base "
            "params['base'], most significant digit first, as a list of ints; [0] for n == 0."
        ),
        domain="n: int in 0..5000.",
        returns="list of int",
        include=("persist", "droot"),
    ),
    Template(
        id="digits2",
        family=FAMILY,
        workflow="W3",
        target="dsum",
        concept="decimal digit sum",
        bugs={
            "mod_nine": Bug(edits=(("n % 10", "n % 9"),)),
            "drop_lead": Bug(edits=(("while n > 0:", "while n > 9:"),)),
            "start_one": Bug(edits=(("    total = 0\n", "    total = 1\n"),)),
        },
        spec="dsum(n) returns the sum of the decimal digits of the non-negative int n (0 for 0).",
        domain="n: int in 0..10**7.",
        returns="int",
        include=("persist",),
        caller="droot",
    ),
    Template(
        id="digits3",
        family=FAMILY,
        workflow="W4",
        target="luhn",
        concept="Luhn checksum",
        bugs={"stub_raise": Bug(stub="raise"), "stub_none": Bug(stub="none")},
        spec=(
            "luhn(number) takes a non-negative int. Write its decimal digits; going from the "
            "rightmost digit leftwards, index them 0, 1, 2, ...; double each digit at an odd "
            "index and subtract 9 from a doubled value greater than 9. Return True when the "
            "sum of all resulting values is divisible by 10, else False. The number 0 has the "
            "single digit 0, so it gives True."
        ),
        domain="number: int in 0..10**17.",
        returns="bool",
        include=("same", "dsum"),
    ),
)
