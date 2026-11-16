"""Family ``rpn``: postfix programs (ints and operator strings) run on a stack."""

from __future__ import annotations

from typing import Any

from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template


def _apply_op(p: Params, op: str, a: Any, b: Any, bug: Any = None) -> Any:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return a // b


def _eval_rpn(p: Params, tk: list[Any], bug: Any = None) -> Any:
    stack: list[Any] = []
    for t in tk:
        if isinstance(t, int):
            stack.append([t] if bug == "wrapped" else t)
        else:
            b = stack.pop()
            a = stack.pop()
            stack.append(_apply_op(p, t, a, b, bug))
    return stack[1] if bug == "second" else stack[-1]


def _depth(p: Params, tk: list[Any], bug: Any = None) -> int:
    size = most = 0
    for t in tk:
        size = size + 1 if isinstance(t, int) else size - 1
        most = max(most, size)
    return most


def _valid(p: Params, tk: list[Any], bug: Any = None) -> bool:
    size = 0
    for t in tk:
        if isinstance(t, int):
            size += 1
        elif size < (1 if bug == "short_one" else 2):
            return False
        else:
            size -= 2 if bug == "drop_two" else 1
    return size >= 1 if bug == "leftovers" else size == 1


def _run_all(p: Params, programs: list[Any], bug: Any = None) -> list[Any]:
    return [_eval_rpn(p, prog, bug) for prog in programs]


FAMILY = Family(
    id="rpn",
    domain="postfix (reverse Polish) programs evaluated on a stack",
    modules={
        "main": ("stack_calc", "rpn", "postfix"),
        "machine": ("machine", "programs", "interp"),
    },
    docs={
        "main": "Postfix programs: lists of ints and operator strings.",
        "machine": "Running many postfix programs.",
    },
    names={
        "apply_op": ("apply_op", "combine_top", "binary_step"),
        "eval_rpn": ("eval_postfix", "run_rpn", "evaluate_tokens"),
        "depth": ("max_depth", "stack_depth", "peak_depth"),
        "valid": ("is_valid", "well_formed", "check_program"),
        "run_all": ("run_all", "results_of", "evaluate_all"),
        "tk": ("tokens", "program", "items"),
    },
    consts={},
    funcs=(
        Func(
            "apply_op",
            "main",
            "def $apply_op(op, a, b):\n"
            "    \"\"\"a op b for op one of '+', '-', '*' and '//'.\"\"\"\n"
            "    if op == '+':\n"
            "        return a + b\n"
            "    if op == '-':\n"
            "        return a - b\n"
            "    if op == '*':\n"
            "        return a * b\n"
            "    return a // b",
            _apply_op,
            hidden=[("+", 2, 3), ("*", 4, 5), ("//", 9, 2)],
        ),
        Func(
            "eval_rpn",
            "main",
            "def $eval_rpn($tk):\n"
            '    """Value of a postfix program: an int is pushed; an operator pops b, then a,\n'
            '    and pushes a op b. The value is the top of the stack at the end."""\n'
            "    stack = []\n"
            "    for t in $tk:\n"
            "        if isinstance(t, int):\n"
            "            stack.append(t)\n"
            "        else:\n"
            "            b = stack.pop()\n"
            "            a = stack.pop()\n"
            "            stack.append($apply_op(t, a, b))\n"
            "    return stack[-1]",
            _eval_rpn,
            hidden=[
                ([5],),
                ([2, 3, "+"],),
                ([7, 2, "-"],),
                ([2, 3, 4, "*", "+"],),
                ([9, 2, "//"],),
                ([1, 2, "-", 3, "*"],),
                ([6, 2, 2, "*", "-"],),
            ],
            uses=("apply_op",),
        ),
        Func(
            "depth",
            "main",
            "def $depth($tk):\n"
            '    """Largest number of values on the stack while the program runs (each int\n'
            '    adds one value; each operator takes two and adds one)."""\n'
            "    size = 0\n"
            "    most = 0\n"
            "    for t in $tk:\n"
            "        size = size + 1 if isinstance(t, int) else size - 1\n"
            "        most = max(most, size)\n"
            "    return most",
            _depth,
            public=[
                ("nested", ([1, 2, 3, "*", "+"],)),
                ("flat", ([1, 2, "+", 3, "+"],)),
                ("empty", ([],)),
            ],
            hidden=[
                ([4],),
                ([1, 2, "-"],),
                ([1, 2, 3, 4, "+", "+", "+"],),
                ([5, 5, "*", 5, 5, "*", "+"],),
                ([1, 1, "+", 1, "+", 1, "+"],),
                ([2, 3, 4, "*", 5, 6, "+", "-", "+"],),
            ],
        ),
        Func(
            "valid",
            "machine",
            "def $valid($tk):\n"
            '    """True when no operator runs short of values and one value is left at the\n'
            '    end."""\n'
            "    size = 0\n"
            "    for t in $tk:\n"
            "        if isinstance(t, int):\n"
            "            size += 1\n"
            "        elif size < 2:\n"
            "            return False\n"
            "        else:\n"
            "            size -= 1\n"
            "    return size == 1",
            _valid,
            public=[
                ("good", ([3, 4, "+"],)),
                ("leftover", ([3, 4],)),
                ("short", ([3, "+", 4],)),
                ("long_ok", ([1, 2, 3, "+", "+"],)),
            ],
            hidden=[
                ([],),
                ([7],),
                (["+"],),
                ([1, "-"],),
                ([1, 2, "*", "*"],),
                ([1, 2, 3, "+"],),
                ([1, 2, "+", 3, "*"],),
                ([5, 5, 5, 5, "+", "+", "+"],),
                ([5, "-", 6],),
                ([2, "*", 7],),
            ],
        ),
        Func(
            "run_all",
            "machine",
            "def $run_all(programs):\n"
            '    """The value of each program, in order."""\n'
            "    return [$eval_rpn(prog) for prog in programs]",
            _run_all,
            public=[
                ("mixed", ([[7, 2, "-"], [4, 1, "+"]],)),
                ("products", ([[2, 3, "*"], [4]],)),
            ],
            hidden=[
                ([],),
                ([[5]],),
                ([[1, 2, "+"], [3, 4, "*"]],),
                ([[9, 3, "-", 2, "*"]],),
                ([[8, 2, "//"], [2, 8, "//"]],),
                ([[6, 3, "//", 1, "+"]],),
            ],
            uses=("eval_rpn",),
        ),
    ),
)

TEMPLATES = (
    Template(
        id="rpn1",
        family=FAMILY,
        workflow="W5",
        target="eval_rpn",
        concept="evaluate a postfix program",
        bugs={
            "second": Bug(edits=(("return stack[-1]", "return stack[1]"),)),
            "wrapped": Bug(edits=(("stack.append(t)", "stack.append([t])"),)),
        },
        spec=(
            "eval_rpn(tokens) evaluates a well-formed postfix program. tokens is a list of ints "
            "and operator strings '+', '-', '*', '//'. Process tokens in order on a stack: an "
            "int is pushed; an operator pops b (top) then a and pushes a + b, a - b, a * b or "
            "a // b, where // is Python floor division (it rounds towards negative infinity, "
            "e.g. -5 // 3 == -2). Return the top of the stack at the end."
        ),
        domain=(
            "tokens: well-formed program (exactly one value left at the end) of 1..9 tokens, "
            "ints in 0..9; the divisor b of every '//' is non-zero when it is applied."
        ),
        returns="int",
        caller="run_all",
    ),
    Template(
        id="rpn2",
        family=FAMILY,
        workflow="W1",
        target="valid",
        concept="well-formedness of a postfix program",
        bugs={
            "leftovers": Bug(edits=(("return size == 1", "return size >= 1"),)),
            "short_one": Bug(edits=(("elif size < 2:", "elif size < 1:"),)),
            "drop_two": Bug(edits=(("            size -= 1", "            size -= 2"),)),
        },
        spec=(
            "valid(tokens) checks a postfix program of ints and operator strings. Walk the "
            "tokens with a counter starting at 0: an int adds 1; an operator needs the counter "
            "to be at least 2 (otherwise return False at once, whatever follows) and then "
            "lowers it by 1. Return True when the counter is exactly 1 at the end, so an empty "
            "list gives False. Every token that is not an int is an operator."
        ),
        domain="tokens: list of 0..9 items, each an int in 0..9 or one of '+', '-', '*'.",
        returns="bool",
        include=("depth",),
    ),
    Template(
        id="rpn3",
        family=FAMILY,
        workflow="W4",
        target="depth",
        concept="maximum stack depth of a postfix program",
        bugs={"stub_raise": Bug(stub="raise"), "stub_none": Bug(stub="none")},
        spec=(
            "depth(tokens) returns the largest stack size reached while running a postfix "
            "program, starting from an empty stack: each int adds 1 to the size and each "
            "operator string lowers it by 1 (it takes two values and pushes one). An empty "
            "program gives 0."
        ),
        domain=(
            "tokens: the empty list, or a well-formed postfix program (exactly one value left "
            "at the end) of 1..9 tokens (ints 0..9 and '+', '-', '*')."
        ),
        returns="int",
        include=("run_all",),
    ),
)
