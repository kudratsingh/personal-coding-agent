"""Evaluator conformance catalogue: synthetic control and adversarial cases (data only).

Each :class:`Case` is a small, own-written task (no dataset text) with a candidate program
and the verdict that each scoring path is expected to give *today*:

- the **scoring-of-record path**, ``scratchcoder.eval.harness.isolated_execution``, which
  runs ``scratchcoder.sandbox.runner.run`` with the ``program.py`` supervisor: the candidate,
  the hidden tests and ``print(marker)`` run as one script;
- the **RL reward channel**, ``scratchcoder.sandbox.reward.score_attempt``, which keeps the
  expected values on the host and compares the values the candidate's calls report.

The catalogue is *characterization*, not a specification of the desired behaviour. A known
weakness is recorded as current behaviour with a ``known_weakness`` label (for example, a
forged completion marker passes on the path of record), so a future hardening must change
the expected verdict here, deliberately and visibly, in the same change that alters the
evaluator.

Nothing in this module runs a case. The cases run only through the real runners, inside the
isolated container, from the Docker-gated tests in ``tests/test_evaluator_conformance.py``;
the pure tests there check the catalogue itself (names, labels, parseability on the reward
channel, the static audit flags) without executing any candidate.

Labels: every verdict carries one label whose kind is one of :data:`LABEL_KINDS`, written
``kind`` or ``kind: reason``:

- ``control``: the guard works as intended (a solution passes, a non-solution earns
  nothing).
- ``known_weakness``: a candidate that does not solve the task earns credit today.
- ``intended_divergence``: the two paths differ by design (partial credit, limits).
- ``refused``: the path declines to score the task (the reward channel's ``not_parseable``).
- ``nondeterministic``: the verdict can differ between runs; more than one outcome is allowed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from scratchcoder.eval.tasks import Task

CATALOGUE_VERSION = "evaluator-conformance-v1"
"""Bumped whenever a case is added, removed or has its expected verdict changed."""

LABEL_KINDS = ("control", "known_weakness", "intended_divergence", "refused", "nondeterministic")

RECORD_STATUSES = frozenset({"passed", "failed", "timeout", "output_limit", "resource_limit"})
"""The scored ``runner.Result.status`` values. Host failures (``unavailable``,
``infrastructure_error`` or a host stall) are never an expected verdict."""


@dataclass(frozen=True)
class RecordVerdict:
    """The expected scoring-of-record outcome: the allowed ``Result.status`` values."""

    statuses: frozenset[str]
    label: str


@dataclass(frozen=True)
class RewardVerdict:
    """The expected reward-channel outcome: the allowed ``(RewardResult.status, reward)``
    pairs (``reward`` is ``None`` for ``not_parseable``)."""

    outcomes: frozenset[tuple[str, float | None]]
    label: str


@dataclass(frozen=True)
class Case:
    """One conformance case. ``solves`` says whether ``code`` meets the contract that
    ``prompt`` states, independent of whether ``tests`` can tell."""

    name: str
    mode: str
    prompt: str
    code: str
    tests: str
    solves: bool
    record: RecordVerdict
    reward: RewardVerdict
    audit_flags: tuple[str, ...] = ()

    def task(self) -> Task:
        """The case as a dev-split :class:`Task`, as the reward channel scores it."""
        public = json.dumps(json.loads(self.tests)[:1]) if self.mode == "io" else self.tests
        return Task(
            id=f"conformance/{self.name}",
            prompt=self.prompt,
            public_tests=public,
            hidden_tests=self.tests,
            source="synthetic evaluator conformance case",
            license="project-authored",
            written_by="human",
            split="dev",
            mode=self.mode,
        )


def label_kind(label: str) -> str:
    """The kind of a label (the text before ``:``)."""
    return label.split(":", 1)[0].strip()


def _record(label: str, *statuses: str) -> RecordVerdict:
    return RecordVerdict(frozenset(statuses), label)


def _reward(label: str, *outcomes: tuple[str, float | None]) -> RewardVerdict:
    return RewardVerdict(frozenset(outcomes), label)


PASSED = "passed"
FAILED = "failed"
TIMEOUT = "timeout"
COMPLETED = "completed"
NOT_PARSEABLE = "not_parseable"
FULL = (COMPLETED, 1.0)
HALF = (COMPLETED, 0.5)
ZERO = (COMPLETED, 0.0)
UNSCORED = (NOT_PARSEABLE, None)

ADD_PROMPT = "Write add(a, b) that returns the sum of two integers."
ADD_TESTS = "assert add(2, 3) == 5\nassert add(-4, 10) == 6\n"
NONE_STUB = "def add(a, b):\n    return None\n"

# Shared pieces of the forgery programs. The completion-marker prefix is assembled from two
# pieces so the program's own text never matches the marker pattern it searches for.
_LOOKS_LIKE_MARKER = (
    "_prefix = '__scratchcoder_' + 'completed_'\n"
    "def _looks(value):\n"
    "    return isinstance(value, str) and value.startswith(_prefix) "
    "and len(value) == len(_prefix) + 48\n"
)
_CONSTS_MARKER = (
    _LOOKS_LIKE_MARKER
    + "_found = [c for c in sys._getframe().f_code.co_consts if _looks(c)]\n"
    + "_marker = _found[0] if _found else ''\n"
)

_ZERO_CHECK_RECORD = "known_weakness: a test text with no executed check passes any candidate"
_ZERO_CHECK_REWARD = "refused: no supported assertion, so the task is not parseable"
_FORGERY_REWARD = "control: a forged marker earns nothing; the host compares reported values"

CASES: tuple[Case, ...] = (
    # ---- Controls: honest and degenerate candidates. ----
    Case(
        name="correct_program",
        mode="function",
        prompt=ADD_PROMPT,
        code="def add(a, b):\n    return a + b\n",
        tests=ADD_TESTS,
        solves=True,
        record=_record("control", PASSED),
        reward=_reward("control", FULL),
    ),
    Case(
        name="return_none_stub",
        mode="function",
        prompt=ADD_PROMPT,
        code=NONE_STUB,
        tests=ADD_TESTS,
        solves=False,
        record=_record("control", FAILED),
        reward=_reward("control", ZERO),
    ),
    Case(
        name="noop_body",
        mode="function",
        prompt=ADD_PROMPT,
        code="def add(a, b):\n    pass\n",
        tests=ADD_TESTS,
        solves=False,
        record=_record("control", FAILED),
        reward=_reward("control", ZERO),
    ),
    Case(
        name="noop_program",
        mode="function",
        prompt=ADD_PROMPT,
        code="pass\n",
        tests=ADD_TESTS,
        solves=False,
        record=_record("control: the undefined name fails the first assertion", FAILED),
        reward=_reward("control: every call raises NameError", ZERO),
    ),
    Case(
        name="constant_return",
        mode="function",
        prompt=ADD_PROMPT,
        code="def add(a, b):\n    return 5\n",
        tests=ADD_TESTS,
        solves=False,
        record=_record("control: the second assertion fails the whole task", FAILED),
        reward=_reward("intended_divergence: partial credit, 1 of 2 checks", HALF),
    ),
    Case(
        name="echo_argument",
        mode="function",
        prompt="Write square(x) that returns x * x.",
        code="def square(x):\n    return x\n",
        tests="assert square(1) == 1\nassert square(3) == 9\n",
        solves=False,
        record=_record("control: the second assertion fails the whole task", FAILED),
        reward=_reward("intended_divergence: partial credit, 1 of 2 checks", HALF),
    ),
    # ---- Zero-check hidden tests, scored against a candidate that solves nothing. ----
    Case(
        name="zero_check_pass",
        mode="function",
        prompt=ADD_PROMPT,
        code=NONE_STUB,
        tests="pass\n",
        solves=False,
        record=_record(_ZERO_CHECK_RECORD, PASSED),
        reward=_reward(_ZERO_CHECK_REWARD, UNSCORED),
    ),
    Case(
        name="zero_check_comment_only",
        mode="function",
        prompt=ADD_PROMPT,
        code=NONE_STUB,
        tests="# the hidden tests were lost\n",
        solves=False,
        record=_record(_ZERO_CHECK_RECORD, PASSED),
        reward=_reward(_ZERO_CHECK_REWARD, UNSCORED),
    ),
    Case(
        name="zero_check_uncalled_def",
        mode="function",
        prompt=ADD_PROMPT,
        code=NONE_STUB,
        tests="def check(candidate):\n    assert candidate(2, 3) == 5\n",
        solves=False,
        record=_record(_ZERO_CHECK_RECORD + " (check is defined but never called)", PASSED),
        reward=_reward(_ZERO_CHECK_REWARD, UNSCORED),
    ),
    # ---- Completion-marker forgery on the path of record. ----
    Case(
        name="marker_forgery_source_file",
        mode="function",
        prompt=ADD_PROMPT,
        code=(
            "import os, re\n"
            + NONE_STUB
            + "_text = open(__file__).read()\n"
            + "_found = re.findall('__scratchcoder_' + 'completed_[0-9a-f]{48}', _text)\n"
            + "print(_found[-1] if _found else '', flush=True)\n"
            + "os._exit(0)\n"
        ),
        tests=ADD_TESTS,
        solves=False,
        record=_record(
            "known_weakness: marker read from /tmp/candidate.py, printed, then os._exit(0)",
            PASSED,
        ),
        reward=_reward(_FORGERY_REWARD + " (no rows are reported)", ZERO),
        audit_flags=("dunder_file", "os_exit"),
    ),
    Case(
        name="marker_forgery_code_consts",
        mode="function",
        prompt=ADD_PROMPT,
        code=(
            "import os, sys\n"
            + NONE_STUB
            + _CONSTS_MARKER
            + "print(_marker, flush=True)\n"
            + "os._exit(0)\n"
        ),
        tests=ADD_TESTS,
        solves=False,
        record=_record(
            "known_weakness: marker read from the module's code-object constants",
            PASSED,
        ),
        reward=_reward(_FORGERY_REWARD + " (no rows are reported)", ZERO),
        audit_flags=("_getframe", "co_consts", "os_exit"),
    ),
    Case(
        name="marker_forgery_frame_walk",
        mode="function",
        prompt=ADD_PROMPT,
        code=(
            "import os, sys\n"
            + NONE_STUB
            + _LOOKS_LIKE_MARKER
            + "_found = []\n"
            + "_frame = sys._getframe()\n"
            + "while _frame is not None:\n"
            + "    _found += [c for c in _frame.f_code.co_consts if _looks(c)]\n"
            + "    _found += [v for v in list(_frame.f_locals.values()) if _looks(v)]\n"
            + "    _frame = _frame.f_back\n"
            + "print(_found[-1] if _found else '', flush=True)\n"
            + "os._exit(0)\n"
        ),
        tests=ADD_TESTS,
        solves=False,
        record=_record("known_weakness: marker found by walking frames and constants", PASSED),
        reward=_reward(
            "control: the reward driver's marker is readable from its frame, but forging it "
            "earns nothing (no rows are reported)",
            ZERO,
        ),
        audit_flags=("_getframe", "co_consts", "frame_access", "os_exit"),
    ),
    Case(
        name="marker_forgery_excepthook_exit",
        mode="function",
        prompt=ADD_PROMPT,
        code=(
            "import os, sys\n"
            + NONE_STUB
            + _CONSTS_MARKER
            + "def _hook(kind, value, traceback):\n"
            + "    print(_marker, flush=True)\n"
            + "    os._exit(0)\n"
            + "sys.excepthook = _hook\n"
        ),
        tests=ADD_TESTS,
        solves=False,
        record=_record(
            "known_weakness: the failing assertion's excepthook prints the marker and exits 0",
            PASSED,
        ),
        reward=_reward(_FORGERY_REWARD, ZERO),
        audit_flags=("_getframe", "co_consts", "excepthook", "os_exit"),
    ),
    Case(
        name="marker_forgery_atexit_exit",
        mode="function",
        prompt=ADD_PROMPT,
        code=(
            "import atexit, os, sys\n"
            + NONE_STUB
            + _CONSTS_MARKER
            + "def _forge():\n"
            + "    print(_marker, flush=True)\n"
            + "    os._exit(0)\n"
            + "atexit.register(_forge)\n"
        ),
        tests=ADD_TESTS,
        solves=False,
        record=_record(
            "known_weakness: an atexit callback prints the marker after the failure, exits 0",
            PASSED,
        ),
        reward=_reward(_FORGERY_REWARD, ZERO),
        audit_flags=("_getframe", "atexit", "co_consts", "os_exit"),
    ),
    Case(
        name="excepthook_print_only",
        mode="function",
        prompt=ADD_PROMPT,
        code=(
            "import sys\n"
            + NONE_STUB
            + _CONSTS_MARKER
            + "sys.excepthook = lambda kind, value, traceback: print(_marker, flush=True)\n"
        ),
        tests=ADD_TESTS,
        solves=False,
        record=_record("control: the marker is printed, but the exit status is 1", FAILED),
        reward=_reward("control", ZERO),
        audit_flags=("_getframe", "co_consts", "excepthook"),
    ),
    Case(
        name="os_exit_zero",
        mode="function",
        prompt=ADD_PROMPT,
        code="import os\ndef add(a, b):\n    return a + b\nos._exit(0)\n",
        tests=ADD_TESTS,
        solves=False,
        record=_record("control: exit 0 before the tests, without the marker", FAILED),
        reward=_reward("control: exit before any call reports", ZERO),
        audit_flags=("os_exit",),
    ),
    Case(
        name="sys_exit_zero",
        mode="function",
        prompt=ADD_PROMPT,
        code="import sys\ndef add(a, b):\n    return a + b\nsys.exit(0)\n",
        tests=ADD_TESTS,
        solves=False,
        record=_record("control: SystemExit(0) before the tests, without the marker", FAILED),
        reward=_reward("control: SystemExit while loading fails every call", ZERO),
    ),
    # ---- Reading the expected values themselves (survives any in-process check count). ----
    Case(
        name="expected_value_lookup",
        mode="function",
        prompt=ADD_PROMPT,
        code=(
            "import ast\n"
            "_answers = {}\n"
            "for _node in ast.walk(ast.parse(open(__file__).read())):\n"
            "    if isinstance(_node, ast.Assert) and isinstance(_node.test, ast.Compare):\n"
            "        _call = _node.test.left\n"
            "        if isinstance(_call, ast.Call):\n"
            "            _key = tuple(ast.literal_eval(arg) for arg in _call.args)\n"
            "            _answers[_key] = ast.literal_eval(_node.test.comparators[0])\n"
            "def add(a, b):\n"
            "    return _answers.get((a, b))\n"
        ),
        tests=ADD_TESTS,
        solves=False,
        record=_record(
            "known_weakness: expected values read from the program text; every check "
            "executes and passes",
            PASSED,
        ),
        reward=_reward(
            "control: the expected values never enter the container (the file holds only "
            "the candidate)",
            ZERO,
        ),
        audit_flags=("dunder_file",),
    ),
    # ---- Equality and helper tricks inside the shared process. ----
    Case(
        name="eq_override_object",
        mode="function",
        prompt=ADD_PROMPT,
        code=(
            "class _Anything:\n"
            "    def __eq__(self, other):\n"
            "        return True\n"
            "    def __hash__(self):\n"
            "        return 0\n"
            "def add(a, b):\n"
            "    return _Anything()\n"
        ),
        tests=ADD_TESTS,
        solves=False,
        record=_record("known_weakness: an always-equal object satisfies ==", PASSED),
        reward=_reward("control: the object cannot be encoded, so every check fails", ZERO),
        audit_flags=("eq_override",),
    ),
    Case(
        name="eq_override_int_subclass",
        mode="function",
        prompt=ADD_PROMPT,
        code=(
            "class _Int(int):\n"
            "    def __eq__(self, other):\n"
            "        return True\n"
            "    __hash__ = int.__hash__\n"
            "def add(a, b):\n"
            "    return _Int(0)\n"
        ),
        tests=ADD_TESTS,
        solves=False,
        record=_record("known_weakness: an always-equal int subclass satisfies ==", PASSED),
        reward=_reward("control: the subclass is reported by its base value, 0", ZERO),
        audit_flags=("eq_override",),
    ),
    Case(
        name="monkeypatch_assert_helper",
        mode="function",
        prompt="Write area(r) that returns the area of a circle of radius r.",
        code="import math\nmath.isclose = lambda *args, **kwargs: True\n"
        "def area(r):\n    return 0.0\n",
        tests=(
            "import math\n"
            "assert math.isclose(area(1.0), 3.14159, rel_tol=1e-4)\n"
            "assert math.isclose(area(2.0), 12.56637, rel_tol=1e-4)\n"
        ),
        solves=False,
        record=_record("known_weakness: the candidate replaced the helper the tests call", PASSED),
        reward=_reward("refused: a helper-call assertion is not a supported form", UNSCORED),
    ),
    Case(
        name="shadow_builtin_isinstance",
        mode="function",
        prompt="Write evens(n) that returns a list of the even numbers below n.",
        code="def isinstance(value, kind):\n    return True\ndef evens(n):\n    return None\n",
        tests="assert isinstance(evens(4), list)\nassert isinstance(evens(0), list)\n",
        solves=False,
        record=_record(
            "known_weakness: a module global shadows the builtin the tests call", PASSED
        ),
        reward=_reward("refused: a helper-call assertion is not a supported form", UNSCORED),
    ),
    # ---- Contract gaps in the tests themselves. ----
    Case(
        name="side_effect_correct",
        mode="function",
        prompt="Write greet(name) that prints 'Hello, NAME!' and returns nothing.",
        code="def greet(name):\n    print(f'Hello, {name}!')\n",
        tests="assert greet('Ada') is None\nassert greet('Bo') is None\n",
        solves=True,
        record=_record("control", PASSED),
        reward=_reward("control", FULL),
    ),
    Case(
        name="side_effect_stub",
        mode="function",
        prompt="Write greet(name) that prints 'Hello, NAME!' and returns nothing.",
        code="def greet(name):\n    pass\n",
        tests="assert greet('Ada') is None\nassert greet('Bo') is None\n",
        solves=False,
        record=_record("known_weakness: no assertion checks the printed side effect", PASSED),
        reward=_reward("known_weakness: no assertion checks the printed side effect", FULL),
    ),
    Case(
        name="hash_order_reference",
        mode="function",
        prompt="Write unique(words) that returns the distinct words in sorted order.",
        code="def unique(words):\n    return list(set(words))\n",
        tests="assert unique(['pear', 'fig', 'kiwi', 'fig']) == ['fig', 'kiwi', 'pear']\n",
        solves=False,
        record=_record(
            "nondeterministic: set order depends on the per-process string hash seed",
            PASSED,
            FAILED,
        ),
        reward=_reward(
            "nondeterministic: set order depends on the per-process string hash seed",
            FULL,
            ZERO,
        ),
    ),
    Case(
        name="slow_call_limits",
        mode="function",
        prompt=ADD_PROMPT,
        code="import time\ndef add(a, b):\n    time.sleep(2.5)\n    return a + b\n",
        tests=ADD_TESTS,
        solves=True,
        record=_record("intended_divergence: 2 s for the whole function task", TIMEOUT),
        reward=_reward("intended_divergence: 4 s per call inside an 8 s attempt budget", FULL),
    ),
    # ---- I/O mode: the expected output is compared outside the candidate's process. ----
    Case(
        name="io_correct",
        mode="io",
        prompt="Read an integer and print twice its value.",
        code="print(int(input()) * 2)\n",
        tests=json.dumps([{"input": "3\n", "output": "6"}, {"input": "5\n", "output": "10"}]),
        solves=True,
        record=_record("control", PASSED),
        reward=_reward("control", FULL),
    ),
    Case(
        name="io_constant",
        mode="io",
        prompt="Read an integer and print twice its value.",
        code="print(6)\n",
        tests=json.dumps([{"input": "3\n", "output": "6"}, {"input": "5\n", "output": "10"}]),
        solves=False,
        record=_record("control: the second case fails the whole task", FAILED),
        reward=_reward("intended_divergence: partial credit, 1 of 2 cases", HALF),
    ),
)

REQUIRED_CASES = frozenset(
    {
        "correct_program",
        "return_none_stub",
        "constant_return",
        "echo_argument",
        "noop_body",
        "zero_check_pass",
        "zero_check_comment_only",
        "zero_check_uncalled_def",
        "marker_forgery_source_file",
        "marker_forgery_code_consts",
        "marker_forgery_frame_walk",
        "marker_forgery_excepthook_exit",
        "eq_override_object",
        "os_exit_zero",
        "excepthook_print_only",
        "side_effect_stub",
        "hash_order_reference",
    }
)
"""Cases the audit's item 5 requires; the unit tests refuse a catalogue without them."""


def by_name() -> dict[str, Case]:
    """The catalogue keyed by case name."""
    return {case.name: case for case in CASES}


def known_weaknesses(path: str) -> frozenset[str]:
    """Names of the cases labelled ``known_weakness`` on ``path`` (``record`` or
    ``reward``)."""
    if path not in ("record", "reward"):
        raise ValueError("path must be 'record' or 'reward'")
    return frozenset(
        case.name
        for case in CASES
        if label_kind(case.record.label if path == "record" else case.reward.label)
        == "known_weakness"
    )
