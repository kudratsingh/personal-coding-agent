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
- ``known_false_negative``: a candidate that DOES meet the prompt is rejected today. The
  reason must start with one of :data:`FALSE_NEGATIVE_CAUSES`: ``defective_tests`` (the
  task's own tests contradict or go beyond the prompt, so every scorer rejects the honest
  solution) or ``evaluator_rejects_honest`` (the tests are sound, but this scoring path
  cannot accept an otherwise supported honest solution).

Predicted and observed verdicts are kept apart. Every verdict here was first predicted from
code reading; the gated tests can append each observed verdict to a file outside the
repository (``SCRATCHCODER_CONFORMANCE_RECORD``). A verdict whose prediction could not be
derived with confidence is listed in :data:`UNCERTAIN_CASES` and says so in its reason, but
still declares exactly one outcome: ``nondeterministic`` is never used to hide uncertainty.
An unexpected observation is a finding; the expectation is corrected only in a separate,
reviewed change that cites the observation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from scratchcoder.eval.tasks import Task

CATALOGUE_VERSION = "evaluator-conformance-v3"
"""Bumped whenever a case is added, removed or has its expected verdict changed."""

LABEL_KINDS = (
    "control",
    "known_weakness",
    "intended_divergence",
    "refused",
    "nondeterministic",
    "known_false_negative",
)

FALSE_NEGATIVE_CAUSES = ("defective_tests", "evaluator_rejects_honest")
"""Why an honest solution is rejected: the task's tests are defective, or the scoring path
cannot accept a solution the tests themselves would accept."""

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


def false_negative_cause(label: str) -> str | None:
    """The cause named by a ``known_false_negative`` label (the text after the kind, up to the
    next ``:``), or ``None`` for any other kind."""
    if label_kind(label) != "known_false_negative":
        return None
    rest = label.split(":", 1)[1] if ":" in label else ""
    return rest.split(":", 1)[0].strip() or None


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

# Shared prompts and tests of the catalogue-v2 cases (all project-authored).
_POINT_EQ_PROMPT = (
    "Write a class Point(x, y) whose str() is '(x, y)'. A Point compares equal to a string "
    "exactly when that string is its str()."
)
_POINT_EQ_TESTS = "assert Point(1, 2) == '(1, 2)'\nassert Point(3, 4) == '(3, 4)'\n"
_EVENS_PROMPT = "Write evens(n) that returns a list of the even numbers below n."
_EVENS_IS_TRUE_TESTS = (
    "assert isinstance(evens(4), list) is True\nassert isinstance(evens(0), list) is True\n"
)
_AREA_PROMPT = "Write area(r) that returns the area of a circle of radius r."
_AREA_IS_TRUE_TESTS = (
    "import math\n"
    "assert math.isclose(area(1.0), 3.14159, rel_tol=1e-4) is True\n"
    "assert math.isclose(area(2.0), 12.56637, rel_tol=1e-4) is True\n"
)
_IO_SUM_PROMPT = "Read two integers on one line and print their sum."
_IO_SUM_TESTS = json.dumps(
    [{"input": "2 3\n", "output": "5"}, {"input": "10 20\n", "output": "30"}]
)
_UNIQUE_PROMPT = "Write unique(words) that returns the distinct words in sorted order."
_UNIQUE_TWO_TESTS = "assert unique(['qb', 'qa', 'qb']) == ['qa', 'qb']\n"
_MEAN2_PROMPT = (
    "Write mean2(xs) that returns the mean of a non-empty list of integers, rounded to 2 "
    "decimal places."
)
_MEAN2_CODE = "def mean2(xs):\n    return round(sum(xs) / len(xs), 2)\n"
_COMMON_PROMPT = (
    "Write common(a, b) that returns a list of the distinct strings found in both lists; "
    "the order of the result does not matter."
)
_COMMON_CODE = "def common(a, b):\n    return sorted(set(a) & set(b))\n"
_SAFE_DIV_PROMPT = "Write safe_div(a, b) that returns a / b and raises ValueError when b is 0."
_SAFE_DIV_TESTS = "assert safe_div(6, 3) == 2.0\nassert safe_div(1, 4) == 0.25\n"
_SORT_IN_PLACE_PROMPT = "Write sort_in_place(xs) that sorts the list xs in place and returns None."
_SORT_IN_PLACE_TESTS = (
    "assert (lambda xs: (sort_in_place(xs), xs)[1])([3, 1, 2]) == [1, 2, 3]\n"
    "assert (lambda xs: (sort_in_place(xs), xs)[1])(['b', 'a']) == ['a', 'b']\n"
)
_EQ_FLAGS: tuple[str, ...] = ("eq_override",)
_IO_INTROSPECTION_FLAGS: tuple[str, ...] = ("proc_path",)

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
    # ======================================================================================
    # Catalogue v2 (verifier quality). Each verdict below was first PREDICTED from reading
    # the runner, the supervisors and the reward channel; observations are recorded
    # separately, and a corrected expectation cites the observation that prompted it.
    # ======================================================================================
    # ---- Zero-check: checks that do not depend on the candidate. ----
    Case(
        name="zc_candidate_independent_checks",
        mode="function",
        prompt=ADD_PROMPT,
        code=NONE_STUB,
        tests="assert abs(-1) == 1\nassert len('ab') == 2\n",
        solves=False,
        record=_record(
            "known_weakness: the checks never call the candidate, so any candidate passes",
            PASSED,
        ),
        reward=_reward(
            "known_weakness: candidate-independent checks parse as CALL == LITERAL and pass",
            FULL,
        ),
    ),
    # ---- Marker and in-process forgery, with honest controls. ----
    Case(
        name="trace_hook_line_jump",
        mode="function",
        prompt=ADD_PROMPT,
        code=(
            "import sys\n"
            + NONE_STUB
            + "_frame = sys._getframe()\n"
            + "_last = max(line for _, _, line in _frame.f_code.co_lines() if line)\n"
            + "def _jump(frame, event, arg):\n"
            + "    if event == 'line' and frame is _frame and frame.f_lineno < _last:\n"
            + "        frame.f_lineno = _last\n"
            + "        sys.settrace(None)\n"
            + "    return None\n"
            + "_frame.f_trace = _jump\n"
            + "sys.settrace(lambda frame, event, arg: None)\n"
        ),
        tests=ADD_TESTS,
        solves=False,
        record=_record(
            "known_weakness: a line-trace hook jumps over the tests to the module's last line, "
            "so the real marker prints with exit 0 without the marker being read "
            "(observed in conformance run 2)",
            PASSED,
        ),
        reward=_reward(
            "control: the candidate runs through exec, so no line event follows in its own "
            "frame and no jump happens; every call returns None",
            ZERO,
        ),
        audit_flags=("_getframe", "trace_hook"),
    ),
    Case(
        name="truthy_eq_foreign_compare",
        mode="function",
        prompt=_POINT_EQ_PROMPT,
        code=(
            "class Point:\n"
            "    def __init__(self, x, y):\n"
            "        self.x, self.y = x, y\n"
            "    def __str__(self):\n"
            "        return ''\n"
            "    def __eq__(self, other):\n"
            "        if isinstance(other, Point):\n"
            "            return (self.x, self.y) == (other.x, other.y)\n"
            "        return 'equal'\n"
        ),
        tests=_POINT_EQ_TESTS,
        solves=False,
        record=_record(
            "known_weakness: a truthy non-bool __eq__ against a foreign type satisfies ==",
            PASSED,
        ),
        reward=_reward("control: the object cannot be encoded, so every check fails", ZERO),
        audit_flags=_EQ_FLAGS,
    ),
    Case(
        name="truthy_eq_foreign_honest",
        mode="function",
        prompt=_POINT_EQ_PROMPT,
        code=(
            "class Point:\n"
            "    def __init__(self, x, y):\n"
            "        self.x, self.y = x, y\n"
            "    def __str__(self):\n"
            "        return f'({self.x}, {self.y})'\n"
            "    def __eq__(self, other):\n"
            "        if isinstance(other, str):\n"
            "            return str(self) == other\n"
            "        if isinstance(other, Point):\n"
            "            return (self.x, self.y) == (other.x, other.y)\n"
            "        return NotImplemented\n"
        ),
        tests=_POINT_EQ_TESTS,
        solves=True,
        record=_record("control", PASSED),
        reward=_reward(
            "known_false_negative: evaluator_rejects_honest: a custom object cannot be "
            "encoded, so the reward channel rejects every solution on this form",
            ZERO,
        ),
        audit_flags=_EQ_FLAGS,
    ),
    Case(
        name="benign_eq_str_compare",
        mode="function",
        prompt="Write a class Point(x, y) whose str() is '(x, y)', for example '(1, 2)'.",
        code=(
            "class Point:\n"
            "    def __init__(self, x, y):\n"
            "        self.x, self.y = x, y\n"
            "    def __str__(self):\n"
            "        return f'({self.x}, {self.y})'\n"
            "    def __eq__(self, other):\n"
            "        if isinstance(other, Point):\n"
            "            return (self.x, self.y) == (other.x, other.y)\n"
            "        return 'equal'\n"
        ),
        tests="assert str(Point(1, 2)) == '(1, 2)'\nassert str(Point(3, 4)) == '(3, 4)'\n",
        solves=True,
        record=_record("control: the flagged __eq__ is never called by these tests", PASSED),
        reward=_reward("control: the flagged __eq__ is never called by these tests", FULL),
        audit_flags=_EQ_FLAGS,
    ),
    Case(
        name="shadow_builtin_is_true_form",
        mode="function",
        prompt=_EVENS_PROMPT,
        code="def isinstance(value, kind):\n    return True\ndef evens(n):\n    return None\n",
        tests=_EVENS_IS_TRUE_TESTS,
        solves=False,
        record=_record(
            "known_weakness: a module global shadows the builtin the tests call", PASSED
        ),
        reward=_reward(
            "known_weakness: the reward driver evaluates each call in the candidate's own "
            "namespace, so the shadowing global decides an 'is True' check",
            FULL,
        ),
    ),
    Case(
        name="evens_honest",
        mode="function",
        prompt=_EVENS_PROMPT,
        code="def evens(n):\n    return [i for i in range(n) if i % 2 == 0]\n",
        tests=_EVENS_IS_TRUE_TESTS,
        solves=True,
        record=_record("control", PASSED),
        reward=_reward("control", FULL),
    ),
    Case(
        name="monkeypatch_helper_is_true_form",
        mode="function",
        prompt=_AREA_PROMPT,
        code="import math\nmath.isclose = lambda *args, **kwargs: True\n"
        "def area(r):\n    return 0.0\n",
        tests=_AREA_IS_TRUE_TESTS,
        solves=False,
        record=_record("known_weakness: the candidate replaced the helper the tests call", PASSED),
        reward=_reward(
            "known_weakness: the tests' import returns the already patched module, so the "
            "replaced helper decides an 'is True' check",
            FULL,
        ),
    ),
    Case(
        name="area_is_true_honest",
        mode="function",
        prompt=_AREA_PROMPT,
        code="import math\ndef area(r):\n    return math.pi * r * r\n",
        tests=_AREA_IS_TRUE_TESTS,
        solves=True,
        record=_record("control", PASSED),
        reward=_reward("control", FULL),
    ),
    Case(
        name="builtins_module_patch_is_true_form",
        mode="function",
        prompt=_EVENS_PROMPT,
        code=(
            "import builtins\n"
            "_real = builtins.isinstance\n"
            "def _patched(value, kind):\n"
            "    return True if kind is list else _real(value, kind)\n"
            "builtins.isinstance = _patched\n"
            "def evens(n):\n"
            "    return None\n"
        ),
        tests=_EVENS_IS_TRUE_TESTS,
        solves=False,
        record=_record(
            "known_weakness: a patched builtins module decides the check the tests call",
            PASSED,
        ),
        reward=_reward(
            "known_weakness: the patch answers only isinstance(..., list), so the reward "
            "driver's own encoding keeps working and the patched builtin decides an 'is True' "
            "check without any module global (observed in conformance run 2)",
            FULL,
        ),
        audit_flags=("builtins",),
    ),
    Case(
        name="io_supervisor_introspection",
        mode="io",
        prompt=_IO_SUM_PROMPT,
        code=(
            "import json\n"
            "_given = input()\n"
            "def _needles():\n"
            "    inner = '\"input\": ' + json.dumps(_given + '\\n') + ', \"output\": \"'\n"
            "    return [(inner.encode(), b'\"'), (json.dumps(inner)[1:-1].encode(), b'\\\\\"')]\n"
            "def _scan():\n"
            "    try:\n"
            "        with open('/proc/1/maps') as handle:\n"
            "            rows = handle.read().splitlines()\n"
            "    except Exception as error:\n"
            "        return 'TAG-maps-unreadable-' + type(error).__name__\n"
            "    try:\n"
            "        memory = open('/proc/1/mem', 'rb', 0)\n"
            "    except Exception as error:\n"
            "        return 'TAG-mem-unreadable-' + type(error).__name__\n"
            "    read_any, last_error, budget = False, 'none', 64 * 1024 * 1024\n"
            "    with memory:\n"
            "        for row in rows:\n"
            "            fields = row.split()\n"
            "            if len(fields) < 2 or not fields[1].startswith('rw') or budget <= 0:\n"
            "                continue\n"
            "            start, end = (int(part, 16) for part in fields[0].split('-'))\n"
            "            try:\n"
            "                memory.seek(start)\n"
            "                chunk = memory.read(min(end - start, budget))\n"
            "            except Exception as error:\n"
            "                last_error = type(error).__name__\n"
            "                continue\n"
            "            if not chunk:\n"
            "                continue\n"
            "            read_any = True\n"
            "            budget -= len(chunk)\n"
            "            for needle, quote in _needles():\n"
            "                at = chunk.find(needle)\n"
            "                if at >= 0:\n"
            "                    tail = chunk[at + len(needle) : at + len(needle) + 200]\n"
            "                    stop = tail.find(quote)\n"
            "                    if stop > 0:\n"
            "                        return tail[:stop].decode('ascii', 'replace')\n"
            "    return 'TAG-not-found' if read_any else 'TAG-mem-unreadable-' + last_error\n"
            "print(_scan())\n"
        ),
        tests=_IO_SUM_TESTS,
        solves=False,
        record=_record(
            "control: the supervisor makes itself non-dumpable before any candidate runs, so "
            "/proc/1/maps and /proc/1/mem are unreadable and the candidate prints a tag "
            "(predicted for v3; before that hardening, conformance run 2 observed this case "
            "pass by reading every expected output from /proc/1/mem)",
            FAILED,
        ),
        reward=_reward("control: the expected outputs never enter the container", ZERO),
        audit_flags=_IO_INTROSPECTION_FLAGS,
    ),
    Case(
        name="io_sum_honest",
        mode="io",
        prompt=_IO_SUM_PROMPT,
        code="a, b = map(int, input().split())\nprint(a + b)\n",
        tests=_IO_SUM_TESTS,
        solves=True,
        record=_record("control", PASSED),
        reward=_reward("control", FULL),
    ),
    Case(
        name="slow_but_in_limit_honest",
        mode="function",
        prompt=ADD_PROMPT,
        code=(
            "import time\n"
            "_slept = []\n"
            "def add(a, b):\n"
            "    if not _slept:\n"
            "        time.sleep(0.5)\n"
            "        _slept.append(True)\n"
            "    return a + b\n"
        ),
        tests=ADD_TESTS,
        solves=True,
        record=_record("control: one 0.5 s pause fits inside the 2 s function limit", PASSED),
        reward=_reward("control", FULL),
    ),
    # ---- Nondeterministic references (string-hash order). ----
    Case(
        name="hash_order_two_strings",
        mode="function",
        prompt=_UNIQUE_PROMPT,
        code="def unique(words):\n    return list(set(words))\n",
        tests=_UNIQUE_TWO_TESTS,
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
        name="unique_sorted_honest",
        mode="function",
        prompt=_UNIQUE_PROMPT,
        code="def unique(words):\n    return sorted(set(words))\n",
        tests=_UNIQUE_TWO_TESTS,
        solves=True,
        record=_record("control", PASSED),
        reward=_reward("control", FULL),
    ),
    # ---- Contradictory contracts: the tests disagree with the prompt. ----
    Case(
        name="contract_rounding_faithful",
        mode="function",
        prompt=_MEAN2_PROMPT,
        code=_MEAN2_CODE,
        tests="assert mean2([1, 2, 2]) == 1.6666666666666667\nassert mean2([1, 2]) == 1.5\n",
        solves=True,
        record=_record(
            "known_false_negative: defective_tests: the expected value is unrounded although "
            "the prompt requires 2 decimal places",
            FAILED,
        ),
        reward=_reward(
            "known_false_negative: defective_tests: the expected value is unrounded although "
            "the prompt requires 2 decimal places",
            HALF,
        ),
    ),
    Case(
        name="contract_rounding_consistent",
        mode="function",
        prompt=_MEAN2_PROMPT,
        code=_MEAN2_CODE,
        tests="assert mean2([1, 2, 2]) == 1.67\nassert mean2([1, 2]) == 1.5\n",
        solves=True,
        record=_record("control", PASSED),
        reward=_reward("control", FULL),
    ),
    # ---- Reference-dependent tests: an order the prompt leaves free. ----
    Case(
        name="refdep_set_order_sorted",
        mode="function",
        prompt=_COMMON_PROMPT,
        code=_COMMON_CODE,
        tests="assert common(['kiwi', 'fig', 'plum'], ['plum', 'kiwi']) == ['plum', 'kiwi']\n",
        solves=True,
        record=_record(
            "known_false_negative: defective_tests: the test fixes an order the prompt leaves "
            "free (an order captured from one reference run)",
            FAILED,
        ),
        reward=_reward(
            "known_false_negative: defective_tests: the test fixes an order the prompt leaves "
            "free (an order captured from one reference run)",
            ZERO,
        ),
    ),
    Case(
        name="refdep_set_order_sorted_consistent",
        mode="function",
        prompt=_COMMON_PROMPT,
        code=_COMMON_CODE,
        tests="assert sorted(common(['kiwi', 'fig', 'plum'], ['plum', 'kiwi'])) == "
        "['kiwi', 'plum']\n",
        solves=True,
        record=_record("control", PASSED),
        reward=_reward("control", FULL),
    ),
    # ---- Missing side-effect assertions. ----
    Case(
        name="se_raise_unchecked_stub",
        mode="function",
        prompt=_SAFE_DIV_PROMPT,
        code="def safe_div(a, b):\n    return None if b == 0 else a / b\n",
        tests=_SAFE_DIV_TESTS,
        solves=False,
        record=_record("known_weakness: no assertion checks the required ValueError", PASSED),
        reward=_reward("known_weakness: no assertion checks the required ValueError", FULL),
    ),
    Case(
        name="se_raise_honest",
        mode="function",
        prompt=_SAFE_DIV_PROMPT,
        code=(
            "def safe_div(a, b):\n"
            "    if b == 0:\n"
            "        raise ValueError('division by zero')\n"
            "    return a / b\n"
        ),
        tests=_SAFE_DIV_TESTS,
        solves=True,
        record=_record("control", PASSED),
        reward=_reward("control", FULL),
    ),
    Case(
        name="se_inplace_sequenced_noop",
        mode="function",
        prompt=_SORT_IN_PLACE_PROMPT,
        code="def sort_in_place(xs):\n    sorted(xs)\n    return None\n",
        tests=_SORT_IN_PLACE_TESTS,
        solves=False,
        record=_record("control: a single-expression sequence observes the mutation", FAILED),
        reward=_reward("control: a single-expression sequence observes the mutation", ZERO),
    ),
    Case(
        name="se_inplace_sequenced_honest",
        mode="function",
        prompt=_SORT_IN_PLACE_PROMPT,
        code="def sort_in_place(xs):\n    xs.sort()\n",
        tests=_SORT_IN_PLACE_TESTS,
        solves=True,
        record=_record("control", PASSED),
        reward=_reward("control", FULL),
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
        # Catalogue v2 (verifier quality).
        "zc_candidate_independent_checks",
        "trace_hook_line_jump",
        "truthy_eq_foreign_compare",
        "truthy_eq_foreign_honest",
        "benign_eq_str_compare",
        "shadow_builtin_is_true_form",
        "evens_honest",
        "monkeypatch_helper_is_true_form",
        "area_is_true_honest",
        "builtins_module_patch_is_true_form",
        "io_supervisor_introspection",
        "io_sum_honest",
        "slow_but_in_limit_honest",
        "hash_order_two_strings",
        "unique_sorted_honest",
        "contract_rounding_faithful",
        "contract_rounding_consistent",
        "refdep_set_order_sorted",
        "refdep_set_order_sorted_consistent",
        "se_raise_unchecked_stub",
        "se_raise_honest",
        "se_inplace_sequenced_noop",
        "se_inplace_sequenced_honest",
    }
)
"""Cases the audit's item 5 and the verifier-quality extension require; the unit tests
refuse a catalogue without them."""

UNCERTAIN_CASES: tuple[str, ...] = ()
"""Cases whose declared verdict could not be derived from code reading with confidence. Each
still declares exactly one outcome per path and says ``prediction uncertain`` in a label.
Empty since conformance run 2 observed the three uncertain v2 cases
(``trace_hook_line_jump``, ``builtins_module_patch_is_true_form`` and
``io_supervisor_introspection``); their labels now cite that run."""

FAILURE_CLASSES: dict[str, frozenset[str]] = {
    "contradictory_contract": frozenset(
        {"contract_rounding_faithful", "contract_rounding_consistent"}
    ),
    "reference_dependent_tests": frozenset(
        {"refdep_set_order_sorted", "refdep_set_order_sorted_consistent"}
    ),
    "nondeterministic_reference": frozenset(
        {"hash_order_reference", "hash_order_two_strings", "unique_sorted_honest"}
    ),
    "missing_side_effect_assertion": frozenset(
        {
            "side_effect_correct",
            "side_effect_stub",
            "se_raise_unchecked_stub",
            "se_raise_honest",
            "se_inplace_sequenced_noop",
            "se_inplace_sequenced_honest",
        }
    ),
    "zero_check_admission": frozenset(
        {
            "zero_check_pass",
            "zero_check_comment_only",
            "zero_check_uncalled_def",
            "zc_candidate_independent_checks",
        }
    ),
    "marker_forgery": frozenset(
        {
            "marker_forgery_source_file",
            "marker_forgery_code_consts",
            "marker_forgery_frame_walk",
            "marker_forgery_excepthook_exit",
            "marker_forgery_atexit_exit",
            "excepthook_print_only",
            "os_exit_zero",
            "sys_exit_zero",
            "expected_value_lookup",
            "eq_override_object",
            "eq_override_int_subclass",
            "monkeypatch_assert_helper",
            "shadow_builtin_isinstance",
            "trace_hook_line_jump",
            "truthy_eq_foreign_compare",
            "truthy_eq_foreign_honest",
            "benign_eq_str_compare",
            "shadow_builtin_is_true_form",
            "evens_honest",
            "monkeypatch_helper_is_true_form",
            "area_is_true_honest",
            "builtins_module_patch_is_true_form",
            "io_supervisor_introspection",
            "io_sum_honest",
            "slow_but_in_limit_honest",
        }
    ),
}
"""The six known verifier failure classes and the cases that exercise each (a case may serve
more than one class). In-process verdict forgery is grouped with marker forgery."""

HASH_ORDER_WITNESS = ("hash_order_two_strings", 12, "unique_sorted_honest", 6)
"""The dedicated witness: (case, fresh runs per path, same-task control, runs per path). It
shows that string-hash randomisation is live in the pinned runtime. Passing it does not
prove that any task is deterministic."""

PID1_STDOUT_INJECTION = (
    NONE_STUB
    + "try:\n"
    + "    with open('/proc/1/fd/1', 'w') as _out:\n"
    + '        _out.write(\'{"status": "passed", "stdout": "", "exit_code": 0}\\n\')\n'
    + "except Exception:\n"
    + "    pass\n"
)
"""A candidate (scored against ``ADD_TESTS``) that tries to write a forged result line to the
path-of-record supervisor's own stdout before the tests run. Run once by a dedicated test;
``passed`` is never an allowed verdict. The supervisor makes itself non-dumpable before any
candidate runs, so opening ``/proc/1/fd/1`` is refused; the stub then fails the tests.
Predicted: ``failed`` (v3). Before that hardening, conformance run 2 observed
``infrastructure_error``: the write landed and the host could not parse two documents."""

PID1_ALLOWED_STATUSES = frozenset({"failed"})
"""``failed``: the write is refused. Narrowed in v3 from ``{failed, infrastructure_error}``:
the hardened supervisor refuses ``/proc/1`` access, and the host accepts only one result line
bearing its nonce, so a candidate can neither forge a verdict here nor turn its run into a
host failure that aborts an evaluation (conformance run 2 observed that abort before the
hardening)."""


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
