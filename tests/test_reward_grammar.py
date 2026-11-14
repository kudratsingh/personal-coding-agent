"""The reward channel's supported-check grammar (reward-grammar-v2): what it keeps, what it
refuses and why. Pure: nothing here runs a candidate."""

import pytest

from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox import conformance
from scratchcoder.sandbox.reward import (
    GRAMMAR_V1,
    GRAMMAR_V2_REASONS,
    REWARD_GRAMMAR_VERSION,
    UNSUPPORTED_REASONS,
    FunctionChecks,
    Unsupported,
    check_forms,
    degenerate_baseline,
    explain_function_checks,
    explain_io_cases,
    is_parseable,
    unsupported_reason,
)


def task(hidden: str, mode: str = "function") -> Task:
    return Task(
        id="grammar/fixture",
        prompt="Write the function the tests call.",
        public_tests="assert True" if mode == "function" else hidden,
        hidden_tests=hidden,
        source="synthetic",
        license="project-authored",
        written_by="human",
        split="train",
        mode=mode,
    )


# Legitimate checks that stay supported, builtin wrappers among them: against an informative
# literal the expected value never enters the container, so a replaced wrapper still has to
# produce it unseen.
SUPPORTED = [
    "assert f(3) == 9",
    "assert 9 == f(3)",
    "assert f(3) is None",
    "assert f(3) is True",
    "assert not f(3)",
    "assert f(3)[0] is True",
    "assert f(3)[1:] == [2, 3]",
    "assert len(f(3)) == 3",
    "assert sorted(f([3, 1, 2])) == [1, 2, 3]",
    "assert round(f(2.0), 2) == 12.57",
    "assert set(f([1, 1, 2])) == {1, 2}",
    "assert abs(f(-2)) == 2",
    "assert list(f(3)) == [0, 1, 2]",
    "assert tuple(f(3)) == (0, 1, 2)",
    "assert str(f(3)) == '3'",
    "assert repr(Point(1, 2)) == 'Point(1, 2)'",
    "assert Solution().solve(4) == 16",
    "assert Account(10).balance == 10",
    "assert f(3).count(1) == 2",
    "import math\nassert f(math.sqrt(4)) == 2",
    "assert f([x * x for x in range(3)]) == 5",
    "assert f(sorted([3, 1])) is True",
    "assert f(lambda y: y + 1) == 2",
]


@pytest.mark.parametrize("hidden", SUPPORTED)
def test_legitimate_checks_stay_supported(hidden: str) -> None:
    assert isinstance(explain_function_checks(hidden), FunctionChecks), hidden
    assert unsupported_reason(task(hidden)) is None


REFUSED: list[tuple[str, str, str | None]] = [
    # The demonstrated shadowing attacks' check shapes (conformance run 2).
    (
        "assert isinstance(evens(4), list) is True",
        "guessable_wrapper",
        "is True on builtin isinstance()",
    ),
    (
        "import math\nassert math.isclose(area(1.0), 3.14159, rel_tol=1e-4) is True",
        "guessable_wrapper",
        "is True on imported math.isclose()",
    ),
    # The same shapes through ==, not or other guessable answers.
    (
        "assert isinstance(f(4), list) == True",
        "guessable_wrapper",
        "== True on builtin isinstance()",
    ),
    ("assert bool(f(3)) is True", "guessable_wrapper", "is True on builtin bool()"),
    ("assert not any(f(3))", "guessable_wrapper", "not on builtin any()"),
    ("assert len(f(3)) == False", "guessable_wrapper", "== False on builtin len()"),
    ("assert f(3).startswith('a') is True", "guessable_wrapper", "is True on method .startswith()"),
    ("assert Stack().pop() is None", "guessable_wrapper", "is None on method .pop()"),
    (
        "assert Book(1900).is_classic() == True",
        "guessable_wrapper",
        "== True on method .is_classic()",
    ),
    ("assert Account(0).open is False", "guessable_wrapper", "is False on attribute .open"),
    ("assert (f(1) + 1) is None", "guessable_wrapper", "is None on BinOp"),
    # Checks that never depend on the candidate.
    ("assert abs(-1) == 1", "candidate_independent", "== literal on builtin abs()"),
    ("import math\nassert math.sqrt(4) == 2.0", "candidate_independent", None),
    ("assert [x for x in range(3)] == [0, 1, 2]", "candidate_independent", None),
    ("from solution import f\nassert f(1) == 2", "candidate_independent", None),
    ("from math import *\nassert f(1) == 2", "star_import", None),
]


@pytest.mark.parametrize("hidden,reason,form", REFUSED)
def test_refused_checks_are_explicit_exclusions(hidden: str, reason: str, form: str | None) -> None:
    refused = explain_function_checks(hidden)
    assert isinstance(refused, Unsupported) and refused.reason == reason, refused
    if form is not None:
        assert refused.form == form
    assert not is_parseable(task(hidden))
    assert degenerate_baseline(task(hidden)) is None  # never a silent zero or success
    if reason != "star_import" and "from solution" not in hidden:
        # The previous grammar accepted every one of these: v2 is what refuses them.
        assert isinstance(explain_function_checks(hidden, grammar=GRAMMAR_V1), FunctionChecks)


def test_one_refused_check_excludes_the_whole_task() -> None:
    hidden = "assert f(1) == 2\nassert isinstance(f(2), int) is True\nassert f(3) == 4"
    refused = explain_function_checks(hidden)
    assert isinstance(refused, Unsupported)
    assert (refused.reason, refused.index) == ("guessable_wrapper", 1)
    assert [row[1] for row in check_forms(hidden)] == [
        "supported",
        "guessable_wrapper",
        "supported",
    ]


@pytest.mark.parametrize(
    "hidden,reason",
    [
        ("assert f(1) ==", "syntax"),
        ("x = 1\nassert f(1) == 2", "statement"),
        ("assert f(1) > 2", "form"),
        ("assert f(1) == g(2)", "form"),
        ("import math", "no_checks"),
    ],
)
def test_earlier_refusals_keep_their_own_reasons(hidden: str, reason: str) -> None:
    refused = explain_function_checks(hidden)
    assert isinstance(refused, Unsupported) and refused.reason == reason
    assert reason not in GRAMMAR_V2_REASONS


@pytest.mark.parametrize(
    "hidden,reason",
    [
        ("not json", "io_malformed"),
        ("[]", "io_case_count"),
        ('[{"input": "1"}]', "io_malformed"),
    ],
)
def test_io_refusals_name_their_reason(hidden: str, reason: str) -> None:
    refused = explain_io_cases(hidden)  # Task() itself refuses such cases.
    assert isinstance(refused, Unsupported) and refused.reason == reason


def test_the_grammar_is_versioned_and_reasons_are_closed() -> None:
    assert REWARD_GRAMMAR_VERSION == "reward-grammar-v2"
    assert GRAMMAR_V2_REASONS <= set(UNSUPPORTED_REASONS)
    with pytest.raises(ValueError):
        explain_function_checks("assert f(1) == 2", grammar="reward-grammar-v0")


@pytest.mark.parametrize(
    "name",
    [
        "zc_candidate_independent_checks",
        "shadow_builtin_is_true_form",
        "monkeypatch_helper_is_true_form",
        "builtins_module_patch_is_true_form",
    ],
)
def test_the_demonstrated_attacks_tasks_are_refused(name: str) -> None:
    case = conformance.by_name()[name]
    refused = unsupported_reason(case.task())
    assert refused is not None and refused.reason in GRAMMAR_V2_REASONS
