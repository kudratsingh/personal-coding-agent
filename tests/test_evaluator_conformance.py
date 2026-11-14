"""Evaluator conformance suite: the catalogue in ``scratchcoder.sandbox.conformance``.

The pure tests check the catalogue itself (names, labels, parseability on the reward channel,
the static audit flags, the pinned list of known weaknesses). They parse candidate text but
never run it.

The tests at the bottom are gated behind ``SCRATCHCODER_SANDBOX_TESTS=1`` (an explicitly
enabled disposable Docker worker) and run every case through the two real scoring paths in
the pinned container: the scoring-of-record path (``harness.isolated_execution``, the
``program.py`` supervisor) and the RL reward channel (``reward.score_attempt``). Each
observed verdict must be one the catalogue declares, so a change in either evaluator that
flips a verdict, including closing a known weakness, fails here until the catalogue is
updated with it. On the owner's Mac they run only inside a passing execution window:

    env -u SSH_AUTH_SOCK uv run --locked python -m tools.mac_window_check exec \\
        --image-id <recorded image ID> --dotenv .env -- \\
        env SCRATCHCODER_SANDBOX_TESTS=1 uv run --locked pytest tests/test_evaluator_conformance.py
"""

import ast
import json
import os
from typing import Any

import pytest

from scratchcoder.eval.harness import is_host_failure, isolated_execution
from scratchcoder.sandbox import conformance, reward
from scratchcoder.sandbox.conformance import (
    CASES,
    LABEL_KINDS,
    RECORD_STATUSES,
    REQUIRED_CASES,
    Case,
    label_kind,
)
from scratchcoder.sandbox.runner import Limits

NAMES = [case.name for case in CASES]
SCORED_REWARD_STATUSES = {
    reward.COMPLETED,
    reward.RESOURCE_KILL,
    reward.MALFORMED_REPORT,
    reward.NOT_PARSEABLE,
}


def test_catalogue_is_versioned_and_complete() -> None:
    assert conformance.CATALOGUE_VERSION == "evaluator-conformance-v1"
    assert len(NAMES) == len(set(NAMES))
    assert REQUIRED_CASES <= set(NAMES)
    assert all(name.isidentifier() and name == name.lower() for name in NAMES)
    assert conformance.by_name() == {case.name: case for case in CASES}


def test_every_verdict_uses_the_declared_vocabulary() -> None:
    for case in CASES:
        assert case.mode in ("function", "io"), case.name
        assert label_kind(case.record.label) in LABEL_KINDS, case.name
        assert label_kind(case.reward.label) in LABEL_KINDS, case.name
        assert case.record.statuses and case.record.statuses <= RECORD_STATUSES, case.name
        assert case.reward.outcomes, case.name
        for status, value in case.reward.outcomes:
            assert status in SCORED_REWARD_STATUSES, case.name
            assert status not in reward.HOST_FAILURES, case.name
            assert (value is None) == (status == reward.NOT_PARSEABLE), case.name
            assert value is None or 0.0 <= value <= 1.0, case.name


def _record_label_agrees(case: Case) -> None:
    kind = label_kind(case.record.label)
    statuses = case.record.statuses
    assert (kind == "nondeterministic") == (len(statuses) > 1), case.name
    assert kind != "refused", case.name  # the path of record scores every well-formed task
    if kind == "control":
        assert (statuses == {"passed"}) if case.solves else ("passed" not in statuses), case.name
    if kind == "known_weakness":
        assert not case.solves and statuses == {"passed"}, case.name


def _reward_label_agrees(case: Case) -> None:
    kind = label_kind(case.reward.label)
    outcomes = case.reward.outcomes
    values = {value for _, value in outcomes}
    assert (kind == "nondeterministic") == (len(outcomes) > 1), case.name
    assert (kind == "refused") == (outcomes == {(reward.NOT_PARSEABLE, None)}), case.name
    if kind == "control":
        assert values == ({1.0} if case.solves else {0.0}), case.name
    if kind == "known_weakness":
        assert not case.solves and any(value for value in values), case.name


@pytest.mark.parametrize("case", CASES, ids=NAMES)
def test_labels_agree_with_the_declared_verdicts(case: Case) -> None:
    """A ``control`` passes a solution and fails a non-solution; a ``known_weakness`` gives a
    non-solution credit; ``refused`` is exactly ``not_parseable``; only ``nondeterministic``
    allows more than one outcome."""
    _record_label_agrees(case)
    _reward_label_agrees(case)


@pytest.mark.parametrize("case", CASES, ids=NAMES)
def test_case_text_is_well_formed_for_both_runners(case: Case) -> None:
    ast.parse(case.code)  # parsed only, never compiled or run
    if case.mode == "function":
        ast.parse(case.tests)
        assert case.tests.strip()  # ``runner.run`` refuses empty tests before running
    else:
        cases = json.loads(case.tests)
        assert isinstance(cases, list) and 1 <= len(cases) <= reward.MAX_IO_CASES
    assert len((case.code + case.tests).encode()) <= Limits().input_bytes
    task = case.task()
    assert task.split == "dev" and task.mode == case.mode and task.hidden_tests == case.tests


@pytest.mark.parametrize("case", CASES, ids=NAMES)
def test_reward_parseability_and_fractions_match_the_catalogue(case: Case) -> None:
    """Host-side parsing only: a case is ``not_parseable`` exactly when the catalogue says the
    reward channel refuses it, and every declared reward is a whole number of checks."""
    parsed = reward.parse_task(case.task())
    refused = case.reward.outcomes == {(reward.NOT_PARSEABLE, None)}
    assert (parsed is None) == refused, case.name
    if parsed is not None:
        for _, value in case.reward.outcomes:
            assert value is not None
            assert value * parsed.total in range(parsed.total + 1), case.name


def test_refused_cases_never_start_a_container() -> None:
    def no_container(_payload: dict[str, Any]) -> reward.ContainerOutcome:
        raise AssertionError("a refused case must not reach the container")

    refused = [case for case in CASES if label_kind(case.reward.label) == "refused"]
    assert refused
    for case in refused:
        result = reward.score_attempt(case.task(), case.code, container=no_container)
        assert (result.status, result.reward) == (reward.NOT_PARSEABLE, None), case.name


@pytest.mark.parametrize("case", CASES, ids=NAMES)
def test_static_audit_flags_are_recorded(case: Case) -> None:
    """The monitoring-only static audit (``reward.audit_attempt``) reads text, never runs it.
    A flag is a signal, not a verdict: ``shadow_builtin_isinstance`` and
    ``monkeypatch_assert_helper`` forge a pass on the path of record without any flag."""
    assert reward.audit_attempt(case.code) == case.audit_flags


def test_known_weaknesses_are_pinned() -> None:
    """Closing (or opening) a weakness must change this list in the same change."""
    assert conformance.known_weaknesses("record") == {
        "zero_check_pass",
        "zero_check_comment_only",
        "zero_check_uncalled_def",
        "marker_forgery_source_file",
        "marker_forgery_code_consts",
        "marker_forgery_frame_walk",
        "marker_forgery_excepthook_exit",
        "marker_forgery_atexit_exit",
        "expected_value_lookup",
        "eq_override_object",
        "eq_override_int_subclass",
        "monkeypatch_assert_helper",
        "shadow_builtin_isinstance",
        "side_effect_stub",
    }
    assert conformance.known_weaknesses("reward") == {"side_effect_stub"}
    with pytest.raises(ValueError):
        conformance.known_weaknesses("benchmark")


def test_unflagged_forgeries_exist_on_the_path_of_record() -> None:
    """The static audit is not a guard: some known weaknesses carry no audit flag at all."""
    unflagged = {
        case.name
        for case in CASES
        if case.name in conformance.known_weaknesses("record") and not case.audit_flags
    }
    assert {"shadow_builtin_isinstance", "monkeypatch_assert_helper"} <= unflagged


# --------------------------------------------------------------------------------------
# Docker-backed tests (opt-in; disposable worker only, inside a passing execution window).
# --------------------------------------------------------------------------------------

integration = pytest.mark.skipif(
    os.environ.get("SCRATCHCODER_SANDBOX_TESTS") != "1",
    reason="requires explicitly enabled disposable Docker worker",
)


@integration
@pytest.mark.parametrize("case", CASES, ids=NAMES)
def test_live_scoring_of_record_matches_the_catalogue(case: Case) -> None:
    result = isolated_execution(case.code, case.tests, case.mode)
    assert not is_host_failure(result), result  # a host failure is a test failure, not a skip
    assert result.status in case.record.statuses, (case.name, result.status, result.stdout)


@integration
@pytest.mark.parametrize("case", CASES, ids=NAMES)
def test_live_reward_channel_matches_the_catalogue(case: Case) -> None:
    result = reward.score_attempt(case.task(), case.code)
    assert not result.is_host_failure, result
    assert (result.status, result.reward) in case.reward.outcomes, (case.name, result)
