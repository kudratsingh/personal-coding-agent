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
        --image-id <recorded image ID> --dotenv <absolute path to .env> \\
        --receipt <run directory>/window-receipt.json -- \\
        env SCRATCHCODER_SANDBOX_TESTS=1 \\
        SCRATCHCODER_CONFORMANCE_RECORD=<run directory>/observed-verdicts.jsonl \\
        uv run --locked pytest tests/test_evaluator_conformance.py

When ``SCRATCHCODER_CONFORMANCE_RECORD`` names a file (keep it outside the repository), every
gated test appends one JSON row per observed verdict, next to the declared one, before it
asserts anything, so a mismatch is recorded as well as reported.
"""

import ast
import json
import os
import time
from typing import Any

import pytest

from scratchcoder.eval.harness import is_host_failure, isolated_execution
from scratchcoder.sandbox import conformance, reward
from scratchcoder.sandbox.conformance import (
    CASES,
    FAILURE_CLASSES,
    FALSE_NEGATIVE_CAUSES,
    LABEL_KINDS,
    RECORD_STATUSES,
    REQUIRED_CASES,
    UNCERTAIN_CASES,
    Case,
    false_negative_cause,
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
    assert conformance.CATALOGUE_VERSION == "evaluator-conformance-v3"
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
    if kind == "known_false_negative":
        assert case.solves and "passed" not in statuses, case.name
        assert false_negative_cause(case.record.label) in FALSE_NEGATIVE_CAUSES, case.name


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
    if kind == "known_false_negative":
        assert case.solves and all(value is not None and value < 1.0 for value in values)
        assert false_negative_cause(case.reward.label) in FALSE_NEGATIVE_CAUSES, case.name


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
        "zc_candidate_independent_checks",
        "trace_hook_line_jump",
        "truthy_eq_foreign_compare",
        "shadow_builtin_is_true_form",
        "monkeypatch_helper_is_true_form",
        "builtins_module_patch_is_true_form",
        "se_raise_unchecked_stub",
    }
    assert conformance.known_weaknesses("reward") == {
        "side_effect_stub",
        "zc_candidate_independent_checks",
        "shadow_builtin_is_true_form",
        "monkeypatch_helper_is_true_form",
        "builtins_module_patch_is_true_form",
        "se_raise_unchecked_stub",
    }
    with pytest.raises(ValueError):
        conformance.known_weaknesses("benchmark")


def test_false_negatives_name_their_cause() -> None:
    """A rejected honest solution says whether the task's tests are defective (no scorer can
    fix that) or the scoring path rejects a solution the tests would accept."""
    causes = {
        (case.name, path): false_negative_cause(label)
        for case in CASES
        for path, label in (("record", case.record.label), ("reward", case.reward.label))
        if label_kind(label) == "known_false_negative"
    }
    assert causes == {
        ("contract_rounding_faithful", "record"): "defective_tests",
        ("contract_rounding_faithful", "reward"): "defective_tests",
        ("refdep_set_order_sorted", "record"): "defective_tests",
        ("refdep_set_order_sorted", "reward"): "defective_tests",
        ("truthy_eq_foreign_honest", "reward"): "evaluator_rejects_honest",
    }
    assert false_negative_cause("control") is None
    assert false_negative_cause("known_false_negative") is None


def test_every_failure_class_has_cases() -> None:
    assert set(FAILURE_CLASSES) == {
        "contradictory_contract",
        "reference_dependent_tests",
        "nondeterministic_reference",
        "missing_side_effect_assertion",
        "zero_check_admission",
        "marker_forgery",
    }
    names = set(NAMES)
    for failure_class, members in FAILURE_CLASSES.items():
        assert members and members <= names, failure_class


def test_uncertain_predictions_are_declared_not_hidden() -> None:
    """An uncertain prediction still declares one outcome per path; it is never disguised as
    ``nondeterministic``, and its reason says it is uncertain."""
    assert set(UNCERTAIN_CASES) <= REQUIRED_CASES
    by_name = conformance.by_name()
    for name in UNCERTAIN_CASES:
        case = by_name[name]
        assert len(case.record.statuses) == 1 and len(case.reward.outcomes) == 1, name
        assert "prediction uncertain" in case.record.label + case.reward.label, name
    for case in CASES:
        if case.name not in UNCERTAIN_CASES:
            assert "prediction uncertain" not in case.record.label + case.reward.label


def test_dedicated_live_cases_are_well_formed() -> None:
    ast.parse(conformance.PID1_STDOUT_INJECTION)  # parsed only
    assert "passed" not in conformance.PID1_ALLOWED_STATUSES
    witness, runs, control, control_runs = conformance.HASH_ORDER_WITNESS
    by_name = conformance.by_name()
    assert label_kind(by_name[witness].record.label) == "nondeterministic"
    assert by_name[control].solves and by_name[control].tests == by_name[witness].tests
    assert (runs, control_runs) == (12, 6)  # fixed in advance; never raised to chase an outcome


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


RECORD_ENV = "SCRATCHCODER_CONFORMANCE_RECORD"
STDOUT_RECORDED = frozenset(
    {"io_supervisor_introspection", "record_pid1_stdout_injection", "trace_hook_line_jump"}
)
DETAIL_BYTES = 300
STDOUT_BYTES = 200


def _observe(
    test: str,
    case: str,
    path: str,
    declared: list[Any],
    status: str,
    value: float | None,
    elapsed: float,
    stdout: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one observed verdict, next to the declared one, to the record file (if any)."""
    target = os.environ.get(RECORD_ENV)
    if not target:
        return
    row: dict[str, Any] = {
        "catalogue": conformance.CATALOGUE_VERSION,
        "test": test,
        "case": case,
        "path": path,
        "declared": declared,
        "status": status,
        "reward": value,
        "elapsed_seconds": round(elapsed, 3),
    }
    row.update(extra or {})
    if stdout is not None and case in STDOUT_RECORDED:
        row["stdout"] = stdout.encode("utf-8", "replace")[:STDOUT_BYTES].decode("utf-8", "replace")
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _run_record(test: str, case: Case) -> str:
    started = time.monotonic()
    result = isolated_execution(case.code, case.tests, case.mode)
    _observe(
        test,
        case.name,
        "record",
        sorted(case.record.statuses),
        result.status,
        None,
        time.monotonic() - started,
        result.stdout,
        {"exit_code": result.exit_code, "host_stalled": result.host_stalled},
    )
    assert not is_host_failure(result), result  # a host failure is a test failure, not a skip
    return result.status


def _run_reward(test: str, case: Case) -> tuple[str, float | None]:
    started = time.monotonic()
    result = reward.score_attempt(case.task(), case.code)
    _observe(
        test,
        case.name,
        "reward",
        sorted([list(outcome) for outcome in case.reward.outcomes], key=str),
        result.status,
        result.reward,
        time.monotonic() - started,
        None,
        {
            "passed": result.passed,
            "total": result.total,
            "unfinished": result.unfinished,
            "outcomes": list(result.outcomes),
            "detail": result.detail.encode("utf-8", "replace")[:DETAIL_BYTES].decode(
                "utf-8", "replace"
            ),
        },
    )
    assert not result.is_host_failure, result
    return result.status, result.reward


@integration
@pytest.mark.parametrize("case", CASES, ids=NAMES)
def test_live_scoring_of_record_matches_the_catalogue(case: Case) -> None:
    status = _run_record("catalogue", case)
    assert status in case.record.statuses, (case.name, status)  # details are in the record row


@integration
@pytest.mark.parametrize("case", CASES, ids=NAMES)
def test_live_reward_channel_matches_the_catalogue(case: Case) -> None:
    outcome = _run_reward("catalogue", case)
    assert outcome in case.reward.outcomes, (case.name, outcome)


@integration
def test_live_hash_order_witness() -> None:
    """String-hash randomisation is live in the pinned runtime: the same program, tests and
    mode give both verdicts across fresh containers, while a sorted honest solution of the
    same task gives one. Passing this does not prove that any task is deterministic."""
    witness_name, runs, control_name, control_runs = conformance.HASH_ORDER_WITNESS
    witness = conformance.by_name()[witness_name]
    control = conformance.by_name()[control_name]
    record_seen = {_run_record("hash_order_witness", witness) for _ in range(runs)}
    reward_seen = {_run_reward("hash_order_witness", witness) for _ in range(runs)}
    control_record = {_run_record("hash_order_witness", control) for _ in range(control_runs)}
    control_reward = {_run_reward("hash_order_witness", control) for _ in range(control_runs)}
    assert control_record == {"passed"} and control_reward == {(reward.COMPLETED, 1.0)}
    assert record_seen == {"passed", "failed"}, ("randomisation not observed", record_seen)
    assert reward_seen == {(reward.COMPLETED, 1.0), (reward.COMPLETED, 0.0)}, (
        "randomisation not observed",
        reward_seen,
    )


@integration
def test_live_record_pid1_stdout_injection() -> None:
    """A candidate that writes a forged result line to the path-of-record supervisor's stdout
    must never be scored ``passed``. Run once, with no harness retries."""
    started = time.monotonic()
    result = isolated_execution(
        conformance.PID1_STDOUT_INJECTION, conformance.ADD_TESTS, "function"
    )
    _observe(
        "pid1_stdout_injection",
        "record_pid1_stdout_injection",
        "record",
        sorted(conformance.PID1_ALLOWED_STATUSES),
        result.status,
        None,
        time.monotonic() - started,
        result.stdout,
        {"exit_code": result.exit_code, "host_stalled": result.host_stalled},
    )
    assert result.status != "passed", result
    assert result.status in conformance.PID1_ALLOWED_STATUSES, result
