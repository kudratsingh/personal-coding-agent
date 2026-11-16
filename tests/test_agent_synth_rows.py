"""Rows from the real harness loop with a scripted policy, the independent-oracle format and
the call ledger. Every sandbox verdict comes from a scripted executor that reads the packed
sources as a literal; nothing generated is executed."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from agent_fixtures import ScriptedExecutor, write_case, write_registry
from synth_fixtures import W1, W3, W5, Chunks, SynthExecutor, claim_errors

from scratchcoder.agent import cases, loop, workspace
from scratchcoder.agent.policy import OraclePolicy
from scratchcoder.agent.synth import independent, instances, rows
from scratchcoder.agent.synth import ledger as call_ledger
from scratchcoder.agent.synth.template import Template
from scratchcoder.data.chat import ASSISTANT_START, PYTHON_END, PYTHON_START
from scratchcoder.data.sft import loss_tokens, row_from_json, row_json

ORACLE = (
    "def inputs(params, rng):\n"
    "    return [(rng.randrange(20),) for _ in range(5)]\n\n\n"
    "def oracle(params, x):\n"
    "    return x * params['k']\n"
)


def _instance(
    tmp_path: Path,
    template: Template = W1,
    bug: str = "factor_up",
    kinds: tuple[str, ...] = ("off_by_one", "wrong_return"),
) -> tuple[instances.Instance, cases.Case, dict[str, Any], Chunks]:
    encoder = Chunks()
    instance = instances.build(template, bug, 0, 0, mutation_kinds=list(kinds), encoder=encoder)
    static = instances.static_check(instance, encoder)
    assert static["problems"] == []
    instances.write(tmp_path, instance, static)
    (tmp_path / cases.REGISTRY_NAME).write_text(
        json.dumps(cases.build_registry(tmp_path)), encoding="utf-8"
    )
    (case,) = cases.load_cases(tmp_path, synthetic=True)
    synth = json.loads((tmp_path / instance.id / "synth.json").read_text(encoding="utf-8"))
    return instance, case, synth, encoder


def _executor(instance: instances.Instance, **options: object) -> SynthExecutor:
    module = instance.module_file(instance.template.target).removesuffix(".py")
    options.setdefault("errors", claim_errors(instance.task_claim))
    return SynthExecutor(
        module,
        instance.name(instance.template.target),
        instance.reference_target,
        instance.designed_failing,
        test_module=instance.public_tests[0].removesuffix(".py"),
        **options,  # type: ignore[arg-type]
    )


def test_an_instance_gives_one_edit_and_one_repair_row_per_mutant(tmp_path: Path) -> None:
    instance, case, synth, encoder = _instance(tmp_path)
    oracle = independent.check_text(ORACLE, "toy1")
    executor = _executor(instance)
    built = rows.build_instance(case, synth, oracle, encoder, encoder, execute=executor)
    assert built.accepted, built.reasons
    assert built.calls == rows.calls_per_instance(2) == len(executor.calls) == 6
    assert [row.id for row in built.rows] == [
        f"agent-sft1/{case.id}/edit",
        f"agent-sft1/{case.id}/repair-1",
        f"agent-sft1/{case.id}/repair-2",
    ]
    reply = [PYTHON_START, *encoder.encode(instance.reference_target), PYTHON_END]
    for row, meta in zip(built.rows, built.meta, strict=True):
        start = row.ids.index(ASSISTANT_START) + 1
        assert list(row.ids[start : start + len(reply)]) == reply
        assert not any(row.mask[:start]) and all(row.mask[start:])
        assert loss_tokens(row) == meta["loss_tokens"] == len(reply) + 1
        assert row_from_json(row_json(row)) == row
        assert meta["reply"] == instance.reference_target
    assert built.meta[0]["mutation_kind"] is None
    assert [meta["mutation_kind"] for meta in built.meta[1:]] == ["off_by_one", "wrong_return"]
    assert "CHANGES SO FAR" not in built.meta[0]["user_text"]
    for meta, mutant in zip(built.meta[1:], instance.mutants, strict=True):
        assert (
            "CHANGES SO FAR" in meta["user_text"]
            and "RESULT run_tests: failed" in meta["user_text"]
        )
        changed = [
            line for line in mutant.source.split("\n") if line not in instance.reference_target
        ]
        assert all("+" + line in meta["user_text"] for line in changed)
    # One grading call holds every check; the original's correctness check runs last.
    grading = executor.calls[4][1]
    assert "# hidden preservation checks" in grading and "_SC_INDEPENDENT = " in grading
    assert repr(ORACLE) in grading
    assert executor.calls[5][1] == case.correctness


def test_the_row_user_turn_is_the_harness_prompt(tmp_path: Path) -> None:
    instance, case, synth, encoder = _instance(tmp_path)
    policy = rows.ScriptedPolicy([instance.reference_target])
    executor = _executor(instance)
    loop.run_instance(
        case, policy, loop.SINGLE, loop.GREEDY, encoder, execute=executor, trace=loop.Trace()
    )
    prompt = policy.prompts[0]
    rendered = rows.render_row(
        encoder, encoder, prompt, instance.reference_target, case=case, files=case.files
    )
    assert rendered.ids[: len(prompt.ids)] == prompt.ids
    with pytest.raises(workspace.EditRefused):
        rows.render_row(
            encoder,
            encoder,
            prompt,
            instance.reference_target.replace(instance.name("scale"), "renamed", 1),
            case=case,
            files=case.files,
        )


def test_a_mutant_that_passes_the_public_tests_rejects_the_instance(tmp_path: Path) -> None:
    instance, case, synth, encoder = _instance(tmp_path)
    executor = _executor(instance, public_passes=[instance.mutants[0].source])
    oracle = independent.check_text(ORACLE, "toy1")
    built = rows.build_instance(case, synth, oracle, encoder, encoder, execute=executor)
    assert not built.accepted and built.rows == []
    assert "mutant_1_passed_public" in built.reasons
    assert built.calls == 3  # F0, the mutant's run, its grading; the original check is saved


def test_an_original_that_passes_correctness_rejects_the_instance(tmp_path: Path) -> None:
    instance, case, synth, encoder = _instance(tmp_path)

    class Lenient(SynthExecutor):
        def __call__(self, program: str, tests: str, mode: str):  # type: ignore[no-untyped-def]
            result = super().__call__(program, tests, mode)
            if tests == case.correctness:
                return type(result)("passed")
            return result

    executor = Lenient(
        instance.module_file("scale").removesuffix(".py"),
        instance.name("scale"),
        instance.reference_target,
        instance.designed_failing,
        test_module=instance.public_tests[0].removesuffix(".py"),
        errors=claim_errors(instance.task_claim),
    )
    oracle = independent.check_text(ORACLE, "toy1")
    built = rows.build_instance(case, synth, oracle, encoder, encoder, execute=executor)
    assert built.reasons == ["original_passes_correctness"]


def test_a_traceback_case_needs_its_exception_in_the_first_observation(tmp_path: Path) -> None:
    instance, case, synth, encoder = _instance(
        tmp_path, W5, "wrong_index", ("wrong_return", "off_by_one")
    )
    traceback = synth["traceback"]
    module_file = instance.module_file("first_item")
    frames = [[instance.public_tests[0], 5, "t"], [module_file, 3, instance.name("first_item")]]
    oracle = independent.check_text(
        "def inputs(params, rng):\n    return [([1, 2],)]\n\n\ndef oracle(params, xs):\n"
        "    return xs[0]\n",
        "toy5",
    )
    good = _executor(instance, frames=frames, error=("IndexError", "list index out of range"))
    assert rows.build_instance(case, synth, oracle, encoder, encoder, execute=good).accepted
    assert traceback["error_line"] == "IndexError: list index out of range"
    wrong = _executor(instance, frames=frames, error=("KeyError", "'x'"))
    built = rows.build_instance(case, synth, oracle, encoder, encoder, execute=wrong)
    assert "traceback_not_observed" in built.reasons


def test_negative_controls_run_the_full_grading_script(tmp_path: Path) -> None:
    instance, case, synth, encoder = _instance(tmp_path)
    oracle = independent.check_text(ORACLE, "toy1")
    executor = _executor(instance)
    found = rows.run_controls(case, synth, oracle, execute=executor)
    assert [row["control"] for row in found] == ["none", "delete", "echo", "constant"]
    assert all(row["ok"] and row["status"] == "failed" for row in found)
    assert all("_SC_INDEPENDENT" in call[1] for call in executor.calls)


def test_grade_checks_default_is_unchanged_and_one_combined_call_is_allowed(tmp_path: Path) -> None:
    write_case(tmp_path / "cases")
    write_registry(tmp_path / "cases")
    (case,) = cases.load_cases(tmp_path / "cases")
    executor = ScriptedExecutor()
    result = loop.run_instance(
        case,
        OraclePolicy(),
        loop.REPAIR_3,
        loop.GREEDY,
        Chunks(),
        execute=executor,
        trace=loop.Trace(),
    )
    assert set(result.grade) == {"public", "correctness", "preservation"} and result.solved
    executor = ScriptedExecutor()
    result = loop.run_instance(
        case,
        OraclePolicy(),
        loop.REPAIR_3,
        loop.GREEDY,
        Chunks(),
        execute=executor,
        trace=loop.Trace(),
        grade_checks=("correctness",),
    )
    assert set(result.grade) == {"public", "correctness"} and result.sandbox_calls == 3
    for bad in ((), ("correctness", "correctness"), ("other",)):
        with pytest.raises(ValueError, match="grade_checks"):
            loop.run_instance(
                case,
                OraclePolicy(),
                loop.SINGLE,
                loop.GREEDY,
                Chunks(),
                execute=executor,
                trace=loop.Trace(),
                grade_checks=bad,
            )


def test_oracle_files_are_read_as_text_and_checked_statically(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    (tmp_path / "toy1.py").write_text(
        ORACLE + f"\nopen({str(marker)!r}, 'w').write('x')\n", encoding="utf-8"
    )
    loaded = independent.load(tmp_path, "toy1")
    assert not marker.exists()  # Loading never runs the file.
    assert loaded.sha256 and not loaded.has_properties
    block = independent.block(loaded, "toolbox", "scale", {"k": 3}, 11)
    first = ast.parse(block).body[0]
    assert isinstance(first, ast.Assign) and isinstance(first.targets[0], ast.Name)
    assert first.targets[0].id == "_SC_INDEPENDENT"
    assert "from toolbox import scale as _sc_target" in block
    assert "_sc_random.Random(11)" in block and "_sc_random.Random(12)" in block
    for text, problem in (
        ("def inputs(params, rng):\n    return []\n", "lacks oracle"),
        ("import numpy\n" + ORACLE, "non-stdlib"),
        ("from . import x\n" + ORACLE, "relative import"),
        ("def oracle(:\n", "does not parse"),
    ):
        with pytest.raises(independent.OracleError, match=problem):
            independent.check_text(text, "toy1")
    with pytest.raises(independent.OracleError, match="no oracle file"):
        independent.load(tmp_path, "toy9")


def _write_sums(directory: Path, *, skip: str = "", extra: str = "") -> str:
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(directory.iterdir())
        if path.name not in {independent.SUMS, skip}
    ]
    (directory / independent.SUMS).write_text("".join(lines) + extra, encoding="utf-8")
    return hashlib.sha256((directory / independent.SUMS).read_bytes()).hexdigest()


def test_the_independent_directory_is_pinned_by_its_sums_file(tmp_path: Path) -> None:
    (tmp_path / "toy1.py").write_text(ORACLE, encoding="utf-8")
    (tmp_path / "README.md").write_text("notes\n", encoding="utf-8")
    with pytest.raises(independent.OracleError, match="no SHA256SUMS"):
        independent.pin(tmp_path)
    digest = _write_sums(tmp_path)
    pinned = independent.pin(tmp_path, expected=digest)
    assert pinned.sums_sha256 == digest and set(pinned.files) == {"toy1.py", "README.md"}
    assert independent.load(tmp_path, "toy1", pinned).sha256 == pinned.files["toy1.py"]
    with pytest.raises(independent.OracleError, match="not the pinned"):
        independent.pin(tmp_path, expected="0" * 64)
    (tmp_path / "toy1.py").write_text(ORACLE + "\n# changed\n", encoding="utf-8")
    with pytest.raises(independent.OracleError, match="does not match its SHA256SUMS"):
        independent.pin(tmp_path)
    # A file changed after the pin was taken is refused when it is loaded.
    with pytest.raises(independent.OracleError, match="pinned SHA256SUMS"):
        independent.load(tmp_path, "toy1", pinned)
    _write_sums(tmp_path, skip="README.md")
    with pytest.raises(independent.OracleError, match="does not list README.md"):
        independent.pin(tmp_path)
    _write_sums(tmp_path, extra=f"{'a' * 64}  toy2.py\n")
    with pytest.raises(independent.OracleError, match="lists missing toy2.py"):
        independent.pin(tmp_path)
    for bad in ("nothex  toy1.py\n", f"{'a' * 64}  ../toy1.py\n"):
        (tmp_path / independent.SUMS).write_text(bad, encoding="utf-8")
        with pytest.raises(independent.OracleError, match="malformed"):
            independent.pin(tmp_path)


def test_the_ledger_counts_before_each_call_and_refuses_over_the_cap(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    book = call_ledger.Ledger.open(path, cap=3)
    assert book.add_prior("earlier/validation.json", 1, {"sha256": "x"})
    assert not book.add_prior("earlier/validation.json", 1, {"sha256": "x"})
    seen: list[str] = []

    def inner(program: str, tests: str, mode: str):  # type: ignore[no-untyped-def]
        seen.append(program)
        from scratchcoder.sandbox.runner import Result

        return Result("passed")

    counted = book.execute(inner, run="r1", context=lambda: "inst-1")
    counted("a", "t", "function")
    counted("b", "t", "function")
    with pytest.raises(call_ledger.CallCapReached):
        counted("c", "t", "function")
    assert seen == ["a", "b"]
    again = call_ledger.Ledger.open(path, cap=3)
    assert again.total == 3 and again.remaining == 0
    assert again.summary()["by_run"] == {"r1": 2} and again.summary()["prior"] == 1
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["prior", "call", "call"]
    assert lines[1]["context"] == "inst-1" and "program_sha256" in lines[1]


def test_prior_calls_are_counted_from_run_records(tmp_path: Path) -> None:
    run = tmp_path / "validation-run"
    run.mkdir()
    (run / "validation.json").write_text(
        json.dumps({"cases": [{"calls": [{"host_failures": 0}, {"host_failures": 2}]}]})
    )
    (run / "scorecard.json").write_text(
        json.dumps({"instances": [{"sandbox_calls": 4, "host_failures": 1}]})
    )
    controls = tmp_path / "controls-run"
    controls.mkdir()
    (controls / "controls.json").write_text(
        json.dumps({"sandbox_calls": 5, "rows": [{"host_failures": 1}, {}]})
    )
    book = call_ledger.Ledger.open(tmp_path / "ledger.jsonl")
    added = call_ledger.prior_from_records(book, [run, controls])
    assert [(row["source"], row["calls"]) for row in added] == [
        ("validation-run/validation.json", 4),
        ("validation-run/scorecard.json", 5),
        ("controls-run/controls.json", 6),
    ]
    assert call_ledger.prior_from_records(book, [run, controls]) == []
    assert book.total == 15
    # A second record directory with the same base name under another parent is counted;
    # a byte-for-byte copy of a counted record is skipped and reported, not counted twice.
    other = tmp_path / "elsewhere" / "validation-run"
    other.mkdir(parents=True)
    (other / "validation.json").write_text(
        json.dumps({"cases": [{"calls": [{"host_failures": 1}]}]})
    )
    (other / "scorecard.json").write_bytes((run / "scorecard.json").read_bytes())
    skipped: list[dict[str, object]] = []
    added = call_ledger.prior_from_records(book, [other], skipped=skipped)
    assert [(row["source"], row["calls"]) for row in added] == [
        ("validation-run/validation.json", 2)
    ]
    assert [row["reason"] for row in skipped] == [f"copy of {(run / 'scorecard.json').resolve()}"]
    assert book.total == 17


def test_an_executing_run_needs_a_seeded_ledger_the_approved_cap_and_the_lock(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    with pytest.raises(call_ledger.LedgerError, match="approved"):
        call_ledger.Ledger.open(path, cap=call_ledger.DEFAULT_CAP + 1)
    with call_ledger.Ledger.locked(path) as book:
        with pytest.raises(call_ledger.LedgerError, match="does not exist"):
            book.require_seeded()
    path.write_text("")
    with call_ledger.Ledger.locked(path) as book:
        with pytest.raises(call_ledger.LedgerError, match="no prior entries"):
            book.require_seeded()
        book.add_prior("earlier/validation.json", 3, {"sha256": "y", "path": "/p"})
        book.require_seeded()
        # A second run cannot take the ledger while the first holds it.
        with pytest.raises(call_ledger.LedgerError, match="in use"):
            with call_ledger.Ledger.locked(path):
                pass
    with call_ledger.Ledger.locked(path) as book:
        assert book.total == 3


def test_a_task_claim_that_the_observation_contradicts_rejects_the_instance(
    tmp_path: Path,
) -> None:
    instance, case, synth, encoder = _instance(tmp_path)
    oracle = independent.check_text(ORACLE, "toy1")
    assert synth["task_claim"] == instance.task_claim and instance.task_claim
    # The original really returns something other than the value the task statement names.
    claim = instance.task_claim
    wrong = _executor(instance, errors={claim["test"]: ("AssertionError", claim["got"] + "1")})
    built = rows.build_instance(case, synth, oracle, encoder, encoder, execute=wrong)
    assert "task_claim_not_observed" in built.reasons and built.rows == []
    assert built.calls == 5  # F0, two mutants, the reference, G; O is not spent.
    right = _executor(instance)
    assert rows.build_instance(case, synth, oracle, encoder, encoder, execute=right).accepted
    # A cross-module task is checked the same way, on the caller's test.
    other = tmp_path / "w3"
    other.mkdir()
    instance, case, synth, encoder = _instance(other, W3, "factor_up")
    assert synth["task_claim"]["test"].startswith("test_" + instance.name("scale_all"))
    assert rows.build_instance(
        case, synth, oracle, encoder, encoder, execute=_executor(instance)
    ).accepted


def test_observation_blocks_keep_each_failing_test_apart() -> None:
    observation = "\n".join(
        [
            "RESULT run_tests: failed (2 of 3 tests)",
            "FAIL test_a (test_m.py:4)",
            "    got = f([])",
            "IndexError: list index out of range",
            "FAIL test_b (test_m.py:8)",
            "    got = f([1])",
            "KeyError: 'x'",
            "  in m.py:3 head: return xs[1]",
            "(+1 more failing)",
        ]
    )
    blocks = rows.failure_blocks(observation)
    assert list(blocks) == ["test_a", "test_b"]
    traceback = {
        "error_line": "IndexError: list index out of range",
        "raise_file": "m.py",
        "raise_function": "head",
        "test": "test_a",
    }
    # The error comes from test_a, the frame from test_b: not one observed traceback.
    assert not rows._traceback_seen(observation, traceback)
    assert rows._traceback_seen(
        observation, {**traceback, "test": "test_b", "error_line": "KeyError: 'x'"}
    )
    assert rows.claim_seen(observation, {"test": "test_b", "raised": "KeyError"})
    assert not rows.claim_seen(observation, {"test": "test_a", "raised": "KeyError"})
    assert not rows.claim_seen(observation, {"test": "test_c", "got": "1"})
    long = "x" * 500
    cut = f"FAIL test_a (t.py:1)\n    got = f()\nAssertionError: {long[:400]}"
    assert rows.claim_seen(cut, {"test": "test_a", "got": long})
    assert not rows.claim_seen(cut, {"test": "test_a", "got": "y" + long})
