"""Agent harness v0 end to end with scripted executors and fake policies: the loop, its
failure classes, case loading and validation, the scorecard and the dev-run tool. No
workspace or model code is executed anywhere in these tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from agent_fixtures import (
    CORRECTNESS,
    FIXED_DOUBLE,
    PRESERVATION,
    ScriptedExecutor,
    write_case,
    write_registry,
)
from test_agent_workspace import Bytes

from scratchcoder.agent import cases as agent_cases
from scratchcoder.agent import loop, scorecard, validation, workspace
from scratchcoder.agent.cases import Case
from scratchcoder.agent.policy import Generate, ModelPolicy, OraclePolicy, turn_seed
from scratchcoder.agent.prompt import Prompt
from scratchcoder.data.chat import PYTHON_END, PYTHON_START
from scratchcoder.sandbox import runner


@pytest.fixture
def case(tmp_path: Path) -> Case:
    write_case(tmp_path)
    write_registry(tmp_path)
    return agent_cases.load_cases(tmp_path)[0]


class Scripted:
    """A policy that replies with the given code blocks in turn (None = no block)."""

    name = "scripted"

    def __init__(self, replies: Sequence[str | None]) -> None:
        self.replies = list(replies)
        self.prompts: list[Prompt] = []

    def propose(
        self,
        case: Case,
        turn: int,
        prompt: Prompt,
        max_new_tokens: int,
        decoding: loop.Decoding,
        files: Mapping[str, str],
    ) -> loop.Proposal:
        self.prompts.append(prompt)
        code = self.replies.pop(0)
        return loop.Proposal(code, code or "", 10, "stop", 0.25)


def run(
    case: Case,
    policy: loop.Policy,
    condition: loop.Condition = loop.REPAIR_3,
    executor: ScriptedExecutor | None = None,
) -> tuple[loop.InstanceResult, loop.Trace, ScriptedExecutor]:
    executor = executor or ScriptedExecutor()
    trace = loop.Trace()
    result = loop.run_instance(
        case,
        policy,
        condition,
        loop.GREEDY,
        Bytes(),
        execute=executor,
        trace=trace,
        sleep=lambda _: None,
    )
    return result, trace, executor


# ----------------------------------------------------------------------------- the loop


def test_the_oracle_solves_the_case_through_the_full_loop(case: Case) -> None:
    result, trace, executor = run(case, OraclePolicy())
    assert result.solved and result.scored and result.failure_class is None
    assert result.ended == "public_passed" and len(result.turns) == 1
    assert result.grade == {"public": "passed", "correctness": "passed", "preservation": "passed"}
    assert result.touched == ["arith.py"] and "+    return x + x\n" in result.diff
    # Feedback before the edit, feedback after it, correctness, preservation.
    assert [tests for _, tests, _ in executor.calls][2:] == [CORRECTNESS, PRESERVATION]
    assert all(mode == "function" for _, _, mode in executor.calls)
    assert result.feedback_before.startswith("RESULT run_tests: failed (2 of 2 tests)")
    assert result.feedback_after.startswith("RESULT run_tests: passed")
    assert [event["event"] for event in trace.events] == [
        "start",
        "sandbox",
        "feedback",
        "prompt",
        "generation",
        "edit",
        "sandbox",
        "feedback",
        "sandbox",
        "sandbox",
        "end",
    ]


def test_hidden_checks_never_reach_a_prompt_or_the_trace(case: Case) -> None:
    policy = Scripted(["def triple(x):\n    return x", None, FIXED_DOUBLE])
    result, trace, _ = run(case, policy)
    assert result.solved
    text = json.dumps(trace.events) + "".join(prompt.text for prompt in policy.prompts)
    hidden = [line for line in (CORRECTNESS + PRESERVATION).splitlines() if "assert" in line]
    assert hidden and not any(line in text for line in hidden)
    assert "double(3)" not in text and "clamp(5, 0, 3)" not in text


def test_repair_retries_after_refusals_and_single_stops_after_one_turn(case: Case) -> None:
    replies = ["def triple(x):\n    return x", None, FIXED_DOUBLE]
    repaired, trace, _ = run(case, Scripted(list(replies)))
    assert repaired.solved and [turn.edit for turn in repaired.turns] == [
        "unknown_name",
        "no_python_block",
        "applied",
    ]
    second = [event for event in trace.events if event["event"] == "prompt"][1]
    assert "EDIT refused: the function is not one of the functions shown" in second["text"]
    single, _, _ = run(case, Scripted(list(replies)), loop.SINGLE)
    assert not single.solved and single.failure_class == "tool_interface"
    assert single.ended == "turns_exhausted" and len(single.turns) == 1
    assert single.grade == {"public": "failed", "correctness": "not_run_null_patch"}


def test_changes_so_far_are_cumulative_against_the_original(case: Case) -> None:
    wrong = 'def double(x):\n    """Twice ``x``."""\n    return x * 3'
    scaled = "def scaled(x, low, high):\n    return clamp(double(x), low, high) + 0"
    policy = Scripted([wrong, scaled, FIXED_DOUBLE])
    result, _, _ = run(case, policy)
    assert len(policy.prompts) == 3
    assert "CHANGES SO FAR\narith.py:\n-    return x + x + 1\n+    return x * 3" in (
        policy.prompts[1].text
    )
    assert "    return x * 3" in policy.prompts[1].text.split("SNIPPET arith.py::double")[1]
    # After the second edit the block holds both edits, each against the original (here
    # over the byte encoder's 100-token budget, so the third prompt drops it, logged).
    assert result.turns[1].edit == "applied"
    assert "previous_dropped" in policy.prompts[2].truncations
    files = dict(case.files)
    for source, name in ((wrong, "double"), (scaled, "scaled")):
        edit = workspace.Edit("arith.py", name, source)
        files = workspace.apply_function(files, edit, case.allowed_files)
    previous = loop.blocks_for(case, files, [], "", case.files).previous
    assert "+    return x * 3" in previous
    assert "+    return clamp(double(x), low, high) + 0" in previous
    assert "PREVIOUS ATTEMPT" not in "".join(prompt.text for prompt in policy.prompts)


@pytest.mark.parametrize(
    ("replies", "failure_class"),
    [
        ([None, None, None], "formatting"),
        (["def double(x)", None, "def ("], "formatting"),
        (["def triple(x):\n    return 1"] * 3, "tool_interface"),
        (["def scaled(x, low, high):\n    return low"] * 3, "action_selection"),
        (['def double(x):\n    """Twice."""\n    return 2 * x + 2'] * 3, "model_capability"),
    ],
)
def test_failure_classes(case: Case, replies: list[str | None], failure_class: str) -> None:
    # Both public tests fail, so the prompt shows scaled and double (clamp is over budget).
    result, _, _ = run(case, Scripted(replies))
    assert not result.solved and result.failure_class == failure_class


def test_public_pass_but_preservation_failure_is_unsolved(case: Case) -> None:
    result, _, _ = run(case, OraclePolicy(), executor=ScriptedExecutor(preservation_passes=False))
    assert not result.solved and result.failure_class == "model_capability"
    assert result.grade["preservation"] == "failed"


def test_host_failures_are_retried_and_a_persistent_one_is_unscored(case: Case) -> None:
    retried, _, _ = run(case, OraclePolicy(), executor=ScriptedExecutor(host_failures=2))
    assert retried.solved and retried.host_failures == 2
    lost, _, _ = run(case, OraclePolicy(), executor=ScriptedExecutor(host_failures=99))
    assert not lost.scored and not lost.solved and lost.failure_class == "grading"
    assert lost.ended == "unresolved_feedback"


def test_a_prompt_that_cannot_fit_ends_as_a_context_failure(tmp_path: Path) -> None:
    write_case(tmp_path, task="word " * 200)
    write_registry(tmp_path)
    long_case = agent_cases.load_cases(tmp_path)[0]
    result, _, _ = run(long_case, OraclePolicy())
    assert result.ended == "context" and result.failure_class == "context" and not result.turns


def test_the_generation_budget_caps_and_ends_turns(case: Case) -> None:
    tight = Case(**{**case.__dict__, "budgets": agent_cases.Budgets(max_generated_tokens=15)})
    policy = Scripted([None, None, None])
    result, _, _ = run(tight, policy)
    assert result.ended == "budget_tokens" and len(result.turns) == 2
    assert [turn.generation_cap for turn in result.turns] == [15, 5]


def test_model_policy_seeds_caches_and_extracts_the_first_closed_block(case: Case) -> None:
    calls: list[tuple[int, float, int | None]] = []
    code = list(FIXED_DOUBLE.encode())

    def generate(
        ids: Sequence[int], cap: int, temperature: float, top_p: float, seed: int | None
    ) -> tuple[list[int], str]:
        calls.append((cap, temperature, seed))
        return [PYTHON_START, *code, PYTHON_END], "stop"

    policy = ModelPolicy(generate, Bytes(), clock=iter([0.0, 2.0, 5.0, 9.0]).__next__)
    sample = loop.Decoding("sample-1", 1.0, 1.0, 1)
    single, _, _ = run_with(case, policy, loop.SINGLE, sample)
    repair, _, _ = run_with(case, policy, loop.REPAIR_3, sample)
    assert single.solved and repair.solved
    assert calls == [(324, 1.0, turn_seed(case.id, 1, 1))]  # The repair turn 1 is a cache hit.
    assert repair.turns[0].cached and repair.turns[0].generation_seconds == 0.0
    assert single.generation_seconds == 2.0
    # The reused generation's time is attributed, so the conditions stay comparable.
    assert repair.turns[0].cached_seconds == 2.0 and repair.cached_generation_seconds == 2.0
    card = scorecard.build([single, repair], {})
    assert card["groups"]["model/repair-3"]["all"]["attributed_generation_seconds"] == 2.0
    assert card["groups"]["model/repair-3"]["all"]["generation_seconds"] == 0.0
    assert card["groups"]["model/single"]["all"]["attributed_generation_seconds"] == 2.0
    assert "| 0.0 (+2.0) |" in scorecard.markdown(card)
    greedy, _, _ = run_with(case, policy, loop.SINGLE, loop.GREEDY)
    assert calls[-1] == (324, 0.0, None) and greedy.solved


def run_with(
    case: Case, policy: loop.Policy, condition: loop.Condition, decoding: loop.Decoding
) -> tuple[loop.InstanceResult, loop.Trace, ScriptedExecutor]:
    executor, trace = ScriptedExecutor(), loop.Trace()
    result = loop.run_instance(
        case, policy, condition, decoding, Bytes(), execute=executor, trace=trace
    )
    return result, trace, executor


# ----------------------------------------------------------------------------- cases


def test_registry_hashes_protect_every_case_file(tmp_path: Path) -> None:
    directory = write_case(tmp_path)
    write_registry(tmp_path)
    assert [case.id for case in agent_cases.load_cases(tmp_path)] == ["X01"]
    (directory / "hidden" / "correctness.py").write_text("assert True\n", encoding="utf-8")
    with pytest.raises(agent_cases.CaseError, match="changed since registration"):
        agent_cases.load_cases(tmp_path)
    write_registry(tmp_path)
    (directory / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(agent_cases.CaseError, match="differ from the registry"):
        agent_cases.load_cases(tmp_path)


def test_held_out_locations_are_refused(tmp_path: Path) -> None:
    sealed = tmp_path / "heldout"
    sealed.mkdir()
    with pytest.raises(agent_cases.CaseError, match="held-out"):
        agent_cases.load_cases(sealed)


def test_a_reference_that_changes_code_outside_its_targets_is_refused(tmp_path: Path) -> None:
    from agent_fixtures import REFERENCE

    write_case(tmp_path, reference_text=REFERENCE.replace("Arithmetic helpers", "Changed"))
    with pytest.raises(agent_cases.CaseError, match="outside its target"):
        agent_cases.load_case(next(tmp_path.iterdir()))


def test_validation_runs_original_reference_and_controls(case: Case) -> None:
    executor = ScriptedExecutor()
    report = validation.validate_case(case, execute=executor)
    assert report["valid"] and report["resolved"]
    labels = [(call["label"], call["kind"], call["status"]) for call in report["calls"]]
    assert labels[:3] == [
        ("original", "public", "failed"),
        ("original", "correctness", "failed"),
        ("original", "preservation", "passed"),
    ]
    assert sum(label.startswith("reference-") for label, _, _ in labels) == 9
    assert labels[-2:] == [
        ("control-none:arith.py::double", "correctness", "failed"),
        ("control-echo:arith.py::double", "correctness", "failed"),
    ]
    assert validation.control_source(case.files["arith.py"], "arith.py", "double", "echo") == (
        "def double(x):\n    return x"
    )
    broken = validation.validate_case(case, execute=ScriptedExecutor(preservation_passes=False))
    assert not broken["valid"]


# ----------------------------------------------------------------------------- scorecard


def test_the_scorecard_counts_with_denominators_and_keeps_unscored_apart(case: Case) -> None:
    solved, _, _ = run(case, OraclePolicy())
    lost, _, _ = run(case, OraclePolicy(), executor=ScriptedExecutor(host_failures=99))
    failed, _, _ = run(case, Scripted([None, None, None]))
    for result, label in ((solved, "greedy"), (lost, "sample-0"), (failed, "sample-1")):
        result.decoding = label
        result.policy = "model"
    card = scorecard.build([solved, lost, failed], {"x": 1})
    cell = card["groups"]["model/repair-3"]["per_case"]["X01"]
    assert cell["greedy"]["solved"] == 1 and cell["greedy"]["scored"] == 1
    assert cell["sampled"] == {**cell["sampled"], "scored": 1, "solved": 0, "unscored": 1}
    assert cell["sampled"]["failure_classes"] == {"formatting": 1}
    text = scorecard.markdown(card)
    assert (
        "| model/repair-3 | X01 | 0/1 | 1/1 | formatting 1 of 2 scored (unscored 1 of 3) |" in text
    )


# ----------------------------------------------------------------------------- the tool


def receipt(path: Path, argv: Sequence[str], **changes: object) -> None:
    body: dict[str, object] = {
        "kind": "mac-execution-window-check",
        "status": "passed",
        "checked_at": datetime.now(UTC).isoformat(),
        "command": ["uv", "run", "python", "-m", "tools.agent_dev_run", *argv],
        "checks": {
            "image": {
                "passed": True,
                "image": runner.IMAGE,
                "expected_id": "sha256:" + "3" * 64,
                "found_id": "sha256:" + "3" * 64,
            }
        },
    }
    body.update(changes)
    path.write_text(json.dumps(body), encoding="utf-8")


def fake_loader(
    replies: list[list[int]],
) -> Callable[[Path, Path, int], tuple[Generate, Bytes, Bytes, int]]:
    def load(checkpoint: Path, tokenizer: Path, threads: int) -> tuple[Generate, Bytes, Bytes, int]:
        def generate(
            ids: Sequence[int], cap: int, temperature: float, top_p: float, seed: int | None
        ) -> tuple[list[int], str]:
            return replies[0], "stop"

        return generate, Bytes(), Bytes(), threads

    return load


def tool_args(tmp_path: Path, out: Path, receipt_path: Path) -> list[str]:
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    import hashlib

    return [
        "run",
        "--cases",
        str(tmp_path / "cases"),
        "--tokenizer",
        str(tokenizer),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        hashlib.sha256(b"weights").hexdigest(),
        "--out",
        str(out),
        "--receipt",
        str(receipt_path),
        "--seeds",
        "1",
        "--skip-oracle",
    ]


def test_the_tool_refuses_without_this_commands_window_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import agent_dev_run

    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    write_case(tmp_path / "cases")
    write_registry(tmp_path / "cases")
    executor = ScriptedExecutor()
    path = tmp_path / "receipt.json"
    argv = tool_args(tmp_path, tmp_path / "out", path)
    receipt(path, ["something", "else"])
    loader = fake_loader([[PYTHON_START, *FIXED_DOUBLE.encode(), PYTHON_END]])
    assert agent_dev_run.main(argv, execute=executor, model_loader=loader) == 2
    receipt(path, argv, status="failed")
    assert agent_dev_run.main(argv, execute=executor, model_loader=loader) == 2
    assert not executor.calls and not (tmp_path / "out").exists()
    (tmp_path / "out").mkdir()
    receipt(path, argv)
    assert agent_dev_run.main(argv, execute=executor, model_loader=loader) == 2
    assert not executor.calls


def test_the_tool_validates_runs_and_writes_traces_packets_and_scorecard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import agent_dev_run

    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    write_case(tmp_path / "cases")
    write_registry(tmp_path / "cases")
    out = tmp_path / "out"
    path = tmp_path / "receipt.json"
    argv = [arg for arg in tool_args(tmp_path, out, path) if arg != "--skip-oracle"]
    receipt(path, argv)
    executor = ScriptedExecutor()
    loader = fake_loader([[PYTHON_START, *FIXED_DOUBLE.encode(), PYTHON_END]])
    assert agent_dev_run.main(argv, execute=executor, model_loader=loader) == 0
    card = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
    assert set(card["groups"]) == {"model/repair-3", "model/single", "oracle/repair-3"}
    assert card["groups"]["oracle/repair-3"]["all"]["solved"] == 1
    model = card["groups"]["model/single"]["per_case"]["X01"]
    assert model["greedy"]["solved"] == 1 and model["sampled"]["solved"] == 1
    assert card["identity"]["runtime_image_id"] == "sha256:" + "3" * 64
    assert card["identity"]["generation_cache_hits"] == 2
    traces = sorted(p.name for p in (out / "traces").iterdir())
    assert len(traces) == 5 and "X01--oracle--repair-3--greedy.jsonl" in traces
    packet = json.loads(
        (out / "packets" / "X01--model--single--greedy" / "packet.json").read_text("utf-8")
    )
    assert packet["solved"] and packet["touched_files"] == ["arith.py"]
    assert not any(line in json.dumps(packet) for line in CORRECTNESS.splitlines() if line)
    assert json.loads((out / "validation.json").read_text("utf-8"))["cases"][0]["valid"]
    # A second run into the same directory is refused before anything executes.
    calls = len(executor.calls)
    receipt(path, argv)
    assert agent_dev_run.main(argv, execute=executor, model_loader=loader) == 2
    assert len(executor.calls) == calls


def test_the_tool_refuses_sealed_and_held_out_paths(tmp_path: Path) -> None:
    from tools import agent_dev_run

    assert agent_dev_run.main(["register", "--cases", str(tmp_path / "heldout")]) == 2
    assert (
        agent_dev_run.main(
            ["plan", "--cases", str(tmp_path), "--tokenizer", str(tmp_path / "test.jsonl")]
        )
        == 2
    )


# ----------------------------------------------------------------------------- review fixes


def test_case_paths_must_stay_inside_the_case_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("assert True\n", encoding="utf-8")
    changes: list[tuple[str, object]] = [
        ("hidden", {"correctness": "../outside.py", "preservation": "hidden/preservation.py"}),
        ("repo", ".."),
        ("reference", {"arith.py": "/etc/hosts"}),
    ]
    for field, value in changes:
        root = tmp_path / field
        overrides: dict[str, object] = {field: value}
        write_case(root, "X01", **overrides)
        with pytest.raises(agent_cases.CaseError, match="escapes|relative"):
            agent_cases.load_case(next(root.iterdir()))


def test_a_registry_directory_cannot_leave_the_root(tmp_path: Path) -> None:
    root = tmp_path / "cases"
    write_case(root)
    write_registry(root)
    registry = json.loads((root / "registry.json").read_text("utf-8"))
    registry["cases"][0]["directory"] = "../cases/X01-double"
    (root / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    assert agent_cases.load_cases(root)  # Resolves back inside: allowed.
    registry["cases"][0]["directory"] = ".."
    (root / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(agent_cases.CaseError):
        agent_cases.load_cases(root)


def test_every_file_read_must_be_registered_with_its_hash(tmp_path: Path) -> None:
    directory = write_case(tmp_path)
    write_registry(tmp_path)
    registry = json.loads((tmp_path / "registry.json").read_text("utf-8"))
    files = dict(registry["cases"][0]["files"])
    with pytest.raises(agent_cases.CaseError, match="not the registered file"):
        agent_cases.load_case(
            directory, registered={k: v for k, v in files.items() if "correctness" not in k}
        )
    files["hidden/correctness.py"] = "0" * 64
    with pytest.raises(agent_cases.CaseError, match="not the registered file"):
        agent_cases.load_case(directory, registered=files)


@pytest.mark.parametrize("case_id", ["../evil", "a/b", "", "-x", "x y"])
def test_case_ids_are_safe_file_names(tmp_path: Path, case_id: str) -> None:
    write_case(tmp_path, case_id="X01", id=case_id)
    with pytest.raises(agent_cases.CaseError, match="case id"):
        agent_cases.load_case(next(tmp_path.iterdir()))


def test_output_names_cannot_escape_their_directory(tmp_path: Path) -> None:
    from tools import agent_dev_run

    assert agent_dev_run._under(tmp_path, "X01--model.jsonl") == tmp_path / "X01--model.jsonl"
    for name in ("../x.jsonl", "a/b", "..", ""):
        with pytest.raises(ValueError, match="escapes"):
            agent_dev_run._under(tmp_path, name)


def test_public_tests_that_pass_before_the_fix_are_refused(tmp_path: Path) -> None:
    write_case(tmp_path, public_fails_before_fix=False)
    with pytest.raises(agent_cases.CaseError, match="fail before the fix"):
        agent_cases.load_case(next(tmp_path.iterdir()))


def test_carriage_returns_in_case_files_are_refused(tmp_path: Path) -> None:
    directory = write_case(tmp_path)
    (directory / "repo" / "steps.py").write_bytes(b"def clamp(v, a, b):\r    return v\n")
    with pytest.raises(agent_cases.CaseError, match="carriage return"):
        agent_cases.load_case(directory)


def one_function_bound(
    prompt_tokens: int, reference: list[int], snippet_tokens: int
) -> dict[str, Any]:
    """The fit bound of a case with one showable function, which turn 1 already shows (so no
    further snippet separators)."""
    from scratchcoder.agent import prompt

    return prompt.fit_bound(
        prompt_tokens,
        reference,
        snippet_tokens=snippet_tokens,
        snippets_shown=1,
        header_tokens=[10],
    )


def test_the_fit_bound_counts_both_feedback_blocks_and_the_generation_cap() -> None:
    from scratchcoder.agent import prompt

    ok = one_function_bound(300, [100], 250)
    assert ok["worst_prompt_tokens"] == 300 + 150 + 100 + prompt.SEPARATOR_ALLOWANCE
    assert ok["snippet_separator_tokens"] == 0
    assert ok["generation_cap"] == 324 and ok["fits"]
    # 1.2 x 270 + 2 = 326 > 324: one function too long for a turn, though the total fits.
    too_long = one_function_bound(100, [270], 250)
    assert too_long["total_bound"] <= 1024 and not too_long["fits"]
    # 450 + 150 + 100 + 4 = 704 > 700: the worst-case prompt is over the prompt budget. (A
    # prompt within the budget always leaves the full 324-token cap, since 700 + 324 = 1,024.)
    assert prompt.PROMPT_BUDGET + prompt.GENERATION_CAP == prompt.CONTEXT
    assert not one_function_bound(450, [100], 250)["fits"]


def test_the_fit_bound_grows_the_snippet_block_to_its_budget() -> None:

    # A later turn can show a longer function than turn 1 did (an edit can grow it up to the
    # function budget), so the worst case takes the snippet block at its full budget.
    small = one_function_bound(420, [100], 100)
    assert small["worst_prompt_tokens"] == 420 - 100 + 250 + 150 + 100 + 4
    assert not small["fits"]
    assert one_function_bound(420, [100], 250)["fits"]
    # A turn-1 snippet block already over the budget is kept as it is.
    over = one_function_bound(300, [10], 260)
    assert over["worst_prompt_tokens"] == 300 + 150 + 100 + 4


def test_the_fit_bound_counts_a_separator_for_each_further_snippet() -> None:
    from scratchcoder.agent import prompt

    # Turn 1 shows one snippet; three showable functions could all be shown later, which adds
    # two joins between snippets that the turn-1 prompt never had.
    three = prompt.fit_bound(
        296, [100], snippet_tokens=100, snippets_shown=1, header_tokens=[8] * 3
    )
    assert three["snippet_separator_tokens"] == 2 * prompt.SEPARATOR_TOKENS
    assert three["worst_prompt_tokens"] == 296 - 100 + 250 + 2 * 2 + 150 + 100 + 4 == 704
    assert not three["fits"]  # Without the joins it would have been 700: a fit.
    # Joins already in the turn-1 prompt are not counted twice.
    shown = prompt.fit_bound(
        296, [100], snippet_tokens=100, snippets_shown=3, header_tokens=[8] * 3
    )
    assert shown["snippet_separator_tokens"] == 0 and shown["fits"]
    # The headers bound the count: only so many can share the 250-token snippet budget.
    assert prompt.most_snippets([100, 100, 100, 5]) == 3
    assert prompt.most_snippets([240, 240]) == 1
    assert prompt.most_snippets([]) == 1
    wide = prompt.fit_bound(100, [10], snippet_tokens=50, snippets_shown=1, header_tokens=[10] * 40)
    assert wide["snippet_separator_tokens"] == (25 - 1) * prompt.SEPARATOR_TOKENS


def test_the_fit_bound_separator_count_matches_a_rendered_prompt() -> None:
    from test_agent_workspace import Bytes

    from scratchcoder.agent import prompt

    # Rendered, one more snippet costs its own tokens plus the blank line before it.
    encoder = Bytes()
    first = prompt.Snippet("a.py::f", "def f():\n    return 1")
    second = prompt.Snippet("a.py::g", "def g():\n    return 2")
    one = prompt.build(encoder, prompt.Blocks("Task.", [], [first], ""))
    two = prompt.build(encoder, prompt.Blocks("Task.", [], [first, second], ""))
    added = two.tokens - one.tokens
    cost = len(encoder.encode(prompt.snippet_header(second.key) + second.source))
    assert two.block_tokens["snippet"] - one.block_tokens["snippet"] == cost
    assert 0 < added - cost <= prompt.SEPARATOR_TOKENS
