"""The name prefill (``--prefill-name``), a harness aid: what it writes, where its tokens are
counted, that it is computed from the prompt alone, how it is reported, and that runs without
it are unchanged. Host-only: nothing executes workspace or model code."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from agent_fixtures import (
    ARITH,
    FIXED_DOUBLE,
    REFERENCE,
    ScriptedExecutor,
    write_case,
    write_registry,
)
from test_agent_experiments import card, config
from test_agent_loop import receipt, tool_args
from test_agent_workspace import Bytes

from scratchcoder.agent import cases as agent_cases
from scratchcoder.agent import loop, prompt, scorecard, workspace
from scratchcoder.agent.cases import Case
from scratchcoder.agent.policy import Generate, ModelPolicy, OraclePolicy
from scratchcoder.data.chat import PYTHON_END, PYTHON_START

NO_PREFILL_GOLDEN = "55da56c96617cadfd8568976806957ec071273ea5bbbf3f64e5af52239e56b89"
"""SHA-256 of every generation request and trace event of :func:`requests_and_traces`, recorded
from the code before the name prefill existed."""
PREFIX = b"def double("
REST = FIXED_DOUBLE.encode()[len(PREFIX) :]


@pytest.fixture
def case(tmp_path: Path) -> Case:
    write_case(tmp_path)
    write_registry(tmp_path)
    return agent_cases.load_cases(tmp_path)[0]


def clamp_target_case(root: Path) -> Case:
    """A case whose only target (``steps.py::clamp``) is not the nearest shown function
    (``arith.py::double``, which the failing public tests call directly)."""
    directory = write_case(
        root,
        reference={"steps.py": "reference/steps.py"},
        targets=["steps.py::clamp"],
    )
    (directory / "reference" / "steps.py").write_text(
        '"""Step helpers."""\n\n\ndef clamp(value, low, high):\n'
        '    """``value`` limited to [low, high]."""\n    return min(max(value, low), high)\n',
        encoding="utf-8",
    )
    write_registry(root)
    return agent_cases.load_cases(root)[0]


class Recorder:
    """A generator that records its requests and answers with the rest of the fixed function
    when the request ends with the prefill, else with the whole block."""

    def __init__(self) -> None:
        self.requests: list[tuple[list[int], int]] = []

    def __call__(
        self, ids: Sequence[int], cap: int, temperature: float, top_p: float, seed: int | None
    ) -> tuple[list[int], str]:
        self.requests.append((list(ids), cap))
        if list(ids[-len(PREFIX) - 1 :]) == [PYTHON_START, *PREFIX]:
            return [*REST, PYTHON_END], "stop"
        return [PYTHON_START, *FIXED_DOUBLE.encode(), PYTHON_END], "stop"


def run(
    case: Case, policy: loop.Policy, *, prefill: bool, condition: loop.Condition = loop.REPAIR_3
) -> tuple[loop.InstanceResult, loop.Trace]:
    trace = loop.Trace()
    result = loop.run_instance(
        case,
        policy,
        condition,
        loop.GREEDY,
        Bytes(),
        execute=ScriptedExecutor(),
        trace=trace,
        prefill_name=prefill,
    )
    return result, trace


def events(trace: loop.Trace, kind: str) -> list[dict[str, Any]]:
    return [event for event in trace.events if event["event"] == kind]


# ----------------------------------------------------------------------------- the prefill


def test_the_prefill_is_a_block_opener_and_the_nearest_shown_name() -> None:
    shown = prompt.Prompt((1, 2, 3), "text", {}, ("arith.py::double", "steps.py::clamp"))
    filled = prompt.name_prefill(shown, Bytes())
    assert filled.prefill == (PYTHON_START, *PREFIX)
    assert filled.prefill_text == "<|python_start|>def double("
    # The prompt itself is unchanged; only the room left for generation shrinks.
    assert (filled.ids, filled.sha256, filled.tokens) == (shown.ids, shown.sha256, shown.tokens)
    assert filled.generation_cap() == min(prompt.GENERATION_CAP, 1024 - 3 - len(filled.prefill))
    nothing = prompt.Prompt((1,), "text", {}, ())
    assert prompt.name_prefill(nothing, Bytes()) == nothing


DECORATED = "@functools.cache\ndef double(x):\n    return x + x"


def test_a_prefilled_reply_would_drop_the_decorators_of_the_function_it_replaces() -> None:
    # Why the prefill skips decorated functions: the editor replaces the whole span, decorators
    # included, and a reply opened with ``def double(`` cannot write lines before it.
    files = {"arith.py": "import functools\n\n\n" + DECORATED + "\n"}
    candidates = list(workspace.functions(files["arith.py"], "arith.py").values())
    edit = workspace.parse_reply(FIXED_DOUBLE, candidates, files, ["arith.py"])
    edited = workspace.apply_function(files, edit, ["arith.py"])
    assert "@functools.cache" not in edited["arith.py"]


def test_the_prefill_is_skipped_for_a_decorated_nearest_function() -> None:
    keys = ("arith.py::double", "steps.py::clamp")
    decorated = prompt.Prompt((1, 2, 3), "text", {}, keys, sources=(DECORATED, "def clamp(v):"))
    skipped = prompt.name_prefill(decorated, Bytes())
    assert skipped.prefill == () and skipped.prefill_text == ""
    assert skipped.prefill_skipped == "decorated" in prompt.PREFILL_SKIPS
    assert skipped.generation_cap() == decorated.generation_cap()
    # Only the nearest shown function matters, and an undecorated one is prefilled.
    plain = replace(decorated, sources=(FIXED_DOUBLE, DECORATED))
    assert prompt.name_prefill(plain, Bytes()).prefill == (PYTHON_START, *PREFIX)
    assert prompt.name_prefill(plain, Bytes()).prefill_skipped == ""


def test_built_prompts_carry_the_shown_sources_in_order() -> None:
    first = prompt.Snippet("arith.py::double", DECORATED)
    second = prompt.Snippet("steps.py::clamp", "def clamp(v):\n    return v")
    blocks = prompt.Blocks("Task.", [], [first, second], "")
    for layout in prompt.LAYOUTS:
        built = prompt.build_layout(Bytes(), blocks, layout)
        assert built.shown == (first.key, second.key)
        assert built.sources == (first.source, second.source)


def decorated_case(root: Path) -> Case:
    """The fixture case with ``@functools.cache`` on ``double``, the nearest shown function."""

    def decorate(text: str) -> str:
        return "import functools\n" + text.replace("def double(", "@functools.cache\ndef double(")

    directory = write_case(root, reference_text=decorate(REFERENCE))
    (directory / "repo" / "arith.py").write_text(decorate(ARITH), encoding="utf-8")
    write_registry(root)
    return agent_cases.load_cases(root)[0]


def test_a_decorated_nearest_function_gets_no_prefill_and_the_skip_is_recorded(
    tmp_path: Path,
) -> None:
    case = decorated_case(tmp_path)
    generate = Recorder()
    result, trace = run(case, ModelPolicy(generate, Bytes()), prefill=True)
    assert result.turns and all(turn.shown[0] == "arith.py::double" for turn in result.turns)
    for turn in result.turns:
        assert turn.prefill_skipped == "decorated"
        assert (turn.prefill_tokens, turn.prefill_function, turn.prefill_text) == (0, "", "")
    assert all(ids[-len(PREFIX) - 1 :] != [PYTHON_START, *PREFIX] for ids, _ in generate.requests)
    shown = events(trace, "prompt")
    assert shown and all(event["prefill_skipped"] == "decorated" for event in shown)
    cell = scorecard.build([result], {})["groups"]["model/repair-3"]["all"]
    assert cell["prefilled_generations"] == 0
    assert cell["prefill_skipped_generations"] == cell["generations"] == len(result.turns)
    report = scorecard.prefill_report([("model/repair-3", trace.events)], {case.id: case.targets})
    assert report["model/repair-3"]["generations_prefill_skipped"] == len(result.turns)
    assert report["model/repair-3"]["generations_prefilled"] == 0
    text = scorecard.markdown(scorecard.build([result], {}, prefill=report))
    assert f"| model/repair-3 | 0 of {len(result.turns)} | {len(result.turns)} of " in text


def test_the_reply_is_prefill_plus_generation_and_only_generated_tokens_are_budgeted(
    case: Case,
) -> None:
    generate = Recorder()
    result, trace = run(case, ModelPolicy(generate, Bytes()), prefill=True)
    assert result.solved
    ((ids, cap),) = generate.requests
    turn = result.turns[0]
    assert ids[-len(PREFIX) - 1 :] == [PYTHON_START, *PREFIX]
    assert turn.prefill_tokens == len(PREFIX) + 1 and turn.prefill_function == "arith.py::double"
    assert turn.prefill_text == "<|python_start|>def double("
    # Prefill tokens count against the context, not as generated tokens.
    assert cap == min(prompt.GENERATION_CAP, 1024 - turn.prompt_tokens - turn.prefill_tokens)
    assert turn.generated_tokens == len(REST) + 1 == result.generated_tokens
    (shown,) = events(trace, "prompt")
    assert shown["prefill_function"] == "arith.py::double"
    assert shown["prefill_tokens"] == turn.prefill_tokens
    (generation,) = events(trace, "generation")
    assert generation["tokens"] == turn.generated_tokens
    assert generation["prefill_tokens"] == turn.prefill_tokens
    assert generation["text"] == FIXED_DOUBLE  # Prefill + generated (Bytes drops special IDs).
    packet = loop.packet(result, case, "trace.jsonl")
    assert packet["turns"][0]["prefill_tokens"] == turn.prefill_tokens


def test_the_generated_token_budget_ignores_the_prefill(case: Case) -> None:
    tight = Case(**{**case.__dict__, "budgets": agent_cases.Budgets(max_generated_tokens=15)})
    generate = Recorder()
    result, _ = run(tight, ModelPolicy(generate, Bytes()), prefill=True)
    assert [cap for _, cap in generate.requests] == [15]
    assert result.generated_tokens == len(REST) + 1


class GuardedCase:
    """A case view that refuses every attribute naming a solution or a hidden check."""

    def __init__(self, case: Case) -> None:
        self._case = case

    def __getattr__(self, name: str) -> object:
        if name in {"targets", "reference", "reference_function", "reference_files"} or name in {
            "correctness",
            "preservation",
        }:
            raise AssertionError(f"the policy read case.{name}")
        return getattr(self._case, name)


class Guarded:
    """Passes the model policy a guarded case, proving it never reads targets or references."""

    def __init__(self, inner: ModelPolicy) -> None:
        self.inner = inner
        self.name = inner.name

    def propose(
        self,
        case: Case,
        turn: int,
        prompt: prompt.Prompt,
        max_new_tokens: int,
        decoding: loop.Decoding,
        files: Mapping[str, str],
    ) -> loop.Proposal:
        guarded: Any = GuardedCase(case)
        return self.inner.propose(guarded, turn, prompt, max_new_tokens, decoding, files)


def test_the_prefill_comes_from_the_shown_function_never_from_the_targets(tmp_path: Path) -> None:
    case = clamp_target_case(tmp_path)
    assert case.targets == ("steps.py::clamp",)
    generate = Recorder()
    result, trace = run(case, Guarded(ModelPolicy(generate, Bytes())), prefill=True)
    shown = events(trace, "prompt")
    assert shown and all(event["shown"][0] == "arith.py::double" for event in shown)
    assert all(event["prefill_function"] == "arith.py::double" for event in shown)
    assert all(ids[-len(PREFIX) - 1 :] == [PYTHON_START, *PREFIX] for ids, _ in generate.requests)
    # Reporting after the run sees that the prefilled name was not the target's.
    report = scorecard.prefill_report([("model/repair-3", trace.events)], {case.id: case.targets})
    assert report["model/repair-3"]["instances_prefilled"] == 1
    assert report["model/repair-3"]["instances_prefilled_target_name"] == 0
    assert result.turns[0].prefill_function == "arith.py::double"


def test_a_policy_that_ignores_the_prefill_is_refused(case: Case) -> None:
    with pytest.raises(ValueError, match="prefill"):
        run(case, OraclePolicy(), prefill=True)


def requests_and_traces(root: Path) -> tuple[str, int]:
    """Generation requests and trace events of model and oracle runs without the prefill."""
    write_case(root)
    write_registry(root)
    case = agent_cases.load_cases(root)[0]
    replies = [
        [*b"no block here"],
        [PYTHON_START, *b"def clamp(value, low, high):\n    return value", PYTHON_END],
        [PYTHON_START, *b"def double(x):\n    return x"],
        [PYTHON_START, *FIXED_DOUBLE.encode(), PYTHON_END],
    ]
    calls: list[list[object]] = []

    def generate(
        ids: Sequence[int], cap: int, temperature: float, top_p: float, seed: int | None
    ) -> tuple[list[int], str]:
        calls.append([list(ids), cap, temperature, top_p, seed])
        return list(replies[(len(calls) - 1) % len(replies)]), "stop"

    ticks = itertools.count()

    def clock() -> float:
        return float(next(ticks))

    traces = []
    for recovery in ("off", "intact-function"):
        policy = ModelPolicy(generate, Bytes(), clock=clock, recovery=recovery)
        for decoding in (loop.GREEDY, loop.Decoding("sample-0", 0.6, 0.95, 0)):
            for condition in (loop.SINGLE, loop.REPAIR_3):
                trace = loop.Trace()
                loop.run_instance(
                    case,
                    policy,
                    condition,
                    decoding,
                    Bytes(),
                    execute=ScriptedExecutor(),
                    trace=trace,
                    clock=clock,
                )
                traces.append(trace.events)
    trace = loop.Trace()
    loop.run_instance(
        case,
        OraclePolicy(),
        loop.REPAIR_3,
        loop.GREEDY,
        Bytes(),
        execute=ScriptedExecutor(),
        trace=trace,
        clock=clock,
    )
    traces.append(trace.events)
    blob = json.dumps([calls, traces], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest(), len(calls)


def test_runs_without_the_prefill_are_unchanged(tmp_path: Path) -> None:
    assert requests_and_traces(tmp_path) == (NO_PREFILL_GOLDEN, 8)


# ----------------------------------------------------------------------------- reporting


def prefill_loader() -> Callable[[Path, Path, int], tuple[Generate, Bytes, Bytes, int]]:
    def load(checkpoint: Path, tokenizer: Path, threads: int) -> tuple[Generate, Bytes, Bytes, int]:
        return Recorder(), Bytes(), Bytes(), threads

    return load


def test_the_tool_records_prefill_apart_and_labels_names_harness_supplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import agent_dev_run

    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    write_case(tmp_path / "cases")
    write_registry(tmp_path / "cases")
    runs = []
    for name, extra in (("C", []), ("P", ["--prefill-name"])):
        out = tmp_path / name
        path = tmp_path / f"{name}-receipt.json"
        argv = [arg for arg in tool_args(tmp_path, out, path) if arg != "--skip-oracle"] + [
            "--sampled-temperature",
            "0.6",
            "--sampled-top-p",
            "0.95",
            *extra,
        ]
        receipt(path, argv)
        executor = ScriptedExecutor()
        assert agent_dev_run.main(argv, execute=executor, model_loader=prefill_loader()) == 0
        runs.append(out)
    plain = json.loads((runs[0] / "scorecard.json").read_text("utf-8"))
    filled = json.loads((runs[1] / "scorecard.json").read_text("utf-8"))
    assert plain["identity"]["config"]["prefill_name"] is False and "prefill_report" not in plain
    assert filled["identity"]["config"]["prefill_name"] is True
    single = filled["groups"]["model/single"]["all"]
    assert single["name_following"] == scorecard.HARNESS_SUPPLIED
    assert single["prefilled_generations"] == single["generations"] == 2
    assert single["prefill_tokens"] == 2 * (len(PREFIX) + 1)
    assert single["generated_tokens"] == 2 * (len(REST) + 1) and single["solved"] == 2
    assert filled["identity"]["generation_cache_hits"] == 2
    # The oracle rows check the harness and never get the prefill.
    oracle = filled["groups"]["oracle/repair-3"]["all"]
    assert oracle["prefilled_generations"] == 0 and oracle["name_following"] == "model"
    oracle_trace = (runs[1] / "traces" / "X01--oracle--repair-3--greedy.jsonl").read_text("utf-8")
    assert "prefill" not in oracle_trace
    assert filled["prefill_report"]["model/single"] == {
        "instances": 2,
        "instances_prefilled": 2,
        "instances_prefilled_target_name": 2,
        "generations": 2,
        "generations_prefilled": 2,
        "generations_prefilled_target_name": 2,
        "generations_prefill_skipped": 0,
    }
    text = (runs[1] / "scorecard.md").read_text("utf-8")
    assert scorecard.PREFILL_NOTE in text and "harness-supplied (name prefill)" in text
    assert "| model/single | 2 of 2 | 2 of 2 |" in text
    assert "name prefill on" in text
    assert scorecard.PREFILL_NOTE not in (runs[0] / "scorecard.md").read_text("utf-8")
    md, js = tmp_path / "compare.md", tmp_path / "compare.json"
    command = ["compare", "--runs", *map(str, runs), "--out-md", str(md), "--out-json", str(js)]
    assert agent_dev_run.main(command) == 0
    table = json.loads(js.read_text("utf-8"))
    assert table["comparable"]
    labels = {(row["run"], row["condition"]): row["label"] for row in table["rows"]}
    assert labels[("P", "single")] == "single factor: prefill_name (harness-supplied function name)"
    compared = md.read_text("utf-8")
    assert scorecard.PREFILL_NOTE in compared
    assert "lower temperature plus nucleus sampling (T 0.6, top-p 0.95) | off | on |" in compared
    # The comparison alone reports how often the prefill applied and named a target.
    rows = {(row["run"], row["condition"]): row for row in table["rows"]}
    single = rows[("P", "single")]
    assert (single["prefilled_generations"], single["generations"]) == (2, 2)
    assert single["prefill_skipped_generations"] == 0
    assert single["prefill_report"] == filled["prefill_report"]["model/single"]
    assert rows[("C", "single")]["prefill_report"] is None
    assert "| off | on | 2 of 2 generations | harness-supplied" in compared
    assert "| off | off | 0 of 2 generations | model |" in compared
    tokens = 2 * (len(PREFIX) + 1)
    assert "## Name prefill applicability" in compared
    assert (
        f"| P | single | 2 of 2 generations | 0 of 2 generations | {tokens} | "
        "2 of 2 instances | 2 of 2 generations |"
    ) in compared
    assert "| C | single | 0 of 2 generations | 0 of 2 generations | 0 | n/a | n/a |" in compared


def test_compare_shows_na_prefill_counts_for_a_run_without_generation_counts() -> None:
    sampled = config(sampled_temperature=0.6, sampled_top_p=0.95)
    text = scorecard.comparison_markdown(scorecard.comparison([("C", card(config=sampled))]))
    assert "| off | off | n/a | model |" in text
    assert "## Name prefill applicability" not in text


def test_compare_accepts_prefill_only_as_the_single_differing_factor() -> None:
    sampled = config(sampled_temperature=0.6, sampled_top_p=0.95)
    base = card(config=sampled)
    table = scorecard.comparison(
        [("C", base), ("P", card(config={**sampled, "prefill_name": True}))]
    )
    assert table["comparable"]
    assert table["rows"][1]["factors_changed"] == ["prefill_name"]
    mixed = scorecard.comparison(
        [("C", base), ("X", card(config={**sampled, "prefill_name": True, "recovery": "x"}))]
    )
    assert not mixed["comparable"]
    assert mixed["rows"][1]["noncomparable_fields"] == ["prefill_name_with_another_factor"]
    # Any invariant difference is still refused beside the prefill.
    other = scorecard.comparison(
        [("C", base), ("Y", card(config={**sampled, "prefill_name": True}, threads=6))]
    )
    assert other["rows"][1]["noncomparable_fields"] == ["threads"]
    # A scorecard from before the prefill existed ran without it.
    assert (
        scorecard.comparison([("C", base), ("C2", card(config=sampled))])["rows"][1][
            "factors_changed"
        ]
        == []
    )


def test_compare_markdown_writes_every_failure_count_with_its_denominator() -> None:
    sampled = config(sampled_temperature=0.6, sampled_top_p=0.95)
    run = card(config=sampled)
    cell = {
        **run["groups"]["model/single"]["all"],  # type: ignore[index]
        "instances": 6,
        "scored": 5,
        "failure_classes": {"formatting": 3, "tool_interface": 2},
        "funnel": {
            "per_generation": {"generations": 6, "closed_or_recovered": 4, "parseable": 2},
            "name_failures": {"invented_name": 1},
            "instances_with_harness_defects": {"target_not_shown": 1},
        },
    }
    run["groups"] = {"model/single": {"all": cell, "sampled": cell, "greedy": cell}}
    text = scorecard.comparison_markdown(scorecard.comparison([("C", run)]))
    assert "lower temperature plus nucleus sampling (T 0.6, top-p 0.95)" in text
    assert "formatting 3 of 5 scored, tool_interface 2 of 5 scored" in text
    assert "invented_name 1 of 6 generations" in text
    assert "target_not_shown 1 of 6 instances" in text
    assert "| 6 | 4 of 6 | 2 of 6 |" in text
