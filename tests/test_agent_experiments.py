"""Agent decoding and prompt experiments: the training-shaped layout, conservative recovery of
an intact function, the reply funnel, the refusal audit, run comparison and the separation
from the RL screening protocol. Host-only: nothing executes workspace or model code."""

from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from agent_fixtures import FIXED_DOUBLE, ScriptedExecutor, write_case, write_registry
from test_agent_loop import Scripted, fake_loader, receipt, tool_args
from test_agent_workspace import FILES, Bytes, shown

from scratchcoder.agent import audit, loop, recovery, scorecard
from scratchcoder.agent import cases as agent_cases
from scratchcoder.agent.cases import Case
from scratchcoder.agent.policy import ModelPolicy
from scratchcoder.data.chat import ASSISTANT_END, PYTHON_END, PYTHON_START

GOLDEN_HARNESS = [
    "ecbb4e70a9121ee3784eb5a4cf2af500c3e0a1b2a66ed48075c6cfbdf9c93d25",
    "2c9b9e24a2c3efec9ce69ab58bb3169f2d50b28e634a60e9e6c09a0540655335",
    "ee5e850604cb68cee76bb8c1c259535cf12192fad52e490539ce06e06024d134",
]
"""Prompt SHA-256s of the fixture under the default configuration, recorded from the code
that produced the first development run (before configurations existed)."""
WRONG = 'def double(x):\n    """Twice ``x``."""\n    return x * 3'


@pytest.fixture
def case(tmp_path: Path) -> Case:
    write_case(tmp_path)
    write_registry(tmp_path)
    return agent_cases.load_cases(tmp_path)[0]


def instance(
    case: Case,
    policy: loop.Policy,
    *,
    layout: str = "harness",
    condition: loop.Condition = loop.REPAIR_3,
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
        layout=layout,
    )
    return result, trace


def prompt_hashes(trace: loop.Trace) -> list[str]:
    return [event["prompt_sha256"] for event in trace.events if event["event"] == "prompt"]


# ----------------------------------------------------------------------------- layouts


def test_the_default_configuration_reproduces_the_first_run_prompts(case: Case) -> None:
    _, trace = instance(case, Scripted(["def triple(x):\n    return x", WRONG, FIXED_DOUBLE]))
    assert prompt_hashes(trace) == GOLDEN_HARNESS


def test_the_training_layout_is_task_code_public_test_and_feedback_on_repair(case: Case) -> None:
    policy = Scripted(["def triple(x):\n    return x", FIXED_DOUBLE])
    result, trace = instance(case, policy, layout="training")
    assert result.solved
    first, second = (prompt.text for prompt in policy.prompts)
    assert first.startswith("Bug: double(2) returns 5 instead of 4. Fix the bug.\n\ndef double(x):")
    assert "def test_double():\n    got = double(2)\n    assert got == 4, repr(got)" in first
    assert "RESULT run_tests" not in first and "MAP" not in first and "SNIPPET" not in first
    assert first.endswith("Reply with the complete corrected version of one function shown above.")
    assert (
        "EDIT refused: the function is not one of the functions shown\nRESULT run_tests" in second
    )
    # Same generation cap rule and the same budgets.
    caps = [event["generation_cap"] for event in trace.events if event["event"] == "prompt"]
    assert caps == [324, 324]
    with pytest.raises(ValueError, match="layout"):
        instance(case, Scripted([None]), layout="other")


def test_the_oracle_solves_under_the_training_layout(case: Case) -> None:
    from scratchcoder.agent.policy import OraclePolicy

    result, _ = instance(case, OraclePolicy(), layout="training")
    assert result.solved and result.target_located


# ----------------------------------------------------------------------------- recovery


@pytest.mark.parametrize(
    ("text", "eos", "rule", "kept"),
    [
        # A complete function, then an example call cut off by the length limit.
        (
            "def f(x):\n    return x\n\nprint(f(2), f(3))",
            False,
            "followed_by_top_level",
            "def f(x):\n    return x\n",
        ),
        # The model closed its turn; only the block marker is missing.
        (
            "import re\n\ndef f(x):\n    return x",
            True,
            "eos_whole_parse",
            "import re\n\ndef f(x):\n    return x",
        ),
        # Complete example calls and comments may follow; the longest parseable prefix stays.
        (
            "def f(x):\n    return x\n# usage\nf(1)\nprint(f(1))\n# done\n",
            False,
            "followed_by_top_level",
            "def f(x):\n    return x\n# usage\nf(1)",
        ),
        # Nested defs belong to the outer function.
        (
            "def f(x):\n    def g():\n        return 1\n    return g()\nprint(f)",
            False,
            "followed_by_top_level",
            "def f(x):\n    def g():\n        return 1\n    return g()",
        ),
        # A decorated function keeps its decorator.
        (
            "@cache\ndef f(x):\n    return x\n\n(lambda: 0)()",
            False,
            "followed_by_top_level",
            "@cache\ndef f(x):\n    return x\n",
        ),
    ],
)
def test_recovery_keeps_only_a_verbatim_prefix_of_complete_statements(
    text: str, eos: bool, rule: str, kept: str
) -> None:
    found = recovery.recover_text(text, ended_by_eos=eos)
    assert found is not None and found.rule == rule and found.code == kept
    assert text.startswith(found.code)


@pytest.mark.parametrize(
    ("text", "eos"),
    [
        ("def f(x):\n    return (x +", False),  # body cut mid-expression
        ("def f(x):\n    return x", False),  # complete-looking, but the length ran out
        ("def f(x):\n    return (x +", True),  # EOS but does not parse
        ('x = """\ndef f():\n    return 1\n"""\n', False),  # "def" only inside a string
        ('def f():\n    s = """\ndef g():\n    pass\nend', False),  # cut inside a string
        ("import os\nimport re", True),  # parses but no function
        ("def f(x):\r    return x\rx = 1", False),  # carriage returns are not normalized
        # The dropped tail could change the kept function: a helper defined after it, ...
        ("def f(x):\n    return helper(x)\ndef helper(y):\n    return (y", False),
        # ... a used name bound by a truncated assignment, ...
        ("def f(x):\n    return SCALE * x\nSCALE = (3 +", False),
        # ... f redefined later, ...
        ("def f(x):\n    return 1\nprint(f(1))\ndef f(x):\n    return (2", False),
        # ... a second shown function after another one, or any import or annotation.
        ("def f(x):\n    return x\ndef g(y):\n    return y\ndef h(z):\n    return (", False),
        ("def f(x):\n    return x\nfrom math import (", False),
        ("def f(x):\n    return x\nclass K(", False),
        ("def f(x):\n    return x\n@cache\ndef g(", False),
        ("def f(x):\n    return x\nLIMIT: int = (", False),
        ("def f(x):\n    return x\ncount += f(", False),
        ("def f(x):\n    return x\nif f(1):\n    y = (", False),
        # Tails the normal rules would refuse once closed, or whose meaning is unknown.
        ("def f(x):\n    return x\nprint([x for x in range(3)])", False),
        ("def f(x):\n    return x\nprint(1,\n    (x := 2))", False),
        ("def f(x):\n    return x\nprint(1) + \\\n    (x := 2)", False),
        ("def f(x):\n    return x\nfor(x) in (\n    [1]): pass", False),
        ("def f(x):\n    return x\nfor(x) in range(3)", False),
        ("def f(x):\n    return x\nwith(open('p')) as x: pass", False),
        ("def f(x):\n    return x\nwith(open('p')) as x", False),
        ("def f(x):\n    return x\ndel(x", False),
        ("def f(x):\n    return x\ndel(x)", False),
        ("def f(x):\n    return x\nif(x", False),
        ('def f(x):\n    return x\nprint("\\""); x = ', False),
        ('def f(x):\n    return x\nprint("\\""); x = 1', False),
        ("def f(x):\n    return x\nf(1); import os as", False),
        ("def f(x):\n    return x\nf(1); import os", False),
        ("def f(x):\n    return x\nprint(1)\n\x0cx = 1", False),
        ("def f(x):\x0b\n    return x\nprint(1)", False),
        ("def f(x):\n    return x\nprint(f(", False),  # a truncated call: meaning unknown
    ],
)
def test_recovery_refuses_anything_not_provably_complete(text: str, eos: bool) -> None:
    assert recovery.recover_text(text, ended_by_eos=eos) is None


def test_recovery_applies_only_to_an_open_block_without_an_end_marker() -> None:
    decoder = Bytes()
    body = list(b"def f(x):\n    return x\nprint(1)")
    assert recovery.recover([PYTHON_START, *body], decoder, "length") is not None
    assert recovery.recover(body, decoder, "length") is None  # no block at all
    assert recovery.recover([ASSISTANT_END, PYTHON_START, *body], decoder, "length") is None
    assert (
        recovery.recover([PYTHON_START, *body, ASSISTANT_END, PYTHON_END], decoder, "stop") is None
    )
    closed = list(b"def f(x):\n    return x")
    eos = recovery.recover([PYTHON_START, *closed, ASSISTANT_END], decoder, "stop")
    assert eos is not None and eos.rule == "eos_whole_parse"
    assert recovery.recover([PYTHON_START, *closed], decoder, "length") is None


def model_policy(replies: list[list[int]], mode: str, finish: str = "length") -> ModelPolicy:
    def generate(
        ids: Sequence[int], cap: int, temperature: float, top_p: float, seed: int | None
    ) -> tuple[list[int], str]:
        return replies.pop(0), finish

    return ModelPolicy(generate, Bytes(), recovery=mode)


def test_a_recovered_function_goes_through_the_same_reply_rules(case: Case) -> None:
    unclosed = [PYTHON_START, *(FIXED_DOUBLE + "\n\nprint(double(2))").encode()]
    off, _ = instance(case, model_policy([list(unclosed)] * 3, "off"), condition=loop.SINGLE)
    assert not off.solved and off.turns[0].edit == "no_python_block"
    on, trace = instance(
        case, model_policy([list(unclosed)], "intact-function"), condition=loop.SINGLE
    )
    turn = on.turns[0]
    assert on.solved and turn.recovered and not turn.closed_block
    assert turn.recovery == {"rule": "followed_by_top_level", "end_line": 4, "block_lines": 5}
    generation = next(event for event in trace.events if event["event"] == "generation")
    assert generation["recovered"] and generation["text"].endswith(
        "print(double(2))"
    )  # full original
    # A recovered prefix with two shown functions is refused like a normal reply.
    two = [
        PYTHON_START,
        *(
            FIXED_DOUBLE + "\n\ndef scaled(x, a, b):\n    return 0\n\nprint(scaled(1, 0, 2))"
        ).encode(),
    ]
    refused, _ = instance(case, model_policy([two], "intact-function"), condition=loop.SINGLE)
    assert refused.turns[0].recovered and refused.turns[0].edit == "multiple_functions"


# ----------------------------------------------------------------------------- funnel, audit


def test_the_funnel_counts_every_stage_per_generation_and_per_instance(case: Case) -> None:
    replies = [None, "def double(x)", "def triple(x):\n    return 1"]
    failed, _ = instance(case, Scripted(replies))
    wrong_then_right, _ = instance(case, Scripted([WRONG, FIXED_DOUBLE]))
    found = scorecard.funnel([failed, wrong_then_right])
    assert found["per_generation"] == {
        "generations": 5,
        "closed_or_recovered": 4,
        "parseable": 3,
        "one_shown_function": 2,
        "target_function": 2,
        "applied_target_edit": 2,
        "public_passed": 1,
        "fresh_generations": 5,
        "closed_block": 4,
        "recovered": 0,
        "applied_non_target_edit": 0,
    }
    assert found["per_instance"] == {
        "instances": 2,
        "closed_or_recovered": 2,
        "parseable": 2,
        "one_shown_function": 1,
        "target_function": 1,
        "applied_target_edit": 1,
        "public_passed": 1,
        "solved": 1,
        "applied_non_target_edit": 0,
        "solved_without_target_edit": 0,
    }
    assert found["refusals"] == {"no_python_block": 1, "unknown_name": 1, "unparsable": 1}
    assert found["name_failures"] == {"invented_name": 1}
    assert found["instances_with_harness_defects"] == {}


def test_name_failures_separate_task_following_from_locate_defects() -> None:
    files = dict(FILES)
    only_scaled = shown("arith.py::scaled")
    assert audit.name_failure(["triple"], only_scaled, files, ["arith.py::double"]) == (
        "invented_name"
    )
    assert audit.name_failure(["clamp"], only_scaled, files, ["arith.py::double"]) == (
        "unshown_repo_function"
    )
    assert audit.name_failure(["double"], only_scaled, files, ["arith.py::double"]) == (
        "locate_omitted_target"
    )
    assert audit.name_failure(["test_double"], only_scaled, files, ["arith.py::double"]) == (
        "public_test_name"
    )


def flags(reason: str, code: str | None, tokens: Sequence[int] = ()) -> list[str]:
    return audit.audit_refusal(
        reason,
        tokens=tokens,
        code=code,
        shown=shown("arith.py::double"),
        files=FILES,
        targets=["arith.py::double"],
    )


def test_the_audit_flags_refusals_that_contradict_the_rules() -> None:
    assert flags("unparsable", FIXED_DOUBLE) == ["parse_contradiction"]
    assert flags("no_function", FIXED_DOUBLE) == ["function_contradiction"]
    assert flags("unknown_name", FIXED_DOUBLE) == ["name_contradiction"]
    assert flags("multiple_functions", FIXED_DOUBLE) == ["single_function_refused"]
    # The target already imports clamp; refusing a reply that imports it again differently
    # is a harness defect, not a model failure.
    again = "from steps import clamp, clamp as c2\n\ndef double(x):\n    return clamp(x + x, 0, 9)"
    assert flags("extra_statements", again) == ["import_already_present"]
    closed = (PYTHON_START, *FIXED_DOUBLE.encode(), PYTHON_END)
    assert flags("no_python_block", None, closed) == ["closed_block_missed"]
    # Consistent refusals carry no flag.
    assert flags("unparsable", "def (") == []
    assert flags("unknown_name", "def triple(x): pass") == []
    assert flags("no_python_block", None, (PYTHON_START, *b"def f():")) == []
    uses_new = "import math\n\ndef double(x):\n    return math.fsum([x, x])"
    assert flags("extra_statements", uses_new) == []
    # The same name from another source is a different import: no defect.
    other = "from other import clamp\n\ndef double(x):\n    return clamp(x + x, 0, 9)"
    assert flags("extra_statements", other) == []
    renamed = "import steps as clamp\n\ndef double(x):\n    return clamp(x + x, 0, 9)"
    assert flags("extra_statements", renamed) == []
    aliased = "from steps import clamp as c\n\ndef double(x):\n    return c(x, 0, 9)"
    assert flags("extra_statements", aliased) == []


def test_real_refusals_in_the_loop_are_audited_clean(case: Case) -> None:
    result, trace = instance(case, Scripted([None, "def (", "def triple(x):\n    return 1"]))
    assert all(not turn.audit for turn in result.turns) and result.harness_defects == []
    edits = [event for event in trace.events if event["event"] == "edit"]
    assert [event["audit"] for event in edits] == [[], [], []]
    assert edits[2]["name_failure"] == "invented_name"


# ----------------------------------------------------------------------------- the tool


def test_run_records_the_configuration_and_compare_writes_one_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import agent_dev_run

    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    write_case(tmp_path / "cases")
    write_registry(tmp_path / "cases")
    runs = []
    for name, extra in (
        ("A", []),
        (
            "E",
            [
                "--prompt",
                "training",
                "--sampled-temperature",
                "0.6",
                "--sampled-top-p",
                "0.95",
                "--recovery",
                "intact-function",
            ],
        ),
    ):
        out = tmp_path / name
        path = tmp_path / f"{name}-receipt.json"
        argv = [*tool_args(tmp_path, out, path), *extra]
        receipt(path, argv)
        loader = fake_loader([[PYTHON_START, *FIXED_DOUBLE.encode(), PYTHON_END]])
        assert agent_dev_run.main(argv, execute=ScriptedExecutor(), model_loader=loader) == 0
        runs.append(out)
    first = json.loads((runs[0] / "scorecard.json").read_text("utf-8"))
    second = json.loads((runs[1] / "scorecard.json").read_text("utf-8"))
    assert first["identity"]["config"] == {
        "prompt": "harness",
        "sampled_temperature": 1.0,
        "sampled_top_p": 1.0,
        "recovery": "off",
        "prefill_name": False,
    }
    assert second["identity"]["config"]["recovery"] == "intact-function"
    assert [d["temperature"] for d in second["identity"]["decodings"]] == [0.0, 0.6]
    assert first["groups"]["model/single"]["all"]["funnel"]["per_instance"]["solved"] == 2
    md, js = tmp_path / "compare.md", tmp_path / "compare.json"
    command = ["compare", "--runs", *map(str, runs), "--out-md", str(md), "--out-json", str(js)]
    assert agent_dev_run.main(command) == 0
    table = json.loads(js.read_text("utf-8"))
    assert [(row["run"], row["condition"]) for row in table["rows"]] == [
        ("A", "repair-3"),
        ("A", "single"),
        ("E", "repair-3"),
        ("E", "single"),
    ]
    assert table["rows"][1]["sampled_solved"] == [1, 1] and table["rows"][1]["greedy_solved"] == [
        1,
        1,
    ]
    text = md.read_text("utf-8")
    assert (
        "| A | single | baseline | yes | harness | default sampling (T 1.0, top-p 1.0) | off | off "
        "| 0 of 2 generations | model |" in text
    )
    assert (
        "| E | single | combined: prompt + sampled decoding, lower temperature plus nucleus "
        "sampling (T 0.6, top-p 0.95) + recovery | yes |" in text
    )
    assert table["comparable"] and table["rows"][2]["differs_from_baseline"] == [
        "prompt",
        "sampled_temperature",
        "sampled_top_p",
        "recovery",
    ]
    assert agent_dev_run.main(command) == 2  # never overwrites


# ----------------------------------------------------------------------------- RL separation


def test_the_agent_package_never_imports_or_alters_the_rl_screening_protocol() -> None:
    import scratchcoder.agent as package

    root = Path(package.__file__).parent
    for path in [*root.glob("*.py"), Path(__file__).parents[1] / "tools" / "agent_dev_run.py"]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(name.startswith("scratchcoder.rl") for name in names), path
    import tools.agent_dev_run  # noqa: F401
    from scratchcoder.agent import policy
    from scratchcoder.rl import rollout

    assert rollout.SCREENING_SEED == 2**40 and rollout.ROLLOUT_TOKENS == 256
    defaults = inspect.signature(rollout.screening_rollouts).parameters
    assert defaults["temperature"].default == 1.0 and defaults["top_p"].default == 1.0
    assert defaults["max_new_tokens"].default == rollout.ROLLOUT_TOKENS
    assert policy.AGENT_SEED != rollout.SCREENING_SEED


def test_boundaries_are_logical_lines_not_backslash_continuations() -> None:
    assert recovery.logical_line_starts("a = 1 + \\\n2\nb = (3,\n4)\nc = 5\n") == [0, 2, 4]
    # The column-0 "1" continues the return statement, so it is kept with the function.
    found = recovery.recover_text(
        "def f(x):\n    return x + \\\n1\nprint(f(1))", ended_by_eos=False
    )
    assert found is not None and found.code == "def f(x):\n    return x + \\\n1"
    # A truncated tail stops the tokenizer without losing the starts before it.
    assert recovery.logical_line_starts('x = 1\ny = """\nz = 2') == [0, 1]


def test_funnel_stages_nest_and_non_target_edits_stay_beside_them(case: Case) -> None:
    non_target = "def scaled(x, low, high):\n    return low"
    results = [
        instance(case, Scripted(replies))[0]
        for replies in (
            [non_target] * 3,  # applied, but never to a target
            [None, "def (", FIXED_DOUBLE],
            [WRONG, non_target, FIXED_DOUBLE],
            ["def triple(x):\n    return 1", WRONG, WRONG],
        )
    ]
    found = scorecard.funnel(results)
    for level, stages in (
        ("per_generation", scorecard.GENERATION_STAGES),
        ("per_instance", scorecard.INSTANCE_STAGES),
    ):
        counts = [found[level][stage] for stage in stages]
        pairs = zip(counts, counts[1:], strict=False)
        assert all(later <= earlier for earlier, later in pairs), (level, counts)
    assert found["per_generation"]["applied_non_target_edit"] == 4
    assert found["per_instance"]["applied_non_target_edit"] == 1
    assert found["per_instance"]["applied_target_edit"] == 3
    assert found["per_instance"]["solved"] == 2


def card(**identity: object) -> dict[str, object]:
    base: dict[str, object] = {
        "git_commit": "abc",
        "dirty": False,
        "valid_cases": ["X01"],
        "cases_registry_sha256": "r",
        "checkpoint_sha256": "c",
        "tokenizer_sha256": "t",
        "seed_base": 1,
        "decodings": [
            {"label": "greedy", "index": 0, "temperature": 0.0, "top_p": 1.0},
            {"label": "sample-0", "index": 0, "temperature": 1.0, "top_p": 1.0},
        ],
        "conditions": [{"name": "single", "edit_turns": 1}],
        "budgets": {"prompt": 700},
        "threads": 1,
        "batch_size": 1,
        "supervisor_sha256": "s",
        "runtime_image_id": "i",
        "precision": "float32",
        "device": "cpu",
        "protocol": "chat-v1",
        "config": {
            "prompt": "harness",
            "sampled_temperature": 1.0,
            "sampled_top_p": 1.0,
            "recovery": "off",
        },
    }
    base.update(identity)
    cell = {
        "solved": 0,
        "scored": 1,
        "unscored": 0,
        "failure_classes": {},
        "generation_seconds": 0.0,
        "sandbox_calls": 1,
        "wall_seconds": 0.0,
    }
    group = {"all": cell, "sampled": cell, "greedy": cell}
    return {"identity": base, "groups": {"model/single": group}}


def config(**changes: object) -> dict[str, object]:
    return {
        "prompt": "harness",
        "sampled_temperature": 1.0,
        "sampled_top_p": 1.0,
        "recovery": "off",
        **changes,
    }


def test_compare_labels_factors_and_flags_runs_that_differ_outside_the_config() -> None:
    table = scorecard.comparison(
        [
            ("A", card()),
            ("B", card(config=config(prompt="training"))),
            ("C", card(config=config(sampled_temperature=0.6, sampled_top_p=0.95))),
            ("D", card(config=config(recovery="intact-function"))),
            ("E", card(config=config(prompt="training", recovery="intact-function"))),
            ("A2", card()),
        ]
    )
    labels = [row["label"] for row in table["rows"]]
    assert labels == [
        "baseline",
        "single factor: prompt",
        "single factor: sampled decoding, lower temperature plus nucleus sampling "
        "(T 0.6, top-p 0.95)",
        "single factor: recovery",
        "combined: prompt + recovery",
        "same configuration as baseline",
    ]
    assert table["comparable"]
    for change in (
        {"checkpoint_sha256": "other"},
        {"threads": 6},
        {"valid_cases": ["X01", "X02"]},
        {"decodings": [{"label": "greedy", "index": 0, "temperature": 0.0, "top_p": 1.0}]},
        {"seed_base": 2},
        {"runtime_image_id": "j"},
        {"git_commit": "def"},
        {"dirty": True},
    ):
        mixed = scorecard.comparison([("A", card()), ("X", card(**change))])
        assert not mixed["comparable"] and mixed["rows"][1]["noncomparable_fields"]
        assert mixed["rows"][0]["comparable"]


def test_the_compare_command_refuses_noncomparable_runs_unless_allowed(tmp_path: Path) -> None:
    from tools import agent_dev_run

    for name, identity in (("A", {}), ("B", {"threads": 6})):
        (tmp_path / name).mkdir()
        (tmp_path / name / "scorecard.json").write_text(json.dumps(card(**identity)), "utf-8")
    md, js = tmp_path / "c.md", tmp_path / "c.json"
    command = [
        "compare",
        "--runs",
        str(tmp_path / "A"),
        str(tmp_path / "B"),
        "--out-md",
        str(md),
        "--out-json",
        str(js),
    ]
    assert agent_dev_run.main(command) == 2 and not md.exists() and not js.exists()
    assert agent_dev_run.main([*command, "--allow-noncomparable"]) == 0
    assert "NO: threads" in md.read_text("utf-8")


@pytest.mark.parametrize(
    "tail",
    [
        ["print(f(1))"],
        ["f(1) if True else g"],
        ["(lambda: 0)()"],
        ["# only a comment", "   # another", ""],
        [],
        ["print(f(1))", "", "# trailing"],
    ],
)
def test_inert_tails_are_complete_expressions_that_bind_nothing(tail: list[str]) -> None:
    assert recovery.tail_is_inert(tail)
    found = recovery.recover_text("def f(x):\n    return x\n" + "\n".join(tail), ended_by_eos=False)
    if any(line.strip() and not line.lstrip().startswith("#") for line in tail):
        assert found is not None and found.code == "def f(x):\n    return x"
