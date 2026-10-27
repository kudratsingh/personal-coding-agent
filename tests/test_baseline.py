import base64
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.runner import Result
from tools.baselines.humaneval import (
    completion_diagnostics,
    encrypted_failures,
    failure_category,
    generate_reference_completion,
    normalize,
    truncate_completion,
    validate_references,
)


def test_humaneval_mapping_and_stop_words() -> None:
    rows = [
        {
            "task_id": "synthetic/0",
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
            "canonical_solution": "    return a+b\n",
            "test": "def check(candidate): assert candidate(1,2) == 3",
        }
    ]
    task = normalize(rows)[0]
    assert task.split == "eval"
    assert task.hidden_tests.endswith("check(add)")
    assert task.solution == "def add(a, b):\n    return a+b\n"
    assert "candidate(1,2)" not in task.public_prompt()
    assert truncate_completion("    return a+b\n\ndef other(): pass") == "    return a+b\n"


def test_newline_ending_reference_prompt_is_stripped_before_generation() -> None:
    calls: list[str] = []
    completion = "\n    return 3\n# discard"

    class Tokenizer:
        eos_token_id = 0

        def __call__(self, text: str, *, return_tensors: str) -> dict[str, torch.Tensor]:
            assert return_tensors == "pt"
            calls.append(text)
            return {"input_ids": torch.tensor([[4, 5]])}

        def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert tokens == [6, 7] and skip_special_tokens
            return completion

    class Model:
        def generate(self, **options: object) -> torch.Tensor:
            assert calls == ["def fixture():"]
            assert options["max_new_tokens"] == 256 and options["do_sample"] is False
            assert options["pad_token_id"] == options["eos_token_id"] == 0
            return torch.tensor([[4, 5, 6, 7]])

    prompt, text, tokens = generate_reference_completion(
        Model(), Tokenizer(), "def fixture():\n \t\n", seed=7, budget=256
    )
    assert prompt == "def fixture():" and text == completion and tokens == [6, 7]
    assert prompt + truncate_completion(text) == "def fixture():\n    return 3"


def test_reference_validation_uses_solution_and_fails_closed() -> None:
    task = Task(
        id="fixture",
        prompt="question",
        public_tests="public",
        hidden_tests="hidden-test",
        source="synthetic",
        license="fixture",
        written_by="human",
        split="dev",
    )
    tasks = [replace(task, solution="known answer")]

    def execute(code: str, tests: str, mode: str) -> Result:
        assert code == "known answer" and tests == "hidden-test" and mode == "function"
        return Result("passed")

    report = validate_references(tasks, execute=execute)
    assert report["metrics"]["greedy_pass_at_1"] == 1
    with pytest.raises(RuntimeError, match="aborted"):
        validate_references(tasks, execute=lambda *_: Result("unavailable"))
    with pytest.raises(ValueError, match="missing"):
        validate_references([task], execute=execute)


def test_failure_categories_never_retain_raw_output() -> None:
    assert failure_category(Result("failed", "private-test\nSyntaxError: secret")) == "SyntaxError"
    assert failure_category(Result("failed", "secret")) == "other_failure"
    assert failure_category(Result("timeout", "secret")) == "timeout"


@pytest.mark.parametrize(
    "completion",
    [
        "    return 3\n",
        "    return 3\n\ndef other(): pass",
        "\n# host-secret",
        "    return (",
        "    return 'host-secret'\nclass Ignore: pass",
        "\x00",
    ],
)
def test_generation_diagnostics_observe_existing_stops_without_source_leak(completion: str) -> None:
    import json

    retained = truncate_completion(completion)
    diagnostics = completion_diagnostics(
        "def fixture():\n", completion, [7, 8], budget=2, eos_token_id=9
    )
    assert diagnostics["retained_characters"] == len(retained)
    assert diagnostics["decoded_characters"] == len(completion)
    assert diagnostics["budget_reached"] and diagnostics["budget_without_eos"]
    assert not diagnostics["eos_seen"]
    assert diagnostics["empty_retained_completion"] == (not retained.strip())
    assert "host-secret" not in json.dumps(diagnostics)
    assert truncate_completion(completion) == retained  # instrumentation never changes the source.


def test_generation_budget_eos_and_first_stop_accounting() -> None:
    row = completion_diagnostics(
        "def fixture():\n",
        "    return 3\n# stop\ndef later(): pass",
        [8, 9],
        budget=2,
        eos_token_id=9,
    )
    assert row["budget_reached"] and row["eos_seen"] and not row["budget_without_eos"]
    assert row["candidate_syntax_valid"] and row["stop_index"] == 4
    short = completion_diagnostics(
        "def fixture():\n", "    return 3", [9], budget=2, eos_token_id=9
    )
    assert not short["budget_reached"] and short["eos_seen"]
    invalid = completion_diagnostics(
        "def fixture():\n", "    return (", [], budget=2, eos_token_id=None
    )
    assert not invalid["candidate_syntax_valid"]
    for tokens in ([True], [-1], [1, 2, 3]):
        with pytest.raises(ValueError, match="generation"):
            completion_diagnostics("", "", tokens, budget=2, eos_token_id=None)


def test_explicit_no_stops_preserves_the_entire_decoded_completion() -> None:
    completion = "\ndef other():\n    return 4\n# keep this too\n"
    assert truncate_completion(completion) == ""  # Legacy default until authorized fix.
    assert truncate_completion(completion, stops=()) == completion
    row = completion_diagnostics(
        "def fixture():\n", completion, [1, 2], budget=2, eos_token_id=9, stops=()
    )
    assert row["retained_characters"] == len(completion)
    assert not row["empty_retained_completion"] and row["stop_index"] is None
    assert row["budget_reached"] and row["budget_without_eos"]


def test_failure_receipt_round_trip_keeps_source_out_of_plaintext(tmp_path: Path) -> None:
    key, certificate = tmp_path / "private.key", tmp_path / "recipient.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=authored-receipt-fixture",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    rows = [
        {"id": "authored/0", "category": "SyntaxError", "completion": "\nπ = 'fixture'\n"},
        {"id": "authored/1", "category": "AssertionError", "completion": ""},
    ]
    encoded = encrypted_failures(rows, certificate)
    assert "fixture" not in encoded and "completion" not in encoded
    clear = subprocess.run(
        ["openssl", "cms", "-decrypt", "-binary", "-inform", "DER", "-inkey", str(key)],
        input=base64.b64decode(encoded),
        capture_output=True,
        check=True,
    ).stdout
    assert json.loads(clear) == {"failures": rows}
