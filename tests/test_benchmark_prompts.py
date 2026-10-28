"""Run 1b's prompt changes leave every protocol but chat-v1's benchmark turn as it was.

The base protocols (``base-pool-completion-v1``, ``base-completion-rstrip-v1``) and chat-v1's
pool prompts are pinned here by the SHA-256 of the prompts they render for authored tasks,
computed with the code before run 1b; run 1's base results were generated under them. The
MBPP+ chat-v1 user turn is EvalPlus's prompt (the task text and the first original test),
and an SFT code row may leave its public tests out by a hash rule; neither reaches an
evaluation prompt of the pool.
"""

import hashlib
import json

import pytest

from scratchcoder.data import chat
from scratchcoder.data.sft import (
    PUBLIC_TESTS_OMITTED_RULE,
    PUBLIC_TESTS_OMITTED_SALT,
    omits_public_tests,
)
from scratchcoder.eval.protocols import BASE_BENCHMARK, BASE_POOL, CHAT, Protocol
from scratchcoder.eval.tasks import Task


class Bytes:
    def encode(self, text: str) -> list[int]:
        return list(text.encode())


def tasks() -> list[Task]:
    common = {"source": "fixture", "license": "fixture", "written_by": "human"}
    return [
        Task(
            id="mbpp/2",
            prompt="Write a function to find the shared elements.\n",
            public_tests=(
                "import math\nassert shared((1, 2), (2, 3)) == (2,)\nassert shared((), ()) == ()"
            ),
            hidden_tests="assert shared((5,), (5,)) == (5,)",
            split="eval",
            solution="def shared(a, b):\n    return tuple(set(a) & set(b))",
            **common,
        ),
        Task(
            id="mbpp/7",
            prompt="Write a python function to identify non-prime numbers.",
            public_tests="\nassert is_not_prime(2) == False\nassert is_not_prime(10) == True",
            hidden_tests="assert is_not_prime(35) == True",
            split="eval",
            solution="def is_not_prime(n):\n    return n > 1",
            **common,
        ),
        Task(
            id="humaneval/0",
            prompt='def add(a, b):\n    """Add two numbers."""\n',
            public_tests="assert callable(add)",
            hidden_tests="assert add(1, 2) == 3",
            split="eval",
            solution='def add(a, b):\n    """Add two numbers."""\n    return a + b',
            entry_point="add",
            **common,
        ),
        Task(
            id="pool/f",
            prompt="  Return x + 1 from inc(x).  ",
            public_tests="assert inc(1) == 2",
            hidden_tests="assert inc(2) == 3",
            split="dev",
            solution="def inc(x):\n    return x + 1",
            **common,
        ),
        Task(
            id="pool/io",
            prompt='Read n and print n """ twice.',
            public_tests=json.dumps([{"input": "1\n", "output": "2\n"}]),
            hidden_tests=json.dumps([{"input": "2\n", "output": "3\n"}]),
            split="dev",
            solution="print(int(input())+1)",
            mode="io",
            **common,
        ),
    ]


BEFORE_RUN_1B = {
    "base-completion-rstrip-v1/benchmark": (
        "7f66a2f01eaac7dd3d7082da766189932d1a5b102b2669dafb175795c4bc182e"
    ),
    "base-pool-completion-v1/pool": (
        "d7fb0e4b3deff1c0adccb903a181c7331f1e4803c6c8f981926510e1dcee009f"
    ),
    "chat-v1/pool": "61dd304ae65fbd9caa319e28b9b718188695410c66321dbec94ec27298b7c203",
}
"""Computed with the code of the commit before run 1b (``basepins``-style: the JSON list of
each protocol's prompt IDs under a byte encoder)."""


def digests() -> dict[str, str]:
    out = {}
    for name, family, ids in (
        (BASE_BENCHMARK, "benchmark", ("mbpp/2", "mbpp/7", "humaneval/0")),
        (BASE_POOL, "pool", ("pool/f", "pool/io")),
        (CHAT, "pool", ("pool/f", "pool/io")),
    ):
        protocol = Protocol(name, family)
        prompts = [protocol.prompt_ids(Bytes(), task) for task in tasks() if task.id in ids]
        out[f"{name}/{family}"] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    return out


def test_the_base_prompts_and_chat_pool_prompts_are_unchanged() -> None:
    assert digests() == BEFORE_RUN_1B


def test_the_base_and_chat_pool_protocol_records_are_unchanged() -> None:
    assert Protocol(BASE_POOL, "pool").sha256() == (
        "88a93535c2528bab4ad2e6fb9fc47d3c105401b480cdb3e0ca4ee0f203bfed8b"
    )
    assert Protocol(BASE_BENCHMARK, "benchmark").sha256() == (
        "3033d99f2319566b96f8f80a215bab215e5281cf43ac585c6f1caa9d88136a9a"
    )
    assert Protocol(CHAT, "pool").sha256() == (
        "45d44b52a51aca900d361b7da8256744b17dab4addb7e5c0a5434602188f3c6b"
    )


def test_the_chat_benchmark_record_changed_and_names_the_mbpp_plus_turn() -> None:
    record = Protocol(CHAT, "benchmark").record()
    assert "mbpp_plus_user_turn" in record["description"]
    assert Protocol(CHAT, "benchmark").sha256() != (
        "3c42f7e24fe2a198b34f4696b678556de676462749d3b56d4efa5c41c1e4b18f"
    )


def test_mbpp_plus_turns_mirror_evalplus_and_name_the_function() -> None:
    first, second, humaneval = tasks()[:3]
    assert chat.benchmark_user_text(first) == (
        '"""\nWrite a function to find the shared elements.\n'
        'assert shared((1, 2), (2, 3)) == (2,)\n"""'
    )
    # Without imports the public tests start with a newline; the first assertion is still found.
    assert chat.benchmark_user_text(second) == (
        '"""\nWrite a python function to identify non-prime numbers.\n'
        'assert is_not_prime(2) == False\n"""'
    )
    assert chat.benchmark_user_text(humaneval) == humaneval.prompt.strip()
    # The base protocol still sees the bare task text.
    assert Protocol(BASE_BENCHMARK, "benchmark").prompt_ids(Bytes(), first) == list(
        b"Write a function to find the shared elements."
    )


def test_an_mbpp_plus_task_without_an_assertion_is_refused() -> None:
    task = tasks()[0]
    broken = Task(**{**task.__dict__, "public_tests": "import math\nprint(1)"})
    with pytest.raises(ValueError, match="not an assertion"):
        chat.benchmark_user_text(broken)
    empty = Task(**{**task.__dict__, "public_tests": "import math"})
    with pytest.raises(ValueError, match="no public test"):
        chat.benchmark_user_text(empty)


def test_rows_may_omit_their_public_tests_but_prompts_never_do() -> None:
    task = tasks()[3]
    with_tests = chat.render_code_task(Bytes(), task)
    without = chat.render_code_task(Bytes(), task, public_tests=False)
    assert chat.task_user_text(task, public_tests=False) == task.prompt.strip()
    assert list(with_tests.ids[: len(chat.task_prompt(Bytes(), task))]) == chat.task_prompt(
        Bytes(), task
    )
    assert len(without.ids) < len(with_tests.ids)
    assert without.assistant_tokens == with_tests.assistant_tokens  # Same loss tokens.


def test_the_omission_rule_is_a_fixed_hash_and_takes_about_a_third() -> None:
    assert PUBLIC_TESTS_OMITTED_SALT in PUBLIC_TESTS_OMITTED_RULE
    identifiers = [f"source/{index}" for index in range(3_000)]
    share = sum(map(omits_public_tests, identifiers)) / len(identifiers)
    assert 0.3 < share < 0.37
    for identifier in identifiers[:50]:
        digest = hashlib.sha256(f"public-tests-omitted-v1|{identifier}".encode()).digest()
        assert omits_public_tests(identifier) == (int.from_bytes(digest[:8], "big") % 3 == 0)
