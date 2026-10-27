"""Reference checks for the evaluation protocols and the hashed completions file format."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from eval_fixtures import (
    CHECKPOINT,
    TOKENIZER,
    chat_tokens,
    io_task,
    pool_task,
    save_task_file,
    write_fixture_completions,
)

from scratchcoder.data import chat
from scratchcoder.eval import protocols
from scratchcoder.eval.completions import (
    CompletionsHeader,
    Sampling,
    prompt_sha256,
    read_completions,
    task_ids_sha256,
    validate_against_tasks,
    write_completions,
)
from scratchcoder.eval.harness import attempt_seed
from scratchcoder.eval.prompt import POOL_STOPS_BY_MODE
from scratchcoder.eval.protocols import (
    BASE_BENCHMARK,
    BASE_POOL,
    CHAT,
    Protocol,
    load_suite_tasks,
    protocol_for,
    record_sha256,
)

# Pinned record hashes. A change here means a protocol changed: completions files made
# under the old record are then refused, so update these only with a deliberate amendment.
PROTOCOL_SHA256 = {
    (CHAT, "pool"): "45d44b52a51aca900d361b7da8256744b17dab4addb7e5c0a5434602188f3c6b",
    (CHAT, "benchmark"): "3c42f7e24fe2a198b34f4696b678556de676462749d3b56d4efa5c41c1e4b18f",
    (BASE_POOL, "pool"): "88a93535c2528bab4ad2e6fb9fc47d3c105401b480cdb3e0ca4ee0f203bfed8b",
    (BASE_BENCHMARK, "benchmark"): (
        "3033d99f2319566b96f8f80a215bab215e5281cf43ac585c6f1caa9d88136a9a"
    ),
}


def test_protocol_records_are_pinned_by_hash() -> None:
    for (name, family), digest in PROTOCOL_SHA256.items():
        protocol = Protocol(name, family)
        assert protocol.sha256() == digest
        assert record_sha256(protocol.record()) == digest
        assert protocol.record()["name"] == name


def test_the_base_pool_stop_table_change_changed_the_record_hash() -> None:
    record = Protocol(BASE_POOL, "pool").record()
    stops = record["description"]["stop_strings_by_mode"]
    assert stops["io"] == ["\nassert", '\n"""', "\n# Test"]
    assert stops["io"] == list(POOL_STOPS_BY_MODE["io"])
    previous = json.loads(json.dumps(record))
    previous["description"]["stop_strings_by_mode"]["io"] = ["\nif __name__", *stops["io"]]
    assert record_sha256(previous) != record_sha256(record)
    assert record["name"] == record["description"]["name"] == "base-pool-completion-v1"


@pytest.mark.parametrize(
    ("name", "suite"),
    [(BASE_POOL, "humaneval"), (BASE_BENCHMARK, "pool"), ("legacy-instruction-v1", "pool")],
)
def test_protocols_are_only_defined_for_their_suites(name: str, suite: str) -> None:
    with pytest.raises(ValueError):
        protocol_for(name, suite)
    with pytest.raises(ValueError, match="suite"):
        protocol_for(CHAT, "arc-easy")


def test_prompts_budgets_and_stops_by_hand() -> None:
    task = pool_task("t/1")
    chat_pool = protocol_for(CHAT, "pool")
    assert chat_pool.prompt_ids(TOKENIZER, task) == chat.task_prompt(TOKENIZER, task)
    assert chat_pool.stop_ids == (chat.PYTHON_END, chat.ASSISTANT_END)
    assert chat_pool.budget(100, 1024) == 512 and chat_pool.budget(700, 1024) == 324
    assert chat_pool.budget(1024, 1024) == 0 and chat_pool.budget(2000, 1024) == 0
    base_pool = protocol_for(BASE_POOL, "pool")
    text = '"""\nWrite add_one(x).\n\nassert add_one(1) == 2\n"""\n'
    assert base_pool.prompt_ids(TOKENIZER, task) == list(text.encode())
    assert base_pool.stop_ids == () and base_pool.cap == 512
    benchmark = pool_task("he/1", prompt="def f(x):\n    '''Doc.'''\n\n", split="eval")
    base_benchmark = protocol_for(BASE_BENCHMARK, "humaneval")
    assert base_benchmark.prompt_ids(TOKENIZER, benchmark) == list(b"def f(x):\n    '''Doc.'''")
    assert base_benchmark.cap == 256 and base_benchmark.budget(10, 1024) == 256
    chat_benchmark = protocol_for(CHAT, "mbpp-plus")
    assert chat_benchmark.prompt_ids(TOKENIZER, benchmark) == chat.benchmark_prompt(
        TOKENIZER, benchmark
    )


def test_extraction_follows_each_protocol() -> None:
    task = pool_task("t/1")
    chat_pool = protocol_for(CHAT, "pool")
    tokens = chat_tokens("def add_one(x):\n    return x + 1")
    assert chat_pool.candidate(task, tokens=tokens, completion="", decoder=TOKENIZER) == (
        "def add_one(x):\n    return x + 1"
    )
    assert (
        chat_pool.candidate(task, tokens=chat_tokens(None), completion="", decoder=TOKENIZER)
        is None
    )
    with pytest.raises(ValueError, match="tokenizer"):
        chat_pool.candidate(task, tokens=tokens, completion="")
    base_pool = protocol_for(BASE_POOL, "pool")
    completion = "def add_one(x):\n    return x + 1\nassert add_one(1) == 2\n"
    assert base_pool.candidate(task, tokens=(), completion=completion) == (
        "def add_one(x):\n    return x + 1"
    )
    guarded = "def main():\n    print(int(input()) + 1)\n\nif __name__ == '__main__':\n    main()"
    assert base_pool.candidate(io_task("t/2"), tokens=(), completion=guarded) == guarded
    benchmark = pool_task("he/1", prompt="def f(x):\n", split="eval")
    base_benchmark = protocol_for(BASE_BENCHMARK, "humaneval")
    assert base_benchmark.candidate(
        benchmark, tokens=(), completion="\n    return x\n\ndef g(): pass"
    ) == ("def f(x):\n    return x\n")


def test_round_trip_identity_and_immutability(tmp_path: Path) -> None:
    tasks = [pool_task("t/2"), pool_task("t/1"), io_task("t/3")]
    tasks_path = save_task_file(tmp_path / "dev.jsonl", tasks)
    sampling = Sampling(samples=2, temperature=0.8, top_p=0.95, seed=3)
    path = tmp_path / "out" / "completions.jsonl"
    digest = write_fixture_completions(
        path, tasks_path, tasks, lambda task, sample: task.solution, sampling=sampling
    )
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert [entry.name for entry in path.parent.iterdir()] == ["completions.jsonl"]
    loaded = read_completions(path, sha256=digest)
    assert loaded.sha256 == digest
    assert [row.key for row in loaded.rows] == [
        (task_id, sample) for task_id in ("t/1", "t/2", "t/3") for sample in (0, 1)
    ]
    first = loaded.row("t/1", 1)
    assert first.seed == attempt_seed(3, "t/1", 1)
    assert first.program == "def add_one(x):\n    return x + 1"
    assert first.prompt_sha256 == prompt_sha256(chat.task_prompt(TOKENIZER, pool_task("t/1")))
    header = loaded.header
    assert header.checkpoint_sha256 == CHECKPOINT and header.sampler_version is not None
    assert header.task_ids_sha256 == task_ids_sha256(["t/1", "t/2", "t/3"])
    lines = path.read_text().splitlines()
    assert json.loads(lines[0])["kind"] == "header"
    assert json.loads(lines[1])["generated_tokens"] == len(loaded.row("t/1", 0).tokens)
    assert [task.id for task in validate_against_tasks(loaded, tasks_path)] == [
        "t/1",
        "t/2",
        "t/3",
    ]
    with pytest.raises(FileExistsError):
        write_completions(path, header, loaded.rows)
    with pytest.raises(ValueError, match="SHA-256"):
        read_completions(path, sha256="0" * 64)


def _rewrite(path: Path, edit: object) -> Path:
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    edit(lines)  # type: ignore[operator]
    target = path.with_name("edited.jsonl")
    target.write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines))
    return target


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda lines: lines[1].update(seed=lines[1]["seed"] + 1), "seed"),
        (lambda lines: lines.pop(), "attempt count"),
        (lambda lines: lines.append(dict(lines[-1])), "attempt count|duplicate"),
        (lambda lines: lines[0].update(schema_version=2), "schema version"),
        (lambda lines: lines[0].update(extra=1), "fields"),
        (lambda lines: lines[1].update(extra=1), "fields"),
        (lambda lines: lines[0]["protocol_record"].update(max_new_tokens="99"), "hash"),
        (lambda lines: lines[0].update(protocol="base-pool-completion-v1"), "protocol"),
        (lambda lines: lines[0].update(split="train"), "split"),
        (lambda lines: lines[0].update(max_new_tokens=600), "cap"),
        (lambda lines: lines[1].update(generated_tokens=0), "generated_tokens"),
        (lambda lines: lines[1].update(stop_reason="eos"), "stop_reason"),
        (lambda lines: lines[1].update(task_id="other"), "task IDs"),
        (lambda lines: lines[0].update(sampler_version="x"), "sampler"),
        (lambda lines: lines.reverse(), "header"),
        (lambda lines: lines[0].update(excluded_tasks=[["humaneval/32"]]), "excluded"),
    ],
)
def test_reading_refuses_inconsistent_files(tmp_path: Path, edit: object, message: str) -> None:
    tasks = [pool_task("t/1"), pool_task("t/2")]
    tasks_path = save_task_file(tmp_path / "dev.jsonl", tasks)
    path = tmp_path / "completions.jsonl"
    write_fixture_completions(path, tasks_path, tasks, lambda task, sample: task.solution)
    with pytest.raises(ValueError, match=message):
        read_completions(_rewrite(path, edit))


def test_a_truncated_file_is_refused(tmp_path: Path) -> None:
    tasks = [pool_task("t/1")]
    tasks_path = save_task_file(tmp_path / "dev.jsonl", tasks)
    path = tmp_path / "completions.jsonl"
    write_fixture_completions(path, tasks_path, tasks, lambda task, sample: task.solution)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="newline"):
        read_completions(path)


def test_the_header_must_match_the_task_file(tmp_path: Path) -> None:
    tasks = [pool_task("t/1"), pool_task("t/2")]
    tasks_path = save_task_file(tmp_path / "dev.jsonl", tasks)
    path = tmp_path / "completions.jsonl"
    write_fixture_completions(path, tasks_path, tasks, lambda task, sample: task.solution)
    completions = read_completions(path)
    other = save_task_file(tmp_path / "other.jsonl", [*tasks, pool_task("t/3")])
    with pytest.raises(ValueError, match="SHA-256"):
        validate_against_tasks(completions, other)
    # Same bytes hash, different task list in the header: refused as a task mismatch.
    fewer = save_task_file(tmp_path / "fewer.jsonl", tasks[:1])
    forged = replace(completions, header=replace(completions.header, task_file_sha256=file(fewer)))
    with pytest.raises(ValueError, match="tasks differ"):
        validate_against_tasks(forged, fewer)


def file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_header_validation_rejects_bad_settings() -> None:
    protocol = protocol_for(CHAT, "pool")
    valid = CompletionsHeader(
        suite="pool",
        split="dev",
        protocol=CHAT,
        protocol_sha256=protocol.sha256(),
        protocol_record=protocol.record(),
        checkpoint_sha256=CHECKPOINT,
        tokenizer_sha256="b" * 64,
        tokenizer_vocab_size=272,
        model_context=1024,
        task_file_sha256="d" * 64,
        task_count=1,
        task_ids_sha256="e" * 64,
        sampling=Sampling(),
        max_new_tokens=512,
        precision="bfloat16",
        device="cuda:0",
        rows=1,
        created_at="now",
    )
    assert valid.to_json()["format"] == "scratchcoder-completions"
    assert CompletionsHeader.from_json(json.loads(json.dumps(valid.to_json()))) == valid
    for change in (
        {"rows": 2},
        {"precision": "float16"},
        {"checkpoint_sha256": "C" * 64},
        {"suite": "arc"},
        {"split": "test", "suite": "humaneval"},
        {"excluded_tasks": ({"task_id": "x"},)},
    ):
        with pytest.raises(ValueError):
            replace(valid, **change)
    with pytest.raises(ValueError, match="greedy"):
        Sampling(samples=8)
    with pytest.raises(ValueError):
        Sampling(samples=8, temperature=0.8, top_p=0)


def test_suite_loading_checks_the_split_and_excludes_task_32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks_path = save_task_file(tmp_path / "dev.jsonl", [pool_task("t/1")])
    assert [task.id for task in load_suite_tasks("pool", "dev", tasks_path)[0]] == ["t/1"]
    with pytest.raises(ValueError, match="only test"):
        load_suite_tasks("pool", "test", tasks_path)
    with pytest.raises(ValueError, match="dev or test"):
        load_suite_tasks("pool", "train", tasks_path)
    release = [
        pool_task(f"humaneval/{index}", split="eval", entry_point="f") for index in (31, 32, 33)
    ]
    monkeypatch.setattr(protocols, "read_release", lambda path, specification: release)
    tasks, excluded = load_suite_tasks("humaneval-plus", "eval", tasks_path)
    assert [task.id for task in tasks] == ["humaneval/31", "humaneval/33"]
    assert excluded == [{"task_id": "humaneval/32", "reason": "known numerical-tolerance failure"}]
    assert load_suite_tasks("mbpp-plus", "eval", tasks_path)[1] == []
    with pytest.raises(ValueError, match="eval"):
        load_suite_tasks("mbpp-plus", "dev", tasks_path)
